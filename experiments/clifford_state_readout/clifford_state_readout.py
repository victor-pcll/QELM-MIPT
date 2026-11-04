#!/usr/bin/env python3
"""Learn ``S_R`` and the purity of the reference qubit from the final state (#292).

Can a QELM-style readout of the final state of the **system** learn, trajectory
by trajectory, the entropy ``S_R`` of the reference qubit ``R``, the
purification order parameter of the measurement-induced transition (MIPT)?
If it could, the learned readout would be a probe of the MIPT. The circuits
are Clifford, so the state is a stabilizer state, simulated with ``stim``
(:mod:`QELM.models.circuits.stabilizer`), and ``N`` can be large.

One trajectory, at one size ``N`` and one measurement probability ``p``:

1. ``R`` (qubit 0) starts in a Bell pair with the middle system site
   ``N // 2``, the other system qubits in ``|0>``.
2. ``task.scramble_steps`` steps without measurement (0 by default), then
   ``task.steps`` measured steps (``2N`` by default). A step is one
   half-layer of random two-qubit Clifford gates on the periodic brickwork,
   then a Z measurement of each system qubit with probability ``p``
   (the step of :func:`~QELM.models.circuits.stabilizer.draw_probe_history`).
   New gates for every trajectory.
3. The target: ``S_R`` in bits. For a stabilizer state it is 0 or 1, and the
   purity is ``Tr(rho_R^2) = 1 - S_R / 2`` (1 or 1/2), an affine function of
   ``S_R``: both targets have the same ``R2``, and the purity has half the
   mean absolute error.
4. The features, the QELM observables of the final state of the system
   (``R`` excluded), :func:`state_features`: ``<P_k>`` for ``P`` in
   ``X, Y, Z`` and every site, and ``<P_i Q_j>`` for every pair ``i < j``
   and ``P, Q`` in ``X, Y, Z``. They are the exact expectation values of the
   trajectory state: 0 or +-1 for a stabilizer state.

The readouts (:data:`READOUTS`), each fitted on a training part of the
trajectories and scored on the rest:

- ``linear``: ridge on the features, the linear QELM;
- ``quadratic``: ridge on the features, their squares (``|<P>|``, 1 when
  ``+-P`` stabilizes the state) and the products of every two one-site
  features;
- ``abs``: ridge on the squares only.

For a stabilizer state the signs carry no information on ``S_R``: ``S_R``
and the stabilizer group up to signs (so every ``|<P>|``) depend only on the
gates and on which sites were measured, while the signs also depend on the
outcomes. The readouts that read the signs (``linear``) can only lose;
``abs`` and ``mlp`` read the outcome-independent part of the state.
- ``mlp``: a small neural network (one hidden layer of 32 units, L2
  penalty, early stopping) on the squares, a classifier whose output is
  its probability that ``S_R = 1``;
- ``chance``: the mean of the training targets;
- ``n_meas``: ridge on the number of measurements of the trajectory and its
  square, a control that reads the record, not the state: within one ``p``
  a trajectory with more measurements is more likely purified, and the
  number of ``Z`` stabilizers of the state tracks it. A readout of the state
  that does no better than ``n_meas`` has found nothing but that count.

Two protocols:

- **per p**: one readout per ``p``. Its ``R2`` is the information on
  ``S_R`` beyond ``p``: the mean of ``S_R`` at that ``p`` scores 0.
- **pooled**: one readout for all ``p`` at once, which does not know ``p``.
  Its mean prediction at each ``p`` is its estimate of the order parameter
  ``<S_R>(p)``. The baseline ``p_mean`` predicts the mean training ``S_R``
  at the trajectory's ``p``: it knows ``p`` but nothing about the
  trajectory, so a readout that only beats it in pooled ``R2`` by reading
  ``p`` from the state (for instance from how many sites carry a ``Z``
  stabilizer) brings nothing beyond ``p``.

Access assumption: the features are exact expectation values of the
trajectory state. An experiment prepares each trajectory state once, so it
would need post-selection on the record (or many copies) to estimate them:
the readout is an upper bound on what a QELM of the state can do.

Run from the repository root:

    uv run python experiments/clifford_state_readout/clifford_state_readout.py N=8

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``: ``metrics.json`` (scores, oracle and predicted
order parameter, per ``p``), optionally ``trajectories.npz`` (features and
targets) and the config.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stim  # type: ignore[import-untyped]
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from QELM.models.circuits.stabilizer import (
    REFERENCE,
    ProbeHistory,
    _do_step,
    draw_probe_history,
)
from QELM.utils import save_config

READOUTS = ("chance", "n_meas", "linear", "abs", "quadratic", "mlp")
PAULIS = "XYZ"
RIDGE_ALPHAS = np.logspace(-4, 4, 17)


@cache
def feature_paulis(N: int) -> tuple[stim.PauliString, ...]:
    """Return the Pauli strings of :func:`state_features`, in order.

    The ``3 N`` one-site strings ``P_k`` (site-major: ``X_0, Y_0, Z_0, X_1,
    ...``), then the ``9 N (N - 1) / 2`` two-site strings ``P_i Q_j`` for
    ``i < j`` (pair-major, then ``P``, then ``Q``). Each acts on the
    ``N + 1`` qubits of the simulator, the identity on ``R`` (qubit 0):
    system site ``k`` is qubit ``k + 1``.

    Args:
        N: Number of system sites.

    Returns:
        ``3 N + 9 N (N - 1) / 2`` Pauli strings.
    """
    strings = []
    for k in range(N):
        for a in PAULIS:
            s = stim.PauliString(N + 1)
            s[k + 1] = a
            strings.append(s)
    for i in range(N):
        for j in range(i + 1, N):
            for a in PAULIS:
                for b in PAULIS:
                    s = stim.PauliString(N + 1)
                    s[i + 1] = a
                    s[j + 1] = b
                    strings.append(s)
    return tuple(strings)


def n_one_site(N: int) -> int:
    """Number of one-site features of :func:`state_features`, ``3 N``."""
    return 3 * N


def state_features(sim: stim.TableauSimulator, N: int) -> np.ndarray:
    """Return the one- and two-site Pauli expectation values of the system.

    Args:
        sim: Simulator of the ``N + 1`` qubits, ``R`` first.
        N: Number of system sites.

    Returns:
        ``<P>`` for each string of :func:`feature_paulis`, 0 or +-1, shape
        ``(3 N + 9 N (N - 1) / 2,)``.
    """
    return np.array(
        [sim.peek_observable_expectation(s) for s in feature_paulis(N)], dtype=float
    )


def reference_entropy(sim: stim.TableauSimulator) -> float:
    """Entropy of ``R`` in bits: 1 if no stabilizer acts on ``R`` alone, else 0."""
    return 1.0 if sim.peek_bloch(REFERENCE) == stim.PauliString(1) else 0.0


def replay(history: ProbeHistory, seed: int) -> stim.TableauSimulator:
    """Prepare the Bell pair and replay ``history`` on a stabilizer simulator.

    Args:
        history: Output of :func:`draw_probe_history`.
        seed: Seed of the ``stim`` measurement outcomes.

    Returns:
        The simulator in the final state, ``R`` on qubit 0, system site
        ``k`` on qubit ``k + 1``.
    """
    sim = stim.TableauSimulator(seed=seed)
    # The encoding: (|00> + |11>) / sqrt(2) on R and the middle site.
    sim.h(REFERENCE)
    sim.cnot(REFERENCE, history.N // 2 + 1)
    for t in range(history.T):
        _do_step(sim, history, t)
    return sim


def simulate_trajectory(
    N: int,
    p: float,
    scramble_steps: int,
    steps: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, int]:
    """Run one trajectory and return its target and features.

    Args:
        N: Number of system sites, even.
        p: Measurement probability per site and measured step.
        scramble_steps: Steps without measurement, first.
        steps: Measured steps.
        rng: Random generator of the gates, the measured sites and the
            ``stim`` outcomes.

    Returns:
        ``(S_R, features, n_measured)``: the entropy of ``R`` in bits (0 or
        1), :func:`state_features` of the final state and the number of
        measurements of the trajectory.
    """
    history = draw_probe_history(N, p, scramble_steps, steps, rng)
    sim = replay(history, int(rng.integers(2**31)))
    return reference_entropy(sim), state_features(sim, N), int(history.measured.sum())


def simulate(
    N: int,
    p_values: Sequence[float],
    n_samples: int,
    scramble_steps: int,
    steps: int,
    seed: int,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate ``n_samples`` trajectories at each ``p``.

    Args:
        N: Number of system sites, even.
        p_values: Measurement probabilities.
        n_samples: Trajectories per ``p``.
        scramble_steps: Steps without measurement.
        steps: Measured steps.
        seed: Seed; each ``p`` has its own stream ``(seed, N, p)``.
        progress: Show a progress bar.

    Returns:
        ``S_R`` (shape ``(P, L)``), ``features`` (shape ``(P, L, F)``),
        ``n_measured`` (shape ``(P, L)``) and ``p`` (shape ``(P,)``).
    """
    F = len(feature_paulis(N))
    P = len(p_values)
    S = np.zeros((P, n_samples))
    X = np.zeros((P, n_samples, F), dtype=np.int8)
    n_meas = np.zeros((P, n_samples), dtype=int)
    for a, p in enumerate(p_values):
        rng = np.random.default_rng([seed, N, round(float(p) * 1e9)])
        for i in tqdm(range(n_samples), desc=f"N={N} p={p}", disable=not progress):
            S[a, i], x, n_meas[a, i] = simulate_trajectory(
                N, float(p), scramble_steps, steps, rng
            )
            X[a, i] = x
    return {
        "S_R": S,
        "features": X,
        "n_measured": n_meas,
        "p": np.asarray(p_values, float),
    }


def quadratic_features(X: np.ndarray, n1: int) -> np.ndarray:
    """Features, their squares and the products of every two one-site features.

    Args:
        X: Features, shape ``(L, F)``; the ``n1`` first columns are the
            one-site ones.
        n1: Number of one-site features.

    Returns:
        Shape ``(L, 2 F + n1 (n1 - 1) / 2)``.
    """
    i, j = np.triu_indices(n1, k=1)
    one = X[:, :n1]
    return np.concatenate([X, X**2, one[:, i] * one[:, j]], axis=1)


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))


def _mlp(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=10.0,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        ),
    )


def fit_predict(
    readout: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n1: int,
    seed: int,
    n_train: np.ndarray | None = None,
    n_test: np.ndarray | None = None,
) -> np.ndarray:
    """Fit one readout of :data:`READOUTS` and predict on ``X_test``.

    Args:
        readout: Name in :data:`READOUTS`.
        X_train: Training features, shape ``(L_train, F)``.
        y_train: Training targets, shape ``(L_train,)``.
        X_test: Test features, shape ``(L_test, F)``.
        n1: Number of one-site features (first columns).
        seed: Seed of the MLP.
        n_train: Number of measurements of each training trajectory, for
            ``n_meas``.
        n_test: The same for the test trajectories.

    Returns:
        Predictions, shape ``(L_test,)``.

    Raises:
        ValueError: If ``readout`` is unknown, or is ``n_meas`` without the
            numbers of measurements.
    """
    if readout == "chance":
        return np.full(len(X_test), float(np.mean(y_train)))
    if readout == "n_meas":
        if n_train is None or n_test is None:
            raise ValueError("n_meas needs n_train and n_test.")
        X_train = np.stack([n_train, n_train**2], axis=1).astype(float)
        X_test = np.stack([n_test, n_test**2], axis=1).astype(float)
        readout = "linear"
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "linear": lambda X: X,
        "abs": lambda X: np.abs(X),
        "quadratic": lambda X: quadratic_features(X, n1),
        "mlp": lambda X: np.abs(X),
    }
    if readout not in transforms:
        raise ValueError(f"Unknown readout {readout!r}, expected one of {READOUTS}.")
    if np.ptp(y_train) == 0:  # one class only: nothing to learn
        return np.full(len(X_test), float(y_train[0]))
    f = transforms[readout]
    if readout == "mlp":
        # S_R is 0 or 1: the network outputs P(S_R = 1 | features).
        mlp = _mlp(seed)
        mlp.fit(f(X_train).astype(float), y_train.astype(int))
        return np.asarray(mlp.predict_proba(f(X_test).astype(float))[:, 1], float)
    model = _ridge()
    model.fit(f(X_train).astype(float), y_train)
    return np.asarray(model.predict(f(X_test).astype(float)), float)


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """``R2`` (NaN when ``y`` is constant) and mean absolute error of ``S_R``.

    The purity ``1 - S_R / 2`` has the same ``R2`` and half the error.
    """
    r2 = float(r2_score(y, pred)) if np.ptp(y) > 0 else float("nan")
    return {"R2": r2, "MAE": float(mean_absolute_error(y, pred))}


def split(L: int, train_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean training mask of ``L`` samples, ``round(train_fraction L)`` true."""
    mask = np.zeros(L, dtype=bool)
    mask[rng.permutation(L)[: int(round(train_fraction * L))]] = True
    return mask


def evaluate(
    data: dict[str, np.ndarray],
    readouts: Sequence[str],
    train_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Score the readouts per ``p`` and pooled over ``p``.

    Args:
        data: Output of :func:`simulate`.
        readouts: Names in :data:`READOUTS`.
        train_fraction: Fraction of the trajectories of each ``p`` used for
            training; the same split serves both protocols.
        seed: Seed of the split and of the MLP.

    Returns:
        ``oracle`` (mean ``S_R`` on the test set per ``p``, with its
        standard error), ``per_p[readout]`` (``R2`` and ``MAE`` per ``p``),
        ``pooled[readout]`` (overall ``R2``, ``R2`` and ``MAE`` per ``p``,
        mean prediction per ``p``; ``p_mean`` is the baseline that knows
        ``p``).
    """
    S, X, p, n = data["S_R"], data["features"], data["p"], data["n_measured"]
    P, L, _ = X.shape
    n1 = _n1_from_features(X.shape[2])
    rng = np.random.default_rng([seed, 292])
    masks = np.stack([split(L, train_fraction, rng) for _ in range(P)])
    test_S = [S[a][~masks[a]] for a in range(P)]
    out: dict[str, Any] = {
        "p": p.tolist(),
        "oracle": [float(t.mean()) for t in test_S],
        "oracle_err": [
            float(t.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else float("nan")
            for t in test_S
        ],
        "oracle_all": S.mean(axis=1).tolist(),
        "measurement_density": data["n_measured"].mean(axis=1).tolist(),
        "per_p": {},
        "pooled": {},
    }
    for r in readouts:
        rows = []
        for a in range(P):
            m = masks[a]
            pred = fit_predict(
                r, X[a][m], S[a][m], X[a][~m], n1, seed, n[a][m], n[a][~m]
            )
            rows.append(scores(S[a][~m], pred))
        out["per_p"][r] = {k: [row[k] for row in rows] for k in ("R2", "MAE")}
    # Pooled: one readout for every p; it is not told p.
    X_tr = np.concatenate([X[a][masks[a]] for a in range(P)])
    y_tr = np.concatenate([S[a][masks[a]] for a in range(P)])
    X_te = np.concatenate([X[a][~masks[a]] for a in range(P)])
    y_te = np.concatenate(test_S)
    n_tr = np.concatenate([n[a][masks[a]] for a in range(P)])
    n_te = np.concatenate([n[a][~masks[a]] for a in range(P)])
    p_te = np.concatenate([np.full((~masks[a]).sum(), a) for a in range(P)])
    preds: dict[str, np.ndarray] = {}
    for r in readouts:
        if r != "chance":
            preds[r] = fit_predict(r, X_tr, y_tr, X_te, n1, seed, n_tr, n_te)
    preds["chance"] = np.full(len(y_te), float(y_tr.mean()))
    preds["p_mean"] = np.array([S[a][masks[a]].mean() for a in p_te])
    for r, pred in preds.items():
        rows = [scores(y_te[p_te == a], pred[p_te == a]) for a in range(P)]
        out["pooled"][r] = {
            "R2_all": scores(y_te, pred)["R2"],
            "R2": [row["R2"] for row in rows],
            "MAE": [row["MAE"] for row in rows],
            "mean_prediction": [float(pred[p_te == a].mean()) for a in range(P)],
        }
    return out


def _n1_from_features(F: int) -> int:
    """Number of one-site features ``3 N`` from ``F = 3 N + 9 N (N - 1) / 2``."""
    for N in range(1, 1000):
        if len_features(N) == F:
            return n_one_site(N)
    raise ValueError(f"No N has {F} features.")


def len_features(N: int) -> int:
    """Number of features of :func:`state_features`, ``3 N + 9 N (N - 1) / 2``."""
    return 3 * N + 9 * N * (N - 1) // 2


def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Simulate and score one size ``N``.

    Args:
        cfg: Config composed from ``conf/clifford_state_readout.yaml``.

    Returns:
        ``(metrics, data)``: the scores of :func:`evaluate` with the run's
        parameters, and the output of :func:`simulate`.

    Raises:
        ValueError: If a readout is unknown.
    """
    N = int(cfg.N)
    readouts = [str(r) for r in cfg.task.readouts]
    unknown = sorted(set(readouts) - set(READOUTS))
    if unknown:
        raise ValueError(f"Unknown readouts {unknown}, expected some of {READOUTS}.")
    steps = int(cfg.task.steps)
    scramble = int(cfg.task.scramble_steps)
    data = simulate(
        N,
        [float(p) for p in cfg.task.p_values],
        int(cfg.task.samples_per_p),
        scramble,
        steps,
        int(cfg.seed),
        progress=bool(cfg.task.get("progress", False)),
    )
    metrics = evaluate(data, readouts, float(cfg.task.train_fraction), int(cfg.seed))
    metrics.update(
        {
            "N": N,
            "seed": int(cfg.seed),
            "steps": steps,
            "scramble_steps": scramble,
            "samples_per_p": int(cfg.task.samples_per_p),
            "train_fraction": float(cfg.task.train_fraction),
            "n_features": len_features(N),
        }
    )
    return metrics, data


def save(
    metrics: dict[str, Any], data: dict[str, np.ndarray], out_dir: Path, arrays: bool
) -> None:
    """Write ``metrics.json`` and, if ``arrays``, ``trajectories.npz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if arrays:
        np.savez_compressed(out_dir / "trajectories.npz", **data)


@hydra.main(
    version_base="1.3", config_path="../../conf", config_name="clifford_state_readout"
)
def main(cfg: DictConfig) -> None:
    """Run one ``N`` and save it to the Hydra run directory."""
    metrics, data = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(metrics, data, out_dir, bool(cfg.task.save_arrays))
    save_config(cfg, out_dir)
    print(f"N: {metrics['N']}  seed: {metrics['seed']}")
    for a, p in enumerate(metrics["p"]):
        r2 = {r: metrics["per_p"][r]["R2"][a] for r in metrics["per_p"]}
        txt = "  ".join(f"{r}={v:.3f}" for r, v in r2.items())
        print(f"p={p:.3f}  <S_R>={metrics['oracle'][a]:.3f}  R2 per p: {txt}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
