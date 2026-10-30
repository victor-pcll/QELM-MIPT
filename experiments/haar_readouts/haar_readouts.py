#!/usr/bin/env python3
"""Learn the charge of a fully random Haar circuit with several readouts.

The task of ``experiments/QELM_M`` (the measurement record ``M`` against the
initial charge sector, learnability of Barratt et al. 2022), with gates that
conserve nothing: Haar-random U(4) gates instead of the TFIM, whose large
transverse field makes the charge almost conserved, or U(1) gates. Without a
conserved charge the record no longer holds ``Q`` as a sum of outcomes, so a
linear readout may miss what it holds. Several readouts of the same record
are compared, each on its own.

For each measurement probability ``p`` of ``readouts_task.p_values`` and
each of its ``readouts_task.samples_per_p`` trajectories:

1. Start from :class:`AncillaChargeSectorEncoding`,
   ``(|0>_R |Q1>_S + |1>_R |Q2>_S) / sqrt(2)`` with ``Q1 = N // 2`` and
   ``Q2 = N // 2 - 1``: the reference qubit ``R`` (qubit 0, ``Nr = 1``)
   entangled with the charge of the ``N`` system qubits.
2. Run ``readouts_task.n_rounds`` rounds, each one brickwork layer of
   Haar-random two-qubit gates (:class:`HaarBrickworkCircuit`, a full
   period: even bonds, then odd bonds) on the system qubits, followed by a
   Z measurement of each system qubit with probability ``p``. Row ``j`` of
   ``M`` holds the outcomes of round ``j``, +-1, and 0 for an unmeasured
   qubit. With ``haar_circuit.fixed_instance: false`` (the default) every
   trajectory draws new gates; ``true`` keeps one draw for the whole run.
3. At the end, the entropy ``S_R`` of ``R`` (von Neumann, bits: 1 while the
   system hides the charge, 0 once the record fixes it, the purification
   order parameter) and the label ``Y``, the Z outcome of ``R`` drawn with
   the Born rule: +1 for ``Q1``, -1 for ``Q2``. ``R`` never evolves, so ``Y``
   is the initial charge sector, +-1 with probability 1/2.

The state stays pure, so it is simulated as a state vector.

The readouts read one or several feature sets (``readouts_task.features``,
:data:`FEATURE_SETS`):

- ``M``: the flattened record (default);
- ``X``, ``Y``, ``Z``: the expectation values ``<P_k>`` of the system qubits
  in the final state, conditioned on the record and on the outcome of
  ``R`` (the state after ``R`` is measured, as in ``experiments/QELM_Q``:
  without that projection the state of the system would be the same mixture
  of both sectors for every label);
- ``XX``, ``YY``, ``ZZ``: the two-site correlators ``<P_i P_j>`` of the
  same state, on every pair ``i < j`` of system qubits;
- ``all``: the six sets above side by side.

At each ``p``, the trajectories are split into training and test sets
(``samples.train_fraction``, stratified on ``Y``) and every readout of
``readouts_task.readouts`` is fitted on every feature set and scored by
two scores on the test set: its ``accuracy`` (the sign of its output
against ``Y``) and the ``R2`` of its output ``s`` against ``Y``
(``sklearn.metrics.r2_score``). The output of a ridge is its prediction of
``Y``; that of a classifier is ``2 P(Y = +1) - 1``, its estimate of the
mean of ``Y`` given the record. ``R2 = 0`` for an output that ignores the
record, 1 for a perfect one, negative for one worse than the mean of ``Y``.

- ``ridge``: the linear readout of the QELM, a ridge regression of ``Y``
  (``RidgeCV``, its alpha chosen by leave-one-out cross-validation over
  :data:`ALPHAS`), the sign of its prediction;
- ``ridge_pairs``: the same ridge on the record and the products of every
  pair of its entries, the quadratic features a parity of two outcomes
  needs;
- ``knn``: ``sqrt(n_train)`` nearest neighbors, weighted by distance;
- ``mlp``: a neural network with two hidden layers of 64 units (ReLU,
  early stopping);
- ``gradient_boosting``: gradient-boosted trees
  (``HistGradientBoostingClassifier``).

Run from the repository root:

    uv run python experiments/haar_readouts/haar_readouts.py N=6 \\
        readouts_task.samples_per_p=500

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``haar_readouts.npz``: ``M`` (``(L, n_rounds, N)``), ``Y``, ``S_R`` and
  ``p`` (the measurement probability of each trajectory), of shape
  ``(L,)``;
- ``metrics.json``: per ``p``, ``S_R_mean_p<p>``, ``S_R_std_p<p>``,
  ``accuracy_baseline_p<p>`` (the majority label of the training set) and
  ``accuracy_<name>_p<p>`` and ``r2_<name>_p<p>`` for each readout and
  feature set, with ``<name>`` the readout for ``M`` and
  ``<readout>_<set>`` for the other sets (``ridge_ZZ``); ``n_rounds``;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import scipy.sparse as sp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from QELM import check_qubit_counts, set_seed
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.evaluation.entropy import von_neumann_entropy
from QELM.evaluation.entropy_regression import stratified_split
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.physics.operators import PAULIS, get_site_operator
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config

# Seed tag of the generators of the gates and the measurements.
RNG_TAG = 260
# Regularizations tried by the ridge readouts.
ALPHAS = np.logspace(-2, 4, 13)

# A readout: (training features, training labels, test features, rng) ->
# output on the test set, an estimate of Y in [-1, 1] for the classifiers.
Readout = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.random.Generator], np.ndarray
]


def _sign(y: np.ndarray) -> np.ndarray:
    return np.where(y >= 0.0, 1.0, -1.0)


def _mean_label(model: Any, X_test: np.ndarray) -> np.ndarray:
    """``2 P(Y = +1) - 1`` of a fitted scikit-learn classifier."""
    plus = list(model.classes_).index(1.0)
    return np.asarray(2.0 * model.predict_proba(X_test)[:, plus] - 1.0)


def _ridge(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    model = RidgeCV(alphas=ALPHAS).fit(X, y)
    return np.asarray(model.predict(X_test), dtype=float)


def _ridge_pairs(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    pairs = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    return _ridge(pairs.fit_transform(X), y, pairs.transform(X_test), rng)


def _knn(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    k = max(1, round(np.sqrt(X.shape[0])))
    model = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X, y)
    return _mean_label(model, X_test)


def _mlp(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        early_stopping=True,
        max_iter=500,
        random_state=int(rng.integers(2**31)),
    ).fit(X, y)
    return _mean_label(model, X_test)


def _gradient_boosting(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    model = HistGradientBoostingClassifier(
        max_iter=300, random_state=int(rng.integers(2**31))
    ).fit(X, y)
    return _mean_label(model, X_test)


# Feature sets of the readouts: the record, and observables of the final state.
OBSERVABLE_SETS = ("X", "Y", "Z", "XX", "YY", "ZZ")
FEATURE_SETS = ("M", *OBSERVABLE_SETS, "all")


def observables(N: int) -> dict[str, list[sp.csr_matrix]]:
    """Return the Pauli operators of every set of :data:`OBSERVABLE_SETS`.

    The state holds ``R`` (qubit 0) and the ``N`` system qubits (``1 .. N``):
    ``P_k`` on each system qubit, ``P_i P_j`` on each pair ``i < j``, sparse.
    """
    n_total = N + 1
    single = {
        name: [get_site_operator(op, k + 1, n_total, sparse=True) for k in range(N)]
        for name, op in PAULIS.items()
    }
    ops: dict[str, list[sp.csr_matrix]] = {}
    for name in OBSERVABLE_SETS:
        if len(name) == 1:
            ops[name] = single[name]
        else:
            P = single[name[0]]
            ops[name] = [
                sp.csr_matrix(P[i] @ P[j]) for i in range(N) for j in range(i + 1, N)
            ]
    return ops


def expectation_values(psi: np.ndarray, ops: list[sp.csr_matrix]) -> np.ndarray:
    """Return ``<psi| O |psi>`` for every operator of ``ops``."""
    return np.array([float(np.real(np.vdot(psi, op @ psi))) for op in ops])


READOUTS: dict[str, Readout] = {
    "ridge": _ridge,
    "ridge_pairs": _ridge_pairs,
    "knn": _knn,
    "mlp": _mlp,
    "gradient_boosting": _gradient_boosting,
}


def trajectory(
    psi0: np.ndarray,
    circuit: HaarBrickworkCircuit,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Run one trajectory from ``psi0``: ``circuit.depth`` rounds of gates
    and measurements of each system qubit with probability ``p``.

    Returns:
        ``(M, S_R, Y, psi)``: the record ``(depth, N)``, the entropy of ``R``
        in bits, its Z outcome, +-1, drawn with the Born rule, and the final
        state after that measurement of ``R``.
    """
    N, N_total = circuit.N, circuit.N_total
    psi = psi0
    M = np.zeros((circuit.depth, N))
    for j in range(circuit.depth):
        psi = circuit.apply_layer(psi, j)
        for k in range(N):
            if rng.random() < p:
                psi, M[j, k] = measure_projective_psi(psi, k + 1, N_total, rng)
    S_R = von_neumann_entropy(psi, [0])
    psi, Y = measure_projective_psi(psi, 0, N_total, rng)
    return M, S_R, float(Y), psi


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories of every ``p``, fit and score the readouts.

    Args:
        cfg: Composed config (``load_config(..., config_name="haar_readouts")``
            or Hydra). Reads ``N``, ``Nr`` (must be 1), ``haar_circuit``,
            ``readouts_task`` and ``samples.train_fraction``.

    Returns:
        The arrays and scores listed in the module docstring.

    Raises:
        ValueError: If ``Nr`` is not 1, a probability is outside ``[0, 1]``,
            a count is too small or a readout is unknown.
    """
    set_seed(cfg)
    N, Nr, N_total = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"haar_readouts needs the reference qubit: Nr=1, got {Nr}.")
    task = cfg.readouts_task
    levels = np.asarray([float(v) for v in task.p_values])
    if levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"readouts_task.p_values must be in [0, 1], got {levels}.")
    if np.unique(levels).size != levels.size:
        raise ValueError(f"readouts_task.p_values must be distinct, got {levels}.")
    per_p = int(task.samples_per_p)
    if per_p < 10:
        raise ValueError(f"readouts_task.samples_per_p must be >= 10, got {per_p}.")
    n_rounds = int(task.n_rounds)
    if n_rounds < 1:
        raise ValueError(f"readouts_task.n_rounds must be >= 1, got {n_rounds}.")
    names = [str(name) for name in task.readouts]
    unknown = [name for name in names if name not in READOUTS]
    if unknown:
        raise ValueError(f"Unknown readouts {unknown}; known: {', '.join(READOUTS)}.")
    sets = [str(name) for name in task.get("features", ["M"])]
    unknown = [name for name in sets if name not in FEATURE_SETS]
    if unknown or not sets:
        raise ValueError(
            f"readouts_task.features must hold sets of {FEATURE_SETS}, got {sets}."
        )
    needed = [name for name in OBSERVABLE_SETS if name in sets or "all" in sets]
    ops = observables(N)

    seed = int(cfg.seed)
    gate_rng = np.random.default_rng([seed, RNG_TAG, 0])
    rng = np.random.default_rng([seed, RNG_TAG, 1])
    block = OmegaConf.to_container(cfg.haar_circuit, resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    params["depth"] = n_rounds
    circuit = HaarBrickworkCircuit(params, gate_rng)
    fixed = bool(params.get("fixed_instance", False))
    psi0 = AncillaChargeSectorEncoding.from_config(cfg).state()

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, n_rounds, N))
    S_R = np.zeros(L)
    Y = np.zeros(L)
    obs = {name: np.zeros((L, len(ops[name]))) for name in needed}
    for i in tqdm(range(L)):
        if not fixed and i > 0:
            circuit.redraw(gate_rng)
        M[i], S_R[i], Y[i], psi = trajectory(psi0, circuit, float(p[i]), rng)
        for name in needed:
            obs[name][i] = expectation_values(psi, ops[name])

    split_rng = np.random.default_rng([seed, RNG_TAG, 2])
    fit_rng = np.random.default_rng([seed, RNG_TAG, 3])
    features = {"M": M.reshape(L, -1), **obs}
    if "all" in sets:
        features["all"] = np.hstack([obs[name] for name in OBSERVABLE_SETS])
    metrics: dict[str, float] = {"n_rounds": float(n_rounds)}
    for level in levels:
        idx = np.flatnonzero(p == level)
        train, test = stratified_split(
            Y[idx], float(cfg.samples.train_fraction), split_rng
        )
        tr, te = idx[train], idx[test]
        key = f"p{level:g}"
        metrics[f"S_R_mean_{key}"] = float(S_R[idx].mean())
        metrics[f"S_R_std_{key}"] = float(S_R[idx].std())
        majority = 1.0 if np.mean(Y[tr]) >= 0.0 else -1.0
        metrics[f"accuracy_baseline_{key}"] = float(np.mean(Y[te] == majority))
        for feature in sets:
            X = features[feature]
            for readout in names:
                name = readout if feature == "M" else f"{readout}_{feature}"
                if np.unique(Y[tr]).size < 2 or np.ptp(X[tr], axis=0).max() < 1e-12:
                    # Nothing to learn from: a single label, or constant
                    # features (the empty record at p = 0). The output is
                    # the mean label of the training set, whose sign is the
                    # majority label.
                    score = np.full(te.size, float(np.mean(Y[tr])))
                else:
                    score = READOUTS[readout](X[tr], Y[tr], X[te], fit_rng)
                metrics[f"accuracy_{name}_{key}"] = float(
                    np.mean(_sign(score) == Y[te])
                )
                metrics[f"r2_{name}_{key}"] = float(r2_score(Y[te], score))
    return {
        "M": M,
        "Y": Y,
        "S_R": S_R,
        "p": p,
        **{f"features_{name}": obs[name] for name in needed},
        "metrics": metrics,
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="haar_readouts")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/haar_readouts.yaml`` and CLI
            overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{Path(__file__).stem}.npz",
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
    )
    metrics = results["metrics"]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_config(cfg, out_dir)
    names = [
        readout if feature == "M" else f"{readout}_{feature}"
        for feature in cfg.readouts_task.get("features", ["M"])
        for readout in cfg.readouts_task.readouts
    ]
    print(f"N: {cfg.N}  rounds: {int(metrics['n_rounds'])}")
    print(f"{'p':>6} {'S_R':>6} " + " ".join(f"{name:>17}" for name in names))
    for level in cfg.readouts_task.p_values:
        key = f"p{float(level):g}"
        accs = " ".join(f"{metrics[f'accuracy_{name}_{key}']:17.3f}" for name in names)
        print(f"{float(level):6.3f} {metrics[f'S_R_mean_{key}']:6.3f} {accs}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
