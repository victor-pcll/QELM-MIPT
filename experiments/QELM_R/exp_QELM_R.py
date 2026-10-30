#!/usr/bin/env python3
"""QELM experiment: purification decoding of a reference qubit (issue #247).

Task 1B of the experimental framework, after the scalable probes of Gullans
& Huse (arXiv:1905.05195, arXiv:1910.00020). Can a ridge decoder of the
local measurement record ``M`` predict the final state of a reference qubit
``R`` that the measurements never touch? Two local, chaotic reservoirs are
compared (issue #243, PR #244): a random Haar brickwork and the
deterministic Floquet Ising chain of Skinner, Ruhman & Nahum (PRX 9, 031009
(2019)). For each measurement probability ``p`` of
``purification_task.p_values`` and each of its
``purification_task.samples_per_p`` trajectories:

1. The system starts in the Néel state ``|0101...>`` and first runs alone,
   with ``R`` (qubit 0, the ancilla, ``Nr = 1``) in ``|0>``, for
   ``purification_task.scramble_rounds`` rounds (default ``4 N``) at the
   same ``p``, as in Gullans & Huse: it reaches its steady state, so that
   ``R`` is hidden in a scrambled state rather than in a product state.
   Then the central system qubit ``N // 2`` is measured, reset to ``|0>``
   and paired with ``R`` in the Bell state ``|00> + |11>``
   (:func:`QELM.evaluation.purification.entangle_reference`). The outcomes
   of this phase are not in the record. With ``scramble_rounds = 0``, ``R``
   is paired with the Néel state at ``t = 0``
   (:func:`QELM.evaluation.purification.reference_bell_state`).

   With ``purification_task.protocol = paper``, the protocol of Gullans &
   Huse (arXiv:1910.00020, reproduced for Clifford circuits by
   ``experiments/purification``, issue #185) instead: ``R`` is paired at
   ``t = 0`` with the central qubit of ``|0...0>``, then the gates run
   ``scramble_rounds`` rounds (default ``2 N``) without measurement.
   ``purification_task.periodic`` closes the chain (bond ``(N - 1, 0)``
   in the odd half-layers, ``N`` even), as in the paper.
2. The ``N`` system qubits alone run ``n_rounds`` rounds
   (``purification_task.n_rounds``, default ``2 N``). A round is one
   half-layer of two-qubit gates, on the even bonds ``(0, 1), (2, 3), ...``
   for even rounds and on the odd bonds ``(1, 2), (3, 4), ...`` for odd ones,
   followed by a projective measurement in Z of each system qubit with
   probability ``p``: the time step of Skinner et al. and of the diagnostics
   of PR #244. ``purification_task.dynamics`` sets the gates:

   - ``haar``: gates drawn from the Haar measure on U(4), new for every
     bond and every round. With ``purification_task.fixed_instance``
     (default) the circuit is drawn once per run, the fixed reservoir of the
     QELM, and only the measurements change between trajectories; ``false``
     draws a new circuit per trajectory;
   - ``floquet``: the same brick everywhere and at every round,
     :func:`QELM.evaluation.chaos_locality.floquet_ising_gate` with the
     couplings of ``purification_task.floquet`` (the paper's values by
     default). It has no randomness: ``fixed_instance`` changes nothing.

   Row ``j`` of ``M`` holds the outcomes of round ``j``, +-1, and 0 for an
   unmeasured qubit. ``R`` is never acted on.
3. At the end, the Bloch vector ``(x, y, z)`` of ``R`` in the state
   conditioned on the record is read, with its entropy ``S_R`` (1 bit: mixed,
   0: purified), and ``R`` is measured once in Z with the Born rule
   (``shot``, +-1), what an experiment records.

Each ``p`` gets its own decoder, as in the learnability experiments: its
trajectories are split into training and test sets
(``samples.train_fraction``,
:func:`~QELM.evaluation.entropy_regression.stratified_split`) and two
ridge regressions (``sklearn.linear_model``, intercept fitted) read the
flattened record. With ``training.alpha = null`` (``conf/QELM_R.yaml``) the
regularization of each is chosen on the training set by efficient
leave-one-out cross-validation (``RidgeCV`` over :data:`ALPHAS`) and written
as ``alpha_shot_p<p>`` and ``alpha_z_p<p>``; a number fixes it.

- a classifier of the shot of ``R``: trained on the shots, it guesses
  ``sign(prediction)``. ``accuracy_p<p>`` is its share of right guesses on
  the test shots; ``accuracy_expected_p<p>`` the mean probability that the
  guesses are right, ``(1 + s z) / 2``, which removes the noise of the test
  shots (:func:`~QELM.evaluation.purification.expected_accuracy`);
- a regression of ``z = <Z_R>``, scored by ``r2_z_p<p>`` and ``mse_z_p<p>``
  (``r2_z`` is NaN when ``z`` does not vary, as at ``p = 0`` where ``z = 0``).

Two ceilings come with them: ``bayes_accuracy_p<p>``, the mean of ``(1 +
|z|) / 2`` on the test set, the best accuracy of any decoder of the record
(:func:`~QELM.evaluation.purification.bayes_accuracy`), and
``bloch_accuracy_p<p>``, the mean of ``(1 + |r|) / 2``, the best accuracy of
a measurement of ``R`` along its own Bloch vector, whatever the axis; the
two differ when ``R`` is purified away from Z. ``majority_accuracy_p<p>`` is
the chance level. In the mixed phase ``R`` stays entangled with the system,
``z`` is close to 0 and every accuracy stays at 1/2; in the pure phase the
record carries ``R`` and the decoder can approach the ceiling.

The unsuffixed ``nmse``, ``nmse_err``, ``r2``, ``y_test`` and ``y_pred``
are those of the ``z`` regression on the test sets pooled over ``p``. Run
from the repository root:

    uv run python experiments/QELM_R/exp_QELM_R.py N=8 \\
        purification_task.dynamics=floquet

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_R.npz``: ``M`` (record, ``(L, n_rounds, N)``), ``bloch``
  (``(L, 3)``), ``z``, ``S_R``, ``shot`` and ``p`` (``(L,)``), ``train`` and
  ``test`` (indices), ``y_test``, ``y_pred`` (``z`` regression) and
  ``shot_pred`` (classifier output) on the test set, ``X`` (the flattened
  record with a bias column);
- ``metrics.json``: per ``p`` the scores above, ``S_R_mean_p<p>``,
  ``S_R_std_p<p>`` and ``purified_p<p>`` (share of trajectories with ``S_R <
  0.1`` bit), ``n_rounds``, ``scramble_rounds``, ``paper_protocol`` and
  ``periodic`` (1 or 0);
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from QELM import check_qubit_counts, set_seed
from QELM.evaluation.chaos_locality import floquet_ising_gate
from QELM.evaluation.entropy_regression import stratified_split
from QELM.evaluation.metrics import MSE, NMSE, R2, NMSE_err, accuracy
from QELM.evaluation.metrics import majority_accuracy as chance_accuracy
from QELM.evaluation.purification import (
    bayes_accuracy,
    entangle_reference,
    expected_accuracy,
    reference_bell_state,
    reference_bloch,
    reference_entropy,
)
from QELM.models.circuits import periodic_bonds
from QELM.models.circuits.base import Layer, apply_layer
from QELM.models.circuits.haar import haar_gate
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config, save_results

# Gates of the system, by purification_task.dynamics.
DYNAMICS = ("haar", "floquet")
# When R is entangled, by purification_task.protocol.
PROTOCOLS = ("steady", "paper")
# Seed tag of the generator of the measurements and the shots of R.
RNG_TAG = 247
# Regularizations tried by RidgeCV when training.alpha is null.
ALPHAS = np.logspace(-2, 4, 13)
# Entropy of R, in bits, below which a trajectory counts as purified.
PURIFIED_BELOW = 0.1
# Below this variance of z on the test set (at p = 0, z = 0 exactly), R2 is
# not defined and r2_z is NaN rather than the 1.0 of QELM.evaluation.metrics.R2.
Z_VAR_MIN = 1e-12
# Scores written per p, as <name>_p<p>.
PER_P_SCORES = (
    "accuracy",
    "accuracy_expected",
    "bayes_accuracy",
    "bloch_accuracy",
    "majority_accuracy",
    "r2_z",
    "mse_z",
    "S_R_mean",
    "S_R_std",
    "purified",
)
# The half-layers of one trajectory, one per round.
Circuit = list[Layer]


def neel_state(N: int) -> np.ndarray:
    """Return ``|0101...>`` on ``N`` qubits, qubit 0 the leftmost factor."""
    index = int("01" * (N // 2) + "0" * (N % 2), 2)
    psi = np.zeros(2**N, dtype=complex)
    psi[index] = 1.0
    return psi


def half_layer_bonds(N: int, t: int, periodic: bool = False) -> list[tuple[int, int]]:
    """Return the bonds of half-layer ``t``, as qubits of the state with ``R`` first.

    System qubit ``k`` is qubit ``k + 1`` of the state. Even ``t``: the
    system bonds ``(0, 1), (2, 3), ...``; odd ``t``: ``(1, 2), (3, 4), ...``,
    and ``(N - 1, 0)`` with ``periodic`` (``N`` even,
    :func:`~QELM.models.circuits.periodic_bonds`).
    """
    if periodic:
        return [(i + 1, j + 1) for i, j in periodic_bonds(N, t)]
    return [(k + 1, k + 2) for k in range(t % 2, N - 1, 2)]


def circuit_sampler(
    dynamics: str,
    N: int,
    n_rounds: int,
    fixed: bool,
    floquet: dict[str, float],
    rng: np.random.Generator,
    periodic: bool = False,
) -> Callable[[], Circuit]:
    """Return a function that gives the ``n_rounds`` half-layers of a trajectory.

    Args:
        dynamics: ``"haar"`` or ``"floquet"``.
        N: Number of system qubits.
        n_rounds: Number of half-layers.
        fixed: Haar only: ``True`` draws the circuit once, ``False`` per call.
        floquet: Couplings of :func:`floquet_ising_gate`.
        rng: Generator of the Haar gates.
        periodic: Periodic boundaries (:func:`half_layer_bonds`).

    Raises:
        ValueError: If ``dynamics`` is unknown.
    """
    if dynamics == "floquet":
        gate = floquet_ising_gate(**floquet)
        circuit = [
            [(i, j, gate) for i, j in half_layer_bonds(N, t, periodic)]
            for t in range(n_rounds)
        ]
        return lambda: circuit
    if dynamics != "haar":
        raise ValueError(
            f"purification_task.dynamics must be one of {DYNAMICS}, got {dynamics!r}."
        )

    def draw() -> Circuit:
        return [
            [(i, j, haar_gate(rng)) for i, j in half_layer_bonds(N, t, periodic)]
            for t in range(n_rounds)
        ]

    if fixed:
        circuit = draw()
        return lambda: circuit
    return draw


def measure_system(
    psi: np.ndarray, p: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Measure each system qubit (1 to ``N``) in Z with probability ``p``, in order.

    ``R``, qubit 0, is never measured.

    Returns:
        ``(psi, outcomes)``: the collapsed state and the outcome of each
        system qubit, +-1, or 0 when it was not measured.
    """
    n_total = psi.shape[0].bit_length() - 1
    outcomes = np.zeros(n_total - 1)
    for k in range(n_total - 1):
        if rng.random() < p:
            psi, outcomes[k] = measure_projective_psi(psi, k + 1, n_total, rng)
    return psi, outcomes


def _with_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _ridge(alpha: float | None) -> Ridge | RidgeCV:
    return RidgeCV(alphas=ALPHAS) if alpha is None else Ridge(alpha=alpha)


def _alpha(model: Ridge | RidgeCV) -> float:
    return float(model.alpha_ if isinstance(model, RidgeCV) else model.alpha)


def decode(
    X: np.ndarray,
    shot: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alpha: float | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit the two ridge decoders of one ``p`` and score them.

    Args:
        X: Flattened records, ``(L, n_features)``.
        shot: Z outcome of ``R``, +-1, ``(L,)``.
        z: ``<Z_R>`` of the conditioned state, ``(L,)``.
        r: Length of the Bloch vector of ``R``, ``(L,)``.
        train: Training indices.
        test: Test indices.
        alpha: Ridge regularization, or ``None`` to choose it by
            leave-one-out cross-validation on the training set.

    Returns:
        ``(scores, shot_pred, z_pred)``: the scores of :data:`PER_P_SCORES`
        about the decoders with ``alpha_shot`` and ``alpha_z``, and the
        outputs of the classifier and of the ``z`` regression on the test
        set.
    """
    classifier = _ridge(alpha).fit(X[train], shot[train])
    shot_pred = np.asarray(classifier.predict(X[test]), dtype=float)
    regression = _ridge(alpha).fit(X[train], z[train])
    z_pred = np.asarray(regression.predict(X[test]), dtype=float)
    scores = {
        "accuracy": accuracy(shot[test], shot_pred),
        "accuracy_expected": expected_accuracy(z[test], shot_pred),
        "bayes_accuracy": bayes_accuracy(z[test]),
        "bloch_accuracy": float(np.mean((1.0 + r[test]) / 2.0)),
        "majority_accuracy": chance_accuracy(shot[train], shot[test]),
        "r2_z": R2(z[test], z_pred) if np.var(z[test]) > Z_VAR_MIN else np.nan,
        "mse_z": MSE(z[test], z_pred),
        "alpha_shot": _alpha(classifier),
        "alpha_z": _alpha(regression),
    }
    return scores, shot_pred, z_pred


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories, fit the decoders of every ``p`` and score them.

    Args:
        cfg: Composed config (``load_config(..., config_name="QELM_R")`` or
            Hydra). Reads ``N``, ``Nr`` (must be 1), ``purification_task``,
            ``samples.train_fraction`` and ``training.alpha``.

    Returns:
        The arrays and scores listed in the module docstring.

    Raises:
        ValueError: If ``Nr`` is not 1, ``N < 2``, ``dynamics`` is unknown,
            a probability is outside ``[0, 1]`` or a count is too small.
    """
    set_seed(cfg)
    N, Nr, N_total = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"exp_QELM_R needs the reference qubit: set Nr=1, got {Nr}.")
    if N < 2:
        raise ValueError(f"exp_QELM_R needs N >= 2 system qubits, got N={N}.")
    task = cfg.purification_task
    levels = np.asarray([float(v) for v in task.p_values])
    if levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"purification_task.p_values must be in [0, 1], got {levels}.")
    if np.unique(levels).size != levels.size:
        raise ValueError(f"purification_task.p_values must be distinct, got {levels}.")
    per_p = int(task.samples_per_p)
    if per_p < 4:
        raise ValueError(f"purification_task.samples_per_p must be >= 4, got {per_p}.")
    n_rounds = 2 * N if task.n_rounds is None else int(task.n_rounds)
    if n_rounds < 1:
        raise ValueError(f"purification_task.n_rounds must be >= 1, got {n_rounds}.")
    protocol = str(task.get("protocol", "steady"))
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"purification_task.protocol must be one of {PROTOCOLS}, got {protocol!r}."
        )
    paper = protocol == "paper"
    periodic = bool(task.get("periodic", False))
    if periodic and N % 2:
        raise ValueError(f"purification_task.periodic needs an even N, got N={N}.")
    default_scramble = 2 * N if paper else 4 * N
    scramble = (
        default_scramble if task.scramble_rounds is None else int(task.scramble_rounds)
    )
    if scramble < 0:
        raise ValueError(
            f"purification_task.scramble_rounds must be >= 0, got {scramble}."
        )

    # The gates, the measurements and the shots have their own generators,
    # so a run is reproducible for a given seed.
    seed = int(cfg.seed)
    rng = np.random.default_rng([seed, RNG_TAG])
    draw_circuit = circuit_sampler(
        str(task.dynamics),
        N,
        scramble + n_rounds,
        bool(task.fixed_instance),
        {str(k): float(v) for k, v in task.floquet.items()},
        np.random.default_rng([seed, RNG_TAG, 2]),
        periodic,
    )
    # Paper protocol: R is paired at t = 0 with the central qubit of
    # |0...0>, then the gates scramble without measurement. Steady protocol:
    # without scrambling, R is paired at t = 0 with the central qubit of the
    # Néel state; with it, the system first runs alone with R in |0>.
    if paper:
        zeros = np.zeros(2**N, dtype=complex)
        zeros[0] = 1.0
        psi0 = reference_bell_state(zeros, N // 2)
    elif scramble == 0:
        psi0 = reference_bell_state(neel_state(N), N // 2)
    else:
        psi0 = np.kron(np.array([1.0, 0.0]), neel_state(N))

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, n_rounds, N))
    bloch = np.zeros((L, 3))
    S_R = np.zeros(L)
    shot = np.zeros(L)
    for i in tqdm(range(L)):
        circuit = draw_circuit()
        psi = psi0
        for j in range(scramble):
            psi = apply_layer(psi, circuit[j])
            if not paper:
                psi, _ = measure_system(psi, float(p[i]), rng)
        if scramble and not paper:
            psi, _ = entangle_reference(psi, N // 2, rng)
        for j in range(n_rounds):
            psi = apply_layer(psi, circuit[scramble + j])
            psi, M[i, j] = measure_system(psi, float(p[i]), rng)
        bloch[i] = reference_bloch(psi)
        S_R[i] = reference_entropy(bloch[i])
        shot[i] = 1.0 if rng.random() < (1.0 + bloch[i, 2]) / 2.0 else -1.0
    z = bloch[:, 2]
    r = np.minimum(np.linalg.norm(bloch, axis=1), 1.0)

    split_rng = np.random.default_rng([seed, RNG_TAG, 1])
    train, test = stratified_split(p, float(cfg.samples.train_fraction), split_rng)
    X = M.reshape(L, -1)
    alpha = None if cfg.training.alpha is None else float(cfg.training.alpha)

    metrics: dict[str, float] = {
        "n_rounds": float(n_rounds),
        "scramble_rounds": float(scramble),
        "paper_protocol": float(paper),
        "periodic": float(periodic),
    }
    z_pred = np.zeros(test.size)
    shot_pred = np.zeros(test.size)
    for level in levels:
        in_p = p == level
        scores, shot_level, z_level = decode(
            X, shot, z, r, train[in_p[train]], test[in_p[test]], alpha
        )
        shot_pred[in_p[test]] = shot_level
        z_pred[in_p[test]] = z_level
        scores["S_R_mean"] = float(S_R[in_p].mean())
        scores["S_R_std"] = float(S_R[in_p].std())
        scores["purified"] = float(np.mean(S_R[in_p] < PURIFIED_BELOW))
        names = (*PER_P_SCORES, "alpha_shot", "alpha_z")
        metrics |= {f"{name}_p{level:g}": scores[name] for name in names}

    y_test = z[test]
    return {
        "M": M,
        "bloch": bloch,
        "z": z,
        "S_R": S_R,
        "shot": shot,
        "p": p,
        "train": train,
        "test": test,
        "X": _with_bias(X),
        "y_test": y_test,
        "y_pred": z_pred,
        "shot_pred": shot_pred,
        "nmse": NMSE(y_test, z_pred),
        "nmse_err": NMSE_err(y_test, z_pred),
        "r2": R2(y_test, z_pred),
        "N_total": float(N_total),
        **metrics,
    }


def scalar_keys(results: dict[str, Any]) -> list[str]:
    """Return the keys of the scalar scores, for metrics.json."""
    skip = {"nmse", "nmse_err", "r2"}
    return [
        key
        for key, value in results.items()
        if isinstance(value, float) and key not in skip
    ]


@hydra.main(version_base="1.3", config_path="../../conf", config_name="QELM_R")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/QELM_R.yaml`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(
        results, out_dir, Path(__file__).stem, extra_metrics=scalar_keys(results)
    )
    save_config(cfg, out_dir)
    dynamics = cfg.purification_task.dynamics
    print(f"L: {results['z'].shape[0]}  N: {cfg.N}  dynamics: {dynamics}")
    print(f"{'p':>6} {'S_R':>6} {'acc':>6} {'acc_exp':>8} {'bayes':>6} {'r2_z':>7}")
    for level in cfg.purification_task.p_values:
        key = f"p{float(level):g}"
        print(
            f"{float(level):6.3f} {results[f'S_R_mean_{key}']:6.3f} "
            f"{results[f'accuracy_{key}']:6.3f} "
            f"{results[f'accuracy_expected_{key}']:8.3f} "
            f"{results[f'bayes_accuracy_{key}']:6.3f} "
            f"{results[f'r2_z_{key}']:7.3f}"
        )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
