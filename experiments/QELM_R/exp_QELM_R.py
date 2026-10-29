#!/usr/bin/env python3
"""QELM experiment: purification decoding of a reference qubit (issue #247).

Task 1B of the experimental framework, after the scalable probes of Gullans
& Huse (arXiv:1905.05195, arXiv:1910.00020). Can a ridge decoder of the
local measurement record ``M`` predict the final state of a reference qubit
``R`` that the measurements never touch? For each measurement probability
``p`` of ``purification_task.p_values`` and each of its
``purification_task.samples_per_p`` trajectories:

1. ``R`` (qubit 0, the ancilla, ``Nr = 1``) is maximally entangled with the
   central system qubit ``N // 2`` of the Néel state ``|0101...>``
   (:func:`QELM.evaluation.purification.reference_bell_state`).
2. The ``N`` system qubits alone run ``n_rounds`` rounds
   (``purification_task.n_rounds``, default ``2 N``) of a unitary step (the
   TFIM ``U(dt)`` with ``dt = purification_task.dt``, or brickwork layers of
   a Haar or Clifford circuit) followed by a projective measurement in Z of
   each system qubit with probability ``p``. Row ``j`` of ``M`` holds the
   outcomes of round ``j``, +-1, and 0 for an unmeasured qubit. ``R`` is
   never acted on. With ``purification_task.fixed_instance`` (default) the
   dynamics are drawn once per run, the fixed reservoir of the QELM, and
   only the measurements change between trajectories.
3. At the end, the Bloch vector ``(x, y, z)`` of ``R`` in the state
   conditioned on the record is read, with its entropy ``S_R`` (1 bit: mixed,
   0: purified), and ``R`` is measured once in Z with the Born rule
   (``shot``, +-1), what an experiment records.

Each ``p`` gets its own decoder, as in the learnability experiments: its
trajectories are split into training and test sets
(``samples.train_fraction``,
:func:`~QELM.evaluation.entropy_regression.stratified_split`) and two
ridge regressions (``sklearn.linear_model.Ridge``, ``alpha =
training.alpha``, intercept fitted) read the flattened record:

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

The TFIM with ``TFIM.basis = X`` conserves the Z parity of the system, and
the two branches of the Bell pair have opposite parities: ``R`` stays
diagonal (``x = y = 0``) and ``z`` is the posterior of the parity given the
record. The TFIM task is then the learnability of a Z2 charge; the Haar
brickwork has no symmetry and is the generic case.

The unsuffixed ``nmse``, ``nmse_err``, ``r2``, ``y_test`` and ``y_pred``
are those of the ``z`` regression on the test sets pooled over ``p``. Run
from the repository root:

    uv run python experiments/QELM_R/exp_QELM_R.py models=haar_circuit N=8

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_R.npz``: ``M`` (record, ``(L, n_rounds, N)``), ``bloch``
  (``(L, 3)``), ``z``, ``S_R``, ``shot`` and ``p`` (``(L,)``), ``train`` and
  ``test`` (indices), ``y_test``, ``y_pred`` (``z`` regression) and
  ``shot_pred`` (classifier output) on the test set, ``X`` (the flattened
  record with a bias column);
- ``metrics.json``: per ``p`` the scores above, ``S_R_mean_p<p>``,
  ``S_R_std_p<p>`` and ``purified_p<p>`` (share of trajectories with ``S_R <
  0.1`` bit), and ``n_rounds``;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import Ridge
from tqdm import tqdm

from QELM import TransverseFieldIsingModel, check_qubit_counts, set_seed
from QELM.evaluation.entropy_regression import stratified_split
from QELM.evaluation.metrics import MSE, NMSE, R2, NMSE_err, accuracy
from QELM.evaluation.metrics import majority_accuracy as chance_accuracy
from QELM.evaluation.purification import (
    bayes_accuracy,
    expected_accuracy,
    reference_bell_state,
    reference_bloch,
    reference_entropy,
)
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.clifford import CliffordBrickworkCircuit
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config, save_results

# Random circuits that can replace the TFIM, by models value.
CIRCUITS: dict[str, type[BrickworkCircuit]] = {
    HaarBrickworkCircuit.name: HaarBrickworkCircuit,
    CliffordBrickworkCircuit.name: CliffordBrickworkCircuit,
}
# Seed tag of the generator of the measurements and the shots of R.
RNG_TAG = 247
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
# One unitary step of a trajectory: the state of R and the system, round j.
Step = Callable[[np.ndarray, int], np.ndarray]


def neel_state(N: int) -> np.ndarray:
    """Return ``|0101...>`` on ``N`` qubits, qubit 0 the leftmost factor."""
    index = int("01" * (N // 2) + "0" * (N % 2), 2)
    psi = np.zeros(2**N, dtype=complex)
    psi[index] = 1.0
    return psi


def _tfim_sampler(
    cfg: DictConfig, N: int, dt: float, fixed: bool
) -> Callable[[], Step]:
    """Return a function that draws a TFIM step ``U(dt)`` on the system qubits.

    The step acts on the rows of the ``(2, 2**N)`` reshaped state, so ``R``
    is left alone. With ``fixed`` the couplings are drawn once.
    """
    block = OmegaConf.to_container(cfg[TransverseFieldIsingModel.name], resolve=True)
    assert isinstance(block, dict)
    params = {**{str(k): v for k, v in block.items()}, "N": N}

    def build() -> Step:
        U_T = TransverseFieldIsingModel(params).propagator(dt=dt).T
        return lambda psi, j: (psi.reshape(2, -1) @ U_T).reshape(-1)

    if fixed:
        step = build()
        return lambda: step
    return build


def _circuit_sampler(cfg: DictConfig, n_rounds: int, fixed: bool) -> Callable[[], Step]:
    """Return a function that draws the gates of a brickwork circuit, per sample.

    The gates act on qubits ``Nr`` to ``N_total - 1``, never on ``R``. Round
    ``j`` applies brickwork layers ``j * k`` to ``(j + 1) * k - 1``, with
    ``k = layers_per_step``. With ``fixed`` the gates are drawn once.

    Raises:
        ValueError: If ``layers_per_step < 1``.
    """
    name = str(cfg.models)
    block = OmegaConf.to_container(cfg[name], resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    per_step = int(params.get("layers_per_step", 1))
    if per_step < 1:
        raise ValueError(f"{name}.layers_per_step must be >= 1, got {per_step}.")
    params["fixed_instance"] = fixed
    circuit = CIRCUITS[name]({**params, "depth": n_rounds * per_step})

    def step(psi: np.ndarray, j: int) -> np.ndarray:
        for t in range(j * per_step, (j + 1) * per_step):
            psi = circuit.apply_layer(psi, t)
        return psi

    first = True

    def draw() -> Step:
        nonlocal first
        # The constructor drew the gates of the first sample.
        if not (first or circuit.fixed_instance):
            circuit.redraw()
        first = False
        return step

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


def decode(
    X: np.ndarray,
    shot: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit the two ridge decoders of one ``p`` and score them.

    Args:
        X: Flattened records, ``(L, n_features)``.
        shot: Z outcome of ``R``, +-1, ``(L,)``.
        z: ``<Z_R>`` of the conditioned state, ``(L,)``.
        r: Length of the Bloch vector of ``R``, ``(L,)``.
        train: Training indices.
        test: Test indices.
        alpha: Ridge regularization.

    Returns:
        ``(scores, shot_pred, z_pred)``: the scores of :data:`PER_P_SCORES`
        about the decoders, and the outputs of the classifier and of the
        ``z`` regression on the test set.
    """
    classifier = Ridge(alpha=alpha).fit(X[train], shot[train])
    shot_pred = np.asarray(classifier.predict(X[test]), dtype=float)
    regression = Ridge(alpha=alpha).fit(X[train], z[train])
    z_pred = np.asarray(regression.predict(X[test]), dtype=float)
    scores = {
        "accuracy": accuracy(shot[test], shot_pred),
        "accuracy_expected": expected_accuracy(z[test], shot_pred),
        "bayes_accuracy": bayes_accuracy(z[test]),
        "bloch_accuracy": float(np.mean((1.0 + r[test]) / 2.0)),
        "majority_accuracy": chance_accuracy(shot[train], shot[test]),
        "r2_z": R2(z[test], z_pred) if np.var(z[test]) > Z_VAR_MIN else np.nan,
        "mse_z": MSE(z[test], z_pred),
    }
    return scores, shot_pred, z_pred


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories, fit the decoders of every ``p`` and score them.

    Args:
        cfg: Composed config (``load_config(..., config_name="QELM_R")`` or
            Hydra). Reads ``models``, ``N``, ``Nr`` (must be 1), the model
            block, ``purification_task``, ``samples.train_fraction`` and
            ``training.alpha``.

    Returns:
        The arrays and scores listed in the module docstring.

    Raises:
        ValueError: If ``Nr`` is not 1, ``N < 2``, ``models`` is not
            supported, a probability is outside ``[0, 1]`` or a count is too
            small.
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
    fixed = bool(task.fixed_instance)

    model_name = str(cfg.models)
    if model_name == TransverseFieldIsingModel.name:
        draw_evolution = _tfim_sampler(cfg, N, float(task.dt), fixed)
    elif model_name in CIRCUITS:
        draw_evolution = _circuit_sampler(cfg, n_rounds, fixed)
    else:
        known = ", ".join([TransverseFieldIsingModel.name, *CIRCUITS])
        raise ValueError(f"models={model_name} is not supported here. Known: {known}.")

    # The dynamics come from the global NumPy state (set_seed), the
    # measurements and the shots from their own generator, so a run is
    # reproducible for a given seed.
    seed = int(cfg.seed)
    rng = np.random.default_rng([seed, RNG_TAG])
    psi0 = reference_bell_state(neel_state(N), N // 2)

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, n_rounds, N))
    bloch = np.zeros((L, 3))
    S_R = np.zeros(L)
    shot = np.zeros(L)
    for i in tqdm(range(L)):
        evolve = draw_evolution()
        psi = psi0
        for j in range(n_rounds):
            psi = evolve(psi, j)
            psi, M[i, j] = measure_system(psi, float(p[i]), rng)
        bloch[i] = reference_bloch(psi)
        S_R[i] = reference_entropy(bloch[i])
        shot[i] = 1.0 if rng.random() < (1.0 + bloch[i, 2]) / 2.0 else -1.0
    z = bloch[:, 2]
    r = np.minimum(np.linalg.norm(bloch, axis=1), 1.0)

    split_rng = np.random.default_rng([seed, RNG_TAG, 1])
    train, test = stratified_split(p, float(cfg.samples.train_fraction), split_rng)
    X = M.reshape(L, -1)
    alpha = float(cfg.training.alpha)

    metrics: dict[str, float] = {"n_rounds": float(n_rounds)}
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
        metrics |= {f"{name}_p{level:g}": scores[name] for name in PER_P_SCORES}

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
    print(f"L: {results['z'].shape[0]}  N: {cfg.N}  model: {cfg.models}")
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
