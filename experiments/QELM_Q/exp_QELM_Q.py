#!/usr/bin/env python3
"""QELM experiment: learn the charge ``Q`` under intermediate Z measurements.

Issue #163. A preparatory noise experiment: strong projective measurements in
the Z basis, applied during the evolution with probability ``p``, disturb the
entanglement of the reservoir, and the readout has to recover the charge from
the final state alone. For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) on the ``Ns = N - 1`` system
   qubits and start from the state of :class:`AncillaChargeSectorEmbedding`,
   ``(|0>_R |Q1>_S + |1>_R |Q2>_S) / sqrt(2)`` with ``Q1 = Ns // 2`` and
   ``Q2 = Ns // 2 - 1``. The ancilla ``R`` is measured in Z first, with the
   Born rule: the outcome fixes the charge sector of the system, the label
   ``Y[i] = Q`` (``Q1`` for ``+1``, ``Q2`` for ``-1``, probability 1/2
   each). ``R`` never evolves (``U = 1_R (x) U_S``, issue #104), so measuring
   it first or last gives the same statistics; measuring it first makes the
   features below depend on ``Q``. Without it the reduced state of ``S``
   would be the same mixture of both sectors for every sample, and no
   readout of ``Tr(Z_k rho)`` could tell the two charges apart.
2. Split the total time ``evolution.t`` into ``nbr_time_evolution`` steps of
   ``t / nbr_time_evolution`` (``projective_measurement.nbr_time_evolution``,
   default 5). Between two steps, each system qubit is measured in Z with
   probability ``projective_measurement.p`` and the state collapses. The
   measurements are intermediate: there is none after the last step, so the
   final state has evolved for ``t / nbr_time_evolution`` since the last
   collapse, and ``nbr_time_evolution - 1`` rounds are applied in total.
3. The features ``X[i]`` are the final expectation values
   ``<Z_k> = Tr(Z_k rho)`` of the system qubits ``k = 1 .. Ns``, plus a bias
   column. The outcomes of the intermediate measurements are **not**
   features: they only disturb the state.

Which qubits are measured and their outcomes are drawn from their own
generator, seeded with ``(seed, 163)``, while the disorder and the label of
``R`` come from the global NumPy state seeded by ``set_seed``. So for a given
seed every ``p`` sees the same Hamiltonians and the same labels, and at
``p = 0`` the run is exactly the standard QELM without measurements, one
unitary ``exp(-i H t)`` per sample (the baseline of issue #163).

``h = 10 J`` makes the TFIM almost conserve the charge (the ``X_i X_j``
couplings change it by 0 or 2), and a measurement in Z commutes with ``Q``.
So the measurements do not erase ``Q`` itself; what they change is the
state the charge is carried by, and how far the final ``<Z_k>`` spread
around their mean.

The half-system entanglement entropy of the final state, ``S_half[i]``
(von Neumann, in bits, of the first ``Ns // 2`` system qubits; the state of
``S`` is pure once ``R`` is measured), tracks how much the measurements
disentangle the reservoir.

A ridge readout (``training`` block) is fitted on the first
``samples.train_size`` samples to predict ``Q`` and scored on the rest with
NMSE, R2 and, the primary metric, ``accuracy``: a prediction counts as right
when it falls on the side of ``(Q1 + Q2) / 2`` of the true charge. With it,
``accuracy_baseline`` (always the majority charge of the training set, the
chance level) and ``balanced_accuracy`` (0.5 for a readout that ignores the
features), as in ``experiments/QELM_M/exp_QELM_M.py`` (issue #130).

Run from the repository root:

    uv run python experiments/QELM_Q/exp_QELM_Q.py N=4 samples.L=500 \
        projective_measurement.p=0.5

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_Q.npz``: ``X`` (features, bias column included), ``Y`` (the
  charge), ``y_test``, ``y_pred``, the weights ``W``, ``S_half`` and
  ``n_measured`` (number of collapses per sample), of shape ``(L,)``;
- ``metrics.json``: ``nmse``, ``nmse_err``, ``r2``, ``accuracy``,
  ``accuracy_baseline``, ``balanced_accuracy``, ``S_half_mean``,
  ``n_measured_mean`` and the shapes used;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from QELM import TransverseFieldIsingModel, check_qubit_counts, set_seed
from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.evaluation import von_neumann_entropy
from QELM.evaluation.metrics import (
    NMSE,
    R2,
    NMSE_err,
    accuracy,
    balanced_accuracy,
    majority_accuracy,
)
from QELM.physics import Z, local_expectation
from QELM.training import ProjectiveMeasurement, RidgeReadout
from QELM.training.measurements.projective import measure_projective_rho
from QELM.utils import save_config, save_results

# Scalars written to metrics.json besides nmse, nmse_err and r2.
EXTRA_METRICS = (
    "accuracy",
    "accuracy_baseline",
    "balanced_accuracy",
    "S_half_mean",
    "n_measured_mean",
)
# Number of time-evolution steps (issue #132).
NBR_TIME_EVOLUTION = "projective_measurement.nbr_time_evolution"
# Second word of the seed of the measurement generator: the issue number.
MEASUREMENT_STREAM = 163


def evolve_with_measurements(
    rho: np.ndarray,
    U: np.ndarray,
    n_steps: int,
    P: ProjectiveMeasurement,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Apply ``n_steps`` steps ``U``, with a measurement round between two.

    Args:
        rho: Initial density matrix of shape ``(2**N, 2**N)``.
        U: One step of the evolution, same shape.
        n_steps: Number of steps, ``>= 1``.
        P: Measurement of the system qubits, each with its probability ``p``.
        rng: Generator of the measurement choices and outcomes.

    Returns:
        The final state, after the last step and without a measurement after
        it, and the number of qubits measured on the way.
    """
    U_dag = U.conj().T
    n_measured = 0
    for j in range(n_steps):
        rho = U @ rho @ U_dag
        if j < n_steps - 1:
            measure = P.measure(rho, rng=rng)
            rho = measure.rho
            n_measured += int(np.sum(np.isfinite(measure.outcomes)))
    return rho, n_measured


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout on ``Q`` and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.t``, ``samples.L``, ``samples.train_size``, ``seed``,
            the TFIM, embedding, measurement and training blocks, and
            ``projective_measurement.nbr_time_evolution`` (default 5).

    Returns:
        The arrays and metrics of the run: ``X`` (with the bias column),
        ``Y`` (the charge ``Q``), ``y_test``, ``y_pred``, ``W``, ``S_half``,
        ``n_measured``, ``nmse``, ``nmse_err``, ``r2``, ``accuracy``,
        ``accuracy_baseline``, ``balanced_accuracy``, ``S_half_mean`` and
        ``n_measured_mean``.
    """
    set_seed(cfg)
    rng = np.random.default_rng([int(cfg.seed), MEASUREMENT_STREAM])

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    N, Ns, Nr = check_qubit_counts(cfg)
    nbr_time_evolution = int(OmegaConf.select(cfg, NBR_TIME_EVOLUTION, default=5))
    if nbr_time_evolution < 1:
        raise ValueError(
            f"{NBR_TIME_EVOLUTION} must be >= 1, got {nbr_time_evolution}."
        )
    dt = float(cfg.evolution.t) / nbr_time_evolution
    rho0 = sector.rho()
    P = ProjectiveMeasurement.from_config(cfg)
    Q1, Q2 = sector.Q1, sector.Q2

    L = int(cfg.samples.L)
    X = np.zeros((L, Ns))
    Y = np.zeros(L)
    S_half = np.zeros(L)
    n_measured = np.zeros(L)
    system = list(range(Nr, N))
    half = system[: Ns // 2]

    # The TFIM block with the system size: R does not evolve (issue #104).
    tfim = OmegaConf.to_container(cfg[TransverseFieldIsingModel.name], resolve=True)
    assert isinstance(tfim, dict)
    tfim_params = {**{str(k): v for k, v in tfim.items()}, "N": Ns}
    identity_R = np.eye(2**Nr)

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel(tfim_params)
        U = np.kron(identity_R, model.propagator(dt=dt))
        # The label first: R measured in Z fixes the charge sector of S.
        rho, r = measure_projective_rho(rho0, 0, N)
        Y[i] = Q1 if r == 1 else Q2
        rho, n_measured[i] = evolve_with_measurements(
            rho, U, nbr_time_evolution, P, rng
        )
        X[i] = [local_expectation(rho, Z, k) for k in system]
        S_half[i] = von_neumann_entropy(rho, half)

    X = np.hstack([X, np.ones((L, 1))])  # bias column: RidgeReadout has no intercept

    train = int(cfg.samples.train_size)
    X_train, y_train = X[:train], Y[:train]
    X_test, y_test = X[train:], Y[train:]

    solver.fit(X_train, y_train)
    y_pred = solver.predict(X_test)
    # The two charges are told apart by the side of their midpoint.
    threshold = (Q1 + Q2) / 2

    return {
        "X": X,
        "Y": Y,
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "S_half": S_half,
        "n_measured": n_measured,
        "nmse": NMSE(y_test, y_pred),
        "nmse_err": NMSE_err(y_test, y_pred),
        "r2": R2(y_test, y_pred),
        "accuracy": accuracy(y_test, y_pred, threshold),
        "accuracy_baseline": majority_accuracy(y_train, y_test, threshold),
        "balanced_accuracy": balanced_accuracy(y_test, y_pred, threshold),
        "S_half_mean": float(np.mean(S_half)),
        "n_measured_mean": float(np.mean(n_measured)),
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(results, out_dir, Path(__file__).stem, extra_metrics=EXTRA_METRICS)
    save_config(cfg, out_dir)
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g} +- {results['nmse_err']:.2g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"accuracy: {results['accuracy']:.6g}")
    print(f"accuracy baseline (majority charge): {results['accuracy_baseline']:.6g}")
    print(f"balanced accuracy: {results['balanced_accuracy']:.6g}")
    print(f"S_half (mean, bits): {results['S_half_mean']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
