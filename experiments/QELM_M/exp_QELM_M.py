#!/usr/bin/env python3
"""QELM experiment: learn the ancilla from the measurement record ``M``.

For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) on the ``Ns = N - 1`` system
   qubits and start from the state of :class:`AncillaChargeSectorEmbedding`:
   the ancilla ``R`` (qubit 0) entangled with the charge of the system.
   ``R`` is a reference qubit, as in Barratt et al. (2022) and Agrawal et
   al. (2024): the TFIM acts on the system only, ``U = 1_R (x) U_S``
   (issue #104). It keeps, in its Z basis, which charge sector the system
   started in.
2. Split the total time ``evolution.t`` into ``nbr_time_evolution`` steps
   of ``t / nbr_time_evolution``
   (``projective_measurement.nbr_time_evolution``, default 5, independent
   of the qubit counts, issues #123 and #132). After
   each step, measure the system qubits with
   :class:`ProjectiveMeasurement` (each with probability
   ``projective_measurement.p``) and keep the collapsed state: row ``j`` of
   ``M[i]`` is the record of the ``j``-th measurement (issue #102).
3. The features ``X[i]`` are the flattened record ``M[i]`` of shape
   ``(nbr_time_evolution, Ns)``, an unmeasured qubit counting as 0 (issue #91).
   The target ``Y[i]`` is the outcome, +1 or -1, of a projective measurement
   of ``R`` in Z after the last step, drawn with the Born rule (issue #104).
   ``R`` never evolves, so measuring it at the end or at the start gives the
   same statistics: ``Y[i]`` is the initial charge sector of the system
   (+1 for ``Ns // 2``, -1 for ``Ns // 2 - 1``), a label fixed before any
   measurement of the system, +1 or -1 with probability 1/2 whatever ``p``.
   The readout learns that hidden charge from the record, the learnability
   task of Barratt et al. ``Z0[i] = <Z_0>`` on the final state, before ``R``
   is measured, is kept too: ``(1 + Z0[i]) / 2`` is the probability that
   ``Y[i] = +1`` given the record and the Hamiltonian.

   Before issue #104 the target was ``<Z_0>`` itself and ``R`` evolved with
   the TFIM. The TFIM conserves the parity of the total charge, so at
   ``p = 1`` the last round fixed ``<Z_0> = +-1`` as the parity of the
   outcomes, whatever the dynamics, and the target changed with ``p``.

4. The entanglement entropy of the ancilla with the system in the final
   state, ``S(rho_R)`` with ``rho_R = Tr_S(rho)``, is recorded twice: the
   von Neumann entropy ``S_vN[i]`` and the Rényi entropy of order 1
   ``S_renyi_1[i]`` (issue #119). Order 1 is the limit ``n -> 1`` of the
   Rényi family, which is the von Neumann entropy, so both columns hold the
   same values; the second one keeps the Rényi family in the outputs. Both
   are in bits. ``S = 0`` when the measurements have purified the ancilla
   (it is fixed by the record), ``S = 1`` when it is still maximally
   entangled with the system: the purification order parameter of the
   measurement-induced phase transition.

5. The system sector ``S`` (the ``Ns`` qubits, ancilla excluded) gets its
   own entropies in the final state (issue #124): the von Neumann entropy
   ``S_vN_sys[i]`` and the Rényi entropy of order 2 ``S_renyi_2_sys[i]`` of
   ``rho_S = Tr_R(rho)``, and the mean over the system qubits ``k`` of the
   single-qubit entropies ``S(rho_k)``, ``S_vN_1q[i]`` and
   ``S_renyi_2_1q[i]``. The global state stays pure (unitary steps and
   projective measurements), so ``S_n(rho_S) = S_n(rho_R)`` for every
   order: ``S_vN_sys`` repeats ``S_vN``, and ``S_renyi_2_sys`` is the
   Rényi-2 entropy of the ancilla as well. The single-qubit means are new
   information: they drop to 0 for the qubits measured at the last step and
   track how entangled each system qubit is with the rest. All in bits.

A ridge readout (``training`` block) is fitted on the first
``samples.train_size`` samples and scored on the rest with NMSE and R2.
``RidgeReadout`` fits no intercept, so a constant column of ones is appended
to the features. Besides NMSE and R2, the readout is scored as a classifier
of the sign of ``Y`` (``accuracy``, issue #124). The two labels are drawn
with probability 1/2 each, but a finite test set is never exactly balanced,
and a readout that learns nothing, as at ``p = 0`` where every feature is
0, predicts the majority sign of its training set (issue #130). So two more
scores go with it: ``accuracy_baseline``, the accuracy of always predicting
the majority sign of the training targets (the chance level of
``accuracy``), and ``balanced_accuracy``, the mean over the two signs of the
share recovered, which is 0.5 for a readout that ignores the features. The
ceiling is ``accuracy_optimal``, the accuracy on the test samples of the
Bayes classifier ``sign(Z0)``, which also knows the Hamiltonian of each
sample: the mean of ``(1 + |Z0|) / 2``. It is 0.5 at ``p = 0`` (the record
holds nothing, ``Z0 = 0``) and 1 when the record fixes the charge.

Run from the repository root:

    uv run python experiments/QELM_M/exp_QELM_M.py N=4 samples.L=500

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_M.npz``: ``M`` (measurement record), ``X`` (features, bias
  column included), ``Y``, ``Z0``,
  ``y_test``, ``y_pred``, the weights ``W`` and the entropies ``S_vN``,
  ``S_renyi_1``, ``S_vN_sys``, ``S_renyi_2_sys``, ``S_vN_1q`` and
  ``S_renyi_2_1q`` of shape ``(L,)``;
- ``metrics.json``: ``nmse``, ``nmse_err``, ``r2``, ``accuracy``,
  ``accuracy_baseline``, ``balanced_accuracy``, ``accuracy_optimal``, the
  shapes used and the
  sample mean ``<name>_mean`` of each entropy;
- ``.hydra/config.yaml``: the full composed config, to reproduce the run.
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
from QELM.evaluation import (
    mean_single_qubit_entropy,
    renyi_entropy,
    von_neumann_entropy,
)
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
from QELM.utils import save_results

# Per-sample entropies, each saved to the .npz with its mean in metrics.json.
ENTROPIES = (
    "S_vN",
    "S_renyi_1",
    "S_vN_sys",
    "S_renyi_2_sys",
    "S_vN_1q",
    "S_renyi_2_1q",
)
# Scalars written to metrics.json besides nmse, nmse_err and r2.
EXTRA_METRICS = (
    "accuracy",
    "accuracy_baseline",
    "balanced_accuracy",
    "accuracy_optimal",
    *(f"{name}_mean" for name in ENTROPIES),
)
# The ancilla R, qubit 0, is the subsystem A of the entropies.
SECTOR_A = [0]
# Number of time-evolution steps, each followed by a measurement (issue #132).
NBR_TIME_EVOLUTION = "projective_measurement.nbr_time_evolution"


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.t``, ``samples.L``, ``samples.train_size``, the TFIM,
            embedding, measurement and training blocks, and the optional
            ``projective_measurement.nbr_time_evolution`` (default 5).

    Returns:
        The arrays and metrics of the run: ``M`` (record of shape
        ``(L, nbr_time_evolution, Ns)``, ``NaN`` for an unmeasured qubit), ``X`` (with
        the bias column), ``Y`` (the measured label of ``R``, +-1), ``Z0``
        (``<Z_0>`` before that measurement), ``y_test``, ``y_pred``, ``W``,
        ``nmse``, ``nmse_err``, ``r2``, ``accuracy``, ``accuracy_baseline``,
        ``balanced_accuracy``, ``accuracy_optimal``,
        the entropies of :data:`ENTROPIES` (bits, one
        per sample) and their means ``<name>_mean``.
    """
    set_seed(cfg)

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    # Qubit counts from the config, not from the model or the embedding
    # (issue #88): N in total, Ns system qubits, Nr ancillas.
    N, Ns, Nr = check_qubit_counts(cfg)
    # The number of time-evolution steps is its own parameter, not tied to
    # N, Ns or Nr (issue #123). It sets when the system is measured, so it
    # lives in the measurement block (issue #132). The old keys fail loudly
    # instead of being silently ignored.
    for old in ("experiment.n_measurements", "experiment.nbr_time_evolution"):
        if OmegaConf.select(cfg, old) is not None:
            raise ValueError(
                f"{old} was moved to {NBR_TIME_EVOLUTION} (issues #123, #132)."
            )
    nbr_time_evolution = int(OmegaConf.select(cfg, NBR_TIME_EVOLUTION, default=5))
    if nbr_time_evolution < 1:
        raise ValueError(
            f"{NBR_TIME_EVOLUTION} must be >= 1, got {nbr_time_evolution}."
        )
    dt = float(cfg.evolution.t) / nbr_time_evolution
    # Every sample starts from the embedded state, not from the collapsed
    # state of the previous sample (issue #101).
    rho0 = sector.rho()
    P = ProjectiveMeasurement.from_config(cfg)

    L = int(cfg.samples.L)
    M = np.zeros((L, nbr_time_evolution, Ns))
    Y = np.zeros(L)
    Z0 = np.zeros(L)
    S = {name: np.zeros(L) for name in ENTROPIES}
    # The system sector: every qubit but the Nr ancillas.
    system = list(range(Nr, N))

    # The TFIM block with the system size: R does not evolve (issue #104).
    tfim = OmegaConf.to_container(cfg[TransverseFieldIsingModel.name], resolve=True)
    assert isinstance(tfim, dict)
    tfim_params = {**{str(k): v for k, v in tfim.items()}, "N": Ns}
    identity_R = np.eye(2**Nr)

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel(tfim_params)
        # U(dt) once per sample: model.evolve rebuilds exp(-i H dt) v from
        # products H @ v at every call, which gets slow when ||H|| dt is
        # large (h = 10 J, dt = t / nbr_time_evolution), while the
        # nbr_time_evolution steps of a sample all use the same U.
        U = np.kron(identity_R, model.propagator(dt=dt))
        U_dag = U.conj().T
        rho = rho0
        for j in range(nbr_time_evolution):
            rho = U @ rho @ U_dag
            measure = P.measure(rho)
            rho = measure.rho
            M[i, j] = measure.outcomes
        # <Z_0> on the ancilla from its 2 x 2 reduced state (issue #95).
        Z0[i] = local_expectation(rho, Z, 0)
        S["S_vN"][i] = von_neumann_entropy(rho, SECTOR_A)
        S["S_renyi_1"][i] = renyi_entropy(rho, 1, SECTOR_A)
        S["S_vN_sys"][i] = von_neumann_entropy(rho, system)
        S["S_renyi_2_sys"][i] = renyi_entropy(rho, 2, system)
        S["S_vN_1q"][i] = mean_single_qubit_entropy(rho, system)
        S["S_renyi_2_1q"][i] = mean_single_qubit_entropy(rho, system, n=2)
        # The label: R measured in Z, after the entropies of the final state
        # (issue #104). R never evolves, so this is its initial outcome.
        _, Y[i] = measure_projective_rho(rho, 0, N)

    # Unmeasured qubits (p < 1) have NaN outcomes; the regression needs
    # numbers, so they count as 0, <Z> of a fully mixed qubit (issue #91).
    X = np.nan_to_num(M.reshape(L, -1), nan=0.0)
    X = np.hstack([X, np.ones((L, 1))])  # bias column: RidgeReadout has no intercept

    train = int(cfg.samples.train_size)
    X_train, y_train = X[:train], Y[:train]
    X_test, y_test = X[train:], Y[train:]

    solver.fit(X_train, y_train)
    y_pred = solver.predict(X_test)

    return {
        "M": M,
        "X": X,
        "Y": Y,
        "Z0": Z0,
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "nmse": NMSE(y_test, y_pred),
        "nmse_err": NMSE_err(y_test, y_pred),
        "r2": R2(y_test, y_pred),
        "accuracy": accuracy(y_test, y_pred),
        "accuracy_baseline": majority_accuracy(y_train, y_test),
        "balanced_accuracy": balanced_accuracy(y_test, y_pred),
        "accuracy_optimal": float(np.mean((1 + np.abs(Z0[train:])) / 2)),
        **S,
        **{f"{name}_mean": float(np.mean(values)) for name, values in S.items()},
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
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g} +- {results['nmse_err']:.2g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"accuracy: {results['accuracy']:.6g}")
    print(f"accuracy baseline (majority sign): {results['accuracy_baseline']:.6g}")
    print(f"balanced accuracy: {results['balanced_accuracy']:.6g}")
    print(f"optimal accuracy (Bayes, sign Z0): {results['accuracy_optimal']:.6g}")
    for name in ENTROPIES:
        print(f"{name} (mean, bits): {results[f'{name}_mean']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
