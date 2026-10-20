#!/usr/bin/env python3
"""QELM experiment: learn the ancilla from the measurement record ``M``.

For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) and start from the state of
   :class:`AncillaChargeSectorEmbedding`: the ancilla ``R`` (qubit 0)
   entangled with the charge of the ``N - 1`` system qubits.
2. Split the total time ``evolution.t`` into ``n_meas`` steps of
   ``t / n_meas`` (``experiment.n_measurements``, default ``Ns``). After
   each step, measure the system qubits with
   :class:`ProjectiveMeasurement` (each with probability
   ``projective_measurement.p``) and keep the collapsed state: row ``j`` of
   ``M[i]`` is the record of the ``j``-th measurement (issue #102).
3. The features ``X[i]`` are the flattened record ``M[i]`` of shape
   ``(n_meas, Ns)``, an unmeasured qubit counting as 0 (issue #91); the
   target ``Y[i]`` is ``<Z_0>`` on the ancilla in the final state.

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

A ridge readout (``training`` block) is fitted on the first
``samples.train_size`` samples and scored on the rest with NMSE and R2.
``RidgeReadout`` fits no intercept, so a constant column of ones is appended
to the features.

Run from the repository root:

    uv run python experiments/QELM_M/exp_QELM_M.py N=4 samples.L=500

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_M.npz``: ``M`` (measurement record), ``X`` (features, bias
  column included), ``Y``,
  ``y_test``, ``y_pred``, the weights ``W`` and the entropies ``S_vN`` and
  ``S_renyi_1`` of shape ``(L,)``;
- ``metrics.json``: ``nmse``, ``r2``, the shapes used and the sample means
  ``S_vN_mean`` and ``S_renyi_1_mean``;
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
from QELM.evaluation import renyi_entropy, von_neumann_entropy
from QELM.evaluation.metrics import NMSE, R2
from QELM.physics import Z, local_expectation
from QELM.training import ProjectiveMeasurement, RidgeReadout
from QELM.utils import save_results

# Scalars written to metrics.json besides nmse and r2.
ENTROPY_METRICS = ("S_vN_mean", "S_renyi_1_mean")
# The ancilla R, qubit 0, is the subsystem A of the entropies.
SECTOR_A = [0]


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.t``, ``samples.L``, ``samples.train_size``, the TFIM,
            embedding, measurement and training blocks, and the optional
            ``experiment.n_measurements`` (default ``Ns = N - Nr``).

    Returns:
        The arrays and metrics of the run: ``M`` (record of shape
        ``(L, n_meas, Ns)``, ``NaN`` for an unmeasured qubit), ``X`` (with
        the bias column), ``Y``, ``y_test``, ``y_pred``, ``W``, ``nmse``,
        ``r2``, the ancilla entropies ``S_vN`` and ``S_renyi_1`` (bits, one
        per sample) and their means ``S_vN_mean`` and ``S_renyi_1_mean``.
    """
    set_seed(cfg)

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    # Qubit counts from the config, not from the model or the embedding
    # (issue #88): N in total, Ns system qubits, Nr ancillas.
    N, Ns, Nr = check_qubit_counts(cfg)
    n_meas = int(OmegaConf.select(cfg, "experiment.n_measurements", default=Ns))
    if n_meas < 1:
        raise ValueError(f"experiment.n_measurements must be >= 1, got {n_meas}.")
    dt = float(cfg.evolution.t) / n_meas
    # Every sample starts from the embedded state, not from the collapsed
    # state of the previous sample (issue #101).
    rho0 = sector.rho()
    P = ProjectiveMeasurement.from_config(cfg)

    L = int(cfg.samples.L)
    M = np.zeros((L, n_meas, Ns))
    Y = np.zeros(L)
    S_vN = np.zeros(L)
    S_renyi_1 = np.zeros(L)

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel.from_config(cfg)
        # U(dt) once per sample: model.evolve rebuilds exp(-i H dt) v from
        # products H @ v at every call, which gets slow when ||H|| dt is
        # large (h = 10 J, dt = t / n_meas), while the n_meas steps of a
        # sample all use the same U.
        U = model.propagator(dt=dt)
        U_dag = U.conj().T
        rho = rho0
        for j in range(n_meas):
            rho = U @ rho @ U_dag
            measure = P.measure(rho)
            rho = measure.rho
            M[i, j] = measure.outcomes
        # <Z_0> on the ancilla from its 2 x 2 reduced state (issue #95).
        Y[i] = local_expectation(rho, Z, 0)
        S_vN[i] = von_neumann_entropy(rho, SECTOR_A)
        S_renyi_1[i] = renyi_entropy(rho, 1, SECTOR_A)

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
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "nmse": NMSE(y_test, y_pred),
        "r2": R2(y_test, y_pred),
        "S_vN": S_vN,
        "S_renyi_1": S_renyi_1,
        "S_vN_mean": float(np.mean(S_vN)),
        "S_renyi_1_mean": float(np.mean(S_renyi_1)),
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(results, out_dir, Path(__file__).stem, extra_metrics=ENTROPY_METRICS)
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"S_vN (mean, bits): {results['S_vN_mean']:.6g}")
    print(f"S_renyi_1 (mean, bits): {results['S_renyi_1_mean']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
