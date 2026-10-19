#!/usr/bin/env python3
"""QELM experiment: learn the ancilla from the measurement record ``M``.

For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) and start from the state of
   :class:`AncillaChargeSectorEmbedding`: the ancilla ``R`` (qubit 0)
   entangled with the charge of the ``N - 1`` system qubits.
2. Split the total time ``evolution.t`` into ``n_meas`` steps of
   ``t / n_meas`` (``experiment.n_measurements``, default ``N - 1``). After
   each step, measure the system qubits with
   :class:`ProjectiveMeasurement` (each with probability
   ``projective_measurement.p``) and keep the collapsed state: row ``j`` of
   ``M[i]`` is the record of the ``j``-th measurement (issue #102).
3. The features ``X[i]`` are the flattened record ``M[i]`` of shape
   ``(n_meas, N - 1)``, an unmeasured qubit counting as 0 (issue #91); the
   target ``Y[i]`` is ``<Z_0>`` on the ancilla in the final state.

A ridge readout (``training`` block) is fitted on the first
``samples.train_size`` samples and scored on the rest with NMSE and R2.
``RidgeReadout`` fits no intercept, so a constant column of ones is appended
to the features.

Run from the repository root:

    uv run python experiments/QELM_M/exp_QELM_M.py TFIM.N=4 samples.L=500

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_M.npz``: ``M`` (measurement record), ``X`` (features, bias
  column included), ``Y``,
  ``y_test``, ``y_pred`` and the weights ``W``;
- ``metrics.json``: ``nmse``, ``r2`` and the shapes used;
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

from QELM import TransverseFieldIsingModel, set_seed
from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.evaluation.metrics import NMSE, R2
from QELM.training import ProjectiveMeasurement, RidgeReadout, SingleObservable
from QELM.utils import save_results


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.t``, ``samples.L``, ``samples.train_size``, the TFIM,
            embedding, measurement and training blocks, and the optional
            ``experiment.n_measurements`` (default ``N - 1``).

    Returns:
        The arrays and metrics of the run: ``M`` (record of shape
        ``(L, n_meas, N - 1)``, ``NaN`` for an unmeasured qubit), ``X`` (with
        the bias column), ``Y``, ``y_test``, ``y_pred``, ``W``, ``nmse`` and ``r2``.
    """
    set_seed(cfg)

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    N = sector.N
    n_meas = int(OmegaConf.select(cfg, "experiment.n_measurements", default=N - 1))
    if n_meas < 1:
        raise ValueError(f"experiment.n_measurements must be >= 1, got {n_meas}.")
    dt = float(cfg.evolution.t) / n_meas
    Z0 = SingleObservable({"N": N, "observables": ["Z"]}).operators()[0]
    # Every sample starts from the embedded state, not from the collapsed
    # state of the previous sample (issue #101).
    rho0 = sector.rho()
    P = ProjectiveMeasurement.from_config(cfg)

    L = int(cfg.samples.L)
    M = np.zeros((L, n_meas, N - 1))
    Y = np.zeros(L)

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel.from_config(cfg)
        rho = rho0
        for j in range(n_meas):
            rho = model.evolve(rho, dt=dt)
            measure = P.measure(rho)
            rho = measure.rho
            M[i, j] = measure.outcomes
        # Full 2**N x 2**N product: costly for large N (issue #95).
        Y[i] = np.real(np.trace(Z0 @ rho))

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
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(results, out_dir, Path(__file__).stem)
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
