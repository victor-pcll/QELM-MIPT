#!/usr/bin/env python3
"""Projective-readout experiment: learn the ancilla from measurement records.

For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) and prepare the initial state
   ``rho`` with :class:`AncillaChargeSectorEmbedding`: the ancilla ``R``
   (qubit 0) entangled with the charge of the ``N - 1`` system qubits.
2. Evolve ``rho`` for ``evolution.n_steps`` steps of ``evolution.dt``.
   After every step but the last, measure the system qubits projectively in
   the Z basis (:class:`ProjectiveMeasurement`) and store the outcomes
   (``+1``, ``-1``, or ``nan`` for a qubit skipped when ``p < 1``). In the
   features ``Phi`` a skipped qubit counts as ``0``: the regression needs
   numbers, and ``0`` is ``<Z>`` of a fully mixed qubit, i.e. no information.
3. After the last step, store ``<Z_k>`` for the system qubits
   ``k = 1 .. N-1`` as the last row of features, and ``<Z_0>`` on the
   ancilla as the target ``Y[i]``.

The features ``Phi`` of shape ``(L, n_steps, N - 1)`` are flattened to
``(L, n_steps * (N - 1))`` and a ridge readout (``training`` block) is fitted
on the first ``samples.train_size`` samples, then scored on the rest with
NMSE and R2.

By default the state collapses after each measurement (``rho`` becomes the
post-measurement state). Pass ``+experiment.backaction=false`` to keep the
unmeasured ``rho`` and only record the outcomes.

Run from the repository root:

    uv run python experiments/projective/projective_readout.py TFIM.N=6 samples.L=1000

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``projective_readout.npz``: ``Phi``, ``Y``, ``M`` (measurement record),
  ``X`` (final ``<Z_k>``), ``y_test``, ``y_pred`` and the weights ``W``;
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
            ``evolution.n_steps``, ``samples.L``, ``samples.train_size``, the
            TFIM, embedding, measurement, readout and training blocks, and
            the optional ``experiment.backaction`` (default ``True``).

    Returns:
        The arrays and metrics of the run: ``Phi``, ``Y``, ``M``, ``X``,
        ``y_test``, ``y_pred``, ``W``, ``nmse`` and ``r2``.
    """
    set_seed(cfg)
    backaction = bool(OmegaConf.select(cfg, "experiment.backaction", default=True))

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    P = ProjectiveMeasurement.from_config(cfg)
    N = sector.N
    # <Z_i> for i = 0 .. N-1, built once: Z_ops[0] is the ancilla.
    Z_ops = SingleObservable({"N": N, "observables": ["Z"]}).operators()

    n_steps = int(cfg.evolution.n_steps)
    N_M = n_steps - 1  # number of measured steps; step N_M is the last one
    temps = int(cfg.samples.L)

    M = np.zeros((temps, N_M, N - 1))
    X = np.zeros((temps, N - 1))
    Phi = np.zeros((temps, n_steps, N - 1))
    Y = np.zeros(temps)

    for i in tqdm(range(temps)):
        model = TransverseFieldIsingModel.from_config(cfg)
        rho = sector.rho()
        for j in range(n_steps):
            rho = model.evolve(rho)
            if j != N_M:
                result = P.measure(rho)
                if backaction:
                    rho = result.rho
                M[i, j] = result.outcomes
                # Unmeasured qubit (p < 1): NaN in the record M, 0 in the
                # features, i.e. no information (issue #91).
                Phi[i, j] = np.nan_to_num(result.outcomes, nan=0.0)
            else:
                for k in range(N - 1):
                    X[i, k] = np.real(np.trace(Z_ops[k + 1] @ rho))
                Phi[i, j] = X[i]
        Y[i] = np.real(np.trace(Z_ops[0] @ rho))

    phi_2D = Phi.reshape(Phi.shape[0], -1)

    train = int(cfg.samples.train_size)
    phi_train, y_train = phi_2D[:train], Y[:train]
    phi_test, y_test = phi_2D[train:], Y[train:]

    solver.fit(phi_train, y_train)
    y_pred = solver.predict(phi_test)

    return {
        "Phi": Phi,
        "Y": Y,
        "M": M,
        "X": X,
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
    save_results(results, out_dir, Path(__file__).stem, features="Phi")
    print(f"Phi: {results['Phi'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
