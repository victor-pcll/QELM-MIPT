#!/usr/bin/env python3
"""Simple QELM experiment: learn the ancilla from the system observables.

No measurement during the evolution. For each sample ``i`` (``samples.L`` in
total):

1. Draw a new TFIM (new disorder ``J_ij``) and prepare the initial state
   with :class:`AncillaChargeSectorEmbedding`: the ancilla ``R`` (qubit 0)
   entangled with the charge of the ``N - 1`` system qubits.
2. Evolve it for the total time ``evolution.t`` in one step,
   ``U(t) = exp(-i H t)``. Without measurements the state stays pure, so the
   state vector is evolved rather than ``rho`` (same result, much cheaper).
3. Store the one-site expectation values ``<O_k>`` on the system qubits
   ``k = 1 .. N-1`` as the features ``X[i]``, for each Pauli ``O`` in
   ``experiment.observables`` (default ``[Z, Y, X]``), and ``<Z_0>`` on the
   ancilla as the target ``Y[i]``.

A ridge readout (``training`` block) is fitted on the first
``samples.train_size`` samples and scored on the rest with NMSE and R2.
``RidgeReadout`` fits no intercept, so a constant column of ones is appended
to the features: the target is an affine, not linear, function of them (for
instance ``<Z_0> ~ c - sum_k <Z_k>`` when ``h >> J`` nearly conserves the
total charge).

Run from the repository root:

    uv run python experiments/exp_QELM.py TFIM.N=4 samples.L=500

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM.npz``: ``X`` (features, bias column included), ``Y``,
  ``y_test``, ``y_pred`` and the weights ``W``;
- ``metrics.json``: ``nmse``, ``r2`` and the shapes used;
- ``.hydra/config.yaml``: the full composed config, to reproduce the run.
"""

from __future__ import annotations

import json
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
from QELM.training import RidgeReadout, SingleObservable

DEFAULT_OBSERVABLES = ["Z", "Y", "X"]


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.t``, ``samples.L``, ``samples.train_size``, the TFIM,
            embedding and training blocks, and the optional
            ``experiment.observables`` (default ``[Z, Y, X]``).

    Returns:
        The arrays and metrics of the run: ``X`` (with the bias column),
        ``Y``, ``y_test``, ``y_pred``, ``W``, ``nmse`` and ``r2``.
    """
    set_seed(cfg)
    labels = list(
        OmegaConf.select(cfg, "experiment.observables", default=DEFAULT_OBSERVABLES)
    )

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    N = sector.N
    t = float(cfg.evolution.t)
    ops = np.array(SingleObservable({"N": N, "observables": labels}).operators())
    ops = ops.reshape(len(labels), N, sector.dim, sector.dim)
    Z0 = SingleObservable({"N": N, "observables": ["Z"]}).operators()[0]
    psi0 = sector.state()

    L = int(cfg.samples.L)
    X = np.zeros((L, len(labels), N - 1))
    Y = np.zeros(L)

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel.from_config(cfg)
        psi = model.evolve(psi0, dt=t)
        X[i] = np.real(np.einsum("a,okab,b->ok", psi.conj(), ops[:, 1:], psi))
        Y[i] = np.real(psi.conj() @ Z0 @ psi)

    X = X.reshape(L, -1)
    X = np.hstack([X, np.ones((L, 1))])  # bias column: RidgeReadout has no intercept

    train = int(cfg.samples.train_size)
    X_train, y_train = X[:train], Y[:train]
    X_test, y_test = X[train:], Y[train:]

    solver.fit(X_train, y_train)
    y_pred = solver.predict(X_test)

    return {
        "X": X,
        "Y": Y,
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "nmse": NMSE(y_test, y_pred),
        "r2": R2(y_test, y_pred),
    }


def save(results: dict[str, Any], out_dir: Path) -> None:
    """Write the arrays to ``exp_QELM.npz`` and the scores to JSON.

    Args:
        results: Output of :func:`run`.
        out_dir: Destination folder, created if needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    np.savez(out_dir / "exp_QELM_Z.npz", **arrays)
    metrics = {
        "nmse": results["nmse"],
        "r2": results["r2"],
        "n_samples": int(results["X"].shape[0]),
        "n_test": int(results["y_test"].shape[0]),
        "n_features": int(results["X"].shape[1]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
