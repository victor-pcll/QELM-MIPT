#!/usr/bin/env python3

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
from QELM.training import ProjectiveMeasurement, RidgeReadout, SingleObservable


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.n_steps``, ``samples.L``, ``samples.train_size``, the
            TFIM, embedding, measurement, readout and training blocks, and
            the optional ``experiment.backaction`` (default ``True``).

    Returns:
        The arrays and metrics of the run: ``Y``, ``X``,
        ``y_test``, ``y_pred``, ``W``, ``nmse`` and ``r2``.
    """
    set_seed(cfg)
    
    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    N = sector.N
    ops = SingleObservable({"N": N, "observables": ["Z", "Y", "X"]}).operators()
    L = int(cfg.samples.L)

    X = np.zeros((L, N - 1, 3))
    Y = np.zeros((L,3))

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel.from_config(cfg)
        rho = sector.rho()
        rho = model.evolve(rho, dt = cfg.evolution.t)
        for k in range(N - 1):
            X[i, k] = [np.real(np.trace(operators[k+1] @ rho)) for operators in ops]
        Y[i] = [np.real(np.trace(operators[0] @ rho)) for operators in ops]

    X = X.reshape(X.shape[0], -1)

    train = int(cfg.samples.train_size)
    X_train, y_train = X[:train], Y[:train]
    X_test, y_test = X[train:], Y[train:]

    solver.fit(X_train, y_train)
    y_pred = solver.predict(X_test)

    return {
        "Y": Y,
        "X": X,
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "nmse": NMSE(y_test, y_pred),
        "r2": R2(y_test, y_pred),
    }


def save(results: dict[str, Any], out_dir: Path) -> None:
    """Write the arrays to ``projective_readout.npz`` and the scores to JSON.

    Args:
        results: Output of :func:`run`.
        out_dir: Destination folder, created if needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    np.savez(out_dir / "projective_readout.npz", **arrays)
    metrics = {
        "nmse": results["nmse"],
        "r2": results["r2"],
        "n_samples": int(results["X"].shape[0]),
        "n_test": int(results["y_test"].shape[0]),
        "n_features": int(np.prod(results["X"].shape[1:])),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    print(f"NMSE: {results['nmse']:.6g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
