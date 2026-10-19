#!/usr/bin/env python3
"""Main entry point for running QELM experiments.

Composes ``conf/config.yaml`` with Hydra, like ``main.py``, and seeds NumPy
from ``cfg.seed``. Put the experiment logic in ``main`` and write its outputs
to ``SCRATCH`` (``experiments/scratch/``), which git ignores.

Run from the repository root:

    uv run python experiments/example/run_experiment.py models=TFIM N=6
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from QELM.seed import set_seed

# Temporary local outputs: everything here except .gitkeep is gitignored.
SCRATCH = Path(__file__).resolve().parents[1] / "scratch"


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the quantum reservoir experiment.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    set_seed(cfg)
    print(f"Running experiment with model: {cfg.models}")
    # Experiment logic goes here...

    # Save outputs to the scratch directory, for example:
    # np.save(SCRATCH / "output.npy", results)


if __name__ == "__main__":
    main()
