"""Pipeline entrypoint.

Scientific parameters come from ``conf/`` YAML and CLI overrides.
``main`` seeds NumPy from ``cfg.seed``, then routes sim type to a model,
training, and evaluation, and writes a checkpoint. Secrets (API keys,
local paths) are loaded from ``.env`` when present.

Run from the repository root:

    uv run python main.py sim_type=TFIM TFIM.N=12 test.dt=0.05
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from QELM.pipeline import run_pipeline
from QELM.seed import set_seed
from QELM.selection import format_selection, load_secrets


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Seed NumPy, load secrets, and run the selected pipeline.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
            ``cfg.seed`` is a scalar integer (default 42) applied to NumPy
            before the pipeline runs.
    """
    set_seed(cfg)
    load_secrets()
    print(format_selection(cfg))
    print(run_pipeline(cfg))


if __name__ == "__main__":
    main()
