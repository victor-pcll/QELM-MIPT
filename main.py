"""Pipeline entrypoint.

Scientific parameters come from ``conf/`` YAML and CLI overrides.
``main`` routes sim type to a model, then training, then evaluation.
Secrets (API keys, local paths) are loaded from ``.env`` when present.

Run from the repository root:

    uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from qrc_env_template.pipeline import run_pipeline
from qrc_env_template.selection import format_selection, load_secrets


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_secrets()
    print(format_selection(cfg))
    print(run_pipeline(cfg))


if __name__ == "__main__":
    main()
