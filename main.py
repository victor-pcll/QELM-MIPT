"""Hydra entrypoint.

Scientific parameters come from ``config/`` YAML and CLI overrides.
Secrets (API keys, local paths) are loaded from ``.env`` when present.

Run from the repository root:

    uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from qrc_env_template.selection import format_selection, load_secrets


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    load_secrets()
    print(format_selection(cfg))


if __name__ == "__main__":
    main()
