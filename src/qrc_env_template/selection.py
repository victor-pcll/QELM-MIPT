"""Read a composed Hydra config. Values come from YAML and CLI overrides."""

from __future__ import annotations

from dotenv import load_dotenv
from omegaconf import DictConfig


def load_secrets() -> None:
    """Load secrets from a local ``.env`` file. Missing files are ignored."""
    load_dotenv()


def format_selection(cfg: DictConfig) -> str:
    """Expose the chosen simulation and test values."""
    sim_type = str(cfg.sim_type)
    sim = cfg[sim_type]
    sim_parts = " ".join(f"{sim_type}.{key}={value}" for key, value in sim.items())
    test_parts = " ".join(f"test.{key}={value}" for key, value in cfg.test.items())
    return f"sim_type={sim_type} {sim_parts} {test_parts}"
