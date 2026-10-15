"""Read a composed Hydra config. Values come from YAML and CLI overrides."""

from __future__ import annotations

from dotenv import load_dotenv
from omegaconf import DictConfig


def load_secrets() -> None:
    """Load secrets from a local ``.env`` file.

    Missing files are ignored. Scientific parameters stay in ``conf/``.
    """
    load_dotenv()


def format_selection(cfg: DictConfig) -> str:
    """Format the chosen simulation and evolution values.

    Args:
        cfg: Composed Hydra config. The active simulation block is
            ``cfg[cfg.models]``. Each exposed value is a scalar
            (``N``, ``J``, ``dt``, ``n_steps``).

    Returns:
        One line of ``key=value`` pairs, starting with ``models``.
    """
    model_name = str(cfg.models)
    sim = cfg[model_name]
    parts = [f"models={model_name}"]
    parts.extend(f"{model_name}.{key}={value}" for key, value in sim.items())
    parts.extend(f"evolution.{key}={value}" for key, value in cfg.evolution.items())
    return " ".join(parts)
