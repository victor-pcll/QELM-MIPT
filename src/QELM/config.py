"""Compose the Hydra config outside ``main.py``, for notebooks and scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# src/QELM/config.py -> repository root, which holds conf/.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "conf"


def load_config(
    overrides: Sequence[str] | None = None,
    config_name: str = "config",
    config_dir: str | Path | None = None,
) -> DictConfig:
    """Compose ``conf/`` with overrides, as ``main.py`` does from the CLI.

    The config directory is found from the package location, so the result
    does not depend on the working directory (Jupyter starts in
    ``notebooks/``). Hydra's global state is cleared first, so a notebook
    cell can be re-run.

    Args:
        overrides: Hydra overrides, such as ``["sim_type=TFIM", "TFIM.N=6"]``.
        config_name: Primary config file in ``config_dir``, without ``.yaml``.
        config_dir: Directory to compose from. ``None`` uses the
            repository's ``conf/``.

    Returns:
        The composed config.

    Raises:
        FileNotFoundError: If the config directory does not exist, for
            example when QELM is installed outside the repository.
    """
    directory = CONFIG_DIR if config_dir is None else Path(config_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Config directory {directory} not found. Pass config_dir= explicitly."
        )
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        return compose(config_name=config_name, overrides=list(overrides or []))
