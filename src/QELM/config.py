"""Compose the Hydra config outside ``main.py``, for notebooks and scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

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
        overrides: Hydra overrides, such as ``["models=TFIM", "N=6"]``.
        config_name: Primary config file in ``config_dir``, without ``.yaml``.
        config_dir: Directory to compose from. ``None`` uses the
            repository's ``conf/``.

    Returns:
        The composed config, checked with :func:`check_qubit_counts`.

    Raises:
        FileNotFoundError: If the config directory does not exist, for
            example when QELM is installed outside the repository.
        ValueError: If the qubit counts disagree (see
            :func:`check_qubit_counts`).
    """
    directory = CONFIG_DIR if config_dir is None else Path(config_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Config directory {directory} not found. Pass config_dir= explicitly."
        )
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    check_qubit_counts(cfg)
    return cfg


# Blocks that carry their own copy of a qubit count, by key. Each one is an
# interpolation of the top-level N or Nr in conf/; overriding the copy
# instead of the source (N=6 instead of N=6) would desynchronize them.
_N_COPIES = ("ancilla_charge_sector", "single_observable", "double_observable")
_NR_COPIES = ("projective_measurement",)


def check_qubit_counts(cfg: DictConfig) -> tuple[int, int, int]:
    """Check the qubit counts ``N``, ``Ns`` and ``Nr`` of a composed config.

    The convention (issue #88): ``N`` is the total number of qubits, ``Nr``
    the number of ancilla qubits and ``Ns = N - Nr`` the number of system
    qubits, all three at the top level of the config. The selected model
    block (``cfg[cfg.models].N``), the embedding and the readouts copy
    ``N``, and the projective measurement copies ``Nr``. Scripts read the
    counts from here, not from a model or an embedding.

    Args:
        cfg: Composed config.

    Returns:
        ``(N, Ns, Nr)``.

    Raises:
        ValueError: If ``Nr`` is negative, ``Ns`` is not ``N - Nr`` or is
            below 1, or a block holds an ``N`` or ``Nr`` different from the
            top-level one, for instance after ``N=6`` instead of
            ``N=6``.
    """
    N, Nr = int(cfg.N), int(cfg.Nr)
    Ns = int(cfg.Ns)
    if Nr < 0:
        raise ValueError(f"Nr must be >= 0, got {Nr}.")
    if Ns != N - Nr or Ns < 1:
        raise ValueError(
            f"Ns must be N - Nr >= 1, got N={N}, Nr={Nr}, Ns={Ns}. "
            "Set N and Nr, never Ns."
        )
    blocks = {f"{cfg.models}.N": N}
    blocks.update({f"{name}.N": N for name in _N_COPIES})
    blocks.update({f"{name}.Nr": Nr for name in _NR_COPIES})
    for key, expected in blocks.items():
        value = OmegaConf.select(cfg, key)
        if value is not None and int(value) != expected:
            source = "N" if key.endswith(".N") else "Nr"
            raise ValueError(
                f"{key}={value} differs from {source}={expected}. "
                f"Override {source}, not {key}."
            )
    return N, Ns, Nr
