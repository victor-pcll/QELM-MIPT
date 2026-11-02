"""Compose the Hydra config outside ``main.py``, for notebooks and scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from QELM.physics.operators import QubitCounts

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
# interpolation of the top-level N, Nr or N_total in conf/; overriding the
# copy instead of the source (TFIM.N=6 instead of N=6) would desynchronize
# them.
_N_COPIES = ("ancilla_charge_sector",)
_NR_COPIES = ("projective_measurement",)
_N_TOTAL_COPIES = ("single_observable", "double_observable")


def check_qubit_counts(cfg: DictConfig) -> QubitCounts:
    """Check the qubit counts ``N``, ``Nr`` and ``N_total`` of a composed config.

    The convention (issue #235, which replaces that of issue #88): ``N`` is
    the number of system qubits, the size of the studied system, ``Nr`` the
    number of ancilla qubits and ``N_total = N + Nr`` the number of qubits
    of the simulated state, all three at the top level of the config. The
    selected model block (``cfg[cfg.models].N``) and the encoding copy
    ``N``, the projective measurement copies ``Nr`` and the readouts, which
    act on the whole state, copy ``N_total`` (as ``n_qubits``). Scripts read
    the counts from here, not from a model or an encoding.

    Args:
        cfg: Composed config.

    Returns:
        ``(N, Nr, N_total)``.

    Raises:
        ValueError: If ``N`` is below 1, ``Nr`` is negative, ``N_total`` is
            not ``N + Nr``, or a block holds a count different from the
            top-level one, for instance after ``TFIM.N=6`` instead of
            ``N=6``.
    """
    N, Nr = int(cfg.N), int(cfg.Nr)
    N_total = int(cfg.N_total)
    if N < 1:
        raise ValueError(f"N, the number of system qubits, must be >= 1, got {N}.")
    if Nr < 0:
        raise ValueError(f"Nr must be >= 0, got {Nr}.")
    if N_total != N + Nr:
        raise ValueError(
            f"N_total must be N + Nr, got N={N}, Nr={Nr}, N_total={N_total}. "
            "Set N and Nr, never N_total."
        )
    blocks: dict[str, tuple[str, int]] = {f"{cfg.models}.N": ("N", N)}
    blocks.update({f"{name}.N": ("N", N) for name in _N_COPIES})
    blocks.update({f"{name}.Nr": ("Nr", Nr) for name in _NR_COPIES})
    blocks.update(
        {f"{name}.n_qubits": ("N_total", N_total) for name in _N_TOTAL_COPIES}
    )
    for key, (source, expected) in blocks.items():
        value = OmegaConf.select(cfg, key)
        if value is not None and int(value) != expected:
            raise ValueError(
                f"{key}={value} differs from {source}={expected}. "
                f"Override {source}, not {key}."
            )
    return QubitCounts(N, Nr, N_total)


def upgrade_legacy_counts(cfg: DictConfig) -> DictConfig:
    """Return a config of a run written before issue #235 in the new convention.

    Runs written before issue #235 hold ``N`` as the total number of qubits,
    ancilla included, and ``Ns = N - Nr`` as the number of system qubits.
    Such a config is recognized by its top-level ``Ns`` and no
    ``N_total``; it is resolved, then rewritten so that ``N`` is the number
    of system qubits and ``N_total`` the total: the top-level ``N``, every
    block whose ``N`` copied the old total (the model, the encoding, a
    circuit) and the readouts, whose ``N`` becomes ``n_qubits``. Every
    other value, ``evolution.n_steps`` included, keeps the value it had in
    the run. A config of the new convention is returned unchanged, so the
    plot scripts read old and new run folders alike.

    Args:
        cfg: Config of one run, typically ``.hydra/config.yaml``.

    Returns:
        ``cfg`` itself for a new config, otherwise a resolved copy.
    """
    if "Ns" not in cfg or "N_total" in cfg:
        return cfg
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    if not isinstance(raw, dict):
        return cfg
    total, system = int(raw["N"]), int(raw.pop("Ns"))
    for name, block in raw.items():
        if not isinstance(block, dict) or block.get("N") != total:
            continue
        if name in _N_TOTAL_COPIES:
            block["n_qubits"] = block.pop("N")
        else:
            block["N"] = system
    raw["N"], raw["N_total"] = system, total
    return OmegaConf.create(raw)
