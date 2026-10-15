"""Write a lightweight checkpoint at the end of a run.

The destination comes from ``cfg.checkpoint``. When ``dir`` is unset, an
active Hydra job uses its timestamped run directory. Otherwise the file
is placed under ``fallback_root/<uuid>/``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from QELM.models.base import EvolutionState, HamiltonianModel
from QELM.training.regression import Trainer

_UNSET = frozenset({"", "null", "none", "???"})


def save_checkpoint(
    cfg: DictConfig,
    *,
    model: HamiltonianModel,
    trainer: Trainer,
    evolution: EvolutionState,
    training_status: str,
) -> Path:
    """Save trainer state and run metadata.

    Args:
        cfg: Composed Hydra config. ``cfg.checkpoint.dir``,
            ``cfg.checkpoint.fallback_root``, ``cfg.checkpoint.use_uuid``,
            and ``cfg.checkpoint.filename`` choose the path. ``cfg.seed``
            is copied into the metadata.
        model: Model whose scalar parameters are stored as metadata.
        trainer: Trainer whose :meth:`~Trainer.state_dict` supplies weights.
        evolution: Evolution record from the model. ``dt`` and ``n_steps``
            are scalars already taken from config by the caller.
        training_status: Status string returned by ``fit``.

    Returns:
        Path of the written ``.npz`` file. The archive holds a ``weights``
        array and a JSON ``metadata`` string. Nothing is uploaded.
    """
    run_id = str(uuid.uuid4())
    path = resolve_checkpoint_path(cfg, run_id=run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainer.state_dict()
    weights = state.get("weights", np.zeros((0,), dtype=np.float64))
    metadata = _metadata(
        cfg,
        model=model,
        trainer=trainer,
        evolution=evolution,
        training_status=training_status,
        run_id=run_id,
    )
    np.savez(
        path,
        weights=np.asarray(weights, dtype=np.float64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def resolve_checkpoint_path(cfg: DictConfig, *, run_id: str) -> Path:
    """Resolve the checkpoint file path from config.

    Args:
        cfg: Composed Hydra config. See :func:`save_checkpoint`.
        run_id: UUID for this save. Used when the directory is not already
            unique, and when ``cfg.checkpoint.use_uuid`` is true.

    Returns:
        File path. Parent directories are not created.

    Raises:
        ValueError: If ``checkpoint.filename`` is not a bare file name.
    """
    filename = _filename(cfg)
    configured = _optional_dir(cfg.checkpoint.dir)
    use_uuid = bool(cfg.checkpoint.use_uuid)
    if configured is not None:
        directory = configured
        nest_uuid = use_uuid
    else:
        hydra_dir = _hydra_run_dir()
        if hydra_dir is not None:
            directory = hydra_dir
            nest_uuid = use_uuid
        else:
            directory = Path(str(cfg.checkpoint.fallback_root)) / run_id
            nest_uuid = False
    if nest_uuid:
        directory = directory / run_id
    return directory / filename


def _filename(cfg: DictConfig) -> str:
    filename = str(cfg.checkpoint.filename)
    if filename == "" or Path(filename).name != filename:
        raise ValueError("checkpoint.filename must be a file name, not a path.")
    return filename


def _optional_dir(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _UNSET:
        return None
    return Path(text)


def _hydra_run_dir() -> Path | None:
    """Return Hydra's run directory when a job has created one."""
    if not HydraConfig.initialized():
        return None
    output_dir = str(HydraConfig.get().runtime.output_dir).strip()
    if output_dir.lower() in _UNSET:
        return None
    return Path(output_dir)


def _metadata(
    cfg: DictConfig,
    *,
    model: HamiltonianModel,
    trainer: Trainer,
    evolution: EvolutionState,
    training_status: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sim_type": str(cfg.sim_type),
        "model": model.name,
        "trainer": trainer.name,
        "seed": int(cfg.seed),
        "training_status": training_status,
        "evolution": _jsonable(dict(evolution)),
        "params": _jsonable(model.params),
    }


def _jsonable(value: object) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = ["resolve_checkpoint_path", "save_checkpoint"]
