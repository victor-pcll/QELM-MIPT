"""Read back the runs of a sweep written by an experiment launcher."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

# Config keys copied into every row, as (row name, dotted config key).
DEFAULT_KEYS: tuple[tuple[str, str], ...] = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("L", "samples.L"),
)


def load_sweep(
    sweep_dir: str | Path,
    keys: Sequence[tuple[str, str]] = DEFAULT_KEYS,
) -> list[dict[str, Any]]:
    """Return one row per run folder of ``sweep_dir``.

    A run folder is a direct subfolder holding ``metrics.json`` and the
    Hydra config ``.hydra/config.yaml``, as written by
    ``experiments/QELM_M/exp_QELM_M.sh`` under
    ``outputs/exp_QELM_M/<date_time>/``. The parameters are read from the
    config, not from the folder name, so renaming a folder changes nothing.

    Args:
        sweep_dir: Folder of one sweep.
        keys: ``(name, dotted config key)`` pairs to copy into each row; a
            key missing from a config gives ``None``.

    Returns:
        Rows sorted by folder name, each with ``run`` (folder name), the
        ``keys`` and every entry of ``metrics.json``.

    Raises:
        FileNotFoundError: If ``sweep_dir`` holds no run folder.
    """
    sweep_dir = Path(sweep_dir)
    rows = []
    for metrics_path in sorted(sweep_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
        row: dict[str, Any] = {"run": run_dir.name}
        for name, key in keys:
            row[name] = OmegaConf.select(cfg, key)
        row.update(json.loads(metrics_path.read_text()))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No */metrics.json run folder in {sweep_dir}.")
    return rows


def summarize(
    rows: Sequence[dict[str, Any]],
    by: Sequence[str] = ("N", "p"),
    metrics: Sequence[str] = ("nmse", "r2"),
) -> list[dict[str, Any]]:
    """Average the metrics of the rows that share the values of ``by``.

    Typically the seeds of a sweep: one output row per ``(N, p)`` with
    ``<metric>_mean``, ``<metric>_std`` (``ddof=1``, ``nan`` for a single
    run), ``<metric>_sem = std / sqrt(n_runs)`` and ``n_runs``.

    Args:
        rows: Output of :func:`load_sweep`.
        by: Row keys that define a group.
        metrics: Row keys to average.

    Returns:
        One row per group, sorted by the values of ``by``.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in by), []).append(row)
    out = []
    for values in sorted(groups):
        members = groups[values]
        summary: dict[str, Any] = dict(zip(by, values, strict=True))
        summary["n_runs"] = len(members)
        for metric in metrics:
            data = np.array([float(m[metric]) for m in members])
            std = float(np.std(data, ddof=1)) if data.size > 1 else float("nan")
            summary[f"{metric}_mean"] = float(np.mean(data))
            summary[f"{metric}_std"] = std
            summary[f"{metric}_sem"] = std / float(np.sqrt(data.size))
        out.append(summary)
    return out


def latest_sweep(root: str | Path) -> Path:
    """Return the most recent ``<date_time>`` folder under ``root``.

    The launchers name each sweep ``%Y-%m-%d_%H-%M-%S``, so the latest is the
    last in alphabetical order.

    Raises:
        FileNotFoundError: If ``root`` holds no subfolder.
    """
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(f"No sweep folder in {root}.")
    return folders[-1]


def new_sweep_dir(root: str | Path, experiment: str) -> Path:
    """Create and return a new ``<root>/<experiment>/<date_time>`` folder.

    The name follows the launchers (``%Y-%m-%d_%H-%M-%S``), so that
    :func:`latest_sweep` finds it. A notebook creates its sweep folder with
    this, passes it to the launcher as ``OUT_DIR`` and reads back that exact
    folder, never another notebook's (issue #151). When the folder already
    exists (two sweeps started in the same second), ``_2``, ``_3``, ... is
    appended.

    Args:
        root: Output root, typically ``outputs``.
        experiment: Folder of the experiment or notebook, a plain name.

    Returns:
        The new, empty folder.

    Raises:
        ValueError: If ``experiment`` is empty or contains a path separator.
    """
    if not experiment or Path(experiment).name != experiment:
        raise ValueError(f"experiment must be a plain name, got {experiment!r}.")
    parent = Path(root) / experiment
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder, k = parent / stamp, 1
    while True:
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            k += 1
            folder = parent / f"{stamp}_{k}"
