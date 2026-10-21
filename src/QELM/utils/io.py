"""Write the results of an experiment run to disk."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def save_results(
    results: Mapping[str, Any],
    out_dir: Path,
    exp_name: str,
    features: str = "X",
    extra_metrics: Sequence[str] = (),
) -> Path:
    """Write the arrays to ``<exp_name>.npz`` and the scores to ``metrics.json``.

    The ``.npz`` name comes from ``exp_name``, so two experiments never write
    the same file (issue #87). The scripts pass ``Path(__file__).stem``.

    Args:
        results: Output of an experiment's ``run``. Every NumPy array goes to
            the ``.npz``; ``nmse``, ``nmse_err`` (its standard error over the
            test samples), ``r2``, ``y_test`` and ``results[features]``
            must be present.
        out_dir: Destination folder, created if needed.
        exp_name: Name of the experiment, used as the ``.npz`` file name.
            Must be a plain name, without a directory or the ``.npz`` suffix.
        features: Key of the feature array: its first axis counts the
            samples and the product of the other axes the features.
        extra_metrics: Keys of other scalar entries of ``results`` to copy
            into ``metrics.json``, after the default ones, so that
            :func:`QELM.utils.load_sweep` reads them back.

    Returns:
        Path of the ``.npz`` file.

    Raises:
        ValueError: If ``exp_name`` is empty, contains a path separator or
            ends with ``.npz``.
        KeyError: If a key of ``extra_metrics`` is missing from ``results``.
    """
    if not exp_name or Path(exp_name).name != exp_name or exp_name.endswith(".npz"):
        raise ValueError(
            f"exp_name must be a plain file name without .npz, got {exp_name!r}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        k: v for k, v in results.items() if isinstance(v, np.ndarray)
    }
    npz_path = out_dir / f"{exp_name}.npz"
    np.savez(npz_path, **arrays)
    X = np.asarray(results[features])
    metrics = {
        "nmse": results["nmse"],
        "nmse_err": results["nmse_err"],
        "r2": results["r2"],
        "n_samples": int(X.shape[0]),
        "n_test": int(np.asarray(results["y_test"]).shape[0]),
        "n_features": int(np.prod(X.shape[1:])),
    }
    for key in extra_metrics:
        metrics[key] = float(results[key])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return npz_path
