"""Regression metrics for the readout, and the scaffold status line."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def metric_status(names: list[str]) -> str:
    """Report which metrics a run would compute.

    Args:
        names: Metric identifiers from ``cfg.evaluation.metrics``. Each
            entry is a scalar string. No score tensor is computed.

    Returns:
        A status line listing the names and ``status=scaffold``.
    """
    return "metrics=" + ",".join(names) + " status=scaffold"


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert both inputs to float arrays of the same, non-empty shape."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shapes differ: y_true {y_true_arr.shape}, y_pred {y_pred_arr.shape}."
        )
    if y_true_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")
    return y_true_arr, y_pred_arr


def MSE(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Squared Error (MSE) metric.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values, same shape as ``y_true``.

    Returns:
        MSE score.
    """
    y_true_arr, y_pred_arr = _as_pair(y_true, y_pred)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def NMSE(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Normalized Mean Squared Error (NMSE) metric: ``MSE / mean(y_true**2)``.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values, same shape as ``y_true``.

    Returns:
        NMSE score.
    """
    y_true_arr, y_pred_arr = _as_pair(y_true, y_pred)
    return MSE(y_true_arr, y_pred_arr) / float(np.mean(y_true_arr**2))


def R(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """R-squared (R2) metric.

    Args:
        y_true: Ground truth values, 1-dimensional.
        y_pred: Predicted values, same shape as ``y_true``.

    Returns:
        R2 score. When ``y_true`` is constant, 1.0 for a perfect prediction
        and 0.0 otherwise.

    Raises:
        ValueError: If the shapes differ, the arrays are empty, or they are
            not 1-dimensional.
    """
    y_true_arr, y_pred_arr = _as_pair(y_true, y_pred)
    if y_true_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays.")

    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1 - ss_res / ss_tot)
