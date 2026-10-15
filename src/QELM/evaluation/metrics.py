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
    """Convert both inputs to 1-dimensional float arrays of the same length.

    Every metric goes through this check, so ``MSE``, ``NMSE`` and ``R2``
    only ever see two validated arrays.

    Raises:
        ValueError: If the shapes differ, the arrays are empty, or they are
            not 1-dimensional.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shapes differ: y_true {y_true_arr.shape}, y_pred {y_pred_arr.shape}."
        )
    if y_true_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true_arr.ndim != 1:
        raise ValueError(
            f"Inputs must be 1-dimensional. Got y_true ndim: {y_true_arr.ndim}, "
            f"y_pred ndim: {y_pred_arr.ndim}."
        )
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

    Raises:
        ValueError: If the inputs are invalid (see :func:`_as_pair`) or
            ``y_true`` is all zeros.
    """
    y_true_arr, y_pred_arr = _as_pair(y_true, y_pred)
    norm = float(np.mean(y_true_arr**2))
    if norm == 0.0:
        raise ValueError("NMSE is undefined when y_true is all zeros.")
    return MSE(y_true_arr, y_pred_arr) / norm


def R2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination R^2 (not the Pearson correlation R).

    ``R2 = 1 - sum((y_true - y_pred)**2) / sum((y_true - mean(y_true))**2)``,
    the same as ``sklearn.metrics.r2_score``.

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

    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1 - ss_res / ss_tot)
