"""Regression metrics for the readout, the ancilla entanglement entropy,
and the scaffold status line."""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse as sp
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


def von_neumann_entropy(
    rho: np.ndarray | sp.csr_matrix,
    N: int,
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Return the Von Neumann entropy of the ancilla ``R`` (qubit 0).

    ``rho_R = Tr_S(rho)`` is the 2x2 reduced density matrix of the first
    qubit, the most significant bit of a basis index, after tracing out the
    ``N - 1`` system qubits. The entropy is ``S = -sum_k l_k log(l_k)`` over
    the eigenvalues ``l_k`` of ``rho_R``. It measures the entanglement
    between ``R`` and ``S`` for a pure global state: 1 bit for
    ``(|0>_R |Q1>_S + |1>_R |Q2>_S) / sqrt(2)`` with orthogonal ``|Q1>`` and
    ``|Q2>``, 0 for a product state.

    Args:
        rho: Global density matrix of shape ``(2**N, 2**N)``, dense or
            sparse. A sparse matrix is traced block by block, without a
            dense copy.
        N: Total number of qubits, ancilla included (``N >= 1``).
        base: Logarithm base: 2 for bits (default), ``np.e`` for nats.
        eps: Eigenvalues ``<= eps`` are dropped, so that rounding noise
            such as ``-1e-16`` never reaches the logarithm.

    Returns:
        ``S(rho_R)`` in the chosen base, between 0 and ``log_base(2)``.

    Raises:
        ValueError: If ``N < 1``, ``rho`` is not ``(2**N, 2**N)``, or
            ``base`` is not a positive number other than 1.
    """
    if N < 1:
        raise ValueError(f"N must be at least 1, got {N}.")
    if not (base > 0 and base != 1):
        raise ValueError(f"base must be positive and different from 1, got {base}.")
    dim = 2**N
    if rho.shape != (dim, dim):
        raise ValueError(
            f"rho must have shape ({dim}, {dim}) for N={N}, got {rho.shape}."
        )
    dim_S = dim // 2
    if sp.issparse(rho):
        # rho_R[a, b] = Tr of the (a, b) block of size dim_S.
        csr = sp.csr_matrix(cast(sp.csr_matrix, rho))
        rho_R = np.array(
            [
                [
                    csr[a * dim_S : (a + 1) * dim_S, b * dim_S : (b + 1) * dim_S]
                    .diagonal()
                    .sum()
                    for b in range(2)
                ]
                for a in range(2)
            ]
        )
    else:
        rho_R = np.trace(np.asarray(rho).reshape(2, dim_S, 2, dim_S), axis1=1, axis2=3)
    eigenvalues = np.linalg.eigvalsh(rho_R)
    valid = eigenvalues[eigenvalues > eps]
    if valid.size == 0:
        return 0.0
    return float(-np.sum(valid * np.log(valid)) / np.log(base))
