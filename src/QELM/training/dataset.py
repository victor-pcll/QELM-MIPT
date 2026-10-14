"""QELM dataset ``(Phi, y)`` built from measured trajectories.

Each sample ``k`` is one trajectory of the circuit. Its features are the
mid-circuit outcomes ``M`` on the system ``S`` followed by the final
projective readout ``X`` of ``S``, so ``Phi = [M || X]``. Its label ``y`` is
the final Z outcome of the reference ancilla ``R``.

Outcomes follow :mod:`QELM.training.measurements`: ``+1`` or ``-1`` for a
measured qubit, ``np.nan`` for a qubit that was not measured.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


class QELMDataset(NamedTuple):
    """Features and labels of ``K`` trajectories.

    Attributes:
        phi: Feature matrix of shape ``(K, n_dynamic + N_S)``.
        y: Ancilla outcomes of shape ``(K,)``.
    """

    phi: np.ndarray
    y: np.ndarray


def create_qelm_dataset(
    dynamic_measurements: ArrayLike,
    final_measurements: ArrayLike,
    ancilla_outcomes: ArrayLike,
    fill_value: float | None = None,
) -> QELMDataset:
    """Build ``Phi = [M || X]`` and pair it with the ancilla labels ``y``.

    Args:
        dynamic_measurements: Mid-circuit outcomes ``M`` of shape
            ``(K, n_dynamic)``, or ``(K, T, N_S)`` (one row of ``N_S``
            outcomes per time step), which is flattened row by row to
            ``(K, T * N_S)``: step 1 first, then step 2, and so on.
            ``n_dynamic`` may be 0.
        final_measurements: Final readout ``X`` of shape ``(K, N_S)``.
        ancilla_outcomes: Ancilla outcomes ``y`` of shape ``(K,)``.
        fill_value: Value that replaces ``np.nan`` (unmeasured qubits) in
            ``Phi``. ``None`` keeps ``np.nan``.

    Returns:
        ``QELMDataset(phi, y)`` with ``phi`` of shape
        ``(K, n_dynamic + N_S)`` as ``float64`` and ``y`` of shape ``(K,)``.

    Raises:
        ValueError: If the arrays do not have the shapes above, if they do
            not have the same number of samples ``K``, or if ``y`` contains
            ``np.nan`` (the ancilla is always measured).
    """
    M = np.asarray(dynamic_measurements, dtype=np.float64)
    X = np.asarray(final_measurements, dtype=np.float64)
    y = np.asarray(ancilla_outcomes, dtype=np.float64)

    if M.ndim == 3:
        M = M.reshape(M.shape[0], -1)
    if M.ndim != 2:
        raise ValueError(
            f"dynamic_measurements must have shape (K, n_dynamic) or "
            f"(K, T, N_S), got {M.shape}."
        )
    if X.ndim != 2:
        raise ValueError(f"final_measurements must have shape (K, N_S), got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"ancilla_outcomes must have shape (K,), got {y.shape}.")
    if not M.shape[0] == X.shape[0] == y.shape[0]:
        raise ValueError(
            f"Sample size mismatch: M={M.shape}, X={X.shape}, y={y.shape}."
        )
    if np.isnan(y).any():
        raise ValueError("ancilla_outcomes must not contain nan.")

    phi = np.hstack([M, X])
    if fill_value is not None:
        phi = np.where(np.isnan(phi), fill_value, phi)
    return QELMDataset(phi=phi, y=y)


def split_dataset(
    dataset: QELMDataset,
    train_fraction: float,
    rng: np.random.Generator | None = None,
) -> tuple[QELMDataset, QELMDataset]:
    """Split a dataset into disjoint train and test sets.

    The train set has ``int(train_fraction * K)`` samples, the same
    rounding as ``samples.train_size`` in ``conf/config.yaml``. Without
    ``rng`` the first samples go to train and the rest to test; with
    ``rng`` the samples are shuffled first. Each sample lands in exactly
    one of the two sets, with its own label.

    Args:
        dataset: Dataset of ``K`` samples.
        train_fraction: Fraction of samples in the train set, in ``[0, 1]``
            (``samples.train_fraction`` in the config).
        rng: Optional generator used to shuffle the samples.

    Returns:
        ``(train, test)``.

    Raises:
        ValueError: If ``train_fraction`` is not in ``[0, 1]``.
    """
    if not 0 <= train_fraction <= 1:
        raise ValueError(f"train_fraction must be in [0, 1], got {train_fraction}.")
    K = dataset.y.shape[0]
    order = np.arange(K) if rng is None else rng.permutation(K)
    n_train = int(train_fraction * K)
    train, test = order[:n_train], order[n_train:]
    return (
        QELMDataset(phi=dataset.phi[train], y=dataset.y[train]),
        QELMDataset(phi=dataset.phi[test], y=dataset.y[test]),
    )
