"""Scores of a readout that predicts the entanglement entropy (issue #230).

The task of ``experiments/QELM_S/exp_QELM_S.py``: from local features of a
trajectory (the measurement record ``M``, the final ``<Z_k>``), predict the
half-chain entanglement entropy ``S`` of its final state, pooled over several
measurement probabilities ``p`` that cross the transition.

``S`` depends first of all on ``p``, and ``p`` can be read from the features
(the density of measured qubits in ``M``). A high pooled ``R2`` can then mean
only that the readout has guessed ``p``. The scores below separate the two:

- :func:`stratified_split`: the same share of every ``p`` in the training
  and the test sets, so the test set is balanced across the transition;
- :func:`one_hot`: the indicators of ``p``. A ridge readout on them predicts
  the training mean of ``S`` at each ``p``, the best any function of ``p``
  alone can do: the baseline that a readout on the features must beat;
- :func:`r2_within`: the ``R2`` at fixed ``p``, measured against the mean of
  ``S`` at that ``p``. The baseline above scores about 0 here, so a positive
  value means that the features carry the trajectory-to-trajectory
  fluctuations of the entanglement, not only ``p``;
- :func:`even_features`: ``[X, X**2]``. ``S`` does not change when every
  outcome flips sign, while a linear readout on ``X`` is odd in the signs;
  the squares (``|M|``, the mask of the measured qubits, and ``<Z_k>**2``)
  give the readout even features, so a failure of the linear readout can be
  told apart from that symmetry;
- :func:`phase_labels`: the phase of a trajectory, +1 below ``p_c`` (volume
  law), -1 above (area law). A single trajectory has no phase: this label is
  a function of ``p`` alone, so classifying it learns ``p``, and the
  one-hot baseline reaches an accuracy of 1. It is kept for comparison with
  the issue, next to the regression.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stratified_split(
    groups: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the samples so that every group keeps the same training share.

    Args:
        groups: Group of each sample, shape ``(L,)`` (here the ``p`` of the
            sample).
        train_fraction: Share of each group in the training set, in
            ``(0, 1)``; each group gets ``max(1, round(train_fraction * n))``
            training samples and keeps at least one test sample.
        rng: Generator for the order of the samples inside each group.

    Returns:
        ``(train, test)``: sorted indices of the training and test samples.

    Raises:
        ValueError: If ``train_fraction`` is not in ``(0, 1)`` or a group has
            fewer than two samples.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    groups = np.asarray(groups)
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for level in np.unique(groups):
        idx = np.flatnonzero(groups == level)
        if idx.size < 2:
            raise ValueError(f"Group {level} has {idx.size} sample, need >= 2.")
        idx = rng.permutation(idx)
        n_train = min(idx.size - 1, max(1, round(train_fraction * idx.size)))
        train.append(idx[:n_train])
        test.append(idx[n_train:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def one_hot(groups: np.ndarray, levels: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the indicators of ``levels``, shape ``(L, len(levels))``.

    Raises:
        ValueError: If a sample is in none of ``levels``.
    """
    groups = np.asarray(groups)
    levels = np.asarray(levels)
    out = (groups[:, None] == levels[None, :]).astype(float)
    if not np.all(out.sum(axis=1) == 1):
        raise ValueError("Every sample must be in exactly one of the levels.")
    return out


def even_features(X: np.ndarray) -> np.ndarray:
    """Return ``[X, X**2]``: the features and their squares, side by side."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, X**2])


def phase_labels(p: np.ndarray, p_c: float) -> np.ndarray:
    """Return +1 for ``p < p_c`` (volume law) and -1 otherwise (area law)."""
    return np.where(np.asarray(p) < p_c, 1.0, -1.0)


def r2_within(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, dict[float, float]]:
    """Return the ``R2`` at fixed group, pooled and per group.

    For a group ``g`` with test samples ``i``, ``R2_g = 1 - sum_i (y_i -
    yhat_i)**2 / sum_i (y_i - mean_g y)**2``: the share of the spread of ``y``
    inside the group that the prediction explains. The pooled value sums the
    two terms over the groups, ``1 - sum_g SSE_g / sum_g SS_g``.

    A group whose ``y`` has no spread (``SS_g <= eps``, for instance ``S = 0``
    for every trajectory at ``p = 1``) gets ``NaN`` and is left out of the
    pooled value, which is ``NaN`` when every group is.

    Args:
        y_true: Targets, shape ``(n,)``.
        y_pred: Predictions, shape ``(n,)``.
        groups: Group of each sample, shape ``(n,)``.
        eps: Smallest spread counted.

    Returns:
        ``(pooled, per_group)`` with ``per_group`` keyed by group value.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    per_group: dict[float, float] = {}
    sse_total = ss_total = 0.0
    for level in np.unique(groups):
        keep = groups == level
        y = y_true[keep]
        ss = float(np.sum((y - y.mean()) ** 2))
        sse = float(np.sum((y - y_pred[keep]) ** 2))
        if ss <= eps:
            per_group[float(level)] = float("nan")
            continue
        per_group[float(level)] = 1.0 - sse / ss
        sse_total += sse
        ss_total += ss
    pooled = 1.0 - sse_total / ss_total if ss_total > 0 else float("nan")
    return pooled, per_group
