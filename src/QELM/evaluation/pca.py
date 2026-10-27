"""Principal component analysis fitted on a training set (issue #225).

:func:`~QELM.evaluation.bayes_optimal.pca_plane` finds the plane of largest
variance of the samples it is given. To look at the geometry a readout
sees, the axes must come from the training samples only and then be applied
to any other samples (the test set), as the readout itself is: this is
:class:`TrainPCA`.

It also handles feature spaces without spread. The measurement record
``M`` of ``experiments/QELM_Q/exp_QELM_Q.py`` is all zeros at ``p = 0``: its
covariance is the zero matrix, and its "principal directions" are whatever
the SVD returns for a zero singular value. Instead of drawing arbitrary
axes, :class:`TrainPCA` marks every direction whose variance is at most
``tol`` times the total variance (or at most ``tol`` in absolute value when
the total is zero) as degenerate: the projection on it is set to 0 and its
share of the variance to 0. A feature space with no spread at all projects
every sample to the origin, with :attr:`TrainPCA.degenerate` set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

# Relative variance below which a principal direction counts as degenerate.
PCA_TOL = 1e-12


@dataclass(frozen=True)
class TrainPCA:
    """Principal axes of a training set, applied to any samples.

    Attributes:
        center: Mean of the training samples, shape ``(d,)``.
        components: Orthonormal principal directions in rows, shape
            ``(n_components, d)``, by decreasing variance, each signed so that
            its largest entry is positive. A degenerate direction is a row of
            zeros.
        variances: Variance of the training samples along each direction,
            shape ``(n_components,)``; 0 for a degenerate one.
        total_variance: Sum of the variances of the training features,
            ``trace(Sigma)``.
    """

    center: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    total_variance: float

    @property
    def explained(self) -> np.ndarray:
        """Share of the total training variance along each direction.

        0 for every direction when the total variance is 0.
        """
        if self.total_variance <= 0.0:
            return np.zeros_like(self.variances)
        return np.asarray(self.variances / self.total_variance)

    @property
    def rank(self) -> int:
        """Number of non-degenerate directions among the kept ones."""
        return int(np.count_nonzero(self.variances))

    @property
    def degenerate(self) -> bool:
        """``True`` if the training samples have no spread at all."""
        return self.rank == 0

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Project samples on the principal axes.

        Args:
            X: Samples, shape ``(n, d)``.

        Returns:
            ``(X - center) @ components.T``, shape ``(n, n_components)``;
            the columns of degenerate directions are 0.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.center.shape[0]:
            raise ValueError(
                f"X must have shape (n, {self.center.shape[0]}), got {X_arr.shape}."
            )
        return np.asarray((X_arr - self.center) @ self.components.T)


def fit_pca(
    X_train: ArrayLike, n_components: int = 2, tol: float = PCA_TOL
) -> TrainPCA:
    """Fit the principal axes of training samples.

    Args:
        X_train: Training samples, shape ``(n, d)`` with ``n >= 1``,
            ``d >= 1``.
        n_components: Number of directions to keep, ``>= 1``. If it exceeds
            ``min(n, d)``, the extra directions are degenerate (zero rows).
        tol: A direction with a variance at most
            ``tol * max(total_variance, 1)`` is degenerate.

    Returns:
        The :class:`TrainPCA`.

    Raises:
        ValueError: If ``X_train`` is not a non-empty 2D array or
            ``n_components < 1``.
    """
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X_train must have shape (n >= 1, d >= 1), got {X.shape}.")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")
    n, d = X.shape
    center = X.mean(axis=0)
    total = float(np.sum(np.var(X, axis=0)))
    _, S, Vt = np.linalg.svd(X - center, full_matrices=False)
    kept = min(n_components, len(S))
    components = np.zeros((n_components, d))
    variances = np.zeros(n_components)
    components[:kept] = Vt[:kept]
    variances[:kept] = S[:kept] ** 2 / n
    # Directions without spread: arbitrary for the SVD, so dropped.
    flat = variances <= tol * max(total, 1.0)
    components[flat] = 0.0
    variances[flat] = 0.0
    # Same sign convention as pca_plane: largest entry positive.
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(n_components), largest])
    signs[signs == 0] = 1.0
    return TrainPCA(
        center=center,
        components=components * signs[:, None],
        variances=variances,
        total_variance=total,
    )


__all__ = ["PCA_TOL", "TrainPCA", "fit_pca"]
