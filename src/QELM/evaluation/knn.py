"""K-nearest-neighbors classifier in NumPy (issue #214).

The Bayes ceilings of the p-sweep use a K-NN with ``K = round(sqrt(n_train))``
and a vote weighted by the inverse distance, the same model as
``sklearn.neighbors.KNeighborsClassifier(n_neighbors=K, weights="distance")``.
:class:`CustomKNN` is that classifier written out, so that the metric can be
changed later (the Mahalanobis distance of issue #213).

The rule, for a test point ``x``:

1. Compute the Euclidean distance ``d_i = ||x - x_i||`` to every training
   point ``x_i``.
2. Keep the ``K`` closest points.
3. Give each kept point the weight ``w_i = 1 / d_i``. If some kept points
   sit exactly on ``x`` (``d_i = 0``), they get the weight 1 and the others 0.
4. The score of a class is the sum of the weights of its kept points.
   ``predict_proba`` divides the scores by their sum; ``predict`` returns the
   class with the largest score, the first class in sorted order on a tie.

Step 2 is ambiguous when several training points sit at the distance of the
``K``-th neighbor: only some of them fit in the ``K`` places. This happens
when the features take few values, for example measurement outcomes ``±1``
on the vertices of the hypercube. scikit-learn then picks the tied points in
an order that depends on its search algorithm (``algorithm="auto"`` uses a
k-d tree at small dimension, which does not pick the same points as
``algorithm="brute"``), so its answer is not unique. :class:`CustomKNN`
offers two well-defined rules, chosen with ``ties``:

- ``"average"`` (default): the ``s`` free places are shared by the ``t``
  tied points, each counted with the weight ``(s / t) w``. The scores are
  then the average of the scores over every choice of ``s`` points among the
  ``t``, so ``predict_proba`` is the average of the K-NN vote over all tie
  breaks, and the result does not depend on the order of the training set.
- ``"index"``: the tied points are taken in the order of the training set,
  lowest index first.

Both rules give the same answer as every scikit-learn algorithm when the
``K``-th distance is not tied, which is the case for continuous features
such as the ``<Z_k>`` of ``exp_QELM_Q.py``.

The distances are computed from the differences ``x - x_i``, as scikit-learn's
k-d and ball trees do, and they agree with them to rounding. scikit-learn's
brute force uses the expansion ``||x||^2 - 2 x.x_i + ||x_i||^2`` instead,
whose squared distance is off by about ``1e-16 ||x||^2``: for two very close
points the distance, hence the weight ``1 / d``, loses digits (``1e-7``
relative for ``d ~ 1e-5`` in the ``<Z_k>`` features). The differences are
exact for points on the hypercube.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike

# Number of test points whose distances are held in memory at once.
_CHUNK = 256

# The rules for the points tied with the K-th neighbor.
TIES = ("average", "index")


def default_n_neighbors(n_train: int) -> int:
    """Return ``K = round(sqrt(n_train))``, at least 1 and at most ``n_train``."""
    if n_train < 1:
        raise ValueError(f"n_train must be positive, got {n_train}")
    return max(1, min(int(round(np.sqrt(n_train))), n_train))


class CustomKNN:
    """K-NN classifier with the Euclidean distance and inverse-distance votes.

    Args:
        n_neighbors: ``K``; ``None`` takes :func:`default_n_neighbors` of the
            training size at :meth:`fit`. A ``K`` larger than the training set
            is reduced to its size.
        ties: How the points tied with the ``K``-th neighbor vote,
            ``"average"`` or ``"index"`` (see the module docstring).

    Attributes:
        classes_: The sorted class labels, set by :meth:`fit`.
        n_neighbors_: The ``K`` used, set by :meth:`fit`.
    """

    def __init__(self, n_neighbors: int | None = None, ties: str = "average") -> None:
        if n_neighbors is not None and int(n_neighbors) < 1:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
        if ties not in TIES:
            raise ValueError(f"ties must be one of {TIES}, got {ties!r}")
        self.n_neighbors = n_neighbors
        self.ties = ties

    def fit(self, X: ArrayLike, y: ArrayLike) -> CustomKNN:
        """Store the training set (lazy learning: nothing else is fitted).

        Args:
            X: Training features, shape ``(n_train, d)``.
            y: Training labels, shape ``(n_train,)``.

        Returns:
            ``self``.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_arr.shape}")
        if y_arr.shape != (X_arr.shape[0],):
            raise ValueError(
                f"y must have shape ({X_arr.shape[0]},), got {y_arr.shape}"
            )
        if X_arr.shape[0] == 0:
            raise ValueError("the training set is empty")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values")
        self.classes_, self._y = np.unique(y_arr, return_inverse=True)
        self._X = X_arr
        n_train = X_arr.shape[0]
        if self.n_neighbors is None:
            self.n_neighbors_ = default_n_neighbors(n_train)
        else:
            self.n_neighbors_ = min(int(self.n_neighbors), n_train)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "_X"):
            raise RuntimeError("CustomKNN is not fitted; call fit first")

    def _check_X(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != self._X.shape[1]:
            raise ValueError(
                f"X must have shape (n, {self._X.shape[1]}), got {X_arr.shape}"
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values")
        return X_arr

    def _chunks(self, X: ArrayLike) -> Iterator[tuple[slice, np.ndarray]]:
        """Yield ``(rows, d2)``: squared distances of blocks of test points."""
        X_arr = self._check_X(X)
        for start in range(0, X_arr.shape[0], _CHUNK):
            rows = slice(start, start + _CHUNK)
            diff = X_arr[rows, None, :] - self._X[None, :, :]
            yield rows, np.einsum("ijk,ijk->ij", diff, diff)

    def kneighbors(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the distances and indices of the ``K`` nearest training points.

        Args:
            X: Test features, shape ``(n, d)``.

        Returns:
            ``(dist, ind)``, both of shape ``(n, K)``, sorted by increasing
            distance and, at equal distance, by increasing training index.
        """
        n = np.asarray(X).shape[0]
        K = self.n_neighbors_
        dist = np.empty((n, K))
        ind = np.empty((n, K), dtype=np.intp)
        for rows, d2 in self._chunks(X):
            order = np.argsort(d2, axis=1, kind="stable")[:, :K]
            ind[rows] = order
            dist[rows] = np.sqrt(np.take_along_axis(d2, order, axis=1))
        return dist, ind

    def _tie_weights(self, d2: np.ndarray) -> np.ndarray:
        """Return the weight of every training point, ``ties="average"``.

        Args:
            d2: Squared distances, shape ``(n, n_train)``.

        Returns:
            Shape ``(n, n_train)``: ``1 / d`` inside the ``K``-th distance,
            ``(s / t) / d`` on it, 0 beyond; on a row with points at distance
            0, those points share the weight instead.
        """
        K = self.n_neighbors_
        d2_K = np.partition(d2, K - 1, axis=1)[:, K - 1 : K]
        inside = d2 < d2_K
        on = d2 == d2_K
        share = (K - inside.sum(axis=1, keepdims=True)) / on.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            inv = 1.0 / np.sqrt(d2)
        w = np.where(inside, inv, 0.0) + np.where(on, share * inv, 0.0)
        exact = d2 == 0.0
        rows = exact.any(axis=1)
        # Points on x: weight 1 each, or K / t each when they fill all K places.
        w[rows] = np.where(exact[rows], np.where(on[rows], share[rows], 1.0), 0.0)
        return w

    def _scores(self, X: ArrayLike) -> np.ndarray:
        """Return the unnormalized vote of each class, shape ``(n, n_classes)``."""
        self._check_fitted()
        n_classes = self.classes_.size
        if self.ties == "average":
            scores = np.empty((np.asarray(X).shape[0], n_classes))
            for rows, d2 in self._chunks(X):
                w = self._tie_weights(d2)
                for c in range(n_classes):
                    scores[rows, c] = np.sum(np.where(self._y == c, w, 0.0), axis=1)
            return scores
        dist, ind = self.kneighbors(X)
        w = _inverse_distance(dist)
        labels = self._y[ind]
        # Same sums as scikit-learn's ``weighted_mode`` on the K neighbors.
        return np.stack(
            [np.sum(np.where(labels == c, w, 0.0), axis=1) for c in range(n_classes)],
            axis=1,
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the weighted vote of each class, normalized to sum to 1.

        Args:
            X: Test features, shape ``(n, d)``.

        Returns:
            Shape ``(n, n_classes)``, columns in the order of ``classes_``.
        """
        if self.ties == "index":
            dist, ind = self.kneighbors(X)
            w = _inverse_distance(dist)
            labels = self._y[ind]
            proba = np.zeros((w.shape[0], self.classes_.size))
            rows = np.arange(w.shape[0])
            # Accumulate neighbor by neighbor, in the order of scikit-learn.
            for j in range(w.shape[1]):
                proba[rows, labels[:, j]] += w[:, j]
        else:
            proba = self._scores(X)
        total = proba.sum(axis=1, keepdims=True)
        total[total == 0.0] = 1.0
        return proba / total

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the class with the largest weighted vote.

        Args:
            X: Test features, shape ``(n, d)``.

        Returns:
            Shape ``(n,)``, labels from ``classes_``; a tie goes to the first
            class in sorted order.
        """
        scores = self._scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of :meth:`predict` on ``(X, y)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


def _inverse_distance(dist: np.ndarray) -> np.ndarray:
    """Return ``1 / d``, or the indicator of ``d = 0`` on rows that have one.

    This is ``weights="distance"`` of scikit-learn.
    """
    with np.errstate(divide="ignore"):
        w = 1.0 / dist
    exact = np.isinf(w)
    rows = exact.any(axis=1)
    w[rows] = exact[rows]
    return w
