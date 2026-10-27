"""Gaussian mixture fitted by expectation-maximization, in NumPy (issue #223).

The Gaussian-mixture ceiling of the Bayes accuracy
(:class:`~QELM.evaluation.bayes_optimal.GaussianMixtureBayes`) and its
choice of the number of Gaussians by BIC
(:mod:`QELM.evaluation.gmm_scan`) used ``sklearn.mixture.GaussianMixture``.
:class:`CustomGMM` is that model written out, so that every step of the fit
can be read and checked: the initialization, the two steps of the
expectation-maximization (EM) algorithm, the stopping rule, the
regularization of the covariances and the count of free parameters in the
BIC.

The model of a sample ``x`` in ``D`` dimensions is a mixture of ``K``
Gaussians with full covariances,

    ``f(x) = sum_k pi_k N(x; mu_k, Sigma_k)``,

and the fit maximizes the log-likelihood ``log L = sum_i log f(x_i)`` of
the ``n`` training samples. EM alternates two steps from an initial guess:

1. **E-step**: the responsibility of component ``k`` for sample ``i`` is its
   posterior, ``r_ik = pi_k N(x_i; mu_k, Sigma_k) / f(x_i)``. It is
   computed in logarithms (log-sum-exp), so that a sample far from every
   component does not underflow to ``0 / 0``.
2. **M-step**: the parameters that maximize the expected log-likelihood
   given the ``r_ik``, with ``n_k = sum_i r_ik``:

   - ``pi_k = n_k / n``;
   - ``mu_k = (1 / n_k) sum_i r_ik x_i``;
   - ``Sigma_k = (1 / n_k) sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T + c I``.

Each EM round cannot decrease ``log L``. The fit stops when the mean
log-likelihood per sample, ``log L / n``, changes by less than ``tol``
between two rounds (or after ``max_iter`` rounds, with a warning).

**Positive definite covariances.** ``sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T``
is a sum of outer products with weights ``r_ik >= 0``, so it is positive
semi-definite, but it is singular when a feature does not vary inside a
component (a qubit measured at ``+-1``) or when a component holds fewer
than ``D + 1`` samples. The constant ``c = reg_covar`` (``1e-6`` by default)
added to its diagonal makes it positive definite: every eigenvalue is at
least ``c``. A covariance whose Cholesky factorization still fails (``c = 0``,
or a rounding error larger than ``c``) raises a ``ValueError``. ``n_k`` gets
``10 eps`` (``eps`` the float64 machine precision), so that a component that
lost all its samples does not divide by 0.

**Initialization.** The EM rounds start from responsibilities ``r_ik``:

- ``init_params="kmeans"`` (default): ``r_ik = 1`` if ``x_i`` is in the
  cluster ``k`` of a k-means partition (:func:`kmeans`, k-means++ seeding
  then Lloyd iterations), else 0;
- ``init_params="random"``: ``r_ik`` drawn uniformly in ``[0, 1)`` and
  normalized over ``k``;
- or the responsibilities passed to :meth:`CustomGMM.fit` as ``resp_init``.

The first M-step turns them into ``pi_k, mu_k, Sigma_k``. EM finds a local
maximum of ``log L`` that depends on this start; with ``n_init > 1`` it is
run from ``n_init`` starts and the best final ``log L`` is kept.

**BIC.** :func:`bic_score` is ``BIC = nu ln n - 2 ln L``, with
:func:`n_parameters` the number of free parameters of the mixture,

    ``nu = K (D + D (D + 1) / 2) + K - 1``:

``D`` per mean, ``D (D + 1) / 2`` per symmetric covariance and ``K - 1``
weights (they sum to 1). The smallest BIC selects the number of Gaussians.

**Agreement with scikit-learn 1.9.** The steps, the order of the
operations, the ``10 eps`` and the stopping rule are those of
``GaussianMixture(covariance_type="full")``. From the same initial
responsibilities, both run the same EM rounds and agree to rounding
(``tests/test_custom_gmm.py``). The k-means partitions differ (scikit-learn
draws its k-means++ seeds with another random generator), so with
``init_params="kmeans"`` the two can reach different local maxima of
``log L`` for the same seed; ``experiments/QELM_Q/test_custom_gmm.py``
measures how much that changes the ceilings on the ``N = 8`` sweep.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

INIT_PARAMS = ("kmeans", "random")

_COVARIANCE_ERROR = (
    "A covariance of the mixture is not positive definite (for instance a "
    "component with a single sample or collapsed samples): decrease "
    "n_components or increase reg_covar."
)


def n_parameters(n_components: int, n_features: int) -> int:
    """Return ``nu``, the free parameters of a mixture of full-covariance Gaussians.

    ``nu = K (D + D (D + 1) / 2) + K - 1``: ``K`` means of ``D`` numbers,
    ``K`` symmetric ``D x D`` covariances of ``D (D + 1) / 2`` numbers, and
    ``K`` weights that sum to 1.

    Args:
        n_components: ``K >= 1``.
        n_features: ``D >= 1``.
    """
    if n_components < 1 or n_features < 1:
        raise ValueError(
            f"Need n_components >= 1 and n_features >= 1, got {n_components} "
            f"and {n_features}."
        )
    K, D = n_components, n_features
    return K * (D + D * (D + 1) // 2) + K - 1


def bic_score(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Return ``BIC = nu ln n - 2 ln L``; the lower, the better.

    Args:
        log_likelihood: ``ln L``, the total log-likelihood of the ``n``
            samples (not the mean per sample).
        n_params: ``nu``, e.g. :func:`n_parameters`.
        n_samples: ``n >= 1``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    return n_params * math.log(n_samples) - 2.0 * log_likelihood


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Return ``log sum exp(a)`` along ``axis``, without overflow or underflow."""
    top = np.max(a, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - top), axis=axis, keepdims=True)) + top
    return np.asarray(np.squeeze(out, axis=axis))


def kmeans(
    X: ArrayLike,
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int = 300,
    tol: float = 1.0e-4,
) -> np.ndarray:
    """Partition ``X`` into ``n_clusters`` clusters by k-means; return the labels.

    1. **k-means++ seeding**: the first center is a sample drawn uniformly;
       each next one is drawn with a probability proportional to the squared
       distance ``d^2`` of a sample to its closest center so far. As in
       scikit-learn, ``2 + floor(ln K)`` candidates are drawn at each step
       and the one that lowers ``sum d^2`` the most is kept.
    2. **Lloyd iterations**: every sample goes to its closest center, and
       every center moves to the mean of its samples, until the centers move
       by less than ``tol`` times the mean variance of the features (in
       total squared distance) or the labels no longer change. A cluster
       left empty gets the sample farthest from its center.

    Args:
        X: Samples, shape ``(n, D)``, ``n >= n_clusters``.
        n_clusters: ``K >= 1``.
        rng: Random generator of the seeding.
        max_iter: Largest number of Lloyd iterations.
        tol: Relative tolerance on the moves of the centers.

    Returns:
        The cluster of every sample, shape ``(n,)``, in ``0 .. K - 1``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n = X_arr.shape[0]
    if not 1 <= n_clusters <= n:
        raise ValueError(f"Need 1 <= n_clusters <= {n} samples, got {n_clusters}.")
    centers = _kmeans_plusplus(X_arr, n_clusters, rng)
    threshold = tol * float(np.mean(np.var(X_arr, axis=0)))
    labels = np.full(n, -1)
    for _ in range(max_iter):
        d2 = _squared_distances(X_arr, centers)
        new_labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            members = new_labels == k
            if members.any():
                new_centers[k] = X_arr[members].mean(axis=0)
            else:
                far = int(np.argmax(d2[np.arange(n), new_labels]))
                new_centers[k] = X_arr[far]
                new_labels[far] = k
        shift = float(np.sum((new_centers - centers) ** 2))
        centers = new_centers
        stable = np.array_equal(new_labels, labels)
        labels = new_labels
        if stable or shift <= threshold:
            break
    return np.asarray(np.argmin(_squared_distances(X_arr, centers), axis=1))


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return ``||x_i - c_k||^2``, shape ``(n, K)``."""
    diff = X[:, None, :] - centers[None, :, :]
    return np.asarray(np.einsum("ikd,ikd->ik", diff, diff))


def _kmeans_plusplus(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``K`` k-means++ centers drawn from ``X``, shape ``(K, D)``."""
    n = X.shape[0]
    n_trials = 2 + int(np.log(K))
    centers = np.empty((K, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        total = closest.sum()
        if total > 0:
            candidates = rng.choice(n, size=n_trials, p=closest / total)
        else:
            # Every sample sits on a center: any sample will do.
            candidates = rng.integers(n, size=n_trials)
        d2 = _squared_distances(X, X[candidates])
        pots = np.minimum(closest[:, None], d2).sum(axis=0)
        best = int(np.argmin(pots))
        centers[k] = X[candidates[best]]
        closest = np.minimum(closest, d2[:, best])
    return centers


class CustomGMM:
    """Mixture of ``K`` full-covariance Gaussians fitted by EM (module docstring).

    Args:
        n_components: ``K``, the number of Gaussians.
        tol: The fit stops when the mean log-likelihood per sample changes
            by less than ``tol`` in one EM round.
        reg_covar: ``c >= 0``, added to the diagonal of every covariance to
            keep it positive definite.
        max_iter: Largest number of EM rounds of one start.
        n_init: Number of starts; the one of largest final log-likelihood is
            kept.
        init_params: ``"kmeans"`` or ``"random"``, the initial
            responsibilities of each start.
        random_state: Seed of ``numpy.random.default_rng`` for the starts,
            for a reproducible fit.

    Attributes:
        weights_: ``pi_k``, shape ``(K,)``.
        means_: ``mu_k``, shape ``(K, D)``.
        covariances_: ``Sigma_k``, shape ``(K, D, D)``.
        precisions_cholesky_: ``L_k^{-T}``, with ``Sigma_k = L_k L_k^T`` the
            Cholesky factorization, shape ``(K, D, D)``:
            ``(x - mu_k)^T Sigma_k^{-1} (x - mu_k) = ||(x - mu_k) L_k^{-T}||^2``.
        converged_: Whether the kept start met ``tol``.
        n_iter_: Number of EM rounds of the kept start.
        lower_bound_: Its final mean log-likelihood per training sample.
        lower_bounds_: Its mean log-likelihood after every EM round.
    """

    def __init__(
        self,
        n_components: int = 1,
        tol: float = 1.0e-3,
        reg_covar: float = 1.0e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: int | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        if tol < 0 or reg_covar < 0:
            raise ValueError(
                f"tol and reg_covar must be >= 0, got {tol} and {reg_covar}."
            )
        if max_iter < 1 or n_init < 1:
            raise ValueError(
                f"max_iter and n_init must be >= 1, got {max_iter} and {n_init}."
            )
        if init_params not in INIT_PARAMS:
            raise ValueError(
                f"init_params must be one of {INIT_PARAMS}, got {init_params!r}."
            )
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X: ArrayLike, resp_init: ArrayLike | None = None) -> CustomGMM:
        """Fit the mixture to ``X`` by EM.

        Args:
            X: Samples, shape ``(n, D)``, with ``n >= max(2, K)``.
            resp_init: Initial responsibilities, shape ``(n, K)``, rows of
                non-negative numbers that sum to 1 (one-hot rows are a
                partition); they replace ``init_params``. Only with
                ``n_init = 1``.

        Returns:
            ``self``.

        Raises:
            ValueError: If the shapes are wrong, or a covariance is not
                positive definite.
        """
        X_arr = self._check_X(X, fitted=False)
        n, D = X_arr.shape
        K = self.n_components
        if n < max(2, K):
            raise ValueError(
                f"Need at least max(2, n_components = {K}) samples, got {n}."
            )
        resp_given: np.ndarray | None = None
        if resp_init is not None:
            resp_given = np.asarray(resp_init, dtype=np.float64)
            if resp_given.shape != (n, K):
                raise ValueError(
                    f"resp_init must have shape ({n}, {K}), got {resp_given.shape}."
                )
            if self.n_init != 1:
                raise ValueError("resp_init needs n_init = 1.")
        rng = np.random.default_rng(self.random_state)
        best: tuple[Any, ...] | None = None
        for _ in range(self.n_init):
            resp = (
                resp_given if resp_given is not None else self._initial_resp(X_arr, rng)
            )
            # The start: the M-step of the initial responsibilities, with the
            # weights n_k / n (scikit-learn does not renormalize them here).
            nk, means, covariances = self._gaussian_parameters(X_arr, resp)
            weights = nk / n
            precisions_chol = _precision_cholesky(covariances)
            lower_bound = -np.inf
            lower_bounds = []
            converged = False
            for n_iter in range(1, self.max_iter + 1):
                previous = lower_bound
                log_norm, log_resp = _log_resp(X_arr, weights, means, precisions_chol)
                nk, means, covariances = self._gaussian_parameters(
                    X_arr, np.exp(log_resp)
                )
                weights = nk / np.sum(nk)
                precisions_chol = _precision_cholesky(covariances)
                lower_bound = float(np.mean(log_norm))
                lower_bounds.append(lower_bound)
                if abs(lower_bound - previous) < self.tol:
                    converged = True
                    break
            if best is None or lower_bound > best[0]:
                best = (
                    lower_bound,
                    weights,
                    means,
                    covariances,
                    precisions_chol,
                    converged,
                    n_iter,
                    lower_bounds,
                )
        assert best is not None
        (
            self.lower_bound_,
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
            self.converged_,
            self.n_iter_,
            self.lower_bounds_,
        ) = best
        if not self.converged_:
            warnings.warn(
                f"CustomGMM did not converge in max_iter = {self.max_iter} EM "
                "rounds: increase max_iter or tol.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def _initial_resp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the responsibilities of one start, shape ``(n, K)``."""
        n, K = X.shape[0], self.n_components
        if self.init_params == "kmeans":
            resp = np.zeros((n, K))
            resp[np.arange(n), kmeans(X, K, rng)] = 1.0
            return resp
        resp = rng.uniform(size=(n, K))
        return np.asarray(resp / resp.sum(axis=1, keepdims=True))

    def _gaussian_parameters(
        self, X: np.ndarray, resp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The M-step: ``n_k``, ``mu_k`` and the regularized ``Sigma_k``."""
        nk = resp.sum(axis=0) + 10 * np.finfo(np.float64).eps
        means = (resp.T @ X) / nk[:, None]
        K, D = means.shape
        covariances = np.empty((K, D, D))
        for k in range(K):
            diff = X - means[k]
            covariances[k] = ((resp[:, k] * diff.T) @ diff) / nk[k]
            covariances[k].flat[:: D + 1] += self.reg_covar
        return nk, means, covariances

    def _check_X(self, X: ArrayLike, fitted: bool = True) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must have shape (n, D), got {X_arr.shape}.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values.")
        if fitted:
            if not hasattr(self, "means_"):
                raise RuntimeError("CustomGMM is not fitted; call fit first.")
            if X_arr.shape[1] != self.means_.shape[1]:
                raise ValueError(
                    f"X must have shape (n, {self.means_.shape[1]}), got {X_arr.shape}."
                )
        return X_arr

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Return ``log f(x)`` of every sample, shape ``(n,)``."""
        X_arr = self._check_X(X)
        return _log_resp(X_arr, self.weights_, self.means_, self.precisions_cholesky_)[
            0
        ]

    def score(self, X: ArrayLike) -> float:
        """Return the mean log-likelihood per sample, ``log L / n``."""
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the responsibilities ``r_ik`` of every sample, shape ``(n, K)``."""
        X_arr = self._check_X(X)
        log_resp = _log_resp(
            X_arr, self.weights_, self.means_, self.precisions_cholesky_
        )[1]
        return np.asarray(np.exp(log_resp))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the component of largest responsibility of every sample."""
        return np.asarray(np.argmax(self.predict_proba(X), axis=1))

    def n_parameters(self) -> int:
        """Return ``nu`` of :func:`n_parameters` for the fitted mixture."""
        K, D = self.means_.shape
        return n_parameters(K, D)

    def bic(self, X: ArrayLike) -> float:
        """Return :func:`bic_score` of the fitted mixture on ``X``."""
        X_arr = self._check_X(X)
        log_likelihood = float(np.sum(self.score_samples(X_arr)))
        return bic_score(log_likelihood, self.n_parameters(), X_arr.shape[0])


def _precision_cholesky(covariances: np.ndarray) -> np.ndarray:
    """Return ``L_k^{-T}`` of every ``Sigma_k = L_k L_k^T``, shape ``(K, D, D)``.

    Raises:
        ValueError: If a ``Sigma_k`` is not positive definite.
    """
    K, D, _ = covariances.shape
    out = np.empty_like(covariances)
    for k in range(K):
        try:
            L = np.linalg.cholesky(covariances[k])
        except np.linalg.LinAlgError as exc:
            raise ValueError(_COVARIANCE_ERROR) from exc
        out[k] = np.linalg.solve(L, np.eye(D)).T
    return out


def _log_resp(
    X: np.ndarray, weights: np.ndarray, means: np.ndarray, precisions_chol: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The E-step: ``log f(x_i)``, shape ``(n,)``, and ``log r_ik``, ``(n, K)``.

    ``log N(x; mu, Sigma) = -1/2 (D log 2 pi + ||(x - mu) P||^2) + log det P``
    with ``P = L^{-T}``, whose determinant is ``det(Sigma)^{-1/2}``.
    """
    n, D = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K))
    for k in range(K):
        P = precisions_chol[k]
        y = X @ P - means[k] @ P
        log_prob[:, k] = np.sum(y**2, axis=1)
    log_det = np.sum(np.log(np.diagonal(precisions_chol, axis1=1, axis2=2)), axis=1)
    weighted = -0.5 * (D * math.log(2 * math.pi) + log_prob) + log_det + np.log(weights)
    log_norm = logsumexp(weighted, axis=1)
    with np.errstate(under="ignore"):
        log_resp = weighted - log_norm[:, None]
    return log_norm, log_resp
