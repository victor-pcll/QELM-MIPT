"""Empirical ceilings of the Bayes accuracy on a feature set (issue #202).

When the ridge readout of ``Q`` from the final ``<Z_k>`` falls to chance
(0.5) at large ``N`` and high measurement rate ``p``, two causes are
possible:

1. **Decoherence**: the features of the two charge sectors, distributed as
   ``f_1(x) = P(x | Q_1)`` and ``f_2(x) = P(x | Q_2)``, overlap entirely.
   No classifier does better than chance: the Bayes error is 50 %.
2. **Learnability**: ``f_1`` and ``f_2`` are still apart, but not on the two
   sides of a hyperplane, so a linear readout cannot tell them apart.

Two classifiers that approach the Bayes rule tell the two cases apart: if
they stay well above 0.5 while the ridge readout is at 0.5, the information
is there but not linearly; if they fall to 0.5 too, it is gone.

- :class:`QDA`, quadratic discriminant analysis written in NumPy: each
  ``f_k`` is modeled as a Gaussian ``N(mu_k, Sigma_k)``, whose mean and
  covariance are those of the training samples of sector ``k``, and a
  sample goes to the sector of largest posterior. It is the Bayes rule when
  the ``f_k`` are Gaussian. Its attributes (``classes_``, ``priors_``,
  ``means_``, ``covariance_``, ``rotations_``, ``scalings_``) and its
  numbers are those of ``sklearn.discriminant_analysis.
  QuadraticDiscriminantAnalysis(store_covariance=True)`` (scikit-learn
  1.9), which the tests use as a cross-check only.
- K nearest neighbors (scikit-learn), with ``K = sqrt(n_train)`` neighbors
  weighted by their inverse distance: no assumption on the shape of the
  ``f_k``, and by the Cover-Hart theorem its error approaches the Bayes
  error when ``K`` and ``n_train / K`` both grow.

- :class:`GaussianMixtureBayes`, the Bayes rule with a mixture of ``K``
  Gaussians per sector (scikit-learn's ``GaussianMixture``, issue #200): for
  multimodal ``f_k``, e.g. the bundles of trajectories that the measurements
  make at high ``p``, where one Gaussian per sector blurs the clusters and
  the QDA can fall below the ridge readout.

:func:`evaluate_bayes_optimal_limit` scores the three on a train/test split.
:func:`pca_plane` and :func:`project_gaussians` bring the data and the
fitted Gaussians onto the plane of largest variance, to draw the contours
of ``f_1`` and ``f_2`` (:func:`QELM.utils.plots.draw_qda_contours`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]

# Defaults of evaluate_bayes_optimal_limit, so that a plot refits the same
# models as the run: the floor of the QDA variances and the number of
# Gaussians per sector of the mixture.
QDA_REG_PARAM = 1.0e-6
GMM_COMPONENTS = 3


class QDA:
    """Quadratic discriminant analysis, written in NumPy.

    For a class ``k`` with ``n_k`` training samples ``x_i``:

    - prior ``pi_k = n_k / n``;
    - mean ``mu_k = (1 / n_k) sum_i x_i``;
    - covariance ``Sigma_k = (1 - r) S_k + r I``, with
      ``S_k = (1 / n_k) sum_i (x_i - mu_k)(x_i - mu_k)^T`` (no ``n_k - 1``,
      as scikit-learn 1.9) and ``r`` the regularization ``reg_param``.

    ``Sigma_k = R_k diag(s_k) R_k^T`` is diagonalized once at the fit: the
    eigenvectors are ``rotations_[k]`` and the variances along them
    ``scalings_[k]``. The discriminant of a sample ``x`` is its
    log-posterior, up to a constant shared by the classes,

        ``delta_k(x) = -1/2 log det Sigma_k
                       - 1/2 (x - mu_k)^T Sigma_k^{-1} (x - mu_k)
                       + log pi_k``,

    with ``log det Sigma_k = sum log s_k`` and the quadratic form
    ``sum_j ((x - mu_k) . R_k[:, j])**2 / s_k[j]``. The predicted class is
    the one of largest ``delta_k``.

    Args:
        reg_param: ``r`` in ``[0, 1]``, which shrinks every covariance
            towards the identity; 0 is the plain estimate.
        tol: A covariance with a variance ``s_k[j] <= tol`` is not full
            rank, and the fit raises (as scikit-learn).

    Attributes:
        classes_: The sorted class labels, shape ``(K,)``.
        priors_: ``pi_k``, shape ``(K,)``.
        means_: ``mu_k``, shape ``(K, d)``.
        covariance_: ``Sigma_k``, shape ``(K, d, d)``.
        rotations_: ``R_k``, eigenvectors in columns, shape ``(K, d, d)``.
        scalings_: ``s_k``, in decreasing order, shape ``(K, d)``.
    """

    def __init__(self, reg_param: float = 0.0, tol: float = 1.0e-4) -> None:
        if not 0.0 <= reg_param <= 1.0:
            raise ValueError(f"reg_param must be in [0, 1], got {reg_param}.")
        self.reg_param = reg_param
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> QDA:
        """Estimate the prior, mean and covariance of every class.

        Args:
            X: Training features, shape ``(n, d)``.
            y: Training labels, shape ``(n,)``, at least two classes.

        Returns:
            ``self``.

        Raises:
            ValueError: If the shapes do not match, there are fewer than two
                classes or a class has a single sample.
            numpy.linalg.LinAlgError: If a covariance is not full rank
                (a variance ``<= tol``): add samples or raise ``reg_param``.
        """
        X_arr, y_arr = _as_samples(X, y)
        n, d = X_arr.shape
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        if self.classes_.size < 2:
            raise ValueError(
                f"Need at least two classes, got {self.classes_.size}: {self.classes_}."
            )
        self.priors_ = counts / n
        if np.any(counts == 1):
            raise ValueError(
                f"Class {self.classes_[counts == 1][0]} has a single sample: its "
                "covariance is not defined."
            )
        means, covariances, rotations, scalings = [], [], [], []
        for label, n_k in zip(self.classes_, counts, strict=True):
            X_k = X_arr[y_arr == label]
            mu = X_k.mean(axis=0)
            centered = X_k - mu
            S = centered.T @ centered / n_k
            sigma = (1.0 - self.reg_param) * S + self.reg_param * np.eye(d)
            # eigh returns increasing eigenvalues; flip to the decreasing order
            # of the singular values that scikit-learn uses.
            s, R = np.linalg.eigh(sigma)
            s, R = s[::-1], R[:, ::-1]
            rank = int(np.sum(s > self.tol))
            if rank < d:
                raise np.linalg.LinAlgError(
                    f"The covariance of class {label} has rank {rank} < {d} "
                    f"(variances <= tol = {self.tol}, {n_k} samples): add "
                    "samples or raise reg_param."
                )
            means.append(mu)
            covariances.append(sigma)
            rotations.append(R)
            scalings.append(s)
        self.means_ = np.array(means)
        self.covariance_ = np.array(covariances)
        self.rotations_ = np.array(rotations)
        self.scalings_ = np.array(scalings)
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return ``delta_k(x)`` for every sample and class.

        Args:
            X: Features, shape ``(n, d)``.

        Returns:
            Shape ``(n, K)``, column ``k`` for ``classes_[k]``. Unlike
            scikit-learn, which returns the difference of the two columns
            for two classes, every column is kept.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.means_.shape[1]:
            raise ValueError(
                f"X must have shape (n, {self.means_.shape[1]}), got {X_arr.shape}."
            )
        deltas = []
        for mu, R, s, prior in zip(
            self.means_, self.rotations_, self.scalings_, self.priors_, strict=True
        ):
            # Coordinates along the eigenvectors, in units of their spread.
            whitened = (X_arr - mu) @ R / np.sqrt(s)
            mahalanobis2 = np.sum(whitened**2, axis=1)
            deltas.append(-0.5 * (np.sum(np.log(s)) + mahalanobis2) + np.log(prior))
        return np.array(deltas).T

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the posterior ``P(k | x)`` of every class, shape ``(n, K)``."""
        delta = self.decision_function(X)
        # Softmax of the discriminants, shifted for stability.
        weights = np.exp(delta - delta.max(axis=1, keepdims=True))
        return np.asarray(weights / weights.sum(axis=1, keepdims=True))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the class of largest ``delta_k`` of every sample, ``(n,)``."""
        return np.asarray(self.classes_[np.argmax(self.decision_function(X), axis=1)])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of :meth:`predict` on ``(X, y)``."""
        X_arr, y_arr = _as_samples(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))


def knn_classifier(
    n_train: int, n_neighbors: int | None = None
) -> KNeighborsClassifier:
    """Return the K-NN estimator of the Bayes rule, not fitted.

    Args:
        n_train: Number of training samples.
        n_neighbors: ``K``; ``None`` takes ``round(sqrt(n_train))``, at least
            1 and at most ``n_train``.

    Returns:
        ``KNeighborsClassifier(n_neighbors=K, weights="distance")``.
    """
    if n_neighbors is None:
        n_neighbors = int(round(np.sqrt(n_train)))
    n_neighbors = max(1, min(int(n_neighbors), n_train))
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")


def evaluate_bayes_optimal_limit(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    reg_param: float = QDA_REG_PARAM,
    n_neighbors: int | None = None,
    gmm_components: int = GMM_COMPONENTS,
    seed: int = 0,
) -> dict[str, float]:
    """Score the QDA, the mixture and the K-NN on a split: ceilings of the accuracy.

    Args:
        X_train: Training features, shape ``(n_train, d)``, without a bias
            column (a constant column has no variance).
        y_train: Training labels, shape ``(n_train,)``, two classes.
        X_test: Test features, shape ``(n_test, d)``.
        y_test: Test labels, shape ``(n_test,)``.
        reg_param: Regularization of the QDA covariances (:class:`QDA`).
            The small default only keeps a feature of zero variance in a
            sector (e.g. a qubit measured last, always ``+-1``) from making
            its covariance singular; it does not change the result
            otherwise.
        n_neighbors: ``K`` of the K-NN, default ``round(sqrt(n_train))``.
        gmm_components: Number of Gaussians per sector of the mixture
            (:class:`GaussianMixtureBayes`).
        seed: ``random_state`` of the mixture, its only random step.

    Returns:
        ``bayes_estimate_qda``, ``bayes_estimate_gmm`` and
        ``bayes_estimate_knn``, the test accuracies, and ``n_neighbors``, the
        ``K`` of the K-NN. ``bayes_estimate_qda`` is ``nan`` when a sector
        has no more training samples than features: its covariance is then
        singular and the QDA would fit the noise of the training set, so it
        gives no ceiling. ``bayes_estimate_gmm`` is ``nan`` when a sector
        has fewer samples than ``gmm_components``.
    """
    X_tr, y_tr = _as_samples(X_train, y_train)
    X_te, y_te = _as_samples(X_test, y_test)
    _, counts = np.unique(y_tr, return_counts=True)
    if counts.min() > X_tr.shape[1]:
        qda = QDA(reg_param=reg_param, tol=0.0).fit(X_tr, y_tr)
        qda_accuracy = qda.score(X_te, y_te)
    else:
        qda_accuracy = float("nan")
    if counts.min() >= gmm_components:
        gmm = GaussianMixtureBayes(n_components=gmm_components, random_state=seed)
        gmm_accuracy = gmm.fit(X_tr, y_tr).score(X_te, y_te)
    else:
        gmm_accuracy = float("nan")
    knn = knn_classifier(len(y_tr), n_neighbors).fit(X_tr, y_tr)
    return {
        "bayes_estimate_qda": qda_accuracy,
        "bayes_estimate_gmm": gmm_accuracy,
        "bayes_estimate_knn": float(np.mean(knn.predict(X_te) == y_te)),
        "n_neighbors": float(knn.n_neighbors),
    }


class GaussianMixtureBayes:
    """Bayes rule with a mixture of ``K`` Gaussians per class (issue #200).

    The QDA models each ``f_k`` by one Gaussian. When the features of a
    sector are multimodal (several bundles of measurement trajectories), one
    Gaussian blurs them, and the QDA can miss a separation that is there.
    Here each ``f_k`` is a mixture

        ``f_k(x) = sum_m w_km N(x; mu_km, Sigma_km)``,

    fitted by expectation-maximization on the training samples of class
    ``k`` (``sklearn.mixture.GaussianMixture``, full covariances). The
    discriminant is the log-posterior up to a shared constant,
    ``delta_k(x) = log f_k(x) + log pi_k``, and the predicted class the one
    of largest ``delta_k``, as in :class:`QDA`. With ``K = 1`` and
    ``reg_covar = 0`` it is the QDA.

    Args:
        n_components: ``K``, the number of Gaussians per class.
        reg_covar: Added to the diagonal of every covariance, so that a
            feature that does not vary within a component (a qubit at
            ``+-1``) does not make it singular.
        random_state: Seed of the initialization (k-means) of each mixture,
            for a reproducible fit.
        n_init: Number of initializations of each mixture; the best
            likelihood is kept.

    Attributes:
        classes_: The sorted class labels, shape ``(C,)``.
        priors_: ``pi_k``, shape ``(C,)``.
        mixtures_: The fitted ``GaussianMixture`` of each class, in the
            order of ``classes_``.
    """

    def __init__(
        self,
        n_components: int = 3,
        reg_covar: float = 1.0e-6,
        random_state: int | None = None,
        n_init: int = 1,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X: ArrayLike, y: ArrayLike) -> GaussianMixtureBayes:
        """Fit the prior and the mixture of every class.

        Args:
            X: Training features, shape ``(n, d)``.
            y: Training labels, shape ``(n,)``, at least two classes.

        Returns:
            ``self``.

        Raises:
            ValueError: If the shapes do not match, there are fewer than two
                classes or a class has fewer samples than ``n_components``.
        """
        X_arr, y_arr = _as_samples(X, y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        if self.classes_.size < 2:
            raise ValueError(
                f"Need at least two classes, got {self.classes_.size}: {self.classes_}."
            )
        if counts.min() < self.n_components:
            raise ValueError(
                f"Class {self.classes_[np.argmin(counts)]} has {counts.min()} "
                f"samples, fewer than n_components = {self.n_components}."
            )
        self.priors_ = counts / len(y_arr)
        self.mixtures_ = [
            GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                n_init=self.n_init,
            ).fit(X_arr[y_arr == label])
            for label in self.classes_
        ]
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return ``delta_k(x) = log f_k(x) + log pi_k``, shape ``(n, C)``."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.array(
            [
                mixture.score_samples(X_arr) + np.log(prior)
                for mixture, prior in zip(self.mixtures_, self.priors_, strict=True)
            ]
        ).T

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the posterior ``P(k | x)`` of every class, shape ``(n, C)``."""
        delta = self.decision_function(X)
        weights = np.exp(delta - delta.max(axis=1, keepdims=True))
        return np.asarray(weights / weights.sum(axis=1, keepdims=True))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the class of largest ``delta_k`` of every sample, ``(n,)``."""
        return np.asarray(self.classes_[np.argmax(self.decision_function(X), axis=1)])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of :meth:`predict` on ``(X, y)``."""
        X_arr, y_arr = _as_samples(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))


def pca_plane(
    X: ArrayLike, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the plane of largest variance of ``X`` (principal components).

    Args:
        X: Samples, shape ``(n, d)`` with ``d >= n_components``.
        n_components: Dimension of the projection.

    Returns:
        ``(center, components, variances)``: the mean of ``X``, shape
        ``(d,)``; the orthonormal principal directions in rows, shape
        ``(n_components, d)``, each signed so that its largest entry is
        positive; the variance of ``X`` along each, decreasing. A sample
        ``x`` projects to ``(x - center) @ components.T``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[1] < n_components:
        raise ValueError(
            f"X must have shape (n, d >= {n_components}), got {X_arr.shape}."
        )
    center = X_arr.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_arr - center, full_matrices=False)
    components = Vt[:n_components]
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(n_components), largest])
    components = components * signs[:, None]
    variances = S[:n_components] ** 2 / X_arr.shape[0]
    return center, components, variances


def project_gaussians(
    means: ArrayLike,
    covariances: ArrayLike,
    center: ArrayLike,
    components: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Gaussians ``N(mu_k, Sigma_k)`` onto a plane.

    The projection ``P (x - c)`` of a Gaussian is the Gaussian
    ``N(P (mu_k - c), P Sigma_k P^T)``: exactly the 2D marginal of ``f_k``
    on the plane, not a slice through it.

    Args:
        means: ``mu_k``, shape ``(K, d)``, e.g. :attr:`QDA.means_`.
        covariances: ``Sigma_k``, shape ``(K, d, d)``.
        center: ``c``, shape ``(d,)``, from :func:`pca_plane`.
        components: ``P``, shape ``(m, d)``, from :func:`pca_plane`.

    Returns:
        The projected means, shape ``(K, m)``, and covariances,
        ``(K, m, m)``.
    """
    P = np.asarray(components, dtype=np.float64)
    mu = (np.asarray(means, dtype=np.float64) - np.asarray(center)) @ P.T
    sigma = P @ np.asarray(covariances, dtype=np.float64) @ P.T
    return mu, sigma


def _as_samples(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X`` as a float ``(n, d)`` array and ``y`` as an ``(n,)`` array.

    Raises:
        ValueError: If ``X`` is not 2-dimensional, ``y`` not 1-dimensional,
            their lengths differ or they are empty.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2 or y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Need X of shape (n, d) and y of shape (n,), got {X_arr.shape} "
            f"and {y_arr.shape}."
        )
    if y_arr.size == 0:
        raise ValueError("X and y must not be empty.")
    return X_arr, y_arr
