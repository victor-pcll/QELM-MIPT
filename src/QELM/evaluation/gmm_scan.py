"""How robust is the Gaussian-mixture ceiling? A scan in ``K``, BIC (issue #215).

:class:`~QELM.evaluation.bayes_optimal.GaussianMixtureBayes` fits the same
number ``K`` of Gaussians to each charge sector, and the runs of
``exp_QELM_Q`` log its accuracy at ``K = 3`` (``bayes_estimate_gmm``). That
is a ceiling of the accuracy only if it does not move when ``K`` changes:

- too few Gaussians blur the bundles of trajectories that the measurements
  make at high ``p`` (``K = 1`` is the QDA, which falls below the ridge
  readout there, #210);
- too many fit the noise of the training set, and the test accuracy falls
  again.

Two tools to check it:

- :class:`BICGaussianMixtureBayes`: the same Bayes rule, but the number of
  Gaussians of each sector is chosen on its training samples by the
  Bayesian information criterion,
  ``BIC(K) = -2 log L_K + n_params(K) log n``, the smallest over
  ``K = 1 .. k_max``. With full covariances in ``d`` dimensions a mixture
  has ``n_params(K) = K d (d + 3) / 2 + K - 1`` free numbers, so BIC only
  adds a Gaussian when the likelihood gains more than it costs. The
  selected ``K`` of a sector (``n_components_``) counts the bundles the data
  can support: it is a measure of the multimodality of ``f_k``.
- :func:`gmm_component_scan`: on one train/test split, the test accuracy of
  :class:`GaussianMixtureBayes` for each ``K`` of a list, of the BIC rule,
  and of the K-NN (no density model, the independent cross-check), with the
  ``K`` selected in each sector.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from QELM.evaluation.bayes_optimal import (
    GaussianMixtureBayes,
    _as_samples,
    gaussian_mixture,
    knn_classifier,
)

# Numbers of Gaussians per sector scanned by gmm_component_scan, and the
# largest one BIC may select.
SCAN_COMPONENTS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
BIC_MAX_COMPONENTS = 8


def bic_curve(
    X: ArrayLike,
    k_max: int = BIC_MAX_COMPONENTS,
    reg_covar: float = 1.0e-6,
    random_state: int | None = None,
    n_init: int = 1,
    backend: str = "sklearn",
) -> tuple[np.ndarray, list[Any]]:
    """Fit mixtures of ``K = 1 .. k_max`` full-covariance Gaussians to ``X``.

    Args:
        X: Samples of one class, shape ``(n, d)``.
        k_max: Largest ``K``; values of ``K`` above ``n`` are skipped.
        reg_covar: As in :class:`GaussianMixtureBayes`.
        random_state: Seed of the initialization of every mixture.
        n_init: Number of initializations of every mixture.
        backend: ``"sklearn"`` or ``"numpy"``, who fits the mixtures
            (:func:`~QELM.evaluation.bayes_optimal.gaussian_mixture`).

    Returns:
        ``bic``, shape ``(k_max,)``, ``bic[K - 1]`` the BIC of the mixture of
        ``K`` Gaussians (``inf`` for a skipped ``K``), and the fitted mixtures
        of ``K = 1 .. min(k_max, n)``.

    Raises:
        ValueError: If ``k_max < 1`` or ``X`` is not a non-empty 2D array.
    """
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}.")
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError(f"X must have shape (n >= 1, d), got {X_arr.shape}.")
    bic = np.full(k_max, np.inf)
    mixtures = []
    for K in range(1, min(k_max, X_arr.shape[0]) + 1):
        mixture = gaussian_mixture(K, reg_covar, random_state, n_init, backend)
        mixture.fit(X_arr)
        bic[K - 1] = mixture.bic(X_arr)
        mixtures.append(mixture)
    return bic, mixtures


class BICGaussianMixtureBayes(GaussianMixtureBayes):
    """Bayes rule with a mixture per class whose size is chosen by BIC.

    As :class:`GaussianMixtureBayes`, but the mixture of class ``k`` has the
    ``K_k`` in ``1 .. k_max`` of smallest BIC on its training samples
    (:func:`bic_curve`), so the two sectors can have different numbers of
    Gaussians.

    Args:
        k_max: Largest number of Gaussians per class.
        reg_covar: As in :class:`GaussianMixtureBayes`.
        random_state: As in :class:`GaussianMixtureBayes`.
        n_init: As in :class:`GaussianMixtureBayes`.
        backend: As in :class:`GaussianMixtureBayes`.

    Attributes:
        classes_: The sorted class labels, shape ``(C,)``.
        priors_: ``pi_k``, shape ``(C,)``.
        mixtures_: The selected mixture of each class.
        n_components_: ``K_k``, the selected number of Gaussians of each
            class, shape ``(C,)``.
        bic_: ``bic_[k, K - 1]``, the BIC of ``K`` Gaussians in class ``k``,
            shape ``(C, k_max)``.
    """

    def __init__(
        self,
        k_max: int = BIC_MAX_COMPONENTS,
        reg_covar: float = 1.0e-6,
        random_state: int | None = None,
        n_init: int = 1,
        backend: str = "sklearn",
    ) -> None:
        super().__init__(
            n_components=k_max,
            reg_covar=reg_covar,
            random_state=random_state,
            n_init=n_init,
            backend=backend,
        )
        self.k_max = k_max

    def fit(self, X: ArrayLike, y: ArrayLike) -> BICGaussianMixtureBayes:
        """Fit the prior and select and fit the mixture of every class.

        Args:
            X: Training features, shape ``(n, d)``.
            y: Training labels, shape ``(n,)``, at least two classes.

        Returns:
            ``self``.

        Raises:
            ValueError: If the shapes do not match or there are fewer than
                two classes.
        """
        X_arr, y_arr = _as_samples(X, y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        if self.classes_.size < 2:
            raise ValueError(
                f"Need at least two classes, got {self.classes_.size}: {self.classes_}."
            )
        self.priors_ = counts / len(y_arr)
        bics, self.mixtures_ = [], []
        for label in self.classes_:
            bic, mixtures = bic_curve(
                X_arr[y_arr == label],
                self.k_max,
                self.reg_covar,
                self.random_state,
                self.n_init,
                self.backend,
            )
            bics.append(bic)
            self.mixtures_.append(mixtures[int(np.argmin(bic))])
        self.bic_ = np.array(bics)
        self.n_components_ = np.argmin(self.bic_, axis=1) + 1
        return self


def gmm_component_scan(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    components: Sequence[int] = SCAN_COMPONENTS,
    k_max: int = BIC_MAX_COMPONENTS,
    seed: int = 0,
) -> dict[str, float]:
    """Score the mixture rule for several ``K``, with BIC, and the K-NN, on one split.

    Args:
        X_train: Training features, shape ``(n_train, d)``, no bias column.
        y_train: Training labels, shape ``(n_train,)``, two classes.
        X_test: Test features, shape ``(n_test, d)``.
        y_test: Test labels, shape ``(n_test,)``.
        components: Numbers ``K`` of Gaussians per sector to score.
        k_max: Largest ``K`` that BIC may select.
        seed: ``random_state`` of every mixture, as in
            :func:`~QELM.evaluation.bayes_optimal.evaluate_bayes_optimal_limit`,
            so that ``accuracy_gmm_K3`` is the run's ``bayes_estimate_gmm``.

    Returns:
        ``accuracy_gmm_K<K>`` for each ``K`` of ``components`` (``nan`` when
        a sector has fewer than ``K`` training samples),
        ``accuracy_gmm_bic``, ``accuracy_knn`` (the K-NN of
        ``bayes_estimate_knn``), and ``k_bic_<i>``, the ``K`` selected by BIC
        in the ``i``-th class in sorted order, with ``k_bic_mean`` their
        mean.
    """
    X_tr, y_tr = _as_samples(X_train, y_train)
    X_te, y_te = _as_samples(X_test, y_test)
    _, counts = np.unique(y_tr, return_counts=True)
    out: dict[str, float] = {}
    for K in components:
        if counts.min() >= K:
            gmm = GaussianMixtureBayes(n_components=K, random_state=seed)
            out[f"accuracy_gmm_K{K}"] = gmm.fit(X_tr, y_tr).score(X_te, y_te)
        else:
            out[f"accuracy_gmm_K{K}"] = float("nan")
    bic = BICGaussianMixtureBayes(k_max=k_max, random_state=seed).fit(X_tr, y_tr)
    out["accuracy_gmm_bic"] = bic.score(X_te, y_te)
    for i, K in enumerate(bic.n_components_):
        out[f"k_bic_{i}"] = float(K)
    out["k_bic_mean"] = float(np.mean(bic.n_components_))
    knn = knn_classifier(len(y_tr)).fit(X_tr, y_tr)
    out["accuracy_knn"] = float(np.mean(knn.predict(X_te) == y_te))
    return out
