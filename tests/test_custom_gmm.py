"""The NumPy Gaussian mixture, ``QELM.evaluation.gmm`` (issue #223)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from QELM.evaluation.bayes_optimal import QDA, GaussianMixtureBayes, gaussian_mixture
from QELM.evaluation.gmm import (
    CustomGMM,
    bic_score,
    kmeans,
    logsumexp,
    n_parameters,
)
from QELM.evaluation.gmm_scan import BICGaussianMixtureBayes, bic_curve


def _blobs(seed: int = 0, binary: bool = True) -> np.ndarray:
    """Three Gaussian clusters in 5 dimensions; the last feature at +-1."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(size=(120, 5)),
            rng.normal(size=(90, 5)) + 3.0,
            0.3 * rng.normal(size=(60, 5)) - 2.0,
        ]
    )
    if binary:
        X[:, 4] = np.sign(X[:, 4])
    return X


def _kmeans_resp(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    """The initial responsibilities that GaussianMixture(random_state=seed) uses."""
    labels = KMeans(n_clusters=K, n_init=1, random_state=seed).fit(X).labels_
    resp = np.zeros((len(X), K))
    resp[np.arange(len(X)), labels] = 1.0
    return resp


@pytest.mark.parametrize("K", [1, 2, 3, 5])
@pytest.mark.parametrize("binary", [False, True])
def test_same_em_as_sklearn_from_the_same_start(K: int, binary: bool) -> None:
    """From scikit-learn's k-means partition, the EM rounds agree to rounding."""
    X = _blobs(binary=binary)
    ref = GaussianMixture(n_components=K, covariance_type="full", random_state=3)
    ref.fit(X)
    ours = CustomGMM(n_components=K).fit(X, resp_init=_kmeans_resp(X, K, 3))
    assert ours.n_iter_ == ref.n_iter_
    assert ours.converged_ == ref.converged_
    np.testing.assert_allclose(ours.lower_bounds_, ref.lower_bounds_, atol=1e-12)
    np.testing.assert_allclose(ours.weights_, ref.weights_, atol=1e-12)
    np.testing.assert_allclose(ours.means_, ref.means_, atol=1e-10)
    np.testing.assert_allclose(ours.covariances_, ref.covariances_, atol=1e-10)
    np.testing.assert_allclose(
        ours.precisions_cholesky_, ref.precisions_cholesky_, rtol=1e-8, atol=1e-8
    )
    X_test = _blobs(seed=1, binary=binary)
    np.testing.assert_allclose(
        ours.score_samples(X_test), ref.score_samples(X_test), rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-8
    )
    assert math.isclose(ours.bic(X), ref.bic(X), rel_tol=1e-12, abs_tol=1e-8)
    assert ours.n_parameters() == ref._n_parameters()


@pytest.mark.parametrize("K,D", [(1, 1), (1, 7), (3, 7), (8, 7), (2, 3)])
def test_n_parameters(K: int, D: int) -> None:
    """``nu = K (D + D (D + 1) / 2) + K - 1``, as scikit-learn counts."""
    assert n_parameters(K, D) == K * (D + D * (D + 1) / 2) + K - 1
    ref = GaussianMixture(n_components=K)
    ref.means_ = np.zeros((K, D))
    assert n_parameters(K, D) == ref._n_parameters()


def test_n_parameters_by_hand() -> None:
    """One Gaussian in 2D: 2 means + 3 covariances + 0 free weights."""
    assert n_parameters(1, 2) == 5
    assert n_parameters(2, 2) == 11
    with pytest.raises(ValueError):
        n_parameters(0, 2)


def test_bic_score_formula() -> None:
    assert bic_score(-10.0, 5, 100) == pytest.approx(5 * math.log(100) + 20.0)
    with pytest.raises(ValueError):
        bic_score(0.0, 1, 0)


def test_single_gaussian_is_the_maximum_likelihood_estimate() -> None:
    """K = 1 converges in one round to the mean and the biased covariance."""
    X = _blobs(binary=False)
    gmm = CustomGMM(n_components=1, reg_covar=0.0).fit(X)
    np.testing.assert_allclose(gmm.means_[0], X.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(
        gmm.covariances_[0], np.cov(X, rowvar=False, bias=True), atol=1e-12
    )
    np.testing.assert_allclose(gmm.weights_, [1.0])
    # It is the QDA density of one class: same log-likelihood.
    qda = QDA().fit(np.vstack([X, X + 100]), np.r_[np.zeros(len(X)), np.ones(len(X))])
    delta = qda.decision_function(X)[:, 0] - np.log(0.5)
    D = X.shape[1]
    np.testing.assert_allclose(
        gmm.score_samples(X), delta - 0.5 * D * math.log(2 * math.pi), atol=1e-9
    )


def test_log_likelihood_never_decreases() -> None:
    """Each EM round cannot lower log L (up to rounding).

    Only without ``reg_covar``: ``Sigma_k + c I`` is not the maximizer of the
    M-step, so near convergence log L can drop by ``~1e-9`` per round.
    """
    X = _blobs(binary=False)
    for init in ("kmeans", "random"):
        gmm = CustomGMM(
            n_components=4, init_params=init, random_state=0, tol=1e-10, reg_covar=0.0
        )
        gmm.fit(X)
        assert np.all(np.diff(gmm.lower_bounds_) >= -1e-10)
        assert gmm.lower_bound_ == gmm.lower_bounds_[-1]


def test_posterior_and_density_are_normalized() -> None:
    X = _blobs()
    gmm = CustomGMM(n_components=3, random_state=0).fit(X)
    resp = gmm.predict_proba(X)
    assert resp.shape == (len(X), 3)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0)
    assert np.all(resp >= 0)
    np.testing.assert_allclose(gmm.weights_.sum(), 1.0)
    assert gmm.predict(X).shape == (len(X),)
    # The density of a 1D mixture integrates to 1.
    x = np.random.default_rng(0).normal(size=(200, 1))
    one_d = CustomGMM(n_components=2, random_state=0).fit(np.vstack([x, x + 4]))
    grid = np.linspace(-10, 15, 20001)[:, None]
    integral = np.trapezoid(np.exp(one_d.score_samples(grid)), grid[:, 0])
    assert integral == pytest.approx(1.0, abs=1e-6)


def test_covariances_stay_positive_definite() -> None:
    """A feature constant inside a component: reg_covar keeps Sigma invertible."""
    X = _blobs()
    X[:, 3] = 1.0
    gmm = CustomGMM(n_components=3, random_state=0, reg_covar=1e-6).fit(X)
    for sigma in gmm.covariances_:
        np.testing.assert_allclose(sigma[3, 3], 1e-6, rtol=1e-6)
        assert np.linalg.eigvalsh(sigma).min() >= 1e-6 * (1 - 1e-6)
    with pytest.raises(ValueError, match="positive definite"):
        CustomGMM(n_components=3, random_state=0, reg_covar=0.0).fit(X)


def test_reproducible_with_the_seed() -> None:
    X = _blobs()
    a = CustomGMM(n_components=4, random_state=5, n_init=3).fit(X)
    b = CustomGMM(n_components=4, random_state=5, n_init=3).fit(X)
    np.testing.assert_array_equal(a.means_, b.means_)
    assert a.lower_bound_ == b.lower_bound_


def test_n_init_keeps_the_best_start() -> None:
    X = _blobs()
    best = CustomGMM(n_components=5, random_state=2, n_init=5).fit(X)
    # The first start of n_init = 5 is the only start of n_init = 1.
    first = CustomGMM(n_components=5, random_state=2, n_init=1).fit(X)
    assert best.lower_bound_ >= first.lower_bound_


def test_own_kmeans_finds_separated_clusters() -> None:
    """Well separated clusters: k-means recovers them, the mixture too."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    truth = np.repeat(np.arange(3), 50)
    X = centers[truth] + rng.normal(scale=0.5, size=(150, 2))
    labels = kmeans(X, 3, np.random.default_rng(1))
    # Same partition up to a renaming of the clusters.
    assert len({(t, c) for t, c in zip(truth, labels, strict=True)}) == 3
    gmm = CustomGMM(n_components=3, random_state=1).fit(X)
    found = gmm.means_[np.argsort(gmm.means_[:, 0] + 2 * gmm.means_[:, 1])]
    np.testing.assert_allclose(found, centers, atol=0.3)
    ref = GaussianMixture(n_components=3, random_state=1).fit(X)
    assert gmm.score(X) == pytest.approx(ref.score(X), abs=1e-6)


def test_logsumexp_is_stable() -> None:
    a = np.array([[1000.0, 1000.0], [-1000.0, -np.inf], [-np.inf, -np.inf]])
    out = logsumexp(a, axis=1)
    np.testing.assert_allclose(out[:2], [1000 + math.log(2), -1000.0])
    assert out[2] == -np.inf


def test_bad_arguments() -> None:
    X = _blobs()
    with pytest.raises(ValueError):
        CustomGMM(n_components=0)
    with pytest.raises(ValueError):
        CustomGMM(init_params="spectral")
    with pytest.raises(ValueError):
        CustomGMM(n_components=3).fit(X, resp_init=np.ones((len(X), 2)))
    with pytest.raises(ValueError):
        CustomGMM(n_components=2, n_init=2).fit(X, resp_init=np.ones((len(X), 2)) / 2)
    with pytest.raises(ValueError):
        CustomGMM(n_components=5).fit(X[:3])
    with pytest.raises(RuntimeError):
        CustomGMM().score_samples(X)
    gmm = CustomGMM().fit(X)
    with pytest.raises(ValueError):
        gmm.score_samples(X[:, :2])


def test_warns_when_not_converged() -> None:
    with pytest.warns(RuntimeWarning, match="did not converge"):
        CustomGMM(n_components=4, max_iter=1, tol=0.0, random_state=0).fit(_blobs())


def test_backend_of_the_bayes_rule() -> None:
    """``backend="numpy"`` fits CustomGMM mixtures; the classes agree on blobs."""
    X = _blobs(binary=False)
    y = np.r_[np.zeros(120), np.ones(150)]
    ours = GaussianMixtureBayes(n_components=2, random_state=0, backend="numpy")
    ref = GaussianMixtureBayes(n_components=2, random_state=0)
    ours.fit(X, y)
    ref.fit(X, y)
    assert all(isinstance(m, CustomGMM) for m in ours.mixtures_)
    assert all(isinstance(m, GaussianMixture) for m in ref.mixtures_)
    assert np.mean(ours.predict(X) == ref.predict(X)) > 0.95
    with pytest.raises(ValueError):
        GaussianMixtureBayes(backend="torch")
    with pytest.raises(ValueError):
        gaussian_mixture(2, backend="torch")


def test_backend_of_the_bic_selection() -> None:
    """Three clear clusters: both backends select K = 3 by BIC."""
    X = _blobs(binary=False)
    for backend in ("sklearn", "numpy"):
        bic, mixtures = bic_curve(X, k_max=5, random_state=0, backend=backend)
        assert int(np.argmin(bic)) + 1 == 3
        assert len(mixtures) == 5
    y = np.r_[np.zeros(120), np.ones(150)]
    rule = BICGaussianMixtureBayes(k_max=4, random_state=0, backend="numpy")
    rule.fit(X, y)
    assert rule.bic_.shape == (2, 4)
    assert all(isinstance(m, CustomGMM) for m in rule.mixtures_)
