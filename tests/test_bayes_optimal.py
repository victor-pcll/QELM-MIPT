"""Bayes ceilings of the accuracy: QDA in NumPy, K-NN, PCA plane (issue #202).

The hand-coded QDA is checked against the formula of its discriminant and
against scikit-learn's ``QuadraticDiscriminantAnalysis(store_covariance=True)``,
used here as a cross-check only.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.discriminant_analysis import (  # type: ignore[import-untyped]
    QuadraticDiscriminantAnalysis,
)

from QELM.evaluation.bayes_optimal import (
    QDA,
    GaussianMixtureBayes,
    evaluate_bayes_optimal_limit,
    knn_classifier,
    pca_plane,
    project_gaussians,
)


def _gaussians(
    n: int, means: list[list[float]], scales: list[list[float]], seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Samples of two (or more) Gaussian classes with random rotations."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, (mu, s) in enumerate(zip(means, scales, strict=True)):
        d = len(mu)
        R, _ = np.linalg.qr(rng.normal(size=(d, d)))
        X.append(rng.normal(size=(n, d)) * s @ R.T + mu)
        y.append(np.full(n, k))
    order = rng.permutation(len(means) * n)
    return np.vstack(X)[order], np.concatenate(y)[order].astype(float)


@pytest.mark.parametrize("reg_param", [0.0, 0.1])
def test_qda_matches_scikit_learn(reg_param: float) -> None:
    """Same priors, means, covariances, posteriors and predictions as sklearn."""
    X, y = _gaussians(200, [[0, 0, 0], [1, 0.5, -0.5]], [[1, 0.5, 0.2], [0.3, 1, 2]])
    X_test, _ = _gaussians(50, [[0, 0, 0], [1, 0.5, -0.5]], [[1, 1, 1]] * 2, 1)
    ours = QDA(reg_param=reg_param).fit(X, y)
    ref = QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=reg_param).fit(
        X, y
    )
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    np.testing.assert_allclose(ours.priors_, ref.priors_)
    np.testing.assert_allclose(ours.means_, ref.means_)
    np.testing.assert_allclose(ours.covariance_, np.array(ref.covariance_), atol=1e-12)
    np.testing.assert_allclose(ours.scalings_, np.array(ref.scalings_), rtol=1e-10)
    # sklearn returns delta_1 - delta_0 for two classes.
    delta = ours.decision_function(X_test)
    np.testing.assert_allclose(
        delta[:, 1] - delta[:, 0], ref.decision_function(X_test), rtol=1e-8
    )
    np.testing.assert_allclose(
        ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-10
    )
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))


def test_discriminant_is_the_gaussian_log_posterior() -> None:
    """delta_k = -1/2 log det S_k - 1/2 (x - mu)^T S_k^-1 (x - mu) + log pi_k."""
    X, y = _gaussians(80, [[0, 0], [2, 1]], [[1, 0.4], [0.5, 1.5]])
    X = X[:150]  # unequal classes, so the priors matter
    y = y[:150]
    qda = QDA().fit(X, y)
    x = np.array([[0.3, -0.2], [1.5, 2.0]])
    for k, label in enumerate(qda.classes_):
        X_k = X[y == label]
        mu = X_k.mean(axis=0)
        S = np.cov(X_k.T, bias=True)
        diff = x - mu
        expected = (
            -0.5 * np.linalg.slogdet(S)[1]
            - 0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.inv(S), diff)
            + np.log(len(X_k) / len(X))
        )
        np.testing.assert_allclose(qda.decision_function(x)[:, k], expected)
        np.testing.assert_allclose(qda.means_[k], mu)
        np.testing.assert_allclose(qda.covariance_[k], S)
    # The eigendecomposition rebuilds the covariance.
    for R, s, S in zip(qda.rotations_, qda.scalings_, qda.covariance_, strict=True):
        np.testing.assert_allclose(R @ np.diag(s) @ R.T, S, atol=1e-12)
        assert np.all(np.diff(s) <= 0)
    np.testing.assert_allclose(qda.predict_proba(x).sum(axis=1), 1.0)


def test_qda_errors() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="two classes"):
        QDA().fit(X, np.zeros(3))
    with pytest.raises(ValueError, match="single sample"):
        QDA().fit(X, np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        QDA().fit(X, np.zeros(2))
    with pytest.raises(ValueError, match="reg_param"):
        QDA(reg_param=1.5)
    # Two points of a class in 2D: a covariance of rank 1.
    X4 = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 0.0], [6.0, 2.0]])
    with pytest.raises(np.linalg.LinAlgError, match="rank"):
        QDA().fit(X4, np.array([0.0, 0.0, 1.0, 1.0]))
    # The regularization makes it full rank.
    QDA(reg_param=0.1).fit(X4, np.array([0.0, 0.0, 1.0, 1.0]))


def test_ceilings_on_separated_and_on_identical_sectors() -> None:
    """Far apart: both at 1 (Scenario B limit). Same law: both near 0.5
    (Scenario A, decoherence)."""
    X, y = _gaussians(300, [[0, 0, 0], [10, 10, 10]], [[1, 1, 1]] * 2)
    out = evaluate_bayes_optimal_limit(X[:400], y[:400], X[400:], y[400:])
    assert out["bayes_estimate_qda"] == 1.0
    assert out["bayes_estimate_gmm"] == 1.0
    assert out["bayes_estimate_knn"] == 1.0
    assert out["n_neighbors"] == 20  # round(sqrt(400))

    X, y = _gaussians(1000, [[0, 0, 0], [0, 0, 0]], [[1, 1, 1]] * 2, seed=3)
    out = evaluate_bayes_optimal_limit(X[:1000], y[:1000], X[1000:], y[1000:])
    assert abs(out["bayes_estimate_qda"] - 0.5) < 0.06
    assert abs(out["bayes_estimate_gmm"] - 0.5) < 0.06
    assert abs(out["bayes_estimate_knn"] - 0.5) < 0.06


def test_learnability_wall_is_seen_by_the_qda_not_by_a_linear_readout() -> None:
    """Same mean, different spread: no hyperplane separates the sectors, the
    QDA does (the learnability scenario of the issue)."""
    X, y = _gaussians(1000, [[0, 0, 0], [0, 0, 0]], [[0.2] * 3, [2.0] * 3], seed=4)
    train = 1500
    out = evaluate_bayes_optimal_limit(X[:train], y[:train], X[train:], y[train:])
    # Least squares on [X, 1], thresholded at the midpoint of the labels.
    phi = np.hstack([X, np.ones((len(X), 1))])
    W, *_ = np.linalg.lstsq(phi[:train], y[:train], rcond=None)
    linear = np.mean((phi[train:] @ W > 0.5) == y[train:])
    assert linear < 0.6
    assert out["bayes_estimate_qda"] > 0.9
    assert out["bayes_estimate_knn"] > 0.85


def test_qda_ceiling_is_nan_without_enough_samples_per_sector() -> None:
    X, y = _gaussians(4, [[0] * 5, [3] * 5], [[1] * 5] * 2)
    out = evaluate_bayes_optimal_limit(X, y, X, y)
    assert np.isnan(out["bayes_estimate_qda"])
    assert 0.0 <= out["bayes_estimate_gmm"] <= 1.0
    assert 0.0 <= out["bayes_estimate_knn"] <= 1.0
    # Fewer samples in a sector than Gaussians: no mixture either.
    out = evaluate_bayes_optimal_limit(X, y, X, y, gmm_components=5)
    assert np.isnan(out["bayes_estimate_gmm"])


def test_mixture_follows_clusters_that_split_one_sector() -> None:
    """Sector 0 in two clusters on both sides of sector 1, as the measured
    <Z_k> at high p: one Gaussian per sector blurs them, the mixture and
    the K-NN do not."""
    rng = np.random.default_rng(5)
    n = 600
    side = rng.choice([-2.0, 2.0], size=n)
    X0 = np.column_stack([side, np.zeros(n)]) + 0.3 * rng.normal(size=(n, 2))
    X1 = np.column_stack([0.3 * rng.normal(size=n), 2.0 * rng.normal(size=n)])
    X = np.vstack([X0, X1])
    y = np.repeat([0.0, 1.0], n)
    order = rng.permutation(2 * n)
    X, y = X[order], y[order]
    out = evaluate_bayes_optimal_limit(X[:800], y[:800], X[800:], y[800:], seed=1)
    assert out["bayes_estimate_gmm"] > 0.97
    assert out["bayes_estimate_knn"] > 0.95
    assert out["bayes_estimate_gmm"] > out["bayes_estimate_qda"] + 0.05
    # The seed fixes the mixture.
    again = evaluate_bayes_optimal_limit(X[:800], y[:800], X[800:], y[800:], seed=1)
    assert again["bayes_estimate_gmm"] == out["bayes_estimate_gmm"]
    gmm = GaussianMixtureBayes(n_components=3, random_state=1).fit(X[:800], y[:800])
    assert gmm.score(X[800:], y[800:]) == out["bayes_estimate_gmm"]


def test_knn_neighbors() -> None:
    assert knn_classifier(100).n_neighbors == 10
    assert knn_classifier(100).weights == "distance"
    assert knn_classifier(3, n_neighbors=10).n_neighbors == 3
    assert knn_classifier(1).n_neighbors == 1


def test_pca_plane_matches_scikit_learn() -> None:
    X, _ = _gaussians(100, [[0, 1, 2, 3]], [[3, 2, 0.5, 0.1]])
    center, components, variances = pca_plane(X)
    ref = PCA(n_components=2).fit(X)
    np.testing.assert_allclose(center, ref.mean_)
    np.testing.assert_allclose(np.abs(components), np.abs(ref.components_), atol=1e-10)
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(variances, ref.explained_variance_ * 99 / 100)
    # Deterministic sign: the largest entry of each direction is positive.
    assert np.all(components[np.arange(2), np.argmax(np.abs(components), 1)] > 0)
    with pytest.raises(ValueError):
        pca_plane(X[:, :1])


def test_projected_gaussian_is_the_marginal_of_the_samples() -> None:
    """P Sigma P^T is the covariance of the projected samples, exactly."""
    X, y = _gaussians(200, [[0, 0, 0, 0], [1, 2, 0, 1]], [[1, 2, 3, 4]] * 2)
    qda = QDA().fit(X, y)
    center, components, _ = pca_plane(X)
    means, covariances = project_gaussians(
        qda.means_, qda.covariance_, center, components
    )
    assert means.shape == (2, 2) and covariances.shape == (2, 2, 2)
    for k, label in enumerate(qda.classes_):
        projected = (X[y == label] - center) @ components.T
        np.testing.assert_allclose(means[k], projected.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(
            covariances[k], np.cov(projected.T, bias=True), atol=1e-12
        )
