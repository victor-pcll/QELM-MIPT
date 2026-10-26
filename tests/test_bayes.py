"""The Bayes posterior estimator of QELM.evaluation.bayes (issue #194)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.bayes import (
    correct_probability,
    posterior_scores,
    prior_entropy,
    qda_posterior,
)
from QELM.training import RidgeReadout


def _gaussians(
    mean: tuple[np.ndarray, np.ndarray],
    scale: tuple[float, float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two isotropic Gaussian classes 0 and 1, alternating, n samples."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    X = np.empty((n, mean[0].size))
    for k in (0, 1):
        X[y == k] = mean[k] + scale[k] * rng.standard_normal((n // 2, mean[k].size))
    return X, y


def test_posterior_rows_are_probabilities() -> None:
    X, y = _gaussians((np.zeros(3), np.ones(3)), (1.0, 1.0))
    classes, proba = qda_posterior(X[:1000], y[:1000], X[1000:])
    np.testing.assert_array_equal(classes, [0, 1])
    assert proba.shape == (1000, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_known_posterior_of_two_one_dimensional_gaussians() -> None:
    """Equal variances and priors: P(1 | x) = 1 / (1 + exp(-(x - 1/2) / s^2))
    for the means 0 and 1, the logistic of the exact Bayes rule."""
    X, y = _gaussians((np.zeros(1), np.ones(1)), (0.5, 0.5), n=200_000)
    x = np.linspace(-1, 2, 7)
    _, proba = qda_posterior(X[:, 0], y, x)
    exact = 1 / (1 + np.exp(-(x - 0.5) / 0.25))
    np.testing.assert_allclose(proba[:, 1], exact, atol=0.02)


def test_separated_classes_are_learned_and_identical_ones_are_not() -> None:
    far, y = _gaussians((np.zeros(2), 10 * np.ones(2)), (1.0, 1.0))
    classes, proba = qda_posterior(far[:1000], y[:1000], far[1000:])
    scores = posterior_scores(classes, proba, y[1000:])
    assert scores["accuracy"] == 1.0 and scores["balanced_accuracy"] == 1.0
    assert scores["entropy_mean"] < 1e-6 and scores["P_corr_mean"] > 1 - 1e-6

    same, y = _gaussians((np.zeros(2), np.zeros(2)), (1.0, 1.0))
    classes, proba = qda_posterior(same[:1000], y[:1000], same[1000:])
    scores = posterior_scores(classes, proba, y[1000:])
    assert scores["accuracy"] == pytest.approx(0.5, abs=0.05)
    assert scores["entropy_mean"] > 0.99


def test_qda_learns_what_a_linear_readout_cannot() -> None:
    """Same mean, different spread (scenario B of issue #194): the charge is
    in the covariance, which no hyperplane separates, but QDA reads it."""
    X, y = _gaussians((np.zeros(3), np.zeros(3)), (0.3, 1.5), n=4000)
    train = 3000
    classes, proba = qda_posterior(X[:train], y[:train], X[train:])
    bayes = posterior_scores(classes, proba, y[train:])["accuracy"]
    phi = np.hstack([X, np.ones((X.shape[0], 1))])
    ridge = RidgeReadout({"alpha": 1e-3})
    ridge.fit(phi[:train], y[:train].astype(float))
    linear = np.mean((ridge.predict(phi[train:]) >= 0.5) == (y[train:] == 1))
    assert linear == pytest.approx(0.5, abs=0.05)
    assert bayes > 0.85


def test_degenerate_class_needs_the_low_tolerance() -> None:
    """A class on a hyperplane (sum of features fixed, as for a conserved
    charge) with small spread: the regularized covariance is above tol."""
    rng = np.random.default_rng(1)
    n = 400
    y = np.arange(n) % 2
    X = 1e-3 * rng.standard_normal((n, 3))
    X -= X.mean(axis=1, keepdims=True)  # each row sums to 0 ...
    X[y == 1] += 0.1  # ... or to 0.3
    classes, proba = qda_posterior(X[:300], y[:300], X[300:], reg_param=1e-6)
    assert posterior_scores(classes, proba, y[300:])["accuracy"] == 1.0
    with pytest.raises(np.linalg.LinAlgError):
        qda_posterior(X[:300], y[:300], X[300:], reg_param=1e-6, tol=1e-4)


def test_correct_probability_reads_the_true_label() -> None:
    classes = np.array([1.0, 2.0])
    proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]])
    np.testing.assert_allclose(
        correct_probability(classes, proba, np.array([1.0, 1.0, 2.0])),
        [0.9, 0.3, 0.5],
    )
    with pytest.raises(ValueError, match="one of classes"):
        correct_probability(classes, proba, np.array([1.0, 3.0, 2.0]))


def test_posterior_scores_on_a_known_case() -> None:
    classes = np.array([0, 1])
    proba = np.array([[0.5, 0.5], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    y = np.array([0, 0, 1, 1])
    scores = posterior_scores(classes, proba, y)
    # argmax picks 0 on the tie: right, right, right, wrong.
    assert scores["accuracy"] == 0.75
    assert scores["balanced_accuracy"] == 0.75
    assert scores["P_corr_mean"] == pytest.approx((0.5 + 1.0 + 0.8 + 0.2) / 4)
    h08 = -(0.8 * np.log2(0.8) + 0.2 * np.log2(0.2))
    assert scores["entropy_mean"] == pytest.approx((1.0 + 0.0 + 2 * h08) / 4)


def test_prior_entropy() -> None:
    assert prior_entropy(np.array([1, 2, 1, 2])) == pytest.approx(1.0)
    assert prior_entropy(np.array([1, 1, 1])) == 0.0
    assert prior_entropy(np.array([0, 0, 0, 1]), base=np.e) == pytest.approx(
        -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    )
    with pytest.raises(ValueError):
        prior_entropy(np.array([0, 1, 2]))


def test_input_checks() -> None:
    X = np.zeros((4, 2))
    with pytest.raises(ValueError, match="two classes"):
        qda_posterior(X, np.zeros(4), X)
    with pytest.raises(ValueError, match="y_train"):
        qda_posterior(X, np.array([0, 1, 0]), X)
    with pytest.raises(ValueError, match="features"):
        qda_posterior(X, np.array([0, 1, 0, 1]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="reg_param"):
        qda_posterior(X, np.array([0, 1, 0, 1]), X, reg_param=2.0)
