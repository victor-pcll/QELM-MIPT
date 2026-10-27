"""Estimators of the Bayes accuracy beyond the QDA: mixtures, SVM (issue #200)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.bayes_optimal import (
    QDA,
    GaussianMixtureBayes,
    benchmark_bayes_estimators,
    evaluate_bayes_optimal_limit,
    svm_rbf_classifier,
)


def _blobs(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Class 0: two blobs at (+-2, 0); class 1: one blob at (0, 0).

    The two classes have the same mean and almost the same covariance, so a
    single Gaussian per class (QDA) barely separates them, while two
    Gaussians for class 0 do.
    """
    rng = np.random.default_rng(seed)
    side = rng.choice([-2.0, 2.0], size=n)
    X0 = np.column_stack([side, np.zeros(n)]) + 0.3 * rng.normal(size=(n, 2))
    X1 = np.column_stack([rng.normal(scale=0.3, size=n), rng.normal(scale=2.0, size=n)])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    order = rng.permutation(2 * n)
    return X[order], y[order]


def test_one_component_is_the_qda() -> None:
    """K = 1 without regularization gives the posterior of the QDA."""
    X, y = _blobs(300)
    X_test, _ = _blobs(100, seed=1)
    gmm = GaussianMixtureBayes(n_components=1, reg_covar=0.0, random_state=0)
    qda = QDA().fit(X, y)
    gmm.fit(X, y)
    np.testing.assert_allclose(gmm.priors_, qda.priors_)
    np.testing.assert_allclose(gmm.mixtures_[1].means_[0], qda.means_[1])
    np.testing.assert_allclose(
        gmm.predict_proba(X_test), qda.predict_proba(X_test), atol=1e-8
    )
    np.testing.assert_array_equal(gmm.predict(X_test), qda.predict(X_test))


def test_mixture_beats_the_qda_on_multimodal_classes() -> None:
    """Two blobs in one class: the mixture finds them, the QDA cannot."""
    X, y = _blobs(400)
    X_test, y_test = _blobs(400, seed=1)
    qda = QDA().fit(X, y).score(X_test, y_test)
    gmm = GaussianMixtureBayes(n_components=2, random_state=0).fit(X, y)
    assert gmm.score(X_test, y_test) > 0.97
    assert gmm.score(X_test, y_test) > qda + 0.05
    proba = gmm.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_mixture_is_reproducible_with_the_seed() -> None:
    X, y = _blobs(200)
    a = GaussianMixtureBayes(n_components=3, random_state=7).fit(X, y)
    b = GaussianMixtureBayes(n_components=3, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


def test_mixture_rejects_bad_inputs() -> None:
    X, y = _blobs(20)
    with pytest.raises(ValueError, match="n_components"):
        GaussianMixtureBayes(n_components=0)
    with pytest.raises(ValueError, match="fewer than n_components"):
        GaussianMixtureBayes(n_components=50).fit(X, y)
    with pytest.raises(ValueError, match="two classes"):
        GaussianMixtureBayes(n_components=1).fit(X, np.zeros(len(y)))


def test_svm_separates_the_blobs() -> None:
    X, y = _blobs(400)
    X_test, y_test = _blobs(400, seed=1)
    assert svm_rbf_classifier().fit(X, y).score(X_test, y_test) > 0.97


def test_benchmark_uses_one_split_for_every_estimator() -> None:
    """Same QDA and K-NN numbers as evaluate_bayes_optimal_limit, same keys."""
    X, y = _blobs(300)
    X_test, y_test = _blobs(150, seed=1)
    out = benchmark_bayes_estimators(X, y, X_test, y_test, gmm_components=(2, 3))
    names = ("qda", "gmm2", "gmm3", "svm", "knn")
    assert set(out) == {f"{k}_{n}" for n in names for k in ("accuracy", "time")}
    ref = evaluate_bayes_optimal_limit(X, y, X_test, y_test)
    assert out["accuracy_qda"] == ref["bayes_estimate_qda"]
    assert out["accuracy_knn"] == ref["bayes_estimate_knn"]
    for name in names:
        assert 0.0 <= out[f"accuracy_{name}"] <= 1.0
        assert out[f"time_{name}"] >= 0.0
    again = benchmark_bayes_estimators(X, y, X_test, y_test, gmm_components=(2, 3))
    for name in names:
        assert again[f"accuracy_{name}"] == out[f"accuracy_{name}"]


def test_benchmark_gives_nan_when_a_sector_is_too_small() -> None:
    """3 samples per sector in 4 dimensions: no QDA, no mixture of 5."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    y = np.array([0.0, 1.0] * 3)
    out = benchmark_bayes_estimators(X, y, X, y, gmm_components=(2, 5))
    assert np.isnan(out["accuracy_qda"]) and np.isnan(out["time_qda"])
    assert np.isnan(out["accuracy_gmm5"])
    assert np.isfinite(out["accuracy_gmm2"])
    assert np.isfinite(out["accuracy_svm"]) and np.isfinite(out["accuracy_knn"])
