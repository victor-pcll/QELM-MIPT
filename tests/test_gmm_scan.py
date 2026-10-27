"""Scan of the mixture ceiling in K and BIC selection (issue #215)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.bayes_optimal import (
    GaussianMixtureBayes,
    evaluate_bayes_optimal_limit,
)
from QELM.evaluation.gmm_scan import (
    SCAN_COMPONENTS,
    BICGaussianMixtureBayes,
    bic_curve,
    gmm_component_scan,
)


def _blobs(centers: list[list[float]], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples around each center, unit-free spread 0.1."""
    return np.vstack([rng.normal(c, 0.1, size=(n, len(c))) for c in centers])


def _two_sectors(
    n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alternating bundles on a line: sector 0 at x = -3, 1, sector 1 at -1, 3.

    Both sectors have the same covariance (variance 4 along x), so one
    Gaussian per sector (the QDA) reduces to a cut at x = 0, right on half
    of the samples only, as the bundles of the measured trajectories (#208).
    """
    rng = np.random.default_rng(seed)

    def draw() -> tuple[np.ndarray, np.ndarray]:
        a = _blobs([[-3.0, 0.0], [1.0, 0.0]], n, rng)
        b = _blobs([[-1.0, 0.0], [3.0, 0.0]], n, rng)
        return np.vstack([a, b]), np.r_[np.zeros(len(a)), np.ones(len(b))]

    X_tr, y_tr = draw()
    X_te, y_te = draw()
    return X_tr, y_tr, X_te, y_te


@pytest.mark.parametrize("n_blobs", [1, 2, 3])
def test_bic_finds_the_number_of_blobs(n_blobs: int) -> None:
    rng = np.random.default_rng(1)
    centers = [[4.0 * k, (-1.0) ** k] for k in range(n_blobs)]
    bic, mixtures = bic_curve(_blobs(centers, 80, rng), k_max=6, random_state=0)
    assert bic.shape == (6,)
    assert len(mixtures) == 6
    assert int(np.argmin(bic)) + 1 == n_blobs


def test_bic_curve_skips_k_above_n_and_checks_inputs() -> None:
    X = np.random.default_rng(0).normal(size=(3, 2))
    bic, mixtures = bic_curve(X, k_max=5, random_state=0)
    assert len(mixtures) == 3
    assert np.all(np.isfinite(bic[:3]))
    assert np.all(np.isinf(bic[3:]))
    with pytest.raises(ValueError, match="k_max"):
        bic_curve(X, k_max=0)
    with pytest.raises(ValueError, match="shape"):
        bic_curve(np.zeros(4))


def test_bic_rule_selects_per_sector_and_classifies() -> None:
    X_tr, y_tr, X_te, y_te = _two_sectors(100)
    model = BICGaussianMixtureBayes(k_max=5, random_state=0).fit(X_tr, y_tr)
    np.testing.assert_array_equal(model.n_components_, [2, 2])
    assert model.bic_.shape == (2, 5)
    assert model.score(X_te, y_te) > 0.99
    # One Gaussian per sector (the QDA) cannot do it.
    qda_like = GaussianMixtureBayes(n_components=1, random_state=0).fit(X_tr, y_tr)
    assert qda_like.score(X_te, y_te) < 0.8


def test_bic_rule_with_k_max_1_is_the_one_gaussian_rule() -> None:
    X_tr, y_tr, X_te, _ = _two_sectors(40, seed=3)
    bic = BICGaussianMixtureBayes(k_max=1, random_state=0).fit(X_tr, y_tr)
    one = GaussianMixtureBayes(n_components=1, random_state=0).fit(X_tr, y_tr)
    np.testing.assert_allclose(bic.decision_function(X_te), one.decision_function(X_te))


def test_bic_rule_needs_two_classes() -> None:
    with pytest.raises(ValueError, match="two classes"):
        BICGaussianMixtureBayes().fit(np.zeros((5, 2)), np.zeros(5))


def test_scan_matches_the_logged_ceilings_and_is_reproducible() -> None:
    X_tr, y_tr, X_te, y_te = _two_sectors(30, seed=5)
    scan = gmm_component_scan(X_tr, y_tr, X_te, y_te, seed=7)
    logged = evaluate_bayes_optimal_limit(X_tr, y_tr, X_te, y_te, seed=7)
    assert scan["accuracy_gmm_K3"] == logged["bayes_estimate_gmm"]
    assert scan["accuracy_knn"] == logged["bayes_estimate_knn"]
    for K in SCAN_COMPONENTS:
        assert 0.0 <= scan[f"accuracy_gmm_K{K}"] <= 1.0
    assert scan["k_bic_0"] == 2.0
    assert scan["k_bic_1"] == 2.0
    assert scan["k_bic_mean"] == 2.0
    assert scan["accuracy_gmm_K1"] < 0.8 < scan["accuracy_gmm_bic"]
    assert scan == gmm_component_scan(X_tr, y_tr, X_te, y_te, seed=7)


def test_scan_gives_nan_when_a_sector_is_smaller_than_k() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    y = np.r_[np.zeros(9), np.ones(3)]
    scan = gmm_component_scan(X, y, X, y, components=(1, 3, 4))
    assert np.isfinite(scan["accuracy_gmm_K3"])
    assert np.isnan(scan["accuracy_gmm_K4"])
    assert 1.0 <= scan["k_bic_1"] <= 3.0
