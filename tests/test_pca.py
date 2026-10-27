"""PCA fitted on a training set (QELM.evaluation.pca, issue #225)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.bayes_optimal import pca_plane
from QELM.evaluation.pca import fit_pca


def test_matches_pca_plane_on_its_own_samples() -> None:
    """Fitted and applied to the same samples, it is pca_plane."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * np.array([3.0, 2.0, 1.0, 0.5, 0.1])
    pca = fit_pca(X, n_components=2)
    center, components, variances = pca_plane(X)
    np.testing.assert_allclose(pca.center, center)
    np.testing.assert_allclose(pca.components, components)
    np.testing.assert_allclose(pca.variances, variances)
    np.testing.assert_allclose(pca.transform(X), (X - center) @ components.T)
    assert pca.total_variance == pytest.approx(float(np.sum(np.var(X, axis=0))))
    assert pca.rank == 2 and not pca.degenerate


def test_axes_come_from_the_training_set_only() -> None:
    """The test samples are projected on the axes of the training samples."""
    X_train = np.array([[-1.0, 0.0], [1.0, 0.0], [-2.0, 0.1], [2.0, -0.1]])
    X_test = np.array([[0.0, 5.0], [3.0, 0.0]])
    pca = fit_pca(X_train, n_components=1)
    # First axis along e_x, centered on the training mean (0, 0).
    np.testing.assert_allclose(np.abs(pca.components[0]), [1.0, 0.0], atol=0.05)
    projected = pca.transform(X_test)
    assert projected.shape == (2, 1)
    # The large spread of X_test along e_y is invisible on the training axis.
    assert abs(projected[0, 0]) < 0.5
    assert projected[1, 0] == pytest.approx(3.0, abs=0.05)


def test_explained_shares_sum_to_one_with_every_component() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    pca = fit_pca(X, n_components=3)
    assert np.sum(pca.explained) == pytest.approx(1.0)
    assert np.all(np.diff(pca.variances) <= 0)
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(3), atol=1e-12)


def test_zero_variance_projects_everything_to_the_origin() -> None:
    """The record M at p = 0 is all zeros: no axis is drawn (issue #225)."""
    X = np.zeros((40, 28))
    pca = fit_pca(X, n_components=2)
    assert pca.degenerate and pca.rank == 0
    assert pca.total_variance == 0.0
    np.testing.assert_array_equal(pca.components, 0.0)
    np.testing.assert_array_equal(pca.explained, 0.0)
    np.testing.assert_array_equal(pca.transform(np.ones((3, 28))), 0.0)


def test_flat_directions_beyond_the_rank_are_zero() -> None:
    """Samples on a line: the second axis has no variance and is dropped."""
    t = np.linspace(-1.0, 1.0, 21)
    X = np.stack([t, 2.0 * t, np.full_like(t, 0.3)], axis=1)
    pca = fit_pca(X, n_components=3)
    assert pca.rank == 1 and not pca.degenerate
    np.testing.assert_array_equal(pca.components[1:], 0.0)
    np.testing.assert_allclose(pca.explained, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.abs(pca.components[0]), [1.0, 2.0, 0.0] / np.sqrt(5.0), atol=1e-12
    )


def test_more_components_than_samples() -> None:
    X = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
    pca = fit_pca(X, n_components=3)
    assert pca.components.shape == (3, 3)
    assert pca.rank == 1


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="X_train"):
        fit_pca(np.zeros(5))
    with pytest.raises(ValueError, match="X_train"):
        fit_pca(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(np.zeros((4, 3)), n_components=0)
    with pytest.raises(ValueError, match="shape"):
        fit_pca(np.ones((4, 3)) * np.arange(4)[:, None]).transform(np.zeros((2, 2)))
