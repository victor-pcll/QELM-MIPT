"""Drawing the estimators of the Bayes accuracy on the PCA plane (issue #202)."""

from __future__ import annotations

import matplotlib
import numpy as np

from QELM.evaluation.bayes_optimal import QDA, GaussianMixtureBayes
from QELM.utils.plots import (
    draw_decision_regions,
    draw_gmm_contours,
    draw_qda_contours,
    ellipse,
    figure_data,
    gaussian_density,
    subplots,
)

matplotlib.use("Agg")


def _data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2 * n, 4)) * [2.0, 1.0, 0.3, 0.1]
    X[n:, 0] += 3.0
    return X, np.repeat([0.0, 1.0], n)


def test_ellipse_is_the_mahalanobis_level_set() -> None:
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    curve = ellipse([1.0, -1.0], sigma, 2.0)
    diff = curve - [1.0, -1.0]
    r2 = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(sigma), diff)
    np.testing.assert_allclose(r2, 4.0)


def test_gaussian_density_is_normalized() -> None:
    sigma = np.array([[1.0, 0.3], [0.3, 0.5]])
    u = np.linspace(-8, 8, 401)
    points = np.stack(np.meshgrid(u, u), axis=-1)
    density = gaussian_density(points, np.array([0.5, 0.0]), sigma)
    np.testing.assert_allclose(np.sum(density) * (u[1] - u[0]) ** 2, 1.0, rtol=1e-6)


def test_qda_and_gmm_contours_keep_their_data() -> None:
    X, y = _data()
    fig, (a, b) = subplots(1, 2)
    qda = QDA().fit(X, y)
    out = draw_qda_contours(a, X, y, qda, names=["Q = 0", "Q = 1"])
    assert out["means"].shape == (2, 2) and out["covariances"].shape == (2, 2, 2)
    gmm = GaussianMixtureBayes(n_components=2, random_state=0).fit(X, y)
    out = draw_gmm_contours(b, X, y, gmm, names=["Q = 0", "Q = 1"])
    for weights, means in zip(out["weights"], out["means"], strict=True):
        np.testing.assert_allclose(np.sum(weights), 1.0)
        assert means.shape == (2, 2)
    labels = {c["label"] for c in figure_data(fig)["panels"][1]["curves"]}
    assert labels == {
        "Q = 0",
        "Q = 1",
        "Q = 0, GMM components (1 sigma)",
        "Q = 1, GMM components (1 sigma)",
    }


def test_decision_regions_of_a_linear_rule() -> None:
    """A rule on the first coordinate: its mistakes are the crosses."""
    X, y = _data()

    def predict(points: np.ndarray) -> np.ndarray:
        return np.where(points[:, 0] > 1.5, 1.0, 0.0)

    fig, ax = subplots()
    out = draw_decision_regions(ax, X, y, predict, [0.0, 1.0])
    accuracy = np.mean(predict(X) == y)
    assert out["accuracy"] == accuracy
    assert 0.0 < out["in_plane"] <= 1.0
    crosses = next(
        c
        for c in figure_data(fig)["panels"][0]["curves"]
        if c["label"] == "misclassified"
    )
    assert len(crosses["x"]) == round((1 - accuracy) * len(y))
