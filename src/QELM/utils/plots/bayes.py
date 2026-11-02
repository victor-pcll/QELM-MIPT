"""The estimators of the Bayes accuracy on the plane of the PCA (issue #202).

The features ``<Z_k>`` of the two charge sectors live in ``N`` dimensions,
too many to draw. Every function here projects the samples onto their plane
of largest variance (:func:`~QELM.evaluation.bayes_optimal.pca_plane`) and
draws them colored by sector, with what one estimator makes of them:

- :func:`draw_qda_contours`: the ellipses
  ``(x - mu_k)^T Sigma_k^{-1} (x - mu_k) = r**2`` of the Gaussian fitted by
  the QDA to each sector, projected exactly
  (:func:`~QELM.evaluation.bayes_optimal.project_gaussians`): the level sets
  of its density ``f_k``. Two clusters of ellipses apart mean the sectors
  are still told apart; ellipses on top of each other mean the information
  is gone from these features.
- :func:`draw_gmm_contours`: the same for a mixture of Gaussians per sector
  (:class:`~QELM.evaluation.bayes_optimal.GaussianMixtureBayes`): the
  contour lines of the projected mixture density, which can wrap several
  disjoint clusters of one sector, and the 1 sigma ellipse of each of its
  components.
- :func:`draw_decision_regions`: the class that a classifier (the ridge
  readout, the K-NN, ...) predicts on the PCA plane, as two shaded regions
  and their boundary, with the test samples it gets wrong marked. The
  classifier works in all ``N`` dimensions: the regions are its slice by
  the plane through the mean of the samples, so a sample far from the plane
  can be predicted otherwise than the region it projects to; the crosses
  mark the true mistakes, from all ``N`` coordinates.

The samples, the ellipses and the crosses are drawn as labelled lines, so
that :func:`~QELM.utils.plots.save_figure` writes their coordinates to the
JSON of the figure; the shaded regions and the mixture contour lines are
not, and the functions return what is needed to redraw them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike

from QELM.evaluation.bayes_optimal import (
    QDA,
    GaussianMixtureBayes,
    pca_plane,
    project_gaussians,
)
from QELM.utils.plots.style import COLORS

# Points along an ellipse.
ELLIPSE_POINTS = 100
# Points per side of the grids of the regions and the mixture densities.
GRID_POINTS = 200
# Margin around the projected samples, as a fraction of their range.
MARGIN = 0.1
# Contour lines of a mixture density, as fractions of its maximum.
DENSITY_LEVELS = (0.05, 0.25, 0.6)


def ellipse(mean: ArrayLike, covariance: ArrayLike, radius: float) -> np.ndarray:
    """Return the ellipse ``(x - mean)^T covariance^{-1} (x - mean) = radius**2``.

    Args:
        mean: Center, shape ``(2,)``.
        covariance: Positive definite, shape ``(2, 2)``.
        radius: Mahalanobis radius, ``1`` for one standard deviation.

    Returns:
        Closed curve, shape ``(ELLIPSE_POINTS, 2)``.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_POINTS)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    # covariance = C C^T maps the unit circle onto the radius-1 ellipse.
    C = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    return np.asarray(np.asarray(mean) + radius * (C @ circle).T)


def gaussian_density(
    points: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Return the density of ``N(mean, covariance)`` at ``points``, ``(..., m)``."""
    diff = points - mean
    inverse = np.linalg.inv(covariance)
    mahalanobis2 = np.einsum("...i,ij,...j->...", diff, inverse, diff)
    norm = np.sqrt((2.0 * np.pi) ** len(mean) * np.linalg.det(covariance))
    return np.asarray(np.exp(-0.5 * mahalanobis2) / norm)


def _project(X: ArrayLike) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``X`` on its PCA plane and the plane (:func:`pca_plane`)."""
    X_arr = np.asarray(X, dtype=np.float64)
    center, components, variances = pca_plane(X_arr)
    plane = {"center": center, "components": components, "variances": variances}
    # Share of the total variance of X along each axis.
    plane["explained"] = variances / (float(np.sum(np.var(X_arr, axis=0))) or np.nan)
    return (X_arr - center) @ components.T, plane


def _draw_samples(
    ax: Axes,
    projected: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    names: Sequence[str],
    colors: Sequence[str],
) -> None:
    """Draw the projected samples, one color per class."""
    for k, (label, name) in enumerate(zip(classes, names, strict=True)):
        points = projected[y == label]
        ax.plot(
            points[:, 0],
            points[:, 1],
            ls="none",
            marker=".",
            ms=4,
            alpha=0.5,
            color=colors[k % len(colors)],
            label=name,
        )


def _label_axes(ax: Axes, plane: dict[str, np.ndarray]) -> None:
    explained = plane["explained"]
    ax.set(
        xlabel=f"PC 1 ({100 * explained[0]:.0f} %)",
        ylabel=f"PC 2 ({100 * explained[1]:.0f} %)",
    )


def _grid(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a ``GRID_POINTS`` square grid over the projected samples.

    Returns:
        The two axes, shape ``(GRID_POINTS,)`` each, and the grid points,
        shape ``(GRID_POINTS, GRID_POINTS, 2)``.
    """
    low, high = projected.min(axis=0), projected.max(axis=0)
    pad = MARGIN * np.where(high > low, high - low, 1.0)
    u = np.linspace(low[0] - pad[0], high[0] + pad[0], GRID_POINTS)
    v = np.linspace(low[1] - pad[1], high[1] + pad[1], GRID_POINTS)
    return u, v, np.stack(np.meshgrid(u, v), axis=-1)


def _names(classes: np.ndarray, names: Sequence[str] | None) -> Sequence[str]:
    return [f"class {c:g}" for c in classes] if names is None else names


def draw_qda_contours(
    ax: Axes,
    X: ArrayLike,
    y: ArrayLike,
    qda: QDA,
    names: Sequence[str] | None = None,
    radii: Sequence[float] = (1.0, 2.0),
    colors: Sequence[str] = COLORS,
) -> dict[str, Any]:
    """Draw samples and QDA ellipses on the PCA plane of ``X``.

    Args:
        ax: Axis to draw on.
        X: Samples, shape ``(n, d)`` with ``d >= 2``, usually the test set;
            the plane is the PCA of these samples.
        y: Their classes, shape ``(n,)``, among ``qda.classes_``.
        qda: A fitted :class:`~QELM.evaluation.bayes_optimal.QDA` on the
            same ``d`` features.
        names: Legend name of each class of ``qda.classes_``, default
            ``"class <label>"``.
        radii: Mahalanobis radii of the ellipses drawn per class.
        colors: Color of each class, in the order of ``qda.classes_``.

    Returns:
        The plane and the projected Gaussians: ``center``, ``components``,
        ``variances``, ``explained`` (of :func:`pca_plane`), ``means`` and
        ``covariances`` (of :func:`project_gaussians`), for the JSON of the
        figure.
    """
    projected, plane = _project(X)
    names = _names(qda.classes_, names)
    _draw_samples(ax, projected, np.asarray(y), qda.classes_, names, colors)
    means, covariances = project_gaussians(
        qda.means_, qda.covariance_, plane["center"], plane["components"]
    )
    for k, name in enumerate(names):
        for r in radii:
            curve = ellipse(means[k], covariances[k], r)
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=colors[k % len(colors)],
                lw=1.5,
                ls="-" if r == radii[0] else "--",
                label=f"{name}, QDA {r:g} sigma",
            )
    _label_axes(ax, plane)
    return {**plane, "means": means, "covariances": covariances}


def draw_gmm_contours(
    ax: Axes,
    X: ArrayLike,
    y: ArrayLike,
    gmm: GaussianMixtureBayes,
    names: Sequence[str] | None = None,
    colors: Sequence[str] = COLORS,
    levels: Sequence[float] = DENSITY_LEVELS,
) -> dict[str, Any]:
    """Draw samples and the mixture density of each class on the PCA plane.

    The projection of a mixture ``sum_m w_m N(mu_m, Sigma_m)`` is the
    mixture of the projected components,
    ``sum_m w_m N(P (mu_m - c), P Sigma_m P^T)``: the exact 2D marginal of
    ``f_k``. Its contour lines are drawn at ``levels`` times its maximum
    on the grid, and the 1 sigma ellipse of each component dotted.

    Args:
        ax: Axis to draw on.
        X: Samples, shape ``(n, d)`` with ``d >= 2``; the plane is their PCA.
        y: Their classes, shape ``(n,)``, among ``gmm.classes_``.
        gmm: A fitted
            :class:`~QELM.evaluation.bayes_optimal.GaussianMixtureBayes`.
        names: Legend name of each class, default ``"class <label>"``.
        colors: Color of each class, in the order of ``gmm.classes_``.
        levels: Contour levels, as fractions of the maximum density.

    Returns:
        The plane (as :func:`draw_qda_contours`) and, per class, the
        ``weights``, ``means`` and ``covariances`` of the projected
        components.
    """
    projected, plane = _project(X)
    names = _names(gmm.classes_, names)
    _draw_samples(ax, projected, np.asarray(y), gmm.classes_, names, colors)
    u, v, points = _grid(projected)
    out: dict[str, Any] = {**plane, "weights": [], "means": [], "covariances": []}
    for k, (mixture, name) in enumerate(zip(gmm.mixtures_, names, strict=True)):
        color = colors[k % len(colors)]
        means, covariances = project_gaussians(
            mixture.means_, mixture.covariances_, plane["center"], plane["components"]
        )
        density = sum(
            w * gaussian_density(points, mu, sigma)
            for w, mu, sigma in zip(mixture.weights_, means, covariances, strict=True)
        )
        ax.contour(
            u, v, density, levels=np.array(levels) * np.max(density), colors=color
        )
        for m, (mu, sigma) in enumerate(zip(means, covariances, strict=True)):
            curve = ellipse(mu, sigma, 1.0)
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=color,
                lw=1.0,
                ls=":",
                label=f"{name}, GMM components (1 sigma)" if m == 0 else "_",
            )
        out["weights"].append(mixture.weights_)
        out["means"].append(means)
        out["covariances"].append(covariances)
    _label_axes(ax, plane)
    return out


def draw_decision_regions(
    ax: Axes,
    X: ArrayLike,
    y: ArrayLike,
    predict: Callable[[np.ndarray], np.ndarray],
    classes: ArrayLike,
    names: Sequence[str] | None = None,
    colors: Sequence[str] = COLORS,
) -> dict[str, Any]:
    """Draw what a classifier predicts on the PCA plane, and its mistakes.

    A point ``z`` of the plane is the sample ``x = c + z P`` (the mean of
    ``X`` plus a combination of the two principal directions); its region
    is the class ``predict`` gives it. The test samples are drawn on top,
    and those that ``predict`` gets wrong from all their coordinates are
    crossed.

    Args:
        ax: Axis to draw on.
        X: Samples, shape ``(n, d)`` with ``d >= 2``; the plane is their PCA.
        y: Their classes, shape ``(n,)``.
        predict: Maps samples of shape ``(m, d)`` to classes, shape ``(m,)``.
        classes: The two (or more) classes, in the order of ``colors``.
        names: Legend name of each class, default ``"class <label>"``.
        colors: Color of each class.

    Returns:
        The plane (as :func:`draw_qda_contours`), ``accuracy`` of
        ``predict`` on ``(X, y)`` and ``in_plane``, the share of the
        variance of ``X`` on the plane: the lower it is, the less the
        regions say about the samples off the plane.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    classes_arr = np.asarray(classes)
    names = _names(classes_arr, names)
    projected, plane = _project(X_arr)
    u, v, points = _grid(projected)
    lifted = plane["center"] + points.reshape(-1, 2) @ plane["components"]
    index = np.searchsorted(classes_arr, predict(lifted)).reshape(points.shape[:2])
    cmap = ListedColormap([colors[k % len(colors)] for k in range(len(classes_arr))])
    edges = np.arange(len(classes_arr) + 1) - 0.5
    ax.contourf(u, v, index, levels=edges, cmap=cmap, alpha=0.15)
    ax.contour(u, v, index, levels=edges[1:-1], colors="black", linewidths=1.0)
    _draw_samples(ax, projected, y_arr, classes_arr, names, colors)
    wrong = predict(X_arr) != y_arr
    ax.plot(
        projected[wrong, 0],
        projected[wrong, 1],
        ls="none",
        marker="x",
        ms=5,
        color="black",
        label="misclassified",
    )
    _label_axes(ax, plane)
    return {
        **plane,
        "accuracy": float(np.mean(~wrong)),
        "in_plane": float(np.sum(plane["explained"])),
    }
