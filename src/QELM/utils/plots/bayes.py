"""Draw the Gaussians fitted by the QDA on the plane of the PCA (issue #202).

The features ``<Z_k>`` of the two charge sectors live in ``Ns`` dimensions,
too many to draw. :func:`draw_qda_contours` projects the samples onto
their plane of largest variance (:func:`~QELM.evaluation.bayes_optimal.pca_plane`)
and draws, for each sector, the samples and the ellipses
``(x - mu_k)^T Sigma_k^{-1} (x - mu_k) = r**2`` of its projected Gaussian
(:func:`~QELM.evaluation.bayes_optimal.project_gaussians`), the level sets
of the density ``f_k`` fitted by the QDA. Two clusters of ellipses apart
mean the sectors are still told apart; ellipses on top of each other mean
the information is gone from these features.

Both the samples and the ellipses are drawn as labelled lines, so that
:func:`~QELM.utils.plots.save_figure` writes their coordinates to the JSON
of the figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from QELM.evaluation.bayes_optimal import QDA, pca_plane, project_gaussians
from QELM.utils.plots.style import COLORS

# Points along an ellipse.
ELLIPSE_POINTS = 100


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
        ``variances`` (of :func:`pca_plane`), ``means`` and ``covariances``
        (of :func:`project_gaussians`), for the JSON of the figure.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    center, components, variances = pca_plane(X_arr)
    means, covariances = project_gaussians(
        qda.means_, qda.covariance_, center, components
    )
    projected = (X_arr - center) @ components.T
    if names is None:
        names = [f"class {label:g}" for label in qda.classes_]
    for k, (label, name) in enumerate(zip(qda.classes_, names, strict=True)):
        color = colors[k % len(colors)]
        points = projected[y_arr == label]
        ax.plot(
            points[:, 0],
            points[:, 1],
            ls="none",
            marker=".",
            ms=4,
            alpha=0.5,
            color=color,
            label=name,
        )
        for r in radii:
            curve = ellipse(means[k], covariances[k], r)
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=color,
                lw=1.5,
                ls="-" if r == radii[0] else "--",
                label=f"{name}, QDA {r:g} sigma",
            )
    # Share of the total variance of X along each axis.
    explained = variances / (float(np.sum(np.var(X_arr, axis=0))) or np.nan)
    ax.set(
        xlabel=f"PC 1 ({100 * explained[0]:.0f} %)",
        ylabel=f"PC 2 ({100 * explained[1]:.0f} %)",
    )
    return {
        "center": center,
        "components": components,
        "variances": variances,
        "means": means,
        "covariances": covariances,
    }
