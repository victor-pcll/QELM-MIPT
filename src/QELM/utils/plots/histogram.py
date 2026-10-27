"""Superimposed histograms of one feature for several datasets (issue #191).

Each dataset, e.g. the values of ``<Z_k>`` over ``n`` samples for several
``n``, is drawn on the same axis as a translucent density histogram with
common bin edges, so the bars of two datasets are comparable bin by bin, and
a Gaussian kernel density estimate (KDE) on top. :func:`save_figure` reads
the bars back as ``histograms`` and the KDE lines as ``curves``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from scipy import stats

# Half-width of the bins drawn around a constant sample, whose range is 0.
POINT_HALF_WIDTH = 0.05


def common_edges(samples: Sequence[ArrayLike], bins: int = 40) -> np.ndarray:
    """Return ``bins + 1`` edges spanning every sample.

    Args:
        samples: The datasets, each of shape ``(n,)``.
        bins: Number of bins.

    Returns:
        Increasing edges from the smallest to the largest value of all the
        samples. If every value is the same ``v`` (a feature fixed by a
        symmetry), the edges span ``v -+ POINT_HALF_WIDTH``.

    Raises:
        ValueError: If there is no value or ``bins < 1``.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in samples])
    if values.size == 0:
        raise ValueError("No value to bin.")
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        lo, hi = lo - POINT_HALF_WIDTH, hi + POINT_HALF_WIDTH
    return np.linspace(lo, hi, bins + 1)


def draw_histograms(
    ax: Axes,
    samples: Mapping[str, ArrayLike],
    colors: Mapping[str, str],
    *,
    edges: ArrayLike | None = None,
    bins: int = 40,
    kde: bool = True,
    alpha: float = 0.3,
) -> list[Line2D]:
    """Draw one density histogram per sample on ``ax``, with its KDE.

    Args:
        ax: Destination axis.
        samples: ``{label: values}``, drawn in this order.
        colors: Color of each label.
        edges: Common bin edges; ``None`` uses :func:`common_edges`.
        bins: Number of bins when ``edges`` is ``None``.
        kde: Whether to draw a Gaussian KDE line over each histogram. A
            constant sample has no KDE (its bandwidth would be 0).
        alpha: Opacity of the bars.

    Returns:
        One legend handle per sample: its KDE line, or a plain line of its
        color when there is no KDE. Pass them to ``ax.legend(handles=...)``.
    """
    if edges is None:
        edges = common_edges(list(samples.values()), bins)
    edges = np.asarray(edges, dtype=float)
    grid = np.linspace(edges[0], edges[-1], 400)
    handles = []
    for label, values in samples.items():
        v = np.asarray(values, dtype=float).ravel()
        color = colors[label]
        _, _, bars = ax.hist(
            v, bins=list(edges), density=True, alpha=alpha, color=color
        )
        # One dataset and the default histtype "bar" give one BarContainer;
        # label it there, where save_figure reads it.
        assert isinstance(bars, BarContainer)
        bars.set_label(f"{label} (histogram)")
        if kde and v.size >= 2 and np.std(v) > 0:
            (line,) = ax.plot(
                grid, stats.gaussian_kde(v)(grid), color=color, lw=2, label=label
            )
        else:
            line = Line2D([], [], color=color, lw=2, label=label)
        handles.append(line)
    return handles


__all__ = ["POINT_HALF_WIDTH", "common_edges", "draw_histograms"]
