"""One plot style for every figure of the project (issue #147).

No grid, a box closed on all four sides, and large titles and labels, so
that a figure stays readable when it is shrunk on a slide or in a report.
:func:`subplots` applies the style before creating the figure;
:func:`style_axes` enforces it on axes created otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Matplotlib settings of the style, applied to ``plt.rcParams``.
STYLE: dict[str, Any] = {
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.linewidth": 1.2,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "savefig.dpi": 150,
}

# Categorical colors in a fixed order: our curves first, then the
# reference curves, so that a curve keeps its color in every panel.
COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#6250d6",
    "#d63a8a",
)


def apply_style() -> None:
    """Set :data:`STYLE` in ``plt.rcParams`` for the figures created next."""
    # rcParams is typed with the literal names of its keys: STYLE is a plain
    # dict of some of them.
    plt.rcParams.update(cast(Any, STYLE))


def style_axes(axes: Axes | Iterable[Axes] | np.ndarray) -> None:
    """Turn the grid off and draw the four spines of every axis given.

    Args:
        axes: One axis, or any nesting of axes (the array of
            ``plt.subplots``).
    """
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(STYLE["axes.linewidth"])
        ax.tick_params(direction="in", top=True, right=True)


def subplots(*args: Any, **kwargs: Any) -> tuple[Figure, Any]:
    """``plt.subplots`` in the project style, with ``constrained_layout``.

    Takes the arguments of ``plt.subplots`` and returns the same
    ``(figure, axes)``.
    """
    apply_style()
    kwargs.setdefault("constrained_layout", True)
    fig, axes = plt.subplots(*args, **kwargs)
    style_axes(axes)
    return fig, axes


def color_map(keys: Iterable[str]) -> dict[str, str]:
    """Give each key a color of :data:`COLORS`, in order, cycling if needed."""
    return {key: COLORS[k % len(COLORS)] for k, key in enumerate(keys)}
