"""Plotting helpers shared by the experiment scripts and the notebooks.

Kept out of ``QELM.utils`` on purpose: Matplotlib is a dev dependency, so
``import QELM`` and ``main.py`` never load it. Import from here:

    from QELM.utils.plots import save_figure, subplots, sweep_curve

- :mod:`~QELM.utils.plots.style`: the project style (no grid, closed box,
  large titles) and its colors;
- :mod:`~QELM.utils.plots.sweep`: curves of a sweep, reference overlays and
  their comparison;
- :mod:`~QELM.utils.plots.io`: :func:`save_figure`, the figure and its data
  in the output folder of the run.
"""

from __future__ import annotations

from QELM.utils.plots.io import figure_data, save_figure
from QELM.utils.plots.style import (
    COLORS,
    STYLE,
    apply_style,
    color_map,
    style_axes,
    subplots,
)
from QELM.utils.plots.sweep import (
    SweepCurve,
    compare_same_size,
    draw_curve,
    draw_references,
    print_comparison,
    sizes,
    sweep_curve,
)

__all__ = [
    "COLORS",
    "STYLE",
    "SweepCurve",
    "apply_style",
    "color_map",
    "compare_same_size",
    "draw_curve",
    "draw_references",
    "figure_data",
    "print_comparison",
    "save_figure",
    "sizes",
    "style_axes",
    "subplots",
    "sweep_curve",
]
