"""Color maps of a metric over a two-parameter sweep (issue #164).

A sweep over two parameters, e.g. the measurement probability ``p`` and the
sample-to-parameter ratio ``alpha_ratio = n_train / n_params``, is averaged
over the seeds with :func:`QELM.utils.summarize`, then
:func:`sweep_grid` puts ``<metric>_mean`` of one size ``N`` on the
rectangular grid of the two parameters, and :func:`draw_heatmap` draws it
with ``pcolormesh``: one cell per grid point, centered on it, so a
logarithmic axis gets cells of equal width on screen (logarithmic binning).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, Normalize


@dataclass(frozen=True)
class SweepGrid:
    """One metric of one size of a sweep, on the grid of two parameters.

    Attributes:
        x: Values of the first parameter, increasing, shape ``(n_x,)``.
        y: Values of the second parameter, increasing, shape ``(n_y,)``.
        mean: Mean of the metric over the seeds, shape ``(n_y, n_x)``:
            ``mean[j, i]`` is the point ``(x[i], y[j])``, ``nan`` where the
            sweep has no run.
        n_runs: Number of runs (seeds) per point, 0 where there is none,
            shape ``(n_y, n_x)``.
    """

    x: np.ndarray
    y: np.ndarray
    mean: np.ndarray
    n_runs: np.ndarray


def sweep_grid(
    summary: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    metric: str,
    size: int | None = None,
    size_key: str = "N",
) -> SweepGrid:
    """Return ``<metric>_mean`` of a summarized sweep on its 2D grid.

    Args:
        summary: Output of :func:`QELM.utils.summarize`, grouped by
            ``(size_key, x_key, y_key)``, with ``metric`` among its metrics.
        x_key: Row key of the parameter of the x axis.
        y_key: Row key of the parameter of the y axis.
        metric: Metric name, ``<metric>_mean`` being read.
        size: Value of ``size_key`` to keep; ``None`` keeps every row.
        size_key: Row key of the size.

    Returns:
        The :class:`SweepGrid`, on every value of ``x_key`` and ``y_key``
        found in the kept rows.

    Raises:
        ValueError: If no row is kept, or two rows fall on the same point
            (the summary was not grouped by both parameters).
    """
    rows = [r for r in summary if size is None or int(r[size_key]) == size]
    if not rows:
        raise ValueError(f"No row with {size_key} = {size}.")
    x = np.array(sorted({float(r[x_key]) for r in rows}))
    y = np.array(sorted({float(r[y_key]) for r in rows}))
    mean = np.full((y.size, x.size), np.nan)
    n_runs = np.zeros((y.size, x.size), dtype=int)
    for r in rows:
        i = int(np.searchsorted(x, float(r[x_key])))
        j = int(np.searchsorted(y, float(r[y_key])))
        if n_runs[j, i]:
            raise ValueError(
                f"Two rows at {x_key} = {x[i]:g}, {y_key} = {y[j]:g}: group the "
                f"summary by ({size_key}, {x_key}, {y_key})."
            )
        mean[j, i] = float(r[f"{metric}_mean"])
        n_runs[j, i] = int(r["n_runs"])
    return SweepGrid(x=x, y=y, mean=mean, n_runs=n_runs)


def cell_edges(centers: Sequence[float] | np.ndarray, log: bool = False) -> np.ndarray:
    """Return the edges of cells centered on increasing values.

    The inner edges are the midpoints of neighbouring values, the outer ones
    are mirrored, so each value sits in the middle of its cell. With
    ``log=True`` the midpoints are geometric, i.e. taken on ``log(x)``: the
    cells of a logarithmic axis have the same width on screen.

    Args:
        centers: Increasing values, shape ``(n,)``; > 0 when ``log``.
        log: Take the midpoints on a logarithmic scale.

    Returns:
        The edges, shape ``(n + 1,)``. A single value gets a cell of width 1
        (a factor 10 on a logarithmic scale) around it.

    Raises:
        ValueError: If the values are empty, not increasing, or not > 0
            with ``log``.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError(f"centers must be a non-empty 1D array, got {c.shape}.")
    if np.any(np.diff(c) <= 0):
        raise ValueError("centers must be strictly increasing.")
    if log:
        if np.any(c <= 0):
            raise ValueError("centers must be > 0 on a logarithmic scale.")
        return np.asarray(10.0 ** cell_edges(np.log10(c)))
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def draw_heatmap(
    ax: Axes,
    grid: SweepGrid,
    label: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    log_color: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> QuadMesh:
    """Draw a :class:`SweepGrid` as a color map, with its colorbar.

    Args:
        ax: Axis to draw on.
        grid: Output of :func:`sweep_grid`.
        label: Label of the colorbar, usually the metric.
        log_x: Logarithmic x axis, with cells of equal width on it.
        log_y: Logarithmic y axis, with cells of equal width on it.
        log_color: Logarithmic color scale, for a metric spanning several
            orders of magnitude (the NMSE); only the values > 0 are drawn.
        vmin: Bottom of the color scale; ``None`` takes the smallest value.
        vmax: Top of the color scale; ``None`` takes the largest value.
        cmap: Matplotlib color map.

    Returns:
        The ``QuadMesh`` of ``ax.pcolormesh``; the points without a run
        (``nan``) are left blank.
    """
    values = np.ma.masked_invalid(grid.mean)
    norm: Normalize
    if log_color:
        values = np.ma.masked_less_equal(values, 0.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        cell_edges(grid.x, log=log_x),
        cell_edges(grid.y, log=log_y),
        values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        # A labelled mesh is saved by QELM.utils.plots.save_figure.
        label=label or "heatmap",
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    colorbar = ax.figure.colorbar(mesh, ax=ax)
    if label is not None:
        colorbar.set_label(label)
    return mesh
