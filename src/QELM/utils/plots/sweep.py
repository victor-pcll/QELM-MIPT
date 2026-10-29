"""Draw the curves of a sweep and overlay reference curves (issue #147).

Shared by the plot scripts of ``experiments/`` and by the notebooks: a sweep
is read with :func:`QELM.utils.load_sweep`, averaged over the seeds with
:func:`QELM.utils.summarize`, then each size ``N`` (number of system qubits)
gives one curve of ``<metric>_mean`` against a swept parameter (``p``,
``gamma``), with an error bar per point.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from QELM.utils.reference import compare_to_reference

# Size in a reference curve name, "N = 6"; the loaders rename the paper's
# notation to ours first.
SIZE = re.compile(r"\bN\s*=\s*(\d+)")

Curves = Mapping[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SweepCurve:
    """One size of a sweep, ordered by the swept parameter.

    Attributes:
        x: Values of the swept parameter, shape ``(n,)``.
        mean: Mean of the metric over the seeds, shape ``(n,)``.
        err: Error bar: the standard error over the seeds when a point has
            several runs, else the error of the single run
            (``<metric>_err``, e.g. a bootstrap error) when the sweep wrote
            one, else 0.
        std: Spread between the seeds (``nan`` with one run), shape ``(n,)``.
        n_runs: Number of runs (seeds) per point, shape ``(n,)``.
        rows: The summary rows, in the order of ``x``.
    """

    x: np.ndarray
    mean: np.ndarray
    err: np.ndarray
    std: np.ndarray
    n_runs: np.ndarray
    rows: tuple[dict[str, Any], ...]

    def where(self, keep: np.ndarray) -> SweepCurve:
        """Return the points where the boolean mask ``keep`` is true."""
        return SweepCurve(
            self.x[keep],
            self.mean[keep],
            self.err[keep],
            self.std[keep],
            self.n_runs[keep],
            tuple(r for r, kept in zip(self.rows, keep, strict=True) if kept),
        )


def sizes(summary: Sequence[Mapping[str, Any]], size_key: str = "N") -> list[int]:
    """Return the sizes of a summary, in increasing order."""
    return sorted({int(r[size_key]) for r in summary})


def sweep_curve(
    summary: Sequence[Mapping[str, Any]],
    size: int,
    x_key: str,
    metric: str,
    size_key: str = "N",
) -> SweepCurve:
    """Return the curve of ``metric`` for one size of a summarized sweep.

    Args:
        summary: Output of :func:`QELM.utils.summarize`, grouped by
            ``(size_key, x_key)``, with ``metric`` (and ``<metric>_err`` when
            the runs report their own error) among its metrics.
        size: Value of ``size_key`` to keep.
        x_key: Row key of the swept parameter.
        metric: Metric name, ``<metric>_mean`` being read.
        size_key: Row key of the size.

    Returns:
        The :class:`SweepCurve`.
    """
    rows = tuple(
        dict(r)
        for r in sorted(
            (r for r in summary if int(r[size_key]) == size), key=lambda r: r[x_key]
        )
    )
    n_runs = np.array([r["n_runs"] for r in rows], dtype=int)
    sem = np.array([r[f"{metric}_sem"] for r in rows], dtype=float)
    single = np.array([r.get(f"{metric}_err_mean", np.nan) for r in rows], dtype=float)
    err = np.nan_to_num(np.where(n_runs > 1, sem, single))
    return SweepCurve(
        x=np.array([r[x_key] for r in rows], dtype=float),
        mean=np.array([r[f"{metric}_mean"] for r in rows], dtype=float),
        err=err,
        std=np.array([r[f"{metric}_std"] for r in rows], dtype=float),
        n_runs=n_runs,
        rows=rows,
    )


def draw_curve(
    ax: Axes,
    curve: SweepCurve,
    color: str | None = None,
    label: str | None = None,
    band: bool = False,
    log: bool = False,
    linestyle: str = "-",
) -> Any:
    """Draw a curve with its error bars, and optionally its seed spread.

    Args:
        ax: Axis to draw on.
        curve: Output of :func:`sweep_curve`.
        color: Line color; ``None`` takes the next color of the axis.
        label: Legend label.
        band: Shade ``mean +- std`` when every point has several runs.
        log: The y axis is logarithmic: a lower bar is stopped at half the
            mean, since a bar below 0 cannot be drawn.
        linestyle: Style of the line joining the points, e.g. ``"--"`` for a
            companion curve (:func:`draw_companion`); the error bars stay
            solid.

    Returns:
        The ``ErrorbarContainer`` of ``ax.errorbar``.
    """
    lower = np.minimum(curve.err, 0.5 * curve.mean) if log else curve.err
    bars = ax.errorbar(
        curve.x,
        curve.mean,
        yerr=[lower, curve.err],
        color=color,
        marker="o",
        capsize=3,
        linestyle=linestyle,
        label=label,
    )
    if band and curve.n_runs.size and np.all(curve.n_runs > 1):
        ax.fill_between(
            curve.x,
            curve.mean - curve.std,
            curve.mean + curve.std,
            color=bars.lines[0].get_color(),
            alpha=0.15,
            lw=0,
        )
    if log:
        ax.set_yscale("log")
    return bars


def draw_companion(
    ax: Axes,
    summary: Sequence[Mapping[str, Any]],
    size: int,
    x_key: str,
    metric: str,
    color: str | None = None,
    label: str | None = None,
    log: bool = False,
    size_key: str = "N",
    linestyle: str = "--",
) -> Any | None:
    """Overlay a second metric of the same sweep on a panel, dashed.

    For a reference level read from the sweep itself, e.g. the chance level
    of the accuracy on the accuracy panel (issue #130). The curve is
    drawn with :func:`draw_curve`, so it keeps its error bars and
    :func:`~QELM.utils.plots.save_figure` saves them.

    Args:
        ax: Axis to draw on.
        summary: Output of :func:`QELM.utils.summarize`, as for
            :func:`sweep_curve`.
        size: Value of ``size_key`` to keep.
        x_key: Row key of the swept parameter.
        metric: Metric to overlay, ``<metric>_mean`` being read.
        color: Line color, usually that of the main curve of the same size.
        label: Legend label.
        log: The y axis is logarithmic, as in :func:`draw_curve`.
        size_key: Row key of the size.
        linestyle: Style of the line, dashed by default; another style
            tells two companions of one curve apart, e.g. the QDA and K-NN
            ceilings of the ridge accuracy (issue #202).

    Returns:
        The ``ErrorbarContainer``, or ``None`` when the summary has no
        ``metric`` for this size (an older sweep): nothing is drawn.
    """
    rows = [r for r in summary if int(r[size_key]) == size]
    if not rows or any(f"{metric}_mean" not in r for r in rows):
        return None
    curve = sweep_curve(rows, size, x_key, metric, size_key=size_key)
    return draw_curve(ax, curve, color, label=label, log=log, linestyle=linestyle)


def draw_references(
    ax: Axes, curves: Curves, colors: Mapping[str, str], source: str
) -> None:
    """Draw reference curves, dashed with square markers.

    Args:
        ax: Axis to draw on.
        curves: ``{name: (x, y)}``, as from :func:`QELM.utils.load_wpd_csv`.
        colors: Color of every curve name.
        source: Prefix of the legend labels, e.g. ``"Barratt et al. 2022"``.
    """
    for name, (x_ref, y_ref) in curves.items():
        ax.plot(
            x_ref,
            y_ref,
            color=colors[name],
            ls="--",
            marker="s",
            ms=5,
            label=f"{source}: {name}",
        )


def draw_residuals(
    ax: Axes,
    curve: SweepCurve,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    color: str | None = None,
    label: str | None = None,
) -> Any:
    """Draw a curve minus a reference curve, with the curve's error bars.

    The reference is interpolated linearly at the points of ``curve``; the
    points outside its range are left out. The residual keeps the error
    bars of ``curve``, which are often far smaller than a marker on the
    scale of the curve itself (issue #152). A dotted line marks 0.

    Args:
        ax: Axis to draw on.
        curve: Output of :func:`sweep_curve`.
        x_ref: Abscissae of the reference curve.
        y_ref: Values of the reference curve.
        color: Line color; ``None`` takes the next color of the axis.
        label: Legend label.

    Returns:
        The ``ErrorbarContainer`` of ``ax.errorbar``.
    """
    order = np.argsort(x_ref)
    x_ref, y_ref = np.asarray(x_ref, float)[order], np.asarray(y_ref, float)[order]
    inside = curve.where((curve.x >= x_ref[0]) & (curve.x <= x_ref[-1]))
    residual = replace(inside, mean=inside.mean - np.interp(inside.x, x_ref, y_ref))
    ax.axhline(0.0, color="gray", ls=":", lw=1.5)
    return draw_curve(ax, residual, color, label=label)


def same_size(curves: Curves, size: int) -> Curves:
    """Return the curves whose name gives ``size`` (``"N = 6"``) or no size."""
    out = {}
    for name, xy in curves.items():
        match = SIZE.search(name)
        if match is None or int(match.group(1)) == size:
            out[name] = xy
    return out


def compare_same_size(
    curve: SweepCurve, size: int, curves: Curves
) -> dict[str, dict[str, float]]:
    """Compare a curve to the reference curves of the same size.

    A reference whose name gives a size (``"N = 6"``) is compared only to the
    curve of that size; one without a size is compared to every curve.

    Returns:
        ``{reference name: stats of QELM.utils.compare_to_reference}``.
    """
    return {
        name: compare_to_reference(curve.x, curve.mean, x_ref, y_ref)
        for name, (x_ref, y_ref) in same_size(curves, size).items()
    }


def print_comparison(comparison: Mapping[str, Any]) -> None:
    """Print a comparison written by the plot scripts, one line per curve.

    Args:
        comparison: ``{metric: {N<N>: {reference: stats}}}``, plus optionally
            ``"binder_crossings"``: ``{metric: {"N<a>-N<b>": p or None}}``.
    """
    for metric, crossings in comparison.get("binder_crossings", {}).items():
        for pair, x in crossings.items():
            where = "no crossing" if x is None else f"at {x:.3f}"
            print(f"{metric} crossing {pair}: {where}")
    for metric, by_size in comparison.items():
        if metric == "binder_crossings":
            continue
        for size, curves in by_size.items():
            for name, stats in curves.items():
                print(
                    f"{metric} {size} vs {name}: "
                    f"max |diff| = {stats['max_abs_diff']:.3g}, "
                    f"rms = {stats['rms_diff']:.3g} ({stats['n_points']} points)"
                )
