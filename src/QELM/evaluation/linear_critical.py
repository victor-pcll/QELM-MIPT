"""The critical ``p`` of the linear readout, from its own metrics (issue #227).

``p_linear*`` is the measurement probability where the ridge readout of the
charge fails most. It is read from the readout alone, with no non-linear
oracle (GMM, K-NN, Bayes), by three estimators on a curve against ``p``:

- ``accuracy``: the minimum of the test accuracy, the bottom of the valley;
- ``binder``: the minimum of the Binder ratio ``B = 1 - mu_4 / (3 mu_2^2)``
  of the test predictions ``y_hat``
  (:func:`QELM.evaluation.charge_sharpening.binder_ratio`): ``B -> 2/3`` when
  ``y_hat`` sits on two sharp peaks, ``B < 0`` when it is one bump with heavy
  tails;
- ``residual``: the maximum of ``1 - R2``, the error of the regression before
  the threshold.

:func:`locate_extremum` gives, for one curve, the extremum on the grid of
``p``, a sub-grid refinement, an interval from the error bars and two flags
of reliability (extremum on the edge of the range, prominence).
:func:`p_linear_table` applies the three estimators to every ``(N, h)`` of a
summarized sweep.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

Kind = Literal["min", "max"]

# (estimator, metric of the summary, extremum). ``residual`` reads the R2 and
# takes 1 - R2, whose error bar is that of the R2.
ESTIMATORS: tuple[tuple[str, str, Kind], ...] = (
    ("accuracy", "accuracy", "min"),
    ("binder", "binder", "min"),
    ("residual", "r2", "max"),
)


@dataclass(frozen=True)
class Extremum:
    """Location of the extremum of a curve ``y(p)``.

    Attributes:
        p_grid: The point of the grid where the mean curve is extremal (the
            first one on a tie): ``p[argmin(y)]`` or ``p[argmax(y)]``.
        p_refined: Vertex of the parabola through that point and its two
            neighbours, kept between them; ``p_grid`` when the extremum is on
            the edge of the range or the three points are aligned.
        value: ``y`` at ``p_grid``.
        p_low: Lower end of the interval of ``p_refined`` over curves
            redrawn within their error bars (16th percentile by default).
        p_high: Upper end of that interval (84th percentile by default).
        at_edge: The extremum is the first or the last point of the grid: the
            curve has no interior extremum in the range, and ``p_grid`` is
            only a bound.
        prominence: Depth of the extremum below the lower end of the curve
            (``min``), or height above its higher end (``max``), in
            units of the combined error bar; ``0`` at an edge, ``inf`` when
            both error bars are 0. A value of a few or more means a well
            defined valley or peak.
    """

    p_grid: float
    p_refined: float
    value: float
    p_low: float
    p_high: float
    at_edge: bool
    prominence: float


def _argext(y: np.ndarray, kind: Kind) -> int:
    return int(np.nanargmin(y) if kind == "min" else np.nanargmax(y))


def parabola_vertex(p: np.ndarray, y: np.ndarray, i: int) -> float:
    """Return the vertex of the parabola through points ``i - 1, i, i + 1``.

    The grid need not be uniform. The vertex is clipped to
    ``[p[i - 1], p[i + 1]]``.

    Args:
        p: Grid, increasing, shape ``(n,)``.
        y: Values, shape ``(n,)``.
        i: Index of the extremum.

    Returns:
        The vertex, or ``p[i]`` when ``i`` is an edge of the grid or the three
        points are aligned (no curvature).
    """
    if i <= 0 or i >= len(p) - 1:
        return float(p[i])
    x0, x1, x2 = p[i - 1], p[i], p[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    # Divided differences: y = y0 + d1 (x - x0) + d2 (x - x0)(x - x1).
    d1 = (y1 - y0) / (x1 - x0)
    d2 = ((y2 - y1) / (x2 - x1) - d1) / (x2 - x0)
    if d2 == 0 or not np.isfinite(d2):
        return float(x1)
    vertex = (x0 + x1) / 2 - d1 / (2 * d2)
    return float(np.clip(vertex, x0, x2))


def locate_extremum(
    p: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    err: Sequence[float] | np.ndarray | None = None,
    kind: Kind = "min",
    n_boot: int = 2000,
    level: float = 0.68,
    rng: np.random.Generator | None = None,
) -> Extremum:
    """Locate the minimum or the maximum of a curve against ``p``.

    The interval ``[p_low, p_high]`` comes from a parametric bootstrap: each
    point is redrawn ``n_boot`` times from a normal law of mean ``y`` and
    width ``err`` (the standard error over the seeds), independently, and the
    refined extremum of every redrawn curve is kept. A flat valley, where
    several points lie within their error bars of the extremum, gives a wide
    interval even when ``p_grid`` looks sharp.

    Args:
        p: Grid of ``p``, increasing, shape ``(n,)``, ``n >= 1``.
        y: Mean of the metric at each ``p``, shape ``(n,)``. ``nan`` points
            are left out.
        err: Error bar of each point, shape ``(n,)``; ``None`` or zeros give
            ``p_low = p_high = p_refined``.
        kind: ``"min"`` or ``"max"``.
        n_boot: Number of redrawn curves.
        level: Probability inside ``[p_low, p_high]``, centred.
        rng: Random generator; a fixed seed (0) when ``None``, so that the
            result is reproducible.

    Returns:
        The :class:`Extremum`.

    Raises:
        ValueError: If the shapes differ, ``p`` is not increasing, every
            ``y`` is ``nan``, ``kind`` is unknown or ``level`` is not in
            ``(0, 1)``.
    """
    if kind not in ("min", "max"):
        raise ValueError(f"kind must be 'min' or 'max', got {kind!r}.")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    e_arr = np.zeros_like(y_arr) if err is None else np.asarray(err, dtype=float)
    if p_arr.ndim != 1 or p_arr.shape != y_arr.shape or e_arr.shape != y_arr.shape:
        raise ValueError(
            f"p, y and err must be 1-D of one shape, got {p_arr.shape}, "
            f"{y_arr.shape} and {e_arr.shape}."
        )
    keep = np.isfinite(y_arr)
    if not keep.any():
        raise ValueError("Every value of y is nan.")
    p_arr, y_arr = p_arr[keep], y_arr[keep]
    e_arr = np.nan_to_num(e_arr[keep])
    if np.any(np.diff(p_arr) <= 0):
        raise ValueError("p must be strictly increasing.")

    i = _argext(y_arr, kind)
    at_edge = i == 0 or i == len(p_arr) - 1
    p_refined = parabola_vertex(p_arr, y_arr, i)

    if at_edge:
        prominence = 0.0
    else:
        # The end closest to the extremum: the lower one for a minimum.
        j = _argext(y_arr[[0, -1]], kind)
        end = (0, len(y_arr) - 1)[j]
        gap = abs(y_arr[end] - y_arr[i])
        scale = float(np.hypot(e_arr[end], e_arr[i]))
        prominence = gap / scale if scale > 0 else float("inf")

    if np.any(e_arr > 0) and n_boot > 0:
        rng = np.random.default_rng(0) if rng is None else rng
        draws = y_arr + e_arr * rng.standard_normal((n_boot, len(y_arr)))
        located = np.array([parabola_vertex(p_arr, d, _argext(d, kind)) for d in draws])
        tail = 100 * (1 - level) / 2
        p_low, p_high = np.percentile(located, [tail, 100 - tail])
    else:
        p_low = p_high = p_refined

    return Extremum(
        p_grid=float(p_arr[i]),
        p_refined=p_refined,
        value=float(y_arr[i]),
        p_low=float(p_low),
        p_high=float(p_high),
        at_edge=bool(at_edge),
        prominence=float(prominence),
    )


def estimator_curve(
    rows: Sequence[Mapping[str, Any]], metric: str, estimator: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(p, y, err)`` of one estimator, sorted by ``p``.

    Args:
        rows: Summary rows of one ``(N, h)``, with ``p``, ``<metric>_mean``
            and ``<metric>_sem`` (:func:`QELM.utils.summarize`).
        metric: Full metric name, e.g. ``"r2_M_Z"``.
        estimator: One of :data:`ESTIMATORS`; ``"residual"`` returns
            ``1 - mean``.
    """
    ordered = sorted(rows, key=lambda r: float(r["p"]))
    p = np.array([float(r["p"]) for r in ordered])
    y = np.array([float(r[f"{metric}_mean"]) for r in ordered])
    err = np.array([float(r.get(f"{metric}_sem", 0.0)) for r in ordered])
    if estimator == "residual":
        y = 1.0 - y
    return p, y, err


def p_linear_table(
    summary: Sequence[Mapping[str, Any]],
    name: str = "M_Z",
    n_boot: int = 2000,
    level: float = 0.68,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return the three estimates of ``p_linear*`` for every ``(N, h)``.

    Args:
        summary: Rows of :func:`QELM.utils.summarize` grouped by
            ``(N, h, p)``, with ``accuracy_<name>``, ``binder_<name>`` and
            ``r2_<name>`` among the metrics. An estimator whose metric is
            missing, or ``nan`` at every ``p``, is left out.
        name: Feature set of the readout, the suffix of the metrics.
        n_boot: As in :func:`locate_extremum`.
        level: As in :func:`locate_extremum`.
        seed: Seed of the bootstrap, one generator for the whole table.

    Returns:
        One row per ``(N, h, estimator)``, sorted by ``N`` then ``h``, with
        ``N``, ``h``, ``estimator``, ``metric`` and the fields of
        :class:`Extremum`.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
    for r in summary:
        groups.setdefault((int(r["N"]), float(r["h"])), []).append(r)
    table = []
    for (N, h), rows in sorted(groups.items()):
        for estimator, score, kind in ESTIMATORS:
            metric = f"{score}_{name}"
            if not all(f"{metric}_mean" in r for r in rows):
                continue
            p, y, err = estimator_curve(rows, metric, estimator)
            if not np.isfinite(y).any():  # e.g. B of predictions all equal
                continue
            found = locate_extremum(p, y, err, kind, n_boot, level, rng)
            table.append(
                {"N": N, "h": h, "estimator": estimator, "metric": metric}
                | asdict(found)
            )
    return table


def merge_summaries(
    *summaries: Sequence[Mapping[str, Any]],
    by: Sequence[str] = ("N", "h", "p"),
) -> list[dict[str, Any]]:
    """Merge summaries of the same sweep, row by row, on the keys ``by``.

    The Binder ratio of the prediction and the scores are summarized
    separately (they are saved in two figure files): this puts them back in
    one row per ``(N, h, p)``. A key of a later summary overwrites the same
    key of an earlier one.

    Raises:
        ValueError: If the summaries do not hold the same ``by`` values.
    """
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    keysets = []
    for summary in summaries:
        keys = set()
        for r in summary:
            key = tuple(float(r[k]) for k in by)
            keys.add(key)
            merged.setdefault(key, {}).update(r)
        keysets.append(keys)
    if any(keys != keysets[0] for keys in keysets[1:]):
        raise ValueError(f"The summaries do not cover the same {tuple(by)}.")
    return [merged[k] for k in sorted(merged)]


__all__ = [
    "ESTIMATORS",
    "Extremum",
    "estimator_curve",
    "locate_extremum",
    "merge_summaries",
    "p_linear_table",
    "parabola_vertex",
]
