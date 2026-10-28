"""Finite-size scaling of the half-chain entropy across the MIPT (issue #239).

Two estimates of the critical measurement probability ``p_c`` from curves
``S(p, N)`` of several sizes ``N``:

- :func:`curve_crossings`: where the curves of two sizes cross. For the
  normalized entropy ``S / (N/2)``, the larger size lies above in the
  volume-law phase (it tends to a nonzero density) and below in the area-law
  phase (``S`` stops growing with ``N``), so the curves cross near ``p_c``;
- :func:`fit_collapse`: the data collapse of Skinner, Ruhman and Nahum, PRX 9,
  031009 (2019), Sec. IV: ``S(p, N) - S(p_c, N) = F((p - p_c) N^(1/nu))``.
  Subtracting ``S(p_c, N)`` removes the ``log N`` growth at the critical
  point. :func:`collapse_cost` measures how far the rescaled curves are from
  one curve, and :func:`fit_collapse` minimizes it over ``(p_c, nu)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import scipy.optimize

# One curve per size: N -> (p, S), p increasing.
Curves = Mapping[int, tuple[Sequence[float], Sequence[float]]]


def curve_crossings(
    p: Sequence[float], y_small: Sequence[float], y_large: Sequence[float]
) -> list[float]:
    """Return the ``p`` where two curves on the same grid cross.

    A crossing is a change of sign of ``y_large - y_small`` between two
    neighbouring points, placed by linear interpolation; a zero difference
    on a grid point is a crossing there.

    Args:
        p: Increasing grid.
        y_small, y_large: The two curves on ``p``.

    Returns:
        The crossing points, in increasing order, possibly empty.
    """
    x = np.asarray(p, dtype=float)
    d = np.asarray(y_large, dtype=float) - np.asarray(y_small, dtype=float)
    out: list[float] = []
    for i in range(d.size - 1):
        if d[i] == 0.0:
            out.append(float(x[i]))
        elif d[i] * d[i + 1] < 0.0:
            out.append(float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i])))
    if d.size and d[-1] == 0.0:
        out.append(float(x[-1]))
    return out


def rescale(curves: Curves, p_c: float, nu: float) -> dict[int, np.ndarray]:
    """Return the rescaled curves ``(x, y)`` of the collapse, per size.

    ``x = (p - p_c) N^(1/nu)`` and ``y = S(p, N) - S(p_c, N)``, with
    ``S(p_c, N)`` interpolated linearly in ``p``.

    Returns:
        ``N -> array of shape (2, n_p)``, rows ``x`` and ``y``.
    """
    out: dict[int, np.ndarray] = {}
    for N, (p, S) in curves.items():
        p_arr, S_arr = np.asarray(p, dtype=float), np.asarray(S, dtype=float)
        x = (p_arr - p_c) * float(N) ** (1.0 / nu)
        y = S_arr - float(np.interp(p_c, p_arr, S_arr))
        out[int(N)] = np.vstack([x, y])
    return out


def collapse_cost(
    curves: Curves, p_c: float, nu: float, window: float | None = None
) -> float:
    """Return the mismatch of the rescaled curves, 0 for a perfect collapse.

    For every ordered pair of sizes ``(a, b)``, the points of ``a`` that fall
    inside the ``x`` range of ``b`` are compared with ``b`` interpolated
    linearly there. The cost is the mean squared difference over all these
    points, divided by the variance of all the ``y`` (so that it does not
    depend on the units of ``S``).

    Args:
        curves: ``N -> (p, S)``.
        p_c, nu: The trial critical point and exponent.
        window: When set, only the points with ``|p - p_c| <= window`` are
            compared: the scaling form holds near ``p_c`` only.

    Returns:
        The cost, or ``inf`` when no two curves overlap.
    """
    scaled = rescale(curves, p_c, nu)
    if window is not None:
        # |p - p_c| <= window is |x| <= window N^(1/nu).
        scaled = {
            N: xy[:, np.abs(xy[0]) <= (window + 1e-12) * float(N) ** (1.0 / nu)]
            for N, xy in scaled.items()
        }
        scaled = {N: xy for N, xy in scaled.items() if xy.shape[1] >= 2}
    residuals: list[np.ndarray] = []
    for a, (xa, ya) in scaled.items():
        for b, (xb, yb) in scaled.items():
            if a == b:
                continue
            inside = (xa >= xb.min()) & (xa <= xb.max())
            if inside.any():
                residuals.append(ya[inside] - np.interp(xa[inside], xb, yb))
    if not residuals:
        return float("inf")
    all_y = np.concatenate([xy[1] for xy in scaled.values()])
    scale = float(np.var(all_y)) or 1.0
    return float(np.mean(np.concatenate(residuals) ** 2) / scale)


def fit_collapse(
    curves: Curves,
    p_range: tuple[float, float],
    nu_range: tuple[float, float] = (0.5, 4.0),
    grid: int = 41,
    window: float | None = None,
) -> dict[str, float]:
    """Find the ``(p_c, nu)`` of the best collapse.

    A grid search over ``p_range x nu_range`` (``grid`` points each, ``nu``
    on a log scale), refined from its best point by Nelder-Mead inside the
    same box.

    Args:
        curves: ``N -> (p, S)``, at least two sizes.
        p_range: Search interval of ``p_c``, inside the ``p`` grid.
        nu_range: Search interval of ``nu``.
        grid: Points per axis of the grid search.
        window: Passed to :func:`collapse_cost`.

    Returns:
        ``{"p_c", "nu", "cost"}``.

    Raises:
        ValueError: With fewer than two sizes.
    """
    if len(curves) < 2:
        raise ValueError(f"A collapse needs at least two sizes, got {len(curves)}.")
    p_grid = np.linspace(*p_range, grid)
    nu_grid = np.geomspace(*nu_range, grid)
    costs = np.array(
        [[collapse_cost(curves, p, nu, window) for nu in nu_grid] for p in p_grid]
    )
    i, j = np.unravel_index(int(np.argmin(costs)), costs.shape)
    lo = np.array([p_range[0], nu_range[0]])
    hi = np.array([p_range[1], nu_range[1]])

    def objective(v: np.ndarray) -> float:
        v = np.clip(v, lo, hi)
        return collapse_cost(curves, float(v[0]), float(v[1]), window)

    res = scipy.optimize.minimize(
        objective,
        np.array([p_grid[i], nu_grid[j]]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-10},
    )
    best = np.clip(res.x, lo, hi)
    cost = objective(best)
    if cost > costs[i, j]:
        best, cost = np.array([p_grid[i], nu_grid[j]]), float(costs[i, j])
    return {"p_c": float(best[0]), "nu": float(best[1]), "cost": float(cost)}
