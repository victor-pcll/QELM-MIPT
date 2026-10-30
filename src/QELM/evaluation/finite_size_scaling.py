"""Crossings and data collapse of an order parameter ``O(p, N)`` (issue #185).

Near a continuous transition at ``p_c``, with correlation-length exponent
``nu``, the curves of different sizes ``N`` fall on one curve::

    O(p, N) = F((p - p_c) N**(1 / nu))

so they cross at ``p_c`` (up to corrections in ``N``). :func:`crossings`
locates the crossing of each pair of sizes, and :func:`fit_collapse` finds
the ``(p_c, nu)`` that best collapse every size on one curve, with the
quality function of Houdayer & Hartmann, PRB 70, 014418 (2004): each point
is compared with the other sizes linearly interpolated at its scaled
abscissa, in units of the error bars. Its minimum is about 1 for a good
collapse.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

# One size: (p, O, error of O), three arrays of the same length, p increasing.
Curve = tuple[np.ndarray, np.ndarray, np.ndarray]


def crossing(x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> float | None:
    """Return the first ``x`` where two curves on the same grid cross.

    The difference is interpolated linearly between grid points; ``nan``
    values are skipped.

    Args:
        x: Common abscissae, increasing.
        y_a: First curve on ``x``.
        y_b: Second curve on ``x``.

    Returns:
        The crossing abscissa, or ``None`` if the curves do not cross.
    """
    d = np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float)
    keep = np.isfinite(d)
    x, d = np.asarray(x, dtype=float)[keep], d[keep]
    for k in range(len(x) - 1):
        if d[k] == 0:
            return float(x[k])
        if d[k] * d[k + 1] < 0:
            return float(x[k] - d[k] * (x[k + 1] - x[k]) / (d[k + 1] - d[k]))
    if len(x) and d[-1] == 0:
        return float(x[-1])
    return None


def crossings(curves: Mapping[int, Curve]) -> dict[tuple[int, int], float | None]:
    """Return the crossing of every pair of sizes, on the ``p`` they share.

    Args:
        curves: ``{N: (p, O, err)}``.

    Returns:
        ``{(N_a, N_b): p of the first crossing, or None}``, ``N_a < N_b``.
    """
    out: dict[tuple[int, int], float | None] = {}
    for a, b in combinations(sorted(curves), 2):
        ya = dict(zip(curves[a][0], curves[a][1], strict=True))
        yb = dict(zip(curves[b][0], curves[b][1], strict=True))
        p = np.array(sorted(set(ya) & set(yb)), dtype=float)
        out[(a, b)] = crossing(
            p, np.array([ya[q] for q in p]), np.array([yb[q] for q in p])
        )
    return out


def crossing_errors(
    curves: Mapping[int, Curve],
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[tuple[int, int], float]:
    """Return the bootstrap error of each crossing of :func:`crossings`.

    Each point is redrawn from a normal law of its error bar, the crossings
    are located again, and the error is half the 16-84 % range of the
    resamples where the pair still crosses: a robust standard deviation,
    since a resample can cross a second time far from the first crossing.

    Args:
        curves: ``{N: (p, O, err)}``.
        n_boot: Bootstrap resamples.
        rng: Random generator of the bootstrap.

    Returns:
        ``{(N_a, N_b): error}``, ``nan`` when fewer than two resamples cross.
    """
    gen = rng if rng is not None else np.random.default_rng(0)
    samples: dict[tuple[int, int], list[float]] = {}
    for _ in range(n_boot):
        noisy = {
            N: (p, np.asarray(y) + gen.normal(size=np.shape(y)) * np.asarray(e), e)
            for N, (p, y, e) in curves.items()
        }
        for pair, value in crossings(noisy).items():
            samples.setdefault(pair, [])
            if value is not None:
                samples[pair].append(value)
    return {
        pair: float(np.subtract(*np.percentile(v, [84, 16])) / 2)
        if len(v) > 1
        else float("nan")
        for pair, v in samples.items()
    }


def collapse_quality(curves: Mapping[int, Curve], p_c: float, nu: float) -> float:
    """Houdayer-Hartmann quality of the collapse with ``(p_c, nu)``.

    Every point ``(x_i, y_i, dy_i)``, with ``x = (p - p_c) N**(1 / nu)``, is
    compared with each other size whose scaled curve brackets ``x_i``,
    linearly interpolated there to ``(Y, dY)``; the quality is the mean of
    ``(y_i - Y)**2 / (dy_i**2 + dY**2)`` over these pairs.

    Args:
        curves: ``{N: (p, O, err)}``, at least two sizes.
        p_c: Trial critical point.
        nu: Trial exponent, positive.

    Returns:
        The quality, ``inf`` if no two sizes overlap or ``nu <= 0``.
    """
    if nu <= 0:
        return float("inf")
    scaled = {
        N: ((np.asarray(p) - p_c) * N ** (1 / nu), np.asarray(y), np.asarray(e))
        for N, (p, y, e) in curves.items()
    }
    terms = []
    for N, (x, y, e) in scaled.items():
        for M, (xm, ym, em) in scaled.items():
            if M == N:
                continue
            for xi, yi, ei in zip(x, y, e, strict=True):
                k = int(np.searchsorted(xm, xi))
                if k == 0 or k == len(xm):
                    continue
                w = (xi - xm[k - 1]) / (xm[k] - xm[k - 1])
                Y = (1 - w) * ym[k - 1] + w * ym[k]
                dY2 = ((1 - w) * em[k - 1]) ** 2 + (w * em[k]) ** 2
                terms.append((yi - Y) ** 2 / max(ei**2 + dY2, 1e-30))
    return float(np.mean(terms)) if terms else float("inf")


@dataclass(frozen=True)
class Collapse:
    """Result of :func:`fit_collapse`.

    Attributes:
        p_c: Best critical point.
        nu: Best exponent.
        quality: :func:`collapse_quality` at the optimum.
        p_c_err: Standard deviation of ``p_c`` over the bootstrap.
        nu_err: Standard deviation of ``nu`` over the bootstrap.
    """

    p_c: float
    nu: float
    quality: float
    p_c_err: float
    nu_err: float


def _optimum(
    curves: Mapping[int, Curve], starts: list[tuple[float, float]]
) -> tuple[float, float, float]:
    """Best Nelder-Mead minimum of the quality over ``starts``: ``(p_c, nu, q)``."""

    def quality(v: np.ndarray) -> float:
        return collapse_quality(curves, float(v[0]), float(v[1]))

    best = (float("nan"), float("nan"), float("inf"))
    for start in starts:
        res = minimize(
            quality,
            [float(start[0]), float(start[1])],
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-6},
        )
        if res.fun < best[2]:
            best = (float(res.x[0]), float(res.x[1]), float(res.fun))
    return best


def fit_collapse(
    curves: Mapping[int, Curve],
    p_c0: float,
    nu0: float,
    n_boot: int = 50,
    rng: np.random.Generator | None = None,
) -> Collapse:
    """Find the ``(p_c, nu)`` that minimize :func:`collapse_quality`.

    Nelder-Mead from ``(p_c0, nu0)`` and four starts around it. The errors
    come from a parametric bootstrap: each point is redrawn from a normal
    law of its error bar and the collapse is fitted again.

    Args:
        curves: ``{N: (p, O, err)}``, at least two sizes.
        p_c0: Starting critical point.
        nu0: Starting exponent.
        n_boot: Bootstrap resamples; 0 gives ``nan`` errors.
        rng: Random generator of the bootstrap.

    Returns:
        The :class:`Collapse`.

    Raises:
        ValueError: With fewer than two sizes.
    """
    if len(curves) < 2:
        raise ValueError(f"A collapse needs at least two sizes, got {len(curves)}.")
    data: dict[int, Curve] = {
        int(N): (
            np.asarray(p, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(e, dtype=float),
        )
        for N, (p, y, e) in curves.items()
    }
    starts = [(p_c0, nu0)] + [
        (p_c0 + dp, nu0 * f) for dp in (-0.02, 0.02) for f in (0.7, 1.4)
    ]
    p_c, nu, quality = _optimum(data, starts)
    gen = rng if rng is not None else np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        noisy: dict[int, Curve] = {
            N: (p, y + gen.normal(size=y.shape) * e, e) for N, (p, y, e) in data.items()
        }
        b = _optimum(noisy, [(p_c, nu)])
        boot.append(b[:2])
    arr = np.array(boot) if boot else np.full((1, 2), np.nan)
    return Collapse(
        p_c=p_c,
        nu=nu,
        quality=quality,
        p_c_err=float(np.std(arr[:, 0])) if boot else float("nan"),
        nu_err=float(np.std(arr[:, 1])) if boot else float("nan"),
    )
