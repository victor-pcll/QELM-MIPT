#!/usr/bin/env python3
"""Error budget of a (p_c, nu) collapse: statistical plus systematic.

Reads the Haar runs of ``experiments/haar_collapse_map`` (``Delta`` after
``T = r N`` half-layers) and, with the exact Houdayer-Hartmann quality
(:func:`QELM.evaluation.finite_size_scaling.hh_quality`), computes:

1. the optimum (grid, then Nelder-Mead);
2. the statistical error from a parametric bootstrap (each point drawn from
   ``N(y, dy)``; the circuits are independent from one ``p`` to the next);
3. the same error times ``sqrt(max(1, S_min))``;
4. the projected half-widths of the region ``S <= 1.3 S_min`` (rule of
   Zabalo et al., PRB 101, 060301(R), 2020) and the ``S_min + 1`` rule of
   Melchert (arXiv:0910.5403), one parameter at a time;
5. the systematic error: half range of the optima over analysis variants
   fixed in advance (sizes kept, window in ``p``, ``Delta`` or ``log Delta``);
6. the extrapolation of Skinner, Ruhman & Nahum (PRX 9, 031009, 2019): the
   optimum without the largest sizes, then a straight line in ``1 / N_max``.

Run from the repository root:

    uv run python experiments/collapse_error_budget/error_budget.py --ratio 2
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from QELM.evaluation.finite_size_scaling import Curve, hh_quality, hh_terms

HERE = Path(__file__).resolve().parent
HAAR_RUNS = HERE.parent / "haar_collapse_map" / "runs"
GRID = (np.linspace(0.10, 0.26, 33), np.linspace(0.6, 3.0, 49))


def load_curves(
    folder: Path,
    ratio: int,
    n_min: int = 0,
    n_max: int = 10**9,
    window: tuple[float, float] = (0.0, 1.0),
    log: bool = False,
) -> dict[int, Curve]:
    """``{N: (p, Delta, err)}`` from ``haar_N*.json`` (mean and SEM per ``p``)."""
    out: dict[int, Curve] = {}
    key = f"delta_T{ratio}N"
    for path in sorted(folder.glob("haar_N*.json")):
        run = json.loads(path.read_text())
        n = int(run["n"])
        if not n_min <= n <= n_max or ratio not in run["ratios"]:
            continue
        rows = [
            r for r in run["per_p"] if window[0] - 1e-9 <= r["p"] <= window[1] + 1e-9
        ]
        p = np.array([r["p"] for r in rows])
        y = np.array([r[key][0] for r in rows])
        e = np.array([r[key][1] for r in rows])
        keep = (y > 0) & (e > 0)
        p, y, e = p[keep], y[keep], e[keep]
        if log:
            y, e = np.log(y), e / y
        o = np.argsort(p)
        out[n] = (p[o], y[o], e[o])
    return dict(sorted(out.items()))


def fit(
    curves: dict[int, Curve],
    start: tuple[float, float] | None,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    subtract: bool = False,
) -> tuple[float, float, float]:
    """Minimum ``(p_c, nu, S_min)``: coarse grid (if given), then Nelder-Mead."""
    if grid is not None:
        pcs, nus = grid
        q = np.array([[hh_quality(curves, a, b, subtract) for b in nus] for a in pcs])
        i, j = np.unravel_index(int(np.argmin(q)), q.shape)
        start = (float(pcs[i]), float(nus[j]))
    assert start is not None
    f: Callable[[np.ndarray], float] = lambda v: hh_quality(  # noqa: E731
        curves, float(v[0]), float(v[1]), subtract
    )
    res = minimize(
        f, start, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6}
    )
    return float(res.x[0]), float(res.x[1]), float(res.fun)


def bootstrap(
    curves: dict[int, Curve],
    best: tuple[float, float, float],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Parametric bootstrap: each point drawn from ``N(y, dy)``, then refit."""
    out = []
    for _ in range(n_boot):
        noisy = {
            n: (p, y + rng.normal(size=y.shape) * e, e)
            for n, (p, y, e) in curves.items()
        }
        out.append(fit(noisy, best[:2])[:2])
    return np.array(out)


def region(
    curves: dict[int, Curve],
    best: tuple[float, float, float],
    factor: float | None = None,
    add: float | None = None,
    half: tuple[float, float] = (0.03, 0.8),
    n: int = 61,
) -> tuple[float, float]:
    """Projected half-widths of ``{S <= factor S_min}`` or ``{S <= S_min + add}``."""
    pcs = np.linspace(best[0] - half[0], best[0] + half[0], n)
    nus = np.linspace(max(best[1] - half[1], 0.2), best[1] + half[1], n)
    q = np.array([[hh_quality(curves, a, b) for b in nus] for a in pcs])
    thr = best[2] * factor if factor is not None else best[2] + (add or 0.0)
    ii, jj = np.nonzero(q <= thr)
    if not len(ii):
        return float("nan"), float("nan")
    return float(np.ptp(pcs[ii]) / 2), float(np.ptp(nus[jj]) / 2)


def melchert(
    curves: dict[int, Curve], best: tuple[float, float, float], add: float = 1.0
) -> list[float]:
    """``S_min + 1`` rule of Melchert: one parameter varied, the other fixed."""
    errs = []
    for k, step in ((0, 1e-4), (1, 1e-3)):
        widths = []
        for sgn in (+1, -1):
            v = list(best[:2])
            for _ in range(20000):
                v[k] += sgn * step
                if hh_quality(curves, v[0], v[1]) > best[2] + add:
                    break
            widths.append(abs(v[k] - best[k]))
        errs.append(max(widths))
    return errs


def budget(runs: Path, ratio: int, n_boot: int) -> dict[str, Any]:
    """Every estimator of the error on ``(p_c, nu)`` for one ``T / N``."""
    rng = np.random.default_rng(0)
    report: dict[str, Any] = {"ratio": ratio}
    base = load_curves(runs, ratio)
    best = fit(base, None, GRID)
    n_terms = len(hh_terms(base, best[0], best[1]))
    boot = bootstrap(base, best, n_boot, rng)
    stat = boot.std(axis=0, ddof=1)
    inflate = np.sqrt(max(best[2], 1.0))
    report["central"] = {
        "sizes": list(base),
        "p_c": best[0],
        "nu": best[1],
        "S_min": best[2],
        "n_terms": n_terms,
    }
    report["stat"] = {
        "bootstrap": stat.tolist(),
        "bootstrap_corr": float(np.corrcoef(boot.T)[0, 1]),
        "bootstrap_x_sqrtS": (stat * inflate).tolist(),
        "region_1.3Smin": region(base, best, factor=1.3),
        "region_Smin+1/N": region(base, best, add=1.0 / n_terms),
        "melchert_Smin+1": melchert(base, best),
    }

    sizes = list(base)
    variants = []
    choices = itertools.product(
        [sizes[0], sizes[1]],  # N_min
        [sizes[-1], sizes[-2]],  # N_max
        [(0.0, 1.0), (0.10, 0.26), (0.12, 0.24)],  # window in p
        [False, True],  # Delta or log Delta
    )
    for n_min, n_max, win, log in choices:
        c = load_curves(runs, ratio, n_min, n_max, win, log)
        if len(c) < 3:
            continue  # HH needs at least three sizes to be smooth
        b = fit(c, best[:2])
        variants.append(
            {
                "n_min": n_min,
                "n_max": n_max,
                "window": win,
                "log": log,
                "p_c": b[0],
                "nu": b[1],
                "S_min": b[2],
            }
        )
    pc_v = np.array([v["p_c"] for v in variants])
    nu_v = np.array([v["nu"] for v in variants])
    report["variants"] = variants
    report["syst"] = {
        "half_range": [float(np.ptp(pc_v) / 2), float(np.ptp(nu_v) / 2)],
        "std": [float(pc_v.std(ddof=1)), float(nu_v.std(ddof=1))],
        "n_variants": len(variants),
    }

    sk = np.array(
        [(n_max, *fit(load_curves(runs, ratio, n_max=n_max), best[:2])[:2])
         for n_max in sizes[2:]]
    )  # fmt: skip
    if len(sk) >= 2:
        cp = np.polyfit(1 / sk[:, 0], sk[:, 1], 1)
        cn = np.polyfit(1 / sk[:, 0], sk[:, 2], 1)
        report["skinner"] = {
            "per_n_max": sk.tolist(),
            "extrapolated": [float(cp[1]), float(cn[1])],
            "uncertainty": [
                float(abs(cp[1] - sk[-1, 1])),
                float(abs(cn[1] - sk[-1, 2])),
            ],
        }

    s, y = report["stat"]["bootstrap_x_sqrtS"], report["syst"]["half_range"]
    report["quote"] = {
        "p_c": [best[0], s[0], y[0], float(np.hypot(s[0], y[0]))],
        "nu": [best[1], s[1], y[1], float(np.hypot(s[1], y[1]))],
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=HAAR_RUNS)
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    report = budget(a.runs, a.ratio, a.n_boot)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"error_budget_haar_T{a.ratio}N.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    q = report["quote"]
    print(
        f"T = {a.ratio}N: p_c = {q['p_c'][0]:.4f} +- {q['p_c'][1]:.4f} (stat) "
        f"+- {q['p_c'][2]:.4f} (syst), nu = {q['nu'][0]:.3f} +- {q['nu'][1]:.3f} "
        f"+- {q['nu'][2]:.3f}, S_min = {report['central']['S_min']:.2f}"
    )


if __name__ == "__main__":
    main()
