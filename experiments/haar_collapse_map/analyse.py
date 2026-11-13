#!/usr/bin/env python3
"""Crossings, collapse and (p_c, nu) quality map of the exact Haar Delta (#348).

Reads ``runs/haar_N<N>.json`` (:mod:`run`), with, per ``p``, the mean and
error of ``Delta = N - log2(1 + W(m))`` after ``T = r N`` half-layers. At
fixed ``T / N`` the order parameter is scale invariant at the transition, so
(as in ``experiments/clifford_collapse_map``)::

    Delta(p, N) = F((p - p_c) N**(1 / nu))

and the curves of all sizes cross at ``p_c``. The cost of a pair
``(p_c, nu)`` is the Houdayer-Hartmann quality ``Q``
(:func:`QELM.evaluation.finite_size_scaling.collapse_quality`).

It writes, in ``--out``:

- ``crossing.png`` / ``.json``: ``Delta`` against ``p`` for every ``N``
  (``T = 2N``) with the pairwise crossings, and the crossings of
  neighbouring sizes against ``1 / N`` for ``T = N, 2N, 4N``;
- ``collapse.png`` / ``.json``: the collapse at the best pair and at the
  literature pair;
- ``collapse_map.png`` / ``.json``: the map of ``Q`` (wide and zoom);
- ``collapse_map_variants.png`` / ``.json``: the zoomed map for other data
  choices;
- ``summary.json``: crossings, optima with bootstrap errors, ``Q`` at the
  literature values, for every case.

Run from the repository root:

    uv run python experiments/haar_collapse_map/analyse.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from QELM.evaluation.finite_size_scaling import (
    Curve,
    collapse_quality,
    crossing_errors,
    crossings,
    fit_collapse,
    quality_map,
)
from QELM.utils.plots import save_figure, subplots

HERE = Path(__file__).resolve().parent
# Zabalo et al., PRB 101, 060301(R) (2020): Haar brickwork, periodic,
# tripartite mutual information of S_1, L = 16 to 24.
P_C_LIT = 0.168
P_C_LIT_ERR = 0.005
NU_LIT = 1.2
NU_LIT_ERR = 0.2
# Same paper, Table I, Renyi-2 (the index of Delta).
P_C_LIT_S2 = 0.162
NU_LIT_S2 = 1.3
WIDE = (np.linspace(0.08, 0.30, 111), np.linspace(0.3, 3.0, 136))
ZOOM_HALF = (0.03, (0.4, 2.4))
LEVELS = (1.5, 2.0)
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#ff7f0e"]


@dataclass(frozen=True)
class Case:
    """One choice of data for the collapse.

    Attributes:
        name: Key in the JSON files.
        label: Title on the figure.
        ratio: ``T / N``.
        min_n: Smallest size kept.
        log: Collapse ``log Delta`` instead of ``Delta``.
        p_window: Only the ``p`` in this closed interval.
    """

    name: str
    label: str
    ratio: int
    min_n: int = 4
    log: bool = False
    p_window: tuple[float, float] = (0.0, 1.0)


MAIN = Case("main", "T = 2N, Delta", 2)
VARIANTS = (
    Case("no_N4", "T = 2N, N >= 6", 2, min_n=6),
    Case("log", "T = 2N, log Delta", 2, log=True),
    Case("window", "T = 2N, 0.10 <= p <= 0.26", 2, p_window=(0.10, 0.26)),
    Case("T4N", "T = 4N, Delta", 4),
    Case("T1N", "T = N, Delta", 1),
    Case("T4N_no_N4", "T = 4N, N >= 6", 4, min_n=6),
)


def load_curves(folder: Path, case: Case) -> dict[int, Curve]:
    """Read ``{N: (p, Delta, err)}`` of a case from the runs of every size.

    Sizes whose run lacks the ratio of the case, or below ``case.min_n``,
    are left out, as are points with a zero error bar.
    """
    curves: dict[int, Curve] = {}
    lo, hi = case.p_window
    key = f"delta_T{case.ratio}N"
    for path in sorted(folder.glob("haar_N*.json")):
        run = json.loads(path.read_text())
        n = int(run["n"])
        if n < case.min_n or case.ratio not in run["ratios"]:
            continue
        rows = [r for r in run["per_p"] if lo - 1e-9 <= r["p"] <= hi + 1e-9]
        p = np.array([r["p"] for r in rows], dtype=float)
        y = np.array([r[key][0] for r in rows], dtype=float)
        e = np.array([r[key][1] for r in rows], dtype=float)
        keep = (y > 0) & (e > 0)
        p, y, e = p[keep], y[keep], e[keep]
        order = np.argsort(p)
        p, y, e = p[order], y[order], e[order]
        if case.log:
            y, e = np.log(y), e / y
        curves[n] = (p, y, e)
    return dict(sorted(curves.items()))


def grid_minimum(q: np.ndarray, p_c: np.ndarray, nu: np.ndarray) -> dict[str, float]:
    """The ``(p_c, nu, Q)`` of the smallest finite entry of a quality map."""
    i, j = np.unravel_index(int(np.argmin(q)), q.shape)
    return {"p_c": float(p_c[i]), "nu": float(nu[j]), "quality": float(q[i, j])}


def zoom_grid(p_c: float) -> tuple[np.ndarray, np.ndarray]:
    """A zoom of the map centred on ``p_c``."""
    half, (lo, hi) = ZOOM_HALF
    return np.linspace(p_c - half, p_c + half, 61), np.linspace(lo, hi, 101)


def pair_crossings(curves: dict[int, Curve], n_boot: int) -> list[dict[str, Any]]:
    """Crossings of every pair of sizes with their bootstrap errors."""
    cross = crossings(curves)
    errs = crossing_errors(curves, n_boot=n_boot, rng=np.random.default_rng(0))
    return [
        {"sizes": list(pair), "p": value, "err": errs.get(pair, float("nan"))}
        for pair, value in cross.items()
    ]


def analyse(curves: dict[int, Curve], wide: bool, n_boot: int) -> dict[str, Any]:
    """Maps, grid minimum, optimum, crossings and literature quality."""
    out: dict[str, Any] = {}
    q_wide = quality_map(curves, *WIDE)
    if wide:
        out["wide"] = {"p_c": WIDE[0], "nu": WIDE[1], "quality": q_wide}
    start = grid_minimum(q_wide, *WIDE)
    zoom = zoom_grid(start["p_c"])
    out["zoom"] = {"p_c": zoom[0], "nu": zoom[1], "quality": quality_map(curves, *zoom)}
    out["grid_min"] = grid_minimum(out["zoom"]["quality"], *zoom)
    g = out["grid_min"]
    fit = fit_collapse(
        curves, g["p_c"], g["nu"], n_boot=n_boot, rng=np.random.default_rng(0)
    )
    out["fit"] = {
        "p_c": fit.p_c,
        "p_c_err": fit.p_c_err,
        "nu": fit.nu,
        "nu_err": fit.nu_err,
        "quality": fit.quality,
    }
    out["quality_at_literature"] = collapse_quality(curves, P_C_LIT, NU_LIT)
    out["quality_at_literature_S2"] = collapse_quality(curves, P_C_LIT_S2, NU_LIT_S2)
    nu_line = np.linspace(0.3, 3.0, 271)
    q_line = [collapse_quality(curves, P_C_LIT, float(v)) for v in nu_line]
    j = int(np.argmin(q_line))
    out["nu_at_literature_p_c"] = {"nu": float(nu_line[j]), "quality": q_line[j]}
    out["crossings"] = pair_crossings(curves, n_boot=200)
    out["sizes"] = sorted(curves)
    return out


def draw_map(ax: Axes, block: dict[str, Any], result: dict[str, Any], fig: Any) -> None:
    """Heat map of ``Q`` with the guide contours, the optimum and the literature."""
    p_c, nu, q = block["p_c"], block["nu"], block["quality"]
    q_min = result["fit"]["quality"]
    finite = q[np.isfinite(q)]
    mesh = ax.pcolormesh(
        p_c, nu, q.T, shading="nearest", cmap="viridis_r",
        norm=LogNorm(vmin=q_min, vmax=float(np.percentile(finite, 98))),
        rasterized=True,
    )  # fmt: skip
    ax.contour(
        p_c, nu, q.T, levels=[q_min * f for f in LEVELS],
        colors="white", linewidths=[1.2, 0.8], linestyles=["-", "--"],
    )  # fmt: skip
    fit = result["fit"]
    ax.errorbar(
        fit["p_c"], fit["nu"], xerr=fit["p_c_err"], yerr=fit["nu_err"],
        marker="o", ms=6, color="#d62728", mec="white", ls="none",
        label=f"best: p_c = {fit['p_c']:.3f}, nu = {fit['nu']:.2f} (Q = {q_min:.1f})",
    )  # fmt: skip
    ax.errorbar(
        P_C_LIT, NU_LIT, xerr=P_C_LIT_ERR, yerr=NU_LIT_ERR, marker="*", ms=12,
        color="#ff7f0e", mec="black", ls="none", ecolor="#ff7f0e",
        label=f"Zabalo et al. (TMI): {P_C_LIT}(5), {NU_LIT}(2)",
    )  # fmt: skip
    ax.plot(
        P_C_LIT_S2, NU_LIT_S2, marker="s", ms=6, color="#ffbb78", mec="black",
        ls="none", label=f"Zabalo et al. (S_2): {P_C_LIT_S2}, {NU_LIT_S2}",
    )  # fmt: skip
    ax.set_xlim(p_c[0], p_c[-1])
    ax.set_ylim(nu[0], nu[-1])
    ax.set_xlabel("p_c")
    ax.set_ylabel("nu")
    fig.colorbar(mesh, ax=ax, label="collapse quality Q (log scale)")


def draw_curves(ax: Axes, curves: dict[int, Curve], cross: list[dict[str, Any]],
                log: bool) -> None:  # fmt: skip
    """``Delta`` against ``p`` for each size, neighbour crossings marked."""
    for n, c in zip(curves, COLORS, strict=False):
        p, y, e = curves[n]
        if log:
            y, e = np.exp(y), e * np.exp(y)
        ax.errorbar(p, y, yerr=e, marker="o", ms=3, lw=1, color=c, label=f"N = {n}")
    sizes = list(curves)
    for row in cross:
        a, b = row["sizes"]
        if row["p"] is None or sizes.index(b) != sizes.index(a) + 1:
            continue
        ya = np.interp(row["p"], curves[a][0], curves[a][1])
        ax.plot(row["p"], np.exp(ya) if log else ya, "kx", ms=8, mew=1.5)
    ax.axvspan(P_C_LIT - P_C_LIT_ERR, P_C_LIT + P_C_LIT_ERR, color="#ff7f0e",
               alpha=0.25, label="p_c, Zabalo et al.")  # fmt: skip
    ax.set_xlabel("p")
    ax.set_ylabel("Delta = N - log2(1 + W(m))")
    ax.legend(fontsize=8)


def draw_crossing_drift(ax: Axes, results: dict[str, dict[str, Any]]) -> None:
    """Crossing of neighbouring sizes against ``1 / mean(N)``, per ``T / N``."""
    for (label, result), c in zip(results.items(), COLORS, strict=False):
        sizes = result["sizes"]
        xs, ys, es = [], [], []
        for row in result["crossings"]:
            a, b = row["sizes"]
            if row["p"] is None or sizes.index(b) != sizes.index(a) + 1:
                continue
            xs.append(2 / (a + b))
            ys.append(row["p"])
            es.append(row["err"])
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1, color=c, label=label)
    ax.axhspan(P_C_LIT - P_C_LIT_ERR, P_C_LIT + P_C_LIT_ERR, color="#ff7f0e",
               alpha=0.25, label="Zabalo et al., L = 16-24")  # fmt: skip
    ax.set_xlim(left=0)
    ax.set_xlabel("1 / mean(N_a, N_b), neighbouring sizes")
    ax.set_ylabel("crossing p of Delta")
    ax.legend(fontsize=8)


def draw_collapse(ax: Axes, curves: dict[int, Curve], p_c: float, nu: float,
                  log: bool, title: str) -> None:  # fmt: skip
    """``Delta`` against ``(p - p_c) N**(1/nu)`` for each size."""
    for n, c in zip(curves, COLORS, strict=False):
        p, y, e = curves[n]
        x = (p - p_c) * n ** (1 / nu)
        if log:
            y, e = np.exp(y), e * np.exp(y)
        ax.errorbar(x, y, yerr=e, marker="o", ms=3, lw=1, color=c, label=f"N = {n}")
    ax.set_yscale("log")
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel(f"(p - {p_c:.3f}) N^(1/{nu:.2f})")
    ax.set_ylabel("Delta")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def _jsonable(value: Any) -> Any:
    """Arrays as lists, non-finite numbers as ``null``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("sizes", "grid_min", "fit", "quality_at_literature",
            "nu_at_literature_p_c",
            "quality_at_literature_S2", "crossings")  # fmt: skip
    return {k: _jsonable(result[k]) for k in keys}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    parser.add_argument("--n-boot", type=int, default=40)
    args = parser.parse_args(argv)

    summary: dict[str, Any] = {
        "literature": {
            "reference": "Zabalo et al., PRB 101, 060301(R) (2020)",
            "tmi_S1": {"p_c": [P_C_LIT, P_C_LIT_ERR], "nu": [NU_LIT, NU_LIT_ERR]},
            "S2_table_I": {"p_c": [P_C_LIT_S2, 0.003], "nu": [NU_LIT_S2, 0.1]},
        }
    }
    curves = load_curves(args.runs, MAIN)
    main_result = analyse(curves, wide=True, n_boot=args.n_boot)
    summary[MAIN.name] = {"label": MAIN.label, **_summary(main_result)}

    drift: dict[str, dict[str, Any]] = {}
    for ratio in (1, 2, 4):
        c = load_curves(args.runs, Case(f"T{ratio}N", "", ratio))
        if len(c) >= 2:
            drift[f"T = {ratio}N"] = {
                "sizes": sorted(c),
                "crossings": pair_crossings(c, n_boot=200),
            }
    fig, axes = subplots(1, 2, figsize=(12, 4.8))
    draw_curves(axes[0], curves, main_result["crossings"], MAIN.log)
    axes[0].set_title("(a) Delta at T = 2N; x: crossing of neighbouring N", fontsize=10)
    draw_crossing_drift(axes[1], drift)
    axes[1].set_title("(b) drift of the crossing with N", fontsize=10)
    save_figure(fig, args.out, "crossing", data=_jsonable(drift))

    fit = main_result["fit"]
    fig, axes = subplots(1, 2, figsize=(12, 4.8))
    draw_collapse(axes[0], curves, fit["p_c"], fit["nu"], MAIN.log,
                  f"(a) best pair, Q = {fit['quality']:.1f}")  # fmt: skip
    draw_collapse(axes[1], curves, P_C_LIT, NU_LIT, MAIN.log,
                  f"(b) Zabalo et al. pair, Q = "
                  f"{main_result['quality_at_literature']:.1f}")  # fmt: skip
    save_figure(fig, args.out, "collapse", data=_summary(main_result))

    fig, axes = subplots(1, 2, figsize=(13, 5))
    draw_map(axes[0], main_result["wide"], main_result, fig)
    axes[0].set_title(f"(a) Q over (p_c, nu): {MAIN.label}, N = "
                      f"{', '.join(map(str, main_result['sizes']))}",
                      fontsize=10)  # fmt: skip
    axes[0].legend(fontsize=7, loc="upper right")
    draw_map(axes[1], main_result["zoom"], main_result, fig)
    axes[1].set_title(
        f"(b) zoom; white: Q = {LEVELS[0]} and {LEVELS[1]} x Q_min", fontsize=10
    )
    save_figure(fig, args.out, "collapse_map", data=_jsonable(main_result))

    fig, axes = subplots(2, 3, figsize=(18, 9))
    variants: dict[str, Any] = {}
    for ax, case in zip(axes.flat, VARIANTS, strict=True):
        c = load_curves(args.runs, case)
        if len(c) < 2:
            ax.set_visible(False)
            continue
        result = analyse(c, wide=False, n_boot=args.n_boot)
        variants[case.name] = _jsonable(result)
        summary[case.name] = {"label": case.label, **_summary(result)}
        draw_map(ax, result["zoom"], result, fig)
        ax.set_title(f"{case.label}, N = {', '.join(map(str, result['sizes']))}",
                     fontsize=10)  # fmt: skip
        ax.legend(fontsize=6, loc="upper right")
    save_figure(fig, args.out, "collapse_map_variants", data=variants)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, row in summary.items():
        if name == "literature":
            continue
        f = row["fit"]
        cr = ", ".join(
            f"{a}/{b}: {x['p']:.3f}" if x["p"] is not None else f"{a}/{b}: -"
            for x in row["crossings"]
            for a, b in [x["sizes"]]
        )
        print(
            f"{name:10s} p_c = {f['p_c']:.4f} +- {f['p_c_err']:.4f}  "
            f"nu = {f['nu']:.3f} +- {f['nu_err']:.3f}  Q = {f['quality']:.2f}  "
            f"Q(lit) = {row['quality_at_literature']:.2f}  "
            f"nu(p_c lit) = {row['nu_at_literature_p_c']['nu']:.2f} "
            f"(Q {row['nu_at_literature_p_c']['quality']:.1f})  [{cr}]"
        )


if __name__ == "__main__":
    main()
