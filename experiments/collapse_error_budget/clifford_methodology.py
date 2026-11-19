#!/usr/bin/env python3
"""Clifford MIPT methodology figure from ``Delta`` only: crossing, collapse, map.

Everything comes from the mode measurement ``W(m)`` through the order
parameter ``Delta = N - log2(1 + W(m))`` of #346 (periodic brickwork of
random Clifford gates, ``Z`` measurements with probability ``p``,
``T = 2N``, ``N = 16, 32, 64, 128``, the values of ``p`` from 0.08 to
0.23, ``--p-max``; above, ``Delta`` of the large sizes is tiny and noisy).
One figure, three panels:

(a) ``Delta`` against ``p`` for every ``N`` (shades of blue, light for small
    ``N``) and the band ``p_c ± δp_c``: the curves cross there, the inset
    zooms on the crossing on a linear axis;
(b) the collapse against ``x = (p - p_c) N^(1/ν)`` at the optimum;
(c) the exact Houdayer-Hartmann quality ``Q`` over ``(p_c, ν)`` (red shades,
    log scale), the optimum ``Q_min`` with its error bars, the contour
    ``Q = 1.3 Q_min`` that defines them (Zabalo et al. 2020), and the
    Gullans & Huse value for the same circuit and the closest quantity (the
    entropy of a reference qubit).

The optimum is refitted on these points (grid then Nelder-Mead, as in
``pc_nu_plane.py``), and its error bars are the projected half-widths of the
region ``Q <= 1.3 Q_min``. A few seconds. Run from this folder:

    uv run python clifford_methodology.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from error_budget import GRID, fit  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pc_nu_plane import CLIFFORD_RUNS, FACTOR, clifford_delta, region  # noqa: E402

from QELM.evaluation.finite_size_scaling import Curve, hh_quality  # noqa: E402

HERE = Path(__file__).resolve().parent
DARK = "#08306b"  # optimum of this work
NU = r"$\nu$"
YLABEL = r"$\Delta = N - \log_2(1 + W(m))$"
# Gullans & Huse, PRL 125, 070606 (2020): entropy of a reference qubit
LIT = ("Gullans & Huse, PRL 2020\n(entropy of a reference qubit)",
       0.1598, 0.0005, 1.30, 0.05)  # fmt: skip
STYLE = {"ms": 3.2, "lw": 0.9, "elinewidth": 0.8, "capsize": 0}


def blues(n: int) -> list[Any]:
    """``n`` shades of blue, light (small ``N``) to dark (large ``N``)."""
    return [plt.get_cmap("Blues")(v) for v in np.linspace(0.38, 0.95, n)]


def q_map(
    curves: dict[int, Curve], row: dict[str, Any], n: int = 81
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Q`` on a grid that holds the 1.3 Q_min region and the literature point."""
    pc, nu = row["p_c"], row["nu"]
    hp = max(4 * row["p_c_err_1.3"], 1.6 * abs(LIT[1] - pc) + LIT[2], 0.002)
    hn = max(4 * row["nu_err_1.3"], 1.6 * abs(LIT[3] - nu) + LIT[4], 0.08)
    pcs = np.linspace(pc - hp, pc + hp, n)
    nus = np.linspace(max(nu - hn, 0.3), nu + hn, n)
    q = np.array([[hh_quality(curves, a, b) for a in pcs] for b in nus])
    return pcs, nus, q


def crossing_panel(ax: Any, curves: dict[int, Curve], pc: float, dpc: float) -> None:
    """(a) ``Delta(p)`` per size, the band ``p_c ± δp_c`` and a linear zoom."""
    sizes = sorted(curves)
    cols = blues(len(sizes))
    ins = ax.inset_axes([0.08, 0.08, 0.4, 0.4])
    lo, hi = pc - 0.02, pc + 0.02
    zoom = []
    for col, n in zip(cols, sizes, strict=True):
        p, y, e = curves[n]
        ax.errorbar(p, y, yerr=e, fmt="o-", color=col, **STYLE)
        m = (p >= lo) & (p <= hi)
        ins.errorbar(p[m], y[m], yerr=e[m], fmt="o-", color=col, **STYLE)
        zoom.append(y[m])
    for a in (ax, ins):
        a.axvspan(pc - dpc, pc + dpc, color="0.6", alpha=0.35, lw=0)
        a.axvline(pc, color="0.35", lw=0.8, ls="--")
    ys = np.concatenate(zoom)
    pad = 0.08 * np.ptp(ys)
    ins.set_xlim(lo, hi)
    ins.set_ylim(ys.min() - pad, ys.max() + pad)
    ins.set_title("zoom near $p_c$ (linear)", fontsize=8, pad=2)
    ins.tick_params(labelsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("$p$")
    ax.set_ylabel(YLABEL)
    ax.set_title(r"(a) $\Delta$ vs $p$: the sizes cross at $p_c$", fontsize=10.5)


def figure(curves: dict[int, Curve], row: dict[str, Any], out: Path) -> dict[str, Any]:
    """The three panels; returns the numbers shown."""
    pc, dpc, nu, dnu = row["p_c"], row["p_c_err_1.3"], row["nu"], row["nu_err_1.3"]
    qmin = row["S_min"]
    sizes = sorted(curves)
    cols = blues(len(sizes))
    ratios = {"width_ratios": [1, 1, 1.12]}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=ratios)

    crossing_panel(a, curves, pc, dpc)

    for col, n in zip(cols, sizes, strict=True):
        p, y, e = curves[n]
        b.errorbar((p - pc) * n ** (1 / nu), y, yerr=e, fmt="o-", color=col, **STYLE)
    b.axvline(0, color="0.35", lw=0.8, ls="--")
    b.set_yscale("log")
    b.set_xlabel(rf"$x = (p - p_c)\,N^{{1/\nu}}$,  $p_c$ = {pc:.4f}, {NU} = {nu:.2f}")
    b.set_ylabel(YLABEL)
    b.set_title(rf"(b) collapse at the optimum ($Q_{{min}}$ = {qmin:.1f})",
                fontsize=10.5)  # fmt: skip

    pcs, nus, q = q_map(curves, row)
    vmax = float(np.percentile(q[np.isfinite(q)], 98))
    im = c.pcolormesh(pcs, nus, q, cmap="Reds", shading="auto",
                      norm=LogNorm(vmin=qmin, vmax=vmax), rasterized=True)  # fmt: skip
    cb = fig.colorbar(im, ax=c, pad=0.02)
    cb.set_label("Houdayer-Hartmann quality $Q$ (log scale)")
    c.contour(pcs, nus, q, levels=[FACTOR * qmin], colors=["k"], linewidths=1.2)
    c.errorbar(pc, nu, xerr=dpc, yerr=dnu, fmt="o", ms=5, color=DARK, mec="white",
               mew=0.8, capsize=3, lw=1.3, zorder=4)  # fmt: skip
    c.errorbar(LIT[1], LIT[3], xerr=LIT[2], yerr=LIT[4], fmt="^", ms=5, color="k",
               mfc="white", capsize=3, lw=1.0, zorder=4)  # fmt: skip
    c.set_xlim(pcs[0], pcs[-1])
    c.set_ylim(nus[0], nus[-1])
    c.set_xlabel("$p_c$")
    c.set_ylabel(NU)
    c.set_title(r"(c) optimization map $Q(p_c, \nu)$", fontsize=10.5)
    c.ticklabel_format(axis="x", useOffset=False)

    for ax in (a, b, c):
        ax.tick_params(labelsize=9)
    handles: list[Any] = [Line2D([], [], color=col, marker="o", ms=3.2, lw=0.9)
                          for col in cols]  # fmt: skip
    labels = [f"N = {n}" for n in sizes]
    handles += [
        Patch(color="0.6", alpha=0.35),
        Line2D([], [], color=DARK, marker="o", ms=5, mec="white", ls="none"),
        Line2D([], [], color="k", lw=1.2),
        Line2D([], [], color="k", marker="^", mfc="white", ms=5, ls="none"),
    ]
    labels += [
        r"$p_c \pm \delta p_c$",
        rf"optimum, $Q_{{min}}$ = {qmin:.1f}"
        f"\n$p_c$ = {pc:.4f} ± {dpc:.4f}\n{NU} = {nu:.2f} ± {dnu:.2f}",
        f"$Q$ = {FACTOR} $Q_{{min}}$\n(defines the error bars)",
        f"{LIT[0]}\n$p_c$ = {LIT[1]} ± {LIT[2]}\n{NU} = {LIT[3]:.2f} ± {LIT[4]}",
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8.8, labelspacing=0.9)  # fmt: skip
    fig.suptitle("Clifford MIPT from the mode measurement W(m): periodic brickwork, "
                 "Z measurements, T = 2N", fontsize=12)  # fmt: skip
    fig.tight_layout()
    path = out / "clifford_methodology_delta.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"file": path.name, "observable": "Delta = N - log2(1 + W(m))",
            "sizes": sizes, "T": "2N", "p_c": pc, "p_c_err": dpc, "nu": nu,
            "nu_err": dnu, "Q_min": qmin, "Q_contour": FACTOR * qmin,
            "literature": LIT}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--p-max", type=float, default=0.23,
                    help="keep only p <= p_max (default 0.23)")  # fmt: skip
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    curves: dict[int, Curve] = {}
    for n, (p, y, e) in clifford_delta(CLIFFORD_RUNS).items():
        m = p <= a.p_max + 1e-9
        curves[n] = (p[m], y[m], e[m])
    best = fit(curves, None, GRID)
    wp, wn, _ = region(curves, False, best)
    row = {"p_c": best[0], "nu": best[1], "S_min": best[2],
           "p_c_err_1.3": wp, "nu_err_1.3": wn}  # fmt: skip
    summary = figure(curves, row, a.out)
    summary["p_max"] = a.p_max
    summary["points_per_N"] = {n: len(c[0]) for n, c in curves.items()}
    (a.out / "clifford_methodology_delta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
