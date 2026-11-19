#!/usr/bin/env python3
"""Clifford figures of the no-readout runs (#378): crossing, collapse, Q map.

Reads ``runs/clifford_N*.json`` of ``run.py`` (periodic Clifford
brickwork, ``Z`` measurement with probability ``p`` after every half-layer,
no final readout, ``T = 4N``, ``N = 32, 64, 128, 256``). Every observable
comes from the Fourier weights ``W_A`` of the effect. One figure per
observable, three panels, in the style of
``experiments/collapse_error_budget/clifford_methodology.py`` (PR #381):

(a) the observable against ``p`` for every ``N`` (shades of blue, light for
    small ``N``), the band ``p_c ± δp_c`` and a linear zoom on the crossing;
(b) the collapse against ``x = (p - p_c) N^(1/ν)`` at the optimum;
(c) the exact Houdayer-Hartmann quality ``Q`` over ``(p_c, ν)`` (red, log
    scale, light near ``Q_min``), the optimum with the projected half-widths
    of the region ``Q <= 1.3 Q_min`` (Zabalo et al. 2020) as error bars, and
    the Gullans & Huse value for the same circuit.

Run from the repository root:

    uv run python experiments/pc_nu_no_readout/clifford_figures.py \
        --runs /mnt/project-files/pc-nu-no-readout/runs
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
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from QELM.evaluation.finite_size_scaling import Curve, hh_quality  # noqa: E402

HERE = Path(__file__).resolve().parent
DARK = "#08306b"  # optimum of this work
NU = r"$\nu$"
FACTOR = 1.3
STYLE = {"ms": 3.2, "lw": 0.9, "elinewidth": 0.8, "capsize": 0}
# Gullans & Huse, PRX 10, 041020 (2020): crossing of I3 (pure initial
# state, t = 4L, L = 128-512) for p_c, collapse for nu.
LIT = ("Gullans & Huse, PRX 2020\n(I3, t = 4L, N = 128-512)",
       0.1593, 0.0005, 1.28, 0.02)  # fmt: skip
OBSERVABLES = {
    "delta": {
        "label": r"$\Delta = S_2$(all references) $= N - \log_2(1 + W)$",
        "short": r"$\Delta$",
        "p_min": 0.0,
        "p_max": 0.21,  # above, Delta of the large sizes is tiny and noisy
        "log": True,
        "inset": [0.08, 0.08, 0.4, 0.4],
    },
    "i3": {
        "label": r"$I_3$ of the reference quarters (Rényi-2, from $W_A$)",
        "short": r"$I_3$",
        # Below 0.14, I3 of the (nearly maximally mixed) references turns back
        # to 0 as p decreases: a second branch that does not scale with N.
        "p_min": 0.14,
        "p_max": 1.0,
        "log": False,
        "inset": [0.55, 0.08, 0.4, 0.4],
    },
}


def load(
    runs: Path, key: str, ratio: int, p_min: float, p_max: float
) -> dict[int, Curve]:
    """``{N: (p, mean, sem)}`` of one observable at ``T = ratio N``."""
    out: dict[int, Curve] = {}
    for path in sorted(runs.glob("clifford_N*.json")):
        run = json.loads(path.read_text())
        name = f"{key}_T{ratio}N"
        rows = [
            r
            for r in run["per_p"]
            if name in r and p_min - 1e-9 <= r["p"] <= p_max + 1e-9
        ]
        p = np.array([r["p"] for r in rows])
        y = np.array([r[name][0] for r in rows])
        e = np.array([r[name][1] for r in rows])
        keep = e > 0
        if key == "delta":
            keep &= y > 0
        out[int(run["n"])] = (p[keep], y[keep], e[keep])
    return dict(sorted(out.items()))


def fit(curves: dict[int, Curve]) -> tuple[float, float, float]:
    """Minimum ``(p_c, nu, Q_min)``: grid, then Nelder-Mead."""
    pcs, nus = np.linspace(0.10, 0.24, 57), np.linspace(0.6, 3.0, 49)
    q = np.array([[hh_quality(curves, a, b) for b in nus] for a in pcs])
    i, j = np.unravel_index(int(np.argmin(q)), q.shape)
    res = minimize(
        lambda v: hh_quality(curves, float(v[0]), float(v[1])),
        (pcs[i], nus[j]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-7},
    )
    return float(res.x[0]), float(res.x[1]), float(res.fun)


def region(
    curves: dict[int, Curve], best: tuple[float, float, float], n: int = 51
) -> tuple[float, float]:
    """Projected half-widths of ``Q <= 1.3 Q_min``, grid refined to its extent."""
    half = (0.01, 0.4)
    wp = wn = 0.0
    for _ in range(5):
        pcs = np.linspace(best[0] - half[0], best[0] + half[0], n)
        nus = np.linspace(max(best[1] - half[1], 0.2), best[1] + half[1], n)
        q = np.array([[hh_quality(curves, a, b) for b in nus] for a in pcs])
        ii, jj = np.nonzero(q <= FACTOR * best[2])
        wp, wn = float(np.ptp(pcs[ii]) / 2), float(np.ptp(nus[jj]) / 2)
        edge = ii.min() == 0 or ii.max() == n - 1 or jj.min() == 0 or jj.max() == n - 1
        if edge:
            half = (2 * half[0], 2 * half[1])
        elif wp < half[0] / 6 or wn < half[1] / 6:
            half = (max(3 * wp, 4 * half[0] / n), max(3 * wn, 4 * half[1] / n))
        else:
            break
    return wp, wn


def blues(n: int) -> list[Any]:
    """``n`` shades of blue, light (small ``N``) to dark (large ``N``)."""
    return [plt.get_cmap("Blues")(v) for v in np.linspace(0.38, 0.95, n)]


def q_map(
    curves: dict[int, Curve], row: dict[str, Any], n: int = 81
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Q`` on a grid that holds the 1.3 Q_min region and the literature point."""
    pc, nu = row["p_c"], row["nu"]
    hp = max(4 * row["p_c_err"], 1.6 * abs(LIT[1] - pc) + LIT[2], 0.002)
    hn = max(4 * row["nu_err"], 1.6 * abs(LIT[3] - nu) + LIT[4], 0.08)
    pcs = np.linspace(pc - hp, pc + hp, n)
    nus = np.linspace(max(nu - hn, 0.3), nu + hn, n)
    q = np.array([[hh_quality(curves, a, b) for a in pcs] for b in nus])
    return pcs, nus, q


def figure(key: str, curves: dict[int, Curve], row: dict[str, Any], out: Path) -> Path:
    """The three panels of one observable."""
    spec = OBSERVABLES[key]
    pc, dpc, nu, dnu, qmin = (row[k] for k in ("p_c", "p_c_err", "nu", "nu_err",
                                               "Q_min"))  # fmt: skip
    sizes = sorted(curves)
    cols = blues(len(sizes))
    ratios = {"width_ratios": [1, 1, 1.12]}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=ratios)

    ins = a.inset_axes(spec["inset"])
    lo, hi = pc - 0.02, pc + 0.02
    zoom = []
    for col, n in zip(cols, sizes, strict=True):
        p, y, e = curves[n]
        a.errorbar(p, y, yerr=e, fmt="o-", color=col, **STYLE)
        m = (p >= lo) & (p <= hi)
        ins.errorbar(p[m], y[m], yerr=e[m], fmt="o-", color=col, **STYLE)
        zoom.append(y[m])
        b.errorbar((p - pc) * n ** (1 / nu), y, yerr=e, fmt="o-", color=col, **STYLE)
    for ax in (a, ins):
        ax.axvspan(pc - dpc, pc + dpc, color="0.6", alpha=0.35, lw=0)
        ax.axvline(pc, color="0.35", lw=0.8, ls="--")
    ys = np.concatenate(zoom)
    pad = 0.08 * np.ptp(ys)
    ins.set_xlim(lo, hi)
    ins.set_ylim(ys.min() - pad, ys.max() + pad)
    ins.set_title("zoom near $p_c$ (linear)", fontsize=8, pad=2)
    ins.tick_params(labelsize=7)
    if spec["log"]:
        a.set_yscale("log")
        b.set_yscale("log")
    a.set_xlabel("$p$")
    a.set_ylabel(spec["label"])
    a.set_title(f"(a) {spec['short']} vs $p$: the sizes cross at $p_c$",
                fontsize=10.5)  # fmt: skip
    b.axvline(0, color="0.35", lw=0.8, ls="--")
    b.set_xlabel(rf"$x = (p - p_c)\,N^{{1/\nu}}$,  $p_c$ = {pc:.4f}, {NU} = {nu:.2f}")
    b.set_ylabel(spec["label"])
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
    fig.suptitle("Clifford MIPT without final readout, from the Fourier weights "
                 "$W_A$: periodic brickwork, Z measurements, T = 4N",
                 fontsize=12)  # fmt: skip
    fig.tight_layout()
    path = out / f"clifford_{key}_T4N.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for key, spec in OBSERVABLES.items():
        curves = load(a.runs, key, 4, spec["p_min"], spec["p_max"])
        best = fit(curves)
        wp, wn = region(curves, best)
        row = {"p_c": best[0], "nu": best[1], "Q_min": best[2], "p_c_err": wp,
               "nu_err": wn}  # fmt: skip
        path = figure(key, curves, row, a.out)
        summary = {
            "file": path.name,
            "observable": key,
            "T": "4N",
            "sizes": sorted(curves),
            "p_range": [float(min(c[0].min() for c in curves.values())),
                        float(max(c[0].max() for c in curves.values()))],
            **row,
            "Q_contour": FACTOR * best[2],
            "literature": LIT,
        }  # fmt: skip
        path.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
