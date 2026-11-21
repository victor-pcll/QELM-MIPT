#!/usr/bin/env python3
"""Figures of the no-readout runs (#378): crossing, collapse, Q map.

Reads ``runs/<model>_N*.json`` of ``run.py`` (periodic brickwork, ``Z``
measurement with probability ``p`` after every half-layer, no final
readout, ``T = 4N``). Every observable is an ``S_2`` of the references, read
from the Fourier weights ``W_A`` of the effect. One figure per circuit and
observable (Delta, S_2(N/2), I3), three panels, in the style of
``experiments/collapse_error_budget/clifford_methodology.py`` (PR #381):

(a) the observable against ``p`` for every ``N`` (shades of blue, light for
    small ``N``), the band ``p_c ± δp_c`` and a linear zoom on the crossing;
(b) the collapse against ``x = (p - p_c) N^(1/ν)`` at the optimum
    (``S_2(N/2) - S_2(p_c, N)`` for the half-chain entropy, which grows with
    ``N`` at ``p_c``);
(c) the exact Houdayer-Hartmann quality ``Q`` over ``(p_c, ν)`` (red, log
    scale, light near ``Q_min``), the optimum with the projected half-widths
    of the region ``Q <= 1.3 Q_min`` (Zabalo et al. 2020) as error bars, and
    the literature value for the same circuit.

Run from the repository root:

    uv run python experiments/pc_nu_no_readout/figures.py \
        --runs /mnt/project-files/pc-nu-no-readout/runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

from QELM.evaluation.finite_size_scaling import (  # noqa: E402
    Curve,
    hh_quality,
    subtract_at,
)

HERE = Path(__file__).resolve().parent
DARK = "#08306b"  # optimum of this work
NU = r"$\nu$"
FACTOR = 1.3
STYLE = {"ms": 3.2, "lw": 0.9, "elinewidth": 0.8, "capsize": 0}
# Gullans & Huse, PRX 10, 041020 (2020): crossing of I3 (pure initial
# state, t = 4L, L = 128-512) for p_c, collapse for nu.
GULLANS = ("Gullans & Huse, PRX 2020\n(I3, t = 4L, N = 128-512)",
           0.1593, 0.0005, 1.28, 0.02)  # fmt: skip
# Zabalo et al., PRB 101, 060301(R) (2020): Renyi-2 I3, t = 4L, L = 8-24.
ZABALO = ("Zabalo et al., PRB 2020\n(Rényi-2 I3, t = 4L, N = 8-24)",
          0.162, 0.003, 1.3, 0.1)  # fmt: skip
LABELS = {
    "delta": (r"$\Delta = S_2$(all references)", r"$\Delta$"),
    "s_half": (r"$S_2(N/2)$ of the references", r"$S_2(N/2)$"),
    "i3": (r"$I_3$ of the reference quarters (Rényi-2)", r"$I_3$"),
}


@dataclass(frozen=True)
class Case:
    """One figure: circuit, observable and the choices of the fit."""

    model: str
    key: str
    p_window: tuple[float, float] = (0.0, 1.0)
    n_min: int = 0
    log: bool = False
    subtract: bool = False
    inset: tuple[float, float, float, float] = (0.08, 0.08, 0.4, 0.4)
    note: str = ""


CASES = [
    # Above p = 0.21, Delta of the large sizes is tiny and noisy.
    Case("clifford", "delta", (0.0, 0.21), log=True),
    Case("clifford", "s_half", (0.10, 0.24), subtract=True,
         inset=(0.55, 0.52, 0.4, 0.4)),
    # Below about 0.15, I3 of the (nearly maximally mixed) references turns
    # back to 0 as p decreases: a second branch that does not scale with N.
    Case("clifford", "i3", (0.15, 1.0), inset=(0.55, 0.08, 0.4, 0.4)),
    Case("haar", "delta", log=True),
    Case("haar", "s_half", subtract=True, inset=(0.55, 0.52, 0.4, 0.4)),
    # N = 4 never crosses the larger sizes; I3 needs 4 | N, so N = 8, 12.
    Case("haar", "i3", n_min=8, inset=(0.55, 0.08, 0.4, 0.4),
         note=", N = 8 and 12 only"),
]  # fmt: skip


def load(runs: Path, case: Case) -> dict[int, Curve]:
    """``{N: (p, mean, sem)}`` of one observable at ``T = 4N``."""
    out: dict[int, Curve] = {}
    name = f"{case.key}_T4N"
    lo, hi = case.p_window
    for path in sorted(runs.glob(f"{case.model}_N*.json")):
        run = json.loads(path.read_text())
        n = int(run["n"])
        if n < case.n_min:
            continue
        rows = [r for r in run["per_p"]
                if name in r and lo - 1e-9 <= r["p"] <= hi + 1e-9]  # fmt: skip
        if len(rows) < 3:
            continue
        p = np.array([r["p"] for r in rows])
        y = np.array([r[name][0] for r in rows])
        e = np.array([r[name][1] for r in rows])
        keep = e > 0
        if case.log:
            keep &= y > 0
        out[n] = (p[keep], y[keep], e[keep])
    return dict(sorted(out.items()))


def fit(curves: dict[int, Curve], sub: bool) -> tuple[float, float, float]:
    """Minimum ``(p_c, nu, Q_min)``: grid, then Nelder-Mead."""
    p_all = np.concatenate([c[0] for c in curves.values()])
    pcs = np.linspace(max(p_all.min(), 0.10), min(p_all.max(), 0.30), 81)
    nus = np.linspace(0.6, 3.0, 49)
    q = np.array([[hh_quality(curves, a, b, sub) for b in nus] for a in pcs])
    i, j = np.unravel_index(int(np.argmin(q)), q.shape)
    res = minimize(
        lambda v: hh_quality(curves, float(v[0]), float(v[1]), sub),
        (pcs[i], nus[j]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-7},
    )
    return float(res.x[0]), float(res.x[1]), float(res.fun)


def region(
    curves: dict[int, Curve],
    sub: bool,
    best: tuple[float, float, float],
    n: int = 51,
) -> tuple[float, float]:
    """Projected half-widths of ``Q <= 1.3 Q_min``, grid refined to its extent."""
    half = (0.01, 0.4)
    wp = wn = 0.0
    for _ in range(5):
        pcs = np.linspace(best[0] - half[0], best[0] + half[0], n)
        nus = np.linspace(max(best[1] - half[1], 0.2), best[1] + half[1], n)
        q = np.array([[hh_quality(curves, a, b, sub) for b in nus] for a in pcs])
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
    curves: dict[int, Curve],
    sub: bool,
    row: dict[str, Any],
    lit: tuple[str, float, float, float, float],
    n: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Q`` on a grid that holds the 1.3 Q_min region and the literature point."""
    pc, nu = row["p_c"], row["nu"]
    hp = max(4 * row["p_c_err"], 1.3 * abs(lit[1] - pc) + lit[2], 0.002)
    hn = max(4 * row["nu_err"], 1.3 * abs(lit[3] - nu) + lit[4], 0.08)
    pcs = np.linspace(pc - hp, pc + hp, n)
    nus = np.linspace(max(nu - hn, 0.3), nu + hn, n)
    q = np.array([[hh_quality(curves, a, b, sub) for a in pcs] for b in nus])
    return pcs, nus, q


def figure(
    case: Case, curves: dict[int, Curve], row: dict[str, Any], out: Path
) -> Path:
    """The three panels of one circuit and observable."""
    label, short = LABELS[case.key]
    lit = GULLANS if case.model == "clifford" else ZABALO
    pc, dpc, nu, dnu, qmin = (row[k] for k in ("p_c", "p_c_err", "nu", "nu_err",
                                               "Q_min"))  # fmt: skip
    sizes = sorted(curves)
    cols = blues(len(sizes))
    ratios = {"width_ratios": [1, 1, 1.12]}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=ratios)

    shown = subtract_at(curves, pc) if case.subtract else curves
    ins = a.inset_axes(case.inset)
    lo, hi = pc - 0.02, pc + 0.02
    zoom = []
    for col, n in zip(cols, sizes, strict=True):
        p, y, e = curves[n]
        a.errorbar(p, y, yerr=e, fmt="o-", color=col, **STYLE)
        m = (p >= lo) & (p <= hi)
        ins.errorbar(p[m], y[m], yerr=e[m], fmt="o-", color=col, **STYLE)
        zoom.append(y[m])
        _, ys, _ = shown[n]
        b.errorbar((p - pc) * n ** (1 / nu), ys, yerr=e, fmt="o-", color=col,
                   **STYLE)  # fmt: skip
    for ax in (a, ins):
        ax.axvspan(pc - dpc, pc + dpc, color="0.6", alpha=0.35, lw=0)
        ax.axvline(pc, color="0.35", lw=0.8, ls="--")
    zs = np.concatenate(zoom)
    if zs.size:
        pad = 0.08 * np.ptp(zs) + 1e-12
        ins.set_ylim(zs.min() - pad, zs.max() + pad)
    ins.set_xlim(lo, hi)
    ins.set_title("zoom near $p_c$ (linear)", fontsize=8, pad=2)
    ins.tick_params(labelsize=7)
    if case.log:
        a.set_yscale("log")
        b.set_yscale("log")
    a.set_xlabel("$p$")
    a.set_ylabel(label)
    crossing = "the sizes cross at $p_c$" if not case.subtract else "grows with N"
    a.set_title(f"(a) {short} vs $p$: {crossing}", fontsize=10.5)
    b.axvline(0, color="0.35", lw=0.8, ls="--")
    b.set_xlabel(rf"$x = (p - p_c)\,N^{{1/\nu}}$,  $p_c$ = {pc:.4f}, {NU} = {nu:.2f}")
    b.set_ylabel(r"$S_2(N/2) - S_2(p_c, N)$" if case.subtract else label)
    b.set_title(rf"(b) collapse at the optimum ($Q_{{min}}$ = {qmin:.1f})",
                fontsize=10.5)  # fmt: skip

    pcs, nus, q = q_map(curves, case.subtract, row, lit)
    finite = q[np.isfinite(q)]
    vmax = max(float(np.percentile(finite, 98)), 1.01 * qmin)
    im = c.pcolormesh(pcs, nus, q, cmap="Reds", shading="auto",
                      norm=LogNorm(vmin=qmin, vmax=vmax), rasterized=True)  # fmt: skip
    cb = fig.colorbar(im, ax=c, pad=0.02)
    cb.set_label("Houdayer-Hartmann quality $Q$ (log scale)")
    c.contour(pcs, nus, q, levels=[FACTOR * qmin], colors=["k"], linewidths=1.2)
    c.errorbar(pc, nu, xerr=dpc, yerr=dnu, fmt="o", ms=5, color=DARK, mec="white",
               mew=0.8, capsize=3, lw=1.3, zorder=4)  # fmt: skip
    c.errorbar(lit[1], lit[3], xerr=lit[2], yerr=lit[4], fmt="^", ms=5, color="k",
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
        f"{lit[0]}\n$p_c$ = {lit[1]} ± {lit[2]}\n{NU} = {lit[3]:.2f} ± {lit[4]}",
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8.8, labelspacing=0.9)  # fmt: skip
    lo_p = min(cv[0].min() for cv in curves.values())
    hi_p = max(cv[0].max() for cv in curves.values())
    fig.suptitle(f"{case.model.capitalize()} MIPT without final readout, "
                 f"{short} from the Fourier weights $W_A$: periodic brickwork, "
                 f"Z measurements, T = 4N, {lo_p:.2f} ≤ p ≤ {hi_p:.2f}{case.note}",
                 fontsize=12)  # fmt: skip
    fig.tight_layout()
    path = out / f"{case.model}_{case.key}_T4N.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        curves = load(a.runs, case)
        best = fit(curves, case.subtract)
        wp, wn = region(curves, case.subtract, best)
        row = {"p_c": best[0], "nu": best[1], "Q_min": best[2], "p_c_err": wp,
               "nu_err": wn}  # fmt: skip
        path = figure(case, curves, row, a.out)
        summary = {
            "file": path.name,
            "model": case.model,
            "observable": case.key,
            "T": "4N",
            "sizes": sorted(curves),
            "p_range": [float(min(cv[0].min() for cv in curves.values())),
                        float(max(cv[0].max() for cv in curves.values()))],
            "subtract": case.subtract,
            **row,
            "Q_contour": FACTOR * best[2],
            "literature": GULLANS if case.model == "clifford" else ZABALO,
        }  # fmt: skip
        path.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps({k: summary[k] for k in ("file", "sizes", "p_c", "p_c_err",
                                                  "nu", "nu_err", "Q_min")}),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
