#!/usr/bin/env python3
"""Methodology figures of the no-readout runs (#378): one PNG per observable.

Data: the runs of ``experiments/pc_nu_no_readout/run.py`` (PR #380), periodic
brickwork with no final readout, read at ``T = 4N``. Every observable is a
Renyi-2 entropy of the references read from the Fourier weights ``W_A`` of
the effect, ``S_2(A) = |A| - log2(1 + W_A)``:

- ``delta``: ``S_2`` of all the references, ``Delta = N - log2(1 + W(m))``;
- ``s_half``: ``S_2(N/2)``, collapse of ``S_2(p, N) - S_2(p_c, N)``;
- ``i3``: Renyi-2 ``I3`` of the quarters;
- ``s_one``: ``S_2`` of one reference.

The fits are not redone: ``p_c``, ``nu``, ``Q_min`` and the half-widths of
the region ``Q <= 1.3 Q_min`` are read from the JSON of the analysis
(``--fits``, the ``provisional.json`` of #378). Only the map of panel (c) is
recomputed with the exact Houdayer-Hartmann quality. Each figure:

(a) observable vs ``p``, one blue shade and one marker per ``N``, band
    ``p_c ± δp_c``, linear zoom on the crossing;
(b) collapse at the optimum;
(c) ``Q(p_c, nu)`` in red, light near ``Q_min`` and dark for high ``Q``, the
    contour ``Q = 1.3 Q_min``, the optimum with its bars, the literature
    value for the same circuit (open triangle).

When the region touches the edge of the search grid (no finite optimum),
panel (b) says so and panel (c) shows the search grid. The runs are not in
git; pass their folder. From the repository root:

    uv run python experiments/collapse_error_budget/no_readout_methodology.py \\
        --runs <runs folder> --fits <provisional.json> --out <folder>
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

from QELM.evaluation.finite_size_scaling import (  # noqa: E402
    Curve,
    hh_quality,
    subtract_at,
)

RATIO = 4
FACTOR = 1.3
BLUE = "#08306b"
MARKERS = ["o"] * 6  # same simple marker for every N, as clifford_methodology
STYLE = {"ms": 3.4, "lw": 0.9, "elinewidth": 0.8, "capsize": 0}
SEARCH = ((0.10, 0.26), (0.6, 3.0))  # search grid of the analysis
NU = r"$\nu$"

OBS: dict[str, dict[str, Any]] = {
    "delta": {"name": r"$\Delta$", "plain": "Delta",
              "label": r"$\Delta = N - \log_2(1 + W(m))$", "sub": False},
    "s_half": {"name": r"$S_2(N/2)$", "plain": "S_2(N/2)",
               "label": r"$S_2(N/2) = N/2 - \log_2(1 + W_{N/2})$", "sub": True},
    "i3": {"name": r"$I_3$", "plain": "I3",
           "label": r"Rényi-2 $I_3$ of the quarters (from $W_A$)", "sub": False},
    "s_one": {"name": r"$S_2(1)$", "plain": "S_2(one reference)",
              "label": r"$S_2$ of one reference $= 1 - \log_2(1 + W_1)$",
              "sub": False},
}  # fmt: skip
GH_PRL = ("Gullans & Huse, PRL 2020\n(entropy of a reference qubit)",
          0.1598, 0.0005, 1.30, 0.05)  # fmt: skip
GH_PRX = ("Gullans & Huse, PRX 2020\n(I3 of quarters, S(ρ)/L)",
          0.1593, 0.0005, 1.28, 0.02)  # fmt: skip
ZABALO = ("Zabalo et al., PRB 2020\n(Rényi-2 I3 of quarters, T = 4N)",
          0.162, 0.003, 1.3, 0.1)  # fmt: skip
LIT = {
    "clifford": {"delta": GH_PRL, "s_one": GH_PRL, "i3": GH_PRX, "s_half": GH_PRX},
    "haar": dict.fromkeys(OBS, ZABALO),
}
TITLE = {"clifford": "Clifford", "haar": "Haar"}


def curves(runs: Path, model: str, obs: str, sizes: list[int]) -> dict[int, Curve]:
    """``{N: (p, mean, SEM)}`` at ``T = 4N`` for the sizes of the fit."""
    out: dict[int, Curve] = {}
    key = f"{obs}_T{RATIO}N"
    for n in sizes:
        r = json.loads((runs / f"{model}_N{n}.json").read_text())
        rows = [x for x in r["per_p"] if key in x]
        p = np.array([x["p"] for x in rows])
        y = np.array([x[key][0] for x in rows])
        e = np.array([x[key][1] for x in rows])
        keep = e > 0
        out[n] = (p[keep], y[keep], e[keep])
    return out


def blues(n: int) -> list[Any]:
    """``n`` shades of blue, light (small ``N``) to dark (large ``N``)."""
    return [plt.get_cmap("Blues")(v) for v in np.linspace(0.4, 0.95, n)]


def free_corner(ax: Any, c: dict[int, Curve]) -> list[float]:
    """Inset box in the corner of ``ax`` that covers the fewest data points."""
    ax.figure.canvas.draw()
    pts = np.concatenate([np.c_[p, y] for p, y, _ in c.values()])
    disp = ax.transData.transform(pts)
    frac = ax.transAxes.inverted().transform(disp)
    boxes = [[0.08, 0.08], [0.56, 0.08], [0.08, 0.54], [0.56, 0.54]]
    count = [
        np.sum(
            (frac[:, 0] > x - 0.04)
            & (frac[:, 0] < x + 0.44)
            & (frac[:, 1] > y - 0.04)
            & (frac[:, 1] < y + 0.44)
        )
        for x, y in boxes
    ]
    x, y = boxes[int(np.argmin(count))]
    return [x, y, 0.38, 0.38]


def log_floor(ax: Any, c: dict[int, Curve]) -> None:
    """Log axis bottom at the smallest value above its own error bar."""
    sig = np.concatenate([y[y > e] for _, y, e in c.values()])
    if len(sig):
        ax.set_ylim(bottom=0.3 * sig.min())


def panel_a(ax: Any, c: dict[int, Curve], o: dict[str, Any], log: bool,
            fit: dict[str, Any] | None) -> None:  # fmt: skip
    """Observable vs ``p``, band ``p_c ± δp_c`` and zoom on the crossing."""
    sizes = sorted(c)
    cols = blues(len(sizes))
    for k, (col, n) in enumerate(zip(cols, sizes, strict=True)):
        p, y, e = c[n]
        ax.errorbar(p, y, yerr=e, fmt=MARKERS[k] + "-", color=col, **STYLE)
    if log:
        ax.set_yscale("log")
        log_floor(ax, c)
    ax.set_xlabel("$p$")
    ax.set_ylabel(o["label"])
    if fit is None:
        ax.set_title(f"(a) {o['name']} vs $p$: no crossing found", fontsize=10.5)
        return
    pc, dpc = fit["p_c"], fit["region_1.3Q"][0]
    in_range = min(c[n][0][0] for n in sizes) <= pc <= max(c[n][0][-1] for n in sizes)
    if not in_range:
        title = f"no crossing in the $p$ range ($p_c$ = {pc:.3f} extrapolated)"
    elif o["sub"]:
        title = "no crossing, grows with $N$"
    else:
        title = "crossing at $p_c$"
    ax.set_title(f"(a) {o['name']} vs $p$: {title}", fontsize=10.5)
    if not in_range:
        return
    ins = ax.inset_axes(free_corner(ax, c))
    for a in (ax, ins):
        a.axvspan(pc - dpc, pc + dpc, color="0.6", alpha=0.35, lw=0)
        a.axvline(pc, color="0.35", lw=0.8, ls="--")
    lo, hi = pc - max(0.025, 2 * dpc), pc + max(0.025, 2 * dpc)
    zoom = []
    for k, (col, n) in enumerate(zip(cols, sizes, strict=True)):
        p, y, e = c[n]
        m = (p >= lo) & (p <= hi)
        ins.errorbar(p[m], y[m], yerr=e[m], fmt=MARKERS[k] + "-", color=col, **STYLE)
        zoom.append(y[m])
    ys = np.concatenate(zoom)
    if len(ys):
        pad = 0.08 * np.ptp(ys) + 1e-12
        ins.set_ylim(ys.min() - pad, ys.max() + pad)
    ins.set_xlim(lo, hi)
    ins.set_title("zoom near $p_c$ (linear)", fontsize=8, pad=2)
    ins.tick_params(labelsize=7)


def panel_b(ax: Any, c: dict[int, Curve], o: dict[str, Any], log: bool,
            fit: dict[str, Any] | None) -> None:  # fmt: skip
    """Collapse at the optimum, or a note when there is none."""
    if fit is None:
        ax.text(0.5, 0.5, "No finite optimum:\nthe region Q ≤ 1.3 Q_min\ntouches the "
                "edge of the search grid,\nso there is no collapse to show.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(b) collapse: none", fontsize=10.5)
        return
    pc, nu = fit["p_c"], fit["nu"]
    shown = subtract_at(c, pc) if o["sub"] else c
    sizes = sorted(c)
    for k, (col, n) in enumerate(zip(blues(len(sizes)), sizes, strict=True)):
        p, y, e = shown[n]
        ax.errorbar((p - pc) * n ** (1 / nu), y, yerr=e, fmt=MARKERS[k] + "-",
                    color=col, **STYLE)  # fmt: skip
    ax.axvline(0, color="0.35", lw=0.8, ls="--")
    if log and not o["sub"]:
        ax.set_yscale("log")
        log_floor(ax, c)
    ax.set_xlabel(rf"$x = (p - p_c)\,N^{{1/\nu}}$,  $p_c$ = {pc:.4f}, {NU} = {nu:.2f}")
    ax.set_ylabel(r"$S_2(p, N) - S_2(p_c, N)$" if o["sub"] else o["label"])
    ax.set_title(rf"(b) collapse at the optimum ($Q_{{min}}$ = {fit['Q_min']:.1f})",
                 fontsize=10.5)  # fmt: skip


def panel_c(ax: Any, fig: Any, c: dict[int, Curve], o: dict[str, Any],
            fit: dict[str, Any] | None, lit: tuple, n: int = 81) -> bool:  # fmt: skip
    """Quality map; returns whether the literature point is on the map."""
    if fit is None:
        pcs, nus = np.linspace(*SEARCH[0], n), np.linspace(*SEARCH[1], n)
    else:
        pc, nu = fit["p_c"], fit["nu"]
        wp, wn = fit["region_1.3Q"]
        hp, hn = max(4 * wp, 0.002), max(4 * wn, 0.08)
        if abs(lit[1] - pc) < 8 * wp + lit[2] and abs(lit[3] - nu) < 8 * wn + lit[4]:
            hp = max(hp, 1.4 * abs(lit[1] - pc) + lit[2])
            hn = max(hn, 1.4 * abs(lit[3] - nu) + lit[4])
        pcs = np.linspace(pc - hp, pc + hp, n)
        nus = np.linspace(max(nu - hn, 0.2), nu + hn, n)
    q = np.array([[hh_quality(c, a, b, o["sub"]) for a in pcs] for b in nus])
    good = q[np.isfinite(q)]
    qmin = fit["Q_min"] if fit is not None else float(good.min())
    im = ax.pcolormesh(pcs, nus, np.where(np.isfinite(q), q, np.nan), cmap="Reds",
                       shading="auto", rasterized=True,
                       norm=LogNorm(vmin=min(qmin, good.min()),
                                    vmax=float(np.percentile(good, 98))))  # fmt: skip
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Houdayer-Hartmann quality $Q$ (log scale)")
    if fit is not None:
        ax.contour(pcs, nus, q, levels=[FACTOR * qmin], colors=["k"], linewidths=1.2)
        ax.errorbar(fit["p_c"], fit["nu"], xerr=fit["region_1.3Q"][0],
                    yerr=fit["region_1.3Q"][1], fmt="o", ms=5, color=BLUE,
                    mec="white", mew=0.8, capsize=3, lw=1.3, zorder=4)  # fmt: skip
    on_map = pcs[0] <= lit[1] <= pcs[-1] and nus[0] <= lit[3] <= nus[-1]
    if on_map:
        ax.errorbar(lit[1], lit[3], xerr=lit[2], yerr=lit[4], fmt="^", ms=5,
                    color="k", mfc="white", capsize=3, lw=1.0, zorder=4)  # fmt: skip
    ax.set_xlim(pcs[0], pcs[-1])
    ax.set_ylim(nus[0], nus[-1])
    ax.set_xlabel("$p_c$")
    ax.set_ylabel(NU)
    title = r"(c) optimization map $Q(p_c, \nu)$"
    ax.set_title(title + (" (search grid)" if fit is None else ""), fontsize=10.5)
    ax.ticklabel_format(axis="x", useOffset=False)
    return on_map


def figure(model: str, obs: str, fit: dict[str, Any], runs: Path, out: Path) -> str:
    """One PNG with the three panels of one observable."""
    o = OBS[obs]
    c = curves(runs, model, obs, fit["sizes"])
    ok = not fit["region_touches_edge"]
    good = fit if ok else None
    allv = np.concatenate([y for _, y, _ in c.values()])
    log = bool(np.all(allv > 0) and allv.max() / allv.min() > 30)
    lit = LIT[model][obs]
    ratios = {"width_ratios": [1, 1, 1.12]}
    fig, (a, b, cc) = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw=ratios)
    panel_a(a, c, o, log, good)
    panel_b(b, c, o, log, good)
    on_map = panel_c(cc, fig, c, o, good, lit)
    for ax in (a, b, cc):
        ax.tick_params(labelsize=9)

    sizes = sorted(c)
    blank = Line2D([], [], ls="none", marker="")
    handles: list[Any] = []
    labels: list[str] = []
    for k, (col, n) in enumerate(zip(blues(len(sizes)), sizes, strict=True)):
        handles.append(Line2D([], [], color=col, marker=MARKERS[k], ms=3.4, lw=0.9))
        labels.append(f"N = {n}")
    if ok:
        pc, nu, (wp, wn) = fit["p_c"], fit["nu"], fit["region_1.3Q"]
        handles += [
            Patch(color="0.6", alpha=0.35),
            Line2D([], [], color=BLUE, marker="o", ms=5, mec="white", ls="none"),
            Line2D([], [], color="k", lw=1.2),
        ]
        labels += [
            r"$p_c \pm \delta p_c$",
            f"optimum ({o['plain']}), $Q_{{min}}$ = {fit['Q_min']:.1f}"
            f"\n$p_c$ = {pc:.4f} ± {wp:.4f}\n{NU} = {nu:.2f} ± {wn:.2f}",
            f"$Q$ = {FACTOR} $Q_{{min}}$\n(defines the error bars)",
        ]
    else:
        labels.append(f"no finite optimum\n(Q_min = {fit['Q_min']:.0f} at the edge)")
        handles.append(blank)
    handles += [Line2D([], [], color="k", marker="^", mfc="white", ms=5, ls="none")]
    labels += [
        f"{lit[0]}\n$p_c$ = {lit[1]} ± {lit[2]}\n{NU} = {lit[3]:.2f} ± {lit[4]}"
        + ("" if on_map else "\n(outside the map)"),
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8.8, labelspacing=0.9)  # fmt: skip
    fig.suptitle(f"{TITLE[model]} MIPT without final readout, from the Fourier weights "
                 f"W_A: periodic brickwork, Z measurements, T = 4N, N = "
                 f"{', '.join(map(str, sizes))}", fontsize=12)  # fmt: skip
    fig.tight_layout()
    name = f"no_readout_{model}_{obs}.png"
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="e.g. clifford/delta")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fits = json.loads(a.fits.read_text())
    for key, fit in fits.items():
        if a.only and key not in a.only:
            continue
        model, obs = key.split("/")
        print(figure(model, obs, fit, a.runs, a.out), flush=True)


if __name__ == "__main__":
    main()
