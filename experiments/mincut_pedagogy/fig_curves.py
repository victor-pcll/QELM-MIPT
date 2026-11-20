"""Figures of the mean min cut against p and of the SRN comparison (issue #383).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_curves.py figures/mincut_data.json figures
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
kw = dict(ms=3.2, lw=0.9, elinewidth=0.8, capsize=0)


def blues(k):
    return plt.get_cmap("Blues")(np.linspace(0.38, 0.95, k))


fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
for row, key in enumerate(("nofinal", "final")):
    final = key == "final"
    mc = D["mincut"][key]
    ns = sorted(int(k) for k in mc)
    col = dict(zip(ns, blues(len(ns))))
    # (a) anatomy at N = 4
    ax = axes[row, 0]
    n = 4
    T = 2 * n
    r = mc[str(n)]
    p = np.array([x["p"] for x in r])
    ax.errorbar(
        p,
        [x["mean"] for x in r],
        [x["sem"] for x in r],
        fmt="o-",
        color=col[4],
        label=None,
        **kw,
    )
    rec = (p * n * T) if not final else (n + p * n * (T - 1))
    ax.plot(p, rec, "--", color="#2a7f62", lw=1.0)
    ax.plot(p, 2 * n - p * n, "-.", color="#c0392b", lw=1.0)
    ax.axhline(2 * n, color="0.6", lw=0.8, ls=":")
    ex = D["exact"][key][str(n)]
    ax.errorbar(
        [x["p"] for x in ex],
        [x["log2rank"] for x in ex],
        [x["sem"] for x in ex],
        fmt="s",
        mfc="white",
        color="#08306b",
        ms=5,
        lw=0,
        elinewidth=0.8,
    )
    ax.set_ylim(-0.3 if not final else 3.5, 8.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("measurement probability $p$")
    ax.set_ylabel(r"$\langle C\rangle$  (bits)")
    ax.set_title(
        ("no final readout" if not final else "with final readout")
        + ": anatomy of the cut, $N = 4$, $T = 8$",
        fontsize=10.5,
    )
    # (b) C / N, all N
    ax = axes[row, 1]
    for n in ns:
        r = mc[str(n)]
        ax.errorbar(
            [x["p"] for x in r],
            [x["mean"] / n for x in r],
            [x["sem"] / n for x in r],
            fmt="o-",
            color=col[n],
            **kw,
        )
    for n, mk in zip((3, 4, 5), ("v", "s", "^")):
        ex = D["exact"][key][str(n)]
        ax.plot(
            [x["p"] for x in ex],
            [x["log2rank"] / n for x in ex],
            mk,
            mfc="white",
            color="0.25",
            ms=4.5,
            lw=0,
        )
    pp = np.linspace(0, 1, 50)
    ax.plot(pp, 2 - pp, "-.", color="#c0392b", lw=1.0)
    ax.set_ylim(-0.05, 2.08)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("measurement probability $p$")
    ax.set_ylabel(r"$\langle C\rangle / N$  (bits per qubit)")
    ax.set_title(
        r"$\langle C\rangle/N$ against $p$, $T = 2N$: sizes do not cross", fontsize=10.5
    )
    # (c) where is the cut
    ax = axes[row, 2]
    for n, ls in ((4, "-"), (32, "--")):
        r = mc[str(n)]
        p = [x["p"] for x in r]
        ax.plot(
            p, [x["records"] for x in r], ls, color="#2a7f62", marker="o", ms=3, lw=0.9
        )
        ax.plot(
            p, [x["input"] for x in r], ls, color="#c0392b", marker="o", ms=3, lw=0.9
        )
        ax.plot(
            p, [x["inside"] for x in r], ls, color="#6a3d9a", marker="o", ms=3, lw=0.9
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("measurement probability $p$")
    ax.set_ylabel("fraction of masks")
    ax.set_title(
        "which cut is the cheapest? (solid $N = 4$, dashed $N = 32$)", fontsize=10.5
    )
for ax, lab in zip(axes.flat, "abcdef"):
    ax.text(
        -0.13, 1.04, f"({lab})", transform=ax.transAxes, fontsize=11, fontweight="bold"
    )
ns = sorted(int(k) for k in D["mincut"]["nofinal"])
col = blues(len(ns))
h = [
    Line2D([], [], color=c, marker="o", ms=3.2, lw=0.9, label=f"$N = {n}$ (min cut)")
    for n, c in zip(ns, col)
]
h += [
    Line2D(
        [],
        [],
        color="0.25",
        marker=m,
        mfc="white",
        lw=0,
        label=f"exact Haar rank, $N = {n}$: " + r"$\langle\log_2 \mathrm{rank}\rangle$",
    )
    for n, m in zip((3, 4, 5), ("v", "s", "^"))
]
h += [
    Line2D(
        [],
        [],
        color="#2a7f62",
        ls="--",
        label="record bound: $K$ (no readout)\nor $N + K_{\\rm eff}$ (readout)",
    ),
    Line2D(
        [],
        [],
        color="#c0392b",
        ls="-.",
        label="input bound $2N - n_1$\n(per qubit: $2 - p$)",
    ),
    Line2D([], [], color="0.6", ls=":", label="dense maximum $2N$ (rank $4^N$)"),
    Line2D([], [], color="#2a7f62", marker="o", ms=3, label="(c, f) cut = all records"),
    Line2D(
        [],
        [],
        color="#c0392b",
        marker="o",
        ms=3,
        label="(c, f) cut = just after layer-1 meas.",
    ),
    Line2D(
        [],
        [],
        color="#6a3d9a",
        marker="o",
        ms=3,
        label="(c, f) cut inside the circuit\n(cheaper than both)",
    ),
]
fig.legend(
    handles=h,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=8.8,
    labelspacing=0.9,
)
fig.suptitle(
    "Min cut $C$ of the space-time graph (rank $= 2^C$ for Haar), open brickwork, "
    "Z measurements after each layer, mean over masks; "
    "top: no final readout, bottom: final Z readout",
    fontsize=11.5,
)
fig.tight_layout()
fig.savefig(f"{OUT}/mincut_vs_p.png", dpi=200, bbox_inches="tight")

# ---------------- Skinner comparison ----------------
fig, axes = plt.subplots(
    1, 3, figsize=(16.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.12])
)
for ax, key, title in (
    (axes[0], "half", "SRN geometry: a measurement slot after each half-layer"),
    (axes[1], "full", "repository geometry: one slot after each full layer"),
):
    s = D["s0"][key]
    ns = sorted(int(k) for k in s)
    col = blues(len(ns))
    for n, c in zip(ns, col):
        r = s[str(n)]
        ax.errorbar(
            [x["p"] for x in r],
            [x["mean"] / n for x in r],
            [x["sem"] / n for x in r],
            fmt="o-",
            color=c,
            label=f"$N = {n}$",
            **kw,
        )
    if key == "half":
        ax.axvline(0.5, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel("measurement probability $p$")
    ax.set_ylabel(r"$\langle S_0(A)\rangle / N$, $A$ = left half")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
ax = axes[2]
mc = D["mincut"]["final"]
ns = sorted(int(k) for k in mc)
col = blues(len(ns))
for n, c in zip(ns, col):
    r = mc[str(n)]
    ax.errorbar(
        [x["p"] for x in r],
        [x["mean"] / n - 1 for x in r],
        [x["sem"] / n for x in r],
        fmt="o-",
        color=c,
        **kw,
    )
ax.set_xlabel("measurement probability $p$")
ax.set_ylabel(r"$\langle C\rangle / N - 1$  (bits per qubit beyond the readout)")
ax.set_title("our operator min cut, final readout, $T = 2N$", fontsize=10.5)
ax.set_xlim(-0.02, 1.02)
for a, lab in zip(axes, "abc"):
    a.text(
        -0.13, 1.04, f"({lab})", transform=a.transAxes, fontsize=11, fontweight="bold"
    )
ns0 = sorted(int(k) for k in D["s0"]["half"])
col0 = blues(len(ns0))
h = [
    Line2D([], [], color=c, marker="o", ms=3.2, lw=0.9, label=f"$N = {n}$ (a, b)")
    for n, c in zip(ns0, col0)
]
h += [
    Line2D([], [], color=c, marker="o", ms=3.2, lw=0.9, label=f"$N = {n}$ (c)")
    for n, c in zip(ns, col)
]
h += [
    Line2D(
        [],
        [],
        color="0.35",
        ls="--",
        label="bond percolation $p_c = 1/2$\n(SRN 2019, $\\nu = 4/3$)",
    )
]
fig.legend(
    handles=h,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=8.8,
    labelspacing=0.75,
)
fig.suptitle(
    "Same tool, different graph: Hartley entropy $S_0$ of a half chain "
    "(Skinner, Ruhman, Nahum 2019) against the effect-space min cut, mean over masks",
    fontsize=11.5,
)
fig.tight_layout()
fig.savefig(f"{OUT}/skinner_comparison.png", dpi=200, bbox_inches="tight")
