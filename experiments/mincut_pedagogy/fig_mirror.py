"""Two mirror figures: rank against the min cut, then E_circuit against C (#383).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_mirror.py figures/rank_data.json \
        figures/ecircuit_data.json figures
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

R = json.load(open(sys.argv[1]))
E = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
NS = [3, 4, 5]
COL = dict(zip(NS, plt.get_cmap("Blues")(np.linspace(0.45, 0.95, 3))))
KW = dict(ms=3.2, lw=0.9, elinewidth=0.8, capsize=0)
SETTINGS = [
    ("nofinal_T=2N", "nofinal_sphere_T=2N", "no final readout, $T = 2N$, dense inputs"),
    ("final_T=N", "final_xz_T=N", "final readout, $T = N$, XZ inputs"),
]


def mean_sem(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


# ---------------- Figure A: algebra of the circuit vs space-time geometry ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
for ax, (rkey, _, title) in zip(axes, SETTINGS):
    for n in NS:
        rows = R[rkey][str(n)]
        p = np.array([r["p"] for r in rows])
        pu = np.unique(p)
        get = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
        c = [mean_sem(get["C"][p == q]) for q in pu]
        h = [mean_sem(np.log2(get["haar"][p == q])) for q in pu]
        cl = [mean_sem(np.log2(get["clifford"][p == q])) for q in pu]
        # expectations of the two bounds over the masks
        pp = np.linspace(0, 1, 101)
        depth = 2 * n if rkey.startswith("nofinal") else n
        rec = pp * n * depth if rkey.startswith("nofinal") else n + pp * n * (depth - 1)
        ax.plot(pp, rec, "--", color=COL[n], lw=0.8)
        ax.plot(pp, 2 * n - pp * n, "-.", color=COL[n], lw=0.8)
        ax.plot(pu, [x[0] for x in c], "-", color=COL[n], lw=1.6)
        ax.errorbar(
            pu, [x[0] for x in h], [x[1] for x in h], fmt="o", color=COL[n], **KW
        )
        ax.errorbar(
            pu,
            [x[0] for x in cl],
            [x[1] for x in cl],
            fmt="s",
            mfc="white",
            color=COL[n],
            **KW,
        )
    ax.set_ylim(-0.3, 10.6)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("measurement probability $p$")
    ax.set_ylabel("bits")
    ax.set_title(title, fontsize=10.5)
for ax, lab in zip(axes, "ab"):
    ax.text(
        -0.11, 1.04, f"({lab})", transform=ax.transAxes, fontsize=11, fontweight="bold"
    )
hd = [Line2D([], [], color=COL[n], lw=1.6, label=f"$N = {n}$") for n in NS]
hd += [
    Line2D([], [], color="0.3", lw=1.6, label=r"min cut $\langle C\rangle$"),
    Line2D(
        [],
        [],
        color="0.3",
        marker="o",
        lw=0,
        ms=4,
        label=r"Haar $\langle\log_2 \mathrm{rank}\rangle$ (exact)",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        marker="s",
        mfc="white",
        lw=0,
        ms=4,
        label=r"Clifford $\langle\log_2 \mathrm{rank}\rangle$ (same masks)",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        ls="--",
        lw=0.8,
        label=r"$\langle$record bound$\rangle$: $pNT$ or $N + pN(T-1)$",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        ls="-.",
        lw=0.8,
        label=r"$\langle$input bound$\rangle$: $N(2 - p)$",
    ),
]
fig.legend(
    handles=hd,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=8.8,
    labelspacing=0.9,
)
fig.suptitle(
    "Step 1. Dense rank of the effect space against the space-time min cut, "
    "mean over masks (24 per point, 10 at $N = 5$)",
    fontsize=11.5,
)
fig.tight_layout()
fig.savefig(f"{OUT}/mirror_1_rank_vs_mincut.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure B: geometry vs learning performance ----------------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
rng = np.random.default_rng(0)
for col, (_, ekey, title) in enumerate(SETTINGS):
    for n in NS:
        rows = E[ekey][str(n)]
        m = rows[0]["M"]
        c = np.array([r["C"] for r in rows])
        e = np.array([r["e_circuit"] for r in rows])
        p = np.array([r["p"] for r in rows])
        pred = 1 - np.minimum(2.0**c - 1, m) / m
        ax = axes[0, col]
        ax.plot(
            c + (n - 4) * 0.18 + rng.uniform(-0.05, 0.05, len(c)),
            e,
            "o",
            ms=2.2,
            color=COL[n],
            alpha=0.5,
        )
        cs = np.arange(c.min(), c.max() + 1)
        ax.plot(
            cs + (n - 4) * 0.18,
            1 - np.minimum(2.0**cs - 1, m) / m,
            "_",
            ms=14,
            mew=2,
            color="#c0392b",
        )
        ax = axes[1, col]
        pu = np.unique(p)
        me = [mean_sem(e[p == q]) for q in pu]
        ax.errorbar(
            pu, [x[0] for x in me], [x[1] for x in me], fmt="o", color=COL[n], **KW
        )
        ax.plot(pu, [pred[p == q].mean() for q in pu], "-", color=COL[n], lw=1.0)
    axes[0, col].set_xlabel("min cut $C$ (bits)")
    axes[0, col].set_ylabel(r"$E_{\rm circuit} = 1 - R^2(\infty)$ per circuit")
    axes[0, col].set_title(title + ": level by level", fontsize=10.5)
    axes[1, col].set_xlabel("measurement probability $p$")
    axes[1, col].set_ylabel(r"$\langle E_{\rm circuit}\rangle$")
    axes[1, col].set_title(title + ": against $p$", fontsize=10.5)
for ax, lab in zip(axes.T.flat, "acbd"):
    ax.text(
        -0.11, 1.04, f"({lab})", transform=ax.transAxes, fontsize=11, fontweight="bold"
    )
hd = [
    Line2D([], [], color=COL[n], marker="o", ms=3.2, lw=0, label=f"$N = {n}$")
    for n in NS
]
hd += [
    Line2D(
        [],
        [],
        color="#c0392b",
        marker="_",
        ms=12,
        mew=2,
        lw=0,
        label="$1 - (2^C - 1)/M$ (a, b)",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        lw=1.0,
        label=r"$\langle 1 - (2^C - 1)/M\rangle$ over masks (c, d)",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        marker="o",
        lw=0,
        ms=4,
        label=r"measured $\langle E_{\rm circuit}\rangle$ (c, d)",
    ),
]
fig.legend(
    handles=hd,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=8.8,
    labelspacing=0.9,
)
fig.suptitle(
    "Step 2. Circuit error of the population ridge against the min cut, Haar, "
    "isotropic targets ($M = 4^N - 1$ dense, $3^N - 1$ XZ)",
    fontsize=11.5,
)
fig.tight_layout()
fig.savefig(f"{OUT}/mirror_2_ecircuit_vs_mincut.png", dpi=200, bbox_inches="tight")
