"""E_circuit against the min cut C, level by level (issue #383).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_ecircuit.py figures/ecircuit_data.json figures
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
ns = [3, 4, 5]
col = dict(zip(ns, plt.get_cmap("Blues")(np.linspace(0.45, 0.95, 3))))
TIT = {
    "final_xz_T=N": "final readout, XZ (angle) inputs, $T = N$",
    "nofinal_sphere_T=2N": "no final readout, dense (sphere) inputs, $T = 2N$",
}
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
rng = np.random.default_rng(0)
summary = {}
for row, (name, blk) in enumerate(D.items()):
    summary[name] = {}
    for n in ns:
        rows = blk[str(n)]
        M = rows[0]["M"]
        dim = 4**n - 1 if "sphere" in name else 3**n - 1
        C = np.array([r["C"] for r in rows])
        E = np.array([r["e_circuit"] for r in rows])
        p = np.array([r["p"] for r in rows])
        vis = np.array([r["visible"] for r in rows])
        pred = 1 - np.minimum(2.0**C - 1, M) / M
        ax = axes[row, 0]
        ax.plot(
            C + (n - 4) * 0.18 + rng.uniform(-0.05, 0.05, len(C)),
            E,
            "o",
            ms=2.2,
            color=col[n],
            alpha=0.5,
        )
        cs = np.arange(C.min(), C.max() + 1)
        ax.plot(
            cs + (n - 4) * 0.18,
            1 - np.minimum(2.0**cs - 1, M) / M,
            "_",
            ms=14,
            mew=2,
            color="#c0392b",
        )
        ax = axes[row, 1]
        pu = np.unique(p)
        me = [E[p == q].mean() for q in pu]
        se = [E[p == q].std(ddof=1) / np.sqrt((p == q).sum()) for q in pu]
        ax.errorbar(pu, me, se, fmt="o", color=col[n], **kw)
        ax.plot(pu, [pred[p == q].mean() for q in pu], "-", color=col[n], lw=1.0)
        ax = axes[row, 2]
        r2 = np.clip(1 - E, 1e-6, None)
        ax.plot(
            np.log2(np.clip(1 - pred, 1e-6, None)) + rng.uniform(-0.06, 0.06, len(C)),
            np.log2(r2),
            "o",
            ms=2.4,
            color=col[n],
            alpha=0.5,
        )
        summary[name][n] = {
            int(c): dict(
                count=int((C == c).sum()),
                e_mean=float(E[C == c].mean()),
                e_sd=float(E[C == c].std()),
                predicted=float(1 - min(2**c - 1, M) / M),
                visible_ranks=sorted(set(vis[C == c].tolist())),
            )
            for c in np.unique(C)
        }
    x = np.linspace(-10, 0, 200)
    axes[row, 2].plot(x, x, "--", color="0.35", lw=0.8)
    axes[row, 0].set_xlabel("min cut $C$ (bits)")
    axes[row, 0].set_ylabel(r"$E_{\rm circuit} = 1 - R^2(\infty)$ per circuit")
    axes[row, 0].set_title(TIT[name] + ": levels", fontsize=10.5)
    axes[row, 1].set_xlabel("measurement probability $p$")
    axes[row, 1].set_ylabel(r"$\langle E_{\rm circuit}\rangle$")
    axes[row, 1].set_title(
        r"mean over circuits: data (dots) vs $\langle 1 - (2^C - 1)/M\rangle$ (lines)",
        fontsize=10.5,
    )
    axes[row, 2].set_xlabel(
        r"predicted $\log_2[(2^C - 1)/M]$ (dense: $\approx C - 2N$)"
    )
    axes[row, 2].set_ylabel(r"$\log_2 R^2(\infty)$")
    axes[row, 2].set_title(
        r"$\log_2 R^2(\infty)$: data vs prediction, one bit less halves $R^2$",
        fontsize=10.5,
    )
    axes[row, 2].set_ylim(-11, 0.5)
    axes[row, 2].set_xlim(-11, 0.5)
for ax, lab in zip(axes.flat, "abcdef"):
    ax.text(
        -0.13, 1.04, f"({lab})", transform=ax.transAxes, fontsize=11, fontweight="bold"
    )
h = [
    Line2D([], [], color=col[n], marker="o", ms=3.2, lw=0.9, label=f"$N = {n}$")
    for n in ns
]
h += [
    Line2D(
        [],
        [],
        color="#c0392b",
        marker="_",
        ms=12,
        mew=2,
        lw=0,
        label="prediction $1 - (2^C - 1)/M$\n$M = 3^N - 1$ (XZ) or $4^N - 1$ (dense)",
    ),
    Line2D([], [], color="0.35", ls="--", label="data = prediction"),
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
    "Circuit error of the population ridge against the min cut, Haar, "
    "isotropic (flat) targets, 40 targets per circuit; "
    "top: XZ inputs saturate at $3^N - 1$",
    fontsize=11.5,
)
fig.tight_layout()
fig.savefig(f"{OUT}/ecircuit_vs_mincut.png", dpi=200, bbox_inches="tight")
json.dump(summary, open(f"{OUT}/ecircuit_vs_mincut_summary.json", "w"), indent=1)
