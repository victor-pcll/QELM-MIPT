"""Pauli-weight profile of the effect space by min-cut level (#383).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_weight_profile.py figures/haar_dense_data.json figures
"""

import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

D = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
NS = sorted(int(k) for k in D)
NCOL = dict(zip(NS, plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(NS)))))

fig, axes = plt.subplots(
    1,
    len(NS) + 1,
    figsize=(17.5, 4.6),
    gridspec_kw=dict(width_ratios=[1] * len(NS) + [1.15]),
)
summary: dict = {}
for ax, n in zip(axes, NS):
    by_c: dict = defaultdict(list)
    for r in D[str(n)]:
        for c in r["circuits"]:
            by_c[c["C"]].append(c["frac_m"])
    levels = sorted(c for c in by_c if c > 0 and len(by_c[c]) >= 5)
    cmap = plt.get_cmap("viridis")
    m = np.arange(1, n + 1)
    big_m = 4**n - 1
    summary[n] = {}
    for k, c in enumerate(levels):
        col = cmap(k / max(1, len(levels) - 1))
        f = np.array(by_c[c])
        ax.errorbar(
            m,
            f.mean(0),
            f.std(0),
            fmt="o-",
            color=col,
            ms=3.2,
            lw=0.9,
            elinewidth=0.8,
            capsize=0,
            label=f"$C = {c}$ ({len(f)})",
        )
        ax.axhline((2**c - 1) / big_m, color=col, lw=0.7, ls="--")
        summary[n][c] = dict(
            count=len(f),
            mean=f.mean(0).tolist(),
            sd=f.std(0).tolist(),
            uniform=(2**c - 1) / big_m,
        )
    ax.set_yscale("log")
    ax.set_xticks(m)
    ax.set_xlabel("Pauli weight $m$")
    ax.set_ylabel(r"$d_m / \mathrm{mult}(m)$")
    ax.set_title(f"$N = {n}$", fontsize=10.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)

ax = axes[-1]
for n in NS:
    rows = D[str(n)]
    p = [r["p"] for r in rows]
    flat = [np.mean([1 - c["r2_flat"] for c in r["circuits"]]) for r in rows]
    iid = [np.mean([1 - c["r2_iid"] for c in r["circuits"]]) for r in rows]
    ax.plot(p, flat, "o-", color=NCOL[n], ms=3.2, lw=0.9)
    ax.plot(p, iid, "s--", color=NCOL[n], ms=3.2, lw=0.9, mfc="white")
ax.set_xlabel("measurement probability $p$")
ax.set_ylabel(r"$\langle E_{\rm circuit}\rangle$ (ratio of means)")
ax.set_title("isotropic (flat) against iid targets", fontsize=10.5)
for a, lab in zip(axes, "abcd"):
    a.text(
        -0.14, 1.04, f"({lab})", transform=a.transAxes, fontsize=11, fontweight="bold"
    )
hd = [
    Line2D([], [], color=NCOL[n], lw=0.9, marker="o", ms=3.2, label=f"$N = {n}$ (d)")
    for n in NS
]
hd += [
    Line2D(
        [],
        [],
        color="0.3",
        ls="--",
        lw=0.7,
        label="(a-c) uniform capture\n$(2^C - 1)/(4^N - 1)$",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        marker="o",
        ms=3.2,
        lw=0.9,
        label="(d) flat: $\\sum_m d_m/(4^N - 1)$",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        marker="s",
        mfc="white",
        ms=3.2,
        ls="--",
        lw=0.9,
        label="(d) iid: $\\sum_m 3^{-m} d_m/(2^N - 1)$",
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
    "Fraction of each Pauli-weight shell captured by the effect space, "
    "by min-cut level "
    "(Haar, dense, $T = 2N$, no final readout; mean and spread over circuits)",
    fontsize=11.5,
)
fig.tight_layout()
fig.savefig(f"{OUT}/weight_profile_by_level.png", dpi=200, bbox_inches="tight")
json.dump(summary, open(f"{OUT}/weight_profile_summary.json", "w"), indent=1)
