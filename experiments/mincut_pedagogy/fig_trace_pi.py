"""E_circuit(p) reduced to the projector: trace and Pauli-weight shape (#388).

Readout error and estimation error are both zero here: exact Heisenberg
spans (infinitely many shots) and the full effect space (``k = inf``). With
the flat prior, ``E_circuit = 1 - Tr(Pi)/M`` with ``Tr(Pi) = d``. Panel (a)
compares ``d(p)`` with the min cut, panel (b) shows the shape of ``Pi`` over
the Pauli weights at a few ``p``, panel (c) the mean ratio between successive
weights against ``p``.

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_trace_pi.py figures/haar_dense_data.json figures
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

D = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
NS = sorted(int(k) for k in D)
NCOL = dict(zip(NS, plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(NS)))))
N_SHAPE = NS[-1]
P_SHAPE = (0.05, 0.1, 0.2, 0.5, 1.0)
PCOL = dict(zip(P_SHAPE, plt.get_cmap("plasma")(np.linspace(0.0, 0.85, len(P_SHAPE)))))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
summary: dict = {}
ax = axes[0]
for n in NS:
    rows = [r for r in D[str(n)] if r["p"] > 0]
    p = np.array([r["p"] for r in rows])
    d = np.array([np.mean([c["d"] for c in r["circuits"]]) for r in rows])
    se = np.array(
        [
            np.std([c["d"] for c in r["circuits"]]) / np.sqrt(len(r["circuits"]))
            for r in rows
        ]
    )
    same = np.array([np.mean([2 ** c["C"] - 1 for c in r["circuits"]]) for r in rows])
    geo = np.array([np.mean([2.0**c - 1 for c in r["geometry_C"]]) for r in rows])
    ax.errorbar(p, d, se, fmt="o", color=NCOL[n], ms=3.4, elinewidth=0.8, capsize=0)
    ax.plot(p, same, "x", color=NCOL[n], ms=5, mew=0.9)
    ax.plot(p, geo, "-", color=NCOL[n], lw=1.0)
    ax.axhline(4**n - 1, color=NCOL[n], lw=0.7, ls=":")
    summary[n] = dict(
        p=p.tolist(),
        d=d.tolist(),
        same_mask=same.tolist(),
        geometry=geo.tolist(),
        max_gap_same=float(np.max(np.abs(d - same))),
    )
ax.set_yscale("log")
ax.set_xlabel("measurement probability $p$")
ax.set_ylabel(r"$d = \mathrm{Tr}\,\Pi = M\,(1 - E_{\rm circuit})$")
ax.set_title(r"mass of $\Pi$ against the min cut", fontsize=10.5)

ax = axes[1]
m = np.arange(1, N_SHAPE + 1)
rows = {r["p"]: r for r in D[str(N_SHAPE)]}
for p in P_SHAPE:
    f = np.array([c["frac_m"] for c in rows[p]["circuits"]])
    cmean = np.mean([c["C"] for c in rows[p]["circuits"]])
    ax.errorbar(
        m,
        f.mean(0),
        f.std(0) / np.sqrt(len(f)),
        fmt="o-",
        color=PCOL[p],
        ms=3.4,
        lw=0.9,
        elinewidth=0.8,
        capsize=0,
        label=rf"$p = {p}$, $\langle C\rangle = {cmean:.1f}$",
    )
f_one = np.mean([c["frac_m"] for c in rows[1.0]["circuits"]], 0)
ax.plot(
    m,
    f_one[0] * 3.0 ** -(m - 1),
    color="0.4",
    ls="--",
    lw=0.8,
    label=r"$\propto 3^{-m}$",
)
ax.set_yscale("log")
ax.set_xticks(m)
ax.set_xlabel("Pauli weight $m$")
ax.set_ylabel(r"$d_m / \mathrm{mult}(m)$")
ax.set_title(f"shape of $\\Pi$ over the weights, $N = {N_SHAPE}$", fontsize=10.5)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")

ax = axes[2]
for n in NS:
    rows_n = [r for r in D[str(n)] if r["p"] > 0]
    p = [r["p"] for r in rows_n]
    ratio = []
    for r in rows_n:
        f = np.mean([c["frac_m"] for c in r["circuits"]], 0)
        ratio.append(float(np.exp(np.mean(np.diff(np.log(f))))))
    ax.plot(p, ratio, "o-", color=NCOL[n], ms=3.4, lw=0.9)
    summary[n]["weight_ratio"] = ratio
ax.axhline(1 / 3, color="0.4", ls="--", lw=0.8)
ax.axhline(1.0, color="0.4", ls=":", lw=0.8)
ax.set_ylim(0.25, 1.05)
ax.set_xlabel("measurement probability $p$")
ax.set_ylabel(
    r"geometric mean of $\frac{d_{m+1}/\mathrm{mult}(m+1)}{d_m/\mathrm{mult}(m)}$"
)
ax.set_title("flatness of the profile", fontsize=10.5)

for a, lab in zip(axes, "abc"):
    a.text(
        -0.14, 1.04, f"({lab})", transform=a.transAxes, fontsize=11, fontweight="bold"
    )
hd = [
    Line2D([], [], color=NCOL[n], lw=1.0, marker="o", ms=3.4, label=f"$N = {n}$")
    for n in NS
]
hd += [
    Line2D(
        [], [], color="0.3", marker="o", ls="none", ms=3.4, label="(a) measured $d$"
    ),
    Line2D(
        [],
        [],
        color="0.3",
        marker="x",
        ls="none",
        ms=5,
        label="(a) $2^C - 1$, same masks",
    ),
    Line2D([], [], color="0.3", lw=1.0, label="(a) $2^C - 1$, 2 000 masks"),
    Line2D(
        [],
        [],
        color="0.3",
        ls=":",
        lw=0.7,
        label="(a) $M = 4^N - 1$\n(c) uniform capture",
    ),
    Line2D([], [], color="0.3", ls="--", lw=0.8, label="(c) only $Z$ strings, $1/3$"),
]
fig.legend(
    handles=hd,
    loc="center left",
    bbox_to_anchor=(0.865, 0.5),
    frameon=False,
    fontsize=8.5,
)
fig.suptitle(
    "Haar, dense inputs, no final readout, $T = 2N$, infinitely many shots, "
    r"$k = \infty$, flat prior: $E_{\rm circuit} = 1 - \mathrm{Tr}\,\Pi / M$",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 0.86, 0.95))
fig.savefig(f"{OUT}/trace_pi.png", dpi=180)
json.dump(summary, open(f"{OUT}/trace_pi_summary.json", "w"), indent=1)
