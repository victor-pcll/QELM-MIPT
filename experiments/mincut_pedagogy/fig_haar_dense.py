"""Haar, dense inputs, no final readout: the two-step figures (#383).

Step 1: exact log2 rank against the min cut C(p). Step 2: measured E_circuit(p)
against the prediction 1 - (2^C - 1)/M averaged over C(p), on one graph.

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_haar_dense.py figures/haar_dense_data.json figures
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
COL = dict(zip(NS, plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(NS)))))
KW = dict(ms=3.2, lw=0, elinewidth=0.8, capsize=0)
LEG = dict(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def stats(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


summary: dict = {}
curves: dict = {}
for n in NS:
    rows = D[str(n)]
    p = np.array([r["p"] for r in rows])
    m = rows[0]["M"]
    cur = dict(
        p=p.tolist(), C=[], log2rank=[], e=[], pred_same=[], pred_geo=[], C_geo=[]
    )
    eq = tot = 0
    for r in rows:
        cs = np.array([c["C"] for c in r["circuits"]], dtype=float)
        rk = np.array([c["rank"] for c in r["circuits"]], dtype=float)
        e = np.array([c["e_circuit"] for c in r["circuits"]])
        cg = np.array(r["geometry_C"], dtype=float)
        eq += int(np.sum(rk == 2**cs))
        tot += len(rk)
        cur["C"].append(stats(cs))
        cur["log2rank"].append(stats(np.log2(rk)))
        cur["e"].append(stats(e))
        cur["pred_same"].append(float(np.mean(1 - (2**cs - 1) / m)))
        cur["pred_geo"].append(float(np.mean(1 - (2**cg - 1) / m)))
        cur["C_geo"].append(float(cg.mean()))
    curves[n] = cur
    gap = np.abs(np.array([x[0] for x in cur["e"]]) - np.array(cur["pred_same"]))
    gap_geo = np.abs(np.array([x[0] for x in cur["e"]]) - np.array(cur["pred_geo"]))
    summary[n] = dict(
        M=m,
        rank_equals_2C=f"{eq}/{tot}",
        max_gap_same_masks=float(gap.max()),
        max_gap_geometry=float(gap_geo.max()),
        **{k: v for k, v in cur.items()},
    )

# ------------------------------ Step 1 ---------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.0))
pp = np.linspace(0, 1, 101)
for n in NS:
    cur = curves[n]
    p = np.array(cur["p"])
    ax.plot(pp, pp * n * 2 * n, "--", color=COL[n], lw=0.8)
    ax.plot(pp, 2 * n - pp * n, "-.", color=COL[n], lw=0.8)
    ax.plot(p, cur["C_geo"], "-", color=COL[n], lw=1.6)
    ax.errorbar(
        p,
        [x[0] for x in cur["log2rank"]],
        [x[1] for x in cur["log2rank"]],
        fmt="o",
        color=COL[n],
        **KW,
    )
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.3, 12.6)
ax.set_xlabel("measurement probability $p$")
ax.set_ylabel("bits")
ax.set_title("Step 1. Exact rank of the effect space against the min cut", fontsize=11)
hd = [Line2D([], [], color=COL[n], lw=1.6, label=f"$N = {n}$") for n in NS]
hd += [
    Line2D(
        [], [], color="0.3", lw=1.6, label=r"min cut $\langle C\rangle$ (2000 masks)"
    ),
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
        ls="--",
        lw=0.8,
        label=r"$\langle$record bound$\rangle$: $pNT$",
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
fig.legend(handles=hd, fontsize=8.8, labelspacing=0.9, **LEG)
fig.text(
    0.5,
    -0.02,
    "Haar U(4), open brickwork, $T = 2N$, Z measurements after each layer, "
    "no final readout",
    ha="center",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(f"{OUT}/haar_dense_step1_rank.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ------------------------------ Step 2 ---------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.0))
for n in NS:
    cur = curves[n]
    p = np.array(cur["p"])
    ax.plot(p, cur["pred_geo"], "-", color=COL[n], lw=1.6)
    ax.plot(p, cur["pred_same"], "x", color=COL[n], ms=4.5, mew=1.0)
    ax.errorbar(
        p,
        [x[0] for x in cur["e"]],
        [x[1] for x in cur["e"]],
        fmt="o",
        color=COL[n],
        **KW,
    )
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("measurement probability $p$")
ax.set_ylabel(r"$E_{\rm circuit} = 1 - R^2(\infty)$")
ax.set_title(
    r"Step 2. $E_{\rm circuit}(p)$: measurement against the min-cut formula",
    fontsize=11,
)
hd = [Line2D([], [], color=COL[n], lw=1.6, label=f"$N = {n}$") for n in NS]
hd += [
    Line2D(
        [],
        [],
        color="0.3",
        marker="o",
        lw=0,
        ms=4,
        label="measured (population ridge,\n40 isotropic targets per circuit)",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        lw=1.6,
        label=r"$\langle 1 - (2^{C} - 1)/(4^N - 1)\rangle_{C \sim P(C\,|\,p)}$"
        "\n(min cut only, 2000 masks)",
    ),
    Line2D(
        [],
        [],
        color="0.3",
        marker="x",
        lw=0,
        ms=4.5,
        label="same formula on the masks\nof the measured circuits",
    ),
]
fig.legend(handles=hd, fontsize=8.8, labelspacing=0.9, **LEG)
fig.text(
    0.5,
    -0.02,
    "Haar U(4), dense (sphere) inputs, $T = 2N$, no final readout",
    ha="center",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(f"{OUT}/haar_dense_step2_ecircuit.png", dpi=200, bbox_inches="tight")
json.dump(summary, open(f"{OUT}/haar_dense_summary.json", "w"), indent=1)
print(
    json.dumps(
        {
            n: {
                k: summary[n][k]
                for k in ("rank_equals_2C", "max_gap_same_masks", "max_gap_geometry")
            }
            for n in NS
        }
    )
)
