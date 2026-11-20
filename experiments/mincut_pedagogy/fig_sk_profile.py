"""Which Pauli weights the k-bit parities reach (#388).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_sk_profile.py figures/sk_profile_data.json figures
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
PS = [r["p"] for r in D[str(NS[0])]]
KLAB = ["$k = 1$", "$k = 2$", "$k = 3$", r"$k = \infty$"]
KCOL = list(plt.get_cmap("viridis")(np.linspace(0.0, 0.85, 3))) + ["0.15"]
PCOL = dict(zip(PS, plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(PS)))))
P_SHOW = 0.3

fig, axes = plt.subplots(2, len(NS), figsize=(14.5, 8.0))
summary: dict = {}
for col, n in enumerate(NS):
    m = np.arange(1, n + 1)
    summary[n] = {}
    for r in D[str(n)]:
        mult = np.array(r["mult"], dtype=float)
        prof = np.array([c["d_m"] for c in r["circuits"]])  # (circ, 4, n)
        mean = prof.mean(0)
        # capture of the weight-m content of S by S_k, ratio of means
        cap = mean[:3] / np.maximum(mean[3], 1e-300)
        cs = [c["C"] for c in r["circuits"]]
        summary[n][r["p"]] = dict(
            C_mean=float(np.mean(cs)),
            K_mean=float(np.mean([c["K"] for c in r["circuits"]])),
            dims_mean=np.mean([c["dims"] for c in r["circuits"]], 0).tolist(),
            d_m_mean=mean.tolist(),
            capture=cap.tolist(),
        )
        if r["p"] == P_SHOW:
            ax = axes[0, col]
            for k in range(4):
                f = prof[:, k] / mult
                ax.errorbar(
                    m,
                    f.mean(0),
                    f.std(0),
                    fmt="o-",
                    color=KCOL[k],
                    ms=3.2,
                    lw=0.9,
                    elinewidth=0.8,
                    capsize=0,
                )
            ax.set_yscale("log")
            ax.set_xticks(m)
            ax.set_xlabel("Pauli weight $m$")
            ax.set_ylabel(r"$d_m^{(k)} / \mathrm{mult}(m)$")
            ax.set_title(
                f"$N = {n}$, $p = {P_SHOW}$, "
                rf"$\langle C\rangle = {np.mean(cs):.1f}$",
                fontsize=10.5,
            )
        ax = axes[1, col]
        for k, ls in zip(range(3), ("-", "--", ":")):
            ax.plot(m, cap[k], marker="o", ls=ls, color=PCOL[r["p"]], ms=3.2, lw=0.9)
    ax = axes[1, col]
    ax.set_ylim(0, 1.03)
    ax.set_xticks(m)
    ax.set_xlabel("Pauli weight $m$")
    ax.set_ylabel(r"$\langle d_m^{(k)}\rangle / \langle d_m^{(\infty)}\rangle$")
    ax.set_title(
        f"$N = {n}$: share of $\\mathcal{{S}}$ reached by $\\mathcal{{S}}_k$",
        fontsize=10.5,
    )
for a, lab in zip(axes.flat, "abcdef"):
    a.text(
        -0.14, 1.04, f"({lab})", transform=a.transAxes, fontsize=11, fontweight="bold"
    )
h1 = [
    Line2D([], [], color=KCOL[k], marker="o", ms=3.2, lw=0.9, label=KLAB[k])
    for k in range(4)
]
h2 = [
    Line2D([], [], color=PCOL[p], lw=0.9, marker="o", ms=3.2, label=f"$p = {p}$")
    for p in PS
]
h2 += [
    Line2D([], [], color="0.3", ls=ls, lw=0.9, label=KLAB[k])
    for k, ls in zip(range(3), ("-", "--", ":"))
]
fig.legend(
    handles=h1,
    loc="center left",
    bbox_to_anchor=(0.905, 0.73),
    frameon=False,
    fontsize=9,
    title="(a-c)",
)
fig.legend(
    handles=h2,
    loc="center left",
    bbox_to_anchor=(0.905, 0.27),
    frameon=False,
    fontsize=9,
    title="(d-f)",
)
fig.suptitle(
    "Haar, dense inputs, no final readout, $T = 2N$: Pauli-weight "
    "profile of the span of the parities of at most $k$ record bits",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 0.9, 0.97))
fig.savefig(f"{OUT}/sk_profile.png", dpi=180)
json.dump(summary, open(f"{OUT}/sk_profile_summary.json", "w"), indent=1)
