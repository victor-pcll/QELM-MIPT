"""Spectral overlap of the target and the circuit, R² = Σ_m P(m) T(m) (#388).

``P(m) = Ω_m mult(m) / Σ Ω mult`` is the share of the target energy at Pauli
weight ``m``; ``T(m) = d_m / mult(m)`` is the share of the weight-``m`` modes
that the effect space transmits; ``T_k(m)`` the same for the parities of at
most ``k`` record bits. Haar, dense inputs, no final readout, ``T = 2N``,
infinitely many shots. Panels (a, b) use ``haar_dense_data.json``, panel (c)
uses ``sk_profile_data.json`` (other circuits).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_overlap.py figures/haar_dense_data.json \\
        figures/sk_profile_data.json figures
"""

import json
import sys
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DENSE = json.load(open(sys.argv[1]))
SK = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
N = 6
P_T = (0.05, 0.2, 1.0)
P_SK = 0.3
m = np.arange(1, N + 1)
mult = np.array([comb(N, k) * 3**k for k in m], dtype=float)


def share(omega: np.ndarray) -> np.ndarray:
    e = omega * mult
    return e / e.sum()


PRIORS = {
    "flat ($\\Omega_m$ constant)": share(np.ones(N)),
    "iid ($\\Omega_m = 3^{-m}$)": share(3.0**-m),
    "degree $\\leq 2$": share((m <= 2).astype(float)),
}
PRCOL = dict(zip(PRIORS, ("#1f4e9c", "#4f9bd9", "#2a9d8f")))
PRMARK = dict(zip(PRIORS, ("s", "D", "^")))
TCOL = dict(zip(P_T, plt.get_cmap("plasma")((0.0, 0.45, 0.8))))
rows = {r["p"]: r for r in DENSE[str(N)]}
T = {p: np.mean([c["frac_m"] for c in rows[p]["circuits"]], 0) for p in P_T}
C = {p: np.mean([c["C"] for c in rows[p]["circuits"]]) for p in P_T}

fig, axes = plt.subplots(1, 3, figsize=(18, 4.9))
summary: dict = {"N": N, "P": {}, "T": {}, "R2": {}, "R2_k": {}}

ax = axes[0]
for name, pm in PRIORS.items():
    keep = pm > 0
    ax.plot(
        m[keep],
        pm[keep],
        PRMARK[name] + "--",
        color=PRCOL[name],
        ms=4,
        lw=0.9,
        mfc="white",
    )
    summary["P"][name] = pm.tolist()
for p in P_T:
    ax.plot(m, T[p], "o-", color=TCOL[p], ms=3.6, lw=1.1)
    summary["T"][p] = T[p].tolist()
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.2)
ax.set_xticks(m)
ax.set_xlabel("Pauli weight $m$")
ax.set_ylabel("share")
ax.set_title(
    "target energy $P(m)$ (open, dashed) and circuit transfer $T(m)$", fontsize=10.5
)

ax = axes[1]
p0 = 0.2
width = 0.26
for j, (name, pm) in enumerate(PRIORS.items()):
    contrib = pm * T[p0]
    r2 = float(contrib.sum())
    summary["R2"][name] = {p: float((pm * T[p]).sum()) for p in P_T}
    ax.bar(
        m + (j - 1) * width,
        contrib,
        width,
        color=PRCOL[name],
        label=f"{name}: $R^2 = {r2:.3f}$",
    )
ax.set_xticks(m)
ax.set_xlabel("Pauli weight $m$")
ax.set_ylabel("$P(m)\\,T(m)$")
ax.set_title(
    f"contributions to $R^2(\\infty)$ at $p = {p0}$ "
    f"($\\langle C\\rangle = {C[p0]:.1f}$)",
    fontsize=10.5,
)
ax.legend(fontsize=8, frameon=False, loc="upper right")

ax = axes[2]
rsk = next(r for r in SK[str(N)] if r["p"] == P_SK)
prof = np.array([c["d_m"] for c in rsk["circuits"]]).mean(0) / mult
cmean = np.mean([c["C"] for c in rsk["circuits"]])
KLAB = ["$k = 1$", "$k = 2$", "$k = 3$", "$k = \\infty$"]
KCOL = list(plt.get_cmap("viridis")(np.linspace(0.0, 0.85, 3))) + ["0.15"]
for k in range(4):
    ls = "-" if k == 3 else ":"
    r2 = {name: float((pm * prof[k]).sum()) for name, pm in PRIORS.items()}
    summary["R2_k"][KLAB[k]] = r2
    lab = KLAB[k] + ": " + ", ".join(f"{v:.3f}" for v in r2.values())
    ax.plot(m, prof[k], "o" + ls, color=KCOL[k], ms=3.6, lw=1.1, label=lab)
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.2)
ax.set_xticks(m)
ax.set_xlabel("Pauli weight $m$")
ax.set_ylabel("$T_k(m) = d_m^{(k)} / \\mathrm{mult}(m)$")
ax.set_title(
    f"readout of order $k$, $p = {P_SK}$ ($\\langle C\\rangle = {cmean:.1f}$)",
    fontsize=10.5,
)
ax.legend(
    fontsize=7.5,
    frameon=False,
    loc="lower left",
    title="$R^2(k)$: flat, iid, degree $\\leq 2$",
    title_fontsize=7.5,
)

for a, lab in zip(axes, "abc"):
    a.text(
        -0.13, 1.04, f"({lab})", transform=a.transAxes, fontsize=11, fontweight="bold"
    )
handles = [
    plt.Line2D(
        [],
        [],
        color=PRCOL[n],
        marker=PRMARK[n],
        mfc="white",
        ls="--",
        lw=0.9,
        ms=4,
        label=n,
    )
    for n in PRIORS
]
handles += [
    plt.Line2D(
        [],
        [],
        color=TCOL[p],
        marker="o",
        lw=1.1,
        ms=3.6,
        label=f"$T(m)$, $p = {p}$, $\\langle C\\rangle = {C[p]:.1f}$",
    )
    for p in P_T
]
fig.legend(
    handles=handles,
    loc="center left",
    bbox_to_anchor=(0.865, 0.5),
    frameon=False,
    fontsize=8.5,
)
fig.suptitle(
    f"Haar, dense inputs, no final readout, $N = {N}$, $T = 2N$, "
    "infinitely many shots: "
    ""
    "$R^2 = \\sum_m P(m)\\,T(m)$",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 0.86, 0.95))
fig.savefig(f"{OUT}/overlap.png", dpi=180)
json.dump(summary, open(f"{OUT}/overlap_summary.json", "w"), indent=1)
