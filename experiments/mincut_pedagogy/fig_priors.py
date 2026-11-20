"""What the flat and iid target priors look like (#388).

``flat`` and ``iid`` are priors on the target, not encodings: the encoding
(``xz`` angles or ``sphere``) fixes the norms ``w_a = E[B_a^2]``, the prior
fixes the variances ``v_a = E[beta_a^2]``. iid: ``beta_a ~ N(0, 1)``, energy
``Omega_a = w_a`` per mode. flat: ``beta_a ~ N(0, 1/w_a)``, energy 1 per
mode. Both priors are drawn from the same Gaussian numbers here, so the only
difference between a flat and an iid sample is the scale of each mode. Every
sample is normalised to unit variance.

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_priors.py figures
"""

import sys
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from QELM.evaluation import fourier_framework as ff  # noqa: E402

OUT = sys.argv[1]
PRIOR_COL = {"flat": "#1f4e9c", "iid": "#4f9bd9"}


def monomials_xz(theta: np.ndarray) -> np.ndarray:
    """``B_a(x)`` for angle inputs, Bloch vector ``(sin t, 0, cos t)`` per site."""
    n = theta.shape[-1]
    ext = np.stack(
        [np.ones_like(theta), np.sin(theta), np.zeros_like(theta), np.cos(theta)], -1
    )
    codes = ff.pauli_codes(n)
    out = np.ones(theta.shape[:-1] + (len(codes),))
    for site in range(n):
        out *= ext[..., site, codes[:, site]]
    return out


def coefficients(n: int, z: np.ndarray, prior: str) -> np.ndarray:
    """``beta_a`` (index 0 the constant, set to 0), unit variance."""
    w = ff.input_weights(n, "xz")
    beta = np.zeros_like(w)
    vis = w > 0
    vis[0] = False
    scale = np.ones(vis.sum()) if prior == "iid" else 1 / np.sqrt(w[vis])
    beta[vis] = z[: vis.sum()] * scale
    return beta / np.sqrt(np.sum(w * beta**2))


fig = plt.figure(figsize=(18, 7.6))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.5, 1.25], wspace=0.55, hspace=0.42)
rng = np.random.default_rng(388)

# (a-d) N = 2 heat maps
grid = np.linspace(0, 2 * np.pi, 181)
t1, t2 = np.meshgrid(grid, grid)
mono2 = monomials_xz(np.stack([t1, t2], -1))
zs = [rng.standard_normal(16) for _ in range(2)]
vmax = 3.0
heat = []
for row, prior in enumerate(("flat", "iid")):
    for col, z in enumerate(zs):
        ax = fig.add_subplot(gs[row, col])
        heat.append(ax)
        f = mono2 @ coefficients(2, z, prior)
        im = ax.pcolormesh(
            t1, t2, f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
        )
        ax.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ax.set_yticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_title(f"{prior}, sample {col + 1}, $N = 2$", fontsize=10)
top, bottom = heat[1].get_position(), heat[3].get_position()
cax = fig.add_axes((top.x1 + 0.006, bottom.y0, 0.006, top.y1 - bottom.y0))
fig.colorbar(im, cax=cax, label="$f$")

# (e, f) N = 6, one-dimensional cut theta_i = t + phi_i
n6 = 6
t = np.linspace(0, 2 * np.pi, 600)
phase = rng.uniform(0, 2 * np.pi, n6)
mono6 = monomials_xz(t[:, None] + phase[None, :])
for row in range(2):
    ax = fig.add_subplot(gs[row, 2])
    z = rng.standard_normal(4**n6)
    for prior in ("flat", "iid"):
        ax.plot(
            t,
            mono6 @ coefficients(n6, z, prior),
            color=PRIOR_COL[prior],
            lw=1.0,
            label=prior,
        )
    ax.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$t$ (every $\theta_i = t + \varphi_i$)")
    ax.set_ylabel("$f$")
    ax.set_title(f"$N = 6$, same Gaussian numbers, sample {row + 1}", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

# (g) energy per mode, (h) energy per weight, N = 6, both encodings
m = np.arange(1, n6 + 1)
ax = fig.add_subplot(gs[0, 3])
for enc, base, ls in (("dense", 3.0, "-"), ("XZ", 2.0, "--")):
    ax.plot(
        m,
        np.ones(n6),
        "s" + ls,
        color=PRIOR_COL["flat"],
        ms=3.5,
        lw=0.9,
        label=f"flat, {enc}",
    )
    ax.plot(
        m,
        base**-m,
        "o" + ls,
        color=PRIOR_COL["iid"],
        ms=3.5,
        lw=0.9,
        label=f"iid, {enc}: ${base:.0f}^{{-m}}$",
    )
ax.set_yscale("log")
ax.set_xticks(m)
ax.set_xlabel("Pauli weight $m$")
ax.set_ylabel(r"energy per mode $\Omega_m = v_m w_m$")
ax.set_title("per mode", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")

ax = fig.add_subplot(gs[1, 3])
width = 0.2
for j, (enc, base) in enumerate((("dense", 3), ("XZ", 2))):
    mult = np.array([comb(n6, k) * base**k for k in m], dtype=float)
    flat = mult / mult.sum()
    iid = np.array([comb(n6, k) for k in m], dtype=float) / (2**n6 - 1)
    hatch = None if enc == "dense" else "//"
    ax.bar(
        m + (2 * j - 1.5) * width,
        flat,
        width,
        color=PRIOR_COL["flat"],
        hatch=hatch,
        edgecolor="white",
        label=f"flat, {enc}",
    )
    ax.bar(
        m + (2 * j - 0.5) * width,
        iid,
        width,
        color=PRIOR_COL["iid"],
        hatch=hatch,
        edgecolor="white",
        label=f"iid, {enc}",
    )
ax.set_xticks(m)
ax.set_xlabel("Pauli weight $m$")
ax.set_ylabel("share of the energy $P(m)$")
ax.set_title("per weight (iid is the same in both encodings)", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

for k, a in enumerate(fig.axes[:4] + fig.axes[5:]):
    a.text(
        -0.2,
        1.06,
        f"({'abcdefgh'[k]})",
        transform=a.transAxes,
        fontsize=11,
        fontweight="bold",
    )
fig.suptitle(
    "Target priors: flat (energy 1 per mode) against iid "
    r"($\beta_a \sim \mathcal{N}(0, 1)$, energy $w_a$ per mode)",
    fontsize=11.5,
)
fig.savefig(f"{OUT}/priors.png", dpi=170, bbox_inches="tight")
