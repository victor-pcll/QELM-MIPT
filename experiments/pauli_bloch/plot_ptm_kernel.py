#!/usr/bin/env python3
"""Figures of the Pauli transfer matrix as the Fourier kernel (issue #369).

Through ``P_a <-> B_a`` (issue #364), a gate ``U`` with Heisenberg Pauli
transfer matrix ``R[a, b] = Tr[P_b U^dag P_a U] / 2^N`` maps the input Fourier
modes to the output features:

    U^dag P_a U = sum_b R_ab P_b    <=>    Tr[P_a U rho(x) U^dag] = sum_b R_ab B_b(x).

Two figures, each written with its data by
:func:`QELM.utils.plots.io.save_figure`:

1. ``ptm_noyau_fourier``: the commutative diagram, the PTM of a Haar gate in
   the Pauli basis and as a kernel on the 81 trigonometric monomials, one row
   of the kernel acting on the 16 input functions, and the numerical check;
2. ``ptm_composition``: Clifford (permutation of modes) against Haar (mixing),
   the dephasing filter, the kernel of a monitored circuit as a product, and
   the two PTMs drawn as bipartite graphs between input modes and outputs.

Usage::

    uv run python experiments/pauli_bloch/plot_ptm_kernel.py \
        experiments/pauli_bloch/figures
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from QELM.encoding import DenseAngleEncoding  # noqa: E402
from QELM.evaluation.fourier_framework import (  # noqa: E402
    pauli_codes,
    pauli_matrix,
    pauli_transfer_matrix,
)
from QELM.evaluation.pauli_bloch import (  # noqa: E402
    circuit_kernel,
    dense_monomials,
    dephasing_ptm,
    frequencies,
    passage_matrix,
    pauli_labels,
    trig_kernel,
    trig_label,
)
from QELM.models.circuits.clifford import random_clifford_gate  # noqa: E402
from QELM.models.circuits.haar import haar_gate  # noqa: E402
from QELM.utils.plots.io import save_figure  # noqa: E402
from QELM.utils.plots.style import apply_style, style_axes  # noqa: E402

CMAP = "RdBu_r"
WEIGHT_COLORS = ("#7f7f7f", "#2a78d6", "#1baf7a")
N = 2


def grid_inputs(s: np.ndarray) -> np.ndarray:
    """Two qubits on the path ``theta_i = phi_i = s_i`` (as in issue #364)."""
    s0, s1 = np.meshgrid(s, s, indexing="ij")
    return np.stack([s0.ravel(), s0.ravel(), s1.ravel(), s1.ravel()], axis=1)


def draw_ptm(ax: Axes, R: np.ndarray, labels: Sequence[str], title: str) -> None:
    im = ax.imshow(R, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=10)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_xlabel("mode d'entrée b  (B_b)")
    ax.set_ylabel("sortie a  (P_a)")
    ax.set_title(title, fontsize=14)
    style_axes(ax)
    plt.colorbar(im, ax=ax, shrink=0.75, label="R_ab")


def _box(ax: Axes, xy: tuple[float, float], text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (xy[0] - 0.15, xy[1] - 0.13),
            0.30,
            0.26,
            boxstyle="round,pad=0.02",
            fc=color,
            ec="k",
            lw=1.5,
        )
    )
    ax.text(*xy, text, ha="center", va="center", fontsize=15)


def _arrow(
    ax: Axes, a: tuple[float, float], b: tuple[float, float], label: str, side: str
) -> None:
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=25, lw=2.5, color="k")
    )
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    off = {"up": (0, 0.06), "down": (0, -0.07), "left": (-0.02, 0), "right": (0.02, 0)}
    ha = {"left": "right", "right": "left"}.get(side, "center")
    ax.text(
        mid[0] + off[side][0],
        mid[1] + off[side][1],
        label,
        ha=ha,
        va="center",
        fontsize=13,
        style="italic",
    )


def draw_diagram(ax: Axes) -> None:
    """The commutative square: operators on top, functions below."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tl, tr, bl, br = (0.25, 0.76), (0.75, 0.76), (0.25, 0.22), (0.75, 0.22)
    _box(ax, tl, "chaîne de Pauli\n$P_b$", "#dbe8f7")
    _box(ax, tr, "$U^\\dagger P_a U = \\sum_b R_{ab}\\, P_b$", "#dbe8f7")
    _box(ax, bl, "mode de Fourier\n$B_b(x) = \\prod_i n_i^{b_i}$", "#fde6d4")
    _box(
        ax,
        br,
        "$f_a(x) = \\mathrm{Tr}[P_a\\, U\\rho(x)U^\\dagger]$\n"
        "$= \\sum_b R_{ab}\\, B_b(x)$",
        "#fde6d4",
    )
    _arrow(ax, (0.41, 0.76), (0.59, 0.76), "PTM R  (Heisenberg)", "up")
    _arrow(ax, (0.41, 0.22), (0.59, 0.22), "noyau de Fourier K = R", "down")
    _arrow(ax, (0.25, 0.62), (0.25, 0.36), "Φ : P ↦ B", "left")
    _arrow(ax, (0.75, 0.62), (0.75, 0.36), "Φ", "right")
    ax.text(0.02, 0.76, "côté\nopérateurs", fontsize=13, va="center", color="#2a5d9f")
    ax.text(0.02, 0.22, "côté\nfonctions", fontsize=13, va="center", color="#b5541a")
    ax.text(
        0.5,
        0.5,
        "le carré commute :\nmême matrice R, deux lectures",
        ha="center",
        va="center",
        fontsize=13,
        bbox={"fc": "white", "ec": "#999999", "boxstyle": "round"},
    )
    ax.set_title(
        "(a) L'isomorphisme Φ transporte la PTM en noyau de Fourier", fontsize=15
    )


# ---------------------------------------------------------------------------
# Figure 7
# ---------------------------------------------------------------------------


def figure_kernel(out: Path, seed: int, row_label: str = "ZI") -> dict:
    rng = np.random.default_rng(seed)
    U = haar_gate(rng)
    R = pauli_transfer_matrix(U)
    labels = pauli_labels(N)
    Kt = trig_kernel(R, N)
    M = passage_matrix(N)

    apply_style()
    fig = plt.figure(figsize=(22, 20), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.7, 1.2, 0.85])
    draw_diagram(fig.add_subplot(gs[0]))

    mid = gs[1].subgridspec(1, 2, width_ratios=[1, 1.25])
    draw_ptm(
        fig.add_subplot(mid[0]),
        R,
        labels,
        "(b) PTM d'une porte Haar U(4), base de Pauli\n"
        "R_ab = Tr[P_b U† P_a U] / 4  (réelle, orthogonale)",
    )
    ax = fig.add_subplot(mid[1])
    # Kernel restricted to the reachable monomials, in the Pauli order: the
    # same numbers as (b), labelled by the trigonometric formulas.
    order_trig = [int(np.flatnonzero(M[a])[0]) for a in range(16)]
    K16 = Kt[np.ix_(order_trig, order_trig)]
    im = ax.imshow(K16, cmap=CMAP, vmin=-1, vmax=1)
    names = [trig_label(c, N) for c in order_trig]
    ax.set_xticks(range(16), names, rotation=90, fontsize=10)
    ax.set_yticks(range(16), names, fontsize=10)
    ax.set_xlabel("monôme d'entrée B_b(θ, φ)")
    ax.set_ylabel("feature de sortie f_a")
    ax.set_title(
        "(c) La même matrice, lue comme noyau de Fourier :\n"
        "étiquettes = monômes trigonométriques de (θ₀, φ₀, θ₁, φ₁)\n"
        f"(les {9**N - 4**N} autres monômes des {9**N} ont lignes et colonnes nulles)",
        fontsize=14,
    )
    style_axes(ax)
    plt.colorbar(im, ax=ax, shrink=0.75, label="K_ab = R_ab")

    # Row action: f_a = sum_b R_ab B_b, drawn term by term.
    a = labels.index(row_label)
    s = np.linspace(0, np.pi, 61)
    B = dense_monomials(grid_inputs(s)).reshape(len(s), len(s), 16)
    f = B @ R[a]
    bot = gs[2].subgridspec(1, 3, width_ratios=[2.4, 0.8, 0.9])
    tiles = bot[0].subgridspec(3, 8, height_ratios=[0.12, 1, 1])
    k_theta, _ = frequencies(N)
    order = np.argsort(-np.abs(R[a]), kind="stable")
    for j, b in enumerate(order):
        axt = fig.add_subplot(tiles[1 + j // 8, j % 8])
        axt.imshow(
            B[:, :, b] * R[a, b],
            cmap=CMAP,
            vmin=-1,
            vmax=1,
            origin="upper",
            extent=(0, 1, 1, 0),
        )
        axt.set_xticks([])
        axt.set_yticks([])
        for spine in axt.spines.values():
            spine.set_edgecolor(WEIGHT_COLORS[k_theta[b]])
            spine.set_linewidth(3)
        axt.set_title(f"{R[a, b]:+.2f} · B_{labels[b]}", fontsize=11)
    ax_title = fig.add_subplot(tiles[0, :], frameon=False)
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    ax_title.set_title(
        f"(d) ligne a = {row_label} du noyau : les 16 modes d'entrée B_b, chacun "
        f"pondéré par R_{{{row_label},b}} (|R| décroissant)",
        fontsize=14,
    )
    axf = fig.add_subplot(bot[1])
    axf.imshow(f, cmap=CMAP, vmin=-1, vmax=1, origin="upper", extent=(0, 1, 1, 0))
    axf.set_xlabel("s₁ / π")
    axf.set_ylabel("s₀ / π")
    axf.set_title(f"(e) Σ des 16 cartes\n= f_{row_label}(x) = ⟨{row_label}⟩ après U")
    style_axes(axf)

    enc = DenseAngleEncoding({"N": N})
    x = rng.uniform(0, np.pi, size=(200, 2 * N))
    direct = np.array(
        [
            [
                np.trace(pauli_matrix(c) @ U @ enc.rho(xi) @ U.conj().T).real
                for c in pauli_codes(N)
            ]
            for xi in x
        ]
    )
    kernel = dense_monomials(x) @ R.T
    err = float(np.abs(direct - kernel).max())
    axc = fig.add_subplot(bot[2])
    axc.scatter(kernel.ravel(), direct.ravel(), s=5, alpha=0.4, color="#2a78d6")
    axc.plot([-1, 1], [-1, 1], color="k", lw=1)
    axc.set_xlabel("Σ_b R_ab B_b(x)  (noyau)")
    axc.set_ylabel("Tr[P_a U ρ(x) U†]  (direct)")
    axc.set_title(f"(f) 16 sorties × 200 entrées\nécart max {err:.0e}")
    style_axes(axc)
    fig.suptitle(
        "La matrice de transfert de Pauli EST le noyau de Fourier "
        "de l'encodage dense (N = 2)",
        fontsize=20,
    )
    data = {
        "strings": labels,
        "ptm": R.tolist(),
        "orthogonality_error": float(np.abs(R @ R.T - np.eye(16)).max()),
        "row": row_label,
        "max_abs_error": err,
        "seed": seed,
    }
    save_figure(fig, out, "ptm_noyau_fourier", data)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# Figure 8
# ---------------------------------------------------------------------------


def draw_bipartite(ax: Axes, R: np.ndarray, labels: Sequence[str], title: str) -> int:
    """Draw ``R`` as edges from the input modes (left) to the outputs (right)."""
    n = len(labels)
    k_theta, _ = frequencies(N)
    y = np.arange(n)[::-1]
    shown = 0
    for a in range(n):
        for b in range(n):
            v = R[a, b]
            if abs(v) < 0.15:
                continue
            shown += 1
            ax.plot(
                [0, 1],
                [y[b], y[a]],
                color="#c0392b" if v > 0 else "#2a78d6",
                lw=0.5 + 4 * abs(v),
                alpha=0.25 + 0.75 * abs(v),
                solid_capstyle="round",
            )
    for side, x in ((0, -0.06), (1, 1.06)):
        ax.scatter([side] * n, y, s=90, c=[WEIGHT_COLORS[k] for k in k_theta], zorder=3)
        for j in range(n):
            ax.text(
                x,
                y[j],
                labels[j],
                ha="right" if side == 0 else "left",
                va="center",
                fontsize=11,
            )
    ax.text(0, n + 0.2, "mode d'entrée B_b", ha="center", fontsize=12)
    ax.text(1, n + 0.2, "sortie f_a", ha="center", fontsize=12)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1, n + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    return shown


def figure_composition(out: Path, seed: int, p: float) -> dict:
    rng = np.random.default_rng(seed + 1)
    labels = pauli_labels(N)
    C = random_clifford_gate(rng)
    R_c = pauli_transfer_matrix(C)
    layers = [[(0, 1, haar_gate(rng))], [(0, 1, haar_gate(rng))]]
    R1 = pauli_transfer_matrix(layers[0][0][2])
    R2 = pauli_transfer_matrix(layers[1][0][2])
    D = dephasing_ptm(N, p)
    K = circuit_kernel(layers, N, p)
    K0 = circuit_kernel(layers, N, 0.0)

    apply_style()
    fig = plt.figure(figsize=(24, 15), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    draw_ptm(
        fig.add_subplot(gs[0, 0]),
        R_c,
        labels,
        "(a) porte de Clifford : permutation signée\n1 mode → 1 mode",
    )
    draw_ptm(
        fig.add_subplot(gs[0, 1]),
        R1,
        labels,
        "(b) porte Haar U₁ : rotation qui mélange\n1 mode → plusieurs modes",
    )
    draw_ptm(
        fig.add_subplot(gs[0, 2]),
        D,
        labels,
        f"(c) mesure Z oubliée, p = {p} : filtre diagonal\n"
        "(1 − p)^(#X+#Y) : amortit les fréquences en φ",
    )
    draw_ptm(
        fig.add_subplot(gs[0, 3]),
        K,
        labels,
        f"(d) circuit U₁ → mesure → U₂ → mesure :\nK = D R₂ D R₁  (p = {p})",
    )
    n_c = draw_bipartite(
        fig.add_subplot(gs[1, 0]),
        R_c,
        labels,
        "(e) Clifford en graphe : une arête par mode",
    )
    n_h = draw_bipartite(
        fig.add_subplot(gs[1, 1]),
        R1,
        labels,
        "(f) Haar en graphe : un mode alimente\nplusieurs sorties (arêtes |R| > 0,15)",
    )
    s = np.linspace(0, np.pi, 81)
    B = dense_monomials(grid_inputs(s)).reshape(len(s), len(s), 16)
    z0 = labels.index("ZI")
    for j, (Kp, name) in enumerate(((K0, "p = 0"), (K, f"p = {p}"))):
        ax = fig.add_subplot(gs[1, 2 + j])
        ax.imshow(
            B @ Kp[z0],
            cmap=CMAP,
            vmin=-1,
            vmax=1,
            origin="upper",
            extent=(0, 1, 1, 0),
        )
        _, k_phi = frequencies(N)
        phi_share = float((Kp[z0, k_phi > 0] ** 2).sum() / (Kp[z0, 1:] ** 2).sum())
        ax.set_xlabel("s₁ / π")
        ax.set_ylabel("s₀ / π")
        ax.set_title(
            f"({'gh'[j]}) feature ⟨Z₀⟩ du circuit, {name}\n"
            f"part des modes avec φ : {phi_share:.2f}",
            fontsize=14,
        )
        style_axes(ax)
    fig.suptitle(
        "Composer des portes et des mesures = multiplier des noyaux de Fourier (N = 2)",
        fontsize=20,
    )
    data = {
        "strings": labels,
        "clifford_nonzero_per_row": [int((np.abs(r) > 1e-9).sum()) for r in R_c],
        "haar_nonzero_per_row": [int((np.abs(r) > 1e-9).sum()) for r in R1],
        "edges_shown": {"clifford": n_c, "haar": n_h},
        "kernel_p": K.tolist(),
        "kernel_check": float(np.abs(K - D @ R2 @ D @ R1).max()),
        "p": p,
        "seed": seed,
    }
    save_figure(fig, out, "ptm_composition", data)
    plt.close(fig)
    return data


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args(argv)
    figure_kernel(args.out, args.seed)
    figure_composition(args.out, args.seed, args.p)


if __name__ == "__main__":
    main()
