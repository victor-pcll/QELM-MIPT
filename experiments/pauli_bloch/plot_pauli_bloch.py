#!/usr/bin/env python3
"""Figures of the Pauli string <-> Bloch/Fourier monomial isomorphism (issue #364).

Six pedagogical figures for the dense encoding (``DenseAngleEncoding``,
``n = (sin t cos f, sin t sin f, cos t)`` per qubit), each written with its
data by :func:`QELM.utils.plots.io.save_figure`:

1. ``un_qubit``: ``I, X, Y, Z`` as functions of ``(theta, phi)``, on the Bloch
   sphere, and in the 3 x 3 lattice of real trigonometric functions;
2. ``deux_qubits``: the 16 strings of two qubits as a multiplication table;
3. ``matrice_passage``: the change of basis Pauli -> trigonometric monomials
   and the three Gram matrices;
4. ``verification``: ``Tr[P_a rho(x)]`` from the encoding against ``B_a(x)``;
5. ``reconstruction``: a QELM feature rebuilt string by string;
6. ``poids_frequence``: Pauli weight = frequency, and how Haar layers and
   Z measurements move the weight of a feature.

Usage::

    uv run python experiments/pauli_bloch/plot_pauli_bloch.py \
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
from matplotlib.colors import LogNorm  # noqa: E402

from QELM.encoding import DenseAngleEncoding  # noqa: E402
from QELM.evaluation.fourier_framework import (  # noqa: E402
    pauli_codes,
    pauli_matrix,
    pauli_vectors,
)
from QELM.evaluation.pauli_bloch import (  # noqa: E402
    PAULI_LETTERS,
    PAULI_TRIG,
    PHI_NAMES,
    THETA_NAMES,
    brickwork_layers,
    dense_monomials,
    frequencies,
    gram_matrix,
    heisenberg_z,
    passage_matrix,
    pauli_labels,
    reconstruction_order,
    residual_variance,
    sphere_weights,
    trig_label,
    weight_profile,
)
from QELM.utils.plots.io import save_figure  # noqa: E402
from QELM.utils.plots.style import apply_style, style_axes, subplots  # noqa: E402

CMAP = "RdBu_r"
# One color per Pauli weight, the same in every figure.
WEIGHT_COLORS = (
    "#7f7f7f",
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#d63a8a",
    "#6250d6",
    "#8c564b",
)
FORMULAS = ("1", "sin θ cos φ", "sin θ sin φ", "cos θ")


def spiral(s: np.ndarray) -> np.ndarray:
    """Return the inputs ``(theta, phi) = (s, s)`` of one qubit, shape ``(len(s), 2)``.

    For ``s`` in ``[0, pi]`` this path runs from the north to the south pole
    while turning by half a turn, inside the bounds of the dense encoding,
    and ``x, y, z`` are three different functions of ``s``.
    """
    return np.stack([s, s], axis=1)


def grid_inputs(s: np.ndarray) -> np.ndarray:
    """Inputs of two qubits on the spiral, ``(len(s)^2, 4)``, ``s_0`` slowest."""
    s0, s1 = np.meshgrid(s, s, indexing="ij")
    return np.stack([s0.ravel(), s0.ravel(), s1.ravel(), s1.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Figure 1: one qubit
# ---------------------------------------------------------------------------


def figure_one_qubit(out: Path) -> dict:
    apply_style()
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.1, 0.9])
    th = np.linspace(0, np.pi, 121)
    ph = np.linspace(0, 2 * np.pi, 241)
    T, F = np.meshgrid(th, ph, indexing="ij")
    x = np.stack([T.ravel(), F.ravel()], axis=1)
    B = dense_monomials(x).reshape(len(th), len(ph), 4)
    u, v = np.mgrid[0 : np.pi : 60j, 0 : 2 * np.pi : 120j]
    sx, sy, sz = np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)
    on_sphere = (np.ones_like(sx), sx, sy, sz)
    cmap = plt.get_cmap(CMAP)
    im = None
    top = []
    for a in range(4):
        ax = fig.add_subplot(gs[0, a])
        top.append(ax)
        im = ax.imshow(
            B[:, :, a],
            origin="lower",
            extent=(0, 2, 0, 1),
            aspect="auto",
            cmap=CMAP,
            vmin=-1,
            vmax=1,
        )
        ax.axvspan(1, 2, color="white", alpha=0.45, lw=0)
        ax.axvline(1, color="k", ls="--", lw=1)
        ax.set_xlabel("φ / π")
        if a == 0:
            ax.set_ylabel("θ / π")
            ax.text(1.5, 0.5, "hors des\nbornes\nφ ∈ [0, π]", ha="center", fontsize=10)
        ax.set_title(f"{PAULI_LETTERS[a]}   ⟷   {FORMULAS[a]}")
        style_axes(ax)

        ax3 = fig.add_subplot(gs[1, a], projection="3d")
        ax3.plot_surface(
            sx,
            sy,
            sz,
            facecolors=cmap((on_sphere[a] + 1) / 2),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        for vec, name in (((1.9, 0, 0), "x"), ((0, 1.9, 0), "y"), ((0, 0, 1.6), "z")):
            ax3.quiver(0, 0, 0, *vec, color="k", lw=1.2, arrow_length_ratio=0.08)
            ax3.text(*(1.08 * np.array(vec)), name, fontsize=12)
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_axis_off()
        ax3.view_init(elev=20, azim=45)
        ax3.set_title(
            "ℓ = 0 : constante" if a == 0 else f"ℓ = 1 : coordonnée {'xyz'[a - 1]}",
            fontsize=13,
        )

        axl = fig.add_subplot(gs[2, a])
        lattice = np.zeros((3, 3))
        for b in range(4):
            lattice[PAULI_TRIG[b]] = 0.35
        lattice[PAULI_TRIG[a]] = 1.0
        axl.imshow(lattice, cmap="Blues", vmin=0, vmax=1.1)
        for uu in range(3):
            for vv in range(3):
                name = " ".join(n for n in (THETA_NAMES[uu], PHI_NAMES[vv]) if n != "1")
                if (uu, vv) in PAULI_TRIG:
                    letter = PAULI_LETTERS[PAULI_TRIG.index((uu, vv))]
                    txt = f"{letter}\n{name or '1'}"
                    color = "white" if (uu, vv) == PAULI_TRIG[a] else "k"
                else:
                    txt, color = f"({name})", "#9a9a9a"
                axl.text(
                    vv, uu, txt, ha="center", va="center", fontsize=10, color=color
                )
        axl.set_xticks(range(3), PHI_NAMES)
        axl.set_yticks(range(3), THETA_NAMES)
        axl.set_xlabel("facteur en φ")
        if a == 0:
            axl.set_ylabel("facteur en θ", labelpad=-2)
        axl.set_title("4 des 9 fonctions de (θ, φ)", fontsize=13)
        style_axes(axl)
    assert im is not None
    fig.colorbar(im, ax=top, shrink=0.9, label="Tr[P ρ(θ, φ)]")
    fig.suptitle(
        "Un qubit : chaque matrice de Pauli est une fonction de l'angle encodé"
    )
    save_figure(
        fig,
        out,
        "un_qubit",
        {
            "pauli": list(PAULI_LETTERS),
            "formula": list(FORMULAS),
            "trig_cell": [list(c) for c in PAULI_TRIG],
        },
    )
    plt.close(fig)
    return {}


# ---------------------------------------------------------------------------
# Figure 2: two qubits as a multiplication table
# ---------------------------------------------------------------------------


def figure_two_qubits(out: Path) -> dict:
    s = np.linspace(0, np.pi, 81)
    one = dense_monomials(spiral(s))  # (len(s), 4)
    B = dense_monomials(grid_inputs(s)).reshape(len(s), len(s), 16)
    fig, axes = subplots(5, 5, figsize=(16, 15.5))
    k_theta, _ = frequencies(2)
    labels = pauli_labels(2)
    corner = axes[0, 0]
    corner.remove()
    ax3 = fig.add_subplot(5, 5, 1, projection="3d")
    u, v = np.mgrid[0 : np.pi : 30j, 0 : 2 * np.pi : 60j]
    ax3.plot_wireframe(
        np.sin(u) * np.cos(v),
        np.sin(u) * np.sin(v),
        np.cos(u),
        color="#cccccc",
        lw=0.4,
        rstride=3,
        cstride=3,
    )
    n = dense_monomials(spiral(s))[:, 1:]
    ax3.plot(n[:, 0], n[:, 1], n[:, 2], color="k", lw=2.5)
    ax3.scatter(*n[0], color="#2a78d6", s=30)
    ax3.scatter(*n[-1], color="#eb6834", s=30)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_axis_off()
    ax3.set_title("chemin θ = φ = s", fontsize=12)
    im = None
    for b in range(4):
        ax = axes[0, b + 1]
        ax.plot(s / np.pi, one[:, b], color="k")
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(f"qubit 1 : {PAULI_LETTERS[b]}", fontsize=14)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("s₁ / π", fontsize=11)
    for a in range(4):
        ax = axes[a + 1, 0]
        ax.plot(one[:, a], s / np.pi, color="k")
        ax.axvline(0, color="#bbbbbb", lw=0.8)
        ax.set_xlim(-1.15, 1.15)
        ax.invert_yaxis()
        ax.set_ylabel(f"qubit 0 : {PAULI_LETTERS[a]}\ns₀ / π", fontsize=13)
        ax.set_yticks([0, 0.5, 1])
        for b in range(4):
            idx = 4 * a + b
            axg = axes[a + 1, b + 1]
            im = axg.imshow(
                B[:, :, idx],
                origin="upper",
                extent=(0, 1, 1, 0),
                cmap=CMAP,
                vmin=-1,
                vmax=1,
            )
            axg.set_xticks([])
            axg.set_yticks([])
            color = WEIGHT_COLORS[k_theta[idx]]
            for spine in axg.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3.5)
            axg.text(
                0.04,
                0.06,
                labels[idx],
                transform=axg.transAxes,
                fontsize=14,
                fontweight="bold",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "lw": 0},
            )
    assert im is not None
    fig.colorbar(im, ax=axes[1:, 1:], shrink=0.6, label="B_ab(s₀, s₁) = n₀^a · n₁^b")
    handles = [
        plt.Line2D([], [], color=WEIGHT_COLORS[k], lw=4, label=f"poids |a| = {k}")
        for k in range(3)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13)
    fig.suptitle(
        "Deux qubits : la chaîne P_a ⊗ P_b est le produit des deux fonctions à un qubit"
    )
    save_figure(fig, out, "deux_qubits", {"strings": labels, "weight": k_theta})
    plt.close(fig)
    return {}


# ---------------------------------------------------------------------------
# Figure 3: change-of-basis and Gram matrices
# ---------------------------------------------------------------------------


def figure_passage(out: Path) -> dict:
    n = 2
    labels = pauli_labels(n)
    M = passage_matrix(n)
    hs = np.array(
        [
            [
                np.trace(pauli_matrix(a) @ pauli_matrix(b)).real / 4
                for b in pauli_codes(n)
            ]
            for a in pauli_codes(n)
        ]
    )
    g_sphere = gram_matrix(n, "sphere")
    g_box = gram_matrix(n, "box")
    apply_style()
    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    k_theta, _ = frequencies(n)
    rows, cols = np.nonzero(M)
    # The 81 monomials as a 9 x 9 table: row = factor of qubit 0, column =
    # factor of qubit 1 (index 3 u + v of f_u(theta) g_v(phi)).
    table = np.full((9, 9), -1)
    for a, col in zip(rows, cols, strict=True):
        table[col // 9, col % 9] = a
    ax.imshow(np.zeros((9, 9)), cmap="Greys", vmin=0, vmax=1)
    for r in range(9):
        for q in range(9):
            a = int(table[r, q])
            if a < 0:
                ax.text(
                    q, r, "·", ha="center", va="center", color="#bbbbbb", fontsize=16
                )
                continue
            ax.add_patch(
                plt.Rectangle(
                    (q - 0.5, r - 0.5),
                    1,
                    1,
                    color=WEIGHT_COLORS[k_theta[a]],
                    alpha=0.85,
                )
            )
            ax.text(
                q,
                r,
                labels[a],
                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold",
            )
    site_names = [
        " ".join(nm for nm in (THETA_NAMES[u], PHI_NAMES[v]) if nm != "1") or "1"
        for u in range(3)
        for v in range(3)
    ]
    ax.set_xticks(
        range(9),
        [t.replace("θ", "θ₁").replace("φ", "φ₁") for t in site_names],
        rotation=30,
        fontsize=10,
    )
    ax.set_yticks(
        range(9),
        [t.replace("θ", "θ₀").replace("φ", "φ₀") for t in site_names],
        fontsize=10,
    )
    ax.set_xlabel("facteur du qubit 1")
    ax.set_ylabel("facteur du qubit 0")
    ax.set_title(
        "(a) Matrice de passage, rangée en table 9 × 9 des 81 monômes "
        "f(θ₀, φ₀) · g(θ₁, φ₁) :\nchaque chaîne de Pauli occupe UNE case, "
        "65 cases (·) ne sont atteintes par aucune chaîne",
        fontsize=14,
    )
    handles = [
        plt.Line2D([], [], color=WEIGHT_COLORS[k], lw=8, label=f"|a| = {k}")
        for k in range(n + 1)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    style_axes(ax)
    panels = (
        (hs, "(b) Tr[P_a P_b] / 4 : base orthonormée\ndes opérateurs", None),
        (
            g_sphere,
            "(c) E[B_a B_b], sphère de Bloch uniforme :\ndiagonale w_a = 3^-|a|",
            None,
        ),
        (
            g_box,
            "(d) E[B_a B_b], θ, φ uniformes dans [0, π] :\n"
            "les monômes ne sont plus orthogonaux",
            None,
        ),
    )
    for j, (mat, title, _) in enumerate(panels):
        axg = fig.add_subplot(gs[1, j])
        if j == 0:
            im = axg.imshow(mat, cmap=CMAP, vmin=-1, vmax=1)
        else:
            im = axg.imshow(
                np.abs(mat) + 1e-6, cmap="magma_r", norm=LogNorm(vmin=1e-3, vmax=1)
            )
        axg.set_xticks(range(16), labels, rotation=90, fontsize=9)
        axg.set_yticks(range(16), labels, fontsize=9)
        axg.set_title(title, fontsize=14)
        fig.colorbar(im, ax=axg, shrink=0.75, label="valeur" if j == 0 else "|valeur|")
        style_axes(axg)
    fig.suptitle(
        "Pauli → Fourier : un changement de base, et le produit scalaire de l'entrée"
    )
    off = g_box - np.diag(np.diag(g_box))
    data = {
        "strings": labels,
        "trig_columns": [trig_label(c, n) for c in cols],
        "max_offdiag_hs": float(np.abs(hs - np.eye(16)).max()),
        "sphere_diag": np.diag(g_sphere).tolist(),
        "box_diag": np.diag(g_box).tolist(),
        "box_max_offdiag": float(np.abs(off).max()),
    }
    save_figure(fig, out, "matrice_passage", data)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# Figure 4: numerical check
# ---------------------------------------------------------------------------


def figure_verification(out: Path, seed: int) -> dict:
    n = 3
    rng = np.random.default_rng(seed)
    enc = DenseAngleEncoding({"N": n})
    x = rng.uniform(0, np.pi, size=(300, 2 * n))
    rhos = np.array([enc.rho(xi) for xi in x])
    trace = pauli_vectors(rhos).real * 2**n  # Tr[rho P_a]
    B = dense_monomials(x)
    diff = np.abs(trace - B)
    k_theta, _ = frequencies(n)
    A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    obs = (A + A.conj().T) / 4
    c = pauli_vectors(obs).real
    x0, x1 = rng.uniform(0, np.pi, size=(2, 2 * n))
    s = np.linspace(0, 1, 200)
    path = x0[None] + s[:, None] * (x1 - x0)[None]
    exact = np.array([np.trace(obs @ enc.rho(xi)).real for xi in path])
    Bp = dense_monomials(path)
    series = Bp @ c
    top = np.argsort(-np.abs(c[1:] * Bp[:, 1:]).max(axis=0))[:5] + 1
    labels = pauli_labels(n)

    fig, axes = subplots(1, 3, figsize=(19, 6.2))
    ax = axes[0]
    for k in range(n + 1):
        sel = k_theta == k
        ax.scatter(
            B[:, sel].ravel(),
            trace[:, sel].ravel(),
            s=6,
            alpha=0.5,
            color=WEIGHT_COLORS[k],
            label=f"|a| = {k}",
        )
    ax.plot([-1, 1], [-1, 1], color="k", lw=1)
    ax.set_xlabel("monôme B_a(x) = Π_i n_i^{a_i} (formule)")
    ax.set_ylabel("Tr[P_a ρ(x)] (état de DenseAngleEncoding)")
    ax.set_title("(a) N = 3 : 64 chaînes × 300 entrées")
    ax.legend(markerscale=3, loc="upper left")
    ax = axes[1]
    bins = np.logspace(-18, -14, 30)
    ax.hist(np.maximum(diff.ravel(), 1e-18), bins=bins, color="#2a78d6")
    ax.set_xscale("log")
    ax.set_xlabel("|Tr[P_a ρ(x)] − B_a(x)|")
    ax.set_ylabel("nombre de couples (a, x)")
    ax.set_title(f"(b) écart maximal : {diff.max():.1e}")
    zeros = int((diff == 0).sum())
    ax.text(
        0.97,
        0.95,
        f"{zeros} écarts exactement nuls\n(placés à 1e-18)\nprécision machine ≈ 1e-16",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    ax = axes[2]
    ax.plot(s, exact, color="k", lw=4, alpha=0.35, label="Tr[O ρ(x)] (direct)")
    ax.plot(s[::6], series[::6], "o", color="#eb6834", ms=6, label="Σ_a c_a B_a(x)")
    for a in top:
        ax.plot(
            s, c[a] * Bp[:, a], lw=1.2, alpha=0.8, label=f"c_{labels[a]} B_{labels[a]}"
        )
    ax.set_xlabel("position le long d'un chemin x(s) dans l'espace des angles")
    ax.set_ylabel("valeur")
    ax.set_title("(c) observable hermitienne aléatoire O")
    ax.legend(fontsize=9, ncol=2, loc="best")
    fig.suptitle(
        "Vérification numérique : l'observable Pauli sur l'état encodé "
        "= la fonction de Bloch"
    )
    data = {
        "max_abs_error_strings": float(diff.max()),
        "max_abs_error_observable": float(np.abs(exact - series).max()),
        "seed": seed,
    }
    save_figure(fig, out, "verification", data)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# Figure 5: step-by-step reconstruction of a QELM feature
# ---------------------------------------------------------------------------


def figure_reconstruction(out: Path, seed: int, steps: Sequence[int]) -> dict:
    n = 2
    layers = brickwork_layers(n, 2, np.random.default_rng(seed))
    c = heisenberg_z(layers, n, 0, 0.0)
    w = sphere_weights(n)
    order = reconstruction_order(c, w)
    if abs(c[0]) < 1e-12:
        # Z is traceless: the identity has no weight, start with the others.
        order = order[1:]
    res = residual_variance(c, w, order)
    labels = pauli_labels(n)
    s = np.linspace(0, np.pi, 101)
    B = dense_monomials(grid_inputs(s)).reshape(len(s), len(s), 16)
    target = B @ c
    lim = float(np.abs(target).max())
    apply_style()
    fig = plt.figure(figsize=(19, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95])
    maps = [("cible : ⟨Z₀⟩ après 2 couches Haar", target)]
    for m in steps:
        part = B[:, :, order[:m]] @ c[order[:m]]
        added = labels[order[m - 1]]
        maps.append((f"{m} chaîne{'s' if m > 1 else ''} (+ {added})", part))
    im = None
    for j, (title, z) in enumerate(maps[:8]):
        ax = fig.add_subplot(gs[j // 4, j % 4])
        im = ax.imshow(
            z, origin="upper", extent=(0, 1, 1, 0), cmap=CMAP, vmin=-lim, vmax=lim
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("s₁ / π")
        ax.set_ylabel("s₀ / π")
        if j:
            err = float(np.sqrt(np.mean((z - target) ** 2)))
            ax.text(
                0.03,
                0.05,
                f"écart rms = {err:.3f}",
                transform=ax.transAxes,
                fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.85, "lw": 0},
            )
        style_axes(ax)
    assert im is not None
    fig.colorbar(im, ax=fig.axes[:8], shrink=0.6, label="valeur de la feature")
    k_theta, _ = frequencies(n)
    ax = fig.add_subplot(gs[2, :3])
    power = w * c**2
    ax.bar(
        range(len(order)),
        power[order],
        color=[WEIGHT_COLORS[k] for k in k_theta[order]],
    )
    ax.set_xticks(range(len(order)), [labels[a] for a in order], rotation=0)
    ax.set_yscale("log")
    ax.set_ylim(max(power[power > 0].min() / 3, 1e-8), 1.5 * power.max())
    ax.set_ylabel("puissance w_a c_a²")
    ax.set_title("coefficients de Fourier/Pauli, dans l'ordre où on les ajoute")
    handles = [
        plt.Line2D([], [], color=WEIGHT_COLORS[k], lw=6, label=f"|a| = {k}")
        for k in range(n + 1)
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3)
    style_axes(ax)
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(range(len(res)), res, "-o", color="k")
    ax.set_xlabel("nombre de chaînes gardées m")
    ax.set_ylabel("variance restante (sphère)")
    ax.set_title("Parseval : 1 − Σ_{gardées} w_a c_a² / Var")
    style_axes(ax)
    fig.suptitle(
        "Reconstruction d'une feature du QELM chaîne de Pauli par chaîne de Pauli "
        "(N = 2, chemin θ_i = φ_i = s_i)"
    )
    data = {
        "order": [labels[a] for a in order],
        "coefficients": {labels[a]: float(c[a]) for a in order},
        "power": {labels[a]: float(power[a]) for a in order},
        "residual_variance": res.tolist(),
        "seed": seed,
    }
    save_figure(fig, out, "reconstruction", data)
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# Figure 6: Pauli weight <-> frequency, and what the circuit does to it
# ---------------------------------------------------------------------------


def _cut(n: int, theta: np.ndarray, phi: float) -> np.ndarray:
    x = np.zeros((len(theta), 2 * n))
    x[:, 0::2] = theta[:, None]
    x[:, 1::2] = phi
    return x


def figure_weight_frequency(
    out: Path,
    seed: int,
    n: int,
    circuits: int,
    depths: Sequence[int],
    ps: Sequence[float],
) -> dict:
    labels4 = pauli_labels(4)
    picks = ["IIII", "ZIII", "XZII", "ZXZI", "XXZX"]
    theta = 2 * np.pi * np.arange(256) / 256
    B = dense_monomials(_cut(4, theta, 0.0))
    rng = np.random.default_rng(seed)
    site = n // 2
    w = sphere_weights(n)
    k_theta, _ = frequencies(n)
    deep = max(depths)
    circ = [brickwork_layers(n, deep, rng) for _ in range(circuits)]

    def share(c: np.ndarray) -> np.ndarray:
        P = weight_profile(c, n, w)
        P[0] = 0.0
        return P / P.sum()

    by_depth = np.array(
        [
            np.mean([share(heisenberg_z(L[:t], n, site, 0.0)) for L in circ], axis=0)
            for t in depths
        ]
    )
    by_p = np.array(
        [
            np.mean([share(heisenberg_z(L, n, site, p)) for L in circ], axis=0)
            for p in ps
        ]
    )
    mean_by_p = by_p @ np.arange(n + 1)
    cut_theta = np.linspace(0, 2 * np.pi, 400)
    Bc = dense_monomials(_cut(n, cut_theta, np.pi / 4))
    show_t = [t for t in (0, 1, 2, deep) if t in depths]

    fig, axes = subplots(2, 3, figsize=(20, 11.5))
    ax = axes[0, 0]
    for j, lab in enumerate(picks):
        a = labels4.index(lab)
        k = int(np.count_nonzero(pauli_codes(4)[a]))
        ax.plot(
            theta / np.pi, B[:, a] + 2.4 * (len(picks) - 1 - j), color=WEIGHT_COLORS[k]
        )
        ax.text(2.03, 2.4 * (len(picks) - 1 - j), lab, va="center", fontsize=12)
    ax.set_xlim(0, 2.3)
    ax.set_yticks([])
    ax.set_xlabel("θ / π  (tous les qubits : θ_i = θ, φ_i = 0)")
    ax.set_title("(a) B_a le long d'une coupe")
    ax = axes[0, 1]
    for j, lab in enumerate(picks):
        a = labels4.index(lab)
        k = int(np.count_nonzero(pauli_codes(4)[a]))
        spec = np.abs(np.fft.rfft(B[:, a]))[:7] / 256
        spec[1:] *= 2
        ax.vlines(np.arange(7) + 0.08 * (j - 2), 0, spec, color=WEIGHT_COLORS[k], lw=4)
        ax.plot([], [], color=WEIGHT_COLORS[k], lw=4, label=f"{lab}  (|a| = {k})")
    ax.set_xlabel("fréquence en θ")
    ax.set_ylabel("|amplitude de Fourier|")
    ax.set_title("(b) fréquence maximale = poids |a|")
    ax.legend()
    ax = axes[0, 2]
    k_all, kphi_all = frequencies(4)
    counts = np.zeros((5, 5))
    for kt, kp in zip(k_all, kphi_all, strict=True):
        counts[kp, kt] += 1
    im = ax.imshow(counts, origin="lower", cmap="Blues")
    for kt in range(5):
        for kp in range(5):
            if counts[kp, kt]:
                ax.text(
                    kt, kp, int(counts[kp, kt]), ha="center", va="center", fontsize=11
                )
    ax.set_xlabel("fréquence en θ = |a|")
    ax.set_ylabel("fréquence en φ = #X + #Y")
    ax.set_title("(c) les 256 chaînes de N = 4 par fréquence")
    fig.colorbar(im, ax=ax, shrink=0.8, label="nombre de chaînes")

    ax = axes[1, 0]
    im = ax.imshow(
        by_depth[:, 1:].T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(min(depths) - 0.5, max(depths) + 0.5, 0.5, n + 0.5),
    )
    ax.set_xlabel("nombre de couches Haar t (p = 0)")
    ax.set_ylabel("poids |a|")
    ax.set_title(f"(d) Z_{site} en Heisenberg :\nla puissance monte en fréquence")
    fig.colorbar(im, ax=ax, shrink=0.8, label="part de la variance")
    ax = axes[1, 1]
    for t in show_t:
        c = heisenberg_z(circ[0][:t], n, site, 0.0)
        ax.plot(cut_theta / np.pi, Bc @ c, label=f"t = {t}")
    ax.set_xlabel(f"θ / π  (θ_i = θ, φ_i = π/4, N = {n})")
    ax.set_ylabel(f"⟨Z_{site}⟩ final")
    ax.set_title("(e) plus de couches :\nla feature oscille plus vite")
    ax.legend()
    ax = axes[1, 2]
    for kk in range(1, n + 1):
        ax.plot(
            ps,
            by_p[:, kk],
            "-o",
            color=WEIGHT_COLORS[min(kk, 6)],
            ms=4,
            label=f"|a| = {kk}",
        )
    ax.set_xlabel("probabilité de mesure p")
    ax.set_ylabel("part de la variance")
    ax2 = ax.twinx()
    ax2.plot(ps, mean_by_p, "--", color="k", lw=2)
    ax2.set_ylabel("poids moyen (tirets)")
    ax.set_title(f"(f) T = {deep} couches : les mesures\ncoupent les hautes fréquences")
    ax.legend(fontsize=9, ncol=2, loc="center right")
    fig.suptitle(
        "Poids de Pauli = fréquence : la dynamique déplace la feature "
        "dans le spectre de Fourier"
    )
    data = {
        "n": n,
        "site": site,
        "circuits": circuits,
        "depths": list(depths),
        "share_by_depth": by_depth.tolist(),
        "ps": list(ps),
        "share_by_p": by_p.tolist(),
        "mean_weight_by_p": mean_by_p.tolist(),
        "seed": seed,
    }
    save_figure(fig, out, "poids_frequence", data)
    plt.close(fig)
    return data


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=6, help="qubits of figure 6 (d-f)")
    parser.add_argument("--circuits", type=int, default=20)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args(argv)
    figure_one_qubit(args.out)
    figure_two_qubits(args.out)
    figure_passage(args.out)
    figure_verification(args.out, args.seed)
    figure_reconstruction(args.out, args.seed, steps=(1, 2, 3, 4, 6, 9, 15))
    ps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    figure_weight_frequency(
        args.out, args.seed, args.n, args.circuits, list(range(args.depth + 1)), ps
    )


if __name__ == "__main__":
    main()
