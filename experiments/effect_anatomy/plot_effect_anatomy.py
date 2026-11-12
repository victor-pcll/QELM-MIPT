"""Pedagogical figures: what the Kraus operator K_m and the effect E_m are (#329).

Everything is computed exactly with dense matrices at small ``N`` (no run
folder to read), in the convention of ``experiments/effect_rank``: open
brickwork, one layer = even bonds then odd bonds, then a ``Z`` measurement of
the sites of the mask; mid-circuit record only, no final readout. Writes,
in ``--out``:

- ``anatomy_clifford.png`` / ``anatomy_haar.png``: ``N = 3``, ``T = 3``,
  one circuit, four records with more and more constraints. Per record: the
  mask and the outcomes, ``|K_m|``, ``|E_m|``, the spectrum of ``E_m``, its
  Pauli coefficients ``c_a / c_0`` and the distribution of
  ``Pr(m | x) / Pr(m)`` over random product inputs;
- ``kraus_vs_effect.png``: ``K_m`` acts on the state, ``E_m = K^dagger K``
  on the input, ``K K^dagger`` is the post-measurement state of the mixed
  input; same spectrum, different eigenvectors;
- ``rank_strings_weight.png``: ``N = 4``, ``T = 4``, Born-drawn records over
  a grid of ``p``, Clifford and Haar: ``log2(1 + W)`` against the rank and
  against the number of non-zero Pauli strings, and against ``p``;
- ``summary.json``: every number quoted in the README.

Usage::

    uv run python experiments/effect_anatomy/plot_effect_anatomy.py \
        --out experiments/effect_anatomy/figures
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from QELM.evaluation.effect_anatomy import (  # noqa: E402
    EffectAnatomy,
    anatomy,
    kraus_operator,
    likelihood_ratio,
    pauli_labels,
    product_states,
    sample_outcomes,
)
from QELM.evaluation.effect_rank import brick_pairs  # noqa: E402
from QELM.models.circuits.clifford import random_clifford_gate  # noqa: E402
from QELM.models.circuits.haar import haar_gate  # noqa: E402

plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 9}
)

CLIFFORD = "#1f5fa8"
HAAR = "#c2571a"
GateDraw = Callable[[np.random.Generator], np.ndarray]


def draw_gates(n: int, depth: int, gate: GateDraw, rng: np.random.Generator) -> list:
    """Gates ``[t][g]`` of an open brickwork."""
    return [[gate(rng) for _ in brick_pairs(n)] for _ in range(depth)]


def pick_masks(
    gates: list, n: int, depth: int, ranks: list[int], seed: int
) -> list[np.ndarray]:
    """One mask per target rank of ``E_m`` (searched over random masks).

    The rank of a Clifford effect does not depend on the outcomes, so the
    search uses the all-zero record; the masks are nested in the number of
    measurements only through the target ranks.
    """
    rng = np.random.default_rng(seed)
    out = []
    for target in ranks:
        if target == 2**n:
            out.append(np.zeros((depth, n), dtype=bool))
            continue
        for _ in range(10_000):
            p = rng.uniform(0.1, 0.6)
            mask = rng.random((depth, n)) < p
            if mask.sum() == 0:
                continue
            kraus = kraus_operator(gates, mask, np.zeros((depth, n), dtype=int))
            if np.trace(kraus.conj().T @ kraus).real < 1e-9:
                continue
            if anatomy(kraus).rank == target:
                out.append(mask)
                break
        else:
            raise RuntimeError(f"no mask of rank {target}")
    return out


def draw_mask(ax: plt.Axes, mask: np.ndarray, outcomes: np.ndarray) -> None:
    """Space-time grid: one column per site, one row per layer, outcomes written."""
    depth, n = mask.shape
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(depth - 0.4, -0.6)
    for site in range(n):
        ax.plot([site, site], [-0.5, depth - 0.5], color="0.75", lw=1, zorder=0)
    for t in range(depth):
        for site in range(n):
            if mask[t, site]:
                ax.add_patch(
                    plt.Rectangle(
                        (site - 0.32, t - 0.32), 0.64, 0.64, color="0.2", zorder=2
                    )
                )
                ax.text(
                    site,
                    t,
                    str(int(outcomes[t, site])),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                    zorder=3,
                )
            else:
                ax.plot(site, t, "o", color="0.75", ms=4, zorder=1)
    ax.set_xticks(range(n), [f"q{i}" for i in range(n)])
    ax.set_yticks(range(depth), [f"t={t + 1}" for t in range(depth)])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def heatmap(ax: plt.Axes, mat: np.ndarray, title: str) -> None:
    """``|mat|`` on a fixed scale 0..max, computational-basis labels."""
    dim = mat.shape[0]
    n = dim.bit_length() - 1
    im = ax.imshow(np.abs(mat), cmap="Blues", vmin=0, vmax=np.abs(mat).max())
    labels = [format(i, f"0{n}b") for i in range(dim)]
    step = 1 if dim <= 8 else 4
    ax.set_xticks(range(0, dim, step), labels[::step], rotation=90, fontsize=7)
    ax.set_yticks(range(0, dim, step), labels[::step], fontsize=7)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def spectrum(ax: plt.Axes, a: EffectAnatomy, color: str) -> None:
    """Eigenvalues of ``E_m`` (``Tr = 2^N``), rank and participation ratio."""
    dim = len(a.eigenvalues)
    ax.bar(range(dim), a.eigenvalues, color=color, width=0.7)
    ax.axhline(1, color="0.5", ls=":", lw=1)
    ax.set_xticks(range(dim))
    ax.set_ylim(0, dim * 1.08)
    ax.set_xlabel("indice de la valeur propre", fontsize=8)
    ax.set_title(
        f"spectre de E_m : rang {a.rank}, r_2 = {a.participation:.2f}", fontsize=9
    )


def pauli_bars(ax: plt.Axes, a: EffectAnatomy, color: str, n: int) -> None:
    """``c_a / c_0`` on the ``4^N`` strings; the non-zero ones are named if few."""
    labels = pauli_labels(n)
    c = a.coefficients
    ax.bar(range(len(c)), c, color=color, width=0.8)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlim(-1, len(c))
    ax.set_xticks([])
    nz = np.flatnonzero(np.abs(c) > 1e-9)
    if len(nz) <= 8:
        for i in nz:
            ax.text(
                i,
                c[i] + (0.05 if c[i] > 0 else -0.05),
                labels[i],
                ha="center",
                va="bottom" if c[i] > 0 else "top",
                fontsize=6.5,
            )
    ax.set_xlabel(
        f"les {len(c)} chaînes de Pauli (ordre IXYZ, q0 en premier)", fontsize=8
    )
    ax.set_title(
        f"c_a / c_0 : {a.n_strings} chaînes non nulles, "
        f"W = {a.weight:.2f}, 1 + W = {1 + a.weight:.2f}",
        fontsize=9,
    )


def ratio_hist(ax: plt.Axes, a: EffectAnatomy, color: str, states: np.ndarray) -> dict:
    """Distribution of ``Pr(m | x) / Pr(m)`` over random product inputs."""
    r = likelihood_ratio(a.effect, states)
    dim = a.effect.shape[0]
    bins = np.linspace(0, dim, 49)
    if np.ptp(r) < 1e-9:
        ax.axvline(1, color=color, lw=3)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(1.2, 0.5, "Pr(m | x) = Pr(m)\npour toute entrée", fontsize=8)
    else:
        ax.hist(r, bins=bins, color=color, density=True)
        ax.set_yscale("log")
    ax.axvline(a.eigenvalues[0], color="0.3", ls="--", lw=1)
    ax.set_xlim(0, dim)
    ax.set_xlabel("Pr(m | x) / Pr(m), entrées produit aléatoires", fontsize=8)
    ax.set_title(
        f"filtre : max = {r.max():.2f} (borne λ_max = {a.eigenvalues[0]:.2f})",
        fontsize=9,
    )
    return {"ratio_max": float(r.max()), "ratio_std": float(r.std())}


def anatomy_figure(
    gate: GateDraw, color: str, name: str, title: str, out: Path, seed: int
) -> list[dict]:
    """Four records of one ``N = 3`` circuit, one row each."""
    n, depth = 3, 3
    rng = np.random.default_rng(seed)
    gates = draw_gates(n, depth, gate, rng)
    # Masks chosen on the Clifford circuit of the same seed so that both
    # figures show the same measurement pattern.
    cliff = draw_gates(n, depth, random_clifford_gate, np.random.default_rng(seed))
    masks = pick_masks(cliff, n, depth, [8, 4, 2, 1], seed)
    states = product_states(n, 20_000, np.random.default_rng(seed + 1))
    fig, axes = plt.subplots(
        len(masks),
        6,
        figsize=(22, 3.3 * len(masks)),
        gridspec_kw={"width_ratios": [0.7, 1, 1, 1, 2.2, 1.4]},
    )
    rows = []
    for row, mask in enumerate(masks):
        outcomes = sample_outcomes(gates, mask, np.random.default_rng(seed + 10 + row))
        a = anatomy(kraus_operator(gates, mask, outcomes))
        ax = axes[row]
        draw_mask(ax[0], mask, outcomes)
        ax[0].set_title(f"{int(mask.sum())} mesures, Pr(m) = {a.prob:.3f}", fontsize=9)
        heatmap(ax[1], a.kraus / np.sqrt(a.prob), "|K_m| / √Pr(m)  (agit sur l'état)")
        heatmap(ax[2], a.effect, "|E_m| = |K†K|, Tr = 2^N  (agit sur l'entrée)")
        spectrum(ax[3], a, color)
        pauli_bars(ax[4], a, color, n)
        stats = ratio_hist(ax[5], a, color, states)
        rows.append(
            {
                "measurements": int(mask.sum()),
                "mask": mask.astype(int).tolist(),
                "outcomes": outcomes.tolist(),
                "prob": a.prob,
                "rank": a.rank,
                "participation": a.participation,
                "eigenvalues": a.eigenvalues.round(6).tolist(),
                "n_strings": a.n_strings,
                "weight": a.weight,
                "nonzero_strings": {
                    pauli_labels(n)[i]: round(float(a.coefficients[i]), 6)
                    for i in np.flatnonzero(np.abs(a.coefficients) > 1e-9)
                    if a.n_strings <= 16
                },
                **stats,
            }
        )
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def kraus_vs_effect(out: Path, seed: int) -> dict:
    """``K``, ``K^dagger K`` (input side) and ``K K^dagger`` (output side)."""
    n, depth = 3, 3
    cliff = draw_gates(n, depth, random_clifford_gate, np.random.default_rng(seed))
    mask = pick_masks(cliff, n, depth, [2], seed + 5)[0]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    summary = {"mask": mask.astype(int).tolist()}
    for row, (gate, color, label) in enumerate(
        ((random_clifford_gate, CLIFFORD, "Clifford"), (haar_gate, HAAR, "Haar"))
    ):
        gates = draw_gates(n, depth, gate, np.random.default_rng(seed))
        outcomes = sample_outcomes(gates, mask, np.random.default_rng(seed + 20))
        a = anatomy(kraus_operator(gates, mask, outcomes))
        k = a.kraus / np.linalg.norm(a.kraus)
        e_in = k.conj().T @ k
        rho_out = k @ k.conj().T
        heatmap(axes[row, 0], k, f"{label} : |K_m| (sortie × entrée)")
        heatmap(axes[row, 1], e_in, "|K†K| = E_m : quelles entrées ?")
        heatmap(axes[row, 2], rho_out, "|K K†| : état après mesure (entrée I/2^N)")
        ev_in = np.sort(np.linalg.eigvalsh(e_in))[::-1]
        ev_out = np.sort(np.linalg.eigvalsh(rho_out))[::-1]
        x = np.arange(2**n)
        axes[row, 3].bar(x - 0.18, ev_in, 0.36, color=color, label="E_m = K†K")
        axes[row, 3].bar(x + 0.18, ev_out, 0.36, color="0.6", label="K K†")
        axes[row, 3].set_xticks(x)
        axes[row, 3].set_title("mêmes valeurs propres", fontsize=9)
        axes[row, 3].legend(frameon=False, fontsize=8)
        # Eigenvectors: overlap of the top eigenvector of each with the basis.
        _, vin = np.linalg.eigh(e_in)
        _, vout = np.linalg.eigh(rho_out)
        summary[label] = {
            "outcomes": outcomes.tolist(),
            "eigenvalues": ev_in.round(6).tolist(),
            "spectra_equal": bool(np.allclose(ev_in, ev_out, atol=1e-10)),
            "commutator_norm": float(np.linalg.norm(e_in @ rho_out - rho_out @ e_in)),
            "top_vectors_overlap": float(abs(vin[:, -1].conj() @ vout[:, -1])),
        }
    fig.tight_layout()
    fig.savefig(out / "kraus_vs_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def statistics(out: Path, seed: int, circuits: int, records: int) -> dict:
    """``N = 4``, ``T = 4``: rank, number of strings and ``W`` over Born records."""
    n, depth = 4, 4
    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    data: dict[str, list[dict]] = {"Clifford": [], "Haar": []}
    rng = np.random.default_rng(seed)
    for label, gate in (("Clifford", random_clifford_gate), ("Haar", haar_gate)):
        for p in grid:
            for _ in range(circuits):
                gates = draw_gates(n, depth, gate, rng)
                mask = rng.random((depth, n)) < p
                for _ in range(records):
                    o = sample_outcomes(gates, mask, rng)
                    a = anatomy(kraus_operator(gates, mask, o))
                    data[label].append(
                        {
                            "p": p,
                            "rank": a.rank,
                            "participation": a.participation,
                            "n_strings": a.n_strings,
                            "weight": a.weight,
                        }
                    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    jit = np.random.default_rng(0)
    summary: dict = {"n": n, "depth": depth, "grid": grid}
    for label, color, dx in (("Clifford", CLIFFORD, -0.08), ("Haar", HAAR, 0.08)):
        rows = data[label]
        lw = np.log2([1 + r["weight"] for r in rows])
        lr = np.log2([r["rank"] for r in rows])
        ls = np.log2([r["n_strings"] for r in rows])
        lp = np.log2([r["participation"] for r in rows])
        axes[0].scatter(
            lr + dx + jit.normal(0, 0.02, len(lr)),
            lw,
            s=10,
            alpha=0.35,
            color=color,
            label=label,
            edgecolors="none",
        )
        axes[1].scatter(
            lw,
            ls + dx + jit.normal(0, 0.02, len(ls)),
            s=10,
            alpha=0.35,
            color=color,
            label=label,
            edgecolors="none",
        )
        mean = [
            np.mean([lw[i] for i, r in enumerate(rows) if r["p"] == p]) for p in grid
        ]
        sem = [
            np.std([lw[i] for i, r in enumerate(rows) if r["p"] == p])
            / np.sqrt(circuits * records)
            for p in grid
        ]
        axes[2].errorbar(grid, mean, sem, fmt="o-", color=color, ms=4, label=label)
        summary[label] = {
            "records": len(rows),
            "mean_log2_1pW": dict(zip(map(str, grid), np.round(mean, 4), strict=True)),
            "identity_residual": float(np.max(np.abs(lw - (n - lp)))),
            "stabilizer_line_residual": float(np.max(np.abs(lw - (n - lr)))),
            "strings_line_residual": float(np.max(np.abs(lw - ls))),
            "max_n_strings": int(max(r["n_strings"] for r in rows)),
            "full_rank_fraction_with_W_positive": float(
                np.mean([r["rank"] == 2**n and r["weight"] > 1e-6 for r in rows])
            ),
        }
    x = np.linspace(0, n, 2)
    axes[0].plot(x, n - x, "k--", lw=1, label="1 + W = 2^N / rang")
    axes[0].set_xlabel("log2 rang(E_m)")
    axes[0].set_ylabel("log2(1 + W)")
    axes[0].set_title("poids de Fourier contre rang")
    axes[0].legend(frameon=False)
    axes[1].plot([0, n], [0, n], "k--", lw=1, label="nombre de chaînes = 1 + W")
    axes[1].axhline(2 * n, color="0.5", ls=":", lw=1, label="4^N chaînes")
    axes[1].set_xlabel("log2(1 + W)")
    axes[1].set_ylabel("log2(nombre de chaînes non nulles)")
    axes[1].set_title("chaînes de Pauli non nulles contre poids")
    axes[1].legend(frameon=False)
    axes[2].axhline(n, color="0.5", ls=":", lw=1)
    axes[2].text(0.0, n + 0.08, "rang 1 : W = 2^N − 1", fontsize=8)
    axes[2].axhline(0, color="0.5", ls=":", lw=1)
    axes[2].text(0.3, 0.08, "E_m = I : W = 0", fontsize=8)
    axes[2].set_xlabel("p")
    axes[2].set_ylabel("moyenne de log2(1 + W) = N − S_2")
    axes[2].set_title(f"à travers p (N = {n}, T = {depth}, Born)")
    axes[2].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rank_strings_weight.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/effect_anatomy/figures")
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--circuits", type=int, default=20)
    parser.add_argument("--records", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "clifford": anatomy_figure(
            random_clifford_gate,
            CLIFFORD,
            "anatomy_clifford.png",
            "Clifford, N = 3, T = 3 : un circuit, "
            "quatre records de plus en plus contraints",
            args.out,
            args.seed,
        ),
        "haar": anatomy_figure(
            haar_gate,
            HAAR,
            "anatomy_haar.png",
            "Haar, N = 3, T = 3 : mêmes masques, portes génériques",
            args.out,
            args.seed,
        ),
        "kraus_vs_effect": kraus_vs_effect(args.out, args.seed),
        "statistics": statistics(args.out, args.seed, args.circuits, args.records),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
