#!/usr/bin/env python3
"""Figures of the rank structure of the Haar E_circuit (issue #362).

Reads ``rank_structure_N4_<prior>.json`` (``rank_structure.py``) and the
samples ``dense_N4_<prior>.npz`` (``run.py``), writes:

- ``rank_levels_<prior>.png``: (a) fraction of Haar circuits at each visible
  rank against ``p``; (b) ``E_circuit`` of each circuit against its rank;
  (c) mean ``E_circuit`` against ``p``, measured and rebuilt from the level
  fractions; (d) min cut ``C`` and its two bounds against ``p``.
- ``ecirc_hist_by_rank_<prior>.png``: the Haar ``E_circuit`` histograms of
  ``run.py``, each sample coloured by the rank of its circuit.

Usage::

    uv run python experiments/r2_budget/plot_rank_structure.py --data outputs/r2_budget
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAMP = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
P_SHOW = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def level_colors(levels: list[int]) -> dict[int, str]:
    pick = np.linspace(0, len(RAMP) - 1, max(len(levels), 1)).round().astype(int)
    return {lev: RAMP[int(k)] for lev, k in zip(levels, pick, strict=True)}


def rank_levels_figure(data: dict, path: Path, title: str, prior: str) -> None:
    ps = np.array(data["p"])
    per = [r["haar"] for r in data["per_p"]]
    vis = [np.array(r["visible"]) for r in per]
    ecirc = [np.array(r["e_circuit"]) for r in per]
    levels = sorted({int(v) for a in vis for v in a})
    colors = level_colors(levels)
    frac = np.array([[np.mean(a == lev) for lev in levels] for a in vis])
    all_v, all_e = np.concatenate(vis), np.concatenate(ecirc)
    e_level = {lev: float(all_e[all_v == lev].mean()) for lev in levels}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    ax = axes[0, 0]
    style(ax)
    bottom = np.zeros(len(ps))
    for k, lev in enumerate(levels):
        ax.bar(
            ps,
            frac[:, k],
            0.04,
            bottom=bottom,
            color=colors[lev],
            edgecolor="white",
            linewidth=0.8,
            label=f"rank {lev}",
        )
        bottom += frac[:, k]
    ax.set_xlabel("p", fontsize=9, color=INK)
    ax.set_ylabel("fraction of Haar circuits", fontsize=9, color=INK)
    ax.set_title("(a) visible rank of the effect space", fontsize=9, color=INK)
    ax.legend(
        fontsize=7,
        frameon=False,
        ncol=len(levels),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )

    ax = axes[0, 1]
    style(ax)
    rng = np.random.default_rng(0)
    for lev in levels:
        e = all_e[all_v == lev]
        x = np.full(e.size, float(lev)) + rng.uniform(-1.2, 1.2, e.size)
        ax.scatter(x, e, s=4, color=colors[lev], alpha=0.5, lw=0)
        ax.plot([lev - 3, lev + 3], [e_level[lev]] * 2, color=INK, lw=1.5)
        ax.annotate(
            f"{e_level[lev]:.2f}",
            (lev + 3.5, e_level[lev]),
            fontsize=7,
            color=INK,
            va="center",
        )
    if prior == "iid":
        mean_r2 = np.concatenate([np.array(r["mean_r2"]) for r in per])
        label = r"$1 - \sum_a w_a [\Pi_C]_{aa} / \sum_a w_a$"
    else:
        mean_r2 = all_v / (3 ** data["n"] - 1)
        label = r"$1 - {\rm rank} / (3^N - 1)$"
    ax.scatter(
        all_v,
        1 - mean_r2,
        s=14,
        marker="D",
        facecolor="none",
        edgecolor=MUTED,
        lw=0.6,
        label=label,
    )
    ax.set_xlabel("visible rank", fontsize=9, color=INK)
    ax.set_ylabel(r"$E_{\rm circuit}$ (mean over 40 targets)", fontsize=9, color=INK)
    ax.set_title(
        r"(b) $E_{\rm circuit}$ of each circuit (all p pooled)", fontsize=9, color=INK
    )
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    style(ax)
    measured = np.array([e.mean() for e in ecirc])
    rebuilt = frac @ np.array([e_level[lev] for lev in levels])
    ax.plot(ps, measured, color=INK, lw=2, marker="o", ms=4, label="measured")
    ax.plot(
        ps,
        rebuilt,
        color="#eb6834",
        lw=1.5,
        ls="--",
        marker="s",
        ms=3,
        label=r"$\sum_{\rm levels}$ fraction(p) × level value",
    )
    ax.set_xlabel("p", fontsize=9, color=INK)
    ax.set_ylabel(r"mean $E_{\rm circuit}$ (Haar)", fontsize=9, color=INK)
    ax.set_title("(c) mean error = mixture of the levels", fontsize=9, color=INK)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    style(ax)
    n = data["n"]
    c_mean = np.array([np.mean(r["mincut"]) for r in per])
    b1 = np.array([np.mean(n + np.array(r["k_eff"])) for r in per])
    b2 = np.array([np.mean(2 * n - np.array(r["n_first"])) for r in per])
    ax.plot(
        ps,
        c_mean,
        color=INK,
        lw=2,
        marker="o",
        ms=4,
        label=r"min cut $C$ ($=\log_2$ dense rank)",
    )
    ax.plot(
        ps, b1, color="#1baf7a", lw=1.5, ls="--", label=r"$N + K_{\rm eff}$ (records)"
    )
    ax.plot(
        ps,
        b2,
        color="#e34948",
        lw=1.5,
        ls="--",
        label=r"$2N - n_1$ (first-layer measurements)",
    )
    ax.axhline(np.log2(3**n), color=MUTED, lw=0.8, ls=":")
    ax.annotate(
        r"$\log_2 3^N$: visible saturation",
        (ps[0], np.log2(3**n) + 0.08),
        fontsize=7,
        color=MUTED,
    )
    ax.set_xlabel("p", fontsize=9, color=INK)
    ax.set_ylabel("bits (mean over circuits)", fontsize=9, color=INK)
    ax.set_title("(d) min cut and its two bounds", fontsize=9, color=INK)
    ax.set_ylim(n - 0.2, 2 * n + 0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def hist_by_rank_figure(data: dict, raw: dict, path: Path, title: str) -> None:
    ps = data["p"]
    cols = [j for j, p in enumerate(ps) if any(abs(p - s) < 1e-9 for s in P_SHOW)]
    cols = cols or list(range(len(ps)))
    per = [r["haar"] for r in data["per_p"]]
    levels = sorted({int(v) for r in per for v in r["visible"]})
    colors = level_colors(levels)
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(
        1, len(cols), figsize=(2.0 * len(cols) + 0.6, 2.9), squeeze=False, sharey=True
    )
    for c, j in enumerate(cols):
        ax = axes[0, c]
        style(ax)
        e = raw[f"haar_circuit_{j}"].astype(float)
        n_t = e.size // len(per[j]["visible"])
        lev_s = np.repeat(np.array(per[j]["visible"]), n_t)
        stacks = [np.clip(e[lev_s == lev], 0, 1) for lev in levels]
        weights = [np.full(s.size, 1.0 / e.size) for s in stacks]
        ax.hist(
            stacks,
            bins=bins,
            weights=weights,
            stacked=True,
            color=[colors[lev] for lev in levels],
            edgecolor="white",
            linewidth=0.4,
            label=[f"rank {lev}" for lev in levels],
        )
        ax.set_title(f"p = {ps[j]:.2f}", fontsize=9, color=INK)
        ax.set_xlabel(r"$E_{\rm circuit}$", fontsize=8, color=MUTED)
        ax.set_xlim(0, 1)
    axes[0, 0].set_ylabel("fraction of samples", fontsize=8, color=INK)
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, lab, loc="upper right", ncol=len(levels), fontsize=7, frameon=False
    )
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/r2_budget"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.data / "figures"
    out.mkdir(parents=True, exist_ok=True)
    prior_text = {
        "iid": r"$\beta_a \sim N(0,1)$",
        "flat": r"$\sqrt{w_a}\,\beta_a \sim N(0,1)$",
    }
    for prior in ("iid", "flat"):
        path = args.data / f"rank_structure_N{args.n}_{prior}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        with np.load(args.data / f"dense_N{args.n}_{prior}.npz") as z:
            raw = {k: z[k] for k in z.files}
        head = (
            f"Haar, N = {data['n']}, T = {data['depth']}, {prior_text[prior]}, "
            "K = ∞, λ → 0+"
        )
        rank_levels_figure(data, out / f"rank_levels_{prior}.png", head, prior)
        hist_by_rank_figure(
            data,
            raw,
            out / f"ecirc_hist_by_rank_{prior}.png",
            head
            + r": $E_{\rm circuit}$ histograms coloured by the rank of the circuit",
        )
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
