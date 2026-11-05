#!/usr/bin/env python3
"""Two summary figures of the early-layers study: Haar U(4) against U(1) (#288).

Reads the ``summary.json`` written by ``plot_early_layers.py`` and draws, for
the Bell encoding and a few values of ``p``:

- ``story_information.png``: where the information sits in the record.
  (a) the share of the ridge's signal carried by each round; (b) ``D_k``,
  the largest advantage over chance of any decoder (Bayes included) of
  rounds ``k .. T`` alone.
- ``story_depth.png``: does a deeper circuit help? The test accuracy of the
  ridge trained on the first ``k`` rounds, which is a circuit of depth
  ``k``, and the Bayes accuracy of those rounds, against ``k``.

    uv run python experiments/early_layers/plot_story.py \\
        experiments/early_layers/figures/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

from QELM.utils.plots import save_figure, subplots

HAAR = "#2a78d6"
U1 = "#eb6834"
CONFIGS = (
    ("haar_bell_N8", "Haar U(4): no conservation", HAAR),
    ("u1_bell_N8", "U(1): charge conserved", U1),
)


def _row(entry: dict[str, Any], p: float) -> dict[str, Any]:
    for row in entry["per_p"]:
        if np.isclose(row["p"], p):
            return dict(row)
    raise ValueError(f"p = {p} not in the summary.")


def plot_information(summary: dict[str, Any], p_values: list[float], out: Path) -> Path:
    """Signal share per round and ``D_k``, Haar against U(1)."""
    fig, axes = subplots(1, 2, figsize=(13, 5))
    styles = ("-", "--", ":")
    T_rounds = summary[CONFIGS[0][0]]["T"]
    for key, label, color in CONFIGS:
        entry = summary[key]
        t = np.arange(1, entry["T"] + 1)
        for p, ls in zip(p_values, styles, strict=False):
            row = _row(entry, p)
            axes[0].errorbar(
                t,
                row["share"],
                yerr=row["share_sem"],
                color=color,
                ls=ls,
                marker="o",
                ms=4,
                capsize=2,
                label=f"{label}, p = {p:g}",
            )
            axes[1].semilogy(
                t, np.maximum(row["D"], 1e-12), color=color, ls=ls, marker="o", ms=4
            )
    axes[0].axhline(0.0, color="0.5", lw=1)
    axes[0].axhline(1.0 / T_rounds, color="0.5", lw=1, ls=":")
    axes[0].text(
        T_rounds,
        1.0 / T_rounds + 0.02,
        "uniform 1/T",
        ha="right",
        color="0.4",
        fontsize=10,
    )
    axes[0].set(
        xlabel="round t",
        ylabel="share of the ridge signal",
        title="(a) Which rounds the ridge reads",
    )
    axes[1].set(
        xlabel="round k",
        ylabel="D_k",
        ylim=(1e-8, 2),
        title="(b) Information left in rounds k..T",
    )
    axes[0].legend(fontsize=9)
    return save_figure(fig, out, "story_information")


def plot_depth(summary: dict[str, Any], p_values: list[float], out: Path) -> Path:
    """Accuracy of the ridge and of Bayes on the first ``k`` rounds."""
    fig, axes = subplots(
        1, len(p_values), figsize=(6.2 * len(p_values), 5), squeeze=False
    )
    for ax, p in zip(axes[0], p_values, strict=True):
        for key, label, color in CONFIGS:
            entry = summary[key]
            row = _row(entry, p)
            k = np.arange(1, entry["T"] + 1)
            ax.errorbar(
                k,
                row["prefix"],
                yerr=row["prefix_sem"],
                color=color,
                marker="o",
                ms=4,
                capsize=2,
                label=f"ridge, {label}",
            )
            ax.plot(
                k,
                row["bayes_prefix"][1:],
                color=color,
                ls="--",
                label=f"Bayes, {label}",
            )
        ax.axhline(0.5, color="0.5", lw=1)
        ax.set(
            xlabel="depth k (rounds used)",
            ylabel="test accuracy",
            title=f"p = {p:g}",
            ylim=(0.45, 1.02),
        )
    axes[0][0].legend(fontsize=9, loc="upper left")
    return save_figure(fig, out, "story_depth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--p", type=float, nargs="+", default=[0.2, 0.6])
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    out = args.out or args.summary.parent
    plot_information(summary, args.p, out)
    plot_depth(summary, args.p, out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
