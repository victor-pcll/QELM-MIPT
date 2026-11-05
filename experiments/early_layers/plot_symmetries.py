#!/usr/bin/env python3
"""Compare the symmetries of the gates: what each one does to the ridge (#288).

Reads the ``summary.json`` of ``plot_early_layers.py`` for a sweep with the
gates ``haar``, ``u1``, ``z2``, ``dipole`` and ``diagonal`` (one encoding,
``N`` and ``T``) and writes to ``--out``:

- ``symmetries_accuracy.png``: (a) test accuracy of the ridge on ``vec(M)``
  (solid) and of Bayes (dashed) against ``p``; (b) overfitting, the train
  minus test accuracy of the ridge;
- ``symmetries_rounds.png``: at one ``p``, (a) the share of the ridge's
  signal per round, (b) the weight norm per round over that of round 1,
  (c) ``D_k``;
- ``symmetries_depth.png``: test accuracy of the ridge (solid) and of Bayes
  (dashed) on the first ``k`` rounds, against ``k``, for two values of ``p``.

    uv run python experiments/early_layers/plot_symmetries.py \\
        outputs/early_layers/symmetries/summary.json --p 0.2 0.6
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

# Gates, label, color, in the order of the legends.
GATES = (
    ("haar", "Haar U(4): no symmetry", "#2a78d6"),
    ("u1", "U(1): charge Q", "#eb6834"),
    ("dipole", "dipole: Q and P = sum i n_i", "#1baf7a"),
    ("z2", "Z2: parity prod Z_i", "#6250d6"),
    ("diagonal", "H = sum h_i Z_i: every Z_i", "#d63a8a"),
)


def _entries(summary: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    found = []
    for gates, label, color in GATES:
        keys = [k for k, e in summary.items() if e["gates"] == gates]
        if keys:
            found.append((summary[keys[0]], label, color))
    if not found:
        raise ValueError("No configuration of the symmetries in the summary.")
    return found


def _legend_below(fig: Any, ax: Any) -> None:
    """One legend for the figure, in a row under the panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), fontsize=11
    )


def _row(entry: dict[str, Any], p: float) -> dict[str, Any]:
    for row in entry["per_p"]:
        if np.isclose(row["p"], p):
            return dict(row)
    raise ValueError(f"p = {p} not in the summary.")


def plot_accuracy(summary: dict[str, Any], out: Path) -> Path:
    """Ridge and Bayes against ``p``, and the overfitting gap."""
    fig, axes = subplots(1, 2, figsize=(13, 5))
    for entry, label, color in _entries(summary):
        p = entry["p"]
        rows = entry["per_p"]
        axes[0].errorbar(
            p,
            [r["accuracy"] for r in rows],
            yerr=[r["accuracy_sem"] for r in rows],
            color=color,
            marker="o",
            capsize=2,
            label=label,
        )
        axes[0].plot(p, [r["bayes_accuracy"] for r in rows], color=color, ls="--")
        if all("train_accuracy" in r for r in rows):
            gap = [r["train_accuracy"] - r["accuracy"] for r in rows]
            sem = [np.hypot(r["train_accuracy_sem"], r["accuracy_sem"]) for r in rows]
            axes[1].errorbar(p, gap, yerr=sem, color=color, marker="o", capsize=2)
    axes[0].axhline(0.5, color="0.5", lw=1)
    axes[0].set(
        xlabel="p",
        ylabel="accuracy",
        xscale="log",
        ylim=(0.45, 1.02),
        title="(a) Ridge (solid) and Bayes (dashed)",
    )
    axes[1].axhline(0.0, color="0.5", lw=1)
    axes[1].set(
        xlabel="p",
        ylabel="train - test accuracy",
        xscale="log",
        title="(b) Overfitting of the ridge",
    )
    _legend_below(fig, axes[0])
    return save_figure(fig, out, "symmetries_accuracy")


def plot_rounds(summary: dict[str, Any], p: float, out: Path) -> Path:
    """Signal share, weight norm and ``D_k`` per round, at one ``p``."""
    fig, axes = subplots(1, 3, figsize=(19, 5))
    for entry, label, color in _entries(summary):
        row = _row(entry, p)
        t = np.arange(1, entry["T"] + 1)
        axes[0].errorbar(
            t,
            row["share"],
            yerr=row["share_sem"],
            color=color,
            marker="o",
            ms=4,
            capsize=2,
            label=label,
        )
        w = np.asarray(row["weight_l2"])
        scale = w[0] if w[0] > 0 else 1.0
        axes[1].errorbar(
            t,
            w / scale,
            yerr=np.asarray(row["weight_l2_sem"]) / scale,
            color=color,
            marker="o",
            ms=4,
            capsize=2,
        )
        axes[2].semilogy(t, np.maximum(row["D"], 1e-12), color=color, marker="o", ms=4)
    T = _entries(summary)[0][0]["T"]
    axes[0].axhline(1.0 / T, color="0.5", lw=1, ls=":")
    axes[0].axhline(0.0, color="0.5", lw=1)
    axes[0].set(
        xlabel="round t",
        ylabel="w_t . c_t / Cov(score, Y)",
        title=f"(a) Share of the ridge signal, p = {p:g}",
    )
    axes[1].set(
        xlabel="round t",
        ylabel="||w_t||_2 / ||w_1||_2",
        title="(b) Weight norm of the ridge",
    )
    axes[2].set(
        xlabel="round k",
        ylabel="D_k",
        ylim=(1e-8, 2),
        title="(c) Information left in rounds k..T",
    )
    _legend_below(fig, axes[0])
    return save_figure(fig, out, "symmetries_rounds")


def plot_depth(summary: dict[str, Any], p_values: list[float], out: Path) -> Path:
    """Ridge and Bayes on the first ``k`` rounds, against ``k``."""
    fig, axes = subplots(
        1, len(p_values), figsize=(6.4 * len(p_values), 5), squeeze=False
    )
    for ax, p in zip(axes[0], p_values, strict=True):
        for entry, label, color in _entries(summary):
            row = _row(entry, p)
            k = np.arange(1, entry["T"] + 1)
            ax.errorbar(
                k,
                row["prefix"],
                yerr=row["prefix_sem"],
                color=color,
                marker="o",
                ms=3,
                capsize=2,
                label=label,
            )
            ax.plot(k, row["bayes_prefix"][1:], color=color, ls="--")
        ax.axhline(0.5, color="0.5", lw=1)
        ax.set(
            xlabel="depth k (rounds used)",
            ylabel="test accuracy",
            title=f"p = {p:g}: ridge (solid), Bayes (dashed)",
            ylim=(0.45, 1.02),
        )
    _legend_below(fig, axes[0][0])
    return save_figure(fig, out, "symmetries_depth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--p", type=float, nargs="+", default=[0.2, 0.6])
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    out = args.out or args.summary.parent
    plot_accuracy(summary, out)
    plot_rounds(summary, args.p[0], out)
    plot_depth(summary, args.p, out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
