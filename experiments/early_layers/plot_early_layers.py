#!/usr/bin/env python3
"""Figures of the early-layers study of the linear QELM on M (issue #288).

Reads every ``metrics.json`` under the sweep folders of ``early_layers.sh``,
groups the runs by configuration (gates, encoding, ``N``), averages over the
circuits (seeds) and writes, to ``--out``, for each configuration
``<gates>_<encoding>_N<N>``:

- ``weights_<key>.png``: per round ``t``, the weight norm ``||w_t||_2`` of
  the ridge on ``vec(M)`` (normalized by its largest round) and the share
  ``w_t . c_t / sum_s w_s . c_s`` of the signal of the model, one curve per
  ``p``;
- ``windows_<key>.png``: for a few ``p``, the accuracy of the ridge
  retrained on the prefix (rounds ``1 .. k``) and on the suffix (rounds
  ``k .. T``), the full model with the rounds after ``k`` blanked, the Bayes
  accuracy of the prefix, the Helstrom bound and the exact suffix bound
  ``(1 + D_k) / 2``;
- ``bounds_<key>.png``: ``D_k`` against ``k`` (log scale) with the Haar
  bound, and the exact RMS linear signal ``sqrt(mean_i c_{k,i}^2)`` with its
  Haar average;

and, over every configuration, ``early_share_vs_p.png``: the share of the
signal carried by rounds 1 to 3 and by the last two rounds, against ``p``.
``summary.json`` holds the averaged curves.

    uv run python experiments/early_layers/plot_early_layers.py \\
        outputs/early_layers/<date_time> --out experiments/early_layers/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

from QELM.utils.plots import save_figure, subplots
from QELM.utils.plots.style import COLORS

# Per-round curves averaged over the circuits.
CURVES = (
    "weight_l2",
    "share",
    "share_bound",
    "ablate_late",
    "ablate_early",
    "prefix",
    "suffix",
    "D",
    "suffix_bound",
    "bayes_prefix",
    "helstrom",
    "charge",
    "haar_mu",
    "haar_D_bound",
    "haar_c_rms",
)
SCALARS = ("accuracy", "train_accuracy", "bayes_accuracy")
# Rounds counted as "early" and "late" in early_share_vs_p.png.
EARLY_ROUNDS = 3
LATE_ROUNDS = 2
LABELS = {
    "haar_bell": "Haar, Bell",
    "haar_ghz": "Haar, GHZ",
    "haar_charge": "Haar, charge",
    "haar_w": "Haar, W",
    "u1_bell": "U(1), Bell",
    "u1_charge": "U(1), charge",
}


def load_runs(folders: list[Path]) -> list[dict[str, Any]]:
    """Every ``metrics.json`` found under ``folders``.

    Raises:
        ValueError: If no run is found.
    """
    runs = []
    for folder in folders:
        for path in sorted(Path(folder).rglob("metrics.json")):
            runs.append(json.loads(path.read_text()))
    if not runs:
        raise ValueError(f"No metrics.json under {folders}.")
    return runs


def _mean_sem(values: list[Any]) -> tuple[list[Any], list[Any]]:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    sem = (
        np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        if len(arr) > 1
        else np.zeros_like(mean)
    )
    return mean.tolist(), np.nan_to_num(sem).tolist()


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average the runs of each configuration, ``p`` by ``p``.

    Returns:
        ``{key: {"gates", "encoding", "N", "T", "n_circuits", "p": [...],
        "per_p": [{curve: mean, curve + "_sem": sem, ...}]}}`` with ``key``
        ``<gates>_<encoding>_N<N>``.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[f"{run['gates']}_{run['encoding']}_N{run['N']}"].append(run)
    out: dict[str, Any] = {}
    for key, members in sorted(groups.items()):
        levels = sorted({e["p"] for run in members for e in run["per_p"]})
        per_p = []
        for level in levels:
            entries = [e for run in members for e in run["per_p"] if e["p"] == level]
            row: dict[str, Any] = {"p": level, "n_circuits": len(entries)}
            for name in (*CURVES, *SCALARS):
                if all(name in e for e in entries):
                    row[name], row[f"{name}_sem"] = _mean_sem(
                        [e[name] for e in entries]
                    )
            share = np.asarray([e["share"] for e in entries], dtype=float)
            bound = np.asarray([e["share_bound"] for e in entries], dtype=float)
            for label, part in (
                ("early", slice(0, EARLY_ROUNDS)),
                ("late", slice(-LATE_ROUNDS, None)),
            ):
                row[f"share_{label}"], row[f"share_{label}_sem"] = _mean_sem(
                    list(share[:, part].sum(axis=1))
                )
                row[f"share_bound_{label}"], _ = _mean_sem(
                    list(bound[:, part].sum(axis=1))
                )
            c = np.asarray([e["c"] for e in entries], dtype=float)
            row["c_rms"], row["c_rms_sem"] = _mean_sem(
                list(np.sqrt(np.mean(c**2, axis=2)))
            )
            row["c_rms_circuits"] = np.sqrt(np.mean(c**2, axis=(0, 2))).tolist()
            per_p.append(row)
        out[key] = {
            "gates": members[0]["gates"],
            "encoding": members[0]["encoding"],
            "N": members[0]["N"],
            "T": members[0]["T"],
            "n_circuits": len(members),
            "p": levels,
            "per_p": per_p,
        }
    return out


def _p_colors(levels: list[float]) -> list[Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.0, 0.9, len(levels))]


def _label(entry: dict[str, Any]) -> str:
    name = f"{entry['gates']}_{entry['encoding']}"
    return f"{LABELS.get(name, name)}, N = {entry['N']}, T = {entry['T']}"


def plot_weights(entry: dict[str, Any], out: Path, key: str) -> Path:
    """Weight norm and signal share per round, one curve per ``p``."""
    fig, axes = subplots(1, 2, figsize=(13, 4.8))
    t = np.arange(1, entry["T"] + 1)
    for row, color in zip(entry["per_p"], _p_colors(entry["p"]), strict=True):
        w = np.asarray(row["weight_l2"])
        scale = w.max() if w.max() > 0 else 1.0
        axes[0].errorbar(
            t,
            w / scale,
            yerr=np.asarray(row["weight_l2_sem"]) / scale,
            color=color,
            marker="o",
            ms=3,
            capsize=2,
            label=f"p = {row['p']:g}",
        )
        axes[1].errorbar(
            t,
            row["share"],
            yerr=row["share_sem"],
            color=color,
            marker="o",
            ms=3,
            capsize=2,
        )
    axes[0].set(
        xlabel="round t",
        ylabel="||w_t||_2 / max_t ||w_t||_2",
        title="Weights of the ridge on M",
    )
    axes[1].axhline(0.0, color="0.5", lw=1)
    axes[1].set(
        xlabel="round t",
        ylabel="w_t . c_t / Cov(score, Y)",
        title="Share of the signal per round",
    )
    axes[0].legend(ncol=2)
    fig.suptitle(_label(entry))
    return save_figure(fig, out, f"weights_{key}", {"per_p": entry["per_p"]})


def plot_windows(entry: dict[str, Any], out: Path, key: str) -> Path:
    """Windows, ablation and exact bounds, for up to four ``p``."""
    wanted = [0.1, 0.2, 0.4, 1.0]
    rows = [r for r in entry["per_p"] if any(np.isclose(r["p"], w) for w in wanted)]
    rows = rows or entry["per_p"][:4]
    fig, axes = subplots(1, len(rows), figsize=(4.6 * len(rows), 4.6), squeeze=False)
    T = entry["T"]
    k = np.arange(1, T + 1)
    for ax, row in zip(axes[0], rows, strict=True):
        curves = (
            ("prefix", "ridge on rounds 1..k", COLORS[0], "-"),
            ("suffix", "ridge on rounds k..T", COLORS[1], "-"),
            ("ablate_late", "full ridge, rounds > k blanked", COLORS[2], "--"),
        )
        for name, label, color, style in curves:
            ax.errorbar(
                k,
                row[name],
                yerr=row[f"{name}_sem"],
                color=color,
                ls=style,
                marker="o",
                ms=3,
                capsize=2,
                label=label,
            )
        ax.plot(
            k, row["bayes_prefix"][1:], color=COLORS[3], label="Bayes on rounds 1..k"
        )
        ax.plot(
            k,
            row["helstrom"][1:],
            color=COLORS[4],
            ls=":",
            label="bound, any decoder knowing 1..k",
        )
        ax.plot(
            k,
            row["suffix_bound"],
            color=COLORS[5],
            ls="-.",
            label="bound (1 + D_k)/2, rounds k..T",
        )
        ax.axhline(0.5, color="0.5", lw=1)
        ax.set(
            xlabel="k", ylabel="accuracy", title=f"p = {row['p']:g}", ylim=(0.45, 1.02)
        )
    axes[0][0].legend(loc="upper right", fontsize=9)
    fig.suptitle(_label(entry))
    return save_figure(fig, out, f"windows_{key}", {"rows": rows})


def plot_bounds(entry: dict[str, Any], out: Path, key: str) -> Path:
    """``D_k`` and the RMS linear signal against ``k``, with the Haar average."""
    fig, axes = subplots(1, 2, figsize=(13, 4.8))
    k = np.arange(1, entry["T"] + 1)
    for row, color in zip(entry["per_p"], _p_colors(entry["p"]), strict=True):
        axes[0].semilogy(
            k,
            np.maximum(row["D"], 1e-16),
            color=color,
            marker="o",
            ms=3,
            label=f"p = {row['p']:g}",
        )
        axes[1].semilogy(
            k, np.maximum(row["c_rms_circuits"], 1e-16), color=color, marker="o", ms=3
        )
        if "haar_D_bound" in row:
            axes[0].semilogy(k, row["haar_D_bound"], color=color, ls=":", lw=1.2)
            axes[1].semilogy(
                k, np.maximum(row["haar_c_rms"], 1e-16), color=color, ls="--", lw=1.2
            )
    axes[0].set(
        xlabel="round k",
        ylabel="D_k",
        title="Exact D_k (dotted: Haar bound)",
        ylim=(1e-8, 2),
    )
    axes[1].set(
        xlabel="round k",
        ylabel="RMS_i E[m_ki Y]",
        title="Linear signal (dashed: Haar average)",
        ylim=(1e-8, 1),
    )
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(_label(entry))
    return save_figure(fig, out, f"bounds_{key}", {"per_p": entry["per_p"]})


def plot_early_share(summary: dict[str, Any], out: Path) -> Path:
    """Share of the signal of rounds 1..3 and of the last two, against ``p``."""
    fig, axes = subplots(1, 2, figsize=(13, 4.8))
    for color, (key, entry) in zip(COLORS * 3, summary.items(), strict=False):
        p = entry["p"]
        early = [r["share_early"] for r in entry["per_p"]]
        late = [r["share_late"] for r in entry["per_p"]]
        axes[0].errorbar(
            p,
            early,
            yerr=[r["share_early_sem"] for r in entry["per_p"]],
            color=color,
            marker="o",
            capsize=2,
            label=_label(entry),
        )
        axes[1].errorbar(
            p,
            late,
            yerr=[r["share_late_sem"] for r in entry["per_p"]],
            color=color,
            marker="o",
            capsize=2,
        )
        axes[1].plot(
            p,
            [r["share_bound_late"] for r in entry["per_p"]],
            color=color,
            ls=":",
            lw=1.2,
        )
    axes[0].set(
        xlabel="p",
        ylabel="share",
        xscale="log",
        title=f"Signal share of rounds 1..{EARLY_ROUNDS}",
    )
    axes[1].set(
        xlabel="p",
        ylabel="share",
        xscale="log",
        yscale="symlog",
        title=f"Signal share of the last {LATE_ROUNDS} rounds (dotted: bound)",
    )
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[0].legend(fontsize=9)
    return save_figure(fig, out, "early_share_vs_p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.folders[0]
    summary = summarize(load_runs(args.folders))
    for key, entry in summary.items():
        plot_weights(entry, out, key)
        plot_windows(entry, out, key)
        plot_bounds(entry, out, key)
    plot_early_share(summary, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
