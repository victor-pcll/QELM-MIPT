#!/usr/bin/env python3
"""Figures of the CNN decoder experiment (issue #293).

Reads the JSON files written by ``cnn_decoder.py`` and draws, in ``--out``
(default: the input folder):

- ``accuracy_vs_p.png``: on the trajectories purified along the axis
  (top row), accuracy of the ridge, the MLP and the CNN against ``p``, one
  panel per readout, one curve per ``N``; the Bayes decoder is at 1. Bottom
  row: the purified fraction ``f = 1 - S_Q`` (the Bayes transition), the
  accuracy over all trajectories of the CNN against its Bayes ceiling
  ``(1 + f) / 2``, and the gain of the CNN over the ridge;
- ``accuracy_vs_train_size.png`` (if ``scaling_*.json`` runs exist): the CNN
  accuracy on purified trajectories against the number of training
  trajectories, one panel per ``p``, one curve per ``N``;
- ``summary.json``: the mean and standard error over circuits of every
  curve.

Run from the repository root:

    uv run python experiments/cnn_decoder/plot_cnn_decoder.py outputs/cnn_decoder
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

READOUTS = {
    "ridge": "ridge on M (linear QELM-M)",
    "mlp": "MLP on M (128/64, #289)",
    "cnn": "CNN on M (3 × 32 conv, periodic)",
}
N_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
GRID = "#dcdbd6"


def load(folder: Path, mode: str) -> dict[int, list[dict]]:
    """Return ``{N: [row, ...]}`` of the ``mode`` runs of ``folder``."""
    runs: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(folder.glob(f"{mode}_N*_seed*_p*.json")):
        row = json.loads(path.read_text())
        runs[int(row["N"])].append(row)
    return dict(sorted(runs.items()))


def curve(rows: list[dict], getter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """Mean and standard error over circuits of ``getter(row)`` at each ``p``."""
    by_p: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_p[row["p"]].append(float(getter(row)))
    p = np.array(sorted(by_p))
    mean, err = [], []
    for x in p:
        v = np.array(by_p[x])
        v = v[~np.isnan(v)]
        mean.append(v.mean() if v.size else np.nan)
        err.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
    return p, np.array(mean), np.array(err)


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_sweep(runs: dict[int, list[dict]], out: Path) -> dict:
    """Draw ``accuracy_vs_p.png``; return its curves."""
    summary: dict = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for k, (N, rows) in enumerate(runs.items()):
        color = N_COLORS[k % len(N_COLORS)]
        n_c = len({r["seed"] for r in rows})
        label = f"N = {N}, T = {2 * N} ({n_c} circuits)"
        s: dict = {}
        for j, readout in enumerate(READOUTS):
            p, m, e = curve(rows, lambda r, ro=readout: r[ro]["acc_pure"])
            axes[0, j].errorbar(p, m, e, color=color, marker="o", ms=4, label=label)
            s[readout] = {"acc_pure": m.tolist(), "err": e.tolist()}
        p, f, fe = curve(rows, lambda r: r["purified"])
        axes[1, 0].errorbar(p, f, fe, color=color, marker="o", ms=4, label=label)
        p, a, ae = curve(rows, lambda r: r["cnn"]["acc_all"])
        p, b, _ = curve(rows, lambda r: r["bayes"]["acc_all"])
        axes[1, 1].errorbar(p, a, ae, color=color, marker="o", ms=4, label=label)
        axes[1, 1].plot(p, b, color=color, ls=":", lw=1.2)
        p, g, ge = curve(rows, lambda r: r["cnn"]["acc_pure"] - r["ridge"]["acc_pure"])
        axes[1, 2].errorbar(p, g, ge, color=color, marker="o", ms=4, label=label)
        s.update(
            p=p.tolist(),
            purified=f.tolist(),
            cnn_acc_all=a.tolist(),
            bayes_acc_all=b.tolist(),
            cnn_minus_ridge=g.tolist(),
            n_circuits=n_c,
            n_traj=rows[0]["n_traj"],
        )
        summary[str(N)] = s
    for j, title in enumerate(READOUTS.values()):
        ax = axes[0, j]
        ax.axhline(1, color="#0b0b0b", ls=":", lw=1.2, label="Bayes optimal")
        ax.axhline(0.5, color="#888", lw=0.8)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("accuracy on purified trajectories")
        _style(ax)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[1, 0].set_ylabel("purified fraction f = 1 − S_Q")
    axes[1, 0].set_title("Bayes transition: R purified at t = T", fontsize=10)
    axes[1, 1].set_ylabel("accuracy over all trajectories")
    axes[1, 1].set_title("CNN (solid) vs Bayes ceiling (1 + f)/2 (dotted)", fontsize=10)
    axes[1, 2].set_ylabel("CNN − ridge, purified trajectories")
    axes[1, 2].axhline(0, color="#888", lw=0.8)
    axes[1, 2].set_title("gain of the CNN over the linear readout", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("measurement rate p")
        _style(ax)
    fig.suptitle(
        "Decoding the reference qubit from the record M, Clifford brickwork, "
        "one fixed circuit per run",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_p.png", dpi=150)
    plt.close(fig)
    return summary


def plot_scaling(runs: dict[int, list[dict]], out: Path) -> dict:
    """Draw ``accuracy_vs_train_size.png``; return its curves."""
    ps = sorted({r["p"] for rows in runs.values() for r in rows})
    fig, axes = plt.subplots(1, len(ps), figsize=(5 * len(ps), 4.2), squeeze=False)
    summary: dict = {}
    for k, (N, rows) in enumerate(runs.items()):
        color = N_COLORS[k % len(N_COLORS)]
        for row in rows:
            ax = axes[0, ps.index(row["p"])]
            n = [r["n_train_pure"] for r in row["rows"]]
            acc = [r["acc_pure"] for r in row["rows"]]
            ax.plot(n, acc, color=color, marker="o", label=f"N = {N}, T = {2 * N}")
            summary[f"N{N}_p{row['p']}_seed{row['seed']}"] = {
                "n_train": [r["n_train"] for r in row["rows"]],
                "n_train_pure": n,
                "acc_pure": acc,
                "purified": row["purified"],
            }
    for ax, p in zip(axes[0], ps, strict=True):
        ax.set_xscale("log")
        ax.axhline(1, color="#0b0b0b", ls=":", lw=1.2, label="Bayes optimal")
        ax.axhline(0.5, color="#888", lw=0.8)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(f"p = {p}", fontsize=10)
        ax.set_xlabel("purified training trajectories")
        _style(ax)
    axes[0, 0].set_ylabel("CNN accuracy on purified trajectories")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("CNN accuracy against training-set size (same test set)")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_train_size.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    """Parse the options and draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.folder
    out.mkdir(parents=True, exist_ok=True)
    summary = {"sweep": plot_sweep(load(args.folder, "sweep"), out)}
    scaling = load(args.folder, "scaling")
    if scaling:
        summary["scaling"] = plot_scaling(scaling, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
