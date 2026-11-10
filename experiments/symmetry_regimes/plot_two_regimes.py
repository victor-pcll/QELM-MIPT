#!/usr/bin/env python3
"""The two regimes of the QELM on one figure (issue #311).

Reads a ``summary.json`` of ``experiments/nonlinear_M/plot_nonlinear_M.py``
(default: ``experiments/nonlinear_M/figures/bell_T16_summary.json``, Haar
U(4), Bell pair with the middle site, Z label, ``N = 8``, ``T = 16``, 10 fixed
circuits, 8000 trajectories per ``p``) and draws three curves only: the QELM
on the state (``<P_k>``, ``<P_i P_j>``), on ``M``, and on both, as test
accuracy and as test R2 of the readout output (two figures). The background
is split at the crossing ``p_x`` of the state and ``M`` curves (linear
interpolation between grid points): blue where the state carries the label,
red where the record does.

    uv run python experiments/symmetry_regimes/plot_two_regimes.py \\
        --out experiments/symmetry_regimes/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent / "nonlinear_M" / "figures" / "bell_T16_summary.json"

STATE, RECORD, BOTH = "ridge_quantum", "ridge_raw", "ridge_M_quantum"
CURVES: tuple[tuple[str, str, str], ...] = (
    (STATE, "QELM on the state: <P_k>, <P_i P_j>", "#1f77b4"),
    (RECORD, "QELM on M", "#d62728"),
    (BOTH, "QELM on M and <P_k>, <P_i P_j>", "#fa5703"),
)
STATE_ZONE, RECORD_ZONE = "#dbe9fb", "#fde8e2"


SCORES = {"accuracy": "test accuracy", "r2": "test R2 of the readout output"}


def crossing(entry: dict[str, Any], score: str = "accuracy") -> float:
    """First ``p > 0`` where the ``M`` curve rises above the state curve.

    Linear interpolation between the two grid points around the sign change
    of ``score(state) - score(M)``, ``score`` in :data:`SCORES`.

    Raises:
        ValueError: If the curves never cross.
    """
    p = np.asarray(entry["p"])
    diff = np.asarray(entry[f"{score}_{STATE}"]) - np.asarray(
        entry[f"{score}_{RECORD}"]
    )
    for i in range(1, p.size):
        if p[i - 1] > 0 and diff[i - 1] > 0 >= diff[i]:
            return float(
                p[i - 1] + diff[i - 1] * (p[i] - p[i - 1]) / (diff[i - 1] - diff[i])
            )
    raise ValueError("The state and M curves do not cross.")


def plot(
    entry: dict[str, Any], out: Path, title: str, score: str = "accuracy"
) -> tuple[Path, float]:
    """Draw the figure of ``score``; return its path and the crossing ``p_x``."""
    p = np.asarray(entry["p"])
    p_x = crossing(entry, score)
    lo, hi = (0.45, 1.02) if score == "accuracy" else (-0.05, 1.02)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axvspan(0.0, p_x, color=STATE_ZONE, lw=0, zorder=0)
    ax.axvspan(p_x, p.max(), color=RECORD_ZONE, lw=0, zorder=0)
    ax.axvline(p_x, color="0.4", lw=0.8, ls="--", zorder=1)
    for name, label, color in CURVES:
        mean = np.asarray(entry[f"{score}_{name}"])
        sem = np.asarray(entry[f"{score}_{name}_sem"])
        ax.plot(
            p, mean, "-", color=color, marker="o", ms=3.5, lw=1.8, label=label, zorder=3
        )
        ax.fill_between(
            p, mean - sem, mean + sem, color=color, alpha=0.25, lw=0, zorder=2
        )
    ax.annotate(
        f"the state carries the label\n(p < p_x = {p_x:.3f})",
        xy=(p_x / 2, lo + 0.92 * (hi - lo)),
        xytext=(p_x + 0.04, lo + 0.92 * (hi - lo)),
        ha="left",
        va="center",
        fontsize=9,
        color="#184f95",
        arrowprops={"arrowstyle": "->", "color": "#184f95", "lw": 0.8},
    )
    ax.text(
        (p_x + p.max()) / 2,
        lo + 0.72 * (hi - lo),
        "the record M carries the label",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#a33a1c",
    )
    ax.set_xlim(0, p.max())
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measurement probability p")
    ax.set_ylabel(SCORES[score])
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.95)
    fig.tight_layout()
    suffix = "" if score == "accuracy" else f"_{score}"
    path = out / f"two_regimes_haar_bell_z{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, p_x


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw the accuracy and R2 figures and write their data next to them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("summary", type=Path, nargs="?", default=DEFAULT)
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    key = next(iter(summary))
    entry = summary[key]
    N = int(key[1:])
    title = (
        f"Bell pair with the middle site, Haar U(4), Z outcome of R, N = {N}, "
        f"T = {int(entry['T'])}, {entry['n_circuits']} fixed circuits"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    paths = []
    for score in SCORES:
        path, p_x = plot(entry, args.out, title, score)
        data = {
            "source": str(args.summary.name),
            "score": score,
            "p_x": p_x,
            "p": entry["p"],
            **{
                f"{score}_{name}{suffix}": entry[f"{score}_{name}{suffix}"]
                for name, _, _ in CURVES
                for suffix in ("", "_sem")
            },
        }
        path.with_suffix(".json").write_text(json.dumps(data, indent=2))
        print(f"Saved {path} (p_x = {p_x:.4f})")
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
