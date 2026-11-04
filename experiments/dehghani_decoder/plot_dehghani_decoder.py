#!/usr/bin/env python3
"""Figures of the Dehghani decoder experiment (issue #287).

Reads the ``N<N>_seed<seed>.json`` files written by ``dehghani_decoder.py``
and draws, in ``--out`` (default: the input folder):

- ``accuracy_vs_p.png``: accuracy of the ridge, the MLP and the Bayes decoder
  along the purification axis, over all trajectories, against ``p``; one
  row per case (record ``M`` alone, ``M`` and the final readout), one
  column per ``N``;
- ``purification_parity.png``: the purified fraction ``1 - S_Q`` of the
  record alone (its crossing in ``N`` is the purification transition), the
  share of weight-1 parities, and the gain of the MLP over the ridge on the
  purified trajectories, against ``p``, for both cases;
- ``summary.json``: the mean and standard error over circuits of every
  curve.

Run from the repository root:

    uv run python experiments/dehghani_decoder/plot_dehghani_decoder.py \
        outputs/dehghani_decoder
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

CASES = {"M": "record M", "M_final": "M + final readout"}
READOUTS = {
    "ridge": ("ridge on M (linear QELM-M)", "#eb6834", "-", "o"),
    "mlp": ("MLP on M (2 × 128/64)", "#6250d6", "--", "s"),
    "bayes": ("Bayes optimal (exact)", "#0b0b0b", ":", None),
}
N_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRID = "#dcdbd6"


def load(folder: Path) -> dict[int, list[dict]]:
    """Return ``{N: [run, ...]}`` for the runs of ``folder``."""
    runs: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(folder.glob("N*_seed*.json")):
        run = json.loads(path.read_text())
        runs[int(run["N"])].append(run)
    return dict(sorted(runs.items()))


def curve(runs: list[dict], getter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error over circuits of ``getter(row)`` at each ``p``."""
    p = np.array([row["p"] for row in runs[0]["rows"]])
    vals = np.array([[getter(row) for row in run["rows"]] for run in runs], dtype=float)
    mean = np.nanmean(vals, axis=0)
    n = np.sum(~np.isnan(vals), axis=0)
    err = (
        np.nanstd(vals, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        if len(runs) > 1
        else np.zeros_like(mean)
    )
    return p, mean, err


def _style(ax: plt.Axes) -> None:
    ax.set_xscale("symlog", linthresh=0.1, linscale=1.5)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_accuracy(runs: dict[int, list[dict]], out: Path) -> None:
    """Accuracy along the purification axis, one panel per case and ``N``."""
    Ns = list(runs)
    fig, axes = plt.subplots(
        2, len(Ns), figsize=(3.3 * len(Ns), 6.4), sharey=True, squeeze=False
    )
    for i, case in enumerate(CASES):
        for j, N in enumerate(Ns):
            ax = axes[i, j]
            for key, (label, color, ls, marker) in READOUTS.items():
                p, m, e = curve(runs[N], lambda r, k=key: r[case][k]["acc_all"])
                ax.plot(
                    p, m, ls=ls, color=color, lw=2, marker=marker, ms=4, label=label
                )
                ax.fill_between(p, m - e, m + e, color=color, alpha=0.15, lw=0)
            ax.axhline(0.5, color="#898781", lw=1, ls="--")
            ax.axvline(0.16, color="#898781", lw=0.8, ls=":")
            _style(ax)
            ax.set_ylim(0.45, 1.02)
            ax.set_title(f"{CASES[case]}, N = {N}, T = {2 * N}", fontsize=10)
            if i == 1:
                ax.set_xlabel("measurement probability p")
            if j == 0:
                ax.set_ylabel("accuracy of σ_P on R")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle(
        "Decoding the reference qubit of a fixed Clifford circuit "
        f"({len(runs[Ns[0]])} circuits per N; dotted line: p_c ≈ 0.16)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "accuracy_vs_p.png", dpi=200)
    plt.close(fig)


def plot_purification_parity(runs: dict[int, list[dict]], out: Path) -> None:
    """Purified fraction, weight-1 share and MLP gain, against ``p``."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for j, N in enumerate(runs):
        c = N_COLORS[j % len(N_COLORS)]
        p, m, e = curve(runs[N], lambda r: r["M"]["purified"])
        axes[0].errorbar(
            p, m, e, color=c, lw=2, marker="o", ms=4, capsize=2, label=f"N = {N}"
        )
        for case, ls in (("M", "-"), ("M_final", "--")):
            p, m, e = curve(runs[N], lambda r, k=case: r[k]["weight_one"])
            axes[1].plot(
                p, m, color=c, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4
            )
            p, m, e = curve(
                runs[N],
                lambda r, k=case: r[k]["mlp"]["acc_pure"] - r[k]["ridge"]["acc_pure"],
            )
            axes[2].plot(
                p, m, color=c, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4
            )
            axes[2].fill_between(p, m - e, m + e, color=c, alpha=0.12, lw=0)
    axes[0].set_ylabel("purified fraction 1 − S_Q (record M)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_ylabel("share of parities of weight ≤ 1")
    axes[2].set_ylabel("acc(MLP) − acc(ridge), purified trajectories")
    axes[2].axhline(0, color="#898781", lw=1)
    axes[1].plot([], [], color="#52514e", ls="-", label="record M")
    axes[1].plot([], [], color="#52514e", ls="--", label="M + final readout")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        _style(ax)
        ax.axvline(0.16, color="#898781", lw=0.8, ls=":")
        ax.set_xlabel("measurement probability p")
    fig.tight_layout()
    fig.savefig(out / "purification_parity.png", dpi=200)
    plt.close(fig)


def summary(runs: dict[int, list[dict]]) -> dict:
    """Mean and standard error of every curve, per ``N`` and case."""
    out: dict = {}
    for N, rs in runs.items():
        out[f"N{N}"] = {"n_circuits": len(rs), "T": 2 * N}
        for case in CASES:
            block: dict = {}
            for name, getter in {
                "purified": lambda r, k=case: r[k]["purified"],
                "weight_one": lambda r, k=case: r[k]["weight_one"],
                **{
                    f"{ro}_{q}": (lambda r, k=case, ro=ro, q=q: r[k][ro][q])
                    for ro in READOUTS
                    for q in ("acc_all", "acc_pure")
                },
            }.items():
                p, m, e = curve(rs, getter)
                block[name] = {"mean": m.round(4).tolist(), "err": e.round(4).tolist()}
            block["p"] = p.tolist()
            out[f"N{N}"][case] = block
    return out


def main() -> None:
    """Parse the folder and draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.folder
    out.mkdir(parents=True, exist_ok=True)
    runs = load(args.folder)
    plot_accuracy(runs, out)
    plot_purification_parity(runs, out)
    (out / "summary.json").write_text(json.dumps(summary(runs), indent=1))


if __name__ == "__main__":
    main()
