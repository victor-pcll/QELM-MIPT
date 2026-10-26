#!/usr/bin/env python3
"""Draw the learnability zone of the unitary QELM against ``h`` (issue #194).

Reads the run folders written by ``exp_QELM_bayes_h.sh`` (one per ``N``,
``h`` and seed), averages the scores over the seeds and draws, for each
``N``, one figure of two panels against the transverse field ``h / J``:

- the accuracy of the chance level (majority charge), of the Ridge readout
  and of the Bayes classifier (QDA on the ``<Z_k>``);
- their mean entropy ``E[H(P_corr)]``, in bits.

The band between the Bayes and the chance curves is filled: it is the
learnability zone, what the ``<Z_k>`` say about the charge beyond a blind
guess. The Ridge curve inside it shows how much of that a linear readout
gets. Error bars are the standard error over the seeds (none with a single
seed). Each figure is ``learnability_zone_N<N>.png`` in the sweep folder,
with its data ``learnability_zone_N<N>.json``
(:func:`QELM.utils.plots.save_figure`).

    uv run python experiments/QELM_bayes_h/plot_bayes_h.py \
        outputs/QELM_bayes_h/<date_time>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    COLORS,
    draw_curve,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("h", "TFIM.h"),
    ("t", "evolution.t"),
    ("L", "samples.L"),
    ("initial", "ancilla_charge_sector.initial"),
)
# (classifier, legend label, color, line style).
CLASSIFIERS = (
    ("chance", "chance (majority charge)", "#7f7f7f", "--"),
    ("ridge", r"Ridge on $\langle Z_k \rangle$", COLORS[0], "-"),
    ("bayes", r"Bayes (QDA) on $\langle Z_k \rangle$", COLORS[1], "-"),
)
# (metric prefix, axis label, panel title).
PANELS = (
    ("accuracy", "accuracy", "Accuracy"),
    ("entropy", r"mean entropy $E[H(P_{corr})]$ (bits)", "Mean entropy"),
)
ZONE_COLOR = COLORS[1]
FIGURE_NAME = "learnability_zone"


def metric_names() -> list[str]:
    """Return the metrics of the runs that the figures read."""
    return [f"{panel}_{name}" for panel, _, _ in PANELS for name, *_ in CLASSIFIERS]


def draw_size(summary: list[dict[str, Any]], N: int) -> Figure:
    """Draw the two panels of one ``N``.

    Args:
        summary: Output of :func:`QELM.utils.summarize` grouped by
            ``(N, h)``, with the metrics of :func:`metric_names`.
        N: Total number of qubits, ancilla included.

    Returns:
        The open figure.
    """
    fig, axes = subplots(1, 2, figsize=(12, 5))
    for ax, (panel, label, title) in zip(axes, PANELS, strict=True):
        curves = {
            name: sweep_curve(summary, N, "h", f"{panel}_{name}")
            for name, *_ in CLASSIFIERS
        }
        ax.fill_between(
            curves["bayes"].x,
            curves["chance"].mean,
            curves["bayes"].mean,
            color=ZONE_COLOR,
            alpha=0.15,
            linewidth=0,
            label="learnability zone",
        )
        for name, legend, color, style in CLASSIFIERS:
            draw_curve(ax, curves[name], color, label=legend, linestyle=style)
        ax.set(xlabel=r"transverse field $h / J$", ylabel=label, title=title)
    axes[0].set_ylim(0.4, 1.02)
    axes[1].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="center right")
    rows = [r for r in summary if int(r["N"]) == N]
    fig.suptitle(f"Unitary QELM, N = {N} (Ns = {N - 1}), t = 10 / J")
    fig.text(0.01, 0.01, f"{rows[0]['n_runs']} seed(s) per point", fontsize=9)
    return fig


def plot_sweep(sweep_dir: Path) -> list[Path]:
    """Write one figure and its data per ``N`` in ``sweep_dir``.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.

    Returns:
        The paths of the PNG files, by increasing ``N``.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    summary = summarize(rows, by=("N", "h"), metrics=metric_names())
    paths = []
    for N in sizes(summary):
        fig = draw_size(summary, N)
        data = {"summary": [r for r in summary if int(r["N"]) == N]}
        paths.append(save_figure(fig, sweep_dir, f"{FIGURE_NAME}_N{N}", data=data))
        plt.close(fig)
    return paths


def main() -> None:
    """Parse the command line and plot one sweep."""
    # No window from the command line: draw to the PNG only.
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/QELM_bayes_h",
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "QELM_bayes_h")
    for path in plot_sweep(sweep_dir):
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
