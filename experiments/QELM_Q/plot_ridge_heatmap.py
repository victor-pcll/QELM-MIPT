#!/usr/bin/env python3
"""Heatmaps of the ridge accuracy over (p, h) for [M], [M, Z] and [Z] (issue #225).

Reads an ``exp_QELM_Q.sh`` sweep of one size ``N`` over the measurement
probability ``p`` and the transverse field ``h`` and draws, for each feature
space of the linear readout, its test accuracy on the charge ``Q`` averaged
over the seeds:

- ``[M]``: the record of the intermediate measurements (``accuracy_M``);
- ``[M, Z]``: the record next to the final ``<Z_k>`` (``accuracy_M_Z``);
- ``[Z]``: the final ``<Z_k>`` alone (``accuracy``).

This is the wall of the linear readout only: where the accuracy falls, no
hyperplane of that feature space separates the two sectors any more. It says
nothing of what a nonlinear rule could still read (the Bayes ceilings of
``plot_bayes_ceiling.py`` and ``gmm_phase_diagram.py``).

Every cell holds its value, and contour lines at :data:`LEVELS` mark where
the accuracy crosses them, interpolated between the cell centers (on
``log h``). Written in the sweep folder with its JSON
(:func:`QELM.utils.plots.save_figure`): ``ridge_heatmap_p_h.png``.

The sweep of the figure (``N = 8``, 11 values of ``p``, 10 fields,
``L = 1600``, about 45 min on 4 cores):

    N_VALUES=8 H_VALUES="0.3 0.5 0.7 1 1.5 2 3 5 7 10" \\
        P_VALUES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" \\
        DELTA=200 STATE=vector MAX_JOBS=4 ./experiments/QELM_Q/exp_QELM_Q.sh
    uv run python experiments/QELM_Q/plot_ridge_heatmap.py \\
        outputs/exp_QELM_Q/<date_time>
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import draw_heatmap, save_figure, subplots, sweep_grid

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("h", "TFIM.h"),
    ("t", "evolution.t"),
    ("final", "projective_measurement.final_measurement"),
)
# Parameters that the runs must share.
FIXED = ("N", "t", "final")
# Feature spaces, one panel each: (metric, title).
MAPS = (
    ("accuracy_M", r"$[M]$"),
    ("accuracy_M_Z", r"$[M, Z]$"),
    ("accuracy", r"$[Z]$"),
)
# Accuracies marked by a contour line.
LEVELS = (0.6, 0.75, 0.9)
FIGURE = "ridge_heatmap_p_h"


def select_runs(
    rows: Sequence[dict[str, Any]], **fixed: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the runs with the given ``N``, ``t``, ``final``; check they are unique.

    Args:
        rows: Output of :func:`QELM.utils.load_sweep` with :data:`KEYS`.
        fixed: Values of :data:`FIXED` to keep; ``None`` keeps every value.

    Returns:
        The kept rows and the single value of each key of :data:`FIXED`.

    Raises:
        ValueError: If no row is left, or a key of :data:`FIXED` still takes
            several values.
    """
    kept = [
        r
        for r in rows
        if all(v is None or r[k] == v for k, v in fixed.items() if k in FIXED)
    ]
    if not kept:
        raise ValueError(f"No run with {fixed}.")
    values = {}
    for key in FIXED:
        found = sorted({r[key] for r in kept}, key=str)
        if len(found) > 1:
            raise ValueError(f"The sweep has several {key} = {found}: pick one.")
        values[key] = found[0]
    return kept, values


def draw_maps(
    rows: Sequence[dict[str, Any]],
    values: dict[str, Any],
    annotate: bool = True,
) -> tuple[Figure, dict[str, Any]]:
    """Draw the ridge accuracy of each feature space over ``(p, h)``.

    Args:
        rows: Runs of one ``(N, t, final)``, from :func:`select_runs`.
        values: Their ``N``, ``t`` and ``final``.
        annotate: Write the value of each cell in it.

    Returns:
        The open figure and, per metric of :data:`MAPS`, the grid drawn:
        ``p``, ``h`` and the mean over the seeds (rows: ``h``, columns:
        ``p``).
    """
    summary = summarize(rows, by=("N", "p", "h"), metrics=[m for m, _ in MAPS])
    log_h = all(float(r["h"]) > 0 for r in rows)
    fig, axes = subplots(1, len(MAPS), figsize=(5.6 * len(MAPS), 4.8), squeeze=False)
    extra: dict[str, Any] = {}
    for ax, (metric, title) in zip(axes[0], MAPS, strict=True):
        grid = sweep_grid(summary, "p", "h", metric)
        draw_heatmap(ax, grid, label="ridge accuracy", log_y=log_h, vmin=0.5, vmax=1.0)
        mean = np.asarray(grid.mean, dtype=float)
        if len(grid.x) > 1 and len(grid.y) > 1 and np.all(np.isfinite(mean)):
            lines = ax.contour(
                grid.x, grid.y, mean, levels=LEVELS, colors="white", linewidths=1.2
            )
            ax.clabel(lines, fmt="%.2f", fontsize=8)
        if annotate:
            for j, h in enumerate(grid.y):
                for i, p in enumerate(grid.x):
                    if np.isfinite(mean[j, i]):
                        ax.text(
                            p,
                            h,
                            f"{mean[j, i]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if mean[j, i] < 0.8 else "black",
                        )
        ax.set(xlabel=r"measurement probability $p$", ylabel=r"$h / J$")
        ax.set_title(title)
        extra[metric] = {
            "p": grid.x.tolist(),
            "h": grid.y.tolist(),
            "mean": mean.tolist(),
        }
    final = ", last round after the last step" if values["final"] else ""
    seeds = len({r["seed"] for r in rows})
    fig.suptitle(
        f"Ridge accuracy on $Q$, N = {values['N']}, t = {values['t']:g} / J{final}, "
        f"{seeds} seed{'s' if seeds > 1 else ''}"
    )
    return fig, extra


def plot_sweep(sweep_dir: Path, annotate: bool = True, **fixed: Any) -> Path:
    """Write :data:`FIGURE` (``.png`` and ``.json``) in the sweep folder.

    Args:
        sweep_dir: Folder of the sweep.
        annotate: As in :func:`draw_maps`.
        fixed: ``N``, ``t``, ``final`` to keep, as in :func:`select_runs`.

    Returns:
        The path of the PNG file.
    """
    rows, values = select_runs(load_sweep(sweep_dir, keys=KEYS), **fixed)
    fig, extra = draw_maps(rows, values, annotate)
    out = save_figure(fig, sweep_dir, FIGURE, data={**values, "maps": extra})
    plt.close(fig)
    return out


def main() -> None:
    """Parse the command line and plot one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/exp_QELM_Q",
    )
    parser.add_argument("--N", type=int, default=None, help="size N to plot")
    parser.add_argument("--t", type=float, default=None, help="evolution.t to plot")
    parser.add_argument(
        "--final",
        choices=("true", "false"),
        default=None,
        help="final_measurement to plot",
    )
    parser.add_argument(
        "--no-values", action="store_true", help="do not write the value of each cell"
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "exp_QELM_Q")
    final = None if args.final is None else args.final == "true"
    png = plot_sweep(
        sweep_dir, annotate=not args.no_values, N=args.N, t=args.t, final=final
    )
    print(f"Figure: {png}")


if __name__ == "__main__":
    main()
