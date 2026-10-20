#!/usr/bin/env python3
"""Overlay a charge-sharpening sweep on the reference curves of Fig. 7.

Reads the run folders written by ``charge_sharpening.sh`` (one per ``N``,
``seed`` and ``gamma``), averages the observables over the seeds (circuit
instances) and draws the three panels of Fig. 7 of Agrawal et al.,
PRX 14, 041012 (2024): ``S_R``, its variance and its Binder ratio against
``gamma``. The reference curves digitized from the paper (issue #137) are
read from ``reference/`` next to this script when present:

- ``agrawal2024_fig7_S_R.csv``: mean ancilla entropy (left panel);
- ``agrawal2024_fig7_S_R_var.csv``: variance of ``S_R`` (center);
- ``agrawal2024_fig7_binder.csv``: Binder ratio (right).

Each is a WebPlotDigitizer CSV export (:func:`QELM.utils.load_wpd_csv`).
For every reference curve the script prints the deviation of our curve at
the reference points (:func:`QELM.utils.compare_to_reference`) and writes it
to ``comparison.json`` in the sweep folder, next to ``charge_sharpening.png``.

Error bars: with one seed, the bootstrap error over trajectories; with
several, the standard error over circuit instances, and the shaded band is
the spread (one standard deviation) between instances. The Binder ratio is
only drawn where ``Var S_R > BINDER_MIN_VAR``; ``comparison.json`` uses the
same points.

    uv run python experiments/charge_sharpening/plot_charge_sharpening.py \
        outputs/charge_sharpening/<date_time>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.utils import (
    compare_to_reference,
    latest_sweep,
    load_sweep,
    load_wpd_csv,
    summarize,
)

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
KEYS = (
    ("N", "N"),
    ("Ns", "Ns"),
    ("seed", "seed"),
    ("gamma", "weak_measurement.gamma"),
    ("L", "samples.L"),
)
# (metric, reference file stem, axis label), one panel each, as in Fig. 7.
PANELS = (
    ("S_R_mean", "agrawal2024_fig7_S_R", r"$S_R$ (nats)"),
    ("S_R_var", "agrawal2024_fig7_S_R_var", r"Var $S_R$ (nats$^2$)"),
    ("binder", "agrawal2024_fig7_binder", r"Binder ratio $B$"),
)
# The digitized exports list (value, gamma), not (gamma, value).
SWAP_XY = True
# The Binder ratio is drawn only where Var S_R exceeds this: elsewhere S_R
# has collapsed on 1 (weak gamma) or 0 (strong gamma) and B is dominated by
# rare outliers (large and negative), which would flatten the panel.
BINDER_MIN_VAR = 1e-3
# Categorical slots in fixed order: ours first, then the reference curves.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6")


def load_references(
    reference_dir: Path,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{metric: {curve name: (x, y)}}`` for the reference files found."""
    refs = {}
    for metric, stem, _ in PANELS:
        path = reference_dir / f"{stem}.csv"
        if path.is_file():
            # The exports hold (value, gamma) pairs: see reference/README.md.
            refs[metric] = load_wpd_csv(
                path, default_name="Agrawal et al. 2024", swap_xy=SWAP_XY
            )
    return refs


def draw_sweep(
    sweep_dir: Path, reference_dir: Path = REFERENCE_DIR
) -> tuple[Figure, dict[str, Any]]:
    """Draw the three panels of Fig. 7 for a sweep, over the reference curves.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.

    Returns:
        The open figure (the notebook shows it, :func:`plot_sweep` saves and
        closes it) and the comparison,
        ``{metric: {N<N>: {curve: deviation stats}}}``.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    metrics = [m for m, _, _ in PANELS]
    summary = summarize(
        rows, by=("N", "gamma"), metrics=[*metrics, *(f"{m}_err" for m in metrics)]
    )
    refs = load_references(reference_dir)
    sizes = sorted({int(row["N"]) for row in summary})
    comparison: dict[str, Any] = {}
    # One color per curve, the same in every panel: ours first, then the
    # reference curves by name (the files do not list them in the same order).
    ref_names = sorted({name for curves in refs.values() for name in curves})
    color_of = {
        key: COLORS[k % len(COLORS)]
        for k, key in enumerate([*(f"N{N}" for N in sizes), *ref_names])
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (metric, _, label) in zip(axes, PANELS, strict=True):
        for N in sizes:
            sel = sorted(
                (r for r in summary if int(r["N"]) == N), key=lambda r: r["gamma"]
            )
            gamma = np.array([r["gamma"] for r in sel], dtype=float)
            mean = np.array([r[f"{metric}_mean"] for r in sel])
            n_runs = np.array([r["n_runs"] for r in sel])
            # One instance: bootstrap error; several: error over instances.
            err = np.where(
                n_runs > 1,
                [r[f"{metric}_sem"] for r in sel],
                [r[f"{metric}_err_mean"] for r in sel],
            )
            if metric == "binder":
                var = np.array([r["S_R_var_mean"] for r in sel])
                keep = var > BINDER_MIN_VAR
                gamma, mean, n_runs, err = (
                    gamma[keep],
                    mean[keep],
                    n_runs[keep],
                    err[keep],
                )
                sel = [r for r, kept in zip(sel, keep, strict=True) if kept]
            color = color_of[f"N{N}"]
            if np.all(n_runs > 1):
                std = np.array([r[f"{metric}_std"] for r in sel])
                ax.fill_between(
                    gamma, mean - std, mean + std, color=color, alpha=0.15, lw=0
                )
            ax.errorbar(
                gamma,
                mean,
                yerr=err,
                color=color,
                lw=2,
                marker="o",
                ms=5,
                capsize=2,
                label=f"this work, L={N - 1}",
            )
            for name, (x_ref, y_ref) in refs.get(metric, {}).items():
                # Compare to the reference of the same size only, when the
                # curve name gives one ("L = 6"); our L is Ns = N - 1.
                size = re.search(r"L\s*=\s*(\d+)", name)
                if size is not None and int(size.group(1)) != N - 1:
                    continue
                stats = compare_to_reference(gamma, mean, x_ref, y_ref)
                comparison.setdefault(metric, {}).setdefault(f"N{N}", {})[name] = stats
        for name, (x_ref, y_ref) in refs.get(metric, {}).items():
            ax.plot(
                x_ref,
                y_ref,
                color=color_of[name],
                lw=2,
                ls="--",
                marker="s",
                ms=5,
                label=f"Agrawal et al. 2024: {name}",
            )
        ax.set_xlabel(r"measurement strength $\gamma$")
        ax.set_ylabel(label)
        if metric == "binder" and metric in refs:
            # Far from the peak B drops to large negative values (rare
            # outliers); keep the scale of the reference curves.
            low = min(float(np.min(y)) for _, y in refs[metric].values())
            ax.set_ylim(1.5 * low, 1.0)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Charge sharpening, Agrawal et al. PRX 14, 041012 (2024), Fig. 7")
    return fig, comparison


def plot_sweep(sweep_dir: Path, reference_dir: Path = REFERENCE_DIR) -> dict[str, Any]:
    """Draw ``charge_sharpening.png`` and write ``comparison.json`` in ``sweep_dir``.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.

    Returns:
        The comparison, ``{metric: {N<N>: {curve: deviation stats}}}``.
    """
    fig, comparison = draw_sweep(sweep_dir, reference_dir)
    fig.savefig(sweep_dir / "charge_sharpening.png", dpi=150)
    plt.close(fig)
    (sweep_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    """Parse the command line and plot one sweep."""
    # No window from the command line: draw to the PNG only. The notebook
    # imports this module without this, so its figures show inline.
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/charge_sharpening",
    )
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "charge_sharpening")
    comparison = plot_sweep(sweep_dir, args.reference_dir)
    for metric, by_size in comparison.items():
        for size, curves in by_size.items():
            for name, stats in curves.items():
                print(
                    f"{metric} {size} vs {name}: max |diff| = "
                    f"{stats['max_abs_diff']:.3g}, rms = {stats['rms_diff']:.3g} "
                    f"({stats['n_points']} points)"
                )
    print(f"Figure: {sweep_dir / 'charge_sharpening.png'}")


if __name__ == "__main__":
    main()
