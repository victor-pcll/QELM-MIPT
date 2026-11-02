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
to ``comparison.json`` in the sweep folder, next to ``charge_sharpening.png``
and the plotted data ``charge_sharpening.json``
(:func:`QELM.utils.plots.save_figure`).

Notation (issues #147 and #235): ``N`` is the number of system qubits, the
ancilla ``R`` excluded, as the paper's ``L``, and ``L`` here is the number
of trajectories per run (``samples.L``). The paper's curves, named
``L = 6`` and ``L = 20`` in the exports, are renamed ``N = 6`` and
``N = 20``, so that our curve ``N6`` is compared to ``N = 6``.

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

from QELM.utils import latest_sweep, load_sweep, load_wpd_csv, summarize
from QELM.utils.plots import (
    color_map,
    compare_same_size,
    draw_curve,
    draw_references,
    print_comparison,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("gamma", "weak_measurement.gamma"),
    ("L", "samples.L"),
)
# (metric, reference file stem, axis label, panel title), as in Fig. 7.
PANELS = (
    ("S_R_mean", "agrawal2024_fig7_S_R", r"$S_R$ (nats)", "Ancilla entropy"),
    ("S_R_var", "agrawal2024_fig7_S_R_var", r"Var $S_R$ (nats$^2$)", "Variance"),
    ("binder", "agrawal2024_fig7_binder", r"Binder ratio $B$", "Binder ratio"),
)
SOURCE = "Agrawal et al. 2024"
# The digitized exports list (value, gamma), not (gamma, value).
SWAP_XY = True
# The Binder ratio is drawn only where Var S_R exceeds this: elsewhere S_R
# has collapsed on 1 (weak gamma) or 0 (strong gamma) and B is dominated by
# rare outliers (large and negative), which would flatten the panel.
BINDER_MIN_VAR = 1e-3
# The paper's system size "L = 6" in a curve name: our N, the ancilla
# excluded (issue #235).
PAPER_SIZE = re.compile(r"\bL\s*=\s*(\d+)")
FIGURE_NAME = "charge_sharpening"


def paper_to_ours(name: str) -> str:
    """Rename the paper's system size ``L = 6`` to our ``N = 6``."""
    return PAPER_SIZE.sub(lambda m: f"N = {int(m.group(1))}", name)


def load_references(
    reference_dir: Path,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{metric: {curve name: (x, y)}}`` for the reference files found.

    The curve names are renamed with :func:`paper_to_ours`.
    """
    refs = {}
    for metric, stem, _, _ in PANELS:
        path = reference_dir / f"{stem}.csv"
        if path.is_file():
            # The exports hold (value, gamma) pairs: see reference/README.md.
            curves = load_wpd_csv(path, default_name=SOURCE, swap_xy=SWAP_XY)
            refs[metric] = {paper_to_ours(n): xy for n, xy in curves.items()}
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
    metrics = [m for m, _, _, _ in PANELS]
    summary = summarize(
        rows, by=("N", "gamma"), metrics=[*metrics, *(f"{m}_err" for m in metrics)]
    )
    refs = load_references(reference_dir)
    n_values = sizes(summary)
    comparison: dict[str, Any] = {}
    # One color per curve, the same in every panel: ours first, then the
    # reference curves by name (the files do not list them in the same order).
    ref_names = sorted({name for curves in refs.values() for name in curves})
    colors = color_map([*(f"N{N}" for N in n_values), *ref_names])

    fig, axes = subplots(1, 3, figsize=(16, 5))
    for ax, (metric, _, label, title) in zip(axes, PANELS, strict=True):
        for N in n_values:
            curve = sweep_curve(summary, N, "gamma", metric)
            if metric == "binder":
                var = np.array([r["S_R_var_mean"] for r in curve.rows])
                curve = curve.where(var > BINDER_MIN_VAR)
            draw_curve(
                ax, curve, colors[f"N{N}"], label=f"this work, N = {N}", band=True
            )
            stats = compare_same_size(curve, N, refs.get(metric, {}))
            if stats:
                comparison.setdefault(metric, {})[f"N{N}"] = stats
        draw_references(ax, refs.get(metric, {}), colors, SOURCE)
        ax.set(xlabel=r"measurement strength $\gamma$", ylabel=label, title=title)
        if metric == "binder" and metric in refs:
            # Far from the peak B drops to large negative values (rare
            # outliers); keep the scale of the reference curves.
            low = min(float(np.min(y)) for _, y in refs[metric].values())
            ax.set_ylim(1.5 * low, 1.0)
    axes[0].legend()
    fig.suptitle("Charge sharpening, Agrawal et al. PRX 14, 041012 (2024), Fig. 7")
    return fig, comparison


def plot_sweep(sweep_dir: Path, reference_dir: Path = REFERENCE_DIR) -> dict[str, Any]:
    """Write the figure, its data and ``comparison.json`` in ``sweep_dir``.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.

    Returns:
        The comparison, ``{metric: {N<N>: {curve: deviation stats}}}``.
    """
    fig, comparison = draw_sweep(sweep_dir, reference_dir)
    save_figure(fig, sweep_dir, FIGURE_NAME)
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
    print_comparison(plot_sweep(sweep_dir, args.reference_dir))
    print(f"Figure: {sweep_dir / f'{FIGURE_NAME}.png'}")


if __name__ == "__main__":
    main()
