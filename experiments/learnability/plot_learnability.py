#!/usr/bin/env python3
"""Overlay a learnability sweep on the reference curves of Fig. 2b.

Reads the run folders written by ``learnability.sh`` (one per ``N``, seed
and ``p``), averages the observables over the seeds and draws, against the
measurement probability ``p``, the panels of Fig. 2b of Barratt et al.,
PRL 129, 200602 (2022): the accuracy of Eve's optimal classifier and, from
its inset, the Binder ratio of ``P_corr``. The reference curves digitized
from the paper (issue #113) are read from ``reference/`` next to this script
when present:

- ``barratt2022_fig2b_accuracy.csv``: accuracy (main panel);
- ``barratt2022_fig2b_binder.csv``: Binder ratio of ``P_corr`` (inset).

Each is a WebPlotDigitizer CSV export (:func:`QELM.utils.load_wpd_csv`).
For every reference curve of the same size, the script prints the deviation
of our curve at the reference points (:func:`QELM.utils.compare_to_reference`).
With several sizes it also locates where their Binder curves cross
(:func:`binder_crossings`), to compare with the paper's ``p_tails ~ 0.1``.
Both go to ``comparison.json`` in the sweep folder, next to
``learnability.png``.

Error bars: with one seed, the bootstrap error over records; with several,
the standard error over seeds.

    uv run python experiments/learnability/plot_learnability.py \
        outputs/learnability/<date_time>
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from QELM.utils import (  # noqa: E402
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
    ("seed", "seed"),
    ("p", "learnability.p"),
    ("n_records", "samples.L"),
)
# (metric, reference file stem, axis label), one panel each, as in Fig. 2b.
PANELS: tuple[tuple[str, str, str], ...] = (
    ("accuracy", "barratt2022_fig2b_accuracy", "accuracy"),
    ("P_corr_binder", "barratt2022_fig2b_binder", r"Binder ratio of $P_{corr}$"),
)
# Transition read off the Binder crossing of the inset of Fig. 2b.
P_TAILS = 0.1
# True if the exports hold (value, p) pairs instead of (p, value), as for the
# Fig. 7 curves of issue #137 (see reference/README.md).
SWAP_XY = False
# Categorical slots in fixed order: ours first, then the reference curves.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#d63a8a")


def load_references(
    reference_dir: Path, panels: Sequence[tuple[str, str, str]] = PANELS
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{metric: {curve name: (p, value)}}`` for the reference files found."""
    refs = {}
    for metric, stem, _ in panels:
        path = reference_dir / f"{stem}.csv"
        if path.is_file():
            refs[metric] = load_wpd_csv(
                path, default_name="Barratt et al. 2022", swap_xy=SWAP_XY
            )
    return refs


def curve_crossing(x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> float | None:
    """Return the first ``x`` where two curves on the same grid cross.

    The curves are interpolated linearly between grid points, and the sign
    change of ``y_a - y_b`` is located inside the first interval where it
    occurs. ``nan`` values are skipped.

    Args:
        x: Common abscissae, increasing.
        y_a: First curve on ``x``.
        y_b: Second curve on ``x``.

    Returns:
        The crossing abscissa, or ``None`` if the curves do not cross.
    """
    keep = np.isfinite(y_a) & np.isfinite(y_b)
    x, d = np.asarray(x, float)[keep], (np.asarray(y_a) - np.asarray(y_b))[keep]
    for k in range(len(x) - 1):
        if d[k] == 0:
            return float(x[k])
        if d[k] * d[k + 1] < 0:
            return float(x[k] - d[k] * (x[k + 1] - x[k]) / (d[k + 1] - d[k]))
    if len(x) and d[-1] == 0:
        return float(x[-1])
    return None


def binder_crossings(
    summary: Sequence[dict[str, Any]], metric: str = "P_corr_binder"
) -> dict[str, float | None]:
    """Return where the curves of ``metric`` of every pair of sizes cross.

    Args:
        summary: Output of :func:`QELM.utils.summarize` by ``("N", "p")``.
        metric: Row key, ``<metric>_mean`` being read.

    Returns:
        ``{"L<a>-L<b>": p of the first crossing, or None}``, on the ``p``
        values that both sizes share.
    """
    sizes = sorted({int(r["N"]) for r in summary})
    out: dict[str, float | None] = {}
    for a, b in itertools.combinations(sizes, 2):
        ya = {r["p"]: r[f"{metric}_mean"] for r in summary if int(r["N"]) == a}
        yb = {r["p"]: r[f"{metric}_mean"] for r in summary if int(r["N"]) == b}
        p = np.array(sorted(set(ya) & set(yb)), dtype=float)
        out[f"L{a}-L{b}"] = curve_crossing(
            p, np.array([ya[q] for q in p]), np.array([yb[q] for q in p])
        )
    return out


def plot_sweep(
    sweep_dir: Path,
    reference_dir: Path = REFERENCE_DIR,
    panels: Sequence[tuple[str, str, str]] = PANELS,
    filename: str = "learnability.png",
) -> dict[str, Any]:
    """Draw the figure and write ``comparison.json`` in ``sweep_dir``.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.
        panels: ``(metric, reference stem, label)``, one panel each.
        filename: Name of the figure in ``sweep_dir``.

    Returns:
        The comparison: ``{metric: {L<N>: {curve: deviation stats}}}`` and,
        under ``"binder_crossings"``, :func:`binder_crossings` of every
        Binder metric drawn.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    metrics = [m for m, _, _ in panels]
    summary = summarize(
        rows, by=("N", "p"), metrics=[*metrics, *(f"{m}_err" for m in metrics)]
    )
    refs = load_references(reference_dir, panels)
    sizes = sorted({int(row["N"]) for row in summary})
    comparison: dict[str, Any] = {}
    # One color per curve, the same in every panel: ours first, then the
    # reference curves by name.
    ref_names = sorted({name for curves in refs.values() for name in curves})
    color_of = {
        key: COLORS[k % len(COLORS)]
        for k, key in enumerate([*(f"L{N}" for N in sizes), *ref_names])
    }

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4), constrained_layout=True
    )
    for ax, (metric, _, label) in zip(np.atleast_1d(axes), panels, strict=True):
        for N in sizes:
            sel = sorted((r for r in summary if int(r["N"]) == N), key=lambda r: r["p"])
            p = np.array([r["p"] for r in sel], dtype=float)
            mean = np.array([r[f"{metric}_mean"] for r in sel])
            n_runs = np.array([r["n_runs"] for r in sel])
            # One seed: bootstrap error; several: error over seeds.
            err = np.where(
                n_runs > 1,
                [r[f"{metric}_sem"] for r in sel],
                [r[f"{metric}_err_mean"] for r in sel],
            )
            ax.errorbar(
                p,
                mean,
                yerr=err,
                color=color_of[f"L{N}"],
                lw=2,
                marker="o",
                ms=5,
                capsize=2,
                label=f"this work, L={N}",
            )
            for name, (x_ref, y_ref) in refs.get(metric, {}).items():
                # Compare to the reference of the same size only, when the
                # curve name gives one ("L = 6").
                size = re.search(r"L\s*=\s*(\d+)", name)
                if size is not None and int(size.group(1)) != N:
                    continue
                stats = compare_to_reference(p, mean, x_ref, y_ref)
                comparison.setdefault(metric, {}).setdefault(f"L{N}", {})[name] = stats
        for name, (x_ref, y_ref) in refs.get(metric, {}).items():
            ax.plot(
                x_ref,
                y_ref,
                color=color_of[name],
                lw=2,
                ls="--",
                marker="s",
                ms=5,
                label=f"Barratt et al. 2022: {name}",
            )
        if "binder" in metric:
            comparison.setdefault("binder_crossings", {})[metric] = binder_crossings(
                summary, metric
            )
            ax.axvline(P_TAILS, color="gray", ls=":", lw=1, label=r"$p_{tails}$ = 0.1")
        if metric.startswith("accuracy"):
            ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel(r"measurement probability $p$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Charge learnability, Barratt et al. PRL 129, 200602 (2022), Fig. 2")
    fig.savefig(sweep_dir / filename, dpi=150)
    plt.close(fig)
    (sweep_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    """Parse the command line and plot one sweep."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/learnability",
    )
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "learnability")
    comparison = plot_sweep(sweep_dir, args.reference_dir)
    for metric, crossings in comparison.pop("binder_crossings", {}).items():
        for pair, p in crossings.items():
            where = "no crossing" if p is None else f"p = {p:.3f}"
            print(f"{metric} crossing {pair}: {where}")
    for metric, by_size in comparison.items():
        for size, curves in by_size.items():
            for name, stats in curves.items():
                print(
                    f"{metric} {size} vs {name}: max |diff| = "
                    f"{stats['max_abs_diff']:.3g}, rms = {stats['rms_diff']:.3g} "
                    f"({stats['n_points']} points)"
                )
    print(f"Figure: {sweep_dir / 'learnability.png'}")


if __name__ == "__main__":
    main()
