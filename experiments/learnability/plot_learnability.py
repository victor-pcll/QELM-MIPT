#!/usr/bin/env python3
"""Overlay a learnability sweep on the reference curves of Fig. 2b and 2c.

Reads the run folders written by ``learnability.sh`` (one per ``N``, seed
and ``p``), averages the observables over the seeds and draws, against the
measurement probability ``p``, two figures after Fig. 2 of Barratt et al.,
PRL 129, 200602 (2022):

- Fig. 2b (issue #113), ``learnability.png``: the accuracy of Eve's optimal
  classifier and, from its inset, the Binder ratio of ``P_corr``;
- Fig. 2c (issue #115), ``learnability_fig2c.png``: the mean entropy
  ``E[H(P_corr)]``, order parameter of the success transition, and, from its
  inset, the Binder ratio of ``H(P_corr)``.

The reference curves digitized from the paper are read from ``reference/``
next to this script when present:

- ``barratt2022_fig2b_accuracy.csv``: accuracy (Fig. 2b);
- ``barratt2022_fig2b_binder.csv``: Binder ratio of ``P_corr`` (inset of 2b);
- ``barratt2022_fig2c_entropy.csv``: mean entropy (Fig. 2c);
- ``barratt2022_fig2c_binder.csv``: Binder ratio of the entropy (inset of 2c).

Each is a WebPlotDigitizer CSV export (:func:`QELM.utils.load_wpd_csv`).
For every reference curve of the same size, the script prints the deviation
of our curve at the reference points (:func:`QELM.utils.compare_to_reference`).
With several sizes it also locates where their Binder curves cross
(:func:`binder_crossings`), to compare with the paper's ``p_tails ~ 0.1``
(Fig. 2b) and ``p_success ~ 0.2`` (Fig. 2c). Both go to ``comparison.json``
(Fig. 2b) and ``comparison_fig2c.json`` (Fig. 2c) in the sweep folder, next
to the figures.

Notation: ``N`` is the chain length everywhere here, the ``L`` of the paper
(no ancilla). The paper's curve names (``L = 6``) are read as ``N = 6``, so
that our curve ``N6`` is compared to the reference ``N = 6``. The number of
records per run is ``samples.L`` in the config, ``n_records`` here.

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

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

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
# Binder panels show 0 <= p <= BINDER_WINDOW * p_c around the transition.
BINDER_WINDOW = 2.0
# Fig. 2c: the mean entropy, order parameter of the success transition.
PANELS_2C: tuple[tuple[str, str, str], ...] = (
    ("entropy_mean", "barratt2022_fig2c_entropy", r"mean entropy $E[H(P_{corr})]$"),
    ("entropy_binder", "barratt2022_fig2c_binder", "Binder ratio of the entropy"),
)
# Transition read off the Binder crossing of the inset of Fig. 2c.
P_SUCCESS = 0.2
# Every figure: (panels, (transition label, p), figure, comparison file).
FIGURES: dict[
    str, tuple[tuple[tuple[str, str, str], ...], tuple[str, float], str, str]
] = {
    "2b": (PANELS, (r"$p_{tails}$", P_TAILS), "learnability.png", "comparison.json"),
    "2c": (
        PANELS_2C,
        (r"$p_{success}$", P_SUCCESS),
        "learnability_fig2c.png",
        "comparison_fig2c.json",
    ),
}
# True if the exports hold (value, p) pairs instead of (p, value), as for the
# Fig. 7 curves of issue #137 (see reference/README.md): the Fig. 2b export
# of issue #113 is one of them.
SWAP_XY = True
# Value axes digitized on the wrong scale, as {stem: (low, high)}: the export
# reads 0 to 1 where the figure's value runs from low to high, so a digitized
# value v stands for low + (high - low) * v. For the accuracy of Fig. 2b
# (issue #113), the two ends are read off the figure: every curve starts at
# 0.59 at p = 0.025 (digitized 0.060 for L = 20) and the L = 20 curve
# saturates at 1 (digitized 1.000).
VALUE_AXIS = {"barratt2022_fig2b_accuracy": (0.564, 1.0)}
# The paper's chain length "L = 6" in a curve name, renamed to our "N = 6".
PAPER_SIZE = re.compile(r"\bL\s*=")
# Categorical slots in fixed order: ours first, then the reference curves.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#d63a8a")


def load_references(
    reference_dir: Path, panels: Sequence[tuple[str, str, str]] = PANELS
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{metric: {curve name: (p, value)}}`` for the reference files found.

    The curve names of the exports use the paper's ``L`` for the chain
    length (``L = 6``); they are renamed to our ``N`` (``N = 6``).
    """
    refs = {}
    for metric, stem, _ in panels:
        path = reference_dir / f"{stem}.csv"
        if path.is_file():
            curves = load_wpd_csv(
                path, default_name="Barratt et al. 2022", swap_xy=SWAP_XY
            )
            low, high = VALUE_AXIS.get(stem, (0.0, 1.0))
            refs[metric] = {
                PAPER_SIZE.sub("N =", name): (x, low + (high - low) * y)
                for name, (x, y) in curves.items()
            }
    return refs


def accuracy_lower_bound(p: np.ndarray, t_f_over_L: float = 1.0) -> np.ndarray:
    """Return the accuracy lower bound of Eq. (1) of Barratt et al. 2022.

    ``alpha = (1 + erf(sqrt(p t_f / L))) / 2``: the accuracy of a classifier
    that treats every measured bit as an independent coin, biased by the
    charge density, for ``Q0 = L / 2`` and ``Q1 = L / 2 - 1``. It is the
    dashed curve of Fig. 2b, with ``t_f = L``.

    Args:
        p: Measurement probabilities.
        t_f_over_L: Ratio ``t_f / L``, 1 in the paper.

    Returns:
        The bound, of the shape of ``p``.
    """
    return np.asarray(0.5 * (1.0 + erf(np.sqrt(np.asarray(p) * t_f_over_L))))


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
        ``{"N<a>-N<b>": p of the first crossing, or None}``, on the ``p``
        values that both sizes share.
    """
    sizes = sorted({int(r["N"]) for r in summary})
    out: dict[str, float | None] = {}
    for a, b in itertools.combinations(sizes, 2):
        ya = {r["p"]: r[f"{metric}_mean"] for r in summary if int(r["N"]) == a}
        yb = {r["p"]: r[f"{metric}_mean"] for r in summary if int(r["N"]) == b}
        p = np.array(sorted(set(ya) & set(yb)), dtype=float)
        out[f"N{a}-N{b}"] = curve_crossing(
            p, np.array([ya[q] for q in p]), np.array([yb[q] for q in p])
        )
    return out


def plot_sweep(
    sweep_dir: Path,
    reference_dir: Path = REFERENCE_DIR,
    panels: Sequence[tuple[str, str, str]] = PANELS,
    filename: str = "learnability.png",
    transition: tuple[str, float] | None = (r"$p_{tails}$", P_TAILS),
    comparison_name: str = "comparison.json",
) -> dict[str, Any]:
    """Draw the figure and write the comparison file in ``sweep_dir``.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.
        panels: ``(metric, reference stem, label)``, one panel each.
        filename: Name of the figure in ``sweep_dir``.
        transition: ``(label, p)`` of the paper's transition, drawn as a
            dotted line on the Binder panels; ``None`` draws none.
        comparison_name: Name of the JSON file of the comparison.

    Returns:
        The comparison: ``{metric: {N<N>: {curve: deviation stats}}}`` and,
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
        for k, key in enumerate([*(f"N{N}" for N in sizes), *ref_names])
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
                color=color_of[f"N{N}"],
                lw=2,
                marker="o",
                ms=5,
                capsize=2,
                label=f"this work, N={N}",
            )
            for name, (x_ref, y_ref) in refs.get(metric, {}).items():
                # Compare to the reference of the same size only, when the
                # curve name gives one ("N = 6").
                size = re.search(r"N\s*=\s*(\d+)", name)
                if size is not None and int(size.group(1)) != N:
                    continue
                stats = compare_to_reference(p, mean, x_ref, y_ref)
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
                label=f"Barratt et al. 2022: {name}",
            )
        if "binder" in metric:
            comparison.setdefault("binder_crossings", {})[metric] = binder_crossings(
                summary, metric
            )
            if transition is not None:
                name, p_c = transition
                ax.axvline(p_c, color="gray", ls=":", lw=1, label=f"{name} = {p_c:g}")
                # As the insets of Fig. 2: 0 <= p <= 2 p_c. Further out the
                # Binder ratio plunges to large negative values (rare
                # outliers) and would flatten the crossing.
                window = [
                    r[f"{metric}_mean"]
                    for r in summary
                    if r["p"] <= BINDER_WINDOW * p_c
                    and np.isfinite(r[f"{metric}_mean"])
                ]
                if window:
                    pad = 0.1 * (max(window) - min(window) or 1.0)
                    ax.set_xlim(0.0, BINDER_WINDOW * p_c)
                    ax.set_ylim(min(window) - pad, max(window) + pad)
        if metric.startswith("accuracy"):
            ax.axhline(0.5, color="gray", ls=":", lw=1)
            p_grid = np.linspace(0.0, max(float(r["p"]) for r in summary), 101)
            ax.plot(
                p_grid,
                accuracy_lower_bound(p_grid),
                color="gray",
                ls="--",
                lw=1.5,
                label="lower bound, Eq. (1)",
            )
        ax.set_xlabel(r"measurement probability $p$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Charge learnability, Barratt et al. PRL 129, 200602 (2022), Fig. 2")
    fig.savefig(sweep_dir / filename, dpi=150)
    plt.close(fig)
    (sweep_dir / comparison_name).write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    """Parse the command line and plot one sweep."""
    # No window from the command line: draw to the PNG only. The notebooks
    # import this module without this, so their figures show inline (#144).
    plt.switch_backend("Agg")
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
    for panels, transition, filename, comparison_name in FIGURES.values():
        comparison = plot_sweep(
            sweep_dir,
            args.reference_dir,
            panels,
            filename,
            transition=transition,
            comparison_name=comparison_name,
        )
        for metric, crossings in comparison.pop("binder_crossings", {}).items():
            for pair, p in crossings.items():
                where = "no crossing" if p is None else f"p = {p:.3f}"
                print(f"{metric} crossing {pair}: {where}")
        for metric, by_size in comparison.items():
            for size, curves in by_size.items():
                for name, stats in curves.items():
                    print(
                        f"{metric} {size} vs {name}: max |diff| = "
                        f"{stats['max_abs_diff']:.3g}, "
                        f"rms = {stats['rms_diff']:.3g} ({stats['n_points']} points)"
                    )
        print(f"Figure: {sweep_dir / filename}")


if __name__ == "__main__":
    main()
