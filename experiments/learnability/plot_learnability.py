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

Notation (issues #147 and #235): ``N`` is the number of system qubits, here
the chain length since there is no ancilla, and ``L`` the number of records per run
(``samples.L`` in the config, ``n_records`` in the rows). The paper writes
its chain length ``L``: its curve names (``L = 6``) are renamed ``N = 6``,
so that our curve ``N6`` is compared to the reference ``N = 6``.

Each figure is written with its data (:func:`QELM.utils.plots.save_figure`):
``learnability.png`` with ``learnability.json``, and so on.

Error bars: with one seed, the bootstrap error over records; with several,
the standard error over seeds. They are often smaller than the markers:
``residuals=True`` (issue #152) adds under each panel our curve minus the
paper's curve of the same size, on a scale where they show.

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
from matplotlib.figure import Figure
from scipy.special import erf

from QELM.utils import latest_sweep, load_sweep, load_wpd_csv, summarize
from QELM.utils.plots import (
    color_map,
    compare_same_size,
    draw_curve,
    draw_references,
    draw_residuals,
    print_comparison,
    same_size,
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
    ("p", "learnability.p"),
    ("n_records", "samples.L"),
)
# (metric, reference file stem, axis label, panel title), one panel each.
Panel = tuple[str, str, str, str]
# Fig. 2b: the accuracy of Eve's classifier and, from its inset, the Binder
# ratio of P_corr.
PANELS: tuple[Panel, ...] = (
    ("accuracy", "barratt2022_fig2b_accuracy", "accuracy", "Accuracy"),
    (
        "P_corr_binder",
        "barratt2022_fig2b_binder",
        r"Binder ratio of $P_{corr}$",
        r"Binder ratio of $P_{corr}$",
    ),
)
SOURCE = "Barratt et al. 2022"
# Transition read off the Binder crossing of the inset of Fig. 2b.
P_TAILS = 0.1
# Binder panels show 0 <= p <= BINDER_WINDOW * p_c around the transition.
BINDER_WINDOW = 2.0
# Fig. 2c: the mean entropy, order parameter of the success transition.
PANELS_2C: tuple[Panel, ...] = (
    (
        "entropy_mean",
        "barratt2022_fig2c_entropy",
        r"mean entropy $E[H(P_{corr})]$ (nats)",
        "Mean entropy",
    ),
    (
        "entropy_binder",
        "barratt2022_fig2c_binder",
        "Binder ratio of the entropy",
        "Binder ratio of the entropy",
    ),
)
# Transition read off the Binder crossing of the inset of Fig. 2c.
P_SUCCESS = 0.2
# Every figure: (panels, (transition label, p), figure, comparison file).
FIGURES: dict[str, tuple[tuple[Panel, ...], tuple[str, float], str, str]] = {
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


def load_references(
    reference_dir: Path, panels: Sequence[Panel] = PANELS
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return ``{metric: {curve name: (p, value)}}`` for the reference files found.

    The curve names of the exports use the paper's ``L`` for the chain
    length (``L = 6``); they are renamed to our ``N`` (``N = 6``).
    """
    refs = {}
    for metric, stem, _, _ in panels:
        path = reference_dir / f"{stem}.csv"
        if path.is_file():
            curves = load_wpd_csv(path, default_name=SOURCE, swap_xy=SWAP_XY)
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
    out: dict[str, float | None] = {}
    for a, b in itertools.combinations(sizes(summary), 2):
        ya = {r["p"]: r[f"{metric}_mean"] for r in summary if int(r["N"]) == a}
        yb = {r["p"]: r[f"{metric}_mean"] for r in summary if int(r["N"]) == b}
        p = np.array(sorted(set(ya) & set(yb)), dtype=float)
        out[f"N{a}-N{b}"] = curve_crossing(
            p, np.array([ya[q] for q in p]), np.array([yb[q] for q in p])
        )
    return out


def draw_sweep(
    sweep_dir: Path,
    reference_dir: Path = REFERENCE_DIR,
    panels: Sequence[Panel] = PANELS,
    transition: tuple[str, float] | None = (r"$p_{tails}$", P_TAILS),
    residuals: bool = False,
) -> tuple[Figure, dict[str, Any]]:
    """Draw one figure of a sweep over the reference curves.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.
        panels: ``(metric, reference stem, label, title)``, one panel each.
        transition: ``(label, p)`` of the paper's transition, drawn as a
            dotted line on the Binder panels; ``None`` draws none.
        residuals: Add under every panel with a reference curve of one of
            our sizes a panel of our curve minus that reference, with our
            error bars (issue #152). The other residual panels are removed.

    Returns:
        The open figure (the notebooks show it, :func:`plot_sweep` saves and
        closes it) and the comparison: ``{metric: {N<N>: {curve: deviation
        stats}}}`` and, under ``"binder_crossings"``, :func:`binder_crossings`
        of every Binder metric drawn.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    metrics = [m for m, _, _, _ in panels]
    summary = summarize(
        rows, by=("N", "p"), metrics=[*metrics, *(f"{m}_err" for m in metrics)]
    )
    refs = load_references(reference_dir, panels)
    n_values = sizes(summary)
    comparison: dict[str, Any] = {}
    # One color per curve, the same in every panel: ours first, then the
    # reference curves by name.
    ref_names = sorted({name for curves in refs.values() for name in curves})
    colors = color_map([*(f"N{N}" for N in n_values), *ref_names])

    if residuals:
        fig, grid = subplots(
            2,
            len(panels),
            figsize=(6.5 * len(panels), 7),
            sharex="col",
            height_ratios=(3, 1),
            squeeze=False,
        )
        axes, residual_axes = grid
    else:
        fig, axes = subplots(1, len(panels), figsize=(6.5 * len(panels), 5))
        axes, residual_axes = np.atleast_1d(axes), [None] * len(panels)
    for ax, res_ax, (metric, _, label, title) in zip(
        axes, residual_axes, panels, strict=True
    ):
        drawn = False
        for N in n_values:
            curve = sweep_curve(summary, N, "p", metric)
            draw_curve(ax, curve, colors[f"N{N}"], label=f"this work, N = {N}")
            stats = compare_same_size(curve, N, refs.get(metric, {}))
            if stats:
                comparison.setdefault(metric, {})[f"N{N}"] = stats
            if res_ax is None:
                continue
            for name, (x_ref, y_ref) in same_size(refs.get(metric, {}), N).items():
                draw_residuals(
                    res_ax,
                    curve,
                    x_ref,
                    y_ref,
                    colors[f"N{N}"],
                    label=f"this work, N = {N} $-$ {SOURCE}: {name}",
                )
                drawn = True
        if res_ax is not None:
            if drawn:
                res_ax.set(
                    xlabel=r"measurement probability $p$", ylabel="this work $-$ paper"
                )
                res_ax.legend(fontsize=9)
            else:
                res_ax.remove()
                # sharex hid the tick labels of the panel above.
                ax.tick_params(labelbottom=True)
        draw_references(ax, refs.get(metric, {}), colors, SOURCE)
        if "binder" in metric:
            comparison.setdefault("binder_crossings", {})[metric] = binder_crossings(
                summary, metric
            )
            if transition is not None:
                name, p_c = transition
                ax.axvline(p_c, color="gray", ls=":", lw=1.5, label=f"{name} = {p_c:g}")
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
            ax.axhline(0.5, color="gray", ls=":", lw=1.5)
            p_grid = np.linspace(0.0, max(float(r["p"]) for r in summary), 101)
            ax.plot(
                p_grid,
                accuracy_lower_bound(p_grid),
                color="gray",
                ls="--",
                lw=1.5,
                label="lower bound, Eq. (1)",
            )
        ax.set(ylabel=label, title=title)
        if not drawn:
            ax.set_xlabel(r"measurement probability $p$")
        ax.legend()
    fig.suptitle("Charge learnability, Barratt et al. PRL 129, 200602 (2022), Fig. 2")
    return fig, comparison


def plot_sweep(
    sweep_dir: Path,
    reference_dir: Path = REFERENCE_DIR,
    panels: Sequence[Panel] = PANELS,
    filename: str = "learnability.png",
    transition: tuple[str, float] | None = (r"$p_{tails}$", P_TAILS),
    comparison_name: str = "comparison.json",
    residuals: bool = False,
) -> dict[str, Any]:
    """Write the figure, its data and the comparison file in ``sweep_dir``.

    The figure goes to ``filename`` and the data it shows to the JSON file
    of the same stem (:func:`QELM.utils.plots.save_figure`).

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        reference_dir: Folder of the reference CSV files; missing files are
            skipped.
        panels: ``(metric, reference stem, label, title)``, one panel each.
        filename: Name of the PNG figure in ``sweep_dir``.
        transition: As in :func:`draw_sweep`.
        comparison_name: Name of the JSON file of the comparison.
        residuals: As in :func:`draw_sweep`.

    Returns:
        The comparison of :func:`draw_sweep`.
    """
    fig, comparison = draw_sweep(
        sweep_dir, reference_dir, panels, transition, residuals=residuals
    )
    save_figure(fig, sweep_dir, Path(filename).stem)
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
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=sorted(FIGURES),
        default=sorted(FIGURES),
        help="figures to draw (issue #151); default: all",
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "learnability")
    for figure in args.figures:
        panels, transition, filename, comparison_name = FIGURES[figure]
        comparison = plot_sweep(
            sweep_dir,
            args.reference_dir,
            panels,
            filename,
            transition=transition,
            comparison_name=comparison_name,
        )
        print_comparison(comparison)
        print(f"Figure: {sweep_dir / filename}")


if __name__ == "__main__":
    main()
