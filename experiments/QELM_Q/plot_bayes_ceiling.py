#!/usr/bin/env python3
"""Draw the Bayes ceilings of a p-sweep of exp_QELM_Q (issue #202).

Reads the run folders written by ``exp_QELM_Q.sh`` (one per ``N``, seed and
``p``), averages over the seeds and draws two figures in the sweep folder:

- ``bayes_ceiling.png``: the accuracy of the ridge readout of ``Q`` from
  ``<Z_k>`` against ``p``, one solid curve per size ``N`` (finite-size
  scaling), with, in the same color, its two ceilings: the QDA
  (``bayes_estimate_qda``, dashed) and the K-NN (``bayes_estimate_knn``,
  dash-dotted), and chance (0.5, dotted). ``--no-ceilings`` leaves the
  ceilings out.
- ``qda_contours.png``: one panel per ``(N, p)``, the test ``<Z_k>`` of one
  run projected on their plane of largest variance (PCA), colored by
  charge sector, and the 1 and 2 sigma ellipses of the Gaussian ``f_k``
  fitted by the QDA on the training samples of each sector.

Reading them together: where the ridge accuracy falls to 0.5,

- ceilings at 0.5 and ellipses on top of each other: the sectors overlap,
  the information is erased (decoherence);
- ceilings well above 0.5 and ellipses apart: the information is in
  ``<Z_k>`` but not linearly (learnability transition).

The QDA of the contours is refitted from the ``.npz`` of the run, on the
same split as the run, so its means and covariances are those behind
``bayes_estimate_qda``. Each figure is written with its data
(:func:`QELM.utils.plots.save_figure`).

The sweep must hold one value of ``TFIM.h``, ``evolution.t`` and
``final_measurement``, or the one to plot is chosen with ``--h``, ``--t``
and ``--final``:

    uv run python experiments/QELM_Q/plot_bayes_ceiling.py \\
        outputs/exp_QELM_Q/<date_time> --h 1 --p 0 0.5 1
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.evaluation.bayes_optimal import QDA
from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    COLORS,
    color_map,
    draw_companion,
    draw_curve,
    draw_qda_contours,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("h", "TFIM.h"),
    ("t", "evolution.t"),
    ("final", "projective_measurement.final_measurement"),
)
# Parameters that the plotted runs must share.
FIXED = ("h", "t", "final")
METRICS = ("accuracy", "bayes_estimate_qda", "bayes_estimate_knn")
# (metric, legend name, line style) of the ceilings.
CEILINGS = (
    ("bayes_estimate_qda", "QDA", "--"),
    ("bayes_estimate_knn", "K-NN", "-."),
)
# Same regularization as evaluate_bayes_optimal_limit.
REG_PARAM = 1.0e-6
NPZ = "exp_QELM_Q.npz"


def select_runs(
    rows: Sequence[dict[str, Any]], **fixed: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the runs with the given ``h``, ``t``, ``final``; check they are unique.

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


def subtitle(values: dict[str, Any]) -> str:
    """Return the fixed parameters of a figure as one line."""
    final = ", last round after the last step" if values["final"] else ""
    return f"h = {values['h']:g} J, t = {values['t']:g} / J{final}"


def draw_ceilings(
    rows: Sequence[dict[str, Any]], values: dict[str, Any], ceilings: bool = True
) -> Figure:
    """Draw the ridge accuracy against ``p``, one curve per ``N``, and its ceilings.

    Args:
        rows: Runs of one ``(h, t, final)``, from :func:`select_runs`.
        values: Their ``h``, ``t`` and ``final``.
        ceilings: Draw the QDA and K-NN ceilings of each size, in its color.

    Returns:
        The open figure.
    """
    summary = summarize(rows, by=("N", "p"), metrics=METRICS)
    Ns = sizes(summary)
    colors = color_map(f"N{N}" for N in Ns)
    fig, ax = subplots(figsize=(7.5, 5.5))
    for N in Ns:
        color = colors[f"N{N}"]
        draw_curve(
            ax,
            sweep_curve(summary, N, "p", "accuracy"),
            color,
            label=f"N = {N}, ridge",
        )
        if not ceilings:
            continue
        for metric, name, style in CEILINGS:
            draw_companion(
                ax,
                summary,
                N,
                "p",
                metric,
                color,
                label=f"N = {N}, {name}",
                linestyle=style,
            )
    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="chance")
    ax.set(
        xlabel=r"measurement probability $p$",
        ylabel=r"accuracy on $Q$ from $\langle Z_k \rangle$",
        title=subtitle(values),
    )
    ax.legend(fontsize=9, ncol=2 if ceilings else 1)
    fig.suptitle("Ridge accuracy and Bayes ceilings")
    return fig


def fit_run(run_dir: Path) -> tuple[QDA, np.ndarray, np.ndarray]:
    """Refit the QDA of a run on its training samples.

    Args:
        run_dir: Folder of one run, holding :data:`NPZ`.

    Returns:
        The fitted QDA, the test ``<Z_k>`` (bias column dropped) and their
        charges.
    """
    with np.load(run_dir / NPZ) as data:
        X, Y, n_test = data["X"][:, :-1], data["Y"], len(data["y_test"])
    train = len(Y) - n_test
    qda = QDA(reg_param=REG_PARAM, tol=0.0).fit(X[:train], Y[:train])
    return qda, X[train:], Y[train:]


def draw_contours(
    sweep_dir: Path,
    rows: Sequence[dict[str, Any]],
    values: dict[str, Any],
    p_values: Sequence[float] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Draw the QDA contours on the PCA plane, one panel per ``(N, p)``.

    Args:
        sweep_dir: Folder of the sweep.
        rows: Runs of one ``(h, t, final)``, from :func:`select_runs`.
        values: Their ``h``, ``t`` and ``final``.
        p_values: Values of ``p`` to draw, default every value of the
            sweep. A point with several seeds draws the smallest seed.

    Returns:
        The open figure and, per panel ``"N<N>_p<p>"``, the plane and the
        projected Gaussians (:func:`~QELM.utils.plots.draw_qda_contours`)
        with the run drawn and its two ceilings.
    """
    Ns = sorted({int(r["N"]) for r in rows})
    ps = sorted({float(r["p"]) for r in rows} if p_values is None else p_values)
    fig, axes = subplots(
        len(Ns),
        len(ps),
        figsize=(4.2 * len(ps), 3.8 * len(Ns)),
        squeeze=False,
    )
    extra: dict[str, Any] = {}
    for i, N in enumerate(Ns):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            runs = sorted(
                (r for r in rows if int(r["N"]) == N and np.isclose(r["p"], p)),
                key=lambda r: r["seed"],
            )
            if not runs:
                ax.set_axis_off()
                continue
            run = runs[0]
            qda, X_test, y_test = fit_run(sweep_dir / run["run"])
            names = [f"Q = {q:g}" for q in qda.classes_]
            plane = draw_qda_contours(
                ax, X_test, y_test, qda, names=names, colors=COLORS[:2]
            )
            ax.set_title(
                f"N = {N}, p = {p:g}\nQDA {run['bayes_estimate_qda']:.2f}, "
                f"ridge {run['accuracy']:.2f}",
                fontsize=12,
            )
            extra[f"N{N}_p{p:g}"] = {
                "run": run["run"],
                "accuracy": run["accuracy"],
                "bayes_estimate_qda": run["bayes_estimate_qda"],
                "bayes_estimate_knn": run["bayes_estimate_knn"],
                **plane,
            }
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        rf"QDA densities on the PCA plane of $\langle Z_k \rangle$, "
        f"{subtitle(values)}"
    )
    return fig, extra


def plot_sweep(
    sweep_dir: Path,
    ceilings: bool = True,
    p_values: Sequence[float] | None = None,
    **fixed: Any,
) -> tuple[Path, Path]:
    """Write ``bayes_ceiling.png`` and ``qda_contours.png`` with their data.

    Args:
        sweep_dir: Folder of the sweep.
        ceilings: As in :func:`draw_ceilings`.
        p_values: As in :func:`draw_contours`.
        fixed: ``h``, ``t``, ``final`` to keep, as in :func:`select_runs`.

    Returns:
        The paths of the two PNG files.
    """
    rows, values = select_runs(load_sweep(sweep_dir, keys=KEYS), **fixed)
    fig = draw_ceilings(rows, values, ceilings)
    accuracy_png = save_figure(fig, sweep_dir, "bayes_ceiling", data=values)
    plt.close(fig)
    fig, extra = draw_contours(sweep_dir, rows, values, p_values)
    contours_png = save_figure(fig, sweep_dir, "qda_contours", data=extra)
    plt.close(fig)
    return accuracy_png, contours_png


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
    parser.add_argument("--h", type=float, default=None, help="TFIM.h to plot")
    parser.add_argument("--t", type=float, default=None, help="evolution.t to plot")
    parser.add_argument(
        "--final",
        choices=("true", "false"),
        default=None,
        help="final_measurement to plot",
    )
    parser.add_argument(
        "--p", type=float, nargs="+", default=None, help="p of the contour panels"
    )
    parser.add_argument(
        "--no-ceilings",
        action="store_true",
        help="draw the ridge accuracy only, without the QDA and K-NN ceilings",
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "exp_QELM_Q")
    final = None if args.final is None else args.final == "true"
    for png in plot_sweep(
        sweep_dir,
        ceilings=not args.no_ceilings,
        p_values=args.p,
        h=args.h,
        t=args.t,
        final=final,
    ):
        print(f"Figure: {png}")


if __name__ == "__main__":
    main()
