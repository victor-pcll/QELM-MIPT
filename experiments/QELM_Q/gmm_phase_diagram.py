#!/usr/bin/env python3
"""The Gaussian-mixture ceiling of one size over the (p, h) plane (issue #215).

Reads an ``exp_QELM_Q.sh`` sweep of one size ``N`` (``N = 8`` for the
issue) over the measurement probability ``p`` and the transverse field
``h``, with several seeds, and checks whether the Bayes ceiling of a mixture
of Gaussians (#208) gives the expected physics everywhere on the plane.

For every run, the models are refitted from ``exp_QELM_Q.npz`` on the split
of the run, with the seed of the run
(:func:`QELM.evaluation.gmm_scan.gmm_component_scan`): the mixture rule for
``K = 1, 2, 3, 4, 6, 8`` Gaussians per sector, the mixture whose ``K`` is
chosen in each sector by BIC, and the K-NN. ``K = 3`` and the K-NN are the
models behind the logged ``bayes_estimate_gmm`` and ``bayes_estimate_knn``;
the script checks that it finds the same numbers, and stops if not. The
seeds are then averaged, with their standard errors.

Figures, each with its JSON (:func:`QELM.utils.plots.save_figure`), in the
sweep folder:

- ``gmm_heatmaps_p_h.png``: maps over ``(p, h)`` of the accuracy of the
  ridge readout, the QDA, the BIC mixture and the K-NN, of the gap
  ``GMM - ridge`` (large where the charge is in ``<Z_k>`` but not linearly:
  the learnability wall) and of ``GMM - K-NN`` (should be about 0 within
  the noise: two estimators of the same Bayes accuracy that share no
  model).
- ``gmm_components_p_h.png``: the number of Gaussians that BIC selects in
  each sector, averaged over the seeds: how many bundles of trajectories
  the data of each sector support.
- ``gmm_vs_K.png``: one panel per ``h``, the mixture accuracy against
  ``K``, one curve per ``p``. A ceiling must have saturated in ``K``.
- ``gmm_vs_p.png``: one panel per ``h``, the ridge, QDA, ``K = 3``, BIC
  and K-NN accuracies against ``p``.
- ``gmm_phase_diagram.json``: the table averaged over the seeds, one row
  per ``(p, h)``.

The sweep of the issue (``N = 8``, 5 fields, 7 values of ``p``, 3 seeds,
``L = 500``, about 2 h on 4 cores with one run per core):

    N_VALUES=8 H_VALUES="0.5 1 2 5 10" P_VALUES="0 0.1 0.2 0.3 0.5 0.7 1" \\
        SEEDS="42 43 44" L=500 ./experiments/QELM_Q/exp_QELM_Q.sh
    uv run python experiments/QELM_Q/gmm_phase_diagram.py \\
        outputs/exp_QELM_Q/<date_time>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.evaluation.gmm_scan import SCAN_COMPONENTS, gmm_component_scan
from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    color_map,
    draw_curve,
    draw_heatmap,
    save_figure,
    subplots,
    sweep_curve,
    sweep_grid,
)

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
NPZ = "exp_QELM_Q.npz"
# The refitted models must give the logged ceilings: refitted -> logged.
CHECKS = {
    "accuracy_gmm_K3": "bayes_estimate_gmm",
    "accuracy_knn": "bayes_estimate_knn",
}
GMM_K = tuple(f"accuracy_gmm_K{K}" for K in SCAN_COMPONENTS)
METRICS = (
    "accuracy",
    "bayes_estimate_qda",
    *GMM_K,
    "accuracy_gmm_bic",
    "accuracy_knn",
    "k_bic_0",
    "k_bic_1",
    "k_bic_mean",
    "gap_gmm_ridge",
    "gap_gmm_knn",
)
# (metric, title, color map, (vmin, vmax)) of the accuracy maps.
MAPS = (
    ("accuracy", "ridge", "viridis", (0.5, 1.0)),
    ("bayes_estimate_qda", "QDA", "viridis", (0.5, 1.0)),
    ("accuracy_gmm_bic", "GMM (BIC)", "viridis", (0.5, 1.0)),
    ("accuracy_knn", "K-NN", "viridis", (0.5, 1.0)),
    ("gap_gmm_ridge", "GMM (BIC) - ridge", "magma", (0.0, 0.5)),
    ("gap_gmm_knn", "GMM (BIC) - K-NN", "RdBu_r", (-0.2, 0.2)),
)
# (metric, legend, line style) of the curves against p.
CURVES = (
    ("accuracy", "ridge", "-"),
    ("bayes_estimate_qda", "QDA", "--"),
    ("accuracy_gmm_K3", "GMM, K = 3", ":"),
    ("accuracy_gmm_bic", "GMM, BIC", "-"),
    ("accuracy_knn", "K-NN", "-."),
)


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


def scan_run(run_dir: Path, row: dict[str, Any]) -> dict[str, float]:
    """Refit the models of one run on its split and score them.

    Args:
        run_dir: Folder of the run, holding :data:`NPZ`.
        row: Its row of :func:`QELM.utils.load_sweep`, with the seed and the
            logged metrics.

    Returns:
        The output of :func:`~QELM.evaluation.gmm_scan.gmm_component_scan`,
        with ``gap_gmm_ridge`` and ``gap_gmm_knn``, the BIC mixture minus the
        ridge readout and minus the K-NN.

    Raises:
        RuntimeError: If a refitted model of :data:`CHECKS` does not give the
            accuracy logged by the run.
    """
    with np.load(run_dir / NPZ) as data:
        X, Y, n_test = data["X"][:, :-1], data["Y"], len(data["y_test"])
    train = len(Y) - n_test
    scan = gmm_component_scan(
        X[:train], Y[:train], X[train:], Y[train:], seed=int(row["seed"])
    )
    for refitted, logged in CHECKS.items():
        same = np.isclose(scan[refitted], float(row[logged]), rtol=0.0, atol=1e-12)
        if not same and not (np.isnan(scan[refitted]) and np.isnan(row[logged])):
            raise RuntimeError(
                f"{run_dir.name}: refitted {refitted} = {scan[refitted]} but the "
                f"run logged {logged} = {row[logged]}."
            )
    scan["gap_gmm_ridge"] = scan["accuracy_gmm_bic"] - float(row["accuracy"])
    scan["gap_gmm_knn"] = scan["accuracy_gmm_bic"] - scan["accuracy_knn"]
    return scan


def subtitle(values: dict[str, Any]) -> str:
    """Return the fixed parameters of a figure as one line."""
    final = ", last round after the last step" if values["final"] else ""
    return f"N = {values['N']}, t = {values['t']:g} / J{final}"


def draw_maps(
    summary: Sequence[dict[str, Any]],
    values: dict[str, Any],
    maps: Sequence[tuple[str, str, str, tuple[float, float]]] = MAPS,
    ncols: int = 3,
) -> Figure:
    """Draw one heatmap over ``(p, h)`` per entry of ``maps``.

    Args:
        summary: Output of :func:`QELM.utils.summarize` by ``(N, p, h)``.
        values: The fixed ``N``, ``t`` and ``final``.
        maps: ``(metric, title, color map, (vmin, vmax))`` of each panel.
        ncols: Panels per row.

    Returns:
        The open figure. ``h`` is on a logarithmic axis when it spans more
        than a factor 4.
    """
    hs = [float(r["h"]) for r in summary]
    log_h = min(hs) > 0 and max(hs) / min(hs) > 4
    nrows = -(-len(maps) // ncols)
    fig, axes = subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False
    )
    for ax, (metric, title, cmap, (vmin, vmax)) in zip(axes.flat, maps, strict=False):
        grid = sweep_grid(summary, "p", "h", metric)
        draw_heatmap(
            ax, grid, label=metric, log_y=log_h, vmin=vmin, vmax=vmax, cmap=cmap
        )
        for j, h in enumerate(grid.y):
            for i, p in enumerate(grid.x):
                if np.isfinite(grid.mean[j, i]):
                    ax.text(
                        p,
                        h,
                        f"{grid.mean[j, i]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if cmap == "magma" else "black",
                    )
        ax.set(xlabel=r"$p$", ylabel=r"$h / J$", title=title)
    for ax in axes.flat[len(maps) :]:
        ax.set_axis_off()
    fig.suptitle(rf"Accuracy on $Q$ from $\langle Z_k \rangle$, {subtitle(values)}")
    return fig


def _rows_of(summary: Sequence[dict[str, Any]], h: float) -> list[dict[str, Any]]:
    return [r for r in summary if np.isclose(float(r["h"]), h)]


def draw_vs_p(summary: Sequence[dict[str, Any]], values: dict[str, Any]) -> Figure:
    """Draw the accuracies of :data:`CURVES` against ``p``, one panel per ``h``."""
    hs = sorted({float(r["h"]) for r in summary})
    colors = color_map(name for _, name, _ in CURVES)
    fig, axes = subplots(1, len(hs), figsize=(4.4 * len(hs), 4.0), squeeze=False)
    N = int(values["N"])
    for ax, h in zip(axes[0], hs, strict=True):
        rows = _rows_of(summary, h)
        for metric, name, style in CURVES:
            curve = sweep_curve(rows, N, "p", metric)
            draw_curve(ax, curve, colors[name], label=name, linestyle=style)
        ax.axhline(0.5, color="gray", ls=":", lw=1.2)
        ax.set(xlabel=r"$p$", title=f"h = {h:g} J", ylim=(0.4, 1.02))
    axes[0, 0].set_ylabel(r"accuracy on $Q$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Ridge readout and Bayes ceilings, {subtitle(values)}")
    return fig


def draw_vs_K(summary: Sequence[dict[str, Any]], values: dict[str, Any]) -> Figure:
    """Draw the mixture accuracy against ``K``: one panel per ``h``, curve per ``p``.

    The BIC mixture of each ``p`` is drawn at its mean selected ``K``, as an
    open marker.
    """
    hs = sorted({float(r["h"]) for r in summary})
    ps = sorted({float(r["p"]) for r in summary})
    colors = color_map(f"p{p:g}" for p in ps)
    fig, axes = subplots(1, len(hs), figsize=(4.4 * len(hs), 4.0), squeeze=False)
    Ks = np.array(SCAN_COMPONENTS, dtype=float)
    for ax, h in zip(axes[0], hs, strict=True):
        for p in ps:
            (row,) = [r for r in _rows_of(summary, h) if np.isclose(float(r["p"]), p)]
            mean = np.array([row[f"{m}_mean"] for m in GMM_K])
            err = np.nan_to_num(np.array([row[f"{m}_sem"] for m in GMM_K]))
            color = colors[f"p{p:g}"]
            ax.errorbar(
                Ks,
                mean,
                yerr=err,
                color=color,
                marker="o",
                capsize=3,
                label=f"p = {p:g}",
            )
            ax.plot(
                row["k_bic_mean_mean"],
                row["accuracy_gmm_bic_mean"],
                marker="D",
                mfc="none",
                color=color,
                ls="none",
            )
        ax.axhline(0.5, color="gray", ls=":", lw=1.2)
        ax.set(
            xlabel=r"Gaussians per sector $K$",
            title=f"h = {h:g} J",
            xticks=Ks,
            ylim=(0.4, 1.02),
        )
    axes[0, 0].set_ylabel(r"mixture accuracy on $Q$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Saturation of the mixture ceiling in K (diamonds: BIC), {subtitle(values)}"
    )
    return fig


def analyze_sweep(sweep_dir: Path, **fixed: Any) -> dict[str, Path]:
    """Refit every run, average over the seeds and write the figures and the table.

    Args:
        sweep_dir: Folder of the sweep.
        fixed: ``N``, ``t``, ``final`` to keep, as in :func:`select_runs`.

    Returns:
        The path of each file written: ``"heatmaps"``, ``"components"``,
        ``"vs_K"``, ``"vs_p"`` (PNG, each with its JSON) and ``"table"``.
    """
    rows, values = select_runs(load_sweep(sweep_dir, keys=KEYS), **fixed)
    for row in rows:
        row.update(scan_run(sweep_dir / row["run"], row))
    summary = summarize(rows, by=("N", "p", "h"), metrics=METRICS)
    out: dict[str, Path] = {}
    figures = {
        "heatmaps": ("gmm_heatmaps_p_h", draw_maps(summary, values)),
        "components": (
            "gmm_components_p_h",
            draw_maps(
                summary,
                values,
                maps=(
                    ("k_bic_0", "BIC components, lower charge", "cividis", (1, 8)),
                    ("k_bic_1", "BIC components, higher charge", "cividis", (1, 8)),
                ),
                ncols=2,
            ),
        ),
        "vs_K": ("gmm_vs_K", draw_vs_K(summary, values)),
        "vs_p": ("gmm_vs_p", draw_vs_p(summary, values)),
    }
    for key, (name, fig) in figures.items():
        out[key] = save_figure(fig, sweep_dir, name, data=values)
        plt.close(fig)
    table = sweep_dir / "gmm_phase_diagram.json"
    table.write_text(
        json.dumps({"fixed": values, "summary": summary}, indent=2, default=str) + "\n"
    )
    out["table"] = table
    return out


def main() -> None:
    """Parse the command line and analyze one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/exp_QELM_Q",
    )
    parser.add_argument("--N", type=int, default=None, help="size N to analyze")
    parser.add_argument("--t", type=float, default=None, help="evolution.t to keep")
    parser.add_argument(
        "--final",
        choices=("true", "false"),
        default=None,
        help="final_measurement to keep",
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(Path("outputs") / "exp_QELM_Q")
    final = None if args.final is None else args.final == "true"
    for path in analyze_sweep(sweep_dir, N=args.N, t=args.t, final=final).values():
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
