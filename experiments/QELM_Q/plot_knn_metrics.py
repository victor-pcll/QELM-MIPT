#!/usr/bin/env python3
"""Compare the ridge readout with the Euclidean and Mahalanobis K-NN (#213).

Reads a sweep of ``exp_QELM_Q.sh`` (one ``N``, any ``h``, ``p``, seeds and
``final_measurement``) and averages over the seeds, with the standard error
as error bars, three accuracies on the same ``<Z_k>`` and split:

- ``accuracy``: the ridge readout (solid line);
- ``bayes_estimate_knn_euclidean``: the Euclidean K-NN (dash-dotted);
- ``bayes_estimate_knn_mahalanobis``: the Mahalanobis K-NN (dotted).

The curves use the runs without a measurement after the last step
(``final_measurement=false``). The runs with it, where the measured qubits
are read right after their collapse (``<Z_k> = +-1`` for every qubit at
``p = 1``), are drawn as open markers at their ``p``, in the colors of the
three models.

Three figures, each with its JSON (:func:`QELM.utils.plots.save_figure`), in
the sweep folder unless ``--out`` is given:

- ``knn_metrics_vs_p.png``: one panel per ``h``, the accuracies against ``p``;
- ``knn_metrics_vs_h.png``: one panel per ``p``, the accuracies against ``h``
  (log axis);
- ``knn_covariance_cond.png``: the condition number of the covariance of the
  training ``<Z_k>`` (``knn_covariance_cond``, log axis) against ``p``, one
  curve per ``h``, open markers for ``final_measurement=true``. A large value
  means that the Mahalanobis metric amplifies directions of tiny spread.

The table of the means is stored under ``"extra"`` of
``knn_metrics_vs_p.json``.

    uv run python experiments/QELM_Q/plot_knn_metrics.py \
        outputs/exp_QELM_Q/<date_time>
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from QELM.utils.plots import color_map, save_figure, subplots
from QELM.utils.sweep import load_sweep, summarize

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("h", "TFIM.h"),
    ("final", "projective_measurement.final_measurement"),
)
# (metrics.json key, legend label, line style).
MODELS: tuple[tuple[str, str, str], ...] = (
    ("accuracy", "ridge", "-"),
    ("bayes_estimate_knn_euclidean", "K-NN, Euclidean", "-."),
    ("bayes_estimate_knn_mahalanobis", "K-NN, Mahalanobis", ":"),
)
COND = "knn_covariance_cond"


def load_summary(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return one row per ``(N, h, p, final)``: means over the seeds.

    Raises:
        ValueError: If the sweep holds several ``N``.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    for row in rows:
        row["final"] = bool(row["final"])
    sizes = {row["N"] for row in rows}
    if len(sizes) != 1:
        raise ValueError(f"The sweep must hold one N, got {sorted(sizes)}.")
    metrics = [key for key, _, _ in MODELS] + [COND]
    return summarize(rows, by=("N", "h", "final", "p"), metrics=metrics)


def _select(
    summary: Sequence[dict[str, Any]], final: bool, **fixed: float
) -> list[dict[str, Any]]:
    return [
        row
        for row in summary
        if row["final"] == final and all(row[k] == v for k, v in fixed.items())
    ]


def _draw(
    ax: Axes, summary: Sequence[dict[str, Any]], axis: str, fixed: dict[str, float]
) -> None:
    """Draw the three models against ``axis`` at ``fixed``, both kinds of run."""
    colors = color_map([key for key, _, _ in MODELS])
    for key, label, style in MODELS:
        rows = sorted(_select(summary, False, **fixed), key=lambda r: r[axis])
        if rows:
            ax.errorbar(
                [r[axis] for r in rows],
                [r[f"{key}_mean"] for r in rows],
                yerr=[r[f"{key}_sem"] for r in rows],
                color=colors[key],
                linestyle=style,
                marker="o",
                markersize=4,
                capsize=2,
                label=label,
            )
        finals = sorted(_select(summary, True, **fixed), key=lambda r: r[axis])
        if finals:
            ax.errorbar(
                [r[axis] for r in finals],
                [r[f"{key}_mean"] for r in finals],
                yerr=[r[f"{key}_sem"] for r in finals],
                color=colors[key],
                linestyle="none",
                marker="s",
                markerfacecolor="none",
                markersize=7,
                capsize=2,
                label=f"{label}, final round",
            )
    ax.axhline(0.5, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_ylim(0.4, 1.02)


def _panels(n: int) -> tuple[Figure, np.ndarray]:
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_vs_p(summary: Sequence[dict[str, Any]]) -> Figure:
    """One panel per ``h``: the three accuracies against ``p``."""
    hs = sorted({row["h"] for row in summary})
    fig, axes = _panels(len(hs))
    N = summary[0]["N"]
    for ax, h in zip(axes.flat, hs, strict=False):
        _draw(ax, summary, "p", {"h": h})
        ax.set_title(f"N = {N}, h = {h:g} J")
        ax.set_xlabel("p")
        ax.set_ylabel("test accuracy on Q")
    axes.flat[0].legend(fontsize=7)
    return fig


def plot_vs_h(summary: Sequence[dict[str, Any]]) -> Figure:
    """One panel per ``p``: the three accuracies against ``h``."""
    ps = sorted({row["p"] for row in summary})
    fig, axes = _panels(len(ps))
    N = summary[0]["N"]
    for ax, p in zip(axes.flat, ps, strict=False):
        _draw(ax, summary, "h", {"p": p})
        ax.set_xscale("log")
        ax.set_title(f"N = {N}, p = {p:g}")
        ax.set_xlabel("h / J")
        ax.set_ylabel("test accuracy on Q")
    axes.flat[0].legend(fontsize=7)
    return fig


def plot_cond(summary: Sequence[dict[str, Any]]) -> Figure:
    """The condition number of the covariance against ``p``, one curve per h."""
    hs = sorted({row["h"] for row in summary})
    colors = color_map([f"{h:g}" for h in hs])
    fig, ax = subplots(figsize=(5.0, 3.6))
    for h in hs:
        color = colors[f"{h:g}"]
        rows = sorted(_select(summary, False, h=h), key=lambda r: r["p"])
        ax.plot(
            [r["p"] for r in rows],
            [r[f"{COND}_mean"] for r in rows],
            color=color,
            marker="o",
            markersize=4,
            label=f"h = {h:g} J",
        )
        finals = _select(summary, True, h=h)
        ax.plot(
            [r["p"] for r in finals],
            [r[f"{COND}_mean"] for r in finals],
            color=color,
            linestyle="none",
            marker="s",
            markerfacecolor="none",
            markersize=7,
            label=f"h = {h:g} J, final round",
        )
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("cond(Sigma) of the training <Z_k>")
    ax.set_title(f"N = {summary[0]['N']} (open squares: final round)")
    ax.legend(fontsize=7)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path, help="Folder of one sweep.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else args.sweep_dir
    summary = load_summary(args.sweep_dir)
    for name, draw in (
        ("knn_metrics_vs_p", plot_vs_p),
        ("knn_metrics_vs_h", plot_vs_h),
        ("knn_covariance_cond", plot_cond),
    ):
        fig = draw(summary)
        extra = {"summary": summary} if name == "knn_metrics_vs_p" else None
        print(save_figure(fig, out, name, data=extra))
        plt.close(fig)


if __name__ == "__main__":
    main()
