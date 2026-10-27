#!/usr/bin/env python3
"""Extract ``p_linear*``, the critical ``p`` of the ridge readout (#227).

Reads the finite-size sweep of notebook 06 (``N = 8, 10, 12``, ``h = J`` and
``2 J``, readout on ``[M, <Z_k>]``) through the two tables the notebook
saves with its figures, so no run is repeated:

- ``metrics_vs_p_N.json``: accuracy and R2 per ``(N, h, p)``;
- ``binder_vs_p_N.json``: Binder ratio ``B`` of the test predictions.

Three estimators of ``p_linear*`` (:mod:`QELM.evaluation.linear_critical`):
the minimum of the accuracy, the minimum of ``B`` and the maximum of
``1 - R2``. Two figures, each with its JSON
(:func:`QELM.utils.plots.save_figure`), the table of the estimates under
``"extra"``:

- ``p_linear_curves.png``: one row per ``h``, one column per ``N``; the
  three metrics on one ``p`` axis (they all lie in ``[-1, 1]``), a vertical
  line at each estimate and its interval shaded;
- ``p_linear_vs_N.png``: one panel per ``h``, the three estimates against
  ``N`` with their intervals. An estimate on the edge of the ``p`` range (no
  interior extremum) is drawn as an open marker.

    uv run python experiments/QELM_Q/plot_p_linear.py \
        outputs/QELM_Q_N_scaling/<date_time>/metrics_vs_p_N.json \
        outputs/QELM_Q_N_scaling/<date_time>/binder_vs_p_N.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.evaluation.linear_critical import (
    ESTIMATORS,
    estimator_curve,
    merge_summaries,
    p_linear_table,
)
from QELM.utils.plots import COLORS, save_figure, subplots

# (estimator, legend label, color).
LABELS: dict[str, tuple[str, str]] = {
    "accuracy": ("accuracy", COLORS[0]),
    "binder": ("Binder ratio B of y_hat", COLORS[2]),
    "residual": ("1 - R2", COLORS[1]),
}


def load_table(path: str | Path) -> list[dict[str, Any]]:
    """Return the summary table saved under ``"extra"`` of a figure JSON."""
    return list(json.loads(Path(path).read_text())["extra"]["table"])


def draw_curves(
    summary: Sequence[Mapping[str, Any]],
    estimates: Sequence[Mapping[str, Any]],
    name: str = "M_Z",
) -> tuple[Figure, np.ndarray]:
    """Draw the three metrics and the three estimates, per ``(h, N)``.

    Args:
        summary: Merged summary (:func:`merge_summaries`), one row per
            ``(N, h, p)``.
        estimates: Output of :func:`p_linear_table` on it.
        name: Feature set, the suffix of the metrics.

    Returns:
        The figure and its axes, shape ``(n_h, n_N)``.
    """
    Ns = sorted({int(r["N"]) for r in summary})
    hs = sorted({float(r["h"]) for r in summary})
    fig, axes = subplots(
        len(hs), len(Ns), figsize=(5 * len(Ns), 4.2 * len(hs)), squeeze=False
    )
    for h, row in zip(hs, axes, strict=True):
        for N, ax in zip(Ns, row, strict=True):
            rows = [r for r in summary if int(r["N"]) == N and float(r["h"]) == h]
            for estimator, score, _ in ESTIMATORS:
                metric = f"{score}_{name}"
                if not all(f"{metric}_mean" in r for r in rows):
                    continue
                label, color = LABELS[estimator]
                p, y, err = estimator_curve(rows, metric, estimator)
                ax.errorbar(
                    p,
                    y,
                    yerr=err,
                    color=color,
                    marker="o",
                    ms=4,
                    capsize=2,
                    label=label,
                )
                found = next(
                    (
                        e
                        for e in estimates
                        if e["N"] == N and e["h"] == h and e["estimator"] == estimator
                    ),
                    None,
                )
                if found is None:  # every value nan
                    continue
                ax.axvspan(found["p_low"], found["p_high"], color=color, alpha=0.12)
                ax.axvline(
                    found["p_refined"],
                    color=color,
                    linestyle=":" if found["at_edge"] else "--",
                    linewidth=1.5,
                )
            for level in (0.0, 2 / 3):  # B of one Gaussian, of two sharp peaks
                ax.axhline(level, color="gray", linestyle=":", linewidth=0.8)
            ax.set(
                xlabel="measurement probability p",
                ylim=(-1.05, 1.05),
                title=f"N = {N}, h = {h:g} J",
            )
        row[0].set_ylabel("metric of the ridge readout (test)")
    axes[0, 0].legend(fontsize="small", loc="lower left")
    fig.suptitle(
        "p_linear*: min of the accuracy, min of B, max of 1 - R2\n"
        "dashed: estimate, shaded: 68% interval, dotted: on the edge of p",
        fontsize="large",
    )
    return fig, axes


def draw_vs_N(estimates: Sequence[Mapping[str, Any]]) -> tuple[Figure, np.ndarray]:
    """Draw the three estimates against ``N``, one panel per ``h``.

    Returns:
        The figure and its axes, shape ``(n_h,)``.
    """
    hs = sorted({float(e["h"]) for e in estimates})
    fig, axes = subplots(1, len(hs), figsize=(5.5 * len(hs), 4.5), squeeze=False)
    offsets = {
        estimator: 0.15 * (k - 1) for k, (estimator, *_) in enumerate(ESTIMATORS)
    }
    for h, ax in zip(hs, axes[0], strict=True):
        for estimator, *_ in ESTIMATORS:
            rows = sorted(
                (e for e in estimates if e["h"] == h and e["estimator"] == estimator),
                key=lambda e: e["N"],
            )
            if not rows:
                continue
            label, color = LABELS[estimator]
            N = np.array([e["N"] for e in rows], dtype=float) + offsets[estimator]
            p = np.array([e["p_refined"] for e in rows])
            low = p - np.array([e["p_low"] for e in rows])
            high = np.array([e["p_high"] for e in rows]) - p
            ax.errorbar(
                N,
                p,
                yerr=[low, high],
                color=color,
                marker="o",
                capsize=3,
                label=label,
            )
            edge = np.array([e["at_edge"] for e in rows])
            if edge.any():
                ax.plot(
                    N[edge],
                    p[edge],
                    "o",
                    ms=10,
                    mfc="white",
                    mec=color,
                    zorder=3,
                )
        ax.set(
            xlabel="N (total qubits)",
            ylabel="p_linear*",
            ylim=(-0.02, 1.02),
            title=f"h = {h:g} J",
        )
        ax.set_xticks(sorted({int(e["N"]) for e in estimates}))
    axes[0, 0].legend(fontsize="small")
    fig.suptitle("p_linear* against N (open marker: extremum on the edge of p)")
    return fig, axes[0]


def print_table(estimates: Sequence[Mapping[str, Any]]) -> None:
    """Print one line per ``(N, h, estimator)``."""
    print(
        f"{'N':>3} {'h':>4} {'estimator':>9} {'p_grid':>6} {'p*':>6} "
        f"{'68% interval':>15} {'value':>7} {'prom.':>6} edge"
    )
    for e in estimates:
        print(
            f"{e['N']:>3} {e['h']:>4g} {e['estimator']:>9} {e['p_grid']:>6.2f} "
            f"{e['p_refined']:>6.3f} [{e['p_low']:.3f}, {e['p_high']:.3f}] "
            f"{e['value']:>7.3f} {e['prominence']:>6.1f} "
            f"{'yes' if e['at_edge'] else ''}"
        )


def plot_p_linear(
    metrics_json: str | Path,
    binder_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    name: str = "M_Z",
    n_boot: int = 2000,
) -> list[dict[str, Any]]:
    """Compute the estimates, save both figures and return the estimates.

    Args:
        metrics_json: ``metrics_vs_p_N.json`` of notebook 06.
        binder_json: ``binder_vs_p_N.json``; without it the Binder estimator
            is left out.
        out_dir: Folder of the figures; that of ``metrics_json`` by default.
        name: Feature set, the suffix of the metrics.
        n_boot: Number of redrawn curves of the bootstrap.
    """
    tables = [load_table(metrics_json)]
    if binder_json is not None:
        tables.append(load_table(binder_json))
    summary = merge_summaries(*tables)
    estimates = p_linear_table(summary, name=name, n_boot=n_boot)
    out = Path(metrics_json).parent if out_dir is None else Path(out_dir)
    fig, _ = draw_curves(summary, estimates, name)
    save_figure(fig, out, "p_linear_curves", data={"estimates": estimates})
    plt.close(fig)
    fig, _ = draw_vs_N(estimates)
    save_figure(fig, out, "p_linear_vs_N", data={"estimates": estimates})
    plt.close(fig)
    return estimates


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("binder_json", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--name", default="M_Z")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    estimates = plot_p_linear(
        args.metrics_json, args.binder_json, args.out, args.name, args.n_boot
    )
    print_table(estimates)


if __name__ == "__main__":
    main()
