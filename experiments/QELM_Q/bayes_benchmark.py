#!/usr/bin/env python3
"""Benchmark the QDA against GMM, SVM and K-NN on a p-sweep (issue #200).

The QDA of issue #202 models each charge sector by one Gaussian in the
space of the final ``<Z_k>``. If the measured trajectories make the sectors
multimodal, it can miss a separation that a richer estimator sees, and its
accuracy is then not the Bayes ceiling. This script checks it on a sweep of
``exp_QELM_Q.sh``: for every run folder, it reads the ``<Z_k>`` and the
charges back from ``exp_QELM_Q.npz``, keeps the split of the run (the first
``train_size`` samples to train, the rest to test) and fits on it, with the
same seed, every estimator of
:func:`QELM.evaluation.bayes_optimal.benchmark_bayes_estimators`:

- ``qda``: one Gaussian per sector (issue #202);
- ``gmm<K>``: a mixture of ``K`` Gaussians per sector, the Bayes rule with
  the log-likelihood of each mixture;
- ``svm``: an SVM with an RBF kernel, a non-linear boundary without a
  density model;
- ``knn``: ``sqrt(train_size)`` nearest neighbors weighted by distance.

It writes in the sweep folder:

- ``bayes_benchmark.json``: one row per run, with the parameters of the
  run, its ridge ``accuracy`` and ``accuracy_<name>``, ``time_<name>`` (fit
  and prediction, in seconds) of every estimator;
- ``accuracy_vs_p_bayes_benchmark.pdf`` (and ``.png`` with its data
  ``.json``, :func:`QELM.utils.plots.save_figure`): one panel per
  ``(h, N)``, the accuracy of every estimator against ``p``, averaged over
  the seeds, with the ridge readout and chance;
- ``time_bayes_benchmark.png`` (and ``.json``): the mean time of every
  estimator against ``N``, on a log scale.

The sweep must hold one value of ``evolution.t`` and of
``final_measurement`` (``--t``, ``--final`` pick one otherwise):

    uv run python experiments/QELM_Q/bayes_benchmark.py \\
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

from QELM.evaluation.bayes_optimal import benchmark_bayes_estimators
from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    COLORS,
    draw_curve,
    save_figure,
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
GMM_COMPONENTS = (3, 5)
# (name, legend, color, line style) of every estimator, the ridge readout
# of exp_QELM_Q first.
ESTIMATORS = (
    ("ridge", "ridge (linear)", "gray", "-"),
    ("qda", "QDA (1 Gaussian)", COLORS[0], "-"),
    *(
        (f"gmm{K}", f"GMM ({K} Gaussians)", COLORS[1 + i], "--")
        for i, K in enumerate(GMM_COMPONENTS)
    ),
    ("svm", "SVM (RBF)", COLORS[3], "-."),
    ("knn", "K-NN", COLORS[4], ":"),
)
NPZ = "exp_QELM_Q.npz"


def benchmark_run(run_dir: Path, seed: int) -> dict[str, float]:
    """Run :func:`benchmark_bayes_estimators` on the split of one run.

    Args:
        run_dir: Folder of one run, holding :data:`NPZ`.
        seed: Seed of the run, given to the mixtures.

    Returns:
        ``accuracy_<name>`` and ``time_<name>`` of every estimator.
    """
    with np.load(run_dir / NPZ) as data:
        # The last column of X is the bias of the ridge readout.
        X, Y, n_test = data["X"][:, :-1], data["Y"], len(data["y_test"])
    train = len(Y) - n_test
    return benchmark_bayes_estimators(
        X[:train],
        Y[:train],
        X[train:],
        Y[train:],
        gmm_components=GMM_COMPONENTS,
        seed=seed,
    )


def benchmark_sweep(
    sweep_dir: Path, t: float | None = None, final: bool | None = None
) -> list[dict[str, Any]]:
    """Benchmark every run of a sweep of one ``t`` and ``final``.

    Args:
        sweep_dir: Folder of the sweep.
        t: ``evolution.t`` to keep, ``None`` for every value.
        final: ``final_measurement`` to keep, ``None`` for every value.

    Returns:
        One row per run: ``run``, :data:`KEYS`, the ridge ``accuracy`` and
        the output of :func:`benchmark_run`, with ``accuracy_ridge`` a copy
        of ``accuracy``.

    Raises:
        ValueError: If no run is left, or ``t`` or ``final`` still takes
            several values.
    """
    rows = [
        r
        for r in load_sweep(sweep_dir, keys=KEYS)
        if (t is None or r["t"] == t) and (final is None or r["final"] == final)
    ]
    if not rows:
        raise ValueError(f"No run with t = {t}, final = {final}.")
    for key in ("t", "final"):
        found = sorted({r[key] for r in rows}, key=str)
        if len(found) > 1:
            raise ValueError(f"The sweep has several {key} = {found}: pick one.")
    out = []
    for r in rows:
        row = {key: r[key] for key in ("run", *(k for k, _ in KEYS))}
        row["accuracy_ridge"] = r["accuracy"]
        row.update(benchmark_run(sweep_dir / r["run"], int(r["seed"])))
        out.append(row)
    return out


def draw_accuracy(rows: Sequence[dict[str, Any]]) -> Figure:
    """Draw the accuracy of every estimator against ``p``, a panel per ``(h, N)``.

    Args:
        rows: Output of :func:`benchmark_sweep`.

    Returns:
        The open figure, rows of panels for ``h``, columns for ``N``.
    """
    hs = sorted({float(r["h"]) for r in rows})
    Ns = sorted({int(r["N"]) for r in rows})
    metrics = [f"accuracy_{name}" for name, *_ in ESTIMATORS]
    fig, axes = subplots(
        len(hs),
        len(Ns),
        figsize=(5.0 * len(Ns), 4.2 * len(hs)),
        squeeze=False,
        sharey=True,
    )
    for i, h in enumerate(hs):
        summary = summarize(
            [r for r in rows if float(r["h"]) == h], by=("N", "p"), metrics=metrics
        )
        for j, N in enumerate(Ns):
            ax = axes[i, j]
            for name, label, color, style in ESTIMATORS:
                curve = sweep_curve(summary, N, "p", f"accuracy_{name}")
                if curve.x.size:
                    draw_curve(ax, curve, color, label=label, linestyle=style)
            ax.axhline(0.5, color="black", ls=":", lw=1.0, label="chance")
            ax.set_title(f"N = {N}, h = {h:g} J")
            ax.set_xlabel(r"measurement probability $p$")
            if j == 0:
                ax.set_ylabel(r"accuracy on $Q$ from $\langle Z_k \rangle$")
    axes[0, 0].legend(fontsize=8)
    t, final = rows[0]["t"], rows[0]["final"]
    last = ", last round after the last step" if final else ""
    fig.suptitle(f"Estimators of the Bayes accuracy, t = {t:g} / J{last}")
    return fig


def draw_time(rows: Sequence[dict[str, Any]]) -> Figure:
    """Draw the mean fit and prediction time of every estimator against ``N``.

    Args:
        rows: Output of :func:`benchmark_sweep`.

    Returns:
        The open figure, log scale, averaged over ``h``, ``p`` and seeds.
    """
    names = [(n, label, c, s) for n, label, c, s in ESTIMATORS if n != "ridge"]
    fig, ax = subplots(figsize=(6.5, 4.5))
    Ns = sorted({int(r["N"]) for r in rows})
    for name, label, color, style in names:
        times = [
            np.nanmean([r[f"time_{name}"] for r in rows if int(r["N"]) == N])
            for N in Ns
        ]
        ax.plot(Ns, times, marker="o", color=color, ls=style, label=label)
    ax.set_yscale("log")
    ax.set(
        xlabel=r"$N$",
        ylabel="fit + prediction time (s)",
        xticks=Ns,
        title="Cost of the estimators, mean over h, p and seeds",
    )
    ax.legend(fontsize=8)
    return fig


def mean_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average accuracy and time of every estimator per ``(N, h)``."""
    table = []
    for N, h in sorted({(int(r["N"]), float(r["h"])) for r in rows}):
        sel = [r for r in rows if int(r["N"]) == N and float(r["h"]) == h]
        entry: dict[str, Any] = {"N": N, "h": h, "n_runs": len(sel)}
        for name, *_ in ESTIMATORS:
            entry[f"accuracy_{name}"] = float(
                np.nanmean([r[f"accuracy_{name}"] for r in sel])
            )
            if name != "ridge":
                entry[f"time_{name}"] = float(
                    np.nanmean([r[f"time_{name}"] for r in sel])
                )
        table.append(entry)
    return table


def benchmark_and_plot(
    sweep_dir: Path, t: float | None = None, final: bool | None = None
) -> tuple[Path, Path, Path]:
    """Write the table and the two figures of :mod:`bayes_benchmark`.

    Args:
        sweep_dir: Folder of the sweep.
        t: As in :func:`benchmark_sweep`.
        final: As in :func:`benchmark_sweep`.

    Returns:
        The paths of ``bayes_benchmark.json``, of the accuracy figure (PDF)
        and of the time figure (PNG).
    """
    rows = benchmark_sweep(sweep_dir, t, final)
    table = sweep_dir / "bayes_benchmark.json"
    table.write_text(
        json.dumps({"runs": rows, "mean": mean_table(rows)}, indent=2) + "\n"
    )
    fig = draw_accuracy(rows)
    save_figure(fig, sweep_dir, "accuracy_vs_p_bayes_benchmark")
    pdf = sweep_dir / "accuracy_vs_p_bayes_benchmark.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    fig = draw_time(rows)
    time_png = save_figure(fig, sweep_dir, "time_bayes_benchmark")
    plt.close(fig)
    return table, pdf, time_png


def main() -> None:
    """Parse the command line and benchmark one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/exp_QELM_Q",
    )
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
    table, pdf, time_png = benchmark_and_plot(sweep_dir, args.t, final)
    for entry in json.loads(table.read_text())["mean"]:
        accuracies = ", ".join(
            f"{name} {entry[f'accuracy_{name}']:.3f}" for name, *_ in ESTIMATORS
        )
        print(f"N = {entry['N']}, h = {entry['h']:g}: {accuracies}")
    for path in (table, pdf, time_png):
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
