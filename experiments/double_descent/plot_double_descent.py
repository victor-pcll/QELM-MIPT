#!/usr/bin/env python3
"""Draw the double descent of a sweep against ``alpha = n_train / n_features``.

Reads the run folders written by ``double_descent.sh`` (one per seed), and
for each model and ratio takes, per seed, the median over the training sets
of the test and train MSE (the MSE has heavy tails near the peak, where a
single training set can give a huge ``||W||``) and the mean of the
classification errors; then averages over the seeds, with the standard
error over the seeds as error bars.

Four panels, ``alpha`` on a log axis and ``alpha = 1`` dashed: the test MSE
(log scale), the train MSE, the test classification error and the train
classification error. The figure is ``double_descent.png`` with its data
``double_descent.json`` (:func:`QELM.utils.plots.save_figure`), and the
accuracies and peak positions go to ``summary.json``, all in the sweep
folder.

    uv run python experiments/double_descent/plot_double_descent.py \
        outputs/double_descent/<date_time>
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

from QELM.utils.plots import color_map, save_figure, subplots

LABELS = {
    "qelm": "QELM",
    "qelm_t0": "QELM, t = 0",
    "elm": "classical ELM",
    "linear": "linear on angles",
}

# (statistic, how the repeats of one seed are reduced, panel title, log y).
PANELS: tuple[tuple[str, str, str, bool], ...] = (
    ("test_mse", "median", "Test MSE", True),
    ("train_mse", "median", "Train MSE", False),
    ("test_error", "mean", "Test error", False),
    ("train_error", "mean", "Train error", False),
)


def load_runs(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return one dict per run folder: its ``metrics.json`` and its arrays.

    A run folder is a direct subfolder holding ``double_descent.npz`` and
    ``metrics.json``. The arrays are under ``"arrays"``.

    Raises:
        FileNotFoundError: If ``sweep_dir`` holds no run.
    """
    runs = []
    for folder in sorted(Path(sweep_dir).iterdir()):
        npz, metrics = folder / "double_descent.npz", folder / "metrics.json"
        if not (npz.is_file() and metrics.is_file()):
            continue
        run = json.loads(metrics.read_text())
        with np.load(npz) as data:
            run["arrays"] = {key: data[key] for key in data.files}
        run["folder"] = folder.name
        runs.append(run)
    if not runs:
        raise FileNotFoundError(f"No double-descent run in {sweep_dir}.")
    return runs


def curve(
    runs: Sequence[dict[str, Any]], model: str, statistic: str, reduce: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(alpha, mean over seeds, standard error over seeds)``.

    Args:
        runs: Output of :func:`load_runs`, every run with the same ratios.
        model: Model name, such as ``"qelm"``.
        statistic: One of the sweep statistics, such as ``"test_mse"``.
        reduce: ``"median"`` or ``"mean"`` over the training sets of a seed.

    Raises:
        ValueError: If the runs do not share the same ratios.
    """
    alpha = runs[0]["arrays"][f"{model}_alpha"]
    per_seed = []
    for run in runs:
        if not np.allclose(run["arrays"][f"{model}_alpha"], alpha):
            raise ValueError(f"The runs do not share the ratios of {model}.")
        values = run["arrays"][f"{model}_{statistic}"]
        per_seed.append(
            np.median(values, axis=1) if reduce == "median" else values.mean(axis=1)
        )
    stack = np.array(per_seed)
    mean = stack.mean(axis=0)
    if len(runs) > 1:
        sem = stack.std(axis=0, ddof=1) / np.sqrt(len(runs))
    else:
        sem = np.full_like(mean, np.nan)
    return alpha, mean, sem


def summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracies, ranks and peak positions, averaged over the seeds."""
    models = list(runs[0]["models"])
    out: dict[str, Any] = {"n_runs": len(runs), "models": {}}
    for model in models:
        alpha, test_mse, _ = curve(runs, model, "test_mse", "median")
        acc = np.array([run[f"accuracy_{model}"] for run in runs])
        out["models"][model] = {
            "n_features": [run[f"n_features_{model}"] for run in runs],
            "n_columns": runs[0][f"n_columns_{model}"],
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(runs) > 1 else float("nan"),
            "peak_alpha_per_seed": [run[f"peak_alpha_{model}"] for run in runs],
            "peak_alpha_of_mean": float(alpha[int(np.argmax(test_mse))]),
        }
    if "accuracy_pixels" in runs[0]:
        out["accuracy_pixels_mean"] = float(
            np.mean([run["accuracy_pixels"] for run in runs])
        )
    out["t0_max_deviation"] = float(max(run["t0_max_deviation"] for run in runs))
    return out


def draw(runs: Sequence[dict[str, Any]]) -> tuple[Figure, Any]:
    """Draw the four panels; return ``(figure, axes)`` of shape ``(2, 2)``."""
    models = list(runs[0]["models"])
    colors = color_map(models)
    fig, axes = subplots(2, 2, figsize=(12, 9))
    for ax, (statistic, reduce, title, log) in zip(axes.ravel(), PANELS, strict=True):
        for model in models:
            alpha, mean, sem = curve(runs, model, statistic, reduce)
            ax.errorbar(
                alpha,
                mean,
                yerr=None if np.all(np.isnan(sem)) else sem,
                marker="o",
                markersize=4,
                capsize=2,
                color=colors[model],
                label=LABELS.get(model, model),
            )
        ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha = N_{train} / N_{features}$")
        ax.set_title(title)
    axes[0, 0].legend()
    fig.suptitle(f"Double descent, pseudo-inverse readout ({len(runs)} seeds)")
    return fig, axes


def plot_sweep(sweep_dir: str | Path) -> dict[str, Any]:
    """Draw and save the figure and ``summary.json``; return the summary."""
    sweep_dir = Path(sweep_dir)
    runs = load_runs(sweep_dir)
    fig, _ = draw(runs)
    table = summary(runs)
    save_figure(fig, sweep_dir, "double_descent", data={"summary": table})
    plt.close(fig)
    (sweep_dir / "summary.json").write_text(json.dumps(table, indent=2) + "\n")
    return table


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: ``plot_double_descent.py <sweep_dir>``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args(argv)
    table = plot_sweep(args.sweep_dir)
    for model, row in table["models"].items():
        print(
            f"{model}: accuracy {row['accuracy_mean']:.4f}, "
            f"peak at alpha = {row['peak_alpha_of_mean']:.3g}"
        )
    print(f"Saved double_descent.png and summary.json to {args.sweep_dir}")


if __name__ == "__main__":
    main()
