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

A second figure, ``confusion_matrices.png`` (with its ``.json``), shows for
each model the confusion matrix on the test set of the readout fitted on
the whole pool, summed over the seeds and normalized by row: entry
``[i, j]`` is the fraction of the test images of digit ``classes[i]``
predicted as ``classes[j]``. A strong diagonal is a good readout; the
off-diagonal entries name the digits it confuses.

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
    "pixels": "linear on pixels",
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

    The runs share the requested ratios (``double_descent.alpha_ratios``),
    but not always the actual ones: ``n_train = round(ratio * rank)`` and
    the rank can differ by one between seeds (a ReLU of the classical ELM
    that is active on every image of one pool adds no independent column).
    The points are matched by index and placed at the mean actual ratio;
    the ratio 1 is exactly 1 for every seed.

    Args:
        runs: Output of :func:`load_runs`, every run with the same ratios.
        model: Model name, such as ``"qelm"``.
        statistic: One of the sweep statistics, such as ``"test_mse"``.
        reduce: ``"median"`` or ``"mean"`` over the training sets of a seed.

    Raises:
        ValueError: If the runs do not scan the same number of ratios.
    """
    alphas = [run["arrays"][f"{model}_alpha"] for run in runs]
    if len({a.shape for a in alphas}) != 1:
        raise ValueError(f"The runs do not share the ratios of {model}.")
    alpha = np.mean(alphas, axis=0)
    per_seed = []
    for run in runs:
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
    out: dict[str, Any] = {
        "n_runs": len(runs),
        "classes": list(runs[0]["classes"]),
        "ridge_alpha": runs[0]["ridge_alpha"],
        "models": {},
    }
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
            "confusion": confusion(runs, model).tolist(),
        }
    if "accuracy_pixels" in runs[0]:
        out["accuracy_pixels_mean"] = float(
            np.mean([run["accuracy_pixels"] for run in runs])
        )
        out["confusion_pixels"] = confusion(runs, "pixels").tolist()
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
    classes = ", ".join(str(d) for d in runs[0]["classes"])
    alpha = runs[0]["ridge_alpha"]
    readout = "unregularized" if alpha == 0 else f"ridge alpha = {alpha:g}"
    fig.suptitle(
        f"Double descent, digits {classes}, {readout} readout ({len(runs)} seeds)"
    )
    return fig, axes


def confusion(runs: Sequence[dict[str, Any]], model: str) -> np.ndarray:
    """Return the confusion matrix of ``model``, summed over the seeds (counts)."""
    return np.sum([run["arrays"][f"confusion_{model}"] for run in runs], axis=0)


def draw_confusion(runs: Sequence[dict[str, Any]]) -> tuple[Figure, Any]:
    """Draw one row-normalized confusion matrix per model.

    Rows are the true digits, columns the predicted ones; each cell shows
    the fraction of the true digit's test images. The pixel baseline is
    the last panel when the runs have it.

    Returns:
        ``(figure, axes)``, one axis per panel.
    """
    models = list(runs[0]["models"])
    if "confusion_pixels" in runs[0]["arrays"]:
        models.append("pixels")
    digits = [str(d) for d in runs[0]["classes"]]
    fig, axes = subplots(1, len(models), figsize=(3.6 * len(models), 4.6))
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        counts = confusion(runs, model)
        fractions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.matshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(len(digits)):
            for j in range(len(digits)):
                ax.text(
                    j,
                    i,
                    f"{fractions[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if fractions[i, j] > 0.5 else "black",
                    fontsize=10,
                )
        ax.set_xticks(range(len(digits)), digits)
        ax.set_yticks(range(len(digits)), digits)
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlabel("predicted digit")
        ax.set_ylabel("true digit")
        ax.set_title(LABELS.get(model, model))
    fig.suptitle(f"Confusion matrices on the test set ({len(runs)} seeds)")
    return fig, axes


def plot_sweep(sweep_dir: str | Path) -> dict[str, Any]:
    """Draw and save the figure and ``summary.json``; return the summary."""
    sweep_dir = Path(sweep_dir)
    runs = load_runs(sweep_dir)
    fig, _ = draw(runs)
    table = summary(runs)
    save_figure(fig, sweep_dir, "double_descent", data={"summary": table})
    plt.close(fig)
    fig, _ = draw_confusion(runs)
    matrices = {key: value["confusion"] for key, value in table["models"].items()}
    if "confusion_pixels" in table:
        matrices["pixels"] = table["confusion_pixels"]
    save_figure(
        fig,
        sweep_dir,
        "confusion_matrices",
        data={"classes": table["classes"], "confusion": matrices},
    )
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
    print(
        "Saved double_descent.png, confusion_matrices.png and summary.json to "
        f"{args.sweep_dir}"
    )


if __name__ == "__main__":
    main()
