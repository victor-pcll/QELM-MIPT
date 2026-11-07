"""Read a sweep of fourier_expressivity.py and draw its figures (issue #309).

    uv run python experiments/fourier_expressivity/plot_fourier_expressivity.py \
        outputs/fourier_expressivity/<date_time>

Each figure is saved in the sweep folder with the data it shows
(:func:`QELM.utils.plots.save_figure`).
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.utils.plots import color_map, save_figure, subplots

MODEL_ORDER = ["haar", "u1", "z2", "clifford"]
MODEL_LABELS = {
    "haar": "Haar U(4)",
    "u1": "U(1)",
    "z2": "Z2 parity",
    "clifford": "Clifford",
}
SCALARS = [
    "rank",
    "rank_fraction",
    "final_rank",
    "mean_degree",
    "final_mean_degree",
    "odd_x_weight",
    "d_eff",
    "feature_variance",
]


def load_runs(sweep: Path) -> list[dict[str, Any]]:
    """Return the ``metrics.json`` of every run of ``sweep``, sorted."""
    runs = [json.loads(f.read_text()) for f in sorted(sweep.glob("*/metrics.json"))]
    if not runs:
        raise FileNotFoundError(f"No */metrics.json in {sweep}.")
    return runs


def summarize(runs: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """Average the measures over the seeds, per model and size.

    Returns:
        ``summary[model][N]`` with ``p`` (the grid), ``n_seeds``, ``T``,
        ``n_features``, and for each scalar measure its mean and standard
        error over the seeds (``<name>_mean``, ``<name>_sem``, arrays over
        ``p``); the profiles ``final_degree_profile`` (``p x (N + 1)``),
        ``degree_accessibility`` (``p x (N + 1)``) and
        ``mean_degree_by_time`` (``p x T``), averaged over the seeds.
    """
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["model"], run["N"])].append(run)
    summary: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for (model, N), group in sorted(groups.items()):
        p = np.array([row["p"] for row in group[0]["per_p"]])
        entry: dict[str, Any] = {
            "p": p,
            "n_seeds": len(group),
            "T": group[0]["T"],
            "n_features": group[0]["n_features"],
        }
        for name in SCALARS:
            data = np.array([[row[name] for row in run["per_p"]] for run in group])
            entry[f"{name}_mean"] = data.mean(axis=0)
            entry[f"{name}_sem"] = (
                data.std(axis=0, ddof=1) / np.sqrt(len(group))
                if len(group) > 1
                else np.zeros(len(p))
            )
        for name in ("final_degree_profile", "degree_accessibility"):
            entry[name] = np.mean(
                [[row[name] for row in run["per_p"]] for run in group], axis=0
            )
        by_time = np.array(
            [[row["mean_degree_by_time"] for row in run["per_p"]] for run in group],
            dtype=float,
        )
        # At p = 0 the record is empty: its rows are nan for every seed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            entry["mean_degree_by_time"] = np.nanmean(by_time, axis=0)
        _summarize_tasks(entry, group)
        summary[model][N] = entry
    return dict(summary)


def _summarize_tasks(entry: dict[str, Any], group: list[dict[str, Any]]) -> None:
    """Add ``tasks[k][name]`` (mean, sem over p) when the runs scored the QELM."""
    first = group[0]["per_p"][0].get("tasks")
    if not first:
        return
    tasks: dict[str, dict[str, np.ndarray]] = {}
    for k, row in first.items():
        tasks[k] = {}
        for name in row:
            data = np.array(
                [[r["tasks"][k][name] for r in run["per_p"]] for run in group]
            )
            tasks[k][f"{name}_mean"] = data.mean(axis=0)
            tasks[k][f"{name}_sem"] = (
                data.std(axis=0, ddof=1) / np.sqrt(len(group))
                if len(group) > 1
                else np.zeros(data.shape[1])
            )
    entry["tasks"] = tasks


def task_curves(
    summary: dict[str, Any],
    k: str,
    measured: str,
    predicted: str | None,
    ylabel: str,
) -> Figure:
    """One panel per dynamics: a QELM score against ``p`` for target ``k``.

    Markers with error bars: ``measured`` (simulated single-shot QELM);
    dashed line: ``predicted`` (from ``C(p)`` and the shot noise), if given.
    """
    models = [
        m for m in _models(summary) if any("tasks" in e for e in summary[m].values())
    ]
    sizes = sorted({N for m in models for N in summary[m]})
    colors = color_map([str(N) for N in sizes])
    fig, axes = subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        for N, entry in sorted(summary[model].items()):
            task = entry.get("tasks", {}).get(k)
            if task is None:
                continue
            ax.errorbar(
                entry["p"],
                task[f"{measured}_mean"],
                yerr=task[f"{measured}_sem"],
                marker="o",
                markersize=3,
                capsize=2,
                linestyle="none" if predicted else "-",
                color=colors[str(N)],
                label=f"N = {N}",
            )
            if predicted:
                ax.plot(
                    entry["p"],
                    task[f"{predicted}_mean"],
                    linestyle="--",
                    color=colors[str(N)],
                )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("p")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    return fig


def shots_curves(summary: dict[str, Any], N: int, k: str = "k1") -> Figure:
    """Predicted R2 against ``p`` for ``S`` averaged shots, at size ``N``."""
    models = [m for m in _models(summary) if "tasks" in summary[m].get(N, {})]
    names = [f"r2_pred_S{s}" for s in (1, 10, 100, 1000)] + ["r2_ideal"]
    labels = ["S = 1", "S = 10", "S = 100", "S = 1000", "S = inf"]
    colors = color_map(labels)
    fig, axes = subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        entry = summary[model][N]
        for name, label in zip(names, labels, strict=True):
            ax.plot(
                entry["p"],
                entry["tasks"][k][f"{name}_mean"],
                marker="o",
                markersize=3,
                color=colors[label],
                label=label,
            )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("p")
    axes[0].set_ylabel(f"predicted R2, target {k}, N = {N}")
    axes[-1].legend(frameon=False)
    return fig


def _models(summary: dict[str, Any]) -> list[str]:
    return [m for m in MODEL_ORDER if m in summary] + sorted(
        m for m in summary if m not in MODEL_ORDER
    )


def curves_per_model(
    summary: dict[str, Any], name: str, ylabel: str, normalize: str | None = None
) -> Figure:
    """One panel per dynamics: ``name`` against ``p``, one curve per ``N``.

    ``normalize="features"`` divides by the number of features ``T N + N``,
    ``"monomials"`` by ``3**N``.
    """
    models = _models(summary)
    sizes = sorted({N for m in models for N in summary[m]})
    colors = color_map([str(N) for N in sizes])
    fig, axes = subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        for N, entry in sorted(summary[model].items()):
            scale = 1.0
            if normalize == "features":
                scale = entry["n_features"]
            elif normalize == "monomials":
                scale = 3.0**N
            ax.errorbar(
                entry["p"],
                entry[f"{name}_mean"] / scale,
                yerr=entry[f"{name}_sem"] / scale,
                marker="o",
                markersize=3,
                capsize=2,
                color=colors[str(N)],
                label=f"N = {N}",
            )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("p")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    return fig


def degree_heatmaps(summary: dict[str, Any], N: int) -> Figure:
    """Degree profile ``W_k`` of the final ``Z_i`` over ``(p, k)``, at size ``N``."""
    models = [m for m in _models(summary) if N in summary[m]]
    fig, axes = subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, model in zip(axes, models, strict=True):
        entry = summary[model][N]
        W = entry["final_degree_profile"][:, 1:]
        image = ax.imshow(
            W.T,
            origin="lower",
            aspect="auto",
            extent=(entry["p"][0], entry["p"][-1], 0.5, N + 0.5),
            vmin=0.0,
            vmax=float(W.max()),
            cmap="viridis",
        )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("p")
    axes[0].set_ylabel("degree k (angles in the monomial)")
    if image is not None:
        fig.colorbar(image, ax=list(axes), label="W_k, final Z_i")
    return fig


def time_heatmap(summary: dict[str, Any], N: int) -> Figure:
    """Mean degree of the record rows of each time ``t`` over ``(p, t)``."""
    models = [m for m in _models(summary) if N in summary[m]]
    fig, axes = subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, model in zip(axes, models, strict=True):
        entry = summary[model][N]
        D = entry["mean_degree_by_time"]
        image = ax.imshow(
            D.T,
            origin="lower",
            aspect="auto",
            extent=(entry["p"][0], entry["p"][-1], -0.5, D.shape[1] - 0.5),
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("p")
    axes[0].set_ylabel("time step t of M_t")
    if image is not None:
        fig.colorbar(image, ax=list(axes), label="mean degree / N")
    return fig


def accessibility_by_degree(
    summary: dict[str, Any], N: int, p_show: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
) -> Figure:
    """``A_k``, the share of the degree-``k`` functions in reach, at size ``N``."""
    models = [m for m in _models(summary) if N in summary[m]]
    colors = color_map([f"{p:g}" for p in p_show])
    fig, axes = subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        entry = summary[model][N]
        for p in p_show:
            idx = int(np.argmin(np.abs(entry["p"] - p)))
            ax.plot(
                np.arange(N + 1),
                entry["degree_accessibility"][idx],
                marker="o",
                color=colors[f"{p:g}"],
                label=f"p = {entry['p'][idx]:g}",
            )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("degree k")
    axes[0].set_ylabel("A_k (share in reach)")
    axes[-1].legend(frameon=False)
    return fig


def task_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of the QELM scores at a few p, per model, N and target degree."""
    rows = []
    for model in _models(summary):
        for N, entry in sorted(summary[model].items()):
            for k, task in sorted(entry.get("tasks", {}).items()):
                for p in (0.0, 0.1, 0.25, 0.5, 1.0):
                    idx = int(np.argmin(np.abs(entry["p"] - p)))
                    rows.append(
                        {
                            "model": model,
                            "N": N,
                            "target": k,
                            "p": float(entry["p"][idx]),
                            **{
                                name[: -len("_mean")]: float(values[idx])
                                for name, values in task.items()
                                if name.endswith("_mean")
                            },
                        }
                    )
    return rows


def table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of the summary table: per model and N, the measures at a few p."""
    rows = []
    for model in _models(summary):
        for N, entry in sorted(summary[model].items()):
            for p in (0.0, 0.1, 0.25, 0.5, 1.0):
                idx = int(np.argmin(np.abs(entry["p"] - p)))
                rows.append(
                    {
                        "model": model,
                        "N": N,
                        "seeds": entry["n_seeds"],
                        "p": float(entry["p"][idx]),
                        "rank/features": float(
                            entry["rank_mean"][idx] / entry["n_features"]
                        ),
                        "final_rank": float(entry["final_rank_mean"][idx]),
                        "k_final/N": float(entry["final_mean_degree_mean"][idx]),
                        "d_eff": float(entry["d_eff_mean"][idx]),
                        "odd_x": float(entry["odd_x_weight_mean"][idx]),
                    }
                )
    return rows


def draw_all(sweep: Path, show: bool = False) -> dict[str, Path]:
    """Draw and save every figure of the sweep; return their paths."""
    summary = summarize(load_runs(sweep))
    largest = max(N for m in summary for N in summary[m])
    figures = {
        "accessible_rank": curves_per_model(
            summary, "rank", "rank C(p) / number of features", "features"
        ),
        "final_mean_degree": curves_per_model(
            summary, "final_mean_degree", "mean degree / N, final Z_i"
        ),
        "record_mean_degree": curves_per_model(
            summary, "mean_degree", "mean degree / N, all features"
        ),
        "effective_dimension": curves_per_model(
            summary, "d_eff", "d_eff (participation ratio)"
        ),
        "odd_x_weight": curves_per_model(
            summary, "odd_x_weight", "weight on odd-sin monomials"
        ),
        f"degree_profile_N{largest}": degree_heatmaps(summary, largest),
        f"degree_by_time_N{largest}": time_heatmap(summary, largest),
        f"accessibility_by_degree_N{largest}": accessibility_by_degree(
            summary, largest
        ),
    }
    with_tasks = [m for m in summary if any("tasks" in e for e in summary[m].values())]
    if with_tasks:
        degrees = sorted(
            {
                k
                for m in with_tasks
                for e in summary[m].values()
                for k in e.get("tasks", {})
            }
        )
        for k in degrees:
            figures[f"r2_ideal_{k}"] = task_curves(
                summary, k, "r2_ideal", None, f"R2 ceiling (S = inf), target {k}"
            )
            figures[f"r2_qelm_{k}"] = task_curves(
                summary,
                k,
                "r2_test",
                "r2_pred_S1",
                f"QELM test R2 (1 shot), target {k}",
            )
            figures[f"accuracy_qelm_{k}"] = task_curves(
                summary,
                k,
                "accuracy_test",
                "accuracy_pred_S1",
                f"QELM test accuracy (1 shot), target {k}",
            )
        largest_task = max(
            N for m in with_tasks for N, e in summary[m].items() if "tasks" in e
        )
        figures[f"r2_vs_shots_N{largest_task}"] = shots_curves(summary, largest_task)
    paths = {name: save_figure(fig, sweep, name) for name, fig in figures.items()}
    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)
    (sweep / "summary_table.json").write_text(
        json.dumps(table(summary), indent=2) + "\n"
    )
    if with_tasks:
        (sweep / "task_table.json").write_text(
            json.dumps(task_table(summary), indent=2) + "\n"
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    for name, path in draw_all(args.sweep).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
