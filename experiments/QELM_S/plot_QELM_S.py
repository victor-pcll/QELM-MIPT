#!/usr/bin/env python3
"""Figures of a sweep of ``exp_QELM_S.sh``: entropy prediction (issue #230).

Reads every run folder of the sweep (one per model, ``N`` and seed),
averages over the seeds (standard error as error bars) and writes four
figures, each with its JSON (:func:`QELM.utils.plots.save_figure`), in the
sweep folder unless ``--out`` is given:

- ``entropy_vs_p.png``: one panel per model, the mean half-chain entropy
  divided by its largest value ``N // 2`` against ``p``, one curve per
  ``N``. In the volume-law phase the curves of different ``N`` stay apart
  from 0 and close to each other; in the area-law phase ``S`` stops growing
  with ``N`` and the ratio falls with ``N``. The dashed line is ``p_c`` of
  the run, when set;
- ``r2_summary.png``: one panel per model, the pooled ``R2`` (filled) and the
  ``R2`` at fixed ``p`` (open) of each feature set, for each ``N``. The ``p``
  set is the baseline: a set above it in pooled ``R2``, or above 0 at fixed
  ``p``, reads more than ``p`` from the trajectory;
- ``r2_within_vs_p.png``: one row per model, one column per feature set, the
  ``R2`` at fixed ``p`` against ``p``, one curve per ``N``;
- ``predicted_vs_exact.png``: one row per model (largest ``N``, first
  seed), one column per readout, the predicted against the exact entropy of
  the test trajectories, colored by ``p``, with the diagonal.

    uv run python experiments/QELM_S/plot_QELM_S.py outputs/exp_QELM_S/<date_time>
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from QELM.utils.plots import color_map, save_figure, subplots
from QELM.utils.sweep import load_sweep

KEYS = (("N", "N"), ("seed", "seed"), ("model", "models"))
# Feature sets of r2_summary.png, in the order of the x axis.
SUMMARY_SETS = (
    "p",
    "n_meas",
    "absM",
    "M",
    "Z",
    "Zshot",
    "M_Z",
    "M_Zshot",
    "M_Z_even",
    "p_n_meas",
    "p_absM",
    "p_M_Z",
    "p_M_Z_even",
)
# Feature sets of r2_within_vs_p.png (the PER_P_SETS of exp_QELM_S.py).
PER_P_SETS = ("p", "M_Z", "M_Z_even", "p_absM", "p_M_Z_even")
# Readouts of predicted_vs_exact.png: (y_pred key of the .npz, title).
SCATTER = (
    ("y_pred_p", "ridge on p (baseline)"),
    ("y_pred_M_Z", "ridge on [M, Z]"),
    ("y_pred_M_Z_even", "ridge on [M, Z] and squares"),
    ("y_pred_p_M_Z_even", "ridge on p, [M, Z] and squares"),
)


def _mean_err(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    err = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(arr.mean()), err


def p_values(run_dir: Path) -> list[float]:
    """Return ``entropy_task.p_values`` of a run, from its Hydra config."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return [float(v) for v in cfg.entropy_task.p_values]


def p_c(run_dir: Path) -> float | None:
    """Return ``entropy_task.p_c`` of a run, or ``None``."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    value = cfg.entropy_task.get("p_c")
    return None if value is None else float(value)


def load_rows(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return one row per run, with its folder, metrics and ``p`` values.

    Raises:
        ValueError: If the folder holds no run.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    if not rows:
        raise ValueError(f"No run in {sweep_dir}.")
    for row in rows:
        row["dir"] = Path(sweep_dir) / row["run"]
        row["p_values"] = p_values(row["dir"])
        row["p_c"] = p_c(row["dir"])
    return rows


def _groups(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    out: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        out.setdefault(str(row["model"]), {}).setdefault(int(row["N"]), []).append(row)
    return {m: dict(sorted(by_n.items())) for m, by_n in sorted(out.items())}


def draw_entropy(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    """Mean ``S / (N // 2)`` against ``p``, one panel per model."""
    groups = _groups(rows)
    fig, axes = subplots(
        1, len(groups), figsize=(5.5 * len(groups), 4.2), squeeze=False
    )
    colors = color_map([str(n) for n in sorted({int(r["N"]) for r in rows})])
    table: dict[str, Any] = {}
    for ax, (model, by_n) in zip(axes[0], groups.items(), strict=True):
        for N, runs in by_n.items():
            ps = runs[0]["p_values"]
            stats = [_mean_err([r.get(f"S_mean_p{p:g}") for r in runs]) for p in ps]
            y = np.array([s[0] for s in stats]) / (N // 2)
            e = np.array([s[1] for s in stats]) / (N // 2)
            ax.errorbar(
                ps, y, yerr=e, marker="o", color=colors[str(N)], label=f"N = {N}"
            )
            table[f"{model}_N{N}"] = {"p": ps, "S_over_Smax": y.tolist()}
        pc = by_n[next(iter(by_n))][0]["p_c"]
        if pc is not None:
            ax.axvline(pc, color="0.4", ls="--", lw=1)
        ax.set_title(model)
        ax.set_xlabel("p")
        ax.set_ylabel(r"$\overline{S_A} / (N/2)$")
        ax.legend(frameon=False)
    return fig, table


def draw_summary(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    """Pooled and fixed-p ``R2`` of every feature set, one panel per model."""
    groups = _groups(rows)
    fig, axes = subplots(
        1, len(groups), figsize=(6.5 * len(groups), 4.5), squeeze=False
    )
    colors = color_map([str(n) for n in sorted({int(r["N"]) for r in rows})])
    x = np.arange(len(SUMMARY_SETS))
    table: dict[str, Any] = {}
    for ax, (model, by_n) in zip(axes[0], groups.items(), strict=True):
        shift = np.linspace(-0.2, 0.2, len(by_n)) if len(by_n) > 1 else [0.0]
        for dx, (N, runs) in zip(shift, by_n.items(), strict=True):
            pooled = [_mean_err([r.get(f"r2_{s}") for r in runs]) for s in SUMMARY_SETS]
            within = [
                _mean_err([r.get(f"r2_within_{s}") for r in runs]) for s in SUMMARY_SETS
            ]
            c = colors[str(N)]
            ax.errorbar(
                x + dx,
                [v[0] for v in pooled],
                yerr=[v[1] for v in pooled],
                fmt="o",
                color=c,
                label=f"N = {N}, pooled",
            )
            ax.errorbar(
                x + dx,
                [v[0] for v in within],
                yerr=[v[1] for v in within],
                fmt="o",
                mfc="none",
                color=c,
                label=f"N = {N}, fixed p",
            )
            table[f"{model}_N{N}"] = {
                "sets": list(SUMMARY_SETS),
                "r2": [v[0] for v in pooled],
                "r2_within": [v[0] for v in within],
            }
        ax.axhline(0.0, color="0.4", lw=1)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks(x, SUMMARY_SETS, rotation=45, ha="right")
        ax.set_ylabel(r"$R^2$ (clipped to $[-1, 1]$)")
        ax.set_title(model)
        ax.legend(frameon=False, fontsize=8)
    return fig, table


def draw_within(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    """``R2`` at fixed ``p`` against ``p``, models by rows, sets by columns."""
    groups = _groups(rows)
    fig, axes = subplots(
        len(groups),
        len(PER_P_SETS),
        figsize=(3.6 * len(PER_P_SETS), 3.4 * len(groups)),
        squeeze=False,
        sharey=True,
    )
    colors = color_map([str(n) for n in sorted({int(r["N"]) for r in rows})])
    table: dict[str, Any] = {}
    for row_axes, (model, by_n) in zip(axes, groups.items(), strict=True):
        for ax, name in zip(row_axes, PER_P_SETS, strict=True):
            for N, runs in by_n.items():
                ps = runs[0]["p_values"]
                stats = [
                    _mean_err([r.get(f"r2_within_{name}_p{p:g}") for r in runs])
                    for p in ps
                ]
                ax.errorbar(
                    ps,
                    [s[0] for s in stats],
                    yerr=[s[1] for s in stats],
                    marker="o",
                    ms=3,
                    color=colors[str(N)],
                    label=f"N = {N}",
                )
                table[f"{model}_{name}_N{N}"] = {"p": ps, "r2": [s[0] for s in stats]}
            ax.axhline(0.0, color="0.4", lw=1)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(f"{model}: {name}", fontsize=11)
            ax.set_xlabel("p")
        row_axes[0].set_ylabel(r"$R^2$ at fixed p")
        row_axes[0].legend(frameon=False, fontsize=8)
    return fig, table


def draw_scatter(
    rows: list[dict[str, Any]], exp_name: str = "exp_QELM_S"
) -> plt.Figure:
    """Predicted against exact ``S`` of the test set, colored by ``p``."""
    groups = _groups(rows)
    fig, axes = subplots(
        len(groups),
        len(SCATTER),
        figsize=(3.8 * len(SCATTER), 3.6 * len(groups)),
        squeeze=False,
    )
    for row_axes, (model, by_n) in zip(axes, groups.items(), strict=True):
        N = max(by_n)
        run = sorted(by_n[N], key=lambda r: r["seed"])[0]
        data = np.load(run["dir"] / f"{exp_name}.npz")
        y, p = data["y_test"], data["p"][data["test"]]
        hi = max(float(N // 2), float(y.max()))
        for ax, (key, title) in zip(row_axes, SCATTER, strict=True):
            if key not in data:
                ax.set_visible(False)
                continue
            points = ax.scatter(y, data[key], c=p, s=6, cmap="viridis")
            ax.plot([0, hi], [0, hi], color="0.4", lw=1)
            ax.set_title(f"{model}, N = {N}\n{title}", fontsize=10)
            ax.set_xlabel("exact S (bits)")
        row_axes[0].set_ylabel("predicted S (bits)")
        fig.colorbar(points, ax=row_axes[-1], label="p")
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="folder of one exp_QELM_S.sh sweep")
    parser.add_argument("--out", type=Path, default=None, help="figure folder")
    args = parser.parse_args(argv)
    out = args.out or args.sweep
    rows = load_rows(args.sweep)
    for name, draw in (
        ("entropy_vs_p", draw_entropy),
        ("r2_summary", draw_summary),
        ("r2_within_vs_p", draw_within),
    ):
        fig, table = draw(rows)
        save_figure(fig, out, name, data=table)
        plt.close(fig)
    fig = draw_scatter(rows)
    save_figure(fig, out, "predicted_vs_exact")
    plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
