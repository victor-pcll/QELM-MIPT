#!/usr/bin/env python3
"""Figures of a sweep of ``exp_QELM_R.sh``: purification decoding (issue #247).

Reads every run folder of the sweep (one per dynamics, ``N`` and seed),
averages over the seeds (standard error as error bars) and writes three
figures, each with its JSON (:func:`QELM.utils.plots.save_figure`), in the
sweep folder unless ``--out`` is given. Each has one panel per dynamics
(Haar, Floquet), with and without the scrambling phase before ``R`` is
entangled when the sweep holds both, one color per ``N``, and the dashed
vertical line at ``p_c`` when the run sets it:

- ``purification_vs_p.png``: the mean entropy of the reference qubit,
  ``S_R``, against ``p``. 1 bit while the system hides ``R`` (mixed,
  volume-law phase), 0 once the measurements have purified it; the curves of
  growing ``N`` steepen around the transition;
- ``accuracy_vs_p.png``: the accuracy of the ridge decoder of the record on
  the Z outcome of ``R`` (filled circles, the expected accuracy
  ``accuracy_expected``, free of the noise of the test shots) and the Bayes
  ceiling of any decoder of the record (dashed, ``bayes_accuracy``), against
  ``p``. Both start at the chance level 1/2 (grey line); the gap between them
  is what a linear readout misses;
- ``r2_z_vs_p.png``: ``R2`` of the ridge regression of ``<Z_R>`` on the
  test set, against ``p`` (not defined at ``p = 0``, where ``<Z_R> = 0``).

    uv run python experiments/QELM_R/plot_QELM_R.py outputs/exp_QELM_R/<date_time>
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

KEYS = (("N", "N"), ("seed", "seed"), ("model", "purification_task.dynamics"))
# Panel order of the dynamics; others follow in alphabetical order.
MODEL_ORDER = ("haar", "haar_scrambled", "floquet", "floquet_scrambled")
# Panel titles of the dynamics, with or without the scrambling phase.
TITLES = {
    "haar": "Haar brickwork",
    "haar_scrambled": "Haar, R entangled after scrambling",
    "floquet": "Floquet (Skinner et al. 2019)",
    "floquet_scrambled": "Floquet, R entangled after scrambling",
}


def _mean_err(values: Sequence[float | None]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    err = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(arr.mean()), err


def load_rows(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return one row per run, with its folder, metrics, ``p`` values and ``p_c``.

    Raises:
        ValueError: If the folder holds no run.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    if not rows:
        raise ValueError(f"No run in {sweep_dir}.")
    for row in rows:
        row["dir"] = Path(sweep_dir) / row["run"]
        cfg = OmegaConf.load(row["dir"] / ".hydra" / "config.yaml")
        task = cfg.purification_task
        row["p_values"] = [float(v) for v in task.p_values]
        pc = task.get("p_c")
        row["p_c"] = None if pc is None else float(pc)
    return rows


def panel(row: dict[str, Any]) -> str:
    """Return the panel of a run: its dynamics, ``_scrambled`` when ``R`` was
    entangled after a scrambling phase (``scramble_rounds > 0``)."""
    scrambled = float(row.get("scramble_rounds") or 0.0) > 0.0
    return f"{row['model']}_scrambled" if scrambled else str(row["model"])


def _groups(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    out: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        out.setdefault(panel(row), {}).setdefault(int(row["N"]), []).append(row)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    models = sorted(out, key=lambda m: (order.get(m, len(order)), m))
    return {m: dict(sorted(out[m].items())) for m in models}


def curve(
    runs: list[dict[str, Any]], name: str
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return ``(p, mean, err)`` of the score ``<name>_p<p>`` over the seeds."""
    ps = runs[0]["p_values"]
    stats = [_mean_err([r.get(f"{name}_p{p:g}") for r in runs]) for p in ps]
    return ps, np.array([s[0] for s in stats]), np.array([s[1] for s in stats])


def _figure(
    rows: list[dict[str, Any]],
    lines: Sequence[tuple[str, dict[str, Any]]],
    ylabel: str,
    hline: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """One panel per model; for each ``N``, one curve per ``(score, style)``."""
    groups = _groups(rows)
    fig, axes = subplots(
        1, len(groups), figsize=(5.2 * len(groups), 4.2), squeeze=False
    )
    colors = color_map([str(n) for n in sorted({int(r["N"]) for r in rows})])
    table: dict[str, Any] = {}
    for ax, (model, by_n) in zip(axes[0], groups.items(), strict=True):
        for N, runs in by_n.items():
            for k, (name, style) in enumerate(lines):
                ps, y, e = curve(runs, name)
                label = f"N = {N}" if k == 0 else None
                ax.errorbar(ps, y, yerr=e, color=colors[str(N)], label=label, **style)
                table[f"{model}_N{N}_{name}"] = {
                    "p": ps,
                    "mean": y.tolist(),
                    "err": e.tolist(),
                    "n_seeds": len(runs),
                }
        pc = next(iter(by_n.values()))[0]["p_c"]
        if pc is not None:
            ax.axvline(pc, color="0.4", ls="--", lw=1)
        if hline is not None:
            ax.axhline(hline, color="0.6", lw=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(TITLES.get(model, model))
        ax.set_xlabel("p")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)
    return fig, table


def draw_purification(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    """Mean entropy of ``R`` against ``p``."""
    return _figure(
        rows,
        [("S_R_mean", {"marker": "o", "ms": 4})],
        r"$\overline{S_R}$ (bits)",
        ylim=(-0.03, 1.03),
    )


def draw_accuracy(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    """Expected accuracy of the ridge decoder and the Bayes ceiling against ``p``."""
    return _figure(
        rows,
        [
            ("accuracy_expected", {"marker": "o", "ms": 4}),
            ("bayes_accuracy", {"ls": "--", "lw": 1.2}),
        ],
        "accuracy on the Z outcome of R\n(ridge: circles, Bayes ceiling: dashed)",
        hline=0.5,
        ylim=(0.4, 1.02),
    )


def draw_r2(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]]:
    """``R2`` of the ridge regression of ``<Z_R>`` against ``p``."""
    return _figure(
        rows,
        [("r2_z", {"marker": "o", "ms": 4})],
        r"$R^2$ of $\langle Z_R \rangle$ (test)",
        hline=0.0,
        ylim=(-0.5, 1.02),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="folder of one exp_QELM_R.sh sweep")
    parser.add_argument("--out", type=Path, default=None, help="figure folder")
    args = parser.parse_args(argv)
    out = args.out or args.sweep
    rows = load_rows(args.sweep)
    for name, draw in (
        ("purification_vs_p", draw_purification),
        ("accuracy_vs_p", draw_accuracy),
        ("r2_z_vs_p", draw_r2),
    ):
        fig, table = draw(rows)
        save_figure(fig, out, name, data=table)
        plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
