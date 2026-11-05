#!/usr/bin/env python3
"""Figures of the convergence in ``PER_P`` (issue #267).

Reads the run folders of ``per_p_convergence.py`` (``per_p_convergence.npz``
and ``resolved_config.json``) and writes, with
:func:`QELM.utils.plots.save_figure` (``<name>.png`` next to
``<name>.json``):

- ``convergence_profile``: the test accuracy against ``PER_P`` (log2 axis)
  for the regimes of ``convergence.regimes``, mean over the circuits with a
  band of +-1 sigma (spread between circuits), the training accuracy
  dashed, and the zone ``PER_P >= PER_P*`` (statistical plateau at every
  ``p``) shaded in the accent color :data:`ACCENT`;
- ``convergence_steps``: ``|mu(2 P) - mu(P)|`` against ``P``, with the
  standard error of the paired difference and the tolerance;
- ``convergence_heatmap``: the mean accuracy over the plane ``(PER_P, p)``;
- ``convergence_spectrum``: the effective rank of the covariance of the
  training features against ``PER_P``, and its spectrum at the largest
  ``PER_P``;
- ``per_p_scaling`` (with several ``N``): ``PER_P*`` and the plateau
  accuracy against ``N``.

Run from the repository root:

    uv run python experiments/per_p_convergence/plot_per_p_convergence.py \\
        outputs/per_p_convergence/N8 [outputs/per_p_convergence/N6 ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib
import numpy as np

from QELM.utils.plots import save_figure, subplots
from QELM.utils.plots.heatmap import SweepGrid, draw_heatmap
from QELM.utils.plots.style import COLORS

# Accent color of the statistical plateau, asked by the issue.
ACCENT = "#fa5703"
HERE = Path(__file__).resolve().parent


def _experiment() -> ModuleType:
    """Import per_p_convergence.py, which is not a package.

    Registered under its name, with its folder on the path, so that the
    worker processes of its ``run`` can import it by name.
    """
    if "per_p_convergence" in sys.modules:
        return sys.modules["per_p_convergence"]
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "per_p_convergence", HERE / "per_p_convergence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["per_p_convergence"] = module
    spec.loader.exec_module(module)
    return module


def load_run(folder: str | Path) -> dict[str, Any]:
    """Read one run folder: its arrays, its config and its summary."""
    folder = Path(folder)
    with np.load(folder / "per_p_convergence.npz") as data:
        results: dict[str, Any] = {key: data[key] for key in data.files}
    cfg = json.loads((folder / "resolved_config.json").read_text())
    tolerance = float(cfg["convergence"]["tolerance"])
    return {
        "folder": folder,
        "N": int(cfg["N"]),
        "n_rounds": int(cfg["convergence"]["n_rounds"]),
        "regimes": [float(p) for p in cfg["convergence"].get("regimes", [])],
        "results": results,
        "summary": _experiment().summarize(results, tolerance),
    }


def _regimes(run: dict[str, Any]) -> list[float]:
    p_values = [float(p) for p in run["results"]["p_values"]]
    chosen = [p for p in run["regimes"] if p in p_values]
    return chosen or p_values


def plateau_start(run: dict[str, Any]) -> tuple[int | None, str]:
    """Start of the shaded plateau, and what it is.

    The universal ``PER_P*`` (every ``p`` of the run) when it exists;
    otherwise the largest ``PER_P*`` of the regimes, when each has one.
    """
    summary = run["summary"]
    if summary["per_p_star"] is not None:
        return int(summary["per_p_star"]), "every p"
    stars = [summary["by_p"][f"{p:g}"]["per_p_star"] for p in _regimes(run)]
    if stars and all(s is not None for s in stars):
        return int(max(stars)), "the regimes"
    return None, ""


def _log2_axis(ax: Any, grid: np.ndarray) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid)
    ax.set_xticklabels([f"{int(g)}" for g in grid], rotation=45)


def plot_profile(run: dict[str, Any], out_dir: Path) -> Path:
    """Accuracy against ``PER_P``: the regimes with the plateau zone (left),
    every ``p`` of the run (right)."""
    summary = run["summary"]
    grid = np.asarray(summary["per_p_values"], dtype=float)
    fig, axes = subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for k, p in enumerate(_regimes(run)):
        row = summary["by_p"][f"{p:g}"]
        mu, sigma = np.asarray(row["mu"]), np.asarray(row["sigma"])
        color = COLORS[k % len(COLORS)]
        ax.plot(grid, mu, "o-", color=color, label=f"p = {p:g}, test")
        ax.fill_between(grid, mu - sigma, mu + sigma, color=color, alpha=0.2)
        ax.plot(grid, row["mu_train"], "--", color=color, alpha=0.7)
    star, scope = plateau_start(run)
    if star is not None:
        ax.axvspan(star, grid[-1], color=ACCENT, alpha=0.15, lw=0)
        ax.axvline(star, color=ACCENT, lw=2, label=f"plateau of {scope}: {star}")
    ax.axhline(0.5, color="gray", lw=1, ls=":")
    ax.plot([], [], "--", color="gray", label="train")
    _log2_axis(ax, grid)
    ax.set_xlabel("PER_P (trajectories per p)")
    ax.set_ylabel("ridge accuracy")
    ax.set_title(
        f"N = {run['N']}, T = {run['n_rounds']}, {summary['n_circuits']} circuits"
    )
    ax.legend(loc="best")
    ax = axes[1]
    cmap = matplotlib.colormaps["viridis"]
    keys = list(summary["by_p"])
    for k, p in enumerate(keys):
        row = summary["by_p"][p]
        color = cmap(k / max(1, len(keys) - 1))
        ax.errorbar(
            grid,
            row["mu"],
            yerr=row["sem"],
            fmt="o-",
            ms=4,
            capsize=2,
            color=color,
            label=f"p = {p}",
        )
        if row["per_p_star"] is not None:
            i = int(np.flatnonzero(grid == row["per_p_star"])[0])
            ax.plot(grid[i], row["mu"][i], "o", ms=11, mfc="none", mec=ACCENT, mew=2)
    ax.axhline(0.5, color="gray", lw=1, ls=":")
    _log2_axis(ax, grid)
    ax.set_xlabel("PER_P (trajectories per p)")
    ax.set_ylabel("ridge accuracy")
    ax.set_title("Every p (error bars: sem; circles: PER_P*)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    return save_figure(fig, out_dir, "convergence_profile", data=summary)


def plot_steps(run: dict[str, Any], out_dir: Path) -> Path:
    """``mu(2 P) - mu(P)`` against ``P``, with its error and the tolerance band.

    Every ``p`` of the run, the regimes in color; the band ``+-tolerance``
    in the accent color.
    """
    summary = run["summary"]
    grid = np.asarray(summary["per_p_values"], dtype=float)[:-1]
    tol = summary["tolerance"]
    fig, ax = subplots(figsize=(7.5, 5))
    ax.axhspan(-tol, tol, color=ACCENT, alpha=0.15, lw=0, label=f"|step| < {tol}")
    regimes = [f"{p:g}" for p in _regimes(run)]
    for p, row in summary["by_p"].items():
        if p not in regimes:
            ax.plot(grid, row["delta"], "-", color="gray", alpha=0.35, lw=1)
    for k, p in enumerate(regimes):
        row = summary["by_p"][p]
        ax.errorbar(
            grid,
            row["delta"],
            yerr=row["sem_delta"],
            fmt="o-",
            capsize=3,
            color=COLORS[k % len(COLORS)],
            label=f"p = {p}",
        )
    ax.axhline(0.0, color="black", lw=0.8)
    if len(summary["by_p"]) > len(regimes):
        ax.plot([], [], "-", color="gray", alpha=0.5, lw=1, label="other p")
    _log2_axis(ax, grid)
    ax.set_xlabel("PER_P")
    ax.set_ylabel("mu(2 PER_P) - mu(PER_P)")
    ax.set_title("Step of the mean accuracy")
    ax.legend(loc="best")
    return save_figure(fig, out_dir, "convergence_steps", data=summary)


def plot_heatmap(run: dict[str, Any], out_dir: Path) -> Path:
    """Mean accuracy over the plane ``(PER_P, p)``."""
    results = run["results"]
    grid = SweepGrid(
        x=np.asarray(results["per_p_values"], dtype=float),
        y=np.asarray(results["p_values"], dtype=float),
        mean=np.asarray(results["accuracy"]).mean(axis=0),
        n_runs=np.full(np.asarray(results["accuracy"]).shape[1:], len(results["S_R"])),
    )
    fig, ax = subplots(figsize=(7.5, 5))
    draw_heatmap(ax, grid, label="ridge accuracy", log_x=True, vmin=0.5, vmax=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid.x)
    ax.set_xticklabels([f"{int(g)}" for g in grid.x], rotation=45)
    ax.set_xlabel("PER_P")
    ax.set_ylabel("p")
    for p, row in run["summary"]["by_p"].items():
        if row["per_p_star"] is not None:
            ax.plot(
                row["per_p_star"], float(p), "o", ms=9, mfc="none", mec=ACCENT, mew=2
            )
    (marker,) = ax.plot([], [], "o", mfc="none", mec=ACCENT, mew=2, label="PER_P*")
    ax.legend(handles=[marker], loc="upper right")
    ax.set_title(f"Accuracy(p, PER_P), N = {run['N']}")
    return save_figure(fig, out_dir, "convergence_heatmap", data=run["summary"])


def plot_spectrum(run: dict[str, Any], out_dir: Path) -> Path:
    """Effective rank against ``PER_P``, and the spectrum at the largest one."""
    results, summary = run["results"], run["summary"]
    grid = np.asarray(summary["per_p_values"], dtype=float)
    p_values = [float(p) for p in results["p_values"]]
    fig, axes = subplots(1, 2, figsize=(13, 5))
    for k, p in enumerate(_regimes(run)):
        color = COLORS[k % len(COLORS)]
        a = p_values.index(p)
        axes[0].plot(
            grid,
            summary["by_p"][f"{p:g}"]["effective_rank"],
            "o-",
            color=color,
            label=f"p = {p:g}",
        )
        eig = np.asarray(results["eigenvalues"])[:, a, -1].mean(axis=0)
        axes[1].plot(np.arange(1, eig.size + 1), eig, color=color, label=f"p = {p:g}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(grid)
    axes[0].set_xticklabels([f"{int(g)}" for g in grid], rotation=45)
    axes[0].set_xlabel("PER_P")
    axes[0].set_ylabel("effective rank of X^T X")
    axes[0].set_title("Participation ratio")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("eigenvalue of X^T X / n")
    axes[1].set_title(f"Spectrum at PER_P = {int(grid[-1])}")
    for ax in axes:
        ax.legend(loc="best")
    return save_figure(fig, out_dir, "convergence_spectrum", data=summary)


def scaling_table(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """``PER_P*`` and the plateau accuracy of every regime against ``N``."""
    runs = sorted(runs, key=lambda r: r["N"])
    regimes = sorted({p for r in runs for p in _regimes(r)})
    table: dict[str, Any] = {"N": [r["N"] for r in runs], "by_p": {}}
    for p in regimes:
        stars, plateau, sem = [], [], []
        for r in runs:
            row = r["summary"]["by_p"].get(f"{p:g}")
            stars.append(None if row is None else row["per_p_star"])
            plateau.append(None if row is None else row["mu"][-1])
            sem.append(None if row is None else row["sem"][-1])
        table["by_p"][f"{p:g}"] = {
            "per_p_star": stars,
            "mu_plateau": plateau,
            "sem_plateau": sem,
        }
    return table


def projection_law(
    table: dict[str, Any], targets: tuple[int, ...] = (6, 8, 10, 12)
) -> dict[str, Any]:
    """Fit ``log2 PER_P* = a + b N`` per regime and project it to ``targets``.

    A fit needs ``PER_P*`` at two sizes at least; with ``b`` doublings of
    ``PER_P*`` per added qubit, ``b = 1`` is the ``2**N`` growth expected
    from a concentration of the features. The projection is an
    extrapolation from a few small sizes, and ``PER_P*`` is a value of the
    grid (a power of 2 times the first value), so ``b`` is coarse.

    Returns:
        Per regime: ``slope`` (``b``), ``intercept`` (``a``) and
        ``projection`` (``{N: PER_P*}``), or ``None`` without enough sizes.
    """
    law: dict[str, Any] = {}
    for p, row in table["by_p"].items():
        pairs = [(n, s) for n, s in zip(table["N"], row["per_p_star"]) if s]
        if len({n for n, _ in pairs}) < 2:
            law[p] = None
            continue
        Ns = np.array([n for n, _ in pairs], dtype=float)
        logs = np.log2([s for _, s in pairs])
        slope, intercept = np.polyfit(Ns, logs, 1)
        law[p] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "projection": {
                int(n): float(2 ** (intercept + slope * n)) for n in targets
            },
        }
    return law


def plot_scaling(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    """``PER_P*`` and the accuracy at the largest ``PER_P`` against ``N``."""
    table = scaling_table(runs)
    fig, axes = subplots(1, 2, figsize=(13, 5))
    for k, (p, row) in enumerate(table["by_p"].items()):
        color = COLORS[k % len(COLORS)]
        Ns = [n for n, s in zip(table["N"], row["per_p_star"]) if s is not None]
        stars = [s for s in row["per_p_star"] if s is not None]
        axes[0].plot(Ns, stars, "o-", color=color, label=f"p = {p}")
        keep = [m is not None for m in row["mu_plateau"]]
        axes[1].errorbar(
            np.asarray(table["N"])[keep],
            np.asarray(row["mu_plateau"], dtype=float)[keep],
            yerr=np.asarray(row["sem_plateau"], dtype=float)[keep],
            fmt="o-",
            capsize=3,
            color=color,
            label=f"p = {p}",
        )
    axes[0].set_yscale("log", base=2)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("PER_P*")
    axes[0].set_title("Saturation threshold")
    axes[1].axhline(0.5, color="gray", lw=1, ls=":")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("accuracy at the largest PER_P")
    axes[1].set_title("Plateau accuracy (T = N)")
    for ax in axes:
        ax.set_xticks(table["N"])
        ax.legend(loc="best")
    return save_figure(fig, out_dir, "per_p_scaling", data=table)


def plot_all(folders: list[str | Path], out_dir: str | Path) -> list[Path]:
    """Draw every figure of the runs in ``folders`` into ``out_dir``.

    The figures of one run are drawn for the run of largest ``N`` closest to
    8 (the reference size of the issue); ``per_p_scaling`` is drawn when
    the runs hold several ``N``.
    """
    out = Path(out_dir)
    runs = [load_run(folder) for folder in folders]
    main = min(runs, key=lambda r: (abs(r["N"] - 8), -len(r["results"]["p_values"])))
    paths = [
        plot_profile(main, out),
        plot_steps(main, out),
        plot_heatmap(main, out),
        plot_spectrum(main, out),
    ]
    if len({r["N"] for r in runs}) > 1:
        paths.append(plot_scaling(runs, out))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="run folders of per_p_convergence")
    parser.add_argument(
        "--out", default=None, help="output folder (default: the first run folder)"
    )
    args = parser.parse_args()
    for path in plot_all(args.folders, args.out or args.folders[0]):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
