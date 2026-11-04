#!/usr/bin/env python3
"""Figures of a sweep of ``clifford_state_readout.sh`` (issue #292).

Reads every ``N<N>_seed<seed>/metrics.json`` of a sweep folder, averages the
seeds (error bars: standard error over seeds) and writes, in that folder:

- ``order_parameter.png``: the oracle ``<S_R>(p)`` per ``N`` (top) and the
  mean prediction of the pooled readouts (bottom: one readout for all ``p``,
  which is not told ``p``), with the crossing of consecutive sizes;
- ``r2_per_p.png``: ``R2`` of each readout trained at each ``p``, against
  ``p``, one panel per readout, one curve per ``N``;
- ``r2_pooled.png``: ``R2`` over all ``p`` of the pooled readouts against
  ``N``, with the ``p_mean`` baseline (knows ``p``, not the trajectory);
- ``summary.json``: the averaged curves and the crossings.

Run from the repository root:

    uv run python experiments/clifford_state_readout/plot_clifford_state_readout.py \
        outputs/clifford_state_readout/<date_time>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from QELM.utils.plots import color_map, save_figure, subplots

RUN_RE = re.compile(r"N(\d+)_seed(\d+)$")
STATE_READOUTS = ("linear", "abs", "quadratic", "mlp")
PER_P_READOUTS = ("n_meas", *STATE_READOUTS)
POOLED_READOUTS = ("p_mean", "n_meas", *STATE_READOUTS)
LABELS = {
    "chance": "chance",
    "p_mean": "mean $S_R$ at $p$ (knows $p$)",
    "n_meas": "number of measurements (record)",
    "linear": r"linear: ridge on $\langle P\rangle$",
    "abs": r"ridge on $|\langle P\rangle|$",
    "quadratic": "ridge, quadratic features",
    "mlp": r"MLP on $|\langle P\rangle|$",
}


def load_runs(sweep_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Return the ``metrics.json`` of each run, grouped by ``N``.

    Raises:
        FileNotFoundError: If the folder holds no run.
    """
    runs: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(sweep_dir.glob("N*_seed*/metrics.json")):
        match = RUN_RE.match(path.parent.name)
        if match:
            runs.setdefault(int(match.group(1)), []).append(
                json.loads(path.read_text())
            )
    if not runs:
        raise FileNotFoundError(f"No N<N>_seed<seed>/metrics.json in {sweep_dir}.")
    return dict(sorted(runs.items()))


def mean_err(rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds (NaN-aware) and standard error, per column."""
    a = np.array(rows, dtype=float)
    with np.errstate(invalid="ignore"):
        n = np.sum(np.isfinite(a), axis=0)
        mean = np.nanmean(a, axis=0) if a.shape[0] else a
        err = (
            np.nanstd(a, axis=0, ddof=1) / np.sqrt(n)
            if a.shape[0] > 1
            else np.zeros(a.shape[1])
        )
    return mean, np.nan_to_num(err)


def crossing(p: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float | None:
    """First ``p`` where ``y1 - y2`` changes sign, by linear interpolation.

    Returns ``None`` if the curves do not cross.
    """
    d = y1 - y2
    for k in range(len(p) - 1):
        if d[k] == 0:
            return float(p[k])
        if d[k] * d[k + 1] < 0:
            return float(p[k] - d[k] * (p[k + 1] - p[k]) / (d[k + 1] - d[k]))
    return None


def summarize(runs: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Average the runs of each ``N`` and find the crossings.

    Returns:
        ``p``, then per ``N``: the oracle, the per-p and pooled scores and
        mean predictions (mean and error over seeds); ``crossings``: for the
        oracle and each pooled readout, the crossing ``p`` of each pair of
        consecutive sizes.
    """
    p = np.array(next(iter(runs.values()))[0]["p"])
    out: dict[str, Any] = {"p": p.tolist(), "sizes": {}, "crossings": {}}
    for N, rs in runs.items():
        entry: dict[str, Any] = {"seeds": [r["seed"] for r in rs]}
        m, e = mean_err([r["oracle_all"] for r in rs])
        entry["oracle"] = {"mean": m.tolist(), "err": e.tolist()}
        entry["per_p"] = {}
        for name in rs[0]["per_p"]:
            entry["per_p"][name] = {}
            for key in ("R2", "MAE"):
                m, e = mean_err([r["per_p"][name][key] for r in rs])
                entry["per_p"][name][key] = {"mean": m.tolist(), "err": e.tolist()}
        entry["pooled"] = {}
        for name in rs[0]["pooled"]:
            d: dict[str, Any] = {}
            for key in ("R2", "MAE", "mean_prediction"):
                m, e = mean_err([r["pooled"][name][key] for r in rs])
                d[key] = {"mean": m.tolist(), "err": e.tolist()}
            m, e = mean_err([[r["pooled"][name]["R2_all"]] for r in rs])
            d["R2_all"] = {"mean": float(m[0]), "err": float(e[0])}
            entry["pooled"][name] = d
        out["sizes"][str(N)] = entry
    sizes = list(runs)
    curves = {"oracle": lambda s: np.array(out["sizes"][str(s)]["oracle"]["mean"])}
    for name in POOLED_READOUTS:
        if name in out["sizes"][str(sizes[0])]["pooled"]:
            curves[name] = lambda s, n=name: np.array(
                out["sizes"][str(s)]["pooled"][n]["mean_prediction"]["mean"]
            )
    for name, curve in curves.items():
        out["crossings"][name] = {
            f"{a}-{b}": crossing(p, curve(a), curve(b))
            for a, b in zip(sizes, sizes[1:], strict=False)
        }
    return out


def draw_order_parameter(summary: dict[str, Any]) -> Figure:
    """Oracle ``<S_R>(p)`` (top) and pooled mean predictions (bottom)."""
    p = np.array(summary["p"])
    sizes = list(summary["sizes"])
    colors = color_map([f"N{N}" for N in sizes])
    readouts = [
        r
        for r in ("abs", "mlp", "quadratic")
        if r in summary["sizes"][sizes[0]]["pooled"]
    ]
    fig, axes = subplots(1, 1 + len(readouts), figsize=(4.2 * (1 + len(readouts)), 3.8))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for N in sizes:
        o = summary["sizes"][N]["oracle"]
        ax.errorbar(
            p,
            o["mean"],
            yerr=o["err"],
            color=colors[f"N{N}"],
            marker="o",
            ms=3,
            capsize=2,
            label=f"N = {N}",
        )
    ax.set_title(r"oracle $\langle S_R\rangle$")
    for ax, r in zip(axes[1:], readouts, strict=True):
        for N in sizes:
            d = summary["sizes"][N]["pooled"][r]["mean_prediction"]
            ax.errorbar(
                p,
                d["mean"],
                yerr=d["err"],
                color=colors[f"N{N}"],
                marker="o",
                ms=3,
                capsize=2,
                label=f"N = {N}",
            )
            o = summary["sizes"][N]["oracle"]["mean"]
            ax.plot(p, o, color=colors[f"N{N}"], ls=":", lw=1)
        ax.set_title(f"pooled readout: {LABELS[r]}", fontsize=9)
    for ax in axes:
        ax.set_xlabel("$p$")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel(r"$\langle S_R\rangle$ (bits)  /  mean prediction")
    axes[0].legend(fontsize=8)
    return fig


def draw_r2_per_p(summary: dict[str, Any]) -> Figure:
    """``R2`` of the per-p readouts against ``p``, one panel per readout."""
    p = np.array(summary["p"])
    sizes = list(summary["sizes"])
    colors = color_map([f"N{N}" for N in sizes])
    names = [r for r in PER_P_READOUTS if r in summary["sizes"][sizes[0]]["per_p"]]
    fig, axes = subplots(1, len(names), figsize=(3.6 * len(names), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, names, strict=True):
        for N in sizes:
            d = summary["sizes"][N]["per_p"][r]["R2"]
            ax.errorbar(
                p,
                d["mean"],
                yerr=d["err"],
                color=colors[f"N{N}"],
                marker="o",
                ms=3,
                capsize=2,
                label=f"N = {N}",
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(LABELS[r], fontsize=9)
        ax.set_xlabel("$p$")
    axes[0].set_ylabel(r"$R^2$ of $S_R$ (test), trained at each $p$")
    axes[0].set_ylim(-0.1, 0.4)
    axes[0].legend(fontsize=8)
    return fig


def draw_r2_pooled(summary: dict[str, Any]) -> Figure:
    """Pooled ``R2`` over all ``p`` against ``N``, per readout."""
    sizes = [int(N) for N in summary["sizes"]]
    names = [
        r for r in POOLED_READOUTS if r in summary["sizes"][str(sizes[0])]["pooled"]
    ]
    colors = color_map(names)
    fig, ax = subplots(figsize=(5.2, 3.8))
    for r in names:
        m = [summary["sizes"][str(N)]["pooled"][r]["R2_all"]["mean"] for N in sizes]
        e = [summary["sizes"][str(N)]["pooled"][r]["R2_all"]["err"] for N in sizes]
        ls = "--" if r == "p_mean" else "-"
        ax.errorbar(
            sizes,
            m,
            yerr=e,
            color=colors[r],
            ls=ls,
            marker="o",
            ms=4,
            capsize=2,
            label=LABELS[r],
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$R^2$ of $S_R$ over all $p$ (test)")
    ax.legend(fontsize=8)
    return fig


def main() -> None:
    """Draw the figures of the sweep folder given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    summary = summarize(load_runs(args.sweep_dir))
    (args.sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_figure(draw_order_parameter(summary), args.sweep_dir, "order_parameter")
    save_figure(draw_r2_per_p(summary), args.sweep_dir, "r2_per_p")
    save_figure(draw_r2_pooled(summary), args.sweep_dir, "r2_pooled")
    for name, cross in summary["crossings"].items():
        txt = "  ".join(
            f"{k}: {v:.3f}" if v is not None else f"{k}: -" for k, v in cross.items()
        )
        print(f"crossing {name}: {txt}")
    print(f"Figures and summary.json in {args.sweep_dir}")


if __name__ == "__main__":
    main()
