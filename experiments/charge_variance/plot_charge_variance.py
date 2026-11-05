#!/usr/bin/env python3
"""Figures of a sweep of ``charge_variance.sh`` (issue #299).

Reads every ``N<N>_seed<seed>/metrics.json`` of a sweep folder, averages the
seeds (error bars: standard error over seeds) and writes, in that folder:

- ``order_parameter.png``: the exact ``4 <Var(Q)>(p)`` and the ratio
  ``1 - <Var^2> / (3 <Var>^2)`` per ``N``, with the crossings of
  consecutive sizes, and the mean prediction of the pooled readouts on
  exact features (one readout for every ``p``, not told ``p``);
- ``r2_per_p.png``: ``R2`` of each readout trained at each ``p`` on exact
  features, one panel per readout, one curve per ``N``;
- ``r2_pooled.png``: ``R2`` over all ``p`` of the pooled readouts against
  ``N``, one panel per feature set (exact, then each shot count);
- ``shots.png``: ``R2`` per ``p`` of the plug-in, direct and quadratic
  readouts at each shot count, largest ``N``;
- ``summary.json``: the averaged curves and the crossings.

Run from the repository root:

    uv run python experiments/charge_variance/plot_charge_variance.py \
        outputs/charge_variance/<date_time>
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
PER_P_READOUTS = ("n_meas", "sep", "linear_z", "linear_zz", "quadratic_z")
POOLED_READOUTS = (
    "p_mean",
    "n_meas",
    "sep",
    "linear_z",
    "linear_zz",
    "quadratic_z",
    "plugin",
    "direct",
)
ORDER_READOUTS = ("linear_zz", "quadratic_z")
SHOT_READOUTS = ("plugin", "direct", "quadratic_z")
LABELS = {
    "chance": "chance",
    "p_mean": "mean Var(Q) at p (knows p)",
    "n_meas": "number of measurements (record)",
    "sep": "gate-ignorant Bayes on the record, P(1-P)",
    "linear_z": r"linear on $\langle Z_i\rangle$",
    "linear_zz": r"linear on $\langle Z_i\rangle, \langle Z_iZ_j\rangle$",
    "quadratic_z": r"quadratic on $\langle Z_i\rangle$",
    "plugin": "plug-in identity, no fit",
    "direct": "sample variance of the shots",
}
ORACLES = {
    "var4": r"$4\langle \mathrm{Var}(Q)\rangle$",
    "ratio": r"$1 - \langle V^2\rangle / 3\langle V\rangle^2$",
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
        all_nan = n == 0
        safe = np.where(all_nan[None, :], 0.0, a)
        mean = np.where(all_nan, np.nan, np.nanmean(safe, axis=0))
        err = (
            np.nanstd(safe, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            if a.shape[0] > 1
            else np.zeros(a.shape[1])
        )
    return mean, np.nan_to_num(err)


def crossing(p: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float | None:
    """First ``p > 0`` where ``y1 - y2`` changes sign, by linear interpolation.

    ``p = 0`` is skipped: there every size has ``Var(Q) = 1/4`` exactly.
    Returns ``None`` if the curves do not cross.
    """
    d = np.asarray(y1, float) - np.asarray(y2, float)
    for k in range(len(p) - 1):
        if p[k] <= 0 or not (np.isfinite(d[k]) and np.isfinite(d[k + 1])):
            continue
        if d[k] == 0:
            return float(p[k])
        if d[k] * d[k + 1] < 0:
            return float(p[k] - d[k] * (p[k + 1] - p[k]) / (d[k + 1] - d[k]))
    return None


def _scores(rs: list[dict[str, Any]], fs: str, proto: str, name: str) -> dict:
    keys = ("R2", "MAE") + (("mean_prediction",) if proto == "pooled" else ())
    d: dict[str, Any] = {}
    for key in keys:
        m, e = mean_err([r["results"][fs][proto][name][key] for r in rs])
        d[key] = {"mean": m.tolist(), "err": e.tolist()}
    if proto == "pooled":
        m, e = mean_err([[r["results"][fs][proto][name]["R2_all"]] for r in rs])
        d["R2_all"] = {"mean": float(m[0]), "err": float(e[0])}
    return d


def summarize(runs: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Average the runs of each ``N`` and find the crossings.

    Returns:
        ``p``, ``feature_sets``, per ``N`` the oracle order parameters (on
        every trajectory, and ``oracle_test`` on the test trajectories that
        the pooled readouts predict) and
        the per-p and pooled scores of each feature set (mean and error
        over seeds), and ``crossings``: for each oracle order parameter and
        each pooled readout of each feature set, the crossing ``p`` of each
        pair of consecutive sizes.
    """
    first = next(iter(runs.values()))[0]
    p = np.array(first["p"])
    sets = list(first["feature_sets"])
    out: dict[str, Any] = {
        "p": p.tolist(),
        "feature_sets": sets,
        "sizes": {},
        "crossings": {},
    }
    for N, rs in runs.items():
        entry: dict[str, Any] = {
            "seeds": [r["seed"] for r in rs],
            "max_leakage": max(r["max_leakage"] for r in rs),
            "oracle": {},
            "results": {},
        }
        entry["oracle_test"] = {}
        for key in ORACLES:
            m, e = mean_err([r["oracle_all"][key] for r in rs])
            entry["oracle"][key] = {"mean": m.tolist(), "err": e.tolist()}
            # On the test trajectories only, those of the pooled predictions.
            m, e = mean_err([r["oracle"][key] for r in rs])
            entry["oracle_test"][key] = {"mean": m.tolist(), "err": e.tolist()}
        for fs in sets:
            res = rs[0]["results"][fs]
            entry["results"][fs] = {
                proto: {name: _scores(rs, fs, proto, name) for name in res[proto]}
                for proto in ("per_p", "pooled")
            }
        out["sizes"][str(N)] = entry
    sizes = [str(N) for N in runs]
    pairs = list(zip(sizes, sizes[1:], strict=False))

    def cross(curve: Any) -> dict[str, float | None]:
        return {f"{a}-{b}": crossing(p, curve(a), curve(b)) for a, b in pairs}

    for key in ORACLES:
        for name in ("oracle", "oracle_test"):
            out["crossings"][f"{name}_{key}"] = cross(
                lambda s, k=key, o=name: np.array(out["sizes"][s][o][k]["mean"])
            )
    for fs in sets:
        for name in out["sizes"][sizes[0]]["results"][fs]["pooled"]:
            if name == "p_mean":
                continue
            out["crossings"][f"{fs}_{name}"] = cross(
                lambda s, f=fs, n=name: (
                    4
                    * np.array(
                        out["sizes"][s]["results"][f]["pooled"][n]["mean_prediction"][
                            "mean"
                        ]
                    )
                )
            )
    return out


def _errorbar(ax: Any, x: Any, d: dict[str, Any], **kw: Any) -> None:
    ax.errorbar(x, d["mean"], yerr=d["err"], marker="o", ms=3, capsize=2, **kw)


def draw_order_parameter(summary: dict[str, Any]) -> Figure:
    """Exact order parameters (left) and pooled mean predictions (right)."""
    p = np.array(summary["p"])
    sizes = list(summary["sizes"])
    colors = color_map([f"N{N}" for N in sizes])
    pooled = summary["sizes"][sizes[0]]["results"]["exact"]["pooled"]
    readouts = [r for r in ORDER_READOUTS if r in pooled]
    n = len(ORACLES) + len(readouts)
    fig, axes = subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, ORACLES.items(), strict=False):
        for N in sizes:
            o = summary["sizes"][N]["oracle"][key]
            _errorbar(ax, p, o, color=colors[f"N{N}"], label=f"N = {N}")
        ax.set_title(f"exact state: {label}", fontsize=9)
    axes[1].set_ylim(-1.0, 0.75)
    for ax, r in zip(axes[len(ORACLES) :], readouts, strict=True):
        for N in sizes:
            d = summary["sizes"][N]["results"]["exact"]["pooled"][r]["mean_prediction"]
            scaled = {"mean": 4 * np.array(d["mean"]), "err": 4 * np.array(d["err"])}
            _errorbar(ax, p, scaled, color=colors[f"N{N}"], label=f"N = {N}")
            o = summary["sizes"][N]["oracle"]["var4"]["mean"]
            ax.plot(p, o, color=colors[f"N{N}"], ls=":", lw=1)
        ax.set_title(f"pooled readout: {LABELS[r]}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.set_xlabel("$p$")
    axes[0].set_ylabel("order parameter  /  4 x mean prediction")
    axes[0].legend(fontsize=8)
    return fig


def draw_r2_per_p(summary: dict[str, Any], fs: str = "exact") -> Figure:
    """``R2`` of the per-p readouts against ``p``, one panel per readout."""
    p = np.array(summary["p"])
    sizes = list(summary["sizes"])
    colors = color_map([f"N{N}" for N in sizes])
    per_p = summary["sizes"][sizes[0]]["results"][fs]["per_p"]
    names = [r for r in PER_P_READOUTS if r in per_p]
    fig, axes = subplots(1, len(names), figsize=(3.6 * len(names), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, names, strict=True):
        for N in sizes:
            d = summary["sizes"][N]["results"][fs]["per_p"][r]["R2"]
            _errorbar(ax, p, d, color=colors[f"N{N}"], label=f"N = {N}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(LABELS[r], fontsize=9)
        ax.set_xlabel("$p$")
    axes[0].set_ylabel(r"$R^2$ of Var(Q) (test), trained at each $p$")
    axes[0].set_ylim(-0.3, 1.05)
    axes[0].legend(fontsize=8)
    return fig


def draw_r2_pooled(summary: dict[str, Any]) -> Figure:
    """Pooled ``R2`` over all ``p`` against ``N``, one panel per feature set."""
    sizes = list(summary["sizes"])
    sets = summary["feature_sets"]
    colors = color_map(list(POOLED_READOUTS))
    # The palette cycles after six colors: the two estimators of the
    # identity get neutral ones.
    colors.update({"plugin": "black", "direct": "gray"})
    fig, axes = subplots(1, len(sets), figsize=(4.4 * len(sets), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fs in zip(axes, sets, strict=True):
        pooled = summary["sizes"][sizes[0]]["results"][fs]["pooled"]
        for r in (r for r in POOLED_READOUTS if r in pooled):
            d = [
                summary["sizes"][N]["results"][fs]["pooled"][r]["R2_all"] for N in sizes
            ]
            unfitted = r in ("sep", "plugin", "direct")
            ax.errorbar(
                [int(N) for N in sizes],
                [x["mean"] for x in d],
                yerr=[x["err"] for x in d],
                color=colors[r],
                ls="--" if r == "p_mean" else (":" if unfitted else "-"),
                marker="s" if unfitted else "o",
                ms=4,
                capsize=2,
                label=LABELS[r],
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"features: {fs}", fontsize=9)
        ax.set_xlabel("$N$")
    axes[0].set_ylabel(r"$R^2$ of Var(Q) over all $p$ (test)")
    axes[0].set_ylim(-0.1, 1.05)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return fig


def draw_shots(summary: dict[str, Any]) -> Figure:
    """``R2`` per ``p`` at each shot count, largest ``N``, per-p protocol."""
    p = np.array(summary["p"])
    N = list(summary["sizes"])[-1]
    sets = [fs for fs in summary["feature_sets"] if fs != "exact"]
    fig, axes = subplots(1, max(len(sets), 1), figsize=(4.4 * max(len(sets), 1), 3.6))
    axes = np.atleast_1d(axes)
    colors = color_map(list(SHOT_READOUTS))
    for ax, fs in zip(axes, sets, strict=False):
        per_p = summary["sizes"][N]["results"][fs]["per_p"]
        for r in (r for r in SHOT_READOUTS if r in per_p):
            _errorbar(ax, p, per_p[r]["R2"], color=colors[r], label=LABELS[r])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"N = {N}, {fs}", fontsize=9)
        ax.set_xlabel("$p$")
        ax.set_ylim(-0.3, 1.05)
    axes[0].set_ylabel(r"$R^2$ of Var(Q) (test), at each $p$")
    axes[0].legend(fontsize=8)
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
    if len(summary["feature_sets"]) > 1:
        save_figure(draw_shots(summary), args.sweep_dir, "shots")
    for name, cross in summary["crossings"].items():
        txt = "  ".join(
            f"{k}: {v:.3f}" if v is not None else f"{k}: -" for k, v in cross.items()
        )
        print(f"crossing {name}: {txt}")
    print(f"Figures and summary.json in {args.sweep_dir}")


if __name__ == "__main__":
    main()
