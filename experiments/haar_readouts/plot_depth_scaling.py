#!/usr/bin/env python3
"""Figures of a sweep of ``depth_scaling.sh``: the depth ``T`` of the QELM.

Issue #268. Reads every run folder of the sweep (one per ``N``, depth rule
``T(N)`` and seed, each with one fixed Haar circuit), averages over the
seeds (standard error as error bars) and writes, in the sweep folder unless
``--out`` is given, each figure as ``<name>.png`` and its data
``<name>.json`` (:func:`QELM.utils.plots.save_figure`):

- ``depth_profiles``: one column per depth rule (``T = 10``, ``T = N``,
  ``T = 2N``), one curve per ``N``. Top: the test accuracy of the QELM ridge
  on the charge sector against ``p``, with the sigmoid fit of
  :func:`fit_sigmoid` (thin line). Bottom: the mean entropy of the
  reference qubit ``S_R``, the purification order parameter of the MIPT;
- ``depth_crossings``: left, the steepest slope ``|d acc / dp|`` of each
  accuracy curve (the fit, at its midpoint ``p_0``) against ``N``; middle,
  the crossing points ``p_x`` of the accuracy curves of successive sizes;
  right, the same for ``S_R``. The rule whose crossings agree best (the
  smallest spread over the pairs of sizes) is drawn in :data:`ACCENT`;
- ``depth_collapse``: the accuracy of every rule and ``N`` against ``p``
  and against ``p N T``, the mean number of measurements of the circuit;
- ``depth_cost``: left, the simulation time of one run against ``T``, one
  curve per ``N``; right, the steepest accuracy slope against that time;
- ``depth_edges``: the variance over the trajectories of ``<Z_i>`` against
  the site ``i``, at the largest ``N`` and a few ``p``, one panel per rule:
  a site outside the light cone of the rest of the chain keeps the same
  distribution whatever the record;
- ``depth_conditioning``: ``log10`` of the condition number of the training
  Gram matrix ``X^T X`` of the QELM features against ``p``.

The summary of every curve (fit, crossings, time) goes to
``depth_summary.json``.

    uv run python experiments/haar_readouts/plot_depth_scaling.py \\
        outputs/haar_depth/<date_time> [outputs/haar_depth/<other>]
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import expit

from QELM.utils.plots import color_map, save_figure, subplots
from QELM.utils.sweep import load_sweep

KEYS = (("N", "N"), ("seed", "seed"), ("rule", "readouts_task.depth_rule"))
# Accent color of the rule with the sharpest crossing (issue #268).
ACCENT = "#fa5703"
# Order and titles of the depth rules.
RULE_TITLES = {"10": "T = 10", "N": "T = N", "2N": "T = 2N"}
# Score of the QELM: the ridge on the six observable sets.
SCORE = "ridge_all"
# A crossing of two accuracy curves is searched only where one of them is
# at least this far above chance: on the plateau at 1/2 they cross by noise.
ABOVE_CHANCE = 0.1
# p of the edge figure.
EDGE_P = (0.05, 0.1, 0.2)


def rule_key(rule: Any) -> str:
    """Name of a depth rule, as in :data:`RULE_TITLES` (``10``, ``N``, ``2N``)."""
    return str(rule).strip().replace(" ", "")


def _rule_order(rule: str) -> tuple[int, str]:
    order = list(RULE_TITLES)
    return (order.index(rule) if rule in order else len(order), rule)


def load_rows(sweeps: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Return one row per run of the sweeps, with its rule, ``T`` and ``p``.

    Runs of several sweeps that share ``N``, rule and seed are merged, the
    last one winning.

    Raises:
        ValueError: If a run has no depth rule, or two merged runs differ in
            their ``p`` values.
    """
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sweep in sweeps:
        for row in load_sweep(sweep, keys=KEYS):
            if row["rule"] is None:
                raise ValueError(f"{sweep}/{row['run']}: no readouts_task.depth_rule.")
            cfg = OmegaConf.load(Path(sweep) / row["run"] / ".hydra" / "config.yaml")
            row["p_values"] = [float(v) for v in cfg.readouts_task.p_values]
            row["rule"] = rule_key(row["rule"])
            row["N"] = int(row["N"])
            key = (row["N"], row["rule"], row["seed"])
            if key in merged and merged[key]["p_values"] != row["p_values"]:
                raise ValueError(f"Runs {key} differ in their p values.")
            merged[key] = row
    return list(merged.values())


def groups(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Runs grouped by rule, then by ``N`` (over the seeds), both sorted."""
    out: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        out.setdefault(row["rule"], {}).setdefault(row["N"], []).append(row)
    return {
        rule: dict(sorted(out[rule].items())) for rule in sorted(out, key=_rule_order)
    }


def curve(
    runs: list[dict[str, Any]], name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(p, mean, err)`` of the score ``<name>_p<p>`` over the seeds."""
    ps = np.asarray(runs[0]["p_values"])
    mean, err = [], []
    for p in ps:
        values = np.asarray(
            [r[f"{name}_p{p:g}"] for r in runs if f"{name}_p{p:g}" in r], dtype=float
        )
        mean.append(float(values.mean()) if values.size else np.nan)
        err.append(
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
    return ps, np.asarray(mean), np.asarray(err)


def sigmoid(p: np.ndarray, p0: float, w: float, a: float) -> np.ndarray:
    """``1/2 + a / (1 + exp((p - p0) / w))``: from ``1/2 + a`` to chance."""
    return np.asarray(0.5 + a * expit(-(p - p0) / w))


def fit_sigmoid(
    p: np.ndarray, acc: np.ndarray, err: np.ndarray | None = None
) -> dict[str, float]:
    """Fit :func:`sigmoid` to an accuracy curve.

    Returns:
        ``p0`` (the midpoint, the inflection point), ``w`` (the width),
        ``a`` (the height above chance at ``p = 0``) and ``slope``, the
        steepest ``|d acc / dp| = a / (4 w)``, at ``p0``. ``w`` is kept
        above a quarter of the smallest step of ``p``: a drop sharper than
        the grid gets that bound. NaN if the fit fails.
    """
    ok = np.isfinite(acc)
    p, acc = p[ok], acc[ok]
    half = 0.5 + (np.max(acc) - 0.5) / 2.0
    below = np.flatnonzero(acc < half)
    p0 = float(p[below[0]]) if below.size else float(p[-1])
    # The drop cannot be resolved below the grid: the width stays above a
    # quarter of its smallest step.
    w_min = float(np.min(np.diff(np.unique(p)))) / 4.0 if p.size > 1 else 1e-4
    sigma = None
    if err is not None:
        e = np.asarray(err)[ok]
        sigma = np.maximum(e, 0.01)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                sigmoid,
                p,
                acc,
                p0=(p0, max(0.02, 2.0 * w_min), max(float(np.max(acc)) - 0.5, 0.01)),
                bounds=((0.0, w_min, 0.0), (1.0, 1.0, 0.5)),
                sigma=sigma,
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        nan = float("nan")
        return {"p0": nan, "w": nan, "a": nan, "slope": nan}
    p0_fit, w, a = (float(v) for v in popt)
    return {"p0": p0_fit, "w": w, "a": a, "slope": a / (4.0 * w)}


def crossing(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, floor: float | None = None
) -> float | None:
    """First ``p`` where the curves ``a`` and ``b`` cross (linear
    interpolation), or ``None``.

    Points where both curves are equal are skipped (``p = 0``, where every
    fixed circuit reads the sector exactly). With ``floor``, only the points
    where ``max(a, b) >= floor`` are searched.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    keep = np.isfinite(d) & (np.abs(d) > 1e-12)
    if floor is not None:
        keep &= np.maximum(a, b) >= floor
    idx = np.flatnonzero(keep)
    for i, j in zip(idx[:-1], idx[1:], strict=False):
        if np.sign(d[i]) != np.sign(d[j]):
            t = d[i] / (d[i] - d[j])
            return float(p[i] + t * (p[j] - p[i]))
    return None


def crossings(
    by_n: dict[int, list[dict[str, Any]]], name: str, floor: float | None
) -> list[dict[str, Any]]:
    """Crossings of the curves ``name`` of successive sizes."""
    sizes = sorted(by_n)
    out = []
    for n1, n2 in zip(sizes[:-1], sizes[1:], strict=False):
        p, a, _ = curve(by_n[n1], name)
        _, b, _ = curve(by_n[n2], name)
        out.append({"N": [n1, n2], "p_x": crossing(p, a, b, floor)})
    return out


def spread(points: list[dict[str, Any]]) -> float | None:
    """Spread (max - min) of the crossing points; ``None`` if fewer than two."""
    found = [c["p_x"] for c in points if c["p_x"] is not None]
    if len(found) < 2:
        return None
    return float(max(found) - min(found))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit, crossings and time of every rule and size (``depth_summary.json``)."""
    summary: dict[str, Any] = {"rules": {}}
    for rule, by_n in groups(rows).items():
        sizes: dict[str, Any] = {}
        for n, runs in by_n.items():
            p, acc, err = curve(runs, f"accuracy_{SCORE}")
            times = [float(r["time_simulation_s"]) for r in runs]
            sizes[str(n)] = {
                "T": int(runs[0]["n_rounds"]),
                "n_seeds": len(runs),
                "fit": fit_sigmoid(p, acc, err),
                "time_simulation_s": float(np.mean(times)),
                "time_per_trajectory_s": float(
                    np.mean([r["time_per_trajectory_s"] for r in runs])
                ),
            }
        acc_x = crossings(by_n, f"accuracy_{SCORE}", 0.5 + ABOVE_CHANCE)
        s_x = crossings(by_n, "S_R_mean", None)
        summary["rules"][rule] = {
            "sizes": sizes,
            "accuracy_crossings": acc_x,
            "accuracy_crossing_spread": spread(acc_x),
            "S_R_crossings": s_x,
            "S_R_crossing_spread": spread(s_x),
        }
    rules = summary["rules"]
    summary["sharpest_accuracy_crossing"] = _best(rules, "accuracy_crossing_spread")
    summary["sharpest_S_R_crossing"] = _best(rules, "S_R_crossing_spread")
    return summary


def _best(rules: dict[str, Any], key: str) -> str | None:
    finite = {r: v[key] for r, v in rules.items() if v[key] is not None}
    return min(finite, key=lambda r: finite[r]) if finite else None


def _size_colors(rows: list[dict[str, Any]]) -> dict[int, str]:
    sizes = sorted({row["N"] for row in rows})
    named = color_map([str(n) for n in sizes])
    return {n: named[str(n)] for n in sizes}


def _rule_colors(rules: Sequence[str], best: str | None) -> dict[str, str]:
    grey = ("0.25", "0.55", "0.75", "0.4")
    out = {}
    others = [r for r in rules if r != best]
    for k, rule in enumerate(others):
        out[rule] = grey[k % len(grey)]
    if best is not None:
        out[best] = ACCENT
    return out


def draw_profiles(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    """Figure 1: accuracy and ``S_R`` against ``p``, one column per rule."""
    by_rule = groups(rows)
    colors = _size_colors(rows)
    fig, axes = subplots(
        2, len(by_rule), figsize=(5.2 * len(by_rule), 8.4), squeeze=False
    )
    table: dict[str, Any] = {}
    for col, (rule, by_n) in enumerate(by_rule.items()):
        ax_a, ax_s = axes[0, col], axes[1, col]
        for n, runs in by_n.items():
            T = int(runs[0]["n_rounds"])
            label = f"N = {n} (T = {T})"
            p, acc, err = curve(runs, f"accuracy_{SCORE}")
            ax_a.errorbar(
                p,
                acc,
                yerr=err,
                marker="o",
                ms=4,
                color=colors[n],
                label=label,
                ls="none",
            )
            fit = summary["rules"][rule]["sizes"][str(n)]["fit"]
            if np.isfinite(fit["p0"]):
                fine = np.linspace(p.min(), p.max(), 300)
                ax_a.plot(
                    fine,
                    sigmoid(fine, fit["p0"], fit["w"], fit["a"]),
                    color=colors[n],
                    lw=1.2,
                )
            ps, s, es = curve(runs, "S_R_mean")
            ax_s.errorbar(
                ps, s, yerr=es, marker="o", ms=4, color=colors[n], label=label
            )
            table[f"{RULE_TITLES.get(rule, rule)}, N = {n}"] = {
                "T": T,
                "p": p.tolist(),
                "accuracy": acc.tolist(),
                "accuracy_err": err.tolist(),
                "S_R": s.tolist(),
                "S_R_err": es.tolist(),
                "n_seeds": len(runs),
            }
        ax_a.axhline(0.5, color="0.6", lw=1)
        ax_a.set_ylim(0.4, 1.02)
        ax_a.set_title(f"QELM ridge, {RULE_TITLES.get(rule, rule)}")
        ax_a.set_ylabel("accuracy on the charge sector (test)")
        ax_s.set_ylim(-0.03, 1.03)
        ax_s.set_title(f"Purification, {RULE_TITLES.get(rule, rule)}")
        ax_s.set_ylabel(r"$\overline{S_R}$ (bits)")
        for ax in (ax_a, ax_s):
            ax.set_xlabel("p")
            ax.legend(fontsize=9)
    return fig, table


def draw_crossings(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    """Figure 2: steepest slope against ``N`` and the crossings of the
    accuracy and ``S_R`` curves of successive sizes."""
    rules = list(summary["rules"])
    fig, (ax_k, ax_a, ax_s) = subplots(1, 3, figsize=(16.5, 4.6))
    table: dict[str, Any] = {}
    panels = (
        (
            ax_a,
            "accuracy_crossings",
            "sharpest_accuracy_crossing",
            "Crossings of the accuracy curves",
        ),
        (
            ax_s,
            "S_R_crossings",
            "sharpest_S_R_crossing",
            r"Crossings of the $\overline{S_R}$ curves",
        ),
    )
    acc_colors = _rule_colors(rules, summary["sharpest_accuracy_crossing"])
    for rule in rules:
        info = summary["rules"][rule]
        sizes = sorted(int(n) for n in info["sizes"])
        slopes = [info["sizes"][str(n)]["fit"]["slope"] for n in sizes]
        ax_k.plot(
            sizes,
            slopes,
            marker="o",
            color=acc_colors[rule],
            label=RULE_TITLES.get(rule, rule),
        )
        table[f"{rule}: slope"] = {"N": sizes, "slope": slopes}
    ax_k.set_xlabel("N")
    ax_k.set_ylabel(r"steepest $|d\,\mathrm{acc} / dp|$ (fit)")
    ax_k.set_title("Sharpness of the accuracy drop")
    ax_k.legend(fontsize=9)
    for ax, key, best_key, text in panels:
        colors = _rule_colors(rules, summary[best_key])
        for rule in rules:
            points = summary["rules"][rule][key]
            mid = [np.mean(c["N"]) for c in points if c["p_x"] is not None]
            px = [c["p_x"] for c in points if c["p_x"] is not None]
            spread_v = summary["rules"][rule][
                key.replace("crossings", "crossing_spread")
            ]
            label = RULE_TITLES.get(rule, rule)
            label += f" (spread {spread_v:.3f})" if spread_v is not None else ""
            ax.plot(
                mid,
                px,
                marker="s",
                color=colors[rule],
                label=label,
                lw=2.5 if rule == summary[best_key] else 1.5,
            )
            table[f"{rule}: {key}"] = points
        if not any(
            c["p_x"] is not None for r in rules for c in summary["rules"][r][key]
        ):
            ax.text(
                0.5,
                0.5,
                "no crossing\nfor any rule",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="0.3",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("mean N of the pair (N, N + 2)")
        ax.set_ylabel(r"crossing $p_\times$")
        ax.set_title(text)
        ax.legend(fontsize=9)
    return fig, table


def draw_collapse(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    """Accuracy against ``p`` and against ``p N T``, every rule and ``N``.

    ``p N T`` is the mean number of measurements in the whole circuit. Left:
    the curves against ``p``; right: against ``p N T``, where they fall on
    each other if that number alone sets the drop. Colors are the sizes,
    markers the rules.
    """
    colors = _size_colors(rows)
    markers = {"10": "o", "N": "s", "2N": "^"}
    fig, (ax_p, ax_n) = subplots(1, 2, figsize=(11.5, 4.6))
    table: dict[str, Any] = {}
    for rule, by_n in groups(rows).items():
        for n, runs in by_n.items():
            T = int(runs[0]["n_rounds"])
            p, acc, err = curve(runs, f"accuracy_{SCORE}")
            style = {
                "marker": markers.get(rule, "D"),
                "ms": 4,
                "color": colors[n],
                "lw": 1,
            }
            label = f"N = {n}, T = {T} (rule {rule})"
            ax_p.errorbar(p, acc, yerr=err, label=label, **style)
            ax_n.errorbar(p * n * T, acc, yerr=err, label=label, **style)
            fit = summary["rules"][rule]["sizes"][str(n)]["fit"]
            table[label] = {
                "p": p.tolist(),
                "pNT": (p * n * T).tolist(),
                "accuracy": acc.tolist(),
                "p0_N_T": fit["p0"] * n * T,
            }
    for ax in (ax_p, ax_n):
        ax.axhline(0.5, color="0.6", lw=1)
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel("accuracy on the charge sector (test)")
    ax_p.set_xlabel("p")
    ax_p.set_xlim(-0.005, 0.2)
    ax_p.set_title("QELM ridge, every rule and N")
    ax_n.set_xlabel(r"mean number of measurements $p\,N\,T$")
    ax_n.set_xlim(-0.5, 20)
    ax_n.set_title("The same against p N T")
    ax_n.legend(fontsize=7, ncol=2)
    return fig, table


def draw_cost(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    """Figure 3: simulation time against ``T``, and slope against time."""
    colors = _size_colors(rows)
    fig, (ax_t, ax_g) = subplots(1, 2, figsize=(11.5, 4.6))
    table: dict[str, Any] = {}
    markers = {"10": "o", "N": "s", "2N": "^"}
    by_n: dict[int, list[tuple[int, float, float, str]]] = {}
    for rule, info in summary["rules"].items():
        for n, s in info["sizes"].items():
            by_n.setdefault(int(n), []).append(
                (s["T"], s["time_simulation_s"], s["fit"]["slope"], rule)
            )
    for n in sorted(by_n):
        pts = sorted(by_n[n])
        ax_t.plot(
            [q[0] for q in pts],
            [q[1] for q in pts],
            color=colors[n],
            lw=1,
            label=f"N = {n}",
        )
        for T, t, slope, rule in pts:
            m = markers.get(rule, "D")
            ax_t.plot(T, t, marker=m, color=colors[n], ls="none")
            ax_g.plot(t, slope, marker=m, color=colors[n], ls="none")
        table[f"N = {n}"] = [
            {"T": q[0], "time_s": q[1], "slope": q[2], "rule": q[3]} for q in pts
        ]
    for rule, m in markers.items():
        if rule in summary["rules"]:
            ax_g.plot([], [], marker=m, color="0.3", ls="none", label=RULE_TITLES[rule])
    ax_t.set_xlabel("T (rounds)")
    ax_t.set_ylabel("simulation time of one run (s)")
    ax_t.set_yscale("log")
    ax_t.set_title("Cost of the depth")
    ax_t.legend(fontsize=9)
    ax_g.set_xscale("log")
    ax_g.set_xlabel("simulation time of one run (s)")
    ax_g.set_ylabel(r"steepest $|d\,\mathrm{acc} / dp|$")
    ax_g.set_title("Sharpness against cost")
    ax_g.legend(fontsize=9)
    return fig, table


def draw_edges(rows: list[dict[str, Any]]) -> tuple[plt.Figure, dict[str, Any]] | None:
    """Variance of ``<Z_i>`` against the site, at the largest ``N``; ``None``
    if the runs have no ``var_Z_*`` key."""
    n_max = max(row["N"] for row in rows)
    by_rule = {r: by_n[n_max] for r, by_n in groups(rows).items() if n_max in by_n}
    if not any("var_Z_0_p0" in run for runs in by_rule.values() for run in runs):
        return None
    ps_all = by_rule[next(iter(by_rule))][0]["p_values"]
    shown = [p for p in EDGE_P if p in ps_all] or ps_all[1:4]
    colors = color_map([f"{p:g}" for p in shown])
    fig, axes = subplots(
        1, len(by_rule), figsize=(5.2 * len(by_rule), 4.4), squeeze=False, sharey=True
    )
    table: dict[str, Any] = {}
    sites = np.arange(n_max)
    for col, (rule, runs) in enumerate(by_rule.items()):
        ax = axes[0, col]
        for p in shown:
            values = np.asarray([[r[f"var_Z_{k}_p{p:g}"] for k in sites] for r in runs])
            mean = values.mean(axis=0)
            err = (
                values.std(axis=0, ddof=1) / np.sqrt(len(runs))
                if len(runs) > 1
                else np.zeros(n_max)
            )
            ax.errorbar(
                sites + 1,
                mean,
                yerr=err,
                marker="o",
                ms=4,
                color=colors[f"{p:g}"],
                label=f"p = {p:g}",
            )
            edge = float(np.mean(mean[[0, -1]]))
            center = float(np.mean(mean[n_max // 2 - 1 : n_max // 2 + 1]))
            table[f"{rule}, p = {p:g}"] = {
                "site": (sites + 1).tolist(),
                "var_Z": mean.tolist(),
                "err": err.tolist(),
                "edge_over_center": edge / center if center > 0 else float("nan"),
            }
        T = int(runs[0]["n_rounds"])
        ax.set_title(f"N = {n_max}, T = {T} (rule {rule})")
        ax.set_xlabel("site i")
        ax.legend(fontsize=9)
    axes[0, 0].set_ylabel(r"Var$_{\mathrm{traj}}\,\langle Z_i \rangle$")
    return fig, table


def draw_conditioning(
    rows: list[dict[str, Any]],
) -> tuple[plt.Figure, dict[str, Any]] | None:
    """``log10`` condition number of ``X^T X`` against ``p``; ``None`` if the
    runs have no ``log10_cond_all_*`` key."""
    by_rule = groups(rows)
    if not any("log10_cond_all_p0" in r for r in rows):
        return None
    colors = _size_colors(rows)
    fig, axes = subplots(
        1, len(by_rule), figsize=(5.2 * len(by_rule), 4.4), squeeze=False, sharey=True
    )
    table: dict[str, Any] = {}
    for col, (rule, by_n) in enumerate(by_rule.items()):
        ax = axes[0, col]
        for n, runs in by_n.items():
            p, c, e = curve(runs, "log10_cond_all")
            ax.errorbar(
                p, c, yerr=e, marker="o", ms=4, color=colors[n], label=f"N = {n}"
            )
            table[f"{rule}, N = {n}"] = {"p": p.tolist(), "log10_cond": c.tolist()}
        ax.set_title(f"QELM features, {RULE_TITLES.get(rule, rule)}")
        ax.set_xlabel("p")
        ax.legend(fontsize=9)
    axes[0, 0].set_ylabel(r"$\log_{10}\,\mathrm{cond}(X^T X)$ (train)")
    return fig, table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sweeps", type=Path, nargs="+", help="folders of depth_scaling.sh sweeps"
    )
    parser.add_argument("--out", type=Path, default=None, help="figure folder")
    args = parser.parse_args(argv)
    out = args.out or args.sweeps[0]
    rows = load_rows(args.sweeps)
    summary = summarize(rows)
    for name, drawn in (
        ("depth_profiles", draw_profiles(rows, summary)),
        ("depth_crossings", draw_crossings(rows, summary)),
        ("depth_cost", draw_cost(rows, summary)),
        ("depth_collapse", draw_collapse(rows, summary)),
        ("depth_edges", draw_edges(rows)),
        ("depth_conditioning", draw_conditioning(rows)),
    ):
        if drawn is None:
            continue
        fig, table = drawn
        save_figure(fig, out, name, data=table)
        plt.close(fig)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "depth_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
