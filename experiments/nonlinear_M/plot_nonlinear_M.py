#!/usr/bin/env python3
"""Figures of the non-linear features of M (issue #269).

Reads the ``metrics.json`` of every run of one or several sweeps of
``nonlinear_M.sh`` (folders ``N<N>_seed<seed>``), averages them over the
circuits (seeds) of each ``N``, and writes to ``--out`` (default: the first
sweep folder):

- ``accuracy_vs_p_N<N>.png``: the master figure of the issue, the accuracy
  and the ``R2`` of every readout against ``p``, with the Bayes ceiling of
  the observer who knows the circuit and the chance line; the range where
  the non-linear features gain most over ``raw`` is shaded in ``#fa5703``
  (only if that gain exceeds 0.02);
- ``gain_vs_p.png``: ``acc(mode) - acc(raw)`` and the share of the gap to
  the Bayes ceiling that each mode closes, one panel per ``N``;
- ``correlation_spectrum.png``: the root mean square ``|Corr(phi, Y)|`` of
  each group of features (entries, absolute values, pairs) against ``p``,
  over the shuffled-label floor;
- ``correlation_map_N<N>.png``: ``|Corr(m_{t,i}, Y)|`` entry by entry,
  averaged over the circuits, at a few ``p`` (from ``corr_map.json``, or the
  ``.npz`` of older runs);
- ``qelm_comparison_N<N>.png``: the three QELMs alone (the state, ``M``,
  both), ``p`` on a linear axis, when the run has ``ridge_M_quantum``;
- ``summary.json``: the averages and standard errors behind the figures.

    uv run python experiments/nonlinear_M/plot_nonlinear_M.py \\
        outputs/nonlinear_M/<stamp>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#fa5703"
# Readouts in the order of the legend: (metric name, label, color, style).
READOUTS: tuple[tuple[str, str, str, str], ...] = (
    ("ridge_raw", "QELM-M (raw)", "#d62728", "-"),
    ("ridge_abs", "QELM-M (|M|)", "#e6b800", "-"),
    ("ridge_raw_abs", "QELM-M (M, |M|)", "#ff9896", "--"),
    ("ridge_xor_local", "QELM-M (XOR local)", "#2ca02c", "-"),
    ("ridge_xor_cone", "QELM-M (pairs, light cone)", "#17becf", "--"),
    ("ridge_xor_full", "QELM-M (all pairs)", "#006d2c", "-"),
    ("mlp_raw", "MLP on M", "#9467bd", ":"),
    ("ridge_quantum", "QELM on the state", "#1f77b4", "-."),
    ("ridge_M_quantum", "QELM on M and the state", "#fa5703", "-."),
    ("bayes", "Bayes (knows the circuit)", "black", "-"),
)
MODES = ("raw_abs", "abs", "xor_local", "xor_cone", "xor_full")
GROUPS: tuple[tuple[str, str, str], ...] = (
    ("m", "entries m", "#d62728"),
    ("abs", "|m|", "#e6b800"),
    ("pair_space", "m_i m_i+1 (space)", "#2ca02c"),
    ("pair_time", "m_t m_t+1 (time)", "#98df8a"),
    ("pair_all", "all pairs m_a m_b", "#006d2c"),
)
RUN_RE = re.compile(r"N(\d+)_seed(\d+)$")


def load_runs(folders: list[Path]) -> dict[int, list[dict[str, float]]]:
    """``metrics.json`` of every run, grouped by ``N`` (one run per seed)."""
    runs: dict[int, dict[int, dict[str, float]]] = {}
    for folder in folders:
        for path in sorted(folder.glob("N*_seed*/metrics.json")):
            match = RUN_RE.match(path.parent.name)
            if match is None:
                continue
            N, seed = int(match.group(1)), int(match.group(2))
            runs.setdefault(N, {})[seed] = json.loads(path.read_text())
    if not runs:
        raise FileNotFoundError(f"No N<N>_seed<seed>/metrics.json under {folders}.")
    return {N: [by_seed[s] for s in sorted(by_seed)] for N, by_seed in runs.items()}


def p_values(metrics: dict[str, float]) -> np.ndarray:
    """The ``p`` of a run, sorted."""
    keys = [k[len("S_R_mean_p") :] for k in metrics if k.startswith("S_R_mean_p")]
    return np.sort(np.asarray([float(k) for k in keys]))


def stat(runs: list[dict[str, float]], name: str, ps: np.ndarray) -> np.ndarray:
    """Mean and standard error over the runs, shape ``(2, len(ps))``; NaN if absent."""
    values = np.asarray(
        [[run.get(f"{name}_p{p:g}", np.nan) for p in ps] for run in runs], dtype=float
    )
    if np.all(np.isnan(values)):
        return np.full((2, ps.size), np.nan)
    n = np.sum(~np.isnan(values), axis=0)
    mean = np.nanmean(values, axis=0)
    sem = np.nanstd(values, axis=0) / np.sqrt(np.maximum(n, 1))
    return np.vstack([mean, sem])


def summarize(runs: dict[int, list[dict[str, float]]]) -> dict[str, Any]:
    """Averages over the circuits of every metric the figures draw."""
    out: dict[str, Any] = {}
    names = [r[0] for r in READOUTS]
    for N, group in sorted(runs.items()):
        ps = p_values(group[0])
        entry: dict[str, Any] = {
            "n_circuits": len(group),
            "T": group[0].get("T"),
            "fixed_instance": bool(group[0].get("fixed_instance", 1.0)),
            "p": ps.tolist(),
        }
        for name in names:
            for score in ("accuracy", "r2"):
                s = stat(group, f"{score}_{name}", ps)
                if not np.all(np.isnan(s)):
                    entry[f"{score}_{name}"] = s[0].tolist()
                    entry[f"{score}_{name}_sem"] = s[1].tolist()
        for extra in ("S_R_mean", "accuracy_bayes_expected", "corr_floor"):
            entry[extra] = stat(group, extra, ps)[0].tolist()
        for g, _, _ in GROUPS:
            for kind in ("rms", "max"):
                entry[f"corr_{kind}_{g}"] = stat(group, f"corr_{kind}_{g}", ps)[
                    0
                ].tolist()
        out[f"N{N}"] = entry
    return out


def _gain_zone(entry: dict[str, Any]) -> tuple[float, float] | None:
    """The ``p`` range where the best non-linear mode gains most over ``raw``."""
    ps = np.asarray(entry["p"])
    if "accuracy_ridge_raw" not in entry:
        return None
    raw = np.asarray(entry["accuracy_ridge_raw"])
    best = np.full(ps.size, -np.inf)
    for mode in MODES:
        key = f"accuracy_ridge_{mode}"
        if key in entry:
            best = np.maximum(best, np.asarray(entry[key]))
    gain = best - raw
    if not np.isfinite(gain).any() or np.nanmax(gain) <= 0.02:
        return None
    hot = ps[gain >= 0.5 * np.nanmax(gain)]
    return float(hot.min()), float(hot.max())


def plot_accuracy(entry: dict[str, Any], N: int, out: Path) -> Path:
    """Master figure: accuracy and R2 of every readout against ``p``."""
    ps = np.asarray(entry["p"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    zone = _gain_zone(entry)
    for name, label, color, style in READOUTS:
        if f"accuracy_{name}" not in entry:
            continue
        for axis, score in ((ax, "accuracy"), (ax2, "r2")):
            mean = np.asarray(entry[f"{score}_{name}"])
            sem = np.asarray(entry[f"{score}_{name}_sem"])
            axis.plot(ps, mean, style, color=color, label=label, marker="o", ms=3)
            axis.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.15)
    ax.plot(
        ps,
        entry["accuracy_bayes_expected"],
        ":",
        color="black",
        lw=1,
        label="Bayes, E[(1 + |r_z|) / 2]",
    )
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax2.axhline(0.0, color="grey", ls="--", lw=1)
    for axis in (ax, ax2):
        if zone is not None:
            axis.axvspan(*zone, color=ACCENT, alpha=0.15, lw=0)
        axis.set_xlabel("measurement probability p")
        axis.set_xscale("symlog", linthresh=0.05)
        axis.grid(alpha=0.3)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.02)
    ax2.set_ylabel("test R2 of the output against Y")
    ax2.set_ylim(-0.2, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=7, loc="upper right")
    fig.suptitle(
        f"Label of the reference qubit from the record M, Haar U(4), N = {N}, "
        f"T = {int(entry['T'])}, "
        + (
            f"{entry['n_circuits']} fixed circuits"
            if entry["fixed_instance"]
            else f"new gates per trajectory ({entry['n_circuits']} seeds)"
        )
    )
    fig.tight_layout()
    path = out / f"accuracy_vs_p_N{N}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gain(summary: dict[str, Any], out: Path) -> Path:
    """Gain of each mode over ``raw`` and the share of the Bayes gap it closes."""
    Ns = sorted(int(k[1:]) for k in summary)
    fig, axes = plt.subplots(2, len(Ns), figsize=(5 * len(Ns), 7), squeeze=False)
    for col, N in enumerate(Ns):
        entry = summary[f"N{N}"]
        ps = np.asarray(entry["p"])
        if "accuracy_ridge_raw" not in entry:
            continue
        raw = np.asarray(entry["accuracy_ridge_raw"])
        bayes = np.asarray(entry["accuracy_bayes_expected"])
        gap = np.where(bayes - 0.5 > 0.02, bayes - 0.5, np.nan)
        for name, label, color, style in READOUTS:
            if f"accuracy_{name}" not in entry or name == "bayes":
                continue
            acc = np.asarray(entry[f"accuracy_{name}"])
            axes[0, col].plot(
                ps, acc - raw, style, color=color, label=label, marker="o", ms=3
            )
            axes[1, col].plot(
                ps, (acc - 0.5) / gap, style, color=color, marker="o", ms=3
            )
        axes[0, col].axhline(0, color="grey", lw=1)
        axes[1, col].axhline(1, color="black", lw=1, ls=":")
        axes[1, col].axhline(0, color="grey", lw=1)
        axes[0, col].set_title(f"N = {N}")
        axes[0, col].set_ylabel("acc - acc(raw)")
        axes[1, col].set_ylabel("(acc - 1/2) / (acc_Bayes - 1/2)")
        axes[1, col].set_ylim(-0.5, 1.5)
        for axis in axes[:, col]:
            axis.set_xscale("symlog", linthresh=0.05)
            axis.set_xlabel("p")
            axis.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = out / "gain_vs_p.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_spectrum(summary: dict[str, Any], out: Path) -> Path:
    """RMS ``|Corr(phi, Y)|`` per group of features against ``p``."""
    Ns = sorted(int(k[1:]) for k in summary)
    fig, axes = plt.subplots(1, len(Ns), figsize=(5.5 * len(Ns), 4.5), squeeze=False)
    for col, N in enumerate(Ns):
        entry = summary[f"N{N}"]
        ps = np.asarray(entry["p"])
        ax = axes[0, col]
        keep = ps > 0  # no record at p = 0: every correlation is 0
        ps = ps[keep]
        entry = {
            k: np.asarray(v)[keep] if isinstance(v, list) and len(v) == keep.size else v
            for k, v in entry.items()
        }
        for g, label, color in GROUPS:
            ax.plot(
                ps, entry[f"corr_rms_{g}"], color=color, marker="o", ms=3, label=label
            )
            ax.plot(ps, entry[f"corr_max_{g}"], color=color, ls=":", lw=1)
        ax.plot(
            ps, entry["corr_floor"], "--", color="grey", label="1 / sqrt(n) (shuffled)"
        )
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_yscale("log")
        ax.set_xlabel("p")
        ax.set_ylabel("|Corr(phi, Y)|: RMS (solid), max (dotted)")
        ax.set_title(f"N = {N}")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = out / "correlation_spectrum.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


MAP_P = (0.05, 0.1, 0.2, 0.4, 1.0)


def _corr_map(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """``(p, |Corr(m_{t,i}, Y)|)`` of one run, shape ``(n_p, T, N)``."""
    path = run_dir / "corr_map.json"
    if path.exists():
        data = json.loads(path.read_text())
        return np.asarray(data["p"]), np.abs(np.asarray(data["corr_m"]))
    npz = run_dir / "nonlinear_M.npz"
    if not npz.exists():
        return None
    from QELM.features import feature_label_correlation

    arrays = np.load(npz)
    M, Y, p = arrays["M"], arrays["Y"], arrays["p"]
    levels = np.unique(p)
    maps = [
        np.abs(
            feature_label_correlation(
                M[p == level].reshape(int(np.sum(p == level)), -1), Y[p == level]
            )
        ).reshape(M.shape[1:])
        for level in levels
    ]
    return levels, np.asarray(maps)


def plot_corr_map(folders: list[Path], N: int, out: Path) -> Path | None:
    """Heatmaps of ``|Corr(m_{t,i}, Y)|`` averaged over the circuits of ``N``."""
    maps = []
    levels: np.ndarray | None = None
    for folder in folders:
        for run_dir in sorted(folder.glob(f"N{N}_seed*")):
            got = _corr_map(run_dir)
            if got is not None:
                levels, m = got
                maps.append(m)
    if not maps or levels is None:
        return None
    mean = np.mean(maps, axis=0)
    chosen = [i for i, p in enumerate(levels) if any(np.isclose(p, MAP_P))]
    fig, axes = plt.subplots(
        1, len(chosen), figsize=(3.2 * len(chosen), 3.6), squeeze=False
    )
    vmax = float(mean[chosen].max()) or 1.0
    image = None
    for ax, i in zip(axes[0], chosen, strict=True):
        image = ax.imshow(
            mean[i], origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=vmax
        )
        ax.axvline(N // 2, color=ACCENT, lw=1, ls="--")
        ax.set_title(f"p = {levels[i]:g}")
        ax.set_xlabel("site i")
    axes[0, 0].set_ylabel("round t")
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="|Corr(m_ti, Y)|")
    fig.suptitle(
        f"Where M holds the label, N = {N} ({len(maps)} circuits; "
        "dashed: the paired site)"
    )
    path = out / f"correlation_map_N{N}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# The three QELMs of the comparison figure: (metric, label, color).
COMPARISON: tuple[tuple[str, str, str], ...] = (
    ("ridge_quantum", "QELM on the state: <P_k>, <P_i P_j>", "#1f77b4"),
    ("ridge_raw", "QELM on M", "#d62728"),
    ("ridge_M_quantum", "QELM on M and <P_k>, <P_i P_j>", "#fa5703"),
)


def plot_comparison(entry: dict[str, Any], N: int, out: Path) -> Path | None:
    """The three QELMs only (state, record, both), ``p`` on a linear axis."""
    if not all(f"accuracy_{name}" in entry for name, _, _ in COMPARISON):
        return None
    ps = np.asarray(entry["p"])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name, label, color in COMPARISON:
        mean = np.asarray(entry[f"accuracy_{name}"])
        sem = np.asarray(entry[f"accuracy_{name}_sem"])
        ax.plot(ps, mean, "-", color=color, marker="o", ms=4, label=label)
        ax.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xlabel("measurement probability p")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(0, ps.max())
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8)
    gates = (
        f"{entry['n_circuits']} fixed circuits"
        if entry["fixed_instance"]
        else "new gates per trajectory"
    )
    ax.set_title(f"QELM readouts, Haar U(4), N = {N}, T = {int(entry['T'])}, {gates}")
    fig.tight_layout()
    path = out / f"qelm_comparison_N{N}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw every figure; return their paths."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.folders[0]
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.folders)
    summary = summarize(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    paths = [plot_accuracy(summary[f"N{N}"], N, out) for N in sorted(runs)]
    paths += [plot_gain(summary, out), plot_spectrum(summary, out)]
    for N in sorted(runs):
        comparison = plot_comparison(summary[f"N{N}"], N, out)
        if comparison is not None:
            paths.append(comparison)
    for N in sorted(runs):
        heat = plot_corr_map(args.folders, N, out)
        if heat is not None:
            paths.append(heat)
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
