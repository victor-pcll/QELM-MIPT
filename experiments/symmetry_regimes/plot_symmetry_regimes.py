#!/usr/bin/env python3
"""Figures of the state-or-record regimes of the QELM (issue #311).

Reads a sweep of ``symmetry_regimes.sh`` (folders
``<circuit>_<encoding>_<target>/N<N>_seed<seed>/metrics.json``), averages
every readout over the circuits (seeds) of each task, and writes to ``--out``
(default: the sweep folder):

- ``qelm_by_symmetry_<encoding>_<target>.png``: one panel per gate family,
  each the figure of ``plot_nonlinear_M.plot_comparison``: the QELM on the
  state (``<P_k>``, ``<P_i P_j>``), on ``M``, on both, and the Bayes decoder
  of ``M`` against ``p``;
- ``full_overlay_<encoding>_<target>.png``: the QELM on ``M`` and the state
  of every gate family on one axis (solid), with the Bayes decoder of each
  (dotted, same color);
- ``regime_map_<target>.png``: for each gate family and encoding, in three
  bands of ``p``, the accuracy of the QELM on the state minus that on ``M``
  (blue: the state carries the label, red: the record does), and the
  accuracy of the full QELM;
- ``summary.json``: the averages and standard errors behind the figures.

    uv run python experiments/symmetry_regimes/plot_symmetry_regimes.py \\
        outputs/symmetry_regimes/<stamp>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def _load_plot_nonlinear_m() -> ModuleType:
    """Import ``experiments/nonlinear_M/plot_nonlinear_M.py`` (not a package)."""
    path = HERE.parent / "nonlinear_M" / "plot_nonlinear_M.py"
    spec = importlib.util.spec_from_file_location("plot_nonlinear_M", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PNM = _load_plot_nonlinear_m()

# Gate families in the order of the panels, with a label and a color.
CIRCUITS: tuple[tuple[str, str, str], ...] = (
    ("haar", "Haar U(4)", "#2a78d6"),
    ("clifford", "Clifford", "#eb6834"),
    ("u1", "U(1): charge", "#1baf7a"),
    ("z2", "Z2: parity", "#eda100"),
    ("dipole", "charge + dipole", "#e87ba4"),
    ("diagonal", "diagonal: every Z_i", "#008300"),
)
ENCODINGS: dict[str, str] = {
    "bell": "Bell pair with the middle site",
    "charge": "charge sectors N/2, N/2 - 1",
    "ghz": "GHZ",
    "w": "|0...0> against W",
}
TARGETS: dict[str, str] = {"z": "Z outcome of R", "x": "X outcome of R"}
# The readouts of every panel: (metric, label, color).
READOUTS: tuple[tuple[str, str, str], ...] = (
    ("ridge_quantum", "QELM on the state: <P_k>, <P_i P_j>", "#1f77b4"),
    ("ridge_raw", "QELM on M", "#d62728"),
    ("ridge_M_quantum", "QELM on M and <P_k>, <P_i P_j>", "#fa5703"),
    ("bayes", "Bayes optimal on M (knows the circuit)", "black"),
)
# Bands of p of the regime map: (label, low, high), both ends included.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("0 < p <= 0.05", 1e-9, 0.05),
    ("0.1 <= p <= 0.3", 0.1, 0.3),
    ("0.4 <= p <= 1", 0.4, 1.0),
)


def load_sweep(folder: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Averages over the seeds of every task ``(circuit, encoding, target)``."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        parts = task_dir.name.split("_")
        if len(parts) != 3:
            continue
        try:
            runs = PNM.load_runs([task_dir])
        except FileNotFoundError:
            continue
        for N, group in runs.items():
            ps = PNM.p_values(group[0])
            entry: dict[str, Any] = {
                "N": N,
                "T": group[0].get("T"),
                "n_circuits": len(group),
                "p": ps.tolist(),
            }
            for name, _, _ in READOUTS:
                s = PNM.stat(group, f"accuracy_{name}", ps)
                entry[f"accuracy_{name}"] = s[0].tolist()
                entry[f"accuracy_{name}_sem"] = s[1].tolist()
            entry["S_R_mean"] = PNM.stat(group, "S_R_mean", ps)[0].tolist()
            out[(parts[0], parts[1], parts[2])] = entry
    if not out:
        raise FileNotFoundError(f"No <circuit>_<encoding>_<target>/ runs in {folder}.")
    return out


def band_mean(entry: dict[str, Any], name: str, low: float, high: float) -> float:
    """Mean accuracy of ``name`` over the ``p`` in ``[low, high]``."""
    ps = np.asarray(entry["p"])
    mask = (ps >= low) & (ps <= high)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(entry[f"accuracy_{name}"])[mask]))


def _draw_readouts(ax: Any, entry: dict[str, Any]) -> None:
    ps = np.asarray(entry["p"])
    for name, legend, color in READOUTS:
        mean = np.asarray(entry[f"accuracy_{name}"])
        sem = np.asarray(entry[f"accuracy_{name}_sem"])
        ax.plot(ps, mean, "-", color=color, marker="o", ms=3, lw=1.5, label=legend)
        ax.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.axhline(0.75, color="black", ls="--", lw=0.8, label="3/4")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.02)


def plot_by_symmetry(
    sweep: dict[tuple[str, str, str], dict[str, Any]],
    encoding: str,
    target: str,
    out: Path,
) -> Path | None:
    """One panel per gate family: the three QELMs and Bayes against ``p``."""
    present = [c for c in CIRCUITS if (c[0], encoding, target) in sweep]
    if not present:
        return None
    ncol = 3
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(13, 3.6 * nrow), sharex=True, sharey=True, squeeze=False
    )
    for ax, (key, label, _) in zip(axes.flat, present, strict=False):
        entry = sweep[(key, encoding, target)]
        _draw_readouts(ax, entry)
        ax.set_title(f"{label} ({entry['n_circuits']} circuits)", fontsize=10)
    for ax in axes.flat[len(present) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("measurement probability p")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    entry = sweep[(present[0][0], encoding, target)]
    fig.suptitle(
        f"{ENCODINGS.get(encoding, encoding)}, label: {TARGETS.get(target, target)}, "
        f"N = {entry['N']}, T = {int(entry['T'])}, fixed circuits",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.18 if nrow == 1 else 0.09, 1, 0.97))
    path = out / f"qelm_by_symmetry_{encoding}_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_full_overlay(
    sweep: dict[tuple[str, str, str], dict[str, Any]],
    encoding: str,
    target: str,
    out: Path,
) -> Path | None:
    """The full QELM of every gate family on one axis, with its Bayes decoder."""
    present = [c for c in CIRCUITS if (c[0], encoding, target) in sweep]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for key, label, color in present:
        entry = sweep[(key, encoding, target)]
        ps = np.asarray(entry["p"])
        mean = np.asarray(entry["accuracy_ridge_M_quantum"])
        sem = np.asarray(entry["accuracy_ridge_M_quantum_sem"])
        ax.plot(ps, mean, "-", color=color, marker="o", ms=3, lw=2, label=label)
        ax.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
        ax.plot(ps, entry["accuracy_bayes"], ":", color=color, lw=1.5)
    ax.plot([], [], ":", color="black", lw=1.5, label="Bayes on M (dotted)")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("measurement probability p")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    entry = sweep[(present[0][0], encoding, target)]
    ax.set_title(
        f"QELM on M and <P_k>, <P_i P_j>: {ENCODINGS.get(encoding, encoding)}, "
        f"{TARGETS.get(target, target)}\nN = {entry['N']}, T = {int(entry['T'])}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / f"full_overlay_{encoding}_{target}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def regime_table(
    sweep: dict[tuple[str, str, str], dict[str, Any]], target: str
) -> dict[str, Any]:
    """Band means of every readout, gate family by encoding."""
    circuits = [c[0] for c in CIRCUITS if any(k[0] == c[0] for k in sweep)]
    encodings = [e for e in ENCODINGS if any(k[1] == e for k in sweep)]
    table: dict[str, Any] = {"circuits": circuits, "encodings": encodings}
    for band, low, high in BANDS:
        block: dict[str, list[list[float]]] = {}
        for name, _, _ in READOUTS:
            block[name] = [
                [
                    band_mean(sweep[(c, e, target)], name, low, high)
                    if (c, e, target) in sweep
                    else float("nan")
                    for e in encodings
                ]
                for c in circuits
            ]
        table[band] = block
    return table


def plot_regime_map(table: dict[str, Any], target: str, out: Path) -> Path:
    """Heat maps of ``acc(state) - acc(M)`` and of ``acc(full)`` by band of ``p``."""
    circuits, encodings = table["circuits"], table["encodings"]
    labels = dict((c[0], c[1]) for c in CIRCUITS)
    fig, axes = plt.subplots(
        2, len(BANDS), figsize=(4.2 * len(BANDS), 2 * (0.5 * len(circuits) + 1.2))
    )
    for col, (band, _, _) in enumerate(BANDS):
        block = table[band]
        diff = np.asarray(block["ridge_quantum"]) - np.asarray(block["ridge_raw"])
        full = np.asarray(block["ridge_M_quantum"])
        for row, (values, cmap, vmin, vmax) in enumerate(
            ((diff, "RdBu", -0.5, 0.5), (full, "Greys", 0.5, 1.0))
        ):
            ax = axes[row, col]
            image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(circuits)):
                for j in range(len(encodings)):
                    v = values[i, j]
                    if np.isnan(v):
                        continue
                    dark = abs(v) > 0.3 if row == 0 else v > 0.8
                    ax.text(
                        j,
                        i,
                        f"{v:+.2f}" if row == 0 else f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if dark else "black",
                    )
            ax.set_xticks(range(len(encodings)), encodings)
            ax.set_yticks(range(len(circuits)), [labels[c] for c in circuits])
            if col > 0:
                ax.set_yticklabels([])
            ax.set_title(band, fontsize=10)
            if col == len(BANDS) - 1:
                fig.colorbar(
                    image,
                    ax=ax,
                    label="acc(state) - acc(M)" if row == 0 else "acc(M and state)",
                )
    fig.suptitle(
        f"Which features carry the label ({TARGETS.get(target, target)})? "
        "Top: blue = the state, red = the record M. Bottom: the full QELM.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / f"regime_map_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw every figure; return their paths."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.folder
    out.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep(args.folder)
    paths: list[Path] = []
    summary: dict[str, Any] = {"tasks": {}, "regimes": {}}
    for (c, e, t), entry in sorted(sweep.items()):
        summary["tasks"][f"{c}_{e}_{t}"] = entry
    for target in sorted({k[2] for k in sweep}):
        for encoding in ENCODINGS:
            for plot in (plot_by_symmetry, plot_full_overlay):
                path = plot(sweep, encoding, target, out)
                if path is not None:
                    paths.append(path)
        table = regime_table(sweep, target)
        summary["regimes"][target] = table
        paths.append(plot_regime_map(table, target, out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
