#!/usr/bin/env python3
"""The two figures of the tomography task (issue #277): Bloch vector and purity.

Reads the ``N<N>_seed<seed>/metrics.json`` of one or more sweeps of
``bloch_purity.sh`` (one fixed circuit per seed), averages every score over
the circuits (mean and standard error) and draws, for each ``N``:

- ``bloch_vector_N<N>.png``: the ``R2`` of the Bloch vector
  ``(<X_R>, <Y_R>, <Z_R>)`` against ``p``, against the exact ``r(M)``
  (top) and, when the runs have them (``bloch_task.labels``: ``shots`` or
  ``both``), against single-shot Pauli measurements of ``R`` (bottom), where
  the Bayes decoder has a ceiling below 1;
- ``purity_N<N>.png``: the ``R2`` of ``Tr(rho_R^2)`` against ``p``, without
  the Bayes line (``R2 = 1``) so that the scale shows the QELMs (top), and
  the mean absolute error of the predicted purity (bottom), defined at every
  ``p``, with a horizontal line at the top of each QELM curve. Where ``R`` is
  pure on almost every trajectory (``std Tr(rho_R^2) <``
  :data:`PURITY_STD_MIN`, ``p >~ 0.6`` at ``N = 8``) the ``R2`` divides by a
  vanishing variance and is not drawn; the error stays defined.

Curves: QELM on M, QELM on the state, QELM on M and the state, Bayes optimal
on M (the exact target, ``R2 = 1``) and chance (``R2 = 0``). Also writes
``summary.json``, the averaged curves:

    uv run python experiments/bloch_purity/plot_bloch_purity.py \\
        outputs/bloch_purity/<date_time> --out figs
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Curves of both figures, in the order of the legend: (readout, label, color).
CURVES: tuple[tuple[str, str, str], ...] = (
    ("M", "QELM on M", "#d62728"),
    ("state", "QELM on the state: <P_k>, <P_i P_j>", "#1f77b4"),
    ("M_state", "QELM on M and the state", "#fa5703"),
    ("bayes", "Bayes optimal on M (knows the circuit)", "black"),
    ("chance", "chance (mean of the training targets)", "grey"),
)
# Scores averaged over the circuits.
SCORES = (
    "r2_bloch",
    "r2_x",
    "r2_y",
    "r2_z",
    "r2_purity",
    "r2_purity_from_bloch",
    "r2_shots",
    "mse_bloch",
    "mae_purity",
)
# Below this spread of the exact purity over the trajectories of one p, its
# R2 is not drawn (a ratio of two numerical noises).
PURITY_STD_MIN = 1e-2
RUN_RE = re.compile(r"N(\d+)_seed(\d+)$")


def load_runs(folders: list[Path]) -> dict[int, list[dict[str, float]]]:
    """``metrics.json`` of every run, grouped by ``N`` (one run per seed).

    Runs of the same ``N`` and seed in several folders (extra values of
    ``p`` on the same circuit) are merged.

    Raises:
        FileNotFoundError: If no run is found.
    """
    runs: dict[int, dict[int, dict[str, float]]] = {}
    for folder in folders:
        for path in sorted(folder.glob("N*_seed*/metrics.json")):
            match = RUN_RE.match(path.parent.name)
            if match is None:
                continue
            N, seed = int(match.group(1)), int(match.group(2))
            runs.setdefault(N, {}).setdefault(seed, {}).update(
                json.loads(path.read_text())
            )
    if not runs:
        raise FileNotFoundError(f"No N<N>_seed<seed>/metrics.json under {folders}.")
    return {N: [by_seed[s] for s in sorted(by_seed)] for N, by_seed in runs.items()}


def p_values(metrics: dict[str, float]) -> np.ndarray:
    """The ``p`` of a run, sorted."""
    prefix = "purity_mean_p"
    return np.sort([float(k[len(prefix) :]) for k in metrics if k.startswith(prefix)])


def stat(runs: list[dict[str, float]], name: str, ps: np.ndarray) -> np.ndarray:
    """Mean and standard error over the runs, shape ``(2, len(ps))``.

    NaN entries (``R2`` of a target that does not vary) are left out; a
    point with no finite value is NaN.
    """
    values = np.asarray(
        [[run.get(f"{name}_p{p:g}", np.nan) for p in ps] for run in runs], dtype=float
    )
    n = np.sum(np.isfinite(values), axis=0)
    with warnings.catch_warnings():
        # nanmean / nanstd of an all-NaN column.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.where(n > 0, np.nanmean(values, axis=0), np.nan)
        sem = np.where(
            n > 0, np.nanstd(values, axis=0) / np.sqrt(np.maximum(n, 1)), np.nan
        )
    return np.vstack([mean, sem])


def _clean(values: np.ndarray) -> list[float | None]:
    """JSON-safe list: NaN becomes ``null``."""
    return [float(v) if np.isfinite(v) else None for v in values]


def summarize(runs: dict[int, list[dict[str, float]]]) -> dict[str, Any]:
    """Averages over the circuits of every score, per ``N``."""
    out: dict[str, Any] = {}
    for N, group in sorted(runs.items()):
        ps = p_values(group[0])
        entry: dict[str, Any] = {
            "n_circuits": len(group),
            "T": group[0].get("T"),
            "samples_per_p": group[0].get("samples_per_p"),
            "fixed_instance": bool(group[0].get("fixed_instance", 1.0)),
            "p": ps.tolist(),
        }
        for extra in ("purity_mean", "purity_std", "S_R_mean", "bayes_shots_ceiling"):
            s = stat(group, extra, ps)
            entry[extra] = _clean(s[0])
            entry[f"{extra}_sem"] = _clean(s[1])
        for score in SCORES:
            for name, _, _ in CURVES:
                s = stat(group, f"{score}_{name}", ps)
                entry[f"{score}_{name}"] = _clean(s[0])
                entry[f"{score}_{name}_sem"] = _clean(s[1])
        out[f"N{N}"] = entry
    return out


def _array(entry: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in entry[key]], dtype=float)


def _title(entry: dict[str, Any], N: int, what: str) -> str:
    gates = (
        f"{entry['n_circuits']} fixed circuits"
        if entry["fixed_instance"]
        else "new gates per trajectory"
    )
    samples = entry.get("samples_per_p")
    per_p = f", {int(samples)} trajectories per p" if samples else ""
    return (
        f"{what}\nBell pair with the middle site, Haar U(4), N = {N}, "
        f"T = {int(entry['T'])}, {gates}{per_p}"
    )


def _draw_curves(
    ax: Any,
    entry: dict[str, Any],
    score: str,
    mask: np.ndarray | None = None,
    skip: tuple[str, ...] = (),
) -> None:
    ps = np.asarray(entry["p"])
    for name, legend, color in CURVES:
        if name in skip:
            continue
        mean = _array(entry, f"{score}_{name}")
        sem = _array(entry, f"{score}_{name}_sem")
        if mask is not None:
            mean, sem = np.where(mask, mean, np.nan), np.where(mask, sem, np.nan)
        style = "--" if name == "chance" else "-"
        marker = None if name in ("bayes", "chance") else "o"
        ax.plot(ps, mean, style, color=color, marker=marker, ms=4, label=legend)
        if marker is not None:
            ax.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
    ax.set_xlabel("measurement probability p")
    ax.set_xlim(0, ps.max())


def _mark_maxima(ax: Any, entry: dict[str, Any], score: str) -> None:
    """A horizontal line at the top of each QELM curve (the top of the bell),
    labelled with its value and the ``p`` where it is reached."""
    ps = np.asarray(entry["p"])
    for name, _, color in CURVES:
        if name in ("bayes", "chance"):
            continue
        mean = _array(entry, f"{score}_{name}")
        if not np.any(np.isfinite(mean)):
            continue
        k = int(np.nanargmax(mean))
        ax.axhline(mean[k], color=color, ls=":", lw=1)
        ax.annotate(
            f"max {mean[k]:.3f} at p = {ps[k]:g}",
            xy=(ps.max(), mean[k]),
            xytext=(-4, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=color,
        )


def _has(entry: dict[str, Any], key: str) -> bool:
    return key in entry and np.any(np.isfinite(_array(entry, key)))


def plot_bloch(entry: dict[str, Any], N: int, out: Path) -> Path:
    """``R2`` of the Bloch vector of ``R`` against ``p``: exact targets, and
    single-shot labels when the runs have them."""
    shots = _has(entry, "r2_shots_bayes")
    exact = _has(entry, "r2_bloch_bayes")
    panels = [k for k, ok in (("r2_bloch", exact), ("r2_shots", shots)) if ok]
    if not panels:
        raise ValueError("The runs hold neither exact nor single-shot scores.")
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7, 5 + 4 * (len(panels) - 1)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, score in zip(axes, panels, strict=True):
        _draw_curves(ax, entry, score)
        if score == "r2_bloch":
            ax.set_ylim(-0.1, 1.05)
            ax.set_ylabel("test R2 against the exact r(M)")
            ax.set_title("exact targets (<X_R>, <Y_R>, <Z_R>) given M", fontsize=9)
        else:
            ax.set_ylabel("test R2 against one shot of R along X, Y or Z")
            ax.set_title(
                "single-shot labels: R measured once along a random axis; "
                "Bayes R2 ~ E[r_a^2] < 1",
                fontsize=9,
            )
        if ax is not axes[-1]:
            ax.set_xlabel("")
    axes[-1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.suptitle(_title(entry, N, "Tomography of the reference qubit"), fontsize=10)
    fig.tight_layout()
    path = out / f"bloch_vector_N{N}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_purity(entry: dict[str, Any], N: int, out: Path) -> Path:
    """``R2`` and mean absolute error of the purity of ``R`` against ``p``."""
    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(7, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    spread = _array(entry, "purity_std")
    # No Bayes line (R2 = 1) on top, so that the scale shows the QELMs.
    _draw_curves(
        ax, entry, "r2_purity", mask=spread >= PURITY_STD_MIN, skip=("bayes",)
    )
    ax.set_xlabel("")
    ax.set_ylabel("test R2 of the purity Tr(rho_R^2)")
    _draw_curves(ax_err, entry, "mae_purity")
    _mark_maxima(ax_err, entry, "mae_purity")
    ax_err.set_ylim(bottom=-0.005)
    ax_err.set_ylabel("test mean |Tr(rho^2)_pred - Tr(rho^2)|")
    ax_err.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    ax.set_title(_title(entry, N, "Purity of the reference qubit"), fontsize=10)
    fig.tight_layout()
    path = out / f"purity_N{N}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw both figures for every ``N``; return their paths."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.folders[0]
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.folders)
    summary = summarize(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    paths: list[Path] = []
    for N in sorted(runs):
        entry = summary[f"N{N}"]
        paths.append(plot_bloch(entry, N, out))
        if _has(entry, "r2_purity_bayes"):
            paths.append(plot_purity(entry, N, out))
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
