#!/usr/bin/env python3
"""Figures of a sweep of ``decoder_hierarchy.sh`` (issue #307).

Reads every ``N<N>_seed<seed>/metrics.json`` of a sweep folder, averages the
seeds (error bars: standard error over seeds) and writes, in that folder:

- ``decoder_hierarchy.png``: the accuracy of each decoder against ``p``,
  one panel per decoder and one curve per ``N``; the share ``g`` of the
  advantage of Bayes; and every decoder at the largest ``N``;
- ``crossings.png``: the crossing ``p*`` of each pair of consecutive sizes,
  for each decoder, against ``1 / sqrt(N N')``; filled when resolved;
- ``summary.json``: the averaged curves and the crossings.

Run from the repository root:

    uv run python experiments/decoder_hierarchy/plot_decoder_hierarchy.py \
        outputs/decoder_hierarchy/<date_time>
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from QELM.utils.plots import color_map, save_figure, subplots

RUN_RE = re.compile(r"N(\d+)_seed(\d+)$")
DECODERS = ("bayes", "sep", "ridge_M", "ridge_Z", "n_ones")
PANELS = ("bayes", "sep", "ridge_M", "ridge_Z")
SHARES = ("sep", "ridge_M", "n_ones")
LABELS = {
    "bayes": "Bayes, gates known (postBQP)",
    "sep": "SEP, gates averaged (stat-mech)",
    "ridge_M": "ridge on vec(M) (QELM-M)",
    "ridge_Z": r"ridge on final $\langle Z_i\rangle$ (state)",
    "n_ones": "ridge on the counts of 0 and 1 (control)",
}
STYLES = {
    "bayes": "-",
    "sep": "--",
    "ridge_M": "-.",
    "ridge_Z": ":",
    "n_ones": (0, (1, 3)),
}
DECODER_COLORS = {
    "bayes": "black",
    "sep": "tab:blue",
    "ridge_M": "tab:orange",
    "ridge_Z": "tab:green",
    "n_ones": "tab:gray",
}
# A crossing is resolved when the difference is beyond Z_RESOLVED standard
# errors on both sides of it.
Z_RESOLVED = 2.0


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


def mean_err(
    rows: list[list[float]], within: list[list[float]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds and its error, per column.

    The error is the standard error over seeds; with one seed, or when it is
    smaller, the error within the runs (``within``, combined in quadrature
    over seeds) is used instead.
    """
    a = np.array(rows, dtype=float)
    k = a.shape[0]
    with warnings.catch_warnings():
        # Columns where every seed is NaN (the share at p = 0) stay NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(a, axis=0)
        err = (
            np.nanstd(a, axis=0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(a.shape[1])
        )
    if within is not None:
        w = np.sqrt(np.nansum(np.array(within, float) ** 2, axis=0)) / k
        err = np.maximum(err, w)
    return mean, np.nan_to_num(err)


def crossing(
    p: np.ndarray, d: np.ndarray, e: np.ndarray, p_min: float = 0.0
) -> dict[str, Any] | None:
    """First crossing of a difference ``d`` (error ``e``) at ``p > p_min``.

    Returns:
        ``{"p": p*, "resolved": bool}`` or ``None`` when ``d`` keeps its
        sign. Resolved: some point before and some point after the crossing
        are beyond :data:`Z_RESOLVED` errors, with opposite signs.
    """
    d = np.asarray(d, float)
    e = np.asarray(e, float)
    for k in range(len(p) - 1):
        if p[k] <= p_min or not (np.isfinite(d[k]) and np.isfinite(d[k + 1])):
            continue
        if d[k] * d[k + 1] < 0 or d[k] == 0:
            ps = (
                float(p[k] - d[k] * (p[k + 1] - p[k]) / (d[k + 1] - d[k]))
                if d[k]
                else float(p[k])
            )
            z = np.where(e > 0, d / np.where(e > 0, e, 1), 0.0)
            before = z[: k + 1][p[: k + 1] > p_min]
            after = z[k + 1 :]
            s = np.sign(d[k]) if d[k] else -np.sign(d[k + 1])
            resolved = bool(
                np.any(s * before > Z_RESOLVED) and np.any(-s * after > Z_RESOLVED)
            )
            return {"p": ps, "resolved": resolved}
    return None


def summarize(runs: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Average the seeds of each ``N`` and find the crossings of consecutive sizes."""
    first = next(iter(runs.values()))[0]
    p = np.array(first["p"])
    decoders = [d for d in DECODERS if d in first["accuracy"]]
    sizes: dict[int, Any] = {}
    for N, rs in runs.items():
        acc = {}
        for d in decoders:
            m, e = mean_err(
                [r["accuracy"][d] for r in rs], [r["accuracy_err"][d] for r in rs]
            )
            acc[d] = {"mean": m.tolist(), "err": e.tolist()}
        share = {}
        for d in decoders:
            m, e = mean_err([r["share"][d] for r in rs])
            share[d] = {"mean": m.tolist(), "err": e.tolist()}
        sizes[N] = {"seeds": [r["seed"] for r in rs], "accuracy": acc, "share": share}
    crossings: dict[str, dict[str, Any]] = {}
    Ns = list(sizes)
    for d in decoders:
        crossings[d] = {}
        for N1, N2 in zip(Ns[:-1], Ns[1:], strict=True):
            a1, a2 = sizes[N1]["accuracy"][d], sizes[N2]["accuracy"][d]
            diff = np.array(a2["mean"]) - np.array(a1["mean"])
            err = np.hypot(a1["err"], a2["err"])
            crossings[d][f"{N1}-{N2}"] = crossing(p, diff, err)
    return {
        "p": p.tolist(),
        "decoders": decoders,
        "sizes": sizes,
        "crossings": crossings,
    }


def _band(ax: Any, p: np.ndarray, d: dict[str, Any], **kw: Any) -> None:
    m, e = np.array(d["mean"]), np.array(d["err"])
    (line,) = ax.plot(p, m, marker="o", ms=2.5, **kw)
    ax.fill_between(p, m - e, m + e, color=line.get_color(), alpha=0.2, lw=0)


def draw_hierarchy(summary: dict[str, Any]) -> Figure:
    """Accuracy per decoder and ``N``, Bayes share, every decoder at the largest N."""
    p = np.array(summary["p"])
    sizes = [int(N) for N in summary["sizes"]]
    S = {int(k): v for k, v in summary["sizes"].items()}
    colors = color_map([f"N{N}" for N in sizes])
    panels = [d for d in PANELS if d in summary["decoders"]]
    fig, axes = subplots(2, 3, figsize=(13.5, 7.6))
    axes = np.asarray(axes).ravel()
    for ax, d in zip(axes, panels, strict=False):
        for N in sizes:
            _band(ax, p, S[N]["accuracy"][d], color=colors[f"N{N}"], label=f"N = {N}")
            if d != "bayes":
                ax.plot(
                    p,
                    S[N]["accuracy"]["bayes"]["mean"],
                    color=colors[f"N{N}"],
                    ls=":",
                    lw=0.8,
                )
        ax.set_title(LABELS[d], fontsize=9)
        ax.set_ylim(0.48, 1.01)
    axes[0].legend(fontsize=8, loc="lower right")
    ax = axes[4]
    for d in SHARES:
        if d not in summary["decoders"]:
            continue
        for N in sizes:
            sh = S[N]["share"][d]
            m = np.array(sh["mean"])
            ok = p > 0
            ax.plot(p[ok], m[ok], color=colors[f"N{N}"], ls=STYLES[d], marker="o", ms=2)
        ax.plot([], [], color="k", ls=STYLES[d], label=LABELS[d].split(" (")[0])
    ax.set_title(
        "share of the Bayes advantage  g = (acc - 1/2)/(acc_Bayes - 1/2)", fontsize=9
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right", handlelength=4)
    ax = axes[5]
    N = sizes[-1]
    for d in summary["decoders"]:
        _band(
            ax,
            p,
            S[N]["accuracy"][d],
            color=DECODER_COLORS[d],
            ls=STYLES[d],
            label=LABELS[d],
        )
    ax.set_title(f"all decoders, N = {N}", fontsize=9)
    ax.set_ylim(0.48, 1.01)
    ax.legend(fontsize=7, loc="lower right", handlelength=4)
    for k, ax in enumerate(axes):
        ax.set_xlabel("measurement probability $p$")
        ax.axhline(0.5, color="0.6", lw=0.7, ls="--")
        ax.text(
            0.02, 0.96, "abcdef"[k], transform=ax.transAxes, fontweight="bold", va="top"
        )
    axes[0].set_ylabel("accuracy on the charge")
    axes[3].set_ylabel("accuracy on the charge")
    axes[4].set_ylabel("g")
    fig.suptitle(
        "Charge of a monitored U(1) brickwork (new circuit per trajectory, T = N), "
        "three levels of decoders on the same trajectories",
        fontsize=10,
    )
    return fig


def draw_crossings(summary: dict[str, Any]) -> Figure:
    """Crossing ``p*`` of consecutive sizes for each decoder."""
    fig, ax = subplots(1, 1, figsize=(5.4, 4.0))
    for k, d in enumerate(summary["decoders"]):
        for pair, c in summary["crossings"][d].items():
            N1, N2 = (int(x) for x in pair.split("-"))
            x = 1 / np.sqrt(N1 * N2) + 0.002 * (k - 2)
            if c is None:
                ax.plot(x, -0.01, marker="|", ms=12, color=DECODER_COLORS[d])
                continue
            ax.plot(
                x,
                c["p"],
                marker="o",
                ms=6,
                color=DECODER_COLORS[d],
                mfc=DECODER_COLORS[d] if c["resolved"] else "none",
            )
        ax.plot([], [], marker="o", ls="", color=DECODER_COLORS[d], label=LABELS[d])
    ax.set_xlabel(r"$1/\sqrt{N N'}$")
    ax.set_ylabel("crossing $p^*$ of sizes N, N'")
    ax.set_title(
        "filled: resolved (> 2 s.e. on both sides); bar: no crossing", fontsize=8
    )
    ax.legend(fontsize=7)
    return fig


def main() -> None:
    """Draw the figures of the sweep folder given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    summary = summarize(load_runs(args.sweep_dir))
    (args.sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_figure(draw_hierarchy(summary), args.sweep_dir, "decoder_hierarchy")
    save_figure(draw_crossings(summary), args.sweep_dir, "crossings")
    for d, cross in summary["crossings"].items():
        txt = "  ".join(
            f"{k}: {c['p']:.3f}{'*' if c['resolved'] else ''}" if c else f"{k}: -"
            for k, c in cross.items()
        )
        print(f"crossing {d}: {txt}")
    print(f"Figures and summary.json in {args.sweep_dir}")


if __name__ == "__main__":
    main()
