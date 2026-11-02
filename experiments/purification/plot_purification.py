#!/usr/bin/env python3
"""Draw Fig. 1 of Gullans & Huse (2020) from a purification sweep.

Reads the run folders written by ``purification.sh`` (one per ``N``, ``p``
and seed), pools the seeds of each ``(N, p)`` (they split the trajectories
of one point) and draws four panels:

- (a) ``S_Q`` at ``t = 4N`` against ``p`` for every ``N``: the paper's
  Fig. 1(b), whose curves cross at ``p_c = 0.1598(5)``;
- (b) the data collapse against ``(p - p_c) N**(1 / nu)``, with the
  ``(p_c, nu)`` fitted on the sizes ``N >= COLLAPSE_MIN_N``
  (:func:`QELM.evaluation.finite_size_scaling.fit_collapse`), as the inset
  of Fig. 1(b);
- (c) the crossing of each pair of consecutive sizes against ``1 / N``:
  the drift of the crossing towards ``p_c``;
- (d) ``S_Q(t)`` against ``t / N`` at the ``p`` closest to ``p_c``: the
  unitary scrambling up to ``t = 2N`` keeps ``S_Q = 1``, then the
  measurements purify ``R``.

Notation (issue #235): ``N`` is the paper's ``L``, the number of system
qubits; ``L`` here is the number of trajectories.

The figure goes to ``purification_gullans_huse.png`` with its data, and the
crossings and collapses to ``fss.json``, in the sweep folder.

    uv run python experiments/purification/plot_purification.py \
        outputs/purification_gullans_huse/<date_time>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from QELM.evaluation.finite_size_scaling import (
    Curve,
    collapse_quality,
    crossing_errors,
    crossings,
    fit_collapse,
)
from QELM.utils import latest_sweep, load_sweep
from QELM.utils.plots import color_map, save_figure, subplots

KEYS = (("N", "N"), ("p", "probe.p"), ("seed", "seed"), ("L", "samples.L"))
# The paper's values, from the sizes L = 64 to 256.
P_C_PAPER = 0.1598
NU_PAPER = 1.30
SOURCE = "Gullans & Huse 2020"
# The smallest size used by the fitted collapse: smaller sizes carry large
# corrections to scaling.
COLLAPSE_MIN_N = 16
FIGURE_NAME = "purification_gullans_huse"


def pooled_curves(sweep_dir: Path) -> tuple[dict[int, Curve], dict[int, Any]]:
    """Pool the seeds of each ``(N, p)``.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.

    Returns:
        ``{N: (p, S_Q, err)}`` at the final time, and ``{N: (p, t / N,
        S_Q(t))}`` at the ``p`` closest to :data:`P_C_PAPER`.
    """
    # The steady-state runs of the same sweep (issue #257) have no S_Q.
    rows = [r for r in load_sweep(sweep_dir, keys=KEYS) if "S_Q_mean" in r]
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["N"]), float(row["p"])), []).append(row)
    points: dict[int, list[tuple[float, float, float, np.ndarray]]] = {}
    for (N, p), members in sorted(groups.items()):
        n = np.array([m["n_samples"] for m in members], dtype=float)
        mean = float(np.sum(n * [m["S_Q_mean"] for m in members]) / n.sum())
        err = float(
            np.sqrt(np.sum((n * [m["S_Q_err"] for m in members]) ** 2)) / n.sum()
        )
        S_t = np.sum([w * np.asarray(m["S_Q_t"]) for w, m in zip(n, members)], 0)
        points.setdefault(N, []).append((p, mean, err, S_t / n.sum()))
    curves: dict[int, Curve] = {}
    traces: dict[int, Any] = {}
    for N, pts in points.items():
        p = np.array([q[0] for q in pts])
        curves[N] = (p, np.array([q[1] for q in pts]), np.array([q[2] for q in pts]))
        k = int(np.argmin(np.abs(p - P_C_PAPER)))
        S_t = pts[k][3]
        traces[N] = (float(p[k]), np.arange(1, S_t.size + 1) / N, S_t)
    return curves, traces


def analyse(curves: dict[int, Curve], n_boot: int = 50) -> dict[str, Any]:
    """Crossings and collapses of the pooled curves, for ``fss.json``."""
    sizes = sorted(curves)
    pairs = crossings(curves)
    pair_errors = crossing_errors(curves, n_boot=4 * n_boot)
    out: dict[str, Any] = {
        "paper": {"p_c": P_C_PAPER, "nu": NU_PAPER},
        "crossings": {
            f"N{a}-N{b}": pairs[(a, b)] for a, b in zip(sizes, sizes[1:], strict=False)
        },
        "crossing_errors": {
            f"N{a}-N{b}": pair_errors[(a, b)]
            for a, b in zip(sizes, sizes[1:], strict=False)
        },
        "quality_at_paper_values": collapse_quality(curves, P_C_PAPER, NU_PAPER),
    }
    for name, subset in (
        ("collapse_all", sizes),
        (f"collapse_N>={COLLAPSE_MIN_N}", [N for N in sizes if N >= COLLAPSE_MIN_N]),
    ):
        if len(subset) >= 2:
            fit = fit_collapse(
                {N: curves[N] for N in subset}, P_C_PAPER, NU_PAPER, n_boot=n_boot
            )
            out[name] = {"sizes": subset, **fit.__dict__}
    return out


def draw_sweep(sweep_dir: Path, n_boot: int = 50) -> tuple[Figure, dict[str, Any]]:
    """Draw the four panels; return the open figure and the analysis.

    Args:
        sweep_dir: Folder of one sweep, holding the run folders.
        n_boot: Bootstrap resamples of the collapse errors.

    Returns:
        The figure (the notebook shows it, :func:`plot_sweep` saves and
        closes it) and the content of ``fss.json``.
    """
    curves, traces = pooled_curves(sweep_dir)
    fss = analyse(curves, n_boot=n_boot)
    sizes = sorted(curves)
    colors = color_map([f"N{N}" for N in sizes])
    collapse_key = f"collapse_N>={COLLAPSE_MIN_N}"
    fit = fss.get(collapse_key) or fss.get("collapse_all")

    fig, axes = subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for N in sizes:
        p, y, e = curves[N]
        ax.errorbar(
            p, y, yerr=e, color=colors[f"N{N}"], marker="o", capsize=3, label=f"N = {N}"
        )
    ax.axvline(
        P_C_PAPER, color="k", ls=":", lw=1, label=rf"$p_c = {P_C_PAPER}$ ({SOURCE})"
    )
    ax.set(
        xlabel="measurement probability $p$",
        ylabel=r"$\langle S_Q \rangle$ at $t = 4N$ (bits)",
        title="(a) Entropy of the reference qubit",
    )
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    if fit is not None:
        p_c, nu = fit["p_c"], fit["nu"]
        for N in sizes:
            p, y, e = curves[N]
            ax.errorbar(
                (p - p_c) * N ** (1 / nu),
                y,
                yerr=e,
                color=colors[f"N{N}"],
                marker="o",
                capsize=3,
                ls="",
                label=f"N = {N}",
                alpha=1.0 if N in fit["sizes"] else 0.35,
            )
        ax.set_title(
            "(b) Collapse, "
            rf"$p_c = {p_c:.4f} \pm {fit['p_c_err']:.4f}$, "
            rf"$\nu = {nu:.2f} \pm {fit['nu_err']:.2f}$"
        )
    ax.set(xlabel=r"$(p - p_c)\, N^{1/\nu}$", ylabel=r"$\langle S_Q \rangle$")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    pairs = [(a, b) for a, b in zip(sizes, sizes[1:], strict=False)]
    xs = [2 / (a + b) for a, b in pairs]
    ys = [fss["crossings"][f"N{a}-N{b}"] for a, b in pairs]
    es = [fss["crossing_errors"][f"N{a}-N{b}"] for a, b in pairs]
    keep = [y is not None for y in ys]
    ax.errorbar(
        np.array(xs)[keep],
        np.array(ys, dtype=float)[keep],
        yerr=np.nan_to_num(np.array(es, dtype=float)[keep]),
        marker="o",
        capsize=3,
        color="C0",
        label="crossing of consecutive sizes",
    )
    ax.axhline(P_C_PAPER, color="k", ls=":", lw=1, label=SOURCE)
    if fit is not None:
        ax.axhspan(
            fit["p_c"] - fit["p_c_err"],
            fit["p_c"] + fit["p_c_err"],
            color="C1",
            alpha=0.25,
            label="collapse (b)",
        )
    ax.set_xlim(left=0)
    ax.set(
        xlabel=r"$1 / \bar N$ of the pair",
        ylabel=r"crossing $p^*$",
        title="(c) Drift of the crossing",
    )
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    for N in sizes:
        p, t, S_t = traces[N]
        ax.plot(t, S_t, color=colors[f"N{N}"], label=f"N = {N}, p = {p:g}")
    ax.axvline(2, color="0.5", ls="--", lw=1)
    ax.set(
        xlabel=r"$t / N$ (gates per site / $N$)",
        ylabel=r"$\langle S_Q(t) \rangle$",
        title=r"(d) $S_Q(t)$ near $p_c$; measurements from $t = 2N$",
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    fig.suptitle(
        "Reference-qubit probe, Clifford brickwork, "
        "Gullans & Huse PRL 125, 070606 (2020), Fig. 1"
    )
    return fig, fss


def plot_sweep(sweep_dir: Path, n_boot: int = 50) -> dict[str, Any]:
    """Write the figure, its data and ``fss.json`` in ``sweep_dir``.

    Returns:
        The content of ``fss.json``.
    """
    fig, fss = draw_sweep(sweep_dir, n_boot=n_boot)
    save_figure(fig, sweep_dir, FIGURE_NAME, data={"fss": fss})
    plt.close(fig)
    (sweep_dir / "fss.json").write_text(json.dumps(fss, indent=2) + "\n")
    return fss


def main() -> None:
    """Parse the command line and plot one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="sweep folder; default: latest under outputs/purification_gullans_huse",
    )
    parser.add_argument("--n-boot", type=int, default=50)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(
        Path("outputs") / "purification_gullans_huse"
    )
    fss = plot_sweep(sweep_dir, n_boot=args.n_boot)
    print(json.dumps({k: v for k, v in fss.items() if k != "paper"}, indent=2))
    print(f"Figure: {sweep_dir / f'{FIGURE_NAME}.png'}")


if __name__ == "__main__":
    main()
