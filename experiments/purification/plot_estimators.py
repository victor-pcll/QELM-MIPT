#!/usr/bin/env python3
"""Compare the estimators of p_c on one sweep of the Clifford brickwork (#257).

Reads the run folders written by ``purification.sh`` with
``MODES="reference steady"``: the reference-qubit runs (``S_Q``) and the
steady-state runs (``S_half``, ``I3``) of the same circuit, step and
boundaries. Three estimators of the critical point:

- ``S_Q``: crossing of the reference-qubit entropy (Gullans & Huse 2020),
  and its collapse ``S_Q = F((p - p_c) N**(1/nu))``;
- ``I3``: crossing of the tripartite mutual information (Zabalo et al. 2020),
  and its collapse, with the same form;
- ``S_half``: collapse of ``S_half(p, N) - S_half(p_c, N)`` (Li, Chen &
  Fisher 2019), the half-chain entropy growing like ``log N`` at ``p_c``.

Collapses use the sizes ``N >= COLLAPSE_MIN_N``. The figure
``pc_estimators.png`` (and ``.json``) shows the three order parameters
against ``p`` and, in the last panel, every estimate of ``p_c``: the
crossings of consecutive sizes against ``1 / N`` and the collapses as bands.
``estimators.json`` holds the numbers.

    uv run python experiments/purification/plot_estimators.py \
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
    crossing_errors,
    crossings,
    fit_collapse,
)
from QELM.utils import latest_sweep, load_sweep
from QELM.utils.plots import color_map, save_figure, subplots

KEYS = (
    ("N", "N"),
    ("p", "probe.p"),
    ("seed", "seed"),
    ("dynamics", "probe.dynamics"),
)
# Literature values: Clifford (Li, Chen & Fisher 2019; Gullans & Huse 2020)
# and Haar von Neumann (Zabalo et al. 2020, from I_3 and purification).
P_C_LIT = {"clifford": 0.16, "haar": 0.17}
NU_LIT = 1.3
COLLAPSE_MIN_N = 16
# metric: (label, axis label, has a size-independent crossing, subtract O(p_c))
ESTIMATORS = {
    "S_Q": (
        "purification $S_Q$",
        r"$\langle S_Q \rangle$ at $t = 4N$ (bits)",
        True,
        False,
    ),
    "I3": ("tripartite $I_3$", r"$\langle I_3 \rangle$ (bits)", True, False),
    "S_half": (
        "half-chain $S_A$",
        r"$\langle S_{N/2} \rangle$ (bits)",
        False,
        True,
    ),
    # Renyi-2 versions, written by the Haar runs only (equal for Clifford).
    "S_Q2": ("purification $S_Q^{(2)}$", "", True, False),
    "I3_2": ("tripartite $I_3^{(2)}$", "", True, False),
    "S_half2": ("half-chain $S_A^{(2)}$", "", False, True),
}
# The order parameters drawn against p, one panel each.
PANELS = ("S_Q", "I3", "S_half")
COLORS = {
    "S_Q": "C0",
    "I3": "C1",
    "S_half": "C2",
    "S_Q2": "C3",
    "I3_2": "C4",
    "S_half2": "C5",
}
FIGURE_NAME = "pc_estimators"


def pooled(rows: list[dict[str, Any]], metric: str) -> dict[int, Curve]:
    """Pool the seeds of each ``(N, p)`` for one metric.

    Args:
        rows: Output of :func:`QELM.utils.load_sweep`.
        metric: ``S_Q``, ``I3`` or ``S_half``; ``<metric>_mean`` and
            ``<metric>_err`` are read, rows without them are skipped.

    Returns:
        ``{N: (p, mean, err)}``, ``p`` increasing.
    """
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        if f"{metric}_mean" in row:
            groups.setdefault((int(row["N"]), float(row["p"])), []).append(row)
    points: dict[int, list[tuple[float, float, float]]] = {}
    for (N, p), members in sorted(groups.items()):
        n = np.array([m["n_samples"] for m in members], dtype=float)
        mean = np.array([m[f"{metric}_mean"] for m in members])
        err = np.array([m[f"{metric}_err"] for m in members])
        points.setdefault(N, []).append(
            (
                p,
                float(n @ mean / n.sum()),
                float(np.sqrt(np.sum((n * err) ** 2)) / n.sum()),
            )
        )
    return {
        N: tuple(np.array([q[k] for q in pts]) for k in range(3))  # type: ignore[misc]
        for N, pts in points.items()
    }


def analyse(
    curves: dict[str, dict[int, Curve]], p_lit: float = 0.16, n_boot: int = 50
) -> dict[str, Any]:
    """Crossings and collapses of each estimator, for ``estimators.json``.

    ``p_lit`` is the literature value, the start of the collapse fits.
    """
    out: dict[str, Any] = {"literature": {"p_c": p_lit, "nu": NU_LIT}}
    for metric, data in curves.items():
        _, _, crosses, subtract = ESTIMATORS[metric]
        sizes = sorted(data)
        entry: dict[str, Any] = {"sizes": sizes}
        if crosses and len(sizes) >= 2:
            pairs = crossings(data)
            errors = crossing_errors(data, n_boot=4 * n_boot)
            entry["crossings"] = {
                f"N{a}-N{b}": [pairs[(a, b)], errors[(a, b)]]
                for a, b in zip(sizes, sizes[1:], strict=False)
            }
        large = {N: c for N, c in data.items() if N >= COLLAPSE_MIN_N}
        subset = large if len(large) >= 2 else data
        if len(subset) >= 2:
            fit = fit_collapse(subset, p_lit, NU_LIT, n_boot=n_boot, subtract=subtract)
            entry["collapse"] = {"sizes": sorted(subset), **fit.__dict__}
        out[metric] = entry
    return out


def draw_sweep(sweep_dir: Path, n_boot: int = 50) -> tuple[Figure, dict[str, Any]]:
    """Draw the order parameters and the ``p_c`` of every estimator.

    Returns:
        The open figure and the content of ``estimators.json``.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    curves = {m: pooled(rows, m) for m in ESTIMATORS}
    curves = {m: c for m, c in curves.items() if c}
    dynamics = sorted({str(r.get("dynamics") or "clifford") for r in rows})
    p_lit = P_C_LIT.get(dynamics[0], 0.16)
    result = analyse(curves, p_lit=p_lit, n_boot=n_boot)
    all_sizes = sorted({N for c in curves.values() for N in c})
    colors = color_map([f"N{N}" for N in all_sizes])
    fig, axes = subplots(2, 2, figsize=(13, 10))

    for ax, metric in zip(axes.flat[:3], PANELS, strict=True):
        label, ylabel, _, _ = ESTIMATORS[metric]
        for N, (p, y, e) in sorted(curves.get(metric, {}).items()):
            ax.errorbar(
                p,
                y,
                yerr=e,
                color=colors[f"N{N}"],
                marker="o",
                capsize=3,
                label=f"N = {N}",
            )
        ax.axvline(p_lit, color="k", ls=":", lw=1)
        ax.set(xlabel="measurement probability $p$", ylabel=ylabel, title=label)
        if metric == "S_half":
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    for metric, entry in result.items():
        if metric == "literature":
            continue
        color = COLORS[metric]
        label = ESTIMATORS[metric][0]
        if "crossings" in entry:
            sizes = entry["sizes"]
            pairs = list(zip(sizes, sizes[1:], strict=False))
            x = np.array([2 / (a + b) for a, b in pairs])
            vals = [entry["crossings"][f"N{a}-N{b}"] for a, b in pairs]
            keep = np.array([v[0] is not None for v in vals])
            y = np.array([v[0] if v[0] is not None else np.nan for v in vals], float)
            e = np.nan_to_num(np.array([v[1] for v in vals], float))
            ax.errorbar(
                x[keep],
                y[keep],
                yerr=e[keep],
                color=color,
                marker="o",
                capsize=3,
                label=f"{label}, crossing",
            )
        if "collapse" in entry:
            c = entry["collapse"]
            ax.axhspan(
                c["p_c"] - c["p_c_err"],
                c["p_c"] + c["p_c_err"],
                color=color,
                alpha=0.25,
                label=rf"{label}, collapse: $p_c = {c['p_c']:.4f}$, "
                rf"$\nu = {c['nu']:.2f}$",
            )
            ax.axhline(c["p_c"], color=color, lw=1)
    ax.axhline(p_lit, color="k", ls=":", lw=1, label=f"literature {p_lit}")
    ax.set_xlim(left=0)
    ax.set(
        xlabel=r"$1 / \bar N$ of the pair",
        ylabel=r"$p_c$ estimate",
        title="Estimates of $p_c$",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        f"Estimators of $p_c$, {' + '.join(dynamics)} brickwork, half-layer "
        "steps, periodic chain"
    )
    return fig, result


def plot_sweep(sweep_dir: Path, n_boot: int = 50) -> dict[str, Any]:
    """Write the figure, its data and ``estimators.json`` in ``sweep_dir``."""
    fig, result = draw_sweep(sweep_dir, n_boot=n_boot)
    save_figure(fig, sweep_dir, FIGURE_NAME, data={"estimators": result})
    plt.close(fig)
    (sweep_dir / "estimators.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Parse the command line and plot one sweep."""
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dir", nargs="?", type=Path)
    parser.add_argument("--n-boot", type=int, default=50)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or latest_sweep(
        Path("outputs") / "purification_gullans_huse"
    )
    print(json.dumps(plot_sweep(sweep_dir, n_boot=args.n_boot), indent=2))
    print(f"Figure: {sweep_dir / f'{FIGURE_NAME}.png'}")


if __name__ == "__main__":
    main()
