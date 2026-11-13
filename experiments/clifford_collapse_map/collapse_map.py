#!/usr/bin/env python3
"""Heat map of the collapse quality of Delta over the (p_c, nu) plane.

The data are the ``transition`` runs of issue #337 (Clifford brickwork of
Gullans & Huse, ``Z`` measurements with probability ``p``, depth ``T = 2N``
or ``4N``), copied in ``sources/``. The order parameter is ``Delta``, the
entropy of the references after the mid-circuit record. At ``T / N`` fixed it
is scale invariant at the transition, so the scaling form is::

    Delta(p, N) = F((p - p_c) N**(1 / nu))

with no power of ``N`` in front: the curves of all sizes cross at ``p_c``.

For each ``(p_c, nu)`` of a grid, the script computes the Houdayer-Hartmann
quality ``Q`` of the collapse (:func:`QELM.evaluation.finite_size_scaling.
collapse_quality`): every point is compared with the other sizes, linearly
interpolated at its scaled abscissa, in units of the combined error bars,
and ``Q`` is the mean squared deviation. ``Q`` is about 1 for a collapse
within the error bars. Points with ``Delta = 0`` (no error bar) are left out.

It writes, in ``--out``:

- ``collapse_map.png`` / ``.json``: the main case (``T = 2N``,
  ``N = 16`` to ``128``, ``Delta`` itself, all ``p``): the map, a zoom, and
  the collapse at the best pair;
- ``collapse_map_variants.png`` / ``.json``: the zoomed map for four other
  choices (``N >= 32``, ``log Delta``, a ``p`` window, ``T = 4N``);
- ``summary.json``: for every case, the grid minimum, the Nelder-Mead
  optimum with its bootstrap errors, and ``Q`` at the paper values.

Run from the repository root:

    uv run python experiments/clifford_collapse_map/collapse_map.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from QELM.evaluation.finite_size_scaling import (
    Curve,
    collapse_quality,
    fit_collapse,
    quality_map,
)
from QELM.utils.plots import save_figure, subplots

HERE = Path(__file__).resolve().parent
# Gullans & Huse, PRX 10, 041020 (2020), sizes L = 64 to 256.
P_C_PAPER = 0.1598
NU_PAPER = 1.30
# The wide grid and the zoom.
WIDE = (np.linspace(0.13, 0.19, 121), np.linspace(0.5, 3.0, 251))
ZOOM = (np.linspace(0.154, 0.168, 71), np.linspace(1.0, 1.7, 141))
# The guide contours, in units of the minimum quality.
LEVELS = (1.5, 2.0)


@dataclass(frozen=True)
class Case:
    """One choice of data for the collapse.

    Attributes:
        name: Key in the JSON files.
        label: Title on the figure.
        ratio: ``T / N`` of the runs.
        sizes: The ``N`` used.
        log: Collapse ``log Delta`` instead of ``Delta``.
        p_window: Only the ``p`` in this closed interval.
    """

    name: str
    label: str
    ratio: int
    sizes: tuple[int, ...]
    log: bool = False
    p_window: tuple[float, float] = (0.0, 1.0)


MAIN = Case("main", "T = 2N, N = 16-128, Delta", 2, (16, 32, 64, 128))
VARIANTS = (
    Case("no_N16", "T = 2N, N = 32-128", 2, (32, 64, 128)),
    Case("log", "T = 2N, N = 16-128, log Delta", 2, (16, 32, 64, 128), log=True),
    Case(
        "window",
        "T = 2N, N = 16-128, 0.10 <= p <= 0.22",
        2,
        (16, 32, 64, 128),
        p_window=(0.10, 0.22),
    ),  # fmt: skip
    Case("T4N", "T = 4N, N = 16-64", 4, (16, 32, 64)),
)


def load_curves(folder: Path, case: Case) -> dict[int, Curve]:
    """Read ``{N: (p, Delta, err)}`` of a case from the transition runs.

    Args:
        folder: Folder of the ``transition_N<N>_T<T>_s0.json`` files.
        case: Which runs, which ``p``, and whether to take the log.

    Returns:
        One curve per size, ``p`` increasing, the points with ``Delta = 0``
        or a zero error bar left out. With ``case.log``, the curve is
        ``log Delta`` with error ``err / Delta``.

    Raises:
        FileNotFoundError: If a run of the case is missing.
    """
    curves: dict[int, Curve] = {}
    lo, hi = case.p_window
    for n in case.sizes:
        path = folder / f"transition_N{n}_T{case.ratio * n}_s0.json"
        rows = json.loads(path.read_text())["per_p"]
        p = np.array([r["p"] for r in rows], dtype=float)
        y = np.array([r["delta"][0] for r in rows], dtype=float)
        e = np.array([r["delta"][1] for r in rows], dtype=float)
        keep = (y > 0) & (e > 0) & (p >= lo - 1e-9) & (p <= hi + 1e-9)
        p, y, e = p[keep], y[keep], e[keep]
        order = np.argsort(p)
        p, y, e = p[order], y[order], e[order]
        if case.log:
            y, e = np.log(y), e / y
        curves[n] = (p, y, e)
    return curves


def grid_minimum(q: np.ndarray, p_c: np.ndarray, nu: np.ndarray) -> dict[str, float]:
    """The ``(p_c, nu, Q)`` of the smallest finite entry of a quality map."""
    i, j = np.unravel_index(int(np.argmin(q)), q.shape)
    return {"p_c": float(p_c[i]), "nu": float(nu[j]), "quality": float(q[i, j])}


def analyse(curves: dict[int, Curve], zoom_only: bool, n_boot: int) -> dict[str, Any]:
    """Maps, grid minimum, Nelder-Mead optimum and paper-value quality."""
    out: dict[str, Any] = {}
    grids = {"zoom": ZOOM} if zoom_only else {"wide": WIDE, "zoom": ZOOM}
    for key, (p_c, nu) in grids.items():
        q = quality_map(curves, p_c, nu)
        out[key] = {"p_c": p_c, "nu": nu, "quality": q}
    out["grid_min"] = grid_minimum(out["zoom"]["quality"], *ZOOM)
    start = out["grid_min"]
    fit = fit_collapse(
        curves, start["p_c"], start["nu"], n_boot=n_boot,
        rng=np.random.default_rng(0),
    )  # fmt: skip
    out["fit"] = {
        "p_c": fit.p_c,
        "p_c_err": fit.p_c_err,
        "nu": fit.nu,
        "nu_err": fit.nu_err,
        "quality": fit.quality,
    }
    out["quality_at_paper"] = collapse_quality(curves, P_C_PAPER, NU_PAPER)
    return out


def draw_map(ax: Axes, block: dict[str, Any], result: dict[str, Any], fig: Any,
             colorbar: bool = True) -> None:  # fmt: skip
    """Heat map of ``Q`` with the guide contours, the optimum and the paper."""
    p_c, nu, q = block["p_c"], block["nu"], block["quality"]
    q_min = result["fit"]["quality"]
    finite = q[np.isfinite(q)]
    mesh = ax.pcolormesh(
        p_c, nu, q.T, shading="nearest", cmap="viridis_r",
        norm=LogNorm(vmin=q_min, vmax=float(np.percentile(finite, 98))),
        rasterized=True,
    )  # fmt: skip
    ax.contour(
        p_c, nu, q.T, levels=[q_min * f for f in LEVELS],
        colors="white", linewidths=[1.2, 0.8], linestyles=["-", "--"],
    )  # fmt: skip
    fit = result["fit"]
    ax.errorbar(
        fit["p_c"], fit["nu"], xerr=fit["p_c_err"], yerr=fit["nu_err"],
        marker="o", ms=6, color="#d62728", mec="white", ls="none",
        label=f"best: p_c = {fit['p_c']:.4f}, nu = {fit['nu']:.2f} (Q = {q_min:.1f})",
    )  # fmt: skip
    ax.plot(
        P_C_PAPER, NU_PAPER, marker="*", ms=12, color="#ff7f0e", mec="black",
        ls="none", label=f"Gullans-Huse: p_c = {P_C_PAPER}, nu = {NU_PAPER}",
    )  # fmt: skip
    ax.set_xlabel("p_c")
    ax.set_ylabel("nu")
    if colorbar:
        fig.colorbar(mesh, ax=ax, label="collapse quality Q (log scale)")


def draw_collapse(ax: Axes, curves: dict[int, Curve], p_c: float, nu: float,
                  log: bool) -> None:  # fmt: skip
    """``Delta`` against ``(p - p_c) N**(1/nu)`` for each size."""
    sizes = sorted(curves)
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
    for n, c in zip(sizes, colors, strict=False):
        p, y, e = curves[n]
        x = (p - p_c) * n ** (1 / nu)
        if log:
            y, e = np.exp(y), e * np.exp(y)
        ax.errorbar(x, y, yerr=e, marker="o", ms=3, lw=1, color=c, label=f"N = {n}")
    ax.set_yscale("log")
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel(f"(p - {p_c:.4f}) N^(1/{nu:.2f})")
    ax.set_ylabel("Delta")
    ax.legend(fontsize=8)


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """The result with the maps as lists, ``inf`` written as ``null``."""
    out: dict[str, Any] = {}
    for key, value in result.items():
        if (
            isinstance(value, dict)
            and "quality" in value
            and isinstance(value["quality"], np.ndarray)
        ):
            q = value["quality"]
            out[key] = {
                "p_c": value["p_c"].tolist(),
                "nu": value["nu"].tolist(),
                "quality": [
                    [float(v) if np.isfinite(v) else None for v in row] for row in q
                ],  # fmt: skip
            }
        else:
            out[key] = value
    return out


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {k: result[k] for k in ("grid_min", "fit", "quality_at_paper")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=HERE / "sources")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    parser.add_argument("--n-boot", type=int, default=40)
    args = parser.parse_args(argv)

    summary: dict[str, Any] = {"paper": {"p_c": P_C_PAPER, "nu": NU_PAPER}}

    curves = load_curves(args.runs, MAIN)
    main_result = analyse(curves, zoom_only=False, n_boot=args.n_boot)
    summary[MAIN.name] = {"label": MAIN.label, **_summary(main_result)}
    fig, axes = subplots(1, 3, figsize=(17, 4.8))
    draw_map(axes[0], main_result["wide"], main_result, fig)
    axes[0].set_title(f"(a) Q over (p_c, nu): {MAIN.label}", fontsize=10)
    axes[0].legend(fontsize=7, loc="upper right")
    draw_map(axes[1], main_result["zoom"], main_result, fig)
    axes[1].set_title(
        f"(b) zoom; white: Q = {LEVELS[0]} and {LEVELS[1]} x Q_min", fontsize=10
    )
    fit = main_result["fit"]
    draw_collapse(axes[2], curves, fit["p_c"], fit["nu"], MAIN.log)
    axes[2].set_title("(c) collapse at the best pair", fontsize=10)
    save_figure(fig, args.out, "collapse_map", data=_jsonable(main_result))

    fig, axes = subplots(2, 2, figsize=(12, 9))
    variants: dict[str, Any] = {}
    for ax, case in zip(axes.flat, VARIANTS, strict=True):
        c = load_curves(args.runs, case)
        result = analyse(c, zoom_only=True, n_boot=args.n_boot)
        variants[case.name] = _jsonable(result)
        summary[case.name] = {"label": case.label, **_summary(result)}
        draw_map(ax, result["zoom"], result, fig)
        ax.set_title(case.label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    save_figure(fig, args.out, "collapse_map_variants", data=variants)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, row in summary.items():
        if name == "paper":
            continue
        f = row["fit"]
        print(
            f"{name:8s} p_c = {f['p_c']:.4f} +- {f['p_c_err']:.4f}  "
            f"nu = {f['nu']:.3f} +- {f['nu_err']:.3f}  Q = {f['quality']:.2f}  "
            f"Q(paper) = {row['quality_at_paper']:.2f}"
        )


if __name__ == "__main__":
    main()
