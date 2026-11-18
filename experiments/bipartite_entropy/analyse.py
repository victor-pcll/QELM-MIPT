#!/usr/bin/env python3
"""Crossings, collapse and (p_c, nu) map of the bipartite entropy S(N/2) (#358).

Reads the runs of :mod:`run` (``<model>_N<N>_T<T>_s<seed>.json``, seeds
merged), for ``model = clifford, haar`` at ``T = 2N``. Three order
parameters of the state ``psi_m`` of the references after the full readout:

- ``S(N/2)``, the half-chain entropy (von Neumann; Renyi-2 for Haar). It
  grows as ``log N`` at ``p_c``, so the scaling form is that of Li, Chen &
  Fisher, PRB 100, 134306 (2019)::

      S(p, N) - S(p_c, N) = F((p - p_c) N**(1 / nu))

  (:func:`QELM.evaluation.finite_size_scaling.subtract_at`);
- ``I3`` of the four quarters, scale invariant at ``p_c``: the curves cross
  and ``I3(p, N) = F((p - p_c) N**(1 / nu))``;
- the profile ``S(l)``, ``l = 1 .. N/2``, and the growth exponent
  ``alpha = d log S(N/2) / d log N`` between neighbouring sizes.

The cost of a pair ``(p_c, nu)`` is the Houdayer-Hartmann quality ``Q`` of
#346 / #348.

It writes, in ``--out``: ``order.png``, ``collapse.png``,
``collapse_map.png``, ``collapse_map_variants.png``, ``profiles.png`` (each
with its ``.json``) and ``summary.json``.

    uv run python experiments/bipartite_entropy/analyse.py \
        --runs outputs/bipartite_entropy
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from QELM.evaluation.bipartite_entropy import growth_exponent
from QELM.evaluation.finite_size_scaling import (
    Curve,
    collapse_quality,
    crossing_errors,
    crossings,
    fit_collapse,
    quality_map,
    subtract_at,
)
from QELM.utils.plots import save_figure, subplots

HERE = Path(__file__).resolve().parent
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#ff7f0e", "#17becf"]
LEVELS = (1.5, 2.0)

# Reference pairs (p_c, nu) drawn on the maps.
REFERENCES: dict[str, list[tuple[str, float, float, str]]] = {
    "clifford": [
        ("Gullans & Huse (Delta)", 0.1598, 1.30, "*"),
        ("Delta, this circuit (#346)", 0.1610, 1.30, "s"),
    ],
    "haar": [
        ("Zabalo et al. (TMI, S_1)", 0.168, 1.2, "*"),
        ("Zabalo et al. (S_2)", 0.162, 1.3, "P"),
        ("Delta, this circuit (#348)", 0.1805, 1.59, "s"),
    ],
}
LITERATURE_PC = {"clifford": (0.1598, 0.0001), "haar": (0.168, 0.005)}
WIDE = {"clifford": (0.10, 0.24), "haar": (0.08, 0.30)}
NU_WIDE = (0.4, 3.0)


@dataclass(frozen=True)
class Case:
    """One order parameter and one choice of data.

    Attributes:
        name: Key in the JSON files.
        label: Title on the figures.
        key: ``half_s1``, ``half_s2``, ``i3`` or ``i3_2`` (per-sample field).
        subtract: Collapse ``O(p, N) - O(p_c, N)``.
        min_n: Smallest size kept.
        p_window: Only the ``p`` in this closed interval.
    """

    name: str
    label: str
    key: str
    subtract: bool
    min_n: int = 0
    p_window: tuple[float, float] = (0.0, 1.0)


# Window of p for the S(N/2) collapse: the Li-Chen-Fisher form holds near
# p_c only. Deep in the volume law S(N/2) ~ s(p) N, and the error bars there
# are tiny, so a full-range fit is driven by corrections far from p_c.
S_WINDOW = {"clifford": (0.10, 0.24), "haar": (0.10, 0.30)}


def cases(model: str, sizes: list[int]) -> tuple[list[Case], list[Case]]:
    """Main cases (S(N/2), I3) and variants of a model."""
    second = sorted(sizes)[1] if len(sizes) > 1 else 0
    lo, hi = S_WINDOW[model]
    win = f"{lo:.2f} <= p <= {hi:.2f}"
    main = [
        Case("S1", f"S(N/2) - S(p_c, N), {win}", "half_s1", True, p_window=(lo, hi)),
        Case("I3", "I3 of the quarters", "i3", False),
    ]
    variants = [
        Case(
            "S1_drop",
            f"S(N/2), {win}, N >= {second}",
            "half_s1",
            True,
            min_n=second,
            p_window=(lo, hi),
        ),
        Case("S1_full", "S(N/2), every p", "half_s1", True),
    ]
    if model == "haar":
        variants += [
            Case("S2", f"S_2(N/2), {win}", "half_s2", True, p_window=(lo, hi)),
            Case("I3_2", "I3 of S_2", "i3_2", False),
        ]
    else:
        quarter = sorted(n for n in sizes if n % 4 == 0)
        drop = quarter[1] if len(quarter) > 1 else 0
        variants += [
            Case("I3_drop", f"I3, N >= {drop}", "i3", False, min_n=drop),
            Case(
                "I3_window", "I3, 0.10 <= p <= 0.24", "i3", False, p_window=(0.10, 0.24)
            ),
        ]
    return main, variants


def load_runs(folder: Path, model: str) -> dict[int, dict[str, Any]]:
    """Merge the seeds of every size at ``T = 2N``: ``{N: run}``.

    The merged run has, per ``p``, the concatenated per-sample lists and the
    profile mean and error weighted by the number of samples.
    """
    merged: dict[int, dict[str, Any]] = {}
    pattern = re.compile(rf"{model}_N(\d+)_T(\d+)_s\d+\.json")
    for path in sorted(folder.glob(f"{model}_N*_T*_s*.json")):
        m = pattern.fullmatch(path.name)
        if m is None or int(m.group(2)) != 2 * int(m.group(1)):
            continue
        run = json.loads(path.read_text())
        n = int(run["n"])
        slot = merged.setdefault(n, {"n": n, "per_p": {}})
        for row in run["per_p"]:
            acc = slot["per_p"].setdefault(row["p"], {"rows": []})
            acc["rows"].append(row)
    out: dict[int, dict[str, Any]] = {}
    for n, slot in sorted(merged.items()):
        per_p = []
        for p, acc in sorted(slot["per_p"].items()):
            rows = acc["rows"]
            counts = np.array([len(r["half_s1"]) for r in rows], dtype=float)
            total = counts.sum()
            row: dict[str, Any] = {"p": p, "count": int(total)}
            for key in ("s1", "s2"):
                mean = np.array([r[key] for r in rows])
                err = np.array([r[f"{key}_err"] for r in rows])
                row[key] = (counts[:, None] * mean).sum(axis=0) / total
                row[f"{key}_err"] = (
                    np.sqrt(((counts[:, None] * err) ** 2).sum(axis=0)) / total
                )
            for key in ("half_s1", "half_s2", "i3", "i3_2"):
                row[key] = np.concatenate(
                    [np.asarray(r[key], dtype=float) for r in rows]
                )
            per_p.append(row)
        out[n] = {"n": n, "per_p": per_p}
    return out


def curves_of(runs: dict[int, dict[str, Any]], case: Case) -> dict[int, Curve]:
    """``{N: (p, mean, error)}`` of the per-sample field of a case."""
    curves: dict[int, Curve] = {}
    lo, hi = case.p_window
    for n, run in runs.items():
        if n < case.min_n:
            continue
        rows = [r for r in run["per_p"] if lo - 1e-9 <= r["p"] <= hi + 1e-9]
        rows = [r for r in rows if np.all(np.isfinite(r[case.key]))]
        if len(rows) < 3:
            continue
        p = np.array([r["p"] for r in rows])
        y = np.array([np.mean(r[case.key]) for r in rows])
        e = np.array(
            [np.std(r[case.key], ddof=1) / np.sqrt(len(r[case.key])) for r in rows]
        )
        keep = e > 0
        curves[n] = (p[keep], y[keep], e[keep])
    return curves


def grid_minimum(q: np.ndarray, p_c: np.ndarray, nu: np.ndarray) -> dict[str, float]:
    """The ``(p_c, nu, Q)`` of the smallest finite entry of a quality map."""
    i, j = np.unravel_index(
        int(np.nanargmin(np.where(np.isfinite(q), q, np.nan))), q.shape
    )
    return {"p_c": float(p_c[i]), "nu": float(nu[j]), "quality": float(q[i, j])}


def pair_crossings(curves: dict[int, Curve], n_boot: int) -> list[dict[str, Any]]:
    """Crossings of every pair of sizes with their bootstrap errors."""
    cross = crossings(curves)
    errs = crossing_errors(curves, n_boot=n_boot, rng=np.random.default_rng(0))
    return [
        {"sizes": list(pair), "p": value, "err": errs.get(pair, float("nan"))}
        for pair, value in cross.items()
    ]


def analyse(
    model: str, curves: dict[int, Curve], case: Case, n_boot: int
) -> dict[str, Any]:
    """Wide and zoomed maps, optimum with bootstrap errors, crossings."""
    out: dict[str, Any] = {"sizes": sorted(curves)}
    wide = (np.linspace(*WIDE[model], 71), np.linspace(*NU_WIDE, 66))
    q_wide = quality_map(curves, *wide, subtract=case.subtract)
    out["wide"] = {"p_c": wide[0], "nu": wide[1], "quality": q_wide}
    start = grid_minimum(q_wide, *wide)
    zoom = (
        np.linspace(start["p_c"] - 0.025, start["p_c"] + 0.025, 51),
        np.linspace(max(0.4, start["nu"] - 0.9), start["nu"] + 0.9, 61),
    )
    out["zoom"] = {
        "p_c": zoom[0],
        "nu": zoom[1],
        "quality": quality_map(curves, *zoom, subtract=case.subtract),
    }
    g = grid_minimum(out["zoom"]["quality"], *zoom)
    out["grid_min"] = g
    fit = fit_collapse(
        curves,
        g["p_c"],
        g["nu"],
        n_boot=n_boot,
        rng=np.random.default_rng(0),
        subtract=case.subtract,
    )
    out["fit"] = {
        "p_c": fit.p_c,
        "p_c_err": fit.p_c_err,
        "nu": fit.nu,
        "nu_err": fit.nu_err,
        "quality": fit.quality,
    }
    out["quality_at_references"] = {
        label: collapse_quality(curves, pc, nu, case.subtract)
        for label, pc, nu, _ in REFERENCES[model]
    }
    out["crossings"] = pair_crossings(curves, n_boot=200)
    return out


def neighbour(sizes: list[int], row: dict[str, Any]) -> bool:
    a, b = row["sizes"]
    return sizes.index(b) == sizes.index(a) + 1


def draw_curves(
    ax: Axes,
    curves: dict[int, Curve],
    cross: list[dict[str, Any]],
    model: str,
    ylabel: str,
) -> None:
    """The order parameter against ``p`` for each size, neighbour crossings marked."""
    for n, c in zip(curves, COLORS, strict=False):
        p, y, e = curves[n]
        ax.errorbar(p, y, yerr=e, marker="o", ms=3, lw=1, color=c, label=f"N = {n}")
    sizes = sorted(curves)
    for row in cross:
        if row["p"] is None or not neighbour(sizes, row):
            continue
        a = row["sizes"][0]
        ax.plot(
            row["p"],
            np.interp(row["p"], curves[a][0], curves[a][1]),
            "kx",
            ms=8,
            mew=1.5,
        )
    pc, err = LITERATURE_PC[model]
    ax.axvspan(
        pc - max(err, 0.002),
        pc + max(err, 0.002),
        color="#ff7f0e",
        alpha=0.25,
        label="p_c, literature",
    )
    ax.set_xlabel("p")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)


def draw_drift(ax: Axes, results: dict[str, dict[str, Any]], model: str) -> None:
    """Crossing of neighbouring sizes against ``1 / mean(N)``."""
    for (label, result), c in zip(results.items(), COLORS, strict=False):
        sizes = result["sizes"]
        rows = [
            r for r in result["crossings"] if r["p"] is not None and neighbour(sizes, r)
        ]
        ax.errorbar(
            [2 / sum(r["sizes"]) for r in rows],
            [r["p"] for r in rows],
            yerr=[r["err"] for r in rows],
            marker="o",
            ms=5,
            lw=1,
            color=c,
            label=label,
        )
    pc, err = LITERATURE_PC[model]
    ax.axhspan(
        pc - max(err, 0.002),
        pc + max(err, 0.002),
        color="#ff7f0e",
        alpha=0.25,
        label="p_c, literature",
    )
    ax.set_xlim(left=0)
    ax.set_xlabel("1 / mean(N_a, N_b), neighbouring sizes")
    ax.set_ylabel("crossing p")
    ax.legend(fontsize=7)


def draw_collapse(
    ax: Axes, curves: dict[int, Curve], case: Case, result: dict[str, Any]
) -> None:
    """The order parameter against ``(p - p_c) N**(1/nu)`` at the best pair."""
    fit = result["fit"]
    pc, nu = fit["p_c"], fit["nu"]
    data = subtract_at(curves, pc) if case.subtract else curves
    for n, c in zip(data, COLORS, strict=False):
        p, y, e = data[n]
        ax.errorbar(
            (p - pc) * n ** (1 / nu),
            y,
            yerr=e,
            marker="o",
            ms=3,
            lw=1,
            color=c,
            label=f"N = {n}",
        )
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel(f"(p - {pc:.4f}) N^(1/{nu:.2f})")
    ax.set_ylabel(case.label)
    ax.set_title(
        f"{case.label}: p_c = {pc:.4f}, nu = {nu:.2f}, Q = {fit['quality']:.1f}",
        fontsize=9,
    )
    ax.legend(fontsize=7)


def draw_map(
    ax: Axes, block: dict[str, Any], result: dict[str, Any], fig: Any, model: str
) -> None:
    """Heat map of ``Q`` with the guide contours, the optimum and the references."""
    p_c, nu, q = block["p_c"], block["nu"], block["quality"]
    q_min = result["fit"]["quality"]
    finite = q[np.isfinite(q)]
    mesh = ax.pcolormesh(
        p_c,
        nu,
        q.T,
        shading="nearest",
        cmap="viridis_r",
        norm=LogNorm(vmin=max(q_min, 1e-3), vmax=float(np.percentile(finite, 98))),
        rasterized=True,
    )
    ax.contour(
        p_c,
        nu,
        q.T,
        levels=[q_min * f for f in LEVELS],
        colors="white",
        linewidths=[1.2, 0.8],
        linestyles=["-", "--"],
    )
    fit = result["fit"]
    ax.errorbar(
        fit["p_c"],
        fit["nu"],
        xerr=fit["p_c_err"],
        yerr=fit["nu_err"],
        marker="o",
        ms=6,
        color="#d62728",
        mec="white",
        ls="none",
        label=f"best: {fit['p_c']:.4f}, {fit['nu']:.2f} (Q = {q_min:.1f})",
    )
    for label, pc, nv, marker in REFERENCES[model]:
        if p_c[0] <= pc <= p_c[-1] and nu[0] <= nv <= nu[-1]:
            ax.plot(
                pc,
                nv,
                marker=marker,
                ms=9,
                color="#ff7f0e",
                mec="black",
                ls="none",
                label=f"{label}: {pc}, {nv}",
            )
    ax.set_xlim(p_c[0], p_c[-1])
    ax.set_ylim(nu[0], nu[-1])
    ax.set_xlabel("p_c")
    ax.set_ylabel("nu")
    ax.legend(fontsize=6, loc="upper right")
    fig.colorbar(mesh, ax=ax, label="Q (log scale)")


def draw_profiles(ax: Axes, run: dict[str, Any], p_show: list[float]) -> None:
    """``S(l)`` against ``l`` at the largest size, for several ``p``."""
    cmap = __import__("matplotlib").colormaps["coolwarm"]
    rows = [r for r in run["per_p"] if any(abs(r["p"] - q) < 1e-9 for q in p_show)]
    for k, r in enumerate(rows):
        ell = np.arange(len(r["s1"]))
        ax.errorbar(
            ell[1:],
            r["s1"][1:],
            yerr=r["s1_err"][1:],
            marker="o",
            ms=2.5,
            lw=1,
            color=cmap(k / max(1, len(rows) - 1)),
            label=f"p = {r['p']}",
        )
    ax.set_xlabel("l")
    ax.set_ylabel("S(l), bits")
    ax.legend(fontsize=6, ncol=2)


def draw_alpha(ax: Axes, runs: dict[int, dict[str, Any]], model: str) -> dict[str, Any]:
    """``alpha(p) = d log S(N/2) / d log N`` between neighbouring sizes."""
    sizes = sorted(runs)
    p_all = [set(r["p"] for r in runs[n]["per_p"]) for n in sizes]
    common = sorted(set.intersection(*p_all))
    values = np.array(
        [
            [
                np.mean(next(r for r in runs[n]["per_p"] if r["p"] == p)["half_s1"])
                for n in sizes
            ]
            for p in common
        ]
    )
    out: dict[str, Any] = {"p": common, "sizes": sizes, "pairs": []}
    alphas = []
    for v in values:
        mid, alpha = growth_exponent(sizes, v)
        alphas.append(alpha)
    alphas_arr = np.array(alphas)
    for k in range(len(sizes) - 1):
        label = f"{sizes[k]} -> {sizes[k + 1]}"
        ax.plot(
            common,
            alphas_arr[:, k],
            marker="o",
            ms=3,
            lw=1,
            color=COLORS[k % len(COLORS)],
            label=label,
        )
        out["pairs"].append({"sizes": sizes[k : k + 2], "alpha": alphas_arr[:, k]})
    pc, err = LITERATURE_PC[model]
    ax.axvspan(pc - max(err, 0.002), pc + max(err, 0.002), color="#ff7f0e", alpha=0.25)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axhline(1, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("p")
    ax.set_ylabel("alpha = d log S(N/2) / d log N")
    ax.legend(fontsize=6)
    return out


def _jsonable(value: Any) -> Any:
    """Arrays as lists, non-finite numbers as ``null``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("sizes", "grid_min", "fit", "quality_at_references", "crossings")
    return {k: _jsonable(result[k]) for k in keys}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("outputs/bipartite_entropy"))
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    parser.add_argument("--n-boot", type=int, default=40)
    args = parser.parse_args(argv)

    models = [m for m in ("clifford", "haar") if load_runs(args.runs, m)]
    runs = {m: load_runs(args.runs, m) for m in models}
    summary: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    variant_results: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, dict[int, Curve]]] = {}
    for m in models:
        main_cases, variants = cases(m, sorted(runs[m]))
        results[m], variant_results[m], curves[m] = {}, {}, {}
        summary[m] = {}
        for case in main_cases + variants:
            c = curves_of(runs[m], case)
            if len(c) < 2:
                continue
            res = analyse(m, c, case, args.n_boot)
            curves[m][case.name] = c
            (results if case in main_cases else variant_results)[m][case.name] = res
            summary[m][case.name] = {"label": case.label, **_summary(res)}
            f = res["fit"]
            print(
                f"{m:8s} {case.name:10s} p_c = {f['p_c']:.4f} +- {f['p_c_err']:.4f}  "
                f"nu = {f['nu']:.3f} +- {f['nu_err']:.3f}  Q = {f['quality']:.2f}  "
                f"N = {res['sizes']}"
            )
        raw = Case("S1_raw", "S(N/2)", "half_s1", False)
        c_raw = curves_of(runs[m], raw)
        curves[m]["S1_raw"] = c_raw
        summary[m]["S1_raw_crossings"] = _jsonable(pair_crossings(c_raw, n_boot=200))

    by_name = {
        (m, c.name): c for m in models for c in sum(cases(m, sorted(runs[m])), [])
    }

    # Figure 1: S(N/2) and I3 against p, crossing drift.
    fig, axes = subplots(len(models), 3, figsize=(17, 4.6 * len(models)), squeeze=False)
    for row, m in zip(axes, models, strict=True):
        draw_curves(
            row[0],
            curves[m]["S1_raw"],
            summary[m]["S1_raw_crossings"],
            m,
            "S(N/2), bits",
        )
        row[0].set_title(
            f"(a) {m}: S(N/2) at T = 2N; x: crossing of neighbouring N", fontsize=9
        )
        if "I3" in results[m]:
            draw_curves(
                row[1],
                curves[m]["I3"],
                results[m]["I3"]["crossings"],
                m,
                "I3 of the quarters, bits",
            )
        row[1].set_title(f"(b) {m}: tripartite information I3", fontsize=9)
        drift = {"I3": results[m]["I3"]} if "I3" in results[m] else {}
        drift["S(N/2)"] = {
            "sizes": sorted(curves[m]["S1_raw"]),
            "crossings": summary[m]["S1_raw_crossings"],
        }
        draw_drift(row[2], drift, m)
        row[2].set_title(f"(c) {m}: drift of the crossings with N", fontsize=9)
    save_figure(
        fig,
        args.out,
        "order",
        data=_jsonable(
            {
                m: {k: curves[m][k] for k in ("S1_raw", "I3") if k in curves[m]}
                for m in models
            }
        ),
    )

    # Figure 2: collapses at the best pair.
    fig, axes = subplots(len(models), 2, figsize=(12, 4.6 * len(models)), squeeze=False)
    for row, m in zip(axes, models, strict=True):
        for ax, name in zip(row, ("S1", "I3"), strict=True):
            if name in results[m]:
                draw_collapse(ax, curves[m][name], by_name[(m, name)], results[m][name])
                ax.set_title(f"{m}: " + ax.get_title(), fontsize=9)
    save_figure(
        fig,
        args.out,
        "collapse",
        data=_jsonable(
            {m: {k: _summary(v) for k, v in results[m].items()} for m in models}
        ),
    )

    # Figure 3: quality maps (wide, zoom) for S(N/2) and I3.
    fig, axes = subplots(len(models), 4, figsize=(24, 5 * len(models)), squeeze=False)
    for row, m in zip(axes, models, strict=True):
        for k, name in enumerate(("S1", "I3")):
            if name not in results[m]:
                continue
            res = results[m][name]
            sizes = ", ".join(map(str, res["sizes"]))
            draw_map(row[2 * k], res["wide"], res, fig, m)
            row[2 * k].set_title(
                f"{m}, {by_name[(m, name)].label}, N = {sizes}", fontsize=9
            )
            draw_map(row[2 * k + 1], res["zoom"], res, fig, m)
            row[2 * k + 1].set_title(
                f"zoom; white: Q = {LEVELS[0]}, {LEVELS[1]} x Q_min", fontsize=9
            )
    save_figure(fig, args.out, "collapse_map", data=_jsonable(results))

    # Figure 4: variants (zoomed maps).
    fig, axes = subplots(len(models), 4, figsize=(24, 5 * len(models)), squeeze=False)
    for row, m in zip(axes, models, strict=True):
        for ax in row:
            ax.set_visible(False)
        for ax, (name, res) in zip(row, variant_results[m].items(), strict=False):
            ax.set_visible(True)
            draw_map(ax, res["zoom"], res, fig, m)
            ax.set_title(
                f"{m}, {by_name[(m, name)].label}, "
                f"N = {', '.join(map(str, res['sizes']))}",
                fontsize=9,
            )
    save_figure(fig, args.out, "collapse_map_variants", data=_jsonable(variant_results))

    # Figure 5: profiles and growth exponent.
    fig, axes = subplots(len(models), 2, figsize=(12, 4.6 * len(models)), squeeze=False)
    alpha_data: dict[str, Any] = {}
    for row, m in zip(axes, models, strict=True):
        big = max(runs[m])
        p_show = [0.04, 0.1, 0.14, 0.16, 0.18, 0.22, 0.3, 0.4]
        draw_profiles(row[0], runs[m][big], p_show)
        row[0].set_title(f"(a) {m}: S(l), l = 1 .. N/2, N = {big}", fontsize=9)
        alpha_data[m] = draw_alpha(row[1], runs[m], m)
        row[1].set_title(f"(b) {m}: growth exponent of S(N/2)", fontsize=9)
        summary[m]["alpha"] = _jsonable(alpha_data[m])
        summary[m]["profiles"] = _jsonable(
            {
                str(n): [
                    {
                        "p": r["p"],
                        "s1": r["s1"],
                        "s1_err": r["s1_err"],
                        "count": r["count"],
                    }
                    for r in run["per_p"]
                ]
                for n, run in runs[m].items()
            }
        )
    save_figure(fig, args.out, "profiles", data=_jsonable(alpha_data))

    (args.out / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
