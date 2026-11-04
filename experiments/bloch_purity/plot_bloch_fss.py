#!/usr/bin/env python3
"""Finite-size comparison of the tomography task (issue #282).

Reads the ``N<N>_seed<seed>/metrics.json`` of one or more sweeps of
``bloch_purity.sh`` with several ``N`` (one fixed circuit per seed) and
overlays the sizes on one figure, ``bloch_fss.png``:

- (a) ``R2`` of the QELM on M against the exact Bloch vector ``r(M)``;
- (b) ``R2`` of the QELM on M against single-shot labels, divided by the
  Bayes ceiling of the same ``p`` (the fraction of the learnable signal the
  QELM reads);
- (c) mean absolute error of the predicted purity ``Tr(rho_R^2)`` (the same
  for every readout of the raw record, which reads nothing of the purity);
- (d) the purification curve, the mean and the spread of the exact purity.

If the minimum of (a) or the peak of (c) came from the measurement-induced
transition, it would sharpen with ``N`` and the curves of different ``N``
would cross near ``p_c``. To test this, ``fss_summary.json`` holds, for each
``N``, the position of the minimum of (a) and of the peak of (c) (a parabola
through the three points around the extremum of the averaged curve), and,
for each pair of sizes, the ``p`` where the curves cross (a sign change of
the difference of the averaged curves, linearly interpolated). The same is
done for the spread ``purity_std`` (its maximum) and the purification curve
``purity_mean`` (its crossings), the physical quantities behind panel (c): a
readout that reads nothing of the purity has an error equal to the mean
absolute deviation of the purity. Every value comes with a bootstrap over
the circuits (``--n-boot`` resamplings of the seeds of each ``N``, median
and 16-84 % interval).

When the runs have the variance probe (``var_purity_*``),
``purity_variance.png`` draws the variance over the trajectories of the
purity of ``R``, exact (the oracle) and predicted by the QELM, and its peak
(position and height per ``N``) is in ``fss_summary.json``: the decoded
analogue of the charge sharpening of #137.

When the runs have the nonlinear diagnostics
(``bloch_task.nonlinear_readouts=true``), ``bloch_nonlinear.png`` compares,
for each ``N``, the linear QELM on M, the ridge and the MLP on ``[M, |M|]``
and the Bayes decoder.

    uv run python experiments/bloch_purity/plot_bloch_fss.py \\
        outputs/bloch_purity/<date_time> --out figs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# Range of p where the minimum of the Bloch R2 and the peak of the purity
# error are looked for (p = 1 is a second, trivial maximum of the R2).
EXTREMUM_RANGE = (0.05, 0.5)
# Range of p where the crossings of two sizes are looked for.
CROSSING_RANGE = (0.05, 0.5)
# Panels of the figure: (metric, readout, y label). ``ratio`` divides the
# readout by the Bayes ceiling of the same p.
PANELS: tuple[tuple[str, str, str], ...] = (
    ("r2_bloch", "M", "R2 of the QELM on M, exact r(M)"),
    ("r2_shots_ratio", "M", "R2 on single shots / Bayes ceiling"),
    ("mae_purity", "M", "mean |Tr(rho^2)_pred - Tr(rho^2)|"),
    ("purity_mean", "", "exact purity Tr(rho_R^2)"),
)
# Extremum looked for in each metric: min or max.
# (The ratio of panel (b) rises with p and has no extremum.)
EXTREMA = {
    "r2_bloch": "min",
    "mae_purity": "max",
    "purity_std": "max",
    "var_oracle": "max",
    "var_qelm_M": "max",
    "var_qelm_M_abs": "max",
}
# Series analysed: name -> (metric, readout) of :func:`curves`. The variance
# probe (runs with ``var_purity_*``): the oracle variance over the
# trajectories of the exact purity, the variance of the purity predicted by
# the QELM on M (from its Bloch vector, the ridge on the purity itself being
# constant on the raw record) and by the ridge on [M, |M|] (direct), and
# their fractions of the oracle variance.
SERIES: dict[str, tuple[str, str]] = {
    "r2_bloch": ("r2_bloch", "M"),
    "mae_purity": ("mae_purity", "M"),
    "purity_std": ("purity_std", ""),
    "r2_shots_ratio": ("r2_shots_ratio", "M"),
    "purity_mean": ("purity_mean", ""),
    "var_oracle": ("var_purity", "bayes"),
    "var_qelm_M": ("var_purity_from_bloch", "M"),
    "var_qelm_M_abs": ("var_purity", "M_abs"),
    "var_fraction_M": ("var_fraction_from_bloch", "M"),
    "var_fraction_M_abs": ("var_fraction", "M_abs"),
}
# Series whose crossings between sizes are looked for (``purity_mean`` is
# the purification order parameter: it has no extremum).
CROSSED = tuple(SERIES)


def _plot_module() -> ModuleType:
    """``plot_bloch_purity.py``: loading of the runs."""
    path = HERE / "plot_bloch_purity.py"
    spec = importlib.util.spec_from_file_location("plot_bloch_purity_for_fss", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PB = _plot_module()


def curves(runs: list[dict[str, float]], metric: str, readout: str) -> np.ndarray:
    """Per-run curve of a metric, shape ``(n_runs, n_p)``.

    ``r2_shots_ratio`` is ``r2_shots_<readout> / r2_shots_bayes``,
    ``var_fraction[_from_bloch]`` is ``var_purity[_from_bloch]_<readout> /
    var_purity_bayes``;
    ``purity_mean`` and ``purity_std`` take no readout.
    """
    ps = PB.p_values(runs[0])

    def get(name: str) -> np.ndarray:
        return np.asarray(
            [[run.get(f"{name}_p{p:g}", np.nan) for p in ps] for run in runs],
            dtype=float,
        )

    if metric == "r2_shots_ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            return get(f"r2_shots_{readout}") / get("r2_shots_bayes")
    if metric.startswith("var_fraction"):
        suffix = metric[len("var_fraction") :]
        with np.errstate(divide="ignore", invalid="ignore"):
            return get(f"var_purity{suffix}_{readout}") / get("var_purity_bayes")
    if not readout:
        return get(metric)
    return get(f"{metric}_{readout}")


def _nanmean(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(values, axis=0))


def extremum(ps: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    """Position and value of the min or max of ``y`` in
    :data:`EXTREMUM_RANGE`: a parabola through the extremal point and its two
    neighbours (the point itself at the edge of the range)."""
    lo, hi = EXTREMUM_RANGE
    idx = np.flatnonzero((ps >= lo) & (ps <= hi) & np.isfinite(y))
    if idx.size == 0:
        return float("nan"), float("nan")
    sign = 1.0 if kind == "min" else -1.0
    k = int(idx[np.argmin(sign * y[idx])])
    if k == idx[0] or k == idx[-1]:
        return float(ps[k]), float(y[k])
    x3, y3 = ps[k - 1 : k + 2], y[k - 1 : k + 2]
    a, b, c = np.polyfit(x3, y3, 2)
    if sign * a <= 0:
        return float(ps[k]), float(y[k])
    x0 = float(np.clip(-b / (2 * a), x3[0], x3[-1]))
    return x0, float(a * x0**2 + b * x0 + c)


def crossings(ps: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> list[float]:
    """The ``p`` in :data:`CROSSING_RANGE` where ``y1 - y2`` changes sign,
    linearly interpolated between grid points."""
    lo, hi = CROSSING_RANGE
    d = y1 - y2
    out: list[float] = []
    for i in range(ps.size - 1):
        if not (lo <= ps[i] and ps[i + 1] <= hi):
            continue
        if not (np.isfinite(d[i]) and np.isfinite(d[i + 1])):
            continue
        if d[i] == 0:
            out.append(float(ps[i]))
        elif d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            out.append(float(ps[i] + t * (ps[i + 1] - ps[i])))
    return out


def _interval(samples: list[float]) -> dict[str, float | None]:
    """Median and 16-84 % interval of the finite bootstrap samples, and the
    fraction of resamplings that gave a value."""
    arr = np.asarray(samples, dtype=float)
    ok = arr[np.isfinite(arr)]
    if ok.size == 0:
        return {"median": None, "lo": None, "hi": None, "fraction": 0.0}
    q = np.percentile(ok, [16, 50, 84])
    return {
        "median": float(q[1]),
        "lo": float(q[0]),
        "hi": float(q[2]),
        "fraction": float(ok.size / arr.size),
    }


def analyse(
    runs: dict[int, list[dict[str, float]]], n_boot: int = 1000, seed: int = 282
) -> dict[str, Any]:
    """Extrema per ``N`` and crossings per pair of ``N``, with a bootstrap
    over the circuits."""
    rng = np.random.default_rng(seed)
    sizes = sorted(runs)
    ps = {N: PB.p_values(runs[N][0]) for N in sizes}
    out: dict[str, Any] = {"sizes": sizes, "extrema": {}, "crossings": {}}
    for metric in CROSSED:
        data = {N: curves(runs[N], *SERIES[metric]) for N in sizes}
        if not any(np.any(np.isfinite(d)) for d in data.values()):
            continue
        boots = {
            N: [
                _nanmean(data[N][rng.integers(len(runs[N]), size=len(runs[N]))])
                for _ in range(n_boot)
            ]
            for N in sizes
        }
        if metric in EXTREMA:
            kind = EXTREMA[metric]
            ext: dict[str, Any] = {}
            for N in sizes:
                x0, y0 = extremum(ps[N], _nanmean(data[N]), kind)
                samples = [extremum(ps[N], b, kind) for b in boots[N]]
                ext[f"N{N}"] = {
                    "kind": kind,
                    "p": x0,
                    "value": y0,
                    "p_boot": _interval([s[0] for s in samples]),
                    "value_boot": _interval([s[1] for s in samples]),
                }
            out["extrema"][metric] = ext
        cross: dict[str, Any] = {}
        for i, N1 in enumerate(sizes):
            for N2 in sizes[i + 1 :]:
                if not np.array_equal(ps[N1], ps[N2]):
                    continue
                grid = ps[N1]
                found = crossings(grid, _nanmean(data[N1]), _nanmean(data[N2]))
                first = [
                    (c[0] if c else np.nan)
                    for c in (
                        crossings(grid, b1, b2)
                        for b1, b2 in zip(boots[N1], boots[N2], strict=True)
                    )
                ]
                cross[f"N{N1}_N{N2}"] = {
                    "p": found,
                    "first_p_boot": _interval(first),
                }
        out["crossings"][metric] = cross
    return out


def plot_fss(runs: dict[int, list[dict[str, float]]], out: Path) -> Path:
    """The four panels, one curve per ``N`` (mean and standard error over the
    circuits)."""
    sizes = sorted(runs)
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(sizes)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, (metric, readout, ylabel) in zip(axes.flat, PANELS, strict=True):
        for N, color in zip(sizes, colors, strict=True):
            ps = PB.p_values(runs[N][0])
            data = curves(runs[N], metric, readout)
            mean = _nanmean(data)
            n = np.maximum(np.sum(np.isfinite(data), axis=0), 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sem = np.nanstd(data, axis=0) / np.sqrt(n)
            T = int(runs[N][0].get("T", 0))
            ax.plot(ps, mean, "o-", ms=3, color=color, label=f"N = {N}, T = {T}")
            ax.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
            if metric == "purity_mean":
                std = _nanmean(curves(runs[N], "purity_std", ""))
                ax.plot(ps, std, ":", color=color)
        ax.axvspan(0.17, 0.25, color="grey", alpha=0.12, lw=0)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[1, 1].text(
        0.98,
        0.05,
        "dotted: std over trajectories",
        ha="right",
        transform=axes[1, 1].transAxes,
        fontsize=8,
    )
    for ax in axes[1]:
        ax.set_xlabel("measurement probability p")
    axes[0, 0].legend(fontsize=9)
    n_c = min(len(runs[N]) for N in sizes)
    fig.suptitle(
        "Tomography and purity of the reference qubit vs N (Bell pair with the "
        f"middle site, Haar U(4), T = 2N, {n_c} fixed circuits per N, raw M)\n"
        "grey band: p = 0.17-0.25 (p_c of the Haar brickwork)",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / "bloch_fss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# Panels of purity_variance.png: ((series, line style), ...), y label.
VARIANCE_PANELS: tuple[tuple[tuple[tuple[str, str], ...], str], ...] = (
    ((("var_oracle", "-"),), "oracle Var[Tr(rho_R^2)] over trajectories"),
    (
        (("var_qelm_M", "-"), ("var_qelm_M_abs", "--")),
        "Var of the predicted purity",
    ),
    (
        (("var_fraction_M", "-"), ("var_fraction_M_abs", "--")),
        "predicted / oracle variance",
    ),
)


def _peak_label(summary: dict[str, Any], key: str, N: int) -> str:
    entry = summary.get("extrema", {}).get(key, {}).get(f"N{N}")
    if not entry:
        return ""
    return f" (peak p = {entry['p']:.3f}, {entry['value']:.4f})"


def plot_variance(
    runs: dict[int, list[dict[str, float]]], summary: dict[str, Any], out: Path
) -> Path | None:
    """Variance probe: oracle variance of the purity of ``R`` over the
    trajectories, variance of the QELM outputs and their ratio, one curve
    per ``N``; ``None`` if the runs have no ``var_purity_*``."""
    sizes = sorted(runs)
    if not any(k.startswith("var_purity_bayes") for k in runs[sizes[0]][0]):
        return None
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(sizes)))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    for ax, (series, ylabel) in zip(axes, VARIANCE_PANELS, strict=True):
        for N, color in zip(sizes, colors, strict=True):
            ps = PB.p_values(runs[N][0])
            for key, style in series:
                data = curves(runs[N], *SERIES[key])
                mean = _nanmean(data)
                if not np.any(np.isfinite(mean)):
                    continue
                n = np.maximum(np.sum(np.isfinite(data), axis=0), 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sem = np.nanstd(data, axis=0) / np.sqrt(n)
                label = f"N = {N}"
                if len(series) > 1:
                    label += ", QELM on M" if key.endswith("_M") else ", [M, |M|]"
                label += _peak_label(summary, key, N)
                ax.plot(ps, mean, style, marker="o", ms=3, color=color, label=label)
                ax.fill_between(
                    ps, mean - sem, mean + sem, color=color, alpha=0.15, lw=0
                )
        ax.axvspan(0.17, 0.25, color="grey", alpha=0.12, lw=0)
        ax.set_xlabel("measurement probability p")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    n_c = min(len(runs[N]) for N in sizes)
    fig.suptitle(
        "Variance probe: spread over the trajectories of the purity of R, exact "
        f"(oracle) and predicted (T = 2N, {n_c} fixed circuits per N)\n"
        "solid: QELM on M (purity from its Bloch vector); dashed: ridge on "
        "[M, |M|] (purity directly)",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / "purity_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


NONLINEAR_CURVES = (
    ("M", "ridge on M (QELM)", "#d62728"),
    ("M_abs", "ridge on [M, |M|]", "#2ca02c"),
    ("M_mlp", "MLP on [M, |M|]", "#9467bd"),
    ("bayes", "Bayes optimal on M", "black"),
)


def plot_nonlinear(runs: dict[int, list[dict[str, float]]], out: Path) -> Path | None:
    """Linear and nonlinear readouts of M, one column per ``N`` that has
    them; ``None`` if no run has the diagnostics."""
    sizes = [
        N
        for N in sorted(runs)
        if any(k.startswith("r2_bloch_M_mlp") for k in runs[N][0])
    ]
    if not sizes:
        return None
    rows = (
        ("r2_bloch", "R2, exact r(M)"),
        ("r2_shots", "R2, single shots"),
        ("mae_purity", "purity mean abs. error"),
    )
    fig, axes = plt.subplots(
        len(rows),
        len(sizes),
        figsize=(4.5 * len(sizes), 9),
        sharex=True,
        squeeze=False,
    )
    for j, N in enumerate(sizes):
        ps = PB.p_values(runs[N][0])
        for i, (metric, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for name, legend, color in NONLINEAR_CURVES:
                data = curves(runs[N], metric, name)
                mean = _nanmean(data)
                if not np.any(np.isfinite(mean)):
                    continue
                style = "-" if name != "bayes" else "--"
                ax.plot(ps, mean, style, marker="o", ms=3, color=color, label=legend)
            ax.axvspan(0.17, 0.25, color="grey", alpha=0.12, lw=0)
            ax.grid(alpha=0.3)
            if metric == "r2_bloch":
                ax.set_ylim(-0.05, 1.05)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"N = {N}, {len(runs[N])} circuits", fontsize=10)
            if i == len(rows) - 1:
                ax.set_xlabel("measurement probability p")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Is the QELM-Bayes gap a linear-readout effect? Linear vs nonlinear "
        "readouts of M",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / "bloch_nonlinear.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw the figures and write ``fss_summary.json``; return the paths."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out or args.folders[0]
    out.mkdir(parents=True, exist_ok=True)
    runs = PB.load_runs(args.folders)
    summary = analyse(runs, n_boot=args.n_boot)
    (out / "fss_summary.json").write_text(json.dumps(summary, indent=2))
    paths = [plot_fss(runs, out)]
    for extra in (plot_nonlinear(runs, out), plot_variance(runs, summary, out)):
        if extra is not None:
            paths.append(extra)
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
