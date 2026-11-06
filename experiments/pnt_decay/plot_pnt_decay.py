#!/usr/bin/env python3
"""Figures of a sweep of ``pnt_decay.sh`` (issue #305).

Reads every run folder of the sweeps (one per encoding, ``N``, ``T`` and
seed), averages over the seeds (standard error as error bars) and writes, in
the first sweep folder unless ``--out`` is given, each figure of
:data:`FIGURES` as ``<name>.png`` with its data ``<name>.json``, and
``pnt_summary.json``. One row per encoding; color: ``N``; marker: ``T / N``.

- ``signal_decay``: the exact signal against ``n = p N T``.
  ``||Delta||_HS^2`` (relative to ``p = 0``) with ``exp(-n)``; the same
  against ``p V_eff``; the signal of the features ``||s||^2`` (relative to
  ``p = 0``) with its fit ``exp(-gamma n)``; the trace distance of the
  averaged states;
- ``noise``: the mean within-class variance of the features against ``p``
  (log-log) with its fit ``c p^beta``, and ``V_eff / (N T)`` against ``T``;
- ``accuracy_vs_pNT``: the ridge, the oracle linear readout and the
  Gaussian prediction against ``n``;
- ``collapse``: the ridge accuracy and the oracle ``SNR^2`` against
  ``p N T`` and against ``p V_feat`` (the measurements inside the light cone
  of the features), with the spread of the accuracy curves in the titles;
- ``accuracy_vs_snr``: the ridge and the oracle against ``sqrt(SNR^2)``,
  with ``Phi(x / 2)``: the curve every point falls on if the accuracy is set
  by the signal-to-noise ratio alone;
- ``accuracy_per_K``: the ridge accuracy on the trajectories with ``K``
  measurements, against ``K``, one curve per ``p``, for three ``(N, T)``;
- ``midpoints``: the midpoint ``n_0 = p_0 N T`` of each accuracy (sigmoid
  fit) against ``N T``, against ``-ln p_0`` with the line of the model
  ``gamma n_0 = const - beta ln p_0``, ``p_0 V_feat`` against ``-ln p_0``,
  and ``p_0 V_feat`` against its prediction, the ``x`` where the fit
  ``SNR^2 = A exp(-kappa x)`` of the same ``(N, T)`` crosses ``SNR2_MID``.

    uv run python experiments/pnt_decay/plot_pnt_decay.py outputs/pnt_decay/<date_time>

``--volumes <folder> ...``: sweeps run with ``SAMPLE=0`` whose volumes
``V_feat`` fill the runs that lack them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri

from QELM.utils.plots import color_map, save_figure, subplots
from QELM.utils.sweep import load_sweep

KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("encoding", "pnt_task.encoding"),
    ("T", "pnt_task.T"),
)
FIGURES = (
    "signal_decay",
    "noise",
    "accuracy_vs_pNT",
    "collapse",
    "accuracy_vs_snr",
    "accuracy_per_K",
    "midpoints",
)
# Readouts of the accuracy figures, with their titles.
READOUTS = {
    "ridge": "Ridge (trained)",
    "oracle": r"Oracle $w = \Sigma^{-1} s$",
    "gauss": r"Prediction $\Phi(\sqrt{SNR^2}/2)$",
}
# Markers by depth T / N.
MARKERS = {0.5: "v", 1.0: "o", 2.0: "s", 4.0: "D"}
# SNR^2 at which the Gaussian prediction Phi(sqrt(SNR^2) / 2) is 0.75.
SNR2_MID = float((2 * ndtri(0.75)) ** 2)
# Range of n where the decay rate of the feature signal is fitted.
GAMMA_RANGE = (0.5, 8.0)


def _load_depth_plot() -> ModuleType:
    """``plot_depth_scaling.py`` of #268, for its sigmoid fit."""
    path = (
        Path(__file__).resolve().parents[1] / "haar_readouts" / "plot_depth_scaling.py"
    )
    spec = importlib.util.spec_from_file_location("plot_depth_scaling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEPTH = _load_depth_plot()


def load_rows(
    sweeps: Sequence[str | Path], volumes: Sequence[str | Path] = ()
) -> list[dict[str, Any]]:
    """One row per run of the sweeps, with its ``p`` values (from the keys
    ``hs2_p<p>``); runs sharing encoding, ``N``, ``T`` and seed are merged,
    the last one winning. ``volumes``: sweeps run with ``SAMPLE=0``, whose
    first-order volumes ``V_*`` fill the runs that lack them."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sweep in sweeps:
        for row in load_sweep(sweep, keys=KEYS):
            row["N"], row["T"] = int(row["N"]), int(row["n_rounds"])
            row["p_values"] = sorted(
                float(k.removeprefix("hs2_p")) for k in row if k.startswith("hs2_p")
            )
            merged[(row["encoding"], row["N"], row["T"], row["seed"])] = row
    for sweep in volumes:
        for row in load_sweep(sweep, keys=KEYS):
            key = (row["encoding"], int(row["N"]), int(row["n_rounds"]), row["seed"])
            if key in merged:
                for name, value in row.items():
                    if name.startswith("V_"):
                        merged[key].setdefault(name, value)
    return list(merged.values())


def mean_key(runs: list[dict[str, Any]], key: str) -> float:
    """Mean of ``key`` over the runs that have it; NaN if none."""
    values = [float(r[key]) for r in runs if key in r]
    return float(np.mean(values)) if values else float("nan")


def collapse_spread(
    curves: list[tuple[np.ndarray, np.ndarray]], upto: float = 8.0
) -> float:
    """Mean standard deviation across the curves ``(x, y)``, interpolated on
    a common grid of ``x`` in ``[0, upto]``: 0 for a perfect collapse."""
    grid = np.linspace(0.0, upto, 81)
    ys = []
    for x, y in curves:
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() >= 2 and x[keep].max() >= upto:
            ys.append(np.interp(grid, x[keep], y[keep]))
    if len(ys) < 2:
        return float("nan")
    return float(np.mean(np.std(np.asarray(ys), axis=0)))


def cone_fit(points: list[tuple[int, int, float]]) -> float:
    """``c`` of ``V / (N T) = 1 - c N / T`` by least squares (the deficit of
    the backward light cone of local observables, ``c N^2`` rounds-sites)."""
    pts = [(n, t, v) for n, t, v in points if np.isfinite(v)]
    if not pts:
        return float("nan")
    x = np.array([n / t for n, t, _ in pts])
    y = np.array([1.0 - v for *_, v in pts])
    return float(x @ y / (x @ x))


def snr_exp_fit(curves: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """``SNR^2 = A exp(-kappa x)`` by least squares in log over the points of
    the curves ``(x, SNR^2)`` with ``x`` in ``[0.5, 8]``, and the predicted
    midpoint ``x0_predicted`` of the accuracy, where ``SNR^2 = SNR2_MID``."""
    xs, ys = [], []
    for x, snr2 in curves:
        keep = np.isfinite(x) & (snr2 > 0) & (x >= 0.5) & (x <= 8.0)
        xs.extend(x[keep])
        ys.extend(np.log(snr2[keep]))
    if len(xs) < 2:
        nan = float("nan")
        return {"kappa": nan, "A": nan, "x0_predicted": nan}
    slope, log_a = np.polyfit(xs, ys, 1)
    return {
        "kappa": float(-slope),
        "A": float(np.exp(log_a)),
        "x0_predicted": float((log_a - np.log(SNR2_MID)) / -slope),
    }


def groups(
    rows: list[dict[str, Any]],
) -> dict[str, dict[tuple[int, int], list[dict[str, Any]]]]:
    """Runs by encoding, then by ``(N, T)``, sorted."""
    out: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    for row in rows:
        out.setdefault(row["encoding"], {}).setdefault((row["N"], row["T"]), []).append(
            row
        )
    return {e: dict(sorted(g.items())) for e, g in sorted(out.items())}


def curve(
    runs: list[dict[str, Any]], name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(p, mean, err)`` of ``<name>_p<p>`` over the seeds; NaN where no run
    has the key."""
    ps = np.asarray(runs[0]["p_values"])
    mean, err = [], []
    for p in ps:
        v = np.asarray(
            [r[f"{name}_p{p:g}"] for r in runs if f"{name}_p{p:g}" in r], dtype=float
        )
        mean.append(float(v.mean()) if v.size else np.nan)
        err.append(float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)
    return ps, np.asarray(mean), np.asarray(err)


def decay_rate(n: np.ndarray, y: np.ndarray) -> float:
    """Rate ``gamma`` of ``y = exp(-gamma n)`` (least squares in log, through
    ``y(0) = 1``) over :data:`GAMMA_RANGE`; NaN without points."""
    keep = (n >= GAMMA_RANGE[0]) & (n <= GAMMA_RANGE[1]) & (y > 0) & np.isfinite(y)
    if keep.sum() < 2:
        return float("nan")
    x, ly = n[keep], np.log(y[keep])
    return float(-(x @ ly) / (x @ x))


def power_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """``y = c x^beta`` by least squares in log: ``c``, ``beta``."""
    keep = (x > 0) & (y > 0) & np.isfinite(y)
    if keep.sum() < 2:
        return {"c": float("nan"), "beta": float("nan")}
    beta, logc = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return {"c": float(np.exp(logc)), "beta": float(beta)}


def power_law_NT(points: list[tuple[int, int, float]]) -> dict[str, float]:
    """Fit ``p_0 = A N^a T^b`` in log; ``a``, ``b`` with standard errors."""
    pts = [(n, t, v) for n, t, v in points if np.isfinite(v) and v > 0]
    nan = float("nan")
    if len(pts) < 4:
        return {"a": nan, "b": nan, "a_err": nan, "b_err": nan, "A": nan}
    A = np.array([[1.0, np.log(n), np.log(t)] for n, t, _ in pts])
    y = np.log([v for _, _, v in pts])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(pts) - 3, 1)
    cov = np.linalg.inv(A.T @ A) * float(resid @ resid) / dof
    err = np.sqrt(np.diag(cov))
    return {
        "A": float(np.exp(coef[0])),
        "a": float(coef[1]),
        "b": float(coef[2]),
        "a_err": float(err[1]),
        "b_err": float(err[2]),
        "rms_log": float(np.sqrt(np.mean(resid**2))),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates, volumes, noise exponents and midpoints of every group."""
    out: dict[str, Any] = {"encodings": {}}
    for enc, by in groups(rows).items():
        sizes: dict[str, Any] = {}
        all_n, all_s, all_p, all_noise = [], [], [], []
        for (N, T), runs in by.items():
            p, hs2, _ = curve(runs, "hs2")
            _, sig, _ = curve(runs, "signal2")
            _, noise, _ = curve(runs, "noise")
            n = p * N * T
            sig_rel = sig / sig[0] if sig[0] > 0 else sig * np.nan
            v_eff = float(np.mean([r["V_eff"] for r in runs]))
            v_feat = mean_key(runs, "V_feat")
            entry: dict[str, Any] = {
                "N": N,
                "T": T,
                "n_seeds": len(runs),
                "V_eff_over_NT": v_eff / (N * T),
                "V_feat_over_NT": v_feat / (N * T),
                "V_Zfeat_over_NT": mean_key(runs, "V_Zfeat") / (N * T),
                "V_XYfeat_over_NT": mean_key(runs, "V_XYfeat") / (N * T),
                "signal_rate_vfeat": decay_rate(p * v_feat, sig_rel),
                "hs2_rate": decay_rate(n, hs2),
                "hs2_rate_veff": decay_rate(p * v_eff, hs2),
                "signal_rate": decay_rate(n, sig_rel),
                "noise_fit": power_fit(p, noise),
            }
            entry["snr_fit"] = snr_exp_fit([(p * v_feat, curve(runs, "snr2")[1])])
            for name in READOUTS:
                _, acc, err = curve(runs, f"accuracy_{name}")
                fit = DEPTH.fit_sigmoid(p, acc, err)
                entry[f"p0_{name}"] = fit["p0"]
                entry[f"n0_{name}"] = fit["p0"] * N * T
                entry[f"n0feat_{name}"] = fit["p0"] * v_feat
            sizes[f"N{N}_T{T}"] = entry
            keep = p > 0
            all_n.extend(n[keep])
            all_s.extend(sig_rel[keep])
            all_p.extend(p[keep])
            all_noise.extend(noise[keep])
        gamma = decay_rate(np.asarray(all_n), np.asarray(all_s))
        noise_fit = power_fit(np.asarray(all_p), np.asarray(all_noise))
        fits = {}
        for name in READOUTS:
            pts = [(e["N"], e["T"], e[f"p0_{name}"]) for e in sizes.values()]
            law = power_law_NT(pts)
            # Model: gamma n_0 + beta ln p_0 = const, i.e. n_0 linear in ln p_0.
            x = np.array([np.log(v) for *_, v in pts if np.isfinite(v) and v > 0])
            y = np.array([n * t * v for n, t, v in pts if np.isfinite(v) and v > 0])
            line = np.polyfit(x, y, 1).tolist() if x.size >= 2 else [np.nan, np.nan]
            fits[name] = {
                "power_law": law,
                "n0_vs_lnp0": {"slope": line[0], "intercept": line[1]},
                "n0_mean": float(np.nanmean(y)) if y.size else float("nan"),
                "n0_std": float(np.nanstd(y)) if y.size else float("nan"),
            }
        for name in READOUTS:
            pts_f = [
                (e[f"p0_{name}"], e[f"n0feat_{name}"])
                for e in sizes.values()
                if np.isfinite(e[f"n0feat_{name}"]) and e[f"p0_{name}"] > 0
            ]
            if len(pts_f) >= 2:
                slope, icpt = np.polyfit(
                    np.log([q[0] for q in pts_f]), [q[1] for q in pts_f], 1
                )
            else:
                slope = icpt = float("nan")
            fits[name]["n0feat_vs_lnp0"] = {
                "slope": float(slope),
                "intercept": float(icpt),
            }
        spreads = {}
        for label, scale in (("pNT", "NT"), ("pV_feat", "V_feat"), ("pT", "T")):
            curves = []
            for (N, T), runs in by.items():
                p, acc, _ = curve(runs, "accuracy_ridge")
                factor = {
                    "NT": N * T,
                    "T": T * 4.0,
                    "V_feat": mean_key(runs, "V_feat"),
                }[scale]
                curves.append((p * factor, acc))
            spreads[label] = collapse_spread(curves)
        # ln SNR^2 = ln A - kappa x with x = p V_feat, over all the (N, T).
        snr_fit = snr_exp_fit(
            [
                (
                    curve(runs, "snr2")[0] * mean_key(runs, "V_feat"),
                    curve(runs, "snr2")[1],
                )
                for runs in by.values()
            ]
        )
        n0f = [e["n0feat_ridge"] for e in sizes.values()]
        n0f = [v for v in n0f if np.isfinite(v)]
        snr_fit["x0_ridge_mean"] = float(np.mean(n0f)) if n0f else float("nan")
        snr_fit["x0_ridge_std"] = float(np.std(n0f)) if n0f else float("nan")
        out["encodings"][enc] = {
            "snr_vs_vfeat": snr_fit,
            "collapse_spread_ridge": spreads,
            "cone_c": cone_fit(
                [(e["N"], e["T"], e["V_feat_over_NT"]) for e in sizes.values()]
            ),
            "signal_rate_vfeat": decay_rate(
                np.concatenate(
                    [
                        curve(runs, "signal2")[0] * mean_key(runs, "V_feat")
                        for runs in by.values()
                    ]
                ),
                np.concatenate(
                    [
                        curve(runs, "signal2")[1] / curve(runs, "signal2")[1][0]
                        for runs in by.values()
                    ]
                ),
            ),
            "sizes": sizes,
            "signal_rate": gamma,
            "noise_fit": noise_fit,
            "predicted_n0_slope": -noise_fit["beta"] / gamma
            if np.isfinite(gamma) and gamma > 0
            else float("nan"),
            "fits": fits,
        }
    return out


def _style(N: int, T: int, colors: dict[str, str]) -> dict[str, Any]:
    ratio = T / N
    marker = MARKERS.get(min(MARKERS, key=lambda r: abs(r - ratio)), "o")
    return {"color": colors[str(N)], "marker": marker, "ms": 4, "lw": 1}


def _colors(rows: list[dict[str, Any]]) -> dict[str, str]:
    return color_map([str(n) for n in sorted({r["N"] for r in rows})])


def _legend_handles(
    ax: Any, rows: list[dict[str, Any]], colors: dict[str, str]
) -> None:
    for n in sorted({r["N"] for r in rows}):
        ax.plot([], [], color=colors[str(n)], lw=2, label=f"N = {n}")
    ratios = sorted(
        {min(MARKERS, key=lambda r: abs(r - x["T"] / x["N"])) for x in rows}
    )
    for r in ratios:
        ax.plot([], [], marker=MARKERS[r], color="0.3", ls="none", label=f"T = {r:g} N")
    ax.legend(fontsize=8, ncol=2)


def draw_signal(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    colors = _colors(rows)
    fig, axes = subplots(len(by_enc), 4, figsize=(21, 4.6 * len(by_enc)), squeeze=False)
    table: dict[str, Any] = {}
    for r, (enc, by) in enumerate(by_enc.items()):
        a_hs, a_s, a_v, a_d = axes[r]
        for (N, T), runs in by.items():
            st = _style(N, T, colors)
            p, hs2, e_hs = curve(runs, "hs2")
            _, sig, _ = curve(runs, "signal2")
            _, dist, e_d = curve(runs, "trace_distance")
            n = p * N * T
            v = float(np.mean([x["V_eff"] for x in runs]))
            v_feat = mean_key(runs, "V_feat")
            a_hs.errorbar(n, hs2, yerr=e_hs, **st)
            a_s.plot(n, sig / sig[0], **st)
            if np.isfinite(v_feat):
                a_v.plot(p * v_feat, sig / sig[0], **st)
            a_d.errorbar(n, dist, yerr=e_d, **st)
            table[f"{enc}, N = {N}, T = {T}"] = {
                "pNT": n.tolist(),
                "pV_eff": (p * v).tolist(),
                "pV_feat": (p * v_feat).tolist(),
                "hs2": hs2.tolist(),
                "signal2_rel": (sig / sig[0]).tolist(),
                "trace_distance": dist.tolist(),
            }
        x = np.linspace(0, 16, 200)
        gamma = summary["encodings"][enc]["signal_rate"]
        a_hs.plot(x, np.exp(-x), "k--", lw=1.2, label=r"$e^{-n}$")
        a_v.plot(x, np.exp(-x), "k--", lw=1.2, label=r"$e^{-x}$")
        a_s.plot(x, np.exp(-gamma * x), "k--", lw=1.2, label=rf"$e^{{-{gamma:.2f} n}}$")
        a_s.plot(x, np.exp(-x), ":", color="0.4", lw=1.2, label=r"$e^{-n}$")
        for ax in (a_hs, a_v, a_s):
            ax.set_yscale("log")
            ax.set_ylim(1e-7, 2)
        a_hs.set_title(rf"{enc}: $\|\Delta\|_{{HS}}^2$ (exact)")
        a_hs.set_xlabel(r"$n = p\,N\,T$")
        a_hs.set_ylabel("relative to p = 0")
        a_v.set_title(r"$\|s\|^2$ against $p\,V_{\mathrm{feat}}$")
        a_v.set_xlabel(r"$p\,V_{\mathrm{feat}}$ (first-order volume)")
        a_s.set_title(r"Signal of the features $\|s\|^2$")
        a_s.set_xlabel(r"$n = p\,N\,T$")
        a_d.set_title(r"Trace distance $D(\bar\rho_+, \bar\rho_-)$")
        a_d.set_xlabel(r"$n = p\,N\,T$")
        a_d.set_ylim(0, 1.02)
        for ax in (a_hs, a_v, a_s):
            ax.set_xlim(-0.3, 16.5)
            ax.legend(fontsize=9, loc="lower left")
        a_d.set_xlim(-0.3, 16.5)
        _legend_handles(a_d, rows, colors)
    return fig, table


def draw_noise(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    colors = _colors(rows)
    fig, axes = subplots(
        len(by_enc), 2, figsize=(11.5, 4.6 * len(by_enc)), squeeze=False
    )
    table: dict[str, Any] = {}
    for r, (enc, by) in enumerate(by_enc.items()):
        a_n, a_v = axes[r]
        for (N, T), runs in by.items():
            st = _style(N, T, colors)
            p, noise, e = curve(runs, "noise")
            keep = p > 0
            a_n.errorbar(p[keep], noise[keep], yerr=e[keep], **st)
            v = [x["V_eff"] / (N * T) for x in runs]
            a_v.plot(N / T, np.mean(v), **{**st, "ms": 6, "fillstyle": "none"})
            vf = mean_key(runs, "V_feat") / (N * T)
            if np.isfinite(vf):
                a_v.plot(N / T, vf, **{**st, "ms": 6})
            table[f"{enc}, N = {N}, T = {T}"] = {
                "p": p.tolist(),
                "noise": noise.tolist(),
                "V_eff_over_NT": float(np.mean(v)),
            }
        fit = summary["encodings"][enc]["noise_fit"]
        x = np.logspace(-3, 0, 50)
        a_n.plot(
            x,
            fit["c"] * x ** fit["beta"],
            "k--",
            lw=1.2,
            label=rf"${fit['c']:.2f}\,p^{{{fit['beta']:.2f}}}$",
        )
        a_n.set_xscale("log")
        a_n.set_yscale("log")
        a_n.set_xlabel("p")
        a_n.set_ylabel(r"noise $\mathrm{tr}\,\Sigma / d$")
        a_n.set_title(f"{enc}: within-class variance of the features")
        a_n.legend(fontsize=9)
        a_v.axhline(1.0, color="0.6", lw=1)
        c = summary["encodings"][enc]["cone_c"]
        if np.isfinite(c):
            xs = np.linspace(0, 2.1, 50)
            a_v.plot(xs, 1 - c * xs, "k--", lw=1.2, label=rf"$1 - {c:.2f}\,N/T$")
        a_v.set_xlabel("N / T")
        a_v.set_ylabel("volume / (N T)")
        a_v.set_ylim(0, 1.1)
        a_v.set_title(r"$V_{\mathrm{eff}}$ (hollow), $V_{\mathrm{feat}}$ (full)")
        _legend_handles(a_v, rows, colors)
    return fig, table


def draw_accuracy(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    colors = _colors(rows)
    fig, axes = subplots(
        len(by_enc),
        len(READOUTS),
        figsize=(16.5, 4.6 * len(by_enc)),
        squeeze=False,
        sharey=True,
    )
    table: dict[str, Any] = {}
    for r, (enc, by) in enumerate(by_enc.items()):
        for c, (name, title) in enumerate(READOUTS.items()):
            ax = axes[r, c]
            for (N, T), runs in by.items():
                p, acc, err = curve(runs, f"accuracy_{name}")
                ax.errorbar(p * N * T, acc, yerr=err, **_style(N, T, colors))
                table[f"{enc}, N = {N}, T = {T}: {name}"] = {
                    "pNT": (p * N * T).tolist(),
                    "accuracy": acc.tolist(),
                    "err": err.tolist(),
                }
            ax.axhline(0.5, color="0.6", lw=1)
            ax.set_xlim(-0.3, 16.5)
            ax.set_ylim(0.4, 1.02)
            ax.set_xlabel(r"$n = p\,N\,T$")
            ax.set_title(f"{enc}: {title}")
        axes[r, 0].set_ylabel("accuracy on the sector (test)")
        _legend_handles(axes[r, -1], rows, colors)
    return fig, table


def draw_collapse(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    colors = _colors(rows)
    fig, axes = subplots(
        len(by_enc), 4, figsize=(22.0, 4.6 * len(by_enc)), squeeze=False
    )
    table: dict[str, Any] = {}
    panels = (
        ("pNT", r"$n = p\,N\,T$", lambda N, T, runs: N * T),
        ("pV_feat", r"$p\,V_{feat}$", lambda N, T, runs: mean_key(runs, "V_feat")),
    )
    for r, (enc, by) in enumerate(by_enc.items()):
        spreads = summary["encodings"][enc]["collapse_spread_ridge"]
        for c, (label, xlabel, scale) in enumerate(panels):
            a_acc, a_snr = axes[r, c], axes[r, c + 2]
            for (N, T), runs in by.items():
                p, acc, err = curve(runs, "accuracy_ridge")
                _, snr2, snr_err = curve(runs, "snr2")
                x = p * scale(N, T, runs)
                if not np.all(np.isfinite(x)):
                    continue
                a_acc.errorbar(x, acc, yerr=err, **_style(N, T, colors))
                keep = (p > 0) & (snr2 > 0)
                a_snr.errorbar(
                    x[keep], snr2[keep], yerr=snr_err[keep], **_style(N, T, colors)
                )
                table[f"{enc}, N = {N}, T = {T}: {label}"] = {
                    "x": x.tolist(),
                    "accuracy": acc.tolist(),
                    "snr2": snr2.tolist(),
                }
            a_acc.axhline(0.5, color="0.6", lw=1)
            a_acc.set_xlim(-0.3, 12.5)
            a_acc.set_ylim(0.4, 1.02)
            a_acc.set_xlabel(xlabel)
            a_acc.set_ylabel("accuracy on the sector (test)")
            a_acc.set_title(f"{enc}: ridge, spread {spreads[label]:.3f}")
            # Accuracy 0.75 of the Gaussian prediction: SNR^2 = (2 ndtri(0.75))^2.
            a_snr.axhline(SNR2_MID, color="0.6", lw=1, ls="--")
            a_snr.set_yscale("log")
            a_snr.set_xlim(0.0, 12.5)
            a_snr.set_xlabel(xlabel)
            a_snr.set_ylabel(r"$SNR^2 = s^T \Sigma^{-1} s$")
            a_snr.set_title(f"{enc}: oracle signal-to-noise ratio")
        _legend_handles(axes[r, -1], rows, colors)
    return fig, table


def draw_snr(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    colors = _colors(rows)
    fig, axes = subplots(
        len(by_enc), 2, figsize=(11.5, 4.6 * len(by_enc)), squeeze=False, sharey=True
    )
    table: dict[str, Any] = {}
    x = np.linspace(0, 8, 200)
    for r, (enc, by) in enumerate(by_enc.items()):
        residuals = {"ridge": [], "oracle": []}
        for (N, T), runs in by.items():
            p, snr2, _ = curve(runs, "snr2")
            keep = p > 0
            root = np.sqrt(np.clip(snr2[keep], 0, None))
            for c, name in enumerate(("ridge", "oracle")):
                _, acc, err = curve(runs, f"accuracy_{name}")
                axes[r, c].errorbar(
                    root,
                    acc[keep],
                    yerr=err[keep],
                    **{**_style(N, T, colors), "ls": "none"},
                )
                residuals[name].extend(acc[keep] - ndtr(root / 2))
            table[f"{enc}, N = {N}, T = {T}"] = {
                "sqrt_snr2": root.tolist(),
                "ridge": curve(runs, "accuracy_ridge")[1][keep].tolist(),
                "oracle": curve(runs, "accuracy_oracle")[1][keep].tolist(),
            }
        for c, name in enumerate(("ridge", "oracle")):
            ax = axes[r, c]
            res = np.asarray(residuals[name])
            ax.plot(x, ndtr(x / 2), "k--", lw=1.5, label=r"$\Phi(x/2)$")
            ax.axhline(0.5, color="0.6", lw=1)
            ax.set_xlim(0, 8)
            ax.set_ylim(0.4, 1.02)
            ax.set_xlabel(r"$\sqrt{SNR^2} = \sqrt{s^T \Sigma^{-1} s}$")
            ax.set_title(f"{enc}: {READOUTS[name]}")
            ax.text(
                0.97,
                0.05,
                f"mean residual {np.mean(res):+.3f}\n"
                f"rms {np.sqrt(np.mean(res**2)):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
            )
            ax.legend(fontsize=9, loc="upper left")
            summary["encodings"][enc][f"snr_residual_{name}"] = {
                "mean": float(np.mean(res)),
                "rms": float(np.sqrt(np.mean(res**2))),
            }
        axes[r, 0].set_ylabel("accuracy on the sector (test)")
    return fig, table


def draw_per_k(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    # Three (N, T): the largest N at T = N, and the extreme depths.
    picks: list[tuple[int, int]] = []
    keys = sorted({k for by in by_enc.values() for k in by})
    n_max = max(n for n, _ in keys)
    for target in (
        (n_max, n_max),
        min(keys, key=lambda k: k[1] / k[0]),
        max(keys, key=lambda k: k[1] / k[0]),
    ):
        if target in keys and target not in picks:
            picks.append(target)
    if not picks:
        picks = keys[:1]
    fig, axes = subplots(
        len(by_enc),
        len(picks),
        figsize=(5.4 * len(picks), 4.4 * len(by_enc)),
        squeeze=False,
        sharey=True,
    )
    table: dict[str, Any] = {}
    for r, (enc, by) in enumerate(by_enc.items()):
        for c, (N, T) in enumerate(picks):
            ax = axes[r, c]
            runs = by.get((N, T), [])
            if not runs:
                continue
            ps = [p for p in runs[0]["p_values"] if p > 0]
            pcolors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(ps)))
            for p, col in zip(ps, pcolors, strict=True):
                ks, means = [], []
                for k in range(13):
                    v = [
                        x[f"accuracy_ridge_K{k}_p{p:g}"]
                        for x in runs
                        if f"accuracy_ridge_K{k}_p{p:g}" in x
                    ]
                    if len(v) >= max(2, len(runs) // 2):
                        ks.append(k)
                        means.append(float(np.mean(v)))
                if ks:
                    ax.plot(
                        ks,
                        means,
                        marker="o",
                        ms=3,
                        color=col,
                        lw=1,
                        label=f"n = {p * N * T:.3g}",
                    )
                    table[f"{enc}, N = {N}, T = {T}, p = {p:g}"] = {
                        "K": ks,
                        "accuracy": means,
                    }
            ax.axhline(0.5, color="0.6", lw=1)
            ax.set_ylim(0.4, 1.02)
            ax.set_xlabel("K, measurements of the trajectory")
            ax.set_title(f"{enc}, N = {N}, T = {T}")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, ncol=2)
        axes[r, 0].set_ylabel("ridge accuracy at fixed K (test)")
    return fig, table


def draw_midpoints(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[plt.Figure, dict[str, Any]]:
    by_enc = groups(rows)
    colors = _colors(rows)
    fig, axes = subplots(
        len(by_enc), 4, figsize=(22.0, 4.6 * len(by_enc)), squeeze=False
    )
    table: dict[str, Any] = {}
    fills = {"ridge": "full", "oracle": "none", "gauss": "none"}
    for r, (enc, info) in enumerate(summary["encodings"].items()):
        a_nt, a_l, a_f, a_pr = axes[r]
        for name in READOUTS:
            pts = []
            for e in info["sizes"].values():
                N, T, p0 = e["N"], e["T"], e[f"p0_{name}"]
                if not np.isfinite(p0) or p0 <= 0:
                    continue
                st = {
                    **_style(N, T, colors),
                    "ls": "none",
                    "fillstyle": fills[name],
                    "ms": 6,
                }
                if name == "gauss":
                    st["mew"] = 0.6
                a_nt.plot(N * T, p0 * N * T, **st)
                a_l.plot(-np.log(p0), p0 * N * T, **st)
                if np.isfinite(e[f"n0feat_{name}"]):
                    a_f.plot(-np.log(p0), e[f"n0feat_{name}"], **st)
                    a_pr.plot(e["snr_fit"]["x0_predicted"], e[f"n0feat_{name}"], **st)
                pts.append((-np.log(p0), p0 * N * T))
            table[f"{enc}: {name}"] = pts
            line = info["fits"][name]["n0_vs_lnp0"]
            if name == "ridge" and np.isfinite(line["slope"]):
                x = np.linspace(min(q[0] for q in pts), max(q[0] for q in pts), 10)
                a_l.plot(
                    x,
                    line["intercept"] - line["slope"] * x,
                    "k-",
                    lw=1,
                    label=rf"fit: slope {-line['slope']:.2f}",
                )
            line_f = info["fits"][name]["n0feat_vs_lnp0"]
            if name == "ridge" and np.isfinite(line_f["slope"]) and pts:
                x = np.linspace(min(q[0] for q in pts), max(q[0] for q in pts), 10)
                a_f.plot(
                    x,
                    line_f["intercept"] - line_f["slope"] * x,
                    "k-",
                    lw=1,
                    label=rf"fit: slope {-line_f['slope']:.2f}",
                )
        beta = info["noise_fit"]["beta"]
        a_f.plot([], [], " ", label=rf"model $\beta$ = {beta:.2f}")
        a_f.set_xlabel(r"$-\ln p_0$")
        a_f.set_ylabel(r"$p_0 V_{feat}$")
        a_f.set_title(r"Model $p_0 V_{feat} = C - \beta \ln p_0$")
        a_f.legend(fontsize=9, loc="upper left")
        lims = np.array([2.4, 4.4])
        a_pr.plot(lims, lims, "k--", lw=1, label="measured = predicted")
        a_pr.set_xlim(*lims)
        a_pr.set_ylim(*lims)
        a_pr.set_xlabel(r"predicted: $SNR^2(x_0) = 1.82$")
        a_pr.set_ylabel(r"measured $x_0 = p_0 V_{feat}$")
        a_pr.set_title("Midpoint from the signal-to-noise ratio")
        a_pr.legend(fontsize=9, loc="upper left")
        pred = info["predicted_n0_slope"]
        a_l.plot([], [], " ", label=rf"model $\beta/\gamma$ = {pred:.2f}")
        a_nt.set_xscale("log")
        a_nt.set_xlabel("N T")
        a_nt.set_ylabel(r"$n_0 = p_0 N T$")
        a_nt.set_title(f"{enc}: midpoint of the accuracy")
        a_l.set_xlabel(r"$-\ln p_0$")
        a_l.set_ylabel(r"$n_0 = p_0 N T$")
        a_l.set_title(r"Model $\gamma n_0 = C - \beta \ln p_0$")
        a_l.legend(fontsize=9, loc="upper left")
        for name, fill in fills.items():
            a_nt.plot(
                [], [], marker="o", color="0.3", ls="none", fillstyle=fill, label=name
            )
        a_nt.legend(fontsize=9)
    return fig, table


# The function that draws each figure of FIGURES.
DRAWERS = {
    "signal_decay": draw_signal,
    "noise": draw_noise,
    "accuracy_vs_pNT": draw_accuracy,
    "collapse": draw_collapse,
    "accuracy_vs_snr": draw_snr,
    "accuracy_per_K": draw_per_k,
    "midpoints": draw_midpoints,
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweeps", type=Path, nargs="+", help="folders of pnt_decay.sh")
    parser.add_argument("--out", type=Path, default=None, help="figure folder")
    parser.add_argument(
        "--volumes",
        type=Path,
        nargs="*",
        default=[],
        help="sweeps run with SAMPLE=0, whose volumes V_* fill the runs",
    )
    args = parser.parse_args(argv)
    out = args.out or args.sweeps[0]
    rows = load_rows(args.sweeps, args.volumes)
    if not rows:
        raise ValueError("No run in the sweeps.")
    summary = summarize(rows)
    for name in FIGURES:
        fig, table = DRAWERS[name](rows, summary)
        save_figure(fig, out, name, data=table)
        plt.close(fig)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "pnt_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
