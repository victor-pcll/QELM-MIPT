#!/usr/bin/env python3
"""Figures of the KL / Jensen-Shannon sweep (issue #270).

Reads the run folders written by ``kl_divergence.sh`` (one per ``N``, seed
and ``p``), averages over the seeds and writes, in the sweep folder, each
figure with its data (:func:`QELM.utils.plots.save_figure`):

- ``kl_vs_accuracy.png``: one column per ``N``. Top, the Bayes accuracy
  against ``p`` (the "blue curve" of the observer who does not know the
  gates, and the observer who knows them, mean over the ``N_c`` circuits
  with their spread shaded). Bottom, on a log scale, the divergences that
  drive it: the finite part of the Jeffreys distance, the Jensen-Shannon
  divergence ``I(Q; M)`` in bits and the infinite mass ``pi_inf`` (the
  probability that a record gives the charge for certain, which makes the
  KL infinite). The dashed line marks ``p_c`` (``--p-c``).
- ``kl_finite_size.png``: finite-size analysis. (a) ``I(Q; M)`` against
  ``p`` for every ``N``; (b) ``pi_inf`` against ``p``; (c) the decay rate
  ``alpha(p)`` of the fit ``I(Q; M) ~ 2**(-alpha N)`` over the sizes; (d)
  the finite Jeffreys distance against ``p - p_c`` above ``p_c``, log-log,
  with the fit ``D ~ (p - p_c)**beta`` for every ``N``.
- ``kl_bounds.png``: one panel per ``N``, the Bayes error of the observer
  who does not know the gates against ``p``, with its lower and upper
  bounds from Jensen-Shannon and from Bhattacharyya, and the
  Bretagnolle-Huber (Le Cam) bound ``exp(-D_KL) / 4``, 0 at every ``p > 0``
  since the KL is infinite.
- ``scaling.json``: the fitted ``alpha(p)`` and ``beta(N)``.

    uv run python experiments/kl_divergence/plot_kl_divergence.py \\
        outputs/kl_divergence/<date_time> --p-c 0.175
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from QELM.evaluation.kl_divergence import BOUNDS, STATISTICS
from QELM.utils import latest_sweep, load_sweep, summarize
from QELM.utils.plots import (
    COLORS,
    color_map,
    draw_curve,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

KEYS = (("N", "N"), ("seed", "seed"), ("p", "kl.p"))
# Accent of the critical point, as asked in issue #270.
P_C_COLOR = "#fa5703"
P_C = 0.175
# The two observers: prefix of their metrics and legend.
OBSERVERS = (
    ("sep", "gates unknown (Haar-averaged)"),
    ("circuit", "gates known (mean over circuits)"),
)
SHORT = {"sep": "gates unknown", "circuit": "gates known"}


def load_summary(sweep_dir: str | Path) -> list[dict[str, Any]]:
    """Return the sweep averaged over the seeds, one row per ``(N, p)``.

    Every statistic is averaged with its error (``<prefix>_<key>_err``)
    and, for the circuits, its spread between circuits
    (``circuit_<key>_std``). The runs without circuits (``kl.n_circuits=0``)
    have no ``circuit_*`` metric: they get ``nan``.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    metrics: list[str] = []
    for key in (*STATISTICS, *BOUNDS):
        metrics += [f"sep_{key}", f"circuit_{key}"]
        if key in STATISTICS:
            metrics += [f"sep_{key}_err", f"circuit_{key}_err", f"circuit_{key}_std"]
    for row in rows:
        for m in metrics:
            row.setdefault(m, float("nan"))
    return summarize(rows, by=("N", "p"), metrics=metrics)


def _positive(summary: Sequence[dict[str, Any]], metric: str) -> bool:
    """Whether ``metric`` is finite and positive somewhere in the sweep."""
    return any(
        np.isfinite(r[f"{metric}_mean"]) and r[f"{metric}_mean"] > 0 for r in summary
    )


def _has(summary: Sequence[dict[str, Any]], metric: str) -> bool:
    return any(np.isfinite(r[f"{metric}_mean"]) for r in summary)


def _p_c_line(ax: Axes, p_c: float | None) -> None:
    if p_c is not None:
        ax.axvline(
            p_c, color=P_C_COLOR, ls="--", lw=1.5, label=rf"$p_c \approx {p_c:g}$"
        )


def _curve(
    ax: Axes,
    summary: Sequence[dict[str, Any]],
    N: int,
    metric: str,
    color: str,
    label: str,
    linestyle: str = "-",
    log: bool = False,
    spread: bool = False,
) -> None:
    """Draw ``metric`` for size ``N``, skipping the non-finite points."""
    curve = sweep_curve(summary, N, "p", metric)
    keep = np.isfinite(curve.mean) & (curve.mean > 0 if log else True)
    if not keep.any():
        return
    curve = curve.where(keep)
    draw_curve(ax, curve, color=color, label=label, linestyle=linestyle, log=log)
    if spread:
        std = np.array([r[f"{metric}_std_mean"] for r in curve.rows], dtype=float)
        if np.isfinite(std).all():
            low = curve.mean - std
            if log:
                low = np.maximum(low, 0.5 * curve.mean)
            ax.fill_between(curve.x, low, curve.mean + std, color=color, alpha=0.15)


def draw_kl_vs_accuracy(
    summary: Sequence[dict[str, Any]], p_c: float | None = P_C
) -> tuple[Figure, np.ndarray]:
    """Bayes accuracy (top) and divergences (bottom), one column per ``N``."""
    Ns = sizes(summary)
    fig, axes = subplots(
        2, len(Ns), figsize=(5.5 * len(Ns), 8.5), sharex=True, squeeze=False
    )
    for col, N in enumerate(Ns):
        top, bottom = axes[0, col], axes[1, col]
        top.axhline(0.5, color="gray", lw=1, ls=":", label="chance")
        for (prefix, name), color, style in zip(
            OBSERVERS, COLORS[:2], ("-", "--"), strict=True
        ):
            if not _has(summary, f"{prefix}_bayes_accuracy"):
                continue
            _curve(
                top,
                summary,
                N,
                f"{prefix}_bayes_accuracy",
                color,
                f"Bayes accuracy, {name}",
                style,
                spread=prefix == "circuit",
            )
            _curve(
                bottom,
                summary,
                N,
                f"{prefix}_jeffreys_finite",
                COLORS[2],
                f"finite Jeffreys $D_{{sym}}$ [nats], {SHORT[prefix]}",
                style,
                log=True,
                spread=prefix == "circuit",
            )
            _curve(
                bottom,
                summary,
                N,
                f"{prefix}_js_bits",
                COLORS[4],
                f"JS $= I(Q;M)$ [bits], {SHORT[prefix]}",
                style,
                log=True,
                spread=prefix == "circuit",
            )
            _curve(
                bottom,
                summary,
                N,
                f"{prefix}_infinite_mass",
                COLORS[5],
                rf"$\pi_\infty$ (KL $= \infty$), {SHORT[prefix]}",
                style,
                log=True,
                spread=prefix == "circuit",
            )
        _p_c_line(top, p_c)
        _p_c_line(bottom, p_c)
        top.set_title(f"N = {N}")
        top.set_ylim(0.45, 1.02)
        bottom.set_xlabel("measurement probability $p$")
        if col == 0:
            top.set_ylabel("Bayes accuracy $1 - P_e$")
            bottom.set_ylabel("divergence")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    axes[1, 0].legend(loc="lower right", fontsize=8)
    return fig, axes


def fit_alpha(
    summary: Sequence[dict[str, Any]], metric: str = "sep_js_bits"
) -> list[dict[str, float]]:
    """Fit ``log2 metric = c - alpha N`` over the sizes, at every ``p``.

    Returns:
        One ``{"p", "alpha", "alpha_err", "n_sizes"}`` per ``p`` with at
        least two sizes; ``alpha > 0`` means a decay with ``N``.
    """
    out = []
    for p in sorted({float(r["p"]) for r in summary}):
        pts = [
            (r["N"], r[f"{metric}_mean"])
            for r in summary
            if float(r["p"]) == p and r[f"{metric}_mean"] > 0
        ]
        if len(pts) < 2:
            continue
        N, y = np.array(pts, dtype=float).T
        if len(pts) > 2:
            coef, cov = np.polyfit(N, np.log2(y), 1, cov=True)
            err = float(np.sqrt(cov[0, 0]))
        else:
            coef, err = np.polyfit(N, np.log2(y), 1), float("nan")
        out.append(
            {"p": p, "alpha": float(-coef[0]), "alpha_err": err, "n_sizes": len(pts)}
        )
    return out


def fit_beta(
    summary: Sequence[dict[str, Any]],
    p_c: float,
    metric: str = "sep_jeffreys_finite",
) -> list[dict[str, float]]:
    """Fit ``ln metric = c + beta ln(p - p_c)`` for ``p > p_c``, for every ``N``.

    Returns:
        One ``{"N", "beta", "beta_err", "n_points"}`` per size with at least
        two points above ``p_c``.
    """
    out = []
    for N in sizes(summary):
        pts = [
            (float(r["p"]) - p_c, r[f"{metric}_mean"])
            for r in summary
            if r["N"] == N
            and float(r["p"]) > p_c
            and np.isfinite(r[f"{metric}_mean"])
            and r[f"{metric}_mean"] > 0
        ]
        if len(pts) < 2:
            continue
        x, y = np.array(pts, dtype=float).T
        if len(pts) > 2:
            coef, cov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
            err = float(np.sqrt(cov[0, 0]))
        else:
            coef, err = np.polyfit(np.log(x), np.log(y), 1), float("nan")
        out.append(
            {
                "N": int(N),
                "beta": float(coef[0]),
                "beta_err": err,
                "intercept": float(coef[1]),
                "n_points": len(pts),
            }
        )
    return out


def draw_finite_size(
    summary: Sequence[dict[str, Any]], p_c: float | None = P_C
) -> tuple[Figure, np.ndarray, dict[str, Any]]:
    """The four panels of the finite-size analysis, and the fits."""
    Ns = sizes(summary)
    colors = color_map([str(N) for N in Ns])
    fig, axes = subplots(2, 2, figsize=(12, 9.5))
    a, b, c, d = axes.ravel()
    for N in Ns:
        color = colors[str(N)]
        _curve(a, summary, N, "sep_js_bits", color, f"N = {N}", log=True)
        _curve(b, summary, N, "sep_infinite_mass", color, f"N = {N}", log=True)
        if _has(summary, "circuit_js_bits"):
            # Label the dashed curves when the solid ones are all 0 (not drawn).
            label = "" if _positive(summary, "sep_js_bits") else f"N = {N}"
            _curve(a, summary, N, "circuit_js_bits", color, label, "--", log=True)
            _curve(b, summary, N, "circuit_infinite_mass", color, "", "--", log=True)
    if not any(
        _positive(summary, f"{prefix}_infinite_mass") for prefix, _ in OBSERVERS
    ):
        b.text(
            0.5,
            0.5,
            r"$\pi_\infty = 0$ at every $p$: the KL is finite",
            transform=b.transAxes,
            ha="center",
        )
    for ax, title in (
        (a, "(a) $I(Q;M)$ [bits]"),
        (b, r"(b) infinite mass $\pi_\infty$"),
    ):
        _p_c_line(ax, p_c)
        ax.set_xlabel("$p$")
        ax.set_title(title)
    a.legend(title="solid: gates unknown\ndashed: gates known", fontsize=9)

    alphas = {
        prefix: fit_alpha(summary, f"{prefix}_js_bits")
        for prefix, _ in OBSERVERS
        if _has(summary, f"{prefix}_js_bits")
    }
    for (prefix, name), color in zip(OBSERVERS, COLORS[:2], strict=True):
        rows = alphas.get(prefix, [])
        if rows:
            c.errorbar(
                [r["p"] for r in rows],
                [r["alpha"] for r in rows],
                yerr=np.nan_to_num([r["alpha_err"] for r in rows]),
                color=color,
                marker="o",
                capsize=3,
                label=name,
            )
    c.axhline(0.0, color="gray", lw=1, ls=":")
    _p_c_line(c, p_c)
    c.set_xlabel("$p$")
    c.set_ylabel(r"$\alpha$")
    c.set_title(r"(c) $I(Q;M) \sim 2^{-\alpha N}$")
    c.legend(fontsize=9)

    betas: list[dict[str, float]] = []
    prefix = "sep"
    if p_c is not None:
        # The observer who does not know the gates, unless it learns nothing
        # (Haar U(4) gates and the local encoding): then the one who does.
        prefix = "sep" if _positive(summary, "sep_jeffreys_finite") else "circuit"
        metric = f"{prefix}_jeffreys_finite"
        betas = fit_beta(summary, p_c, metric)
        for fit in betas:
            N = int(fit["N"])
            rows = [
                r
                for r in summary
                if r["N"] == N
                and float(r["p"]) > p_c
                and np.isfinite(r[f"{metric}_mean"])
                and r[f"{metric}_mean"] > 0
            ]
            x = np.array([float(r["p"]) - p_c for r in rows])
            y = np.array([r[f"{metric}_mean"] for r in rows])
            d.loglog(x, y, "o", color=colors[str(N)])
            d.loglog(
                x,
                np.exp(fit["intercept"]) * x ** fit["beta"],
                "-",
                color=colors[str(N)],
                label=rf"N = {N}: $\beta = {fit['beta']:.2f}$",
            )
        if betas:
            d.legend(fontsize=9)
    d.set_xlabel("$p - p_c$")
    d.set_ylabel(r"finite $D_{sym}$ [nats]")
    d.set_title(rf"(d) $D_{{sym}} \sim (p - p_c)^\beta$, {SHORT[prefix]}")
    return fig, axes, {"alpha": alphas, "beta": betas, "p_c": p_c}


def draw_bounds(
    summary: Sequence[dict[str, Any]], p_c: float | None = P_C
) -> tuple[Figure, np.ndarray]:
    """Bayes error of the Haar-averaged observer and its bounds, per ``N``."""
    Ns = sizes(summary)
    fig, axes = subplots(1, len(Ns), figsize=(5.5 * len(Ns), 4.8), squeeze=False)
    for ax, N in zip(axes[0], Ns, strict=True):
        _curve(ax, summary, N, "sep_bayes_error", COLORS[0], "Bayes error $P_e$")
        rows = sorted((r for r in summary if r["N"] == N), key=lambda r: r["p"])
        p = np.array([r["p"] for r in rows], dtype=float)
        for key, label, color, style in (
            ("bound_js_upper", "JS upper", COLORS[4], "--"),
            ("bound_js_lower", "JS lower", COLORS[4], ":"),
            ("bound_hellinger_upper", "Bhattacharyya upper", COLORS[2], "--"),
            ("bound_hellinger_lower", "Bhattacharyya lower", COLORS[2], ":"),
            ("bound_le_cam", r"Le Cam $e^{-D_{KL}}/4$", COLORS[5], "-."),
        ):
            ax.plot(
                p,
                [r[f"sep_{key}_mean"] for r in rows],
                color=color,
                ls=style,
                label=label,
            )
        _p_c_line(ax, p_c)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("$p$")
    axes[0, 0].set_ylabel("error probability")
    axes[0, 0].legend(fontsize=8)
    return fig, axes


def plot_sweep(sweep_dir: str | Path, p_c: float | None = P_C) -> dict[str, Any]:
    """Write the three figures and ``scaling.json`` in ``sweep_dir``.

    Returns:
        The fits of :func:`draw_finite_size`.
    """
    sweep_dir = Path(sweep_dir)
    summary = load_summary(sweep_dir)
    fig, _ = draw_kl_vs_accuracy(summary, p_c)
    save_figure(fig, sweep_dir, "kl_vs_accuracy", {"summary": summary})
    plt.close(fig)
    fig, _, fits = draw_finite_size(summary, p_c)
    save_figure(fig, sweep_dir, "kl_finite_size", fits)
    plt.close(fig)
    fig, _ = draw_bounds(summary, p_c)
    save_figure(fig, sweep_dir, "kl_bounds")
    plt.close(fig)
    (sweep_dir / "scaling.json").write_text(json.dumps(fits, indent=2) + "\n")
    return fits


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: ``plot_kl_divergence.py [SWEEP_DIR] [--p-c P]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        help="sweep folder; the latest of outputs/kl_divergence by default",
    )
    parser.add_argument(
        "--p-c", type=float, default=P_C, help="critical point to mark and fit"
    )
    args = parser.parse_args(argv)
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else latest_sweep(Path("outputs") / "kl_divergence")
    )
    fits = plot_sweep(sweep_dir, args.p_c)
    for prefix, rows in fits["alpha"].items():
        for r in rows:
            print(f"alpha[{prefix}](p={r['p']:.3g}) = {r['alpha']:.3f}")
    for r in fits["beta"]:
        print(f"beta(N={r['N']}) = {r['beta']:.3f} +- {r['beta_err']:.2g}")
    print(f"Figures in {sweep_dir}")


if __name__ == "__main__":
    main()
