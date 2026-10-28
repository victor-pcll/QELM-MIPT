#!/usr/bin/env python3
"""Finite-size scaling of the Bayes-Ridge pipeline on ``M`` (issue #233).

Reads a p-sweep of ``exp_QELM_Q.sh`` run with
``FINAL_VALUES=true BALANCED=true`` (the TFIM reservoir, the record ``M``
ending on a collapse, exactly half of the samples per charge sector),
averages over the seeds and, when ``--sep`` points to sweeps of
``experiments/learnability/learnability.py``, the Bayes accuracy of the
symmetric exclusion process (SEP) on random U(1) circuits of the same
number of system qubits ``Ns = N - 1`` and the same two charges.

Three figures, each with its data (:func:`QELM.utils.plots.save_figure`), in
the TFIM sweep folder (or ``--out``):

- ``bayes_ridge_fss.png``: against ``p``, one color per ``N``. Left, the
  TFIM reservoir: the Bayes-Ridge accuracy ``accuracy_M_bayes_ridge``
  (solid), the Bayes rule of the scrambled model alone
  ``accuracy_M_bayes`` (dashed) and, for reference, the ridge on the raw
  record ``accuracy_M`` (dotted). Right, one panel per ``--sep`` sweep:
  the SEP Bayes accuracy of the U(1) circuit, colored by the same ``N``.
- ``bayes_ridge_collapse.png``: the accuracy against ``d' / 2``, the
  discriminability of the two charges under the scrambled model
  (:func:`draw_collapse`), with the prediction ``Phi(d' / 2)`` of that
  model: if the points of every ``N`` fall on one curve, the accuracy
  depends on ``R p / ((1 - p) Ns)`` alone (``R`` the number of rounds),
  and at a fixed ``R`` goes to 1/2 as ``N`` grows.
- ``bayes_ridge_asymptotics.png``: the test of an asymptotic behavior in
  ``N``. Left, the accuracy at each ``p`` against ``1 / N``, with the fit
  ``A_N = A_inf + c / N`` (:func:`fit_inverse_size`); a straight line
  through the points means that the curves converge to a limit ``A_inf(p)``
  at the rate ``1 / N``. Right, the largest slope ``max_p dA / dp`` of each
  curve against ``N`` (:func:`max_slope`): it stays bounded if the curves
  tend to a smooth limit, and grows with ``N`` if they sharpen into a step,
  the signature of a transition at a ``p_c`` where the curves cross.

``bayes_ridge_fss.json`` (extra) holds the per-``(N, p)`` means and errors,
and ``bayes_ridge_asymptotics.json`` the fits, their ``chi2`` per degree of
freedom, the successive gaps ``A_{N'} - A_N`` with their errors, and the
crossings of the curves of two sizes (:func:`crossings`).

    OUT_DIR=outputs/fss233/tfim N_VALUES="6 8 10 12" SEEDS="42 43" \\
        P_VALUES="0.0 0.1 0.2 0.4 0.6 1.0" FINAL_VALUES=true BALANCED=true \\
        STATE=vector L=2000 MAX_JOBS=4 ./experiments/QELM_Q/exp_QELM_Q.sh
    uv run python experiments/QELM_Q/plot_bayes_ridge_fss.py outputs/fss233/tfim \\
        --sep outputs/fss233/sep_matched outputs/fss233/sep_paper
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from QELM.evaluation.bayes_ridge import scrambled_discriminability
from QELM.utils import load_sweep, summarize
from QELM.utils.plots import (
    color_map,
    draw_curve,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)

TFIM_KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "projective_measurement.p"),
    ("L", "samples.L"),
    ("final", "projective_measurement.final_measurement"),
    ("balanced", "samples.balanced"),
    ("steps", "projective_measurement.nbr_time_evolution"),
)
SEP_KEYS = (
    ("N", "N"),
    ("seed", "seed"),
    ("p", "learnability.p"),
    ("depth", "learnability.depth"),
    ("measure_every", "learnability.measure_every"),
)
# (metric, label, line style) of the TFIM panel.
TFIM_CURVES = (
    ("accuracy_M_bayes_ridge", "Bayes-Ridge on $M$", "-"),
    ("accuracy_M_bayes", "Bayes rule (scrambled SEP) on $M$", "--"),
    ("accuracy_M", "ridge on raw $M$", ":"),
)
MAIN = TFIM_CURVES[0][0]


def fit_inverse_size(
    N: Sequence[float], A: Sequence[float], err: Sequence[float]
) -> dict[str, float]:
    """Fit ``A_N = A_inf + c / N`` by weighted least squares.

    Args:
        N: Sizes, at least 2 distinct.
        A: Accuracy at each size.
        err: Standard error of each ``A``; a zero error (an exact 1 or 0.5)
            is raised to the smallest positive one, or to 1e-3.

    Returns:
        ``A_inf``, ``A_inf_err``, ``c``, ``c_err`` and ``chi2_dof`` (``nan``
        with only 2 sizes).
    """
    x = 1.0 / np.asarray(N, dtype=float)
    y = np.asarray(A, dtype=float)
    s = np.asarray(err, dtype=float)
    floor = s[s > 0].min() if np.any(s > 0) else 1e-3
    s = np.where(s > 0, s, floor)
    design = np.column_stack([np.ones_like(x), x]) / s[:, None]
    coef, *_ = np.linalg.lstsq(design, y / s, rcond=None)
    cov = np.linalg.inv(design.T @ design)
    resid = (y - coef[0] - coef[1] * x) / s
    dof = y.size - 2
    return {
        "A_inf": float(coef[0]),
        "A_inf_err": float(np.sqrt(cov[0, 0])),
        "c": float(coef[1]),
        "c_err": float(np.sqrt(cov[1, 1])),
        "chi2_dof": float(resid @ resid / dof) if dof > 0 else float("nan"),
    }


def max_slope(p: Sequence[float], A: Sequence[float]) -> dict[str, float]:
    """Return the largest finite-difference slope ``dA / dp`` and where it is.

    Args:
        p: Increasing measurement probabilities.
        A: Accuracy at each ``p``.

    Returns:
        ``slope`` and ``p_mid``, the middle of the interval where it is.
    """
    p_arr, A_arr = np.asarray(p, dtype=float), np.asarray(A, dtype=float)
    slopes = np.diff(A_arr) / np.diff(p_arr)
    k = int(np.argmax(slopes))
    return {"slope": float(slopes[k]), "p_mid": float((p_arr[k] + p_arr[k + 1]) / 2)}


def crossings(
    p: Sequence[float],
    A_small: Sequence[float],
    A_large: Sequence[float],
    err: Sequence[float],
    n_sigma: float = 2.0,
) -> list[float]:
    """Return the ``p`` where the curves of two sizes cross, significantly.

    The difference ``A_large - A_small`` is kept where it exceeds
    ``n_sigma`` errors; a crossing is a change of its sign between two such
    points, placed by linear interpolation.

    Args:
        p: Increasing measurement probabilities.
        A_small, A_large: Accuracy of the smaller and of the larger size.
        err: Standard error of the difference at each ``p``.
        n_sigma: Significance of a difference.

    Returns:
        The interpolated crossing points, possibly empty.
    """
    d = np.asarray(A_large, dtype=float) - np.asarray(A_small, dtype=float)
    keep = np.abs(d) > n_sigma * np.asarray(err, dtype=float)
    pk, dk = np.asarray(p, dtype=float)[keep], d[keep]
    out = []
    for i in np.flatnonzero(np.sign(dk[1:]) != np.sign(dk[:-1])):
        out.append(float(pk[i] - dk[i] * (pk[i + 1] - pk[i]) / (dk[i + 1] - dk[i])))
    return out


def tfim_summary(sweep: Path) -> list[dict[str, Any]]:
    """Average the TFIM runs of ``sweep`` over the seeds, one row per ``(N, p)``."""
    rows = load_sweep(sweep, TFIM_KEYS)
    for row in rows:
        if row["final"] is not True:
            raise ValueError(
                f"{row['run']}: issue #233 needs final_measurement=true, "
                f"got {row['final']!r}."
            )
    metrics = [m for m, _, _ in TFIM_CURVES]
    summary = summarize(rows, by=("N", "p"), metrics=metrics)
    n_test = {(r["N"], r["p"]): r["n_test"] for r in rows}
    steps = {(r["N"], r["p"]): r["steps"] for r in rows}
    for row in summary:
        # The binomial error of the pooled test sets: the error of one seed,
        # and a floor for the standard error over a few seeds, which two
        # seeds that happen to agree make far too small.
        pooled = n_test[(row["N"], row["p"])] * row["n_runs"]
        for m in metrics:
            a = row[f"{m}_mean"]
            binomial = float(np.sqrt(max(a * (1 - a), 0.0) / pooled))
            row[f"{m}_err_mean"] = binomial
            if np.isfinite(row[f"{m}_sem"]):
                row[f"{m}_sem"] = max(row[f"{m}_sem"], binomial)
        # final_measurement=true: one round after each step.
        row["rounds"] = int(steps[(row["N"], row["p"])])
    return summary


def sep_summary(sweep: Path) -> list[dict[str, Any]]:
    """Average a learnability sweep, labeled by the TFIM size ``N = Ns + 1``."""
    rows = load_sweep(sweep, SEP_KEYS)
    for row in rows:
        row["N"] = int(row["N"]) + 1
    summary = summarize(rows, by=("N", "p"), metrics=("accuracy", "accuracy_err"))
    first = rows[0]
    per_step = 2 if first["measure_every"] == "layer" else 1
    for row in summary:
        row["rounds"] = per_step * int(first["depth"])
    return summary


def _sep_title(rows: Sequence[dict[str, Any]]) -> str:
    first = rows[0]
    if first["measure_every"] == "period":
        return f"SEP Bayes, U(1) circuit, {first['depth']} rounds"
    return "SEP Bayes, U(1) circuit, $t_f = N_s$ (paper)"


def draw_fss(
    tfim: list[dict[str, Any]], seps: Sequence[tuple[str, list[dict[str, Any]]]]
) -> tuple[Any, dict[str, Any]]:
    """Draw ``bayes_ridge_fss.png``; return the figure and its extra data."""
    Ns = sizes(tfim)
    colors = color_map([str(n) for n in Ns])
    fig, axes = subplots(1, 1 + len(seps), figsize=(7 * (1 + len(seps)), 5.5))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for n in Ns:
        for metric, label, style in TFIM_CURVES:
            curve = sweep_curve(tfim, n, "p", metric)
            draw_curve(
                ax,
                curve,
                color=colors[str(n)],
                label=f"N = {n}, {label}" if style == "-" else None,
                linestyle=style,
            )
    for _, label, style in TFIM_CURVES:
        ax.plot([], [], color="0.4", linestyle=style, label=label)
    ax.axhline(0.5, color="0.6", linestyle=":", lw=1)
    ax.set_title("TFIM reservoir, final measurement, $Q$ from $M$ only")
    ax.set_xlabel("measurement probability $p$")
    ax.set_ylabel("test accuracy on $Q$")
    ax.legend(fontsize=8)
    extra: dict[str, Any] = {"tfim": tfim}
    for ax, (title, rows) in zip(axes[1:], seps, strict=True):
        for n in sizes(rows):
            curve = sweep_curve(rows, n, "p", "accuracy")
            draw_curve(
                ax, curve, color=colors.get(str(n)), label=f"N = {n} (Ns = {n - 1})"
            )
        ax.axhline(0.5, color="0.6", linestyle=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("measurement probability $p$")
        ax.set_ylabel("Bayes accuracy on $Q$")
        ax.legend(fontsize=8)
        extra[title] = rows
    for ax in axes:
        ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    return fig, extra


def draw_collapse(
    datasets: Sequence[tuple[str, list[dict[str, Any]], str]],
) -> tuple[Any, dict[str, Any]]:
    """Draw ``bayes_ridge_collapse.png``: accuracy against ``d' / 2``.

    ``d'`` is the discriminability of the two charges under the scrambled
    model (:func:`QELM.evaluation.bayes_ridge.scrambled_discriminability`),
    ``~ 2 sqrt(rounds p / ((1 - p) Ns))``. If the scrambled model holds,
    every ``(N, p)`` falls on ``Phi(d' / 2)``, whatever ``N``: the
    accuracy is a function of ``rounds p / ((1 - p) Ns)`` alone, and goes
    to 1/2 at fixed ``rounds`` as ``N`` grows. ``p = 0`` and ``p = 1`` are
    left out (``d' = 0`` and infinite).

    Args:
        datasets: ``(title, summary, metric)`` per panel; each summary row
            has ``rounds``.

    Returns:
        The figure and its extra data, the points of every panel.
    """
    from scipy.special import erf

    fig, axes = subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.5))
    axes = np.atleast_1d(axes)
    x = np.linspace(0, 3, 200)
    extra: dict[str, Any] = {}
    for ax, (title, rows, metric) in zip(axes, datasets, strict=True):
        Ns = sizes(rows)
        colors = color_map([str(n) for n in Ns])
        points = []
        for n in Ns:
            curve = sweep_curve(rows, n, "p", metric)
            ns = n - 1
            charges = (ns // 2, ns // 2 - 1)
            keep = (curve.x > 0) & (curve.x < 1)
            half_d = [
                scrambled_discriminability(ns, float(p), int(r["rounds"]), charges) / 2
                for p, r in zip(curve.x[keep], np.array(curve.rows)[keep], strict=True)
            ]
            ax.errorbar(
                half_d,
                curve.mean[keep],
                yerr=curve.err[keep],
                color=colors[str(n)],
                marker="o",
                ls="",
                label=f"N = {n}",
            )
            points += [
                {"N": n, "p": float(p), "half_d": h, "A": float(a)}
                for p, h, a in zip(curve.x[keep], half_d, curve.mean[keep], strict=True)
            ]
        ax.plot(
            x, 0.5 * (1 + erf(x / np.sqrt(2))), color="k", lw=1, label=r"$\Phi(d'/2)$"
        )
        ax.set_xlabel(r"$d'/2 \approx \sqrt{R\,p / ((1-p) N_s)}$")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)
        extra[title] = points
    fig.tight_layout()
    return fig, extra


def asymptotics(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Fits in ``1 / N``, gaps between sizes, crossings and slopes of a metric."""
    Ns = sizes(rows)
    curves = {n: sweep_curve(rows, n, "p", metric) for n in Ns}
    ps = sorted(set.intersection(*(set(c.x.tolist()) for c in curves.values())))
    at = {
        n: {
            float(x): (float(m), float(e))
            for x, m, e in zip(c.x, c.mean, c.err, strict=True)
        }
        for n, c in curves.items()
    }
    per_p = []
    for p in ps:
        A = [at[n][p][0] for n in Ns]
        err = [at[n][p][1] for n in Ns]
        gaps = [
            {
                "N": [a, b],
                "gap": at[b][p][0] - at[a][p][0],
                "gap_err": float(np.hypot(at[a][p][1], at[b][p][1])),
            }
            for a, b in zip(Ns[:-1], Ns[1:], strict=True)
        ]
        per_p.append(
            {"p": p, "A": A, "err": err, "gaps": gaps, **fit_inverse_size(Ns, A, err)}
        )
    cross = {}
    for a, b in zip(Ns[:-1], Ns[1:], strict=True):
        cross[f"{a}-{b}"] = crossings(
            ps,
            [at[a][p][0] for p in ps],
            [at[b][p][0] for p in ps],
            [float(np.hypot(at[a][p][1], at[b][p][1])) for p in ps],
        )
    slopes = {n: max_slope(ps, [at[n][p][0] for p in ps]) for n in Ns}
    return {
        "metric": metric,
        "N": Ns,
        "p": ps,
        "per_p": per_p,
        "crossings": cross,
        "max_slope": slopes,
    }


def draw_asymptotics(
    results: Sequence[tuple[str, dict[str, Any]]],
) -> Any:
    """Draw ``bayes_ridge_asymptotics.png`` from :func:`asymptotics` results."""
    fig, axes = subplots(len(results), 2, figsize=(13, 5 * len(results)))
    axes = np.atleast_2d(axes)
    for (title, res), (ax_fit, ax_slope) in zip(results, axes, strict=True):
        Ns = np.array(res["N"], dtype=float)
        inv = np.linspace(0, 1.05 / Ns.min(), 50)
        colors = color_map([f"{p:g}" for p in res["p"]])
        for row in res["per_p"]:
            if row["p"] == 0:
                continue
            color = colors[f"{row['p']:g}"]
            ax_fit.errorbar(
                1 / Ns, row["A"], yerr=row["err"], color=color, marker="o", ls=""
            )
            ax_fit.plot(
                inv,
                row["A_inf"] + row["c"] * inv,
                color=color,
                lw=1,
                label=(
                    f"p = {row['p']:g}: $A_\\infty$ = {row['A_inf']:.3f}"
                    f" ± {row['A_inf_err']:.3f}, $\\chi^2$/dof = {row['chi2_dof']:.2g}"
                ),
            )
        ax_fit.set_xlim(0, None)
        ax_fit.set_xlabel("$1 / N$")
        ax_fit.set_ylabel("accuracy")
        ax_fit.set_title(f"{title}: $A_N = A_\\infty + c / N$")
        ax_fit.legend(fontsize=7)
        slopes = [res["max_slope"][n]["slope"] for n in res["N"]]
        ax_slope.plot(res["N"], slopes, marker="o", color="k")
        ax_slope.set_xlabel("$N$")
        ax_slope.set_ylabel(r"$\max_p \, dA / dp$")
        ax_slope.set_title("steepest slope of the curve")
    fig.tight_layout()
    return fig


def main() -> None:
    """Parse the command line and draw both figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep", type=Path, help="exp_QELM_Q sweep folder")
    parser.add_argument(
        "--sep", type=Path, nargs="*", default=[], help="learnability sweep folders"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output folder (default: sweep)"
    )
    args = parser.parse_args()
    out = args.out or args.sweep
    tfim = tfim_summary(args.sweep)
    seps = []
    for folder in args.sep:
        rows = load_sweep(folder, SEP_KEYS)
        seps.append((_sep_title(rows), sep_summary(folder)))
    fig, extra = draw_fss(tfim, seps)
    print(f"Wrote {save_figure(fig, out, 'bayes_ridge_fss', extra)}")
    plt.close(fig)
    results = [("TFIM Bayes-Ridge", asymptotics(tfim, MAIN))]
    results += [(title, asymptotics(rows, "accuracy")) for title, rows in seps]
    panels = [("TFIM Bayes-Ridge", tfim, MAIN)]
    panels += [(title, rows, "accuracy") for title, rows in seps]
    fig, extra = draw_collapse(panels)
    print(f"Wrote {save_figure(fig, out, 'bayes_ridge_collapse', extra)}")
    plt.close(fig)
    fig = draw_asymptotics(results)
    data = {title: res for title, res in results}
    print(f"Wrote {save_figure(fig, out, 'bayes_ridge_asymptotics', data)}")
    plt.close(fig)
    for title, res in results:
        print(f"--- {title}")
        for row in res["per_p"]:
            gaps = ", ".join(f"{g['gap']:+.3f}±{g['gap_err']:.3f}" for g in row["gaps"])
            print(
                f"p={row['p']:<5g} A_inf={row['A_inf']:.3f}±{row['A_inf_err']:.3f}"
                f" chi2/dof={row['chi2_dof']:.2f}  gaps: {gaps}"
            )
        print("crossings:", res["crossings"])
        print(
            "max slope:", {n: round(v["slope"], 3) for n, v in res["max_slope"].items()}
        )


if __name__ == "__main__":
    main()
