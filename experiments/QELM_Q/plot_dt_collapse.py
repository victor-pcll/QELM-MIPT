#!/usr/bin/env python3
"""Effect of the step ``dt`` on the ``Phi(d' / 2)`` collapse (issue #253).

In issue #233 the Bayes-Ridge accuracy on the record ``M`` of the TFIM
reservoir (``R = 5`` rounds of ``dt = 2 / J``) falls on ``Phi(d' / 2)``, the
prediction of the scrambled model: before each round the configuration is
uniform on the charge sector, whatever the previous rounds found. The
hypothesis is that it works because ``dt = 2 / J`` is about the scrambling
time of the reservoir (``t_scr ~ 2.25 / J``, issue #243). This script reads a
sweep of ``exp_QELM_Q.sh`` over ``T_VALUES`` at a fixed
``nbr_time_evolution = R`` (so ``dt = t / R``) and, with ``--haar``, the same
sweep with ``STEP=haar_u1``: a Haar U(1) unitary at each step, which
realizes the scrambled model exactly, discreteness of the counts included.

Four figures, each with its data (:func:`QELM.utils.plots.save_figure`), in
the sweep folder (or ``--out``):

- ``dt_accuracy_vs_p.png``: the Bayes-Ridge accuracy against ``p``, one
  panel per ``dt``, one color per ``N``; the Haar U(1) control of the same
  ``N`` dashed, when given.
- ``dt_collapse.png``: the same points against ``d' / 2``, one panel per
  ``dt`` (and one for the control), with ``Phi(d' / 2)``
  (``plot_bayes_ridge_fss.draw_collapse``).
- ``dt_two_rules.png``: the memory left in the record. The frozen Bayes
  rule (:func:`QELM.evaluation.bayes_ridge.frozen_log_likelihood`: the
  configuration does not move between rounds, the opposite limit of the
  scrambled model) on the same test records, against the Bayes-Ridge
  accuracy. Left, the mean over ``p`` of ``A_frozen - A_scrambled``
  against ``dt``, the Haar control dashed; right, both against ``p`` at the
  largest ``N``, one color per ``dt``.
- ``dt_deviation.png``: the gap ``A_TFIM - A_ref`` to the scrambled model,
  ``A_ref`` the Haar U(1) control at the same ``(N, p)`` when given, else
  ``Phi(d' / 2)``. Left, its mean over ``p`` against ``dt`` (log scale), one
  color per ``N``, with ``t_scr``; right, against ``p`` at the largest ``N``,
  one color per ``dt``. A gap near 0 means that the step scrambles enough
  for the counting model; a negative gap, that consecutive rounds repeat
  the same information (memory, redundancy).

    OUT_DIR=outputs/dt253/tfim N_VALUES="5 7 9 11" T_VALUES="0.5 1.5 5 10" \\
        SEEDS="42 43" P_VALUES="0.1 0.2 0.4 0.6 0.8" FINAL_VALUES=true \\
        BALANCED=true STATE=vector L=2000 MAX_JOBS=4 \\
        ./experiments/QELM_Q/exp_QELM_Q.sh
    OUT_DIR=outputs/dt253/haar STEP=haar_u1 T_VALUES=10 ... (same values)
    uv run python experiments/QELM_Q/plot_dt_collapse.py outputs/dt253/tfim \\
        --haar outputs/dt253/haar
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from QELM.evaluation.bayes_ridge import (
    bayes_accuracy,
    frozen_log_likelihood,
    scrambled_accuracy,
)
from QELM.utils import load_sweep
from QELM.utils.plots import (
    color_map,
    draw_curve,
    save_figure,
    sizes,
    subplots,
    sweep_curve,
)


def _load_fss() -> ModuleType:
    """Import plot_bayes_ridge_fss.py, next to this file (not a package)."""
    path = Path(__file__).resolve().with_name("plot_bayes_ridge_fss.py")
    spec = importlib.util.spec_from_file_location("plot_bayes_ridge_fss", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fss = _load_fss()
KEYS = (*fss.TFIM_KEYS, ("t", "evolution.t"), ("step", "unitary_step"))
METRIC = fss.MAIN
# Bayes rule of the frozen model on the same test records (issue #253),
# computed here from the saved records M.
FROZEN = "accuracy_M_frozen"
METRICS = (METRIC, FROZEN, "accuracy_M")
# Scrambling time of the TFIM reservoir at N_total = 11, in units of 1 / J
# (issue #243).
T_SCR = 2.25


def frozen_accuracy(run_dir: Path, n_test: int) -> float:
    """Return the accuracy of the frozen Bayes rule on the test records of a run.

    The rule is ``argmax_Q`` of
    :func:`QELM.evaluation.bayes_ridge.frozen_log_likelihood`, the Bayes rule
    when the configuration does not move between rounds; the test records
    are the last ``n_test`` ones, as in ``exp_QELM_Q``.

    Args:
        run_dir: Folder of one run, with its ``exp_QELM_Q.npz``.
        n_test: Number of test samples (``n_test`` of ``metrics.json``).
    """
    data = np.load(run_dir / "exp_QELM_Q.npz")
    M, Y = data["M"].astype(int), data["Y"]
    N = M.shape[2]
    charges = (N // 2, N // 2 - 1)
    log_lik = np.stack([frozen_log_likelihood(M, q) for q in charges], axis=1)
    return bayes_accuracy(log_lik[-n_test:], Y[-n_test:], charges)


def load_rows(sweep: Path) -> list[dict[str, Any]]:
    """:func:`load_sweep` with :data:`KEYS`, plus :data:`FROZEN` in each row."""
    rows = load_sweep(sweep, KEYS)
    for row in rows:
        row[FROZEN] = frozen_accuracy(sweep / row["run"], int(row["n_test"]))
    return rows


def dt_summaries(sweep: Path) -> dict[float, list[dict[str, Any]]]:
    """Average a sweep over the seeds, one summary per step ``dt = t / steps``.

    Args:
        sweep: Folder of an ``exp_QELM_Q.sh`` sweep, with
            ``final_measurement=true``.

    Returns:
        ``dt -> summary``, in increasing ``dt``, each summary with one row per
        ``(N, p)`` (:func:`plot_bayes_ridge_fss.summarize_tfim`) and ``dt``.
    """
    rows = load_rows(sweep)
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        dt = round(float(row["t"]) / int(row["steps"]), 12)
        groups.setdefault(dt, []).append(row)
    out = {}
    for dt in sorted(groups):
        summary = fss.summarize_tfim(groups[dt], METRICS)
        for row in summary:
            row["dt"] = dt
        out[dt] = summary
    return out


def haar_summary(sweep: Path) -> list[dict[str, Any]]:
    """Average the Haar U(1) control over the seeds, one row per ``(N, p)``.

    The Haar step ignores ``dt``, so every ``t`` of the sweep is pooled.

    Raises:
        ValueError: If a run of ``sweep`` does not use ``unitary_step=haar_u1``.
    """
    rows = load_sweep(sweep, KEYS)
    for row in rows:
        if row["step"] != "haar_u1":
            raise ValueError(
                f"{row['run']}: the control needs unitary_step=haar_u1, "
                f"got {row['step']!r}."
            )
    return fss.summarize_tfim(load_rows(sweep), METRICS)


def _point(
    rows: Sequence[dict[str, Any]], N: int, p: float, metric: str = METRIC
) -> tuple[float, float]:
    curve = sweep_curve(rows, N, "p", metric)
    (k,) = np.flatnonzero(np.isclose(curve.x, p))
    return float(curve.mean[k]), float(curve.err[k])


def deviations(
    by_dt: dict[float, list[dict[str, Any]]],
    reference: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return the gap ``A_TFIM - A_ref`` at every ``(dt, N, p)``.

    Args:
        by_dt: Output of :func:`dt_summaries`.
        reference: Output of :func:`haar_summary`, or ``None`` to compare
            with ``Phi(d' / 2)``
            (:func:`QELM.evaluation.bayes_ridge.scrambled_accuracy`), whose
            error is 0.

    Returns:
        One row per point with ``dt``, ``N``, ``p``, ``A``, ``A_ref``,
        ``gap`` and ``gap_err``; a point the reference lacks is left out.
    """
    out = []
    for dt, rows in by_dt.items():
        for row in rows:
            N, p = int(row["N"]), float(row["p"])
            A, err = _point(rows, N, p)
            if reference is None:
                charges = (N // 2, N // 2 - 1)
                ref, ref_err = (
                    scrambled_accuracy(N, p, int(row["rounds"]), charges),
                    0.0,
                )
            else:
                if not any(
                    int(r["N"]) == N and np.isclose(r["p"], p) for r in reference
                ):
                    continue
                ref, ref_err = _point(reference, N, p)
            out.append(
                {
                    "dt": dt,
                    "N": N,
                    "p": p,
                    "A": A,
                    "A_ref": ref,
                    "gap": A - ref,
                    "gap_err": float(np.hypot(err, ref_err)),
                }
            )
    return out


def mean_gap(points: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Mean of the gaps of ``points`` over ``p``, with its error.

    ``0 < p < 1`` only: at ``p = 0`` and ``p = 1`` every step gives 1/2 and 1.
    """
    gaps = np.array([q["gap"] for q in points if 0 < q["p"] < 1])
    errs = np.array([q["gap_err"] for q in points if 0 < q["p"] < 1])
    if gaps.size == 0:
        return {"gap": float("nan"), "gap_err": float("nan"), "n_p": 0}
    return {
        "gap": float(gaps.mean()),
        "gap_err": float(np.sqrt(np.sum(errs**2)) / gaps.size),
        "n_p": int(gaps.size),
    }


def _title(dt: float) -> str:
    return f"TFIM, $dt$ = {dt:g} / J"


def draw_vs_p(
    by_dt: dict[float, list[dict[str, Any]]],
    reference: list[dict[str, Any]] | None,
) -> tuple[Any, dict[str, Any]]:
    """Draw ``dt_accuracy_vs_p.png``; return the figure and its extra data."""
    n_values = sorted({n for rows in by_dt.values() for n in sizes(rows)})
    colors = color_map([str(n) for n in n_values])
    fig, axes = subplots(1, len(by_dt), figsize=(5.5 * len(by_dt), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (dt, rows) in zip(axes, by_dt.items(), strict=True):
        for n in sizes(rows):
            curve = sweep_curve(rows, n, "p", METRIC)
            draw_curve(ax, curve, color=colors[str(n)], label=f"N = {n}")
            if reference is not None and n in sizes(reference):
                ref = sweep_curve(reference, n, "p", METRIC)
                draw_curve(ax, ref, color=colors[str(n)], linestyle="--")
        if reference is not None:
            ax.plot([], [], color="0.4", ls="--", label="Haar U(1) step")
        ax.axhline(0.5, color="0.6", linestyle=":", lw=1)
        ax.set_title(_title(dt))
        ax.set_xlabel("measurement probability $p$")
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Bayes-Ridge accuracy on $Q$")
    fig.tight_layout()
    extra = {f"{dt:g}": rows for dt, rows in by_dt.items()}
    if reference is not None:
        extra["haar_u1"] = reference
    return fig, extra


def draw_deviation(gaps: list[dict[str, Any]], t_scr: float, ref: str) -> Any:
    """Draw ``dt_deviation.png`` from :func:`deviations`."""
    fig, (ax_dt, ax_p) = subplots(1, 2, figsize=(13, 5))
    n_values = sorted({q["N"] for q in gaps})
    dts = sorted({q["dt"] for q in gaps})
    colors = color_map([str(n) for n in n_values])
    for n in n_values:
        means = [
            mean_gap([q for q in gaps if q["N"] == n and q["dt"] == dt]) for dt in dts
        ]
        ax_dt.errorbar(
            dts,
            [m["gap"] for m in means],
            yerr=[m["gap_err"] for m in means],
            color=colors[str(n)],
            marker="o",
            capsize=3,
            label=f"N = {n}",
        )
    ax_dt.axvline(
        t_scr, color="0.4", ls="--", lw=1, label=f"$t_{{scr}}$ = {t_scr:g} / J"
    )
    ax_dt.axhline(0.0, color="0.6", ls=":", lw=1)
    ax_dt.set_xscale("log")
    ax_dt.set_xlabel("step $dt$ (units of 1 / J)")
    ax_dt.set_ylabel(f"mean over $p$ of $A_{{TFIM}} - A_{{{ref}}}$")
    ax_dt.set_title(f"gap to the scrambled model ({ref})")
    ax_dt.legend(fontsize=8)
    largest = n_values[-1]
    dt_colors = color_map([f"{dt:g}" for dt in dts])
    for dt in dts:
        pts = sorted(
            (q for q in gaps if q["N"] == largest and q["dt"] == dt),
            key=lambda q: q["p"],
        )
        ax_p.errorbar(
            [q["p"] for q in pts],
            [q["gap"] for q in pts],
            yerr=[q["gap_err"] for q in pts],
            color=dt_colors[f"{dt:g}"],
            marker="o",
            capsize=3,
            label=f"dt = {dt:g} / J",
        )
    ax_p.axhline(0.0, color="0.6", ls=":", lw=1)
    ax_p.set_xlabel("measurement probability $p$")
    ax_p.set_ylabel(f"$A_{{TFIM}} - A_{{{ref}}}$")
    ax_p.set_title(f"gap against $p$, N = {largest}")
    ax_p.legend(fontsize=8)
    fig.tight_layout()
    return fig


def rule_gaps(
    by_dt: dict[float, list[dict[str, Any]]],
    reference: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Mean over ``p`` of ``A_frozen - A_scrambled`` per ``N`` and ``dt``.

    ``A_scrambled`` is the Bayes-Ridge accuracy, ``A_frozen`` the frozen rule
    (:func:`frozen_accuracy`) on the same records. A positive gap means that
    the record keeps a memory between rounds that the scrambled model throws
    away. The Haar U(1) control, when given, is under ``"haar_u1"``.
    """
    sets = {f"{dt:g}": rows for dt, rows in by_dt.items()}
    if reference is not None:
        sets["haar_u1"] = reference
    out: dict[str, Any] = {}
    for key, rows in sets.items():
        for n in sizes(rows):
            points = []
            for row in (r for r in rows if int(r["N"]) == n):
                p = float(row["p"])
                a, ea = _point(rows, n, p, FROZEN)
                b, eb = _point(rows, n, p, METRIC)
                points.append(
                    {"p": p, "gap": a - b, "gap_err": float(np.hypot(ea, eb))}
                )
            out[f"N={n}, dt={key}"] = {"N": n, "dt": key, **mean_gap(points)}
    return out


def draw_rules(
    by_dt: dict[float, list[dict[str, Any]]],
    gaps: dict[str, Any],
    t_scr: float,
) -> Any:
    """Draw ``dt_two_rules.png``: the frozen rule against the scrambled one."""
    fig, (ax_dt, ax_p) = subplots(1, 2, figsize=(13, 5))
    dts = list(by_dt)
    n_values = sorted({g["N"] for g in gaps.values()})
    colors = color_map([str(n) for n in n_values])
    for n in n_values:
        means = [gaps.get(f"N={n}, dt={dt:g}") for dt in dts]
        kept = [(dt, m) for dt, m in zip(dts, means, strict=True) if m is not None]
        ax_dt.errorbar(
            [dt for dt, _ in kept],
            [m["gap"] for _, m in kept],
            yerr=[m["gap_err"] for _, m in kept],
            color=colors[str(n)],
            marker="o",
            capsize=3,
            label=f"N = {n}",
        )
        haar = gaps.get(f"N={n}, dt=haar_u1")
        if haar is not None:
            ax_dt.axhline(haar["gap"], color=colors[str(n)], ls="--", lw=1)
    if any(g["dt"] == "haar_u1" for g in gaps.values()):
        ax_dt.plot([], [], color="0.4", ls="--", label="Haar U(1) step")
    ax_dt.axvline(
        t_scr, color="0.4", ls=":", lw=1, label=f"$t_{{scr}}$ = {t_scr:g} / J"
    )
    ax_dt.axhline(0.0, color="0.6", ls=":", lw=1)
    ax_dt.set_xscale("log")
    ax_dt.set_xlabel("step $dt$ (units of 1 / J)")
    ax_dt.set_ylabel(r"mean over $p$ of $A_{frozen} - A_{scrambled}$")
    ax_dt.set_title("memory in the record: frozen rule minus Bayes-Ridge")
    ax_dt.legend(fontsize=8)
    largest = max(n_values)
    dt_colors = color_map([f"{dt:g}" for dt in dts])
    for dt, rows in by_dt.items():
        if largest not in sizes(rows):
            continue
        color = dt_colors[f"{dt:g}"]
        draw_curve(
            ax_p,
            sweep_curve(rows, largest, "p", METRIC),
            color=color,
            label=f"dt = {dt:g} / J",
        )
        draw_curve(
            ax_p, sweep_curve(rows, largest, "p", FROZEN), color=color, linestyle="--"
        )
    ax_p.plot([], [], color="0.4", ls="-", label="Bayes-Ridge (scrambled)")
    ax_p.plot([], [], color="0.4", ls="--", label="frozen Bayes rule")
    ax_p.axhline(0.5, color="0.6", ls=":", lw=1)
    ax_p.set_ylim(0.45, 1.02)
    ax_p.set_xlabel("measurement probability $p$")
    ax_p.set_ylabel("accuracy on $Q$")
    ax_p.set_title(f"two rules on the same records, N = {largest}")
    ax_p.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    """Parse the command line and draw the four figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep", type=Path, help="exp_QELM_Q sweep over T_VALUES")
    parser.add_argument(
        "--haar", type=Path, default=None, help="the same sweep with STEP=haar_u1"
    )
    parser.add_argument(
        "--t-scr", type=float, default=T_SCR, help="scrambling time, in 1 / J"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output folder (default: sweep)"
    )
    args = parser.parse_args()
    out = args.out or args.sweep
    by_dt = dt_summaries(args.sweep)
    reference = haar_summary(args.haar) if args.haar else None

    fig, extra = draw_vs_p(by_dt, reference)
    print(f"Wrote {save_figure(fig, out, 'dt_accuracy_vs_p', extra)}")
    plt.close(fig)

    panels = [(_title(dt), rows, METRIC) for dt, rows in by_dt.items()]
    if reference is not None:
        panels.append(("Haar U(1) step (control)", reference, METRIC))
    fig, extra = fss.draw_collapse(panels)
    print(f"Wrote {save_figure(fig, out, 'dt_collapse', extra)}")
    plt.close(fig)

    gaps = deviations(by_dt, reference)
    ref = "Haar" if reference is not None else r"\Phi"
    summary = {
        f"N={n}, dt={dt:g}": mean_gap(
            [q for q in gaps if q["N"] == n and q["dt"] == dt]
        )
        for n in sorted({q["N"] for q in gaps})
        for dt in by_dt
    }
    fig = draw_deviation(gaps, args.t_scr, ref)
    data = {"reference": ref, "t_scr": args.t_scr, "points": gaps, "mean_gap": summary}
    print(f"Wrote {save_figure(fig, out, 'dt_deviation', data)}")
    plt.close(fig)
    for key, m in summary.items():
        print(f"{key:<16} mean gap {m['gap']:+.4f} ± {m['gap_err']:.4f}")

    rules = rule_gaps(by_dt, reference)
    fig = draw_rules(by_dt, rules, args.t_scr)
    print(f"Wrote {save_figure(fig, out, 'dt_two_rules', rules)}")
    plt.close(fig)
    for key, m in rules.items():
        print(f"{key:<22} frozen - scrambled {m['gap']:+.4f} ± {m['gap_err']:.4f}")


if __name__ == "__main__":
    main()
