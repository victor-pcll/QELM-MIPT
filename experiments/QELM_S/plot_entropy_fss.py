#!/usr/bin/env python3
"""Finite-size scaling of the half-chain entropy of an ``exp_QELM_S.sh`` sweep.

Issue #239: does the local Floquet chain of Skinner, Ruhman and Nahum (2019)
show the volume-law to area-law transition that the all-to-all TFIM of PR
#232 does not? Reads every run folder of the sweep (one per model, ``N`` and
seed), averages ``S_mean_p<p>`` over the seeds and writes
``entropy_fss.png`` with its JSON (:func:`QELM.utils.plots.save_figure`), one
row per model:

- left: ``S_A / (N/2)`` against ``p``, one curve per ``N``, with the
  crossings of the curves of every pair of sizes
  (:func:`QELM.evaluation.finite_size.curve_crossings`);
- middle: ``S_A`` in bits against ``p``. In the volume-law phase it grows
  linearly with ``N``, in the area-law phase it stops growing;
- right: the collapse ``S(p, N) - S(p_c, N)`` against
  ``(p - p_c) N^(1/nu)`` at the best ``(p_c, nu)``
  (:func:`QELM.evaluation.finite_size.fit_collapse`), the method of Sec. IV
  of the paper.

The JSON holds, per model, the mean curves, the crossings, the collapse of
the seed-averaged curves with its statistical error (a bootstrap over the
trajectories, ``p_c_stat``, ``nu_stat``) and its systematic error (half the
range of the fits over the choices of the smallest size and of the ``p``
window, ``p_c_sys``, ``nu_sys``, the fits in ``collapse_choices``; issue
#251), both quoted in the title, the collapse of each seed alone (the
spread of the seeds, ``p_c_err`` and ``nu_err``), the collapse
restricted to ``|p - p_c| <= 0.25`` and ``0.15`` (at ``N <= 12`` the
estimate moves with the window, the main finite-size caveat), and the collapse of
the sizes with ``N / 2`` even and with ``N / 2`` odd apart. A brickwork
alternates even and odd bonds, so whether the bond across the cut got its
gate in the last half-layer depends on the parity of ``N / 2`` (and, for
``floquet_circuit``, is fixed by ``exp_QELM_S.floquet_rounds``); with full
periods per round (``haar_circuit``, ``U1_circuit``) the two families of
sizes are offset from each other. The paper's values
for the Floquet model (von Neumann entropy ``S_1``, Table I) are
``p_c = 0.21 +- 0.16`` and ``nu = 1.74 +- 0.45``.

A fourth column draws the cost of the collapse in the ``(p_c, nu)`` plane,
on a log scale, with its minimum and the paper's point. At these sizes the
cost has a long diagonal valley: raising ``p_c`` and ``nu`` together costs
almost nothing, so the window, the sizes or the noise move the minimum
along it (issue #251).

Options (issue #251), for a sweep run with ``AVERAGE_FROM`` set:

- ``--entropy``: the curve to collapse, ``S`` (the final von Neumann
  entropy, default) or a late-time average, ``S1_avg``, ``S2_avg`` or
  ``Sinf_avg`` (the rows of Table I of the paper are ``S_1``, ``S_2`` and
  ``S_inf``);
- ``--min-N``: leave the smaller sizes out of the fit, they carry the
  largest corrections to scaling;
- ``--p-min``: leave the points with ``p < p_min`` out of the fit, the
  volume-law side far from ``p_c``.

When the runs hold ``I3_avg``, the tripartite mutual information of the
four quarters (``N`` a multiple of 4), the script also writes
``i3_crossings.png`` with its JSON: ``I_3`` against ``p`` for each ``N``,
and the crossings of every pair of sizes. ``I_3`` needs no fit: its curves
cross near ``p_c`` with a small drift in ``N`` (Zabalo et al., PRB 101,
060301 (2020)).

    uv run python experiments/QELM_S/plot_entropy_fss.py outputs/exp_QELM_S/<date_time>
    uv run python experiments/QELM_S/plot_entropy_fss.py <sweep> \
        --entropy S1_avg --min-N 8 --p-min 0.05
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from QELM.evaluation.finite_size import (
    bootstrap_collapse,
    collapse_cost,
    curve_crossings,
    fit_collapse,
    rescale,
)
from QELM.utils.plots import color_map, save_figure, subplots
from QELM.utils.sweep import load_sweep

KEYS = (("N", "N"), ("seed", "seed"), ("model", "models"))
# Search box of the collapse.
P_RANGE = (0.05, 0.45)
NU_RANGE = (0.5, 4.0)
# Half-widths |p - p_c| of the windows of the collapses near p_c.
WINDOWS = (0.25, 0.15)
# Table I of Skinner, Ruhman and Nahum, PRX 9, 031009 (2019), S_1.
REFERENCES = {
    "floquet_circuit": {"p_c": 0.21, "p_c_err": 0.16, "nu": 1.74, "nu_err": 0.45},
    "haar_circuit": {"p_c": 0.26, "p_c_err": 0.08, "nu": 2.01, "nu_err": 0.10},
}
# The other Rényi indices of the Floquet model in Table I, by --entropy.
REFERENCES_RENYI = {
    "S2_avg": {
        "floquet_circuit": {"p_c": 0.22, "p_c_err": 0.05, "nu": 2.07, "nu_err": 0.0}
    },
    "Sinf_avg": {
        "floquet_circuit": {"p_c": 0.21, "p_c_err": 0.05, "nu": 2.03, "nu_err": 0.0}
    },
}
# Curves --entropy can collapse: the final S or a late-time average.
ENTROPIES = ("S", "S1_avg", "S2_avg", "Sinf_avg")
# Points per axis of the cost landscape.
LANDSCAPE_GRID = 61
# Choices of fit_choices: p_min is also raised by this, and the p above
# this cut are also left out.
P_MIN_SHIFT = 0.05
P_MAX_CUT = 0.3
# Resamples of the bootstrap of the collapse.
N_BOOT = 100


def load_curves(
    sweep_dir: str | Path, key: str = "S", errors: bool = False
) -> dict[str, dict[int, dict[int, Any]]]:
    """Return ``model -> N -> seed -> (p, <key>_mean)`` from the run folders.

    With ``errors``, the second entry is instead the standard error of that
    mean, ``<key>_std / sqrt(samples_per_p)``, from the spread of the
    trajectories of the run. Runs without ``<key>_mean_p<p>`` (no late-time
    averages, or ``I3`` for ``N`` not a multiple of 4) are skipped.

    Raises:
        ValueError: If the folder holds no run.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    if not rows:
        raise ValueError(f"No run in {sweep_dir}.")
    stat = "std" if errors else "mean"
    out: dict[str, dict[int, dict[int, Any]]] = {}
    for row in rows:
        cfg = OmegaConf.load(Path(sweep_dir) / row["run"] / ".hydra" / "config.yaml")
        p = [float(v) for v in cfg.entropy_task.p_values]
        names = [f"{key}_{stat}_p{v:g}" for v in p]
        if any(name not in row for name in names):
            continue
        S = np.array([float(row[name]) for name in names])
        if errors:
            S /= np.sqrt(int(cfg.entropy_task.samples_per_p))
        order = np.argsort(p)
        curve = (np.asarray(p)[order], S[order])
        by_n = out.setdefault(str(row["model"]), {}).setdefault(int(row["N"]), {})
        by_n[int(row["seed"])] = curve
    return {m: dict(sorted(by_n.items())) for m, by_n in sorted(out.items())}


def restrict(
    by_n: dict[int, dict[int, Any]],
    min_N: int = 0,
    p_min: float = 0.0,
    p_max: float = np.inf,
) -> dict[int, dict[int, Any]]:
    """Keep the sizes ``N >= min_N`` and the points ``p_min <= p <= p_max``."""
    out: dict[int, dict[int, Any]] = {}
    for N, seeds in by_n.items():
        if N < min_N:
            continue
        out[N] = {}
        for seed, (p, S) in seeds.items():
            p = np.asarray(p)
            keep = (p >= p_min - 1e-12) & (p <= p_max + 1e-12)
            out[N][seed] = (p[keep], np.asarray(S)[keep])
    return out


def pooled_errors(errors: dict[int, dict[int, Any]]) -> dict[int, np.ndarray]:
    """Standard error of the seed average, ``sqrt(sum_s e_s**2) / n_seeds``."""
    return {
        N: np.sqrt(sum(np.asarray(e) ** 2 for _, e in seeds.values())) / len(seeds)
        for N, seeds in errors.items()
    }


def fit_choices(
    by_n: dict[int, dict[int, Any]], min_N: int = 0, p_min: float = 0.0
) -> list[dict[str, Any]]:
    """Collapse fits over reasonable choices of the data (issue #251).

    Around the choice ``(min_N, p_min)``: the smallest size kept is
    ``min_N`` or one of the next two sizes (as long as three sizes remain),
    ``p_min`` or ``p_min + P_MIN_SHIFT``, and all the ``p`` above or only
    ``p <= P_MAX_CUT`` (when at least two points of the grid lie above it).
    The spread of the fits is the systematic error of the collapse.
    """
    sizes = sorted(N for N in by_n if N >= min_N)
    starts = [sizes[k] for k in range(3) if len(sizes) - k >= 3] or sizes[:1]
    grid = np.unique(np.concatenate([p for s in by_n.values() for p, _ in s.values()]))
    cuts = [np.inf] + ([P_MAX_CUT] if np.sum(grid > P_MAX_CUT) >= 2 else [])
    fits = []
    for first in starts:
        for low in (p_min, round(p_min + P_MIN_SHIFT, 10)):
            for high in cuts:
                curves = {
                    N: (p, S)
                    for N, (p, S, _) in mean_curves(
                        restrict(by_n, first, low, high)
                    ).items()
                }
                if len(curves) < 2:
                    continue
                choice = {"min_N": first, "p_min": low, "p_max": float(high)}
                fits.append(choice | fit_collapse(curves, P_RANGE, NU_RANGE))
    return fits


def cost_landscape(
    curves: dict[int, Any], grid: int = LANDSCAPE_GRID
) -> dict[str, list[Any]]:
    """Return the collapse cost on a ``(p_c, nu)`` grid over the search box."""
    p_c = np.linspace(*P_RANGE, grid)
    nu = np.linspace(*NU_RANGE, grid)
    cost = [[collapse_cost(curves, a, b) for a in p_c] for b in nu]
    return {"p_c": p_c.tolist(), "nu": nu.tolist(), "cost": cost}


def mean_curves(by_n: dict[int, dict[int, Any]]) -> dict[int, Any]:
    """Average the seeds: ``N -> (p, S, standard error of S)``."""
    out: dict[int, Any] = {}
    for N, seeds in by_n.items():
        p = next(iter(seeds.values()))[0]
        S = np.array([curve[1] for curve in seeds.values()])
        err = S.std(axis=0, ddof=1) / np.sqrt(len(S)) if len(S) > 1 else 0 * S[0]
        out[N] = (p, S.mean(axis=0), err)
    return out


def analyse(
    by_n: dict[int, dict[int, Any]],
    min_N: int = 0,
    p_min: float = 0.0,
    errors: dict[int, dict[int, Any]] | None = None,
) -> dict[str, Any]:
    """Crossings of every size and collapses of the sizes ``N >= min_N``.

    The collapses keep only the points with ``p >= p_min``. The error of
    the collapse has two parts (issue #251):

    - statistical, ``p_c_stat`` and ``nu_stat``: a parametric bootstrap
      (:func:`QELM.evaluation.finite_size.bootstrap_collapse`) with the
      standard error of each point, from the spread of the trajectories
      (``errors``, ``N -> seed -> (p, error)``) or, without it, from the
      spread of the seeds;
    - systematic, ``p_c_sys`` and ``nu_sys``: half the range of ``p_c`` and
      ``nu`` over the fits of :func:`fit_choices`, listed in
      ``collapse_choices``.

    ``p_c_total`` and ``nu_total`` add the two in quadrature.
    """
    mean = mean_curves(by_n)
    sizes = sorted(mean)
    crossings = {
        f"N{a}-N{b}": curve_crossings(
            mean[a][0], mean[a][1] / (a // 2), mean[b][1] / (b // 2)
        )
        for k, a in enumerate(sizes)
        for b in sizes[k + 1 :]
    }
    result: dict[str, Any] = {
        "sizes": sizes,
        "curves": {
            str(N): {"p": p.tolist(), "S": S.tolist(), "S_err": e.tolist()}
            for N, (p, S, e) in mean.items()
        },
        "crossings_S_over_half": crossings,
        "fit": {"min_N": min_N, "p_min": p_min},
    }
    full = by_n
    by_n = restrict(by_n, min_N, p_min)
    sizes = sorted(by_n)
    if len(sizes) >= 2:
        mean = mean_curves(by_n)
        curves = {N: (p, S) for N, (p, S, _) in mean.items()}
        result["collapse"] = fit_collapse(curves, P_RANGE, NU_RANGE)
        result["landscape"] = cost_landscape(curves)
        point_errors = (
            pooled_errors(restrict(errors, min_N, p_min))
            if errors is not None and set(restrict(errors, min_N, p_min)) == set(mean)
            else {N: e for N, (_, _, e) in mean.items()}
        )
        result["collapse"] |= bootstrap_collapse(
            curves, point_errors, P_RANGE, NU_RANGE, n_boot=N_BOOT
        )
        choices = fit_choices(full, min_N, p_min)
        result["collapse_choices"] = choices
        for key in ("p_c", "nu"):
            values = [f[key] for f in choices]
            sys = (max(values) - min(values)) / 2
            stat = result["collapse"][f"{key}_stat"]
            result["collapse"][f"{key}_sys"] = sys
            result["collapse"][f"{key}_total"] = float(np.hypot(stat, sys))
        seeds = sorted(set.intersection(*(set(s) for s in by_n.values())))
        per_seed = [
            fit_collapse({N: by_n[N][s] for N in sizes}, P_RANGE, NU_RANGE)
            for s in seeds
        ]
        result["collapse_per_seed"] = {str(s): f for s, f in zip(seeds, per_seed)}
        # The scaling form holds near p_c only: how p_c and nu move when the
        # fit keeps only the points with |p - p_c| <= window.
        result["collapse_by_window"] = {
            f"{w:g}": fit_collapse(curves, P_RANGE, NU_RANGE, window=w) for w in WINDOWS
        }
        # The brickwork alternates even and odd bonds, so the bond across the
        # cut is in one half-layer or the other with the parity of N / 2:
        # the collapse of each family of sizes alone.
        families = {
            parity: [N for N in sizes if (N // 2) % 2 == parity] for parity in (0, 1)
        }
        result["collapse_by_half_parity"] = {
            f"N/2 {'even' if parity == 0 else 'odd'}": {
                "sizes": family,
                **fit_collapse({N: curves[N] for N in family}, P_RANGE, NU_RANGE),
            }
            for parity, family in families.items()
            if len(family) >= 2
        }
        if len(per_seed) > 1:
            for key in ("p_c", "nu"):
                values = np.array([f[key] for f in per_seed])
                result["collapse"][f"{key}_err"] = float(
                    values.std(ddof=1) / np.sqrt(values.size)
                )
    return result


def draw(
    data: dict[str, dict[int, dict[int, Any]]],
    entropy: str = "S",
    min_N: int = 0,
    p_min: float = 0.0,
    errors: dict[str, dict[int, dict[int, Any]]] | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw ``entropy_fss.png``, one row per model."""
    fig, axes = subplots(len(data), 4, figsize=(20, 3.9 * len(data)), squeeze=False)
    all_sizes = sorted({N for by_n in data.values() for N in by_n})
    colors = color_map([str(N) for N in all_sizes])
    references = REFERENCES_RENYI.get(entropy, REFERENCES)
    table: dict[str, Any] = {}
    for row, (model, by_n) in zip(axes, data.items(), strict=True):
        result = analyse(by_n, min_N, p_min, (errors or {}).get(model))
        ref = references.get(model)
        table[model] = result | {"entropy": entropy, "reference": ref}
        for N, curve in result["curves"].items():
            p, S, e = (np.asarray(curve[k]) for k in ("p", "S", "S_err"))
            half = int(N) // 2
            c = colors[N]
            row[0].errorbar(p, S / half, yerr=e / half, marker="o", ms=3, color=c)
            row[1].errorbar(p, S, yerr=e, marker="o", ms=3, color=c, label=f"N = {N}")
        for points in result["crossings_S_over_half"].values():
            for x in points:
                row[0].axvline(x, color="0.6", lw=0.8, ls=":")
        row[0].set_ylabel(rf"$\overline{{{entropy}}} / (N/2)$".replace("_", r"\_"))
        row[1].set_ylabel(rf"$\overline{{{entropy}}}$ (bits)".replace("_", r"\_"))
        row[1].legend(frameon=False, fontsize=8)
        for ax in row[:2]:
            ax.set_xlabel("p")
            ax.set_title(model)
        fit = result.get("collapse")
        if fit is not None:
            curves = {
                N: (p, S)
                for N, (p, S, _) in mean_curves(restrict(by_n, min_N, p_min)).items()
            }
            for N, (x, y) in rescale(curves, fit["p_c"], fit["nu"]).items():
                row[2].plot(x, y, "o-", ms=3, color=colors[str(N)])
            title = (
                f"collapse N >= {min(curves)}, p >= {p_min:g}:\n"
                f"p_c = {fit['p_c']:.3f} ± {fit['p_c_stat']:.3f} (stat)"
                f" ± {fit['p_c_sys']:.3f} (sys), "
                f"nu = {fit['nu']:.2f} ± {fit['nu_stat']:.2f} (stat)"
                f" ± {fit['nu_sys']:.2f} (sys)"
            )
            if ref is not None:
                title += f"\npaper: p_c = {ref['p_c']} ± {ref['p_c_err']}, "
                title += f"nu = {ref['nu']} ± {ref['nu_err']}"
            row[2].set_title(title, fontsize=9)
            row[2].axvline(0.0, color="0.6", lw=0.8)
            land = result["landscape"]
            levels = np.log10(np.maximum(np.asarray(land["cost"]), 1e-12))
            mesh = row[3].contourf(land["p_c"], land["nu"], levels, 30, cmap="magma_r")
            fig.colorbar(mesh, ax=row[3], label="log10 cost")
            row[3].plot(fit["p_c"], fit["nu"], "c*", ms=12, label="minimum")
            if ref is not None:
                row[3].errorbar(
                    ref["p_c"],
                    ref["nu"],
                    xerr=ref["p_c_err"],
                    yerr=ref["nu_err"],
                    fmt="ws",
                    mec="k",
                    capsize=3,
                    label="paper",
                )
            row[3].legend(frameon=False, fontsize=8)
        row[2].set_xlabel(r"$(p - p_c)\, N^{1/\nu}$")
        row[2].set_ylabel(r"$S(p, N) - S(p_c, N)$")
        row[3].set_xlabel("p_c")
        row[3].set_ylabel("nu")
        row[3].set_title("cost of the collapse", fontsize=9)
    return fig, table


def draw_i3(
    data: dict[str, dict[int, dict[int, Any]]],
) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw ``i3_crossings.png``: ``I_3`` against ``p``, one panel per model."""
    fig, axes = subplots(1, len(data), figsize=(5 * len(data), 3.9), squeeze=False)
    all_sizes = sorted({N for by_n in data.values() for N in by_n})
    colors = color_map([str(N) for N in all_sizes])
    table: dict[str, Any] = {}
    for ax, (model, by_n) in zip(axes[0], data.items(), strict=True):
        mean = mean_curves(by_n)
        sizes = sorted(mean)
        crossings = {
            f"N{a}-N{b}": curve_crossings(mean[a][0], mean[a][1], mean[b][1])
            for k, a in enumerate(sizes)
            for b in sizes[k + 1 :]
        }
        table[model] = {
            "curves": {
                str(N): {"p": p.tolist(), "I3": y.tolist(), "I3_err": e.tolist()}
                for N, (p, y, e) in mean.items()
            },
            "crossings": crossings,
        }
        for N, (p, y, e) in mean.items():
            ax.errorbar(p, y, yerr=e, marker="o", ms=3, color=colors[str(N)])
            ax.lines[-1].set_label(f"N = {N}")
        for points in crossings.values():
            for x in points:
                ax.axvline(x, color="0.6", lw=0.8, ls=":")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set(xlabel="p", ylabel=r"$\overline{I_3}$ (bits)", title=model)
        ax.legend(frameon=False, fontsize=8)
    return fig, table


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments, draw the figures and print the estimates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="folder of an exp_QELM_S.sh sweep")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument(
        "--entropy", choices=ENTROPIES, default="S", help="curve to collapse"
    )
    parser.add_argument("--min-N", type=int, default=0, help="smallest N of the fit")
    parser.add_argument(
        "--p-min", type=float, default=0.0, help="smallest p of the fit"
    )
    args = parser.parse_args(argv)
    out = args.out or args.sweep
    data = load_curves(args.sweep, args.entropy)
    if not data:
        raise ValueError(
            f"No run of {args.sweep} holds {args.entropy}_mean_p<p>: "
            "run the sweep with AVERAGE_FROM set (entropy_task.average_from)."
        )
    errors = load_curves(args.sweep, args.entropy, errors=True)
    fig, table = draw(data, args.entropy, args.min_N, args.p_min, errors)
    print(f"Wrote {save_figure(fig, out, 'entropy_fss', table)}")
    for model, result in table.items():
        print(f"{model}: crossings of S/(N/2) {result['crossings_S_over_half']}")
        if "collapse" in result:
            print(f"{model}: collapse {result['collapse']}")
    i3 = load_curves(args.sweep, "I3_avg")
    if i3:
        fig, table = draw_i3(i3)
        print(f"Wrote {save_figure(fig, out, 'i3_crossings', table)}")
        for model, result in table.items():
            print(f"{model}: crossings of I_3 {result['crossings']}")


if __name__ == "__main__":
    main()
