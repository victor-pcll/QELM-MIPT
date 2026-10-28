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
the seed-averaged curves, the collapse of each seed alone (its spread is
the error bar of ``p_c`` and ``nu`` quoted in the title), the collapse
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

    uv run python experiments/QELM_S/plot_entropy_fss.py outputs/exp_QELM_S/<date_time>
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from QELM.evaluation.finite_size import curve_crossings, fit_collapse, rescale
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


def load_curves(sweep_dir: str | Path) -> dict[str, dict[int, dict[int, Any]]]:
    """Return ``model -> N -> seed -> (p, S_mean)`` from the run folders.

    Raises:
        ValueError: If the folder holds no run.
    """
    rows = load_sweep(sweep_dir, keys=KEYS)
    if not rows:
        raise ValueError(f"No run in {sweep_dir}.")
    out: dict[str, dict[int, dict[int, Any]]] = {}
    for row in rows:
        cfg = OmegaConf.load(Path(sweep_dir) / row["run"] / ".hydra" / "config.yaml")
        p = [float(v) for v in cfg.entropy_task.p_values]
        S = [float(row[f"S_mean_p{v:g}"]) for v in p]
        order = np.argsort(p)
        curve = (np.asarray(p)[order], np.asarray(S)[order])
        by_n = out.setdefault(str(row["model"]), {}).setdefault(int(row["N"]), {})
        by_n[int(row["seed"])] = curve
    return {m: dict(sorted(by_n.items())) for m, by_n in sorted(out.items())}


def mean_curves(by_n: dict[int, dict[int, Any]]) -> dict[int, Any]:
    """Average the seeds: ``N -> (p, S, standard error of S)``."""
    out: dict[int, Any] = {}
    for N, seeds in by_n.items():
        p = next(iter(seeds.values()))[0]
        S = np.array([curve[1] for curve in seeds.values()])
        err = S.std(axis=0, ddof=1) / np.sqrt(len(S)) if len(S) > 1 else 0 * S[0]
        out[N] = (p, S.mean(axis=0), err)
    return out


def analyse(by_n: dict[int, dict[int, Any]]) -> dict[str, Any]:
    """Crossings and collapses of one model."""
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
    }
    if len(sizes) >= 2:
        curves = {N: (p, S) for N, (p, S, _) in mean.items()}
        result["collapse"] = fit_collapse(curves, P_RANGE, NU_RANGE)
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
) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw ``entropy_fss.png``, one row per model."""
    fig, axes = subplots(len(data), 3, figsize=(15, 3.9 * len(data)), squeeze=False)
    all_sizes = sorted({N for by_n in data.values() for N in by_n})
    colors = color_map([str(N) for N in all_sizes])
    table: dict[str, Any] = {}
    for row, (model, by_n) in zip(axes, data.items(), strict=True):
        result = analyse(by_n)
        table[model] = result | {"reference": REFERENCES.get(model)}
        for N, curve in result["curves"].items():
            p, S, e = (np.asarray(curve[k]) for k in ("p", "S", "S_err"))
            half = int(N) // 2
            c = colors[N]
            row[0].errorbar(p, S / half, yerr=e / half, marker="o", ms=3, color=c)
            row[1].errorbar(p, S, yerr=e, marker="o", ms=3, color=c, label=f"N = {N}")
        for points in result["crossings_S_over_half"].values():
            for x in points:
                row[0].axvline(x, color="0.6", lw=0.8, ls=":")
        row[0].set_ylabel(r"$\overline{S_A} / (N/2)$")
        row[1].set_ylabel(r"$\overline{S_A}$ (bits)")
        row[1].legend(frameon=False, fontsize=8)
        for ax in row[:2]:
            ax.set_xlabel("p")
            ax.set_title(model)
        fit = result.get("collapse")
        if fit is not None:
            curves = {
                int(N): (np.asarray(c["p"]), np.asarray(c["S"]))
                for N, c in result["curves"].items()
            }
            for N, (x, y) in rescale(curves, fit["p_c"], fit["nu"]).items():
                row[2].plot(x, y, "o-", ms=3, color=colors[str(N)])
            title = (
                f"collapse: p_c = {fit['p_c']:.3f}"
                + (f" ± {fit['p_c_err']:.3f}" if "p_c_err" in fit else "")
                + f", nu = {fit['nu']:.2f}"
                + (f" ± {fit['nu_err']:.2f}" if "nu_err" in fit else "")
            )
            ref = REFERENCES.get(model)
            if ref is not None:
                title += f"\npaper: p_c = {ref['p_c']} ± {ref['p_c_err']}, "
                title += f"nu = {ref['nu']} ± {ref['nu_err']}"
            row[2].set_title(title, fontsize=9)
            row[2].axvline(0.0, color="0.6", lw=0.8)
        row[2].set_xlabel(r"$(p - p_c)\, N^{1/\nu}$")
        row[2].set_ylabel(r"$S_A(p, N) - S_A(p_c, N)$")
    return fig, table


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments, draw the figure and print the estimates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="folder of an exp_QELM_S.sh sweep")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    args = parser.parse_args(argv)
    out = args.out or args.sweep
    fig, table = draw(load_curves(args.sweep))
    print(f"Wrote {save_figure(fig, out, 'entropy_fss', table)}")
    for model, result in table.items():
        print(f"{model}: crossings of S/(N/2) {result['crossings_S_over_half']}")
        if "collapse" in result:
            print(f"{model}: collapse {result['collapse']}")


if __name__ == "__main__":
    main()
