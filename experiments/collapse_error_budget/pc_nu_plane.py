#!/usr/bin/env python3
"""(p_c, nu) planes for Clifford and Haar: three observables, 1.3 Q_min bars.

For each circuit, three ways to get ``(p_c, nu)`` from the data of the
project, all after ``T = 2N`` half-layers of the periodic brickwork with
``Z`` measurements:

- ``Delta = N - log2(1 + W(m))`` (purification; #346 Clifford, #348 Haar);
- ``I3`` of the four quarters of the references (#358);
- ``S(N/2)`` of the references, collapse of ``S(p, N) - S(p_c, N)`` (#358).

Haar also has ``Delta`` at ``T = 4N`` (#348), the time of Zabalo et al.

Each optimum minimizes the exact Houdayer-Hartmann quality ``Q``
(:func:`QELM.evaluation.finite_size_scaling.hh_quality`). The error bar is
the projected half-width of the region ``Q <= 1.3 Q_min`` (rule of Zabalo et
al., PRB 101, 060301(R), 2020), whose contour is drawn too. Literature: only
the same circuit (Clifford: Gullans & Huse 2020; Haar: Zabalo et al. 2020).

The ``#358`` runs (14 MB of per-sample values) are not in git; pass their
folder with ``--bipartite-runs``. Run from the repository root:

    uv run python experiments/collapse_error_budget/pc_nu_plane.py \\
        --bipartite-runs outputs/bipartite_entropy

``--plot-only`` redraws the figures from ``figures/pc_nu_plane_full.json``
without refitting (the fits take about 15 minutes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from error_budget import GRID, HAAR_RUNS, fit, load_curves  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from QELM.evaluation.finite_size_scaling import Curve, hh_quality  # noqa: E402

HERE = Path(__file__).resolve().parent
CLIFFORD_RUNS = HERE.parent / "clifford_collapse_map" / "sources"
FACTOR = 1.3

LIT: dict[str, list[tuple[str, float, float, float, float, str]]] = {
    "clifford": [
        ("Gullans & Huse, PRX 2020: I3 of quarters (and S(ρ)/L), N = 128-512",
         0.1593, 0.0005, 1.28, 0.02, "v"),
        ("Gullans & Huse, PRL 2020: S_Q of the reference qubit, N = 64-256",
         0.1598, 0.0005, 1.30, 0.05, "<"),
    ],
    "haar": [
        ("Zabalo et al. 2020: I3 of quarters, von Neumann (n = 1), N = 8-24, T = 4N",
         0.168, 0.005, 1.2, 0.2, "v"),
        ("Zabalo et al. 2020: I3 of quarters, Rényi-2, N = 8-24, T = 4N",
         0.162, 0.003, 1.3, 0.1, "<"),
    ],
}  # fmt: skip
MINE = "#1f77b4"
LITC = "#ff7f0e"
MARKERS = ["o", "s", "D", "^"]
STYLES = ["-", "--", ":", "-."]


def bipartite_curves(
    folder: Path,
    model: str,
    key: str,
    window: tuple[float, float],
    multiple: int = 1,
) -> dict[int, Curve]:
    """Mean and SEM per ``(N, p)`` of the per-sample values of #358.

    Runs of the same ``N`` with different seeds are pooled. ``multiple``
    keeps only the sizes divisible by it (``I3`` of quarters needs ``4 | N``).
    """
    acc: dict[int, dict[float, list[float]]] = {}
    for path in sorted(folder.glob(f"{model}_N*_T*_s*.json")):
        r = json.loads(path.read_text())
        n = int(r["n"])
        if n % multiple:
            continue
        for row in r["per_p"]:
            acc.setdefault(n, {}).setdefault(float(row["p"]), []).extend(row[key])
    out: dict[int, Curve] = {}
    for n, per in sorted(acc.items()):
        ps = np.array(
            sorted(q for q in per if window[0] - 1e-9 <= q <= window[1] + 1e-9)
        )
        v = [np.asarray(per[q], float) for q in ps]
        y = np.array([x.mean() for x in v])
        e = np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in v])
        keep = e > 0
        out[n] = (ps[keep], y[keep], e[keep])
    return out


def clifford_delta(folder: Path, sizes: tuple[int, ...] = (16, 32, 64, 128)):
    """``Delta`` of the Clifford transition runs of #346 at ``T = 2N``."""
    out: dict[int, Curve] = {}
    for n in sizes:
        rows = json.loads((folder / f"transition_N{n}_T{2 * n}_s0.json").read_text())
        p = np.array([r["p"] for r in rows["per_p"]])
        y = np.array([r["delta"][0] for r in rows["per_p"]])
        e = np.array([r["delta"][1] for r in rows["per_p"]])
        keep = (y > 0) & (e > 0)
        o = np.argsort(p[keep])
        out[n] = (p[keep][o], y[keep][o], e[keep][o])
    return out


def region(
    curves: dict[int, Curve],
    subtract: bool,
    best: tuple[float, float, float],
    half: tuple[float, float] = (0.02, 0.6),
    n: int = 51,
) -> tuple[float, float, dict[str, Any]]:
    """Region ``Q <= 1.3 Q_min`` on a grid refined to its extent (three passes).

    Returns the projected half-widths in ``p_c`` and ``nu`` and the grid.
    """
    wp = wn = 0.0
    for _ in range(3):
        pcs = np.linspace(best[0] - half[0], best[0] + half[0], n)
        nus = np.linspace(max(best[1] - half[1], 0.2), best[1] + half[1], n)
        q = np.array([[hh_quality(curves, a, b, subtract) for b in nus] for a in pcs])
        ii, jj = np.nonzero(q <= FACTOR * best[2])
        wp = float(np.ptp(pcs[ii]) / 2) if len(ii) else 0.0
        wn = float(np.ptp(nus[jj]) / 2) if len(jj) else 0.0
        edge = n - 1
        touches = ii.min() == 0 or ii.max() == edge or jj.min() == 0 or jj.max() == edge
        if touches:
            half = (half[0] * 2, half[1] * 2)
        elif wp < half[0] / 6 or wn < half[1] / 6:
            half = (max(3 * wp, 4 * half[0] / n), max(3 * wn, 4 * half[1] / n))
        else:
            break
    return wp, wn, {"p_c": pcs.tolist(), "nu": nus.tolist(), "S": q.tolist()}


def methods(bip: Path) -> dict[str, list[tuple[str, dict[int, Curve], bool]]]:
    """``(label, curves, subtract)`` of every point of each plane."""
    return {
        "clifford": [
            ("Δ = N − log2(1+W(m)), N = 16-128", clifford_delta(CLIFFORD_RUNS), False),
            ("I3 of the reference quarters, N = 16-256, 0.10 ≤ p ≤ 0.24",
             bipartite_curves(bip, "clifford", "i3", (0.10, 0.24)), False),
            ("S(N/2) − S(p_c, N), N = 16-256, 0.10 ≤ p ≤ 0.24",
             bipartite_curves(bip, "clifford", "half_s1", (0.10, 0.24)), True),
        ],
        "haar": [
            ("Δ = N − log2(1+W(m)), N = 4-12", load_curves(HAAR_RUNS, 2), False),
            ("I3 of the reference quarters (n = 1), N = 8-20",
             bipartite_curves(bip, "haar", "i3", (0.0, 1.0), multiple=4), False),
            ("S(N/2) − S(p_c, N) (n = 1), N = 8-20, 0.10 ≤ p ≤ 0.30 : "
             "unconstrained (islands)",
             bipartite_curves(bip, "haar", "half_s1", (0.10, 0.30)), True),
            ("Δ at T = 4N, N = 4-10 (same time as Zabalo)",
             load_curves(HAAR_RUNS, 4), False),
        ],
    }  # fmt: skip


def plot(results: dict[str, list[dict[str, Any]]], out: Path) -> None:
    """One plane per circuit, legend on the right in two groups."""
    titles = {
        "clifford": "Clifford, periodic brickwork, Z measurements, T = 2N",
        "haar": "Haar, periodic brickwork, Z measurements, T = 2N (unless stated)",
    }
    for model, rows in results.items():
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        blank = Line2D([], [], ls="none", marker="")
        handles: list[Line2D] = [blank]
        labels = ["This work (blue)"]
        for k, row in enumerate(rows):
            reg = row["region"]
            ax.contour(reg["p_c"], reg["nu"], np.array(reg["S"]).T,
                       levels=[FACTOR * row["S_min"]], colors=[MINE],
                       linewidths=1, linestyles=[STYLES[k]], alpha=0.5)  # fmt: skip
            face = "white" if "4N" in row["label"] else MINE
            ax.errorbar(row["p_c"], row["nu"], xerr=row["p_c_err_1.3"],
                        yerr=row["nu_err_1.3"], fmt=MARKERS[k], ms=8, color=MINE,
                        mfc=face, mec=MINE, capsize=3, lw=1.4)  # fmt: skip
            handles.append(Line2D([], [], ls="none", marker=MARKERS[k], ms=9,
                                  color=MINE, mfc=face, mec=MINE, mew=1.5))  # fmt: skip
            labels.append(
                f"{row['label']}\np_c = {row['p_c']:.4f} ± {row['p_c_err_1.3']:.4f}   "
                f"ν = {row['nu']:.2f} ± {row['nu_err_1.3']:.2f}   "
                f"Q_min = {row['S_min']:.1f}"
            )
        handles.append(blank)
        labels.append("Literature, same circuit (orange)")
        for lab, pc, dpc, nu, dnu, m in LIT[model]:
            ax.errorbar(pc, nu, xerr=dpc, yerr=dnu, fmt=m, ms=8, color=LITC,
                        mec=LITC, capsize=3, lw=1.4)  # fmt: skip
            handles.append(Line2D([], [], ls="none", marker=m, ms=9, color=LITC))
            labels.append(f"{lab}\np_c = {pc} ± {dpc}   ν = {nu} ± {dnu}")
        ax.set_xlabel("p_c")
        ax.set_ylabel("ν")
        ax.set_title(f"{titles[model]}\nblue: this work, orange: literature; "
                     "bars and contours: region Q ≤ 1.3 Q_min (Houdayer-Hartmann)",
                     fontsize=10)  # fmt: skip
        ax.grid(alpha=0.3)
        leg = ax.legend(handles, labels, fontsize=8.5, loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), frameon=False, labelspacing=1.1,
                        handletextpad=0.8, borderaxespad=0.0)  # fmt: skip
        for text in leg.get_texts():
            if text.get_text().startswith(("This work", "Literature")):
                text.set_fontweight("bold")
                text.set_fontsize(9.5)
        fig.savefig(out / f"pc_nu_{model}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bipartite-runs", type=Path,
                    default=Path("outputs/bipartite_entropy"))  # fmt: skip
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from pc_nu_plane_full.json, no refit")  # fmt: skip
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    full = a.out / "pc_nu_plane_full.json"
    if a.plot_only:
        plot(json.loads(full.read_text()), a.out)
        return
    results: dict[str, list[dict[str, Any]]] = {}
    for model, items in methods(a.bipartite_runs).items():
        results[model] = []
        for label, curves, sub in items:
            best = fit(curves, None, GRID, subtract=sub)
            wp, wn, reg = region(curves, sub, best)
            results[model].append(
                {"label": label, "sizes": list(curves), "subtract": sub,
                 "p_c": best[0], "nu": best[1], "S_min": best[2],
                 "p_c_err_1.3": wp, "nu_err_1.3": wn, "region": reg}
            )  # fmt: skip
            print(f"{model:8s} {label[:45]:45s} p_c = {best[0]:.4f} ± {wp:.4f}  "
                  f"nu = {best[1]:.3f} ± {wn:.3f}  Q_min = {best[2]:.2f}",
                  flush=True)  # fmt: skip
    full.write_text(json.dumps(results) + "\n")
    slim: dict[str, Any] = {
        m: [{k: v for k, v in r.items() if k != "region"} for r in rows]
        for m, rows in results.items()
    }
    slim["literature"] = LIT
    (a.out / "pc_nu_plane.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False) + "\n"
    )
    plot(results, a.out)


if __name__ == "__main__":
    main()
