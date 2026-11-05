#!/usr/bin/env python3
"""Figures and crossings of the Clifford finite-size run (calcul 6.A, #303).

Reads the ``<regime>_N<N>_seed<seed>.json`` files of ``clifford_fss.py`` and
writes, into ``--out``:

- ``clifford_fss.png``: one row per regime (``scrambled``, ``T2N``), with five
  panels: the purified fraction ``f`` (oracle), the ridge on ``M`` over all
  trajectories, the Bayes share ``g = 2 acc_pure - 1`` of the ridge, the share
  ``w1`` of weight-1 parities, and ``g`` at ``p = 0.12, 0.16, 0.20`` against
  ``N``;
- ``clifford_fss_summary.json``: means and standard errors over circuits, and
  the crossings of consecutive sizes (same criterion as ``plot_verdict.py``).

Run from the repository root:

    uv run python experiments/qelm_verdict/plot_clifford_fss.py \
        outputs/qelm_verdict/clifford_fss --out experiments/qelm_verdict/figures
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("plot_verdict", HERE / "plot_verdict.py")
assert _spec is not None and _spec.loader is not None
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

KEYS = ("purified", "ridge_acc_all", "ridge_acc_pure", "bayes_acc_all", "weight_one")
REGIMES = {
    "scrambled": "brouillé : $2N$ pas sans mesure, puis $2N$ mesurés",
    "T2N": "$T = 2N$ pas mesurés depuis la paire de Bell",
}


def load(folder: Path) -> dict:
    """``{regime: {"p": array, N: {key: (mean, err, n_circuits)}}}``."""
    raw: dict = defaultdict(lambda: defaultdict(list))
    p_of: dict = {}
    for f in sorted(folder.glob("*_N*_seed*.json")):
        d = json.loads(f.read_text())
        raw[d["regime"]][d["N"]].append(d["rows"])
        p_of[d["regime"]] = np.array([r["p"] for r in d["rows"]])
    out: dict = {}
    for regime, by_n in raw.items():
        out[regime] = {"p": p_of[regime]}
        for n, runs in by_n.items():
            stats = {}
            for k in KEYS:
                a = np.array([[r[k] for r in rows] for rows in runs], float)
                if k == "ridge_acc_pure":
                    a = 2 * a - 1  # the Bayes share g (Bayes is 1 there)
                m = np.nanmean(a, axis=0)
                cnt = np.sum(~np.isnan(a), axis=0)
                e = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
                stats["g" if k == "ridge_acc_pure" else k] = (
                    m,
                    np.nan_to_num(e),
                    len(runs),
                )
            out[regime][n] = stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.folder)
    regimes = [r for r in REGIMES if r in data]
    fig, axes = plt.subplots(
        len(regimes), 5, figsize=(24, 4.4 * len(regimes)), squeeze=False
    )
    summary: dict = {}
    panels = (
        ("purified", "fraction purifiée $f$ (oracle)"),
        ("ridge_acc_all", "ridge sur vec($M$), toutes trajectoires"),
        ("g", "part de Bayes $g = 2\\,acc_{pur} - 1$"),
        ("weight_one", "part $w_1$ des parités de poids $\\leq 1$"),
    )
    for row, regime in enumerate(regimes):
        d = data[regime]
        p = d["p"]
        ns = sorted(k for k in d if k != "p")
        cl = dict(zip(ns, pv.colors(len(ns)), strict=True))
        s: dict = {"n_circuits": {n: d[n]["purified"][2] for n in ns}}
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            for n in ns:
                m, e, nc = d[n][key]
                ax.errorbar(
                    p,
                    m,
                    e,
                    color=cl[n],
                    marker="o",
                    ms=3,
                    capsize=2,
                    label=f"N = {n} ({nc} circuits)",
                )
            ax.axvline(pv.P_C_CLIFFORD, color="k", ls=":", lw=1)
            ax.set_title(f"{title}\n{REGIMES[regime]}", fontsize=9)
            ax.set_xlabel("$p$")
            ax.grid(alpha=0.25)
            if key in ("ridge_acc_all",):
                ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
            if key in ("g", "weight_one"):
                ax.axhline(0, color="0.5", ls="--", lw=0.8)
                ax.set_ylim(-0.15, 1.0)
            s[key] = {
                "mean": {n: d[n][key][0].tolist() for n in ns},
                "err": {n: d[n][key][1].tolist() for n in ns},
                "crossings": pv.consecutive_crossings(
                    p, {n: d[n][key][:2] for n in ns}, lo=0.05, hi=0.3
                ),
            }
        axes[row, 0].legend(fontsize=7)
        ax = axes[row, 4]
        for pp, mk in ((0.12, "s"), (0.16, "o"), (0.20, "^")):
            i = int(np.argmin(abs(p - pp)))
            ax.errorbar(
                ns,
                [d[n]["g"][0][i] for n in ns],
                [d[n]["g"][1][i] for n in ns],
                marker=mk,
                capsize=2,
                label=f"$p = {p[i]:.2f}$",
            )
        ax.axhline(0, color="0.5", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("$N$")
        ax.set_ylim(-0.15, 1.0)
        ax.set_title(f"$g$ contre $N$ près de $p_c$\n{REGIMES[regime]}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        summary[regime] = s
    fig.tight_layout()
    fig.savefig(args.out / "clifford_fss.png", dpi=150)
    plt.close(fig)
    (args.out / "clifford_fss_summary.json").write_text(
        json.dumps(summary, indent=1, default=str)
    )
    for regime, s in summary.items():
        print(regime, s["n_circuits"])
        for key in ("purified", "ridge_acc_all", "g"):
            for pair, r in s[key]["crossings"].items():
                print(
                    f"  {key:14s} {pair:6s} mean={[round(x, 3) for x in r['mean_curve_crossings']]} "  # noqa: E501
                    f"resolved={r['resolved'] and round(r['resolved']['p_star'], 3)}"
                )


if __name__ == "__main__":
    main()
