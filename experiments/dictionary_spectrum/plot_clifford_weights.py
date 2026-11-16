#!/usr/bin/env python3
"""Figure of the weight histogram N(l) of the Clifford effect group (issue #356).

Reads ``clifford_weights_N*.json`` and writes ``clifford_weights.png`` and
its data in the output folder.

Usage::

    uv run python experiments/dictionary_spectrum/plot_clifford_weights.py outputs/dictionary_spectrum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from QELM.evaluation.dictionary_spectrum import (  # noqa: E402
    mean_weight,
    product_z_reference,
    random_strings_null,
)
from QELM.utils.plots.io import save_figure  # noqa: E402
from QELM.utils.plots.style import subplots  # noqa: E402

P_C = 0.161
PICKS = {0.04: "#1f4e99", 0.16: "#8a3fa0", 0.3: "#c0392b"}
SIZE_COLORS = {12: "#9ecae1", 16: "#4292c6", 20: "#08519c", 24: "#08306b"}
LOCAL = 3


def load(folder: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(folder.glob("clifford_weights_N*.json")):
        d = json.loads(path.read_text())
        out[int(d["n"])] = d
    return dict(sorted(out.items()))


def hist(d: dict, key: str, p: float) -> tuple[np.ndarray, np.ndarray]:
    j = int(np.argmin(np.abs(np.asarray(d["p"]) - p)))
    return np.asarray(d[key]["mean"][j]), np.asarray(d[key]["sem"][j])


def draw_row(axes: np.ndarray, d: dict) -> dict:
    n = int(d["n"])
    ell = np.arange(n + 1)
    null_all, _ = random_strings_null(n, n)
    prod = product_z_reference(n)
    table: dict = {}
    ax_lin, ax_log, ax_mid = axes
    for p, c in PICKS.items():
        h, _ = hist(d, "full_all", p)
        hv, _ = hist(d, "full_vis", p)
        hm, _ = hist(d, "mid_all", p)
        tot_mid = hm.sum()
        ax_lin.plot(ell, h / 2.0**n, "-o", color=c, ms=4, label=f"p = {p}")
        ax_log.semilogy(ell, h, "-o", color=c, ms=4, label=f"p = {p}, all")
        ax_log.semilogy(ell, hv, "--", color=c, lw=1.2)
        ax_mid.semilogy(
            ell,
            np.where(hm > 0, hm, np.nan),
            "-o",
            color=c,
            ms=4,
            label=f"p = {p}: sum = 2^{np.log2(tot_mid):.1f}",
        )
        table[str(p)] = {
            "full_all": h.tolist(),
            "full_visible": hv.tolist(),
            "mid_all": hm.tolist(),
            "mid_sum_log2": float(np.log2(tot_mid)),
            "mean_weight_full": mean_weight(h),
            "peak_full": int(np.argmax(h)),
        }
    ax_lin.plot(ell, null_all / 2.0**n, "k--", lw=1, label="random strings (3N/4)")
    ax_lin.plot(ell, prod / 2.0**n, ":", color="grey", lw=1.5, label="all Z (N/2)")
    ax_log.semilogy(ell, null_all, "k--", lw=1, label="random strings")
    ax_log.semilogy(ell, prod, ":", color="grey", lw=1.5, label="all Z")
    ax_lin.set_ylabel("N(l) / 2^N")
    ax_lin.set_title(f"N = {n}, final readout")
    ax_log.set_ylabel("N(l)")
    ax_log.set_ylim(0.1, None)
    ax_log.set_title(f"N = {n}, final readout (dashed: no Y)")
    ax_mid.set_ylabel("N(l)")
    ax_mid.set_ylim(0.1, None)
    ax_mid.set_title(f"N = {n}, no final readout")
    for ax in axes:
        ax.set_xlabel("weight l")
        ax.axvline(0.75 * n, color="k", lw=0.5, alpha=0.4)
        ax.legend(fontsize=7, frameon=False)
    return table


def draw_summary(axes: np.ndarray, data: dict[int, dict]) -> dict:
    ax_rank, ax_mean, ax_local = axes
    table: dict = {}
    for n, d in data.items():
        c = SIZE_COLORS.get(n, "k")
        p = np.asarray(d["p"])
        mid = np.asarray(d["mid_all"]["mean"])
        full = np.asarray(d["full_all"]["mean"])
        fvis = np.asarray(d["full_vis"]["mean"])
        # log2 of the exact per-circuit sums, averaged: 1 - Delta / N.
        rank_mid = 1.0 - np.asarray(d["delta_mean"]) / n
        ax_rank.plot(p, rank_mid, "-o", color=c, ms=4, label=f"N = {n}")
        mw = np.array([mean_weight(h) for h in full]) / n
        mv = np.array([mean_weight(h) for h in fvis]) / n
        ax_mean.plot(p, mw, "-o", color=c, ms=4, label=f"N = {n}, all")
        ax_mean.plot(p, mv, "--s", color=c, ms=3, lw=1)
        loc = full[:, 1 : LOCAL + 1].sum(axis=1) / n
        single = full[:, 1] / n
        ax_local.plot(p, single, "-o", color=c, ms=4, label=f"N = {n}")
        table[str(n)] = {
            "p": p.tolist(),
            "log2_sum_over_N_mid": rank_mid.tolist(),
            "mean_weight_over_N_full": mw.tolist(),
            "mean_weight_over_N_visible": mv.tolist(),
            f"strings_weight_le_{LOCAL}_per_site": loc.tolist(),
            "single_site_strings_per_site": single.tolist(),
            "visible_total_full": fvis.sum(axis=1).tolist(),
            "mid_sum_mean": mid.sum(axis=1).tolist(),
        }
    ax_rank.axhline(1.0, color="k", lw=1, ls="--", label="final readout (exact)")
    ax_rank.set_ylabel("log2(sum_l N(l)) / N = 1 - Delta / N")
    ax_rank.set_title("Size of the dictionary")
    ax_mean.axhline(0.75, color="k", lw=0.8, ls="--")
    ax_mean.axhline(2 / 3, color="k", lw=0.8, ls=":")
    ax_mean.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax_mean.set_ylabel("<l> / N (solid: all, dashed: no Y)")
    ax_mean.set_title("Position of the bulk, final readout")
    ax_local.set_ylabel("N(1) / N")
    ax_local.set_title("Single-site strings per site, final readout")
    for ax in axes:
        ax.axvline(P_C, color="grey", lw=1, alpha=0.5)
        ax.set_xlabel("p")
        ax.legend(fontsize=7, frameon=False)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 20])
    args = parser.parse_args(argv)
    data = load(args.folder)
    rows = [n for n in args.rows if n in data]
    fig, axes = subplots(len(rows) + 1, 3, figsize=(15, 4.2 * (len(rows) + 1)))
    axes = np.atleast_2d(axes)
    extra: dict = {"rows": {}}
    for k, n in enumerate(rows):
        extra["rows"][str(n)] = draw_row(axes[k], data[n])
    extra["summary"] = draw_summary(axes[-1], data)
    path = save_figure(fig, args.folder, "clifford_weights", extra)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
