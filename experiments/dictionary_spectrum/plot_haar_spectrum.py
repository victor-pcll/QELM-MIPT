#!/usr/bin/env python3
"""Figure of the spectrum of the Haar effect dictionary (issue #356).

Reads ``haar_spectrum_N*_seed*.json`` and writes ``haar_spectrum.png`` and
its data in the output folder. Spectra are sorted per seed, then averaged
index by index.

Usage::

    uv run python experiments/dictionary_spectrum/plot_haar_spectrum.py \
        outputs/dictionary_spectrum
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from scipy.special import comb  # noqa: E402

from QELM.utils.plots.io import save_figure  # noqa: E402
from QELM.utils.plots.style import subplots  # noqa: E402

P_C = 0.29
PICKS = {0.04: "#1f4e99", 0.16: "#5b6fb3", 0.3: "#8a3fa0", 0.5: "#c0392b"}
SIZE_STYLE = {4: ("#9ecae1", "--"), 5: ("#6baed6", "-."), 6: ("#08519c", "-")}
TOP = 10


def load(folder: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(folder.glob("haar_spectrum_N*_seed*.json")):
        d = json.loads(path.read_text())
        out[int(d["n"])].append(d)
    return dict(sorted(out.items()))


def rows_at(runs: list[dict], kind: str, p: float, record: str) -> list[dict]:
    out = []
    for d in runs:
        for row in d["per_kind"][kind]:
            if abs(row["p"] - p) < 1e-9:
                out.append(row[record])
    return out


def mean_sorted(rows: list[dict], size: int) -> np.ndarray:
    mus = np.zeros((len(rows), size))
    for k, r in enumerate(rows):
        mu = np.sort(np.asarray(r["mu"]))[::-1][:size]
        mus[k, : mu.size] = mu
    return np.asarray(mus.mean(axis=0))


def top_share(mu: list[float]) -> float:
    m = np.sort(np.asarray(mu))[::-1]
    return float(m[:TOP].sum() / m.sum()) if m.sum() > 0 else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args(argv)
    data = load(args.folder)
    n = args.n
    runs = data[n]
    size = 3**n - 1
    k = np.arange(1, size + 1)
    thresholds = runs[0]["thresholds"]
    fig, axes = subplots(2, 3, figsize=(15, 9))
    extra: dict = {"n": n, "seeds": len(runs), "spectra": {}, "vs_p": {}}

    # Top row: sorted spectra (full, mid), and the degree profiles.
    cap = comb(n, np.arange(n + 1)) * 2.0 ** np.arange(n + 1)
    for j, record in enumerate(("full", "mid")):
        ax = axes[0, j]
        for p, c in PICKS.items():
            haar = mean_sorted(rows_at(runs, "haar", p, record), size)
            cliff = mean_sorted(rows_at(runs, "clifford", p, record), size)
            ax.semilogy(
                k,
                np.where(haar > 0, haar, np.nan),
                "-",
                color=c,
                lw=2,
                label=f"Haar, p = {p}",
            )
            ax.semilogy(k, np.where(cliff > 0, cliff, np.nan), ":", color=c, lw=1.5)
            extra["spectra"][f"{record}_{p}"] = {
                "haar": haar.tolist(),
                "clifford": cliff.tolist(),
            }
        for lam in thresholds:
            ax.axhline(lam, color="grey", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("index k")
        ax.set_ylabel("mu_k (eigenvalues of F)")
        title = "final readout" if record == "full" else "no final readout"
        ax.set_title(f"N = {n}, {title} (dotted: Clifford)")
        ax.legend(fontsize=7, frameon=False)
    ax = axes[0, 2]
    ell = np.arange(n + 1)
    for p, c in PICKS.items():
        rows = rows_at(runs, "haar", p, "full")
        born = np.mean([r["born_profile"] for r in rows], axis=0)
        span = np.mean([r["span_profile"] for r in rows], axis=0)
        ax.plot(ell[1:], (born / cap)[1:], "-o", color=c, ms=4, label=f"p = {p}")
        ax.plot(ell[1:], (span / cap)[1:], "--", color=c, lw=1)
        extra["spectra"][f"profile_{p}"] = {
            "born": born.tolist(),
            "span": span.tolist(),
        }
    ax.set_yscale("log")
    ax.set_xlabel("degree l")
    ax.set_ylabel("fraction of the degree-l modes")
    ax.set_title("Haar, final readout: N_F(l) (solid), P(l) (dashed)")
    ax.legend(fontsize=7, frameon=False)

    # Bottom row: against p, every size.
    for m, mruns in data.items():
        col, ls = SIZE_STYLE.get(m, ("k", "-"))
        p_list = [row["p"] for row in mruns[0]["per_kind"]["haar"]]
        dim = 3**m - 1
        res: dict = defaultdict(list)
        for p in p_list:
            for kind in ("haar", "clifford"):
                rows = rows_at(mruns, kind, p, "full")
                res[f"{kind}_rank"].append(
                    np.mean([r["span_rank"] - 1 for r in rows]) / dim
                )
                res[f"{kind}_pr"].append(
                    np.mean([r["ranks"]["participation"] for r in rows])
                )
                res[f"{kind}_top"].append(np.mean([top_share(r["mu"]) for r in rows]))
                res[f"{kind}_trace"].append(np.mean([r["trace_F"] for r in rows]))
                for t, lam in enumerate(thresholds):
                    res[f"{kind}_above_{lam:g}"].append(
                        np.mean([r["ranks"]["above"][t] for r in rows]) / dim
                    )
        a = axes[1, 0]
        a.plot(
            p_list,
            res["haar_rank"],
            ls,
            color=col,
            marker="o",
            ms=4,
            label=f"N = {m}: span rank",
        )
        for lam, mk in zip(thresholds, ("s", "^", "v"), strict=False):
            a.plot(
                p_list,
                res[f"haar_above_{lam:g}"],
                ls,
                color=col,
                marker=mk,
                ms=4,
                lw=1,
                label=f"N = {m}: mu > {lam:g}",
            )
        b = axes[1, 1]
        b.plot(
            p_list,
            res["haar_pr"],
            ls,
            color=col,
            marker="o",
            ms=4,
            label=f"Haar N = {m}",
        )
        b.plot(
            p_list,
            res["clifford_pr"],
            ls,
            color="#c0392b",
            marker="s",
            ms=3,
            lw=1,
            alpha=0.4 + 0.1 * m / 2,
            label=f"Clifford N = {m}",
        )
        c_ = axes[1, 2]
        c_.plot(
            p_list,
            res["haar_top"],
            ls,
            color=col,
            marker="o",
            ms=4,
            label=f"Haar N = {m}",
        )
        c_.plot(
            p_list,
            res["clifford_top"],
            ls,
            color="#c0392b",
            marker="s",
            ms=3,
            lw=1,
            label=f"Clifford N = {m}",
        )
        extra["vs_p"][str(m)] = {
            "p": p_list,
            **{key: list(v) for key, v in res.items()},
        }
    axes[1, 0].set_ylabel("fraction of the 3^N - 1 modes")
    axes[1, 0].set_title("Haar, final readout: rank and counts above lambda")
    axes[1, 1].set_ylabel("(Tr F)^2 / Tr F^2")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Participation ratio of F, final readout")
    axes[1, 2].set_ylabel(f"share of Tr F in the top {TOP} modes")
    axes[1, 2].set_title("Concentration of the spectrum, final readout")
    for ax in axes[1]:
        ax.axvline(P_C, color="grey", lw=1, alpha=0.5)
        ax.set_xlabel("p")
        ax.legend(fontsize=6, frameon=False)
    path = save_figure(fig, args.folder, "haar_spectrum", extra)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
