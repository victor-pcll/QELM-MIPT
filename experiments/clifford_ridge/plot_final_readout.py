#!/usr/bin/env python3
"""The learning transition with and without the final readout (issue #344).

Puts side by side, for the same circuits, the record with the final ``Z``
readout (``full``) and the mid-circuit record alone (``mid``):

- Clifford, K = inf: ``log2 dim Span{E_m} / N`` (``1`` against
  ``1 - Delta / N``), ``E_perp`` of local targets, and the number of visible
  strings of the group at ``N = 16`` (``population_*.json`` of
  ``clifford_ridge.py``);
- Haar, small ``N``: ``E_perp`` of a degree-1 target and the angle rank
  (``haar_span_*.json``);
- Clifford: the fraction of ``G`` reached by parities of at most ``k`` bits,
  with all bits or with the mid bits only (``parities_*.json`` of
  ``fourier_filter.py``, mid-only runs tagged ``mid``).

Usage::

    uv run python experiments/clifford_ridge/plot_final_readout.py \
        --population outputs/clifford_ridge --haar outputs/haar_span \
        --parities outputs/fourier_filter --out outputs/final_readout
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P_C = 0.161
P_C_HAAR = 0.29
RECORDS = {"full": ("-", "o"), "mid": ("--", "s")}
LABEL = {"full": "with final readout", "mid": "mid-circuit only"}


def load_population(folder: Path) -> dict:
    """``{N: run}`` at ``T = 2N``, chunks merged and sorted by ``p``."""
    chunks: dict = defaultdict(list)
    for path in sorted(folder.glob("population_N*_T*_s*.json")):
        d = json.loads(path.read_text())
        if d["t"] == 2 * d["n"]:
            chunks[d["n"]].append(d)
    runs = {}
    for n, parts in chunks.items():
        rows = {r["p"]: r for part in parts for r in part["per_p"]}
        runs[n] = {"p": sorted(rows), "per_p": [rows[p] for p in sorted(rows)]}
    return runs


def load_haar(folder: Path) -> dict:
    runs: dict = defaultdict(list)
    for path in sorted(folder.glob("haar_span_N*_seed*.json")):
        d = json.loads(path.read_text())
        runs[d["n"]].append(d)
    return dict(runs)


def load_parities(folder: Path) -> dict:
    """``{(N, record): run}``, ``record`` in ``full`` and ``mid``."""
    runs = {}
    for path in sorted(folder.glob("parities_N*_s*.json")):
        d = json.loads(path.read_text())
        runs[(d["n"], "mid" if d.get("mid_only") else "full")] = d
    return runs


def leaf(rows: list, *path: str) -> tuple[np.ndarray, np.ndarray]:
    m, e = [], []
    for row in rows:
        x = row
        for k in path:
            x = x[k]
        m.append(x["mean"])
        e.append(x["sem"])
    return np.asarray(m, dtype=float), np.asarray(e, dtype=float)


def haar_stat(runs: list, record: str, getter) -> tuple:
    """``p``, mean and SEM over seeds of ``getter(row)`` for the Haar kind."""
    p = np.asarray(runs[0]["p"])
    vals = np.array([[getter(r[record]) for r in run["per_kind"]["haar"]]
                     for run in runs])  # fmt: skip
    sem = vals.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0 * p
    return p, vals.mean(axis=0), sem


def palette(count: int, cmap: str = "viridis") -> list:
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.1, 0.85, max(count, 1))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--haar", type=Path, required=True)
    parser.add_argument("--parities", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pop = load_population(args.population)
    haar = load_haar(args.haar)
    par = load_parities(args.parities)
    if not pop:
        raise SystemExit("No population run.")
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ns = sorted(pop)
    cols = dict(zip(ns, palette(len(ns)), strict=True))

    # (a) Clifford: log2 dim Span / N.
    ax = axes[0, 0]
    summary["log2_dim_over_N"] = {}
    for n in ns:
        run = pop[n]
        for rec, (ls, mk) in RECORDS.items():
            m, e = leaf(run["per_p"], rec, "rank")
            ax.errorbar(run["p"], m / n, e / n, color=cols[n], ls=ls, marker=mk,
                        ms=3, label=f"{rec}, N={n}")  # fmt: skip
            summary["log2_dim_over_N"][f"{rec}_N{n}"] = (m / n).tolist()
    ax.axvline(P_C, color="0.4", ls=":", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("log2 dim Span{E_m} / N")
    ax.set_title("Clifford: dimension of the span (full: 1, mid: 1 - Delta/N)",
                 fontsize=10)  # fmt: skip
    ax.legend(fontsize=6, ncol=2)

    # (b) Clifford: E_perp of local targets, largest N.
    ax = axes[0, 1]
    big = ns[-1]
    summary["clifford_e_perp"] = {}
    for fam, c in zip(("Z", "ZZ_1", "ZX_1"), palette(3, "tab10"), strict=True):
        for rec, (ls, mk) in RECORDS.items():
            m, e = leaf(pop[big]["per_p"], rec, "member", fam)
            ax.errorbar(pop[big]["p"], 1 - m, e, color=c, ls=ls, marker=mk, ms=3,
                        label=f"{fam}, {rec}")  # fmt: skip
            summary["clifford_e_perp"][f"{fam}_{rec}_N{big}"] = (1 - m).tolist()
    ax.axvline(P_C, color="0.4", ls=":", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("E_perp (K = inf, lambda -> 0+)")
    ax.set_title(f"Clifford N = {big}: local targets (the two records coincide)",
                 fontsize=10)  # fmt: skip
    ax.legend(fontsize=7, ncol=2)

    # (c) Clifford N = 16: visible strings and d_eff of each group.
    ax = axes[0, 2]
    small = min(n for n in ns if "visible" in pop[n]["per_p"][0]["full"])
    run = pop[small]
    summary["clifford_visible"] = {}
    for rec, (ls, mk) in RECORDS.items():
        vis = np.array([np.sum(r[rec]["visible"]["mean"]) for r in run["per_p"]])
        deff = np.array([r[rec]["d_eff"]["mean"][2] for r in run["per_p"]])
        ax.semilogy(run["p"], vis, color="C0", ls=ls, marker=mk, ms=3,
                    label=f"visible strings, {rec}")  # fmt: skip
        ax.semilogy(run["p"], deff, color="C3", ls=ls, marker=mk, ms=3,
                    label=f"d_eff(lambda = 1/16), {rec}")  # fmt: skip
        summary["clifford_visible"][f"visible_{rec}"] = vis.tolist()
        summary["clifford_visible"][f"d_eff_{rec}"] = deff.tolist()
    ax.axvline(P_C, color="0.4", ls=":", lw=1)
    ax.set_xlabel("p")
    ax.set_title(f"Clifford N = {small}: size of the dictionary at K = inf",
                 fontsize=10)  # fmt: skip
    ax.legend(fontsize=7)

    # (d), (e) Haar: E_perp of a degree-1 target and the angle rank.
    hn = sorted(haar)
    hcols = dict(zip(hn, palette(len(hn), "plasma"), strict=True))
    summary["haar"] = {}
    for ax, (name, getter, scale) in zip(
        axes[1, :2],
        (
            ("E_perp of a degree-1 target", lambda r: r["e_perp"]["k1"], "linear"),
            ("angle rank / 3^N", None, "log"),
        ),
        strict=True,
    ):
        for n in hn:
            g = getter or (lambda r, n=n: r["angle"] / 3**n)
            for rec, (ls, mk) in RECORDS.items():
                p, m, e = haar_stat(haar[n], rec, g)
                ax.errorbar(p, m, e, color=hcols[n], ls=ls, marker=mk, ms=3,
                            label=f"{rec}, N={n}")  # fmt: skip
                summary["haar"][f"{name}_{rec}_N{n}"] = m.tolist()
        ax.axvline(P_C_HAAR, color="0.4", ls=":", lw=1)
        ax.set_yscale(scale)
        ax.set_xlabel("p")
        ax.set_title(f"Haar, open brickwork, T = 2N: {name}", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
    summary["haar"]["p"] = haar[hn[0]][0]["p"] if hn else []

    # (f) Clifford: parities of at most k bits, with and without final bits.
    ax = axes[1, 2]
    summary["parities"] = {}
    pn = sorted({n for n, _ in par})
    pcols = dict(zip(pn, palette(len(pn), "cividis"), strict=True))
    for n in pn:
        for rec, (ls, mk) in RECORDS.items():
            if (n, rec) not in par:
                continue
            run = par[(n, rec)]
            k = run["k_max"]
            m, e = leaf(run["per_p"], f"k{k}", "G_fraction")
            ax.errorbar(run["p"], m, e, color=pcols[n], ls=ls, marker=mk, ms=3,
                        label=f"N={n}, k <= {k}, {LABEL[rec]}")  # fmt: skip
            summary["parities"][f"N{n}_{rec}"] = {"p": run["p"], "G_fraction":
                                                  m.tolist()}  # fmt: skip
    ax.axvline(P_C, color="0.4", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("fraction of the visible strings of G (full record)")
    ax.set_title("Clifford: modes reached by parities of <= k bits", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle(
        "Learning transition with the final readout (solid) and with the "
        "mid-circuit record only (dashed)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out / "with_without_final.png", dpi=150)
    plt.close(fig)
    (args.out / "summary_final_readout.json").write_text(json.dumps(summary))
    print(f"wrote {args.out / 'with_without_final.png'}")


if __name__ == "__main__":
    main()
