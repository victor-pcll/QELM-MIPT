#!/usr/bin/env python3
"""Figures of the ridge at K = infinity on the Clifford QELM (issue #337).

Reads ``population_N*_T*_s*[_tag].json`` (the chunks of the p grid are
merged per ``N, T``) from the output folder of ``clifford_ridge.py`` and
writes the figures and ``summary_population.json`` in ``figures/``.

Usage::

    uv run python experiments/clifford_ridge/plot_population.py \
        outputs/clifford_ridge
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

P_C_LIT = 0.160
NU = 1.3


def palette(count: int, cmap: str = "viridis") -> list:
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.1, 0.9, max(count, 1))]


def crossing(p, a, ea, b, eb) -> tuple[float, bool]:
    """Sign change of ``a - b`` (linear interpolation) and whether it is
    resolved: the difference exceeds two standard errors on both sides."""
    d = np.asarray(a) - np.asarray(b)
    e = np.sqrt(np.asarray(ea) ** 2 + np.asarray(eb) ** 2)
    for i in range(len(p) - 1):
        if d[i] == 0 or d[i] * d[i + 1] < 0:
            x = p[i] - d[i] * (p[i + 1] - p[i]) / (d[i + 1] - d[i])
            ok = abs(d[i]) > 2 * e[i] and abs(d[i + 1]) > 2 * e[i + 1]
            return float(x), bool(ok)
    return float("nan"), False


def load(folder: Path) -> dict:
    """``{(N, T/N): run}`` with the per-p rows of all chunks, sorted by p."""
    chunks: dict = defaultdict(list)
    for path in sorted(folder.glob("population_N*_T*_s*.json")):
        d = json.loads(path.read_text())
        chunks[(d["n"], d["t"] // d["n"])].append(d)
    runs = {}
    for key, parts in chunks.items():
        run = dict(parts[0])
        rows = {r["p"]: r for part in parts for r in part["per_p"]}
        run["per_p"] = [rows[p] for p in sorted(rows)]
        run["p"] = sorted(rows)
        runs[key] = run
    return runs


def series(run: dict, *path: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM of one leaf along the p grid."""
    m, e = [], []
    for row in run["per_p"]:
        leaf = row
        for k in path:
            leaf = leaf[k]
        m.append(leaf["mean"])
        e.append(leaf["sem"])
    return np.asarray(m, dtype=float), np.asarray(e, dtype=float)


def order_parameters(run: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``Delta / N`` and the half-chain entropy of the references / N."""
    n = run["n"]
    rank, e_rank = series(run, "mid", "rank")
    half = run["lengths"].index(n // 2)
    k, e_k = series(run, "full", "local_rank")
    return {
        "delta": (1 - rank / n, e_rank / n),
        "s_half": ((n // 2 - k[:, half]) / n, e_k[:, half] / n),
    }


def pairwise_crossings(runs: dict, ratio: int, name: str) -> list[dict]:
    ns = sorted(n for (n, r) in runs if r == ratio)
    out = []
    for a, b in zip(ns[:-1], ns[1:], strict=True):
        ra, rb = runs[(a, ratio)], runs[(b, ratio)]
        if ra["p"] != rb["p"]:
            continue
        ma, ea = order_parameters(ra)[name]
        mb, eb = order_parameters(rb)[name]
        x, ok = crossing(np.asarray(ra["p"]), ma, ea, mb, eb)
        out.append({"N": [a, b], "p": x, "resolved": ok})
    return out


def fig_order(runs: dict, ratio: int, out: Path, summary: dict) -> float:
    """Order parameters of the transition, from the same groups."""
    ns = sorted(n for (n, r) in runs if r == ratio)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, ax, title in (
        ("delta", axes[0], "Delta / N: input bits not read by the mid record"),
        ("s_half", axes[1], "S(N/2) / N of the references, full record"),
    ):
        for n, c in zip(ns, palette(len(ns)), strict=True):
            m, e = order_parameters(runs[(n, ratio)])[name]
            ax.errorbar(runs[(n, ratio)]["p"], m, e, color=c, ms=3, marker="o",
                        label=f"N={n}")  # fmt: skip
        ax.axvline(P_C_LIT, color="0.4", ls="--", lw=1)
        ax.set_xlabel("p")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        summary[f"crossings_{name}"] = pairwise_crossings(runs, ratio, name)
    fig.tight_layout()
    fig.savefig(out / "population_order.png", dpi=150)
    plt.close(fig)
    resolved = [c["p"] for c in summary["crossings_delta"] if c["resolved"]]
    p_c = float(np.mean(resolved)) if resolved else P_C_LIT
    summary["p_c"] = p_c
    return p_c


def fig_eperp(runs: dict, ratio: int, out: Path, summary: dict, p_c: float) -> None:
    """E_perp = 1 - Pr[a in G] of monomial targets against p, per N."""
    ns = sorted(n for (n, r) in runs if r == ratio)
    fams = [("Z", "Z_i"), ("X", "X_i"), ("ZZ_1", "Z_i Z_i+1"), ("XX_1", "X_i X_i+1"),
            ("ZX_1", "Z_i X_i+1"), ("far", "Z_i Z_i+N/2")]  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    summary["e_perp"] = {}
    for ax, (fam, label) in zip(axes.ravel(), fams, strict=True):
        for n, c in zip(ns, palette(len(ns)), strict=True):
            run = runs[(n, ratio)]
            key = f"ZZ_{n // 2}" if fam == "far" else fam
            for record, ls in (("full", "-"), ("mid", ":")):
                m, e = series(run, record, "member", key)
                ax.errorbar(run["p"], 1 - m, e, color=c, ls=ls, ms=2.5,
                            marker="o" if ls == "-" else None,
                            label=f"N={n}" if record == "full" else None)  # fmt: skip
                summary["e_perp"][f"N{n}_{fam}_{record}"] = (1 - m).tolist()
        ax.axvline(p_c, color="0.4", ls="--", lw=1)
        ax.set_title(f"E_perp of {label} (full: solid, mid only: dotted)", fontsize=9)
        ax.set_xlabel("p")
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("E_perp = 1 - R2, K = inf, lambda -> 0+")
    axes[1, 0].set_ylabel("E_perp = 1 - R2, K = inf, lambda -> 0+")
    fig.tight_layout()
    fig.savefig(out / "population_eperp.png", dpi=150)
    plt.close(fig)


def fig_range(runs: dict, ratio: int, out: Path, summary: dict) -> None:
    """Locality at the largest N: E_perp against the range, local ranks."""
    n = max(n for (n, r) in runs if r == ratio)
    run = runs[(n, ratio)]
    picks = [p for p in (0.04, 0.1, 0.14, 0.16, 0.18, 0.22, 0.3, 0.5) if p in run["p"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    cols = palette(len(picks), "coolwarm")
    summary["range"] = {"N": n, "p": picks}
    for p, c in zip(picks, cols, strict=True):
        row = run["per_p"][run["p"].index(p)]
        for ax, fam in ((axes[0], "ZZ"), (axes[1], "XX")):
            ell = run["ranges"]
            m = [row["full"]["member"][f"{fam}_{x}"]["mean"] for x in ell]
            ax.semilogx(ell, m, color=c, marker="o", ms=3, label=f"p={p}")
            summary["range"][f"{fam}_p{p}"] = m
        k = np.asarray(row["full"]["local_rank"]["mean"])
        lengths = np.asarray(run["lengths"])
        axes[2].semilogx(lengths, k / lengths, color=c, marker="o", ms=3)
        summary["range"][f"local_rank_p{p}"] = k.tolist()
    for ax, fam in ((axes[0], "Z_i Z_i+l"), (axes[1], "X_i X_i+l")):
        ax.set_xlabel("range l")
        ax.set_ylabel("R2 = Pr[a in G], full record")
        ax.set_title(f"N = {n}: R2 of {fam}", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[2].set_xlabel("interval length l")
    axes[2].set_ylabel("k(l) / l = 1 - S(l) / l")
    axes[2].set_title(f"N = {n}: rank of G inside an interval, per site", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "population_range.png", dpi=150)
    plt.close(fig)


def fig_spectrum(runs: dict, ratio: int, out: Path, summary: dict, p_c: float) -> None:
    """Spectrum of the population covariance: weights of the visible strings."""
    ns = sorted(n for (n, r) in runs if r == ratio)
    enum = [n for n in ns if "visible" in runs[(n, ratio)]["per_p"][0]["full"]]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
    summary["spectrum"] = {}
    if enum:
        n = max(enum)
        run = runs[(n, ratio)]
        picks = [p for p in (0.04, 0.1, 0.16, 0.22, 0.3, 0.5) if p in run["p"]]
        for p, c in zip(picks, palette(len(picks), "coolwarm"), strict=True):
            row = run["per_p"][run["p"].index(p)]
            counts = np.asarray(row["full"]["visible"]["mean"])
            w = np.arange(counts.size)
            axes[0].semilogy(w[1:], counts[1:], color=c, marker="o", ms=3,
                             label=f"p={p}")  # fmt: skip
            summary["spectrum"][f"N{n}_counts_p{p}"] = counts.tolist()
        axes[0].set_xlabel("weight |g| (eigenvalue 2^-|g|)")
        axes[0].set_ylabel("number of eigenvalues")
        axes[0].set_title(f"N = {n}: spectrum of Sigma, full record", fontsize=10)
        axes[0].legend(fontsize=7)
        lams = run["lambdas"]
        for j, c in zip(range(0, len(lams), 2), palette(5, "plasma"), strict=False):
            for nn, ls in zip(enum, ("-", "--", ":"), strict=False):
                m, e = series(runs[(nn, ratio)], "full", "d_eff")
                axes[1].errorbar(runs[(nn, ratio)]["p"], m[:, j], e[:, j], color=c,
                                 ls=ls, label=f"lambda=2^-{2 * j}, N={nn}")  # fmt: skip
        axes[1].set_yscale("log")
        axes[1].set_xlabel("p")
        axes[1].set_title("d_eff(lambda) = sum w/(w + lambda)", fontsize=10)
        axes[1].legend(fontsize=6, ncol=2)
        for nn, c in zip(enum, palette(len(enum)), strict=True):
            for record, ls in (("full", "-"), ("mid", ":")):
                m, e = series(runs[(nn, ratio)], record, "rank_eff")
                axes[2].errorbar(runs[(nn, ratio)]["p"], m, e, color=c, ls=ls,
                                 label=f"N={nn} {record}")  # fmt: skip
                summary["spectrum"][f"N{nn}_rank_eff_{record}"] = m.tolist()
        axes[2].set_xlabel("p")
        axes[2].set_title("effective rank sum mu / mu_max", fontsize=10)
        axes[2].legend(fontsize=7)
    for n, c in zip(ns, palette(len(ns)), strict=True):
        run = runs[(n, ratio)]
        m, _ = series(run, "full", "local_visible")
        total = m.sum(axis=1) / n
        axes[3].plot(run["p"], total, color=c, marker="o", ms=3, label=f"N={n}")
        summary["spectrum"][f"N{n}_local_visible_per_site"] = total.tolist()
    span = next(iter(runs.values()))["local_span"]
    axes[3].set_xlabel("p")
    axes[3].set_title(f"visible strings within {span} sites, per site", fontsize=10)
    axes[3].legend(fontsize=7)
    for ax in axes[1:]:
        ax.axvline(p_c, color="0.4", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out / "population_spectrum.png", dpi=150)
    plt.close(fig)


def fig_collapse(runs: dict, ratio: int, out: Path, summary: dict, p_c: float) -> None:
    """Finite-size collapse against ``(p - p_c) N^{1/nu}``."""
    ns = sorted(n for (n, r) in runs if r == ratio)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for n, c in zip(ns, palette(len(ns)), strict=True):
        run = runs[(n, ratio)]
        x = (np.asarray(run["p"]) - p_c) * n ** (1 / NU)
        params = order_parameters(run)
        axes[0].plot(x, params["delta"][0], color=c, marker="o", ms=3, label=f"N={n}")
        axes[1].plot(x, params["s_half"][0], color=c, marker="o", ms=3)
        m, _ = series(run, "full", "member", "Z")
        axes[2].plot(x, 1 - m, color=c, marker="o", ms=3)
    titles = ("Delta / N", "S(N/2) / N", "E_perp of Z_i")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xlabel(f"(p - {p_c:.3f}) N^(1/{NU})")
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color="0.4", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    summary["collapse"] = {"p_c": p_c, "nu": NU}
    fig.tight_layout()
    fig.savefig(out / "population_collapse.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--ratio", type=int, default=2, help="T / N")
    args = parser.parse_args(argv)
    runs = load(args.folder)
    if not any(r == args.ratio for (_, r) in runs):
        raise SystemExit(f"No population run with T = {args.ratio} N.")
    out = args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    p_c = fig_order(runs, args.ratio, out, summary)
    fig_eperp(runs, args.ratio, out, summary, p_c)
    fig_range(runs, args.ratio, out, summary)
    fig_spectrum(runs, args.ratio, out, summary, p_c)
    fig_collapse(runs, args.ratio, out, summary, p_c)
    (out / "summary_population.json").write_text(json.dumps(summary))
    print(f"wrote figures in {out}")


if __name__ == "__main__":
    main()
