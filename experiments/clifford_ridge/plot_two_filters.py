#!/usr/bin/env python3
"""Figures of the two-filter decomposition (issue #349).

Usage::

    uv run python experiments/clifford_ridge/plot_two_filters.py outputs/two_filters

Writes ``figures/two_filters_additivity.png``, ``figures/two_filters_locality.png``
and ``figures/summary_two_filters.json`` inside the given directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P_C = 0.161
PHASE = {0.04: "#1f77b4", 0.16: "#7f7f7f", 0.3: "#d62728"}


def m(v: dict | None) -> float:
    return float("nan") if v is None else float(v["mean"])


def s(v: dict | None) -> float:
    return float("nan") if v is None else float(v["sem"])


def load(root: Path, prefix: str) -> dict[int, dict]:
    out = {}
    for f in sorted(root.glob(f"{prefix}_N*_s*.json")):
        d = json.loads(f.read_text())
        out[int(d["n"])] = d
    return out


def nearest(rows: list[dict], p: float) -> dict:
    return min(rows, key=lambda r: abs(r["p"] - p))


def vline(ax) -> None:
    ax.axvline(P_C, color="k", lw=0.6, ls=":")


def plot_additivity(exact: dict[int, dict], shots: dict | None, out: Path) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    summary: dict = {}
    # (a) cross term at lambda = 0 over all monomials and the dense target.
    ax = axes[0, 0]
    for n, d in sorted(exact.items()):
        ps = [r["p"] for r in d["per_p"]]
        worst = [
            max(row["mono_max_abs_cross"]["max"] for row in r["k"]) for r in d["per_p"]
        ]
        dense = [
            max(abs(row["dense"][0]["cross"]["max"]) for row in r["k"])
            for r in d["per_p"]
        ]
        ax.semilogy(ps, np.maximum(worst, 1e-18), "o-", label=f"N={n}, all monomials")
        ax.semilogy(ps, np.maximum(dense, 1e-18), "s--", label=f"N={n}, dense target")
        summary[f"N{n}_max_cross_lambda0"] = float(max(max(worst), max(dense)))
    vline(ax)
    ax.set(xlabel="p", ylabel="max |cross term|",
           title="(a) population, λ → 0+: the split is exact")  # fmt: skip
    ax.legend(fontsize=7)
    # (b) cross term against lambda, dense target, largest exact N, k = 1.
    n_b = max(exact)
    d = exact[n_b]
    ax = axes[0, 1]
    lams = d["lambdas"]
    for p, col in PHASE.items():
        r = nearest(d["per_p"], p)["k"][0]["dense"]
        x = [lam if lam > 0 else 2.0**-16 for lam in lams]
        ax.semilogx(
            x, [m(v["cross"]) for v in r], "o-", color=col, label=f"p={p}, cross"
        )
        ax.semilogx(x, [m(v["readout"]) for v in r], "s--", color=col, alpha=0.6,
                    label=f"p={p}, readout")  # fmt: skip
    ax.set(xlabel="λ (λ = 0 plotted at 2^-16)", ylabel="term / Var y",
           title=f"(b) λ > 0, N={n_b}, k=1, dense target")  # fmt: skip
    ax.legend(fontsize=7)
    # (c) finite training set (exact features, lambda = 2^-8).
    ax = axes[0, 2]
    for p, col in PHASE.items():
        r = nearest(d["per_p"], p)["k"][0]
        for name, mk in (("dense", "o-"), ("deg1", "s--")):
            v = r[f"{name}_train"]
            ax.semilogx(d["train"], [m(x["cross"]) for x in v], mk, color=col,
                        label=f"p={p}, {name}")  # fmt: skip
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="training inputs", ylabel="cross term / Var y",
           title=f"(c) finite training set, N={n_b}, k=1, λ=2^-8")  # fmt: skip
    ax.legend(fontsize=7)
    summary["finite_train_cross"] = {
        str(p): [m(x["cross"]) for x in nearest(d["per_p"], p)["k"][0]["dense_train"]]
        for p in PHASE
    }
    # (d) finite shots.
    ax = axes[1, 0]
    if shots:
        ks = shots["shots"]
        for p, col in PHASE.items():
            r = nearest(shots["per_p"], p)
            tot = [m(r[f"K{k}"]["total"]) for k in ks]
            sumt = [m(r[f"K{k}"]["circuit"]) + m(r[f"K{k}"]["readout"]) for k in ks]
            ax.semilogx(ks, tot, "o-", color=col, label=f"p={p}, total")
            ax.semilogx(ks, sumt, "s--", color=col, alpha=0.6,
                        label=f"p={p}, circuit + readout")  # fmt: skip
            ax.axhline(m(r["population"]["total"]), color=col, lw=0.6, ls=":")
        ax.set(xlabel="shots K", ylabel="error / Var y",
               title="(d) N=4, k=2, 256 training inputs: finite K")  # fmt: skip
        ax.legend(fontsize=7)
        summary["shots"] = {
            str(r["p"]): {
                f"K{k}": {
                    t: m(r[f"K{k}"][t])
                    for t in ("total", "circuit", "readout", "cross")
                }
                for k in ks
            }  # fmt: skip
            for r in shots["per_p"]
        }
    # (e) any decoder against the linear ridge, monomials.
    ax = axes[1, 1]
    for n, dd in sorted(exact.items()):
        ps = [r["p"] for r in dd["per_p"]]
        for k in range(dd["k_max"]):
            rec = [
                m(r["k"][k]["any_recovered_in_closure_not_F"])
                / max(m(r["k"][k]["in_closure_not_F"]), 1e-12)
                for r in dd["per_p"]
            ]
            ax.plot(ps, rec, "o-", label=f"N={n}, k≤{k + 1}")
    vline(ax)
    ax.set(xlabel="p", ylabel="fraction recovered by a non-linear decoder",
           title="(e) monomials in <D_k> but not in D_k")  # fmt: skip
    ax.legend(fontsize=6, ncol=2)
    # (f) basis target (not multilinear).
    ax = axes[1, 2]
    for n, dd in sorted(exact.items()):
        ps = [r["p"] for r in dd["per_p"]]
        ax.plot(ps, [m(r["S_lin_basis"]) for r in dd["per_p"]], "k--", lw=0.8)
        ax.plot(ps, [m(r["S_any_basis"]) for r in dd["per_p"]], "o-",
                label=f"N={n}, any decoder on S")  # fmt: skip
        ax.plot(ps, [m(r["k"][0]["any_basis"]) for r in dd["per_p"]], "s:",
                label=f"N={n}, any decoder on F^(1)")  # fmt: skip
    vline(ax)
    ax.set(
        xlabel="p",
        ylabel="error / Var y",
        ylim=(0, 1.05),
        title="(f) target x_0² − z_0² (basis of site 0), dashed: linear",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "two_filters_additivity.png", dpi=130)
    plt.close(fig)
    return summary


def plot_locality(loc: dict[int, dict], out: Path) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    summary: dict = {}
    n_main = 16 if 16 in loc else max(loc)
    d = loc[n_main]
    ps = [r["p"] for r in d["per_p"]]
    kmax = d["k_max"]
    # (a) fixed targets: probability to be kept by the circuit / by the ridge.
    ax = axes[0, 0]
    fams = ["Z", "ZZ_1", "ZZ_N/2", f"rand^{4}", f"Z^{n_main // 2}"]
    colors = plt.get_cmap("tab10")
    for j, fam in enumerate(fams):
        if fam not in d["per_p"][0]["families"]:
            continue
        rows = [r["families"][fam] for r in d["per_p"]]
        in_g = np.array([1 - m(r["circuit"]) for r in rows])
        in_1 = in_g - np.array([m(r["readout"][0]) for r in rows])
        in_k = in_g - np.array([m(r["readout"][-1]) for r in rows])
        c = colors(j)
        ax.semilogy(ps, np.maximum(in_g, 1e-4), "o-", color=c, label=f"{fam}: in G")
        ax.semilogy(ps, np.maximum(in_1, 1e-4), "--", color=c)
        ax.semilogy(ps, np.maximum(in_k, 1e-4), ":", color=c)
    vline(ax)
    ax.set(xlabel="p", ylabel=f"1 − E (solid: S, dashed: k=1, dotted: k≤{kmax})",
           title=f"(a) fixed monomial targets, N={n_main}")  # fmt: skip
    ax.legend(fontsize=7)
    summary["families"] = {
        fam: {
            "p": ps,
            "E_circuit": [m(r["families"][fam]["circuit"]) for r in d["per_p"]],
            "E_readout_k": [
                [m(x) for x in r["families"][fam]["readout"]] for r in d["per_p"]
            ],
        }
        for fam in d["per_p"][0]["families"]
    }
    # (b) isotropic window targets.
    ax = axes[0, 1]
    lengths = d["iso_lengths"]
    for p, col in PHASE.items():
        r = nearest(d["per_p"], p)
        ax.plot(lengths, [m(v) for v in r["iso_lengths_circuit"]], "o-", color=col,
                label=f"p={p}: E_circuit")  # fmt: skip
        ax.plot(lengths, [m(v) for v in r["iso_lengths_readout"][0]], "s--",
                color=col, label=f"p={p}: E_readout(1)")  # fmt: skip
        ax.plot(lengths, [m(v) for v in r["iso_lengths_readout"][-1]], "^:",
                color=col, label=f"p={p}: E_readout(≤{kmax})")  # fmt: skip
    ax.set(xlabel="window l", ylabel="error / Var y", yscale="log",
           title=f"(b) isotropic target on l sites, N={n_main}")  # fmt: skip
    ax.legend(fontsize=6)
    # (c) targets drawn inside G, by extent.
    ax = axes[0, 2]
    ext = np.arange(1, n_main + 1)
    for p, col in PHASE.items():
        r = nearest(d["per_p"], p)["inG_readout_by_extent"]
        ax.plot(ext, [m(v) for v in r[0]], "s--", color=col, label=f"p={p}, k=1")
        ax.plot(ext, [m(v) for v in r[-1]], "o-", color=col, label=f"p={p}, k≤{kmax}")
    ax.set(xlabel="extent of the target (sites)", ylabel="E_readout (E_circuit = 0)",
           title=f"(c) targets inside G, N={n_main}")  # fmt: skip
    ax.legend(fontsize=7)
    # (d) in-G targets, short vs long, against p, all N.
    ax = axes[1, 0]
    for n, dd in sorted(loc.items()):
        pp = [r["p"] for r in dd["per_p"]]
        rows = [r["inG_readout_by_extent"][-1] for r in dd["per_p"]]
        short = [np.nanmean([m(v) for v in row[:3]]) for row in rows]
        long_ = [np.nanmean([m(v) for v in row[n // 2 - 1 :]]) for row in rows]
        (h,) = ax.plot(pp, short, "o-", label=f"N={n}, k≤{dd['k_max']}, extent ≤ 3")
        ax.plot(pp, long_, "s--", color=h.get_color(), label="extent ≥ N/2")
        summary[f"N{n}_inG_readout_short_long"] = {
            "p": pp,
            "short": short,
            "long": long_,
        }
    vline(ax)
    ax.set(xlabel="p", ylabel="E_readout of targets inside G",
           title="(d) does the ridge 'light up' above p_c?")  # fmt: skip
    ax.legend(fontsize=6)
    # (e) visible part of G.
    ax = axes[1, 1]
    for n, dd in sorted(loc.items()):
        pp = [r["p"] for r in dd["per_p"]]
        vis = [m(r["G_visible"]) for r in dd["per_p"]]
        ax.semilogy(pp, np.array(vis) / 2**n, "o-", label=f"N={n}")
        summary[f"N{n}_G_visible_over_2N"] = (np.array(vis) / 2**n).tolist()
    vline(ax)
    ax.set(xlabel="p", ylabel="visible strings of G / 2^N",
           title="(e) what the 2^N modes of G hold for XZ inputs")  # fmt: skip
    ax.legend(fontsize=7)
    # (f) extent histogram of the visible strings of G.
    ax = axes[1, 2]
    for p, col in PHASE.items():
        h = np.array([m(v) for v in nearest(d["per_p"], p)["G_extent_hist"]])
        ax.plot(ext, h / h.sum(), "o-", color=col, label=f"p={p}")
    ax.set(xlabel="extent (sites)", ylabel="fraction of visible G",
           title=f"(f) extent of the visible strings of G, N={n_main}")  # fmt: skip
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "two_filters_locality.png", dpi=130)
    plt.close(fig)
    return summary


def plot_budget(shots: dict, out: Path) -> dict:
    """Error budget with three non-negative terms, N = 4, k <= 2.

    ``E_circuit`` and ``E_readout(k)`` are population terms (``lambda -> 0+``,
    ``K = inf``). ``E_estimation = E_total - (E_circuit + E_readout)`` is the
    excess of the trained ridge (finite training set, ``lambda``, ``K``). It
    is non-negative because no predictor in ``Span F`` beats ``P_F``.
    """
    ks = shots["shots"]
    labels = [f"K={k}" for k in ks] + ["K=∞"]
    keys = [f"K{k}" for k in ks] + ["Kinf"]
    p_show = [0.04, 0.16, 0.3]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"circuit": "#4c72b0", "readout": "#dd8452", "estimation": "#55a868"}
    summary: dict = {}
    width = 0.8 / len(p_show)
    x = np.arange(len(keys))
    for j, p in enumerate(p_show):
        r = nearest(shots["per_p"], p)
        circ = m(r["population"]["circuit"])
        read = m(r["population"]["readout"])
        est = np.array([m(r[k]["total"]) - circ - read for k in keys])
        err = np.array([s(r[k]["total"]) for k in keys])
        summary[str(p)] = {
            "E_circuit": circ, "E_readout": read,
            "E_estimation": dict(zip(labels, est.tolist(), strict=True)),
        }  # fmt: skip
        pos = x + (j - 1) * width
        hatch = ["", "//", ".."][j]
        for ax in axes:
            base = np.zeros(len(keys))
            parts = (("circuit", circ), ("readout", read), ("estimation", est))
            for name, val in parts:
                if ax is axes[1] and name == "circuit":
                    continue
                v = np.broadcast_to(val, base.shape)
                ax.bar(pos, v, width, bottom=base, color=colors[name], hatch=hatch,
                       edgecolor="white",
                       label=f"{name}" if j == 0 else None)  # fmt: skip
                base = base + v
            if ax is axes[1]:
                ax.errorbar(pos, base, yerr=err, fmt="none", ecolor="k", lw=0.8)
    for ax in axes:
        ax.set_xticks(x, labels)
    title_a = "(a) E_total = E_circuit + E_readout(k) + E_estimation"
    axes[0].set(ylabel="error / Var y", ylim=(0, 1.05), title=title_a)
    axes[1].set(ylabel="error / Var y",
                title="(b) zoom: what the readout and the training add")  # fmt: skip
    axes[0].legend(loc="lower right", fontsize=8)
    from matplotlib.patches import Patch

    hatches = [Patch(facecolor="white", edgecolor="k", hatch=h, label=f"p = {p}")
               for h, p in zip(["", "//", ".."], p_show, strict=True)]  # fmt: skip
    handles, _ = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=handles + hatches, loc="upper right", fontsize=8)
    fig.suptitle(
        "N = 4 Clifford QELM, readout by parities of ≤ 2 bits, 256 training inputs, "
        "λ = 2^-8; bars: p = 0.04 / 0.16 / 0.3",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "two_filters_budget.png", dpi=130)
    plt.close(fig)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    out = args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    exact = load(args.root, "exact")
    loc = load(args.root, "locality")
    sh = load(args.root, "shots")
    summary = {
        "additivity": plot_additivity(exact, sh.get(4), out),
        "locality": plot_locality(loc, out),
        "budget": plot_budget(sh[4], out) if 4 in sh else {},
    }
    (out / "summary_two_filters.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
