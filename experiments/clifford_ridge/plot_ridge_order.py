#!/usr/bin/env python3
"""Two order parameters of the MIPT read from the ridge at K = infinity (issue #349).

No simulation: the figures re-plot data already computed.

1. **Without the final readout: the rank of the ridge.** The span of the
   features of the mid-circuit record is ``Span G_mid``, of dimension
   ``2^(N - Delta)`` (#344). ``log2(rank) / N = 1 - Delta / N`` is the share
   of the ``N`` input bits that the ridge can represent; ``Delta`` is the
   number it cannot. Data: the ``transition`` runs of #337
   (``transition_N*_T{2N}_s0.json``).
2. **With the final readout: the locality of the ridge dictionary.** The
   rank is ``2^N`` at every ``p``, but only ``2^(l - S(l))`` of the
   representable functions fit in a window of ``l`` sites, with ``S(l)`` the
   entropy of ``l`` reference qubits (#337, #339). ``S(l)`` is the number of
   input bits of the window that need functions reaching outside it. Data:
   ``figures/summary_locality.json`` of #339.

Usage::

    uv run python experiments/clifford_ridge/plot_ridge_order.py \\
        --transition outputs/clifford_ridge \\
        --locality experiments/clifford_ridge/figures/summary_locality.json \\
        --out experiments/clifford_ridge/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P_C = 0.161  # best collapse of #346
NU = 1.30
SIZE_COLORS = {
    16: "#9ecae1",
    32: "#4292c6",
    64: "#08519c",
    128: "#08306b",
    256: "#000000",
}
P_COLORS = {
    "0.04": "#1f77b4", "0.1": "#6baed6", "0.14": "#9e9ac8", "0.16": "#555555",
    "0.18": "#fd8d3c", "0.22": "#f16913", "0.3": "#d62728", "0.5": "#7f0000",
}  # fmt: skip


def load_transition(root: Path) -> dict[int, dict]:
    """``T = 2N`` transition runs, keyed by ``N``."""
    out = {}
    for f in sorted(root.glob("transition_N*_T*_s0.json")):
        d = json.loads(f.read_text())
        if d["t"] == 2 * d["n"]:
            out[int(d["n"])] = d
    return out


def shade_phases(ax, label: bool = True) -> None:
    ax.axvline(P_C, color="k", lw=0.8, ls="--")
    if label:
        ax.text(0.03, 0.04, "volume law\n(information\nhidden)", transform=ax.transAxes,
                fontsize=8, color="#1f4e79", va="bottom")  # fmt: skip
        ax.text(0.97, 0.04, "area law\n(information\nread)", transform=ax.transAxes,
                fontsize=8, color="#7f0000", ha="right", va="bottom")  # fmt: skip


def figure_rank(runs: dict[int, dict], out: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    summary: dict = {}
    for n, d in sorted(runs.items()):
        p = np.array([r["p"] for r in d["per_p"]])
        delta = np.array([r["delta"][0] for r in d["per_p"]])
        err = np.array([r["delta"][1] for r in d["per_p"]])
        c = SIZE_COLORS.get(n, "k")
        axes[0].errorbar(p, 1 - delta / n, yerr=err / n, fmt="o-", ms=3, color=c,
                         label=f"N = {n}")  # fmt: skip
        axes[1].errorbar(p, delta, yerr=err, fmt="o-", ms=3, color=c, label=f"N = {n}")
        x = (p - P_C) * n ** (1 / NU)
        axes[2].plot(x, delta, "o", ms=3, color=c, label=f"N = {n}")
        summary[f"N{n}"] = {"p": p.tolist(), "delta": delta.tolist()}
    ax = axes[0]
    shade_phases(ax)
    ax.set(xlabel="measurement rate p", ylabel="log2(rank of ridge) / N = 1 − Δ/N",
           ylim=(0.6, 1.02),
           title="(a) share of the N input bits the ridge can represent")  # fmt: skip
    ax.legend(fontsize=8, loc="center right")
    ax = axes[1]
    shade_phases(ax, label=False)
    ax.set(xlabel="measurement rate p", ylabel="Δ = N − log2(rank)  (bits)",
           yscale="log",
           title="(b) input bits the ridge cannot represent")  # fmt: skip
    ax.annotate(f"curves cross at p_c ≈ {P_C}", xy=(P_C, 1.7), xytext=(0.2, 12),
                arrowprops={"arrowstyle": "->"}, fontsize=8)  # fmt: skip
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="(p − p_c) N^(1/ν)", ylabel="Δ", yscale="log",
           title=f"(c) one curve for all N  (p_c = {P_C}, ν = {NU})")  # fmt: skip
    ax.legend(fontsize=8)
    fig.suptitle(
        "Order parameter 1 — without the final readout, the rank of the ridge "
        "(dimension of Span{E_m} = 2^(N − Δ)), Clifford QELM, T = 2N, K = ∞",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "ridge_order_rank.png", dpi=150)
    plt.close(fig)
    return summary


def figure_locality(loc: dict, out: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    summary: dict = {}
    n_big = int(loc["N"])
    # (a) S(l) at the largest N: volume, log, area law.
    ax = axes[0]
    for key in ("0.04", "0.1", "0.16", "0.22", "0.3"):
        s = np.array(loc["S"][key])
        ell = np.arange(1, s.size + 1)
        ax.plot(ell, 1 - s / ell, "-", lw=2, color=P_COLORS[key], label=f"p = {key}")
    ax.set(xlabel="window size l (sites)", xscale="log", ylim=(0, 1.02),
           ylabel="1 − S(l)/l",
           title=f"(a) window functions that stay local, N = {n_big}")  # fmt: skip
    ax.legend(fontsize=8)
    # (b) S(N/2) against N.
    ax = axes[1]
    sizes = np.array(loc["N_list"], dtype=float)
    for key, vals in loc["s_half"].items():
        ax.loglog(sizes, vals, "o-", color=P_COLORS.get(key, "k"), label=f"p = {key}")
    ax.loglog(sizes, sizes / 4, "k:", lw=0.8)
    ax.text(sizes[-1] * 0.9, sizes[-1] / 4 * 1.15, "∝ N", fontsize=8, ha="right")
    ax.set_xticks(sizes, [str(int(n)) for n in sizes])
    ax.minorticks_off()
    ax.set(xlabel="N", ylabel="S(N/2)  (bits)",
           title="(b) non-local bits of half the chain")  # fmt: skip
    ax.legend(fontsize=7, ncol=2)
    # (c) growth exponent d log S(N/2) / d log N between successive sizes.
    ax = axes[2]
    keys = sorted(loc["s_half"], key=float)
    p = np.array([float(k) for k in keys])
    vals = np.array([loc["s_half"][k] for k in keys])
    expo = np.diff(np.log(vals), axis=1) / np.diff(np.log(sizes))[None, :]
    for j in range(expo.shape[1]):
        lab = f"N = {int(sizes[j])} → {int(sizes[j + 1])}"
        c = SIZE_COLORS.get(int(sizes[j + 1]), "k")
        ax.plot(p, expo[:, j], "o-", color=c, label=lab)
    shade_phases(ax, label=False)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set(xlabel="measurement rate p", ylabel="d log S(N/2) / d log N",
           ylim=(-0.1, 1.05),
           title="(c) growth exponent: 1 (volume) → 0 (area)")  # fmt: skip
    ax.legend(fontsize=8, loc="upper right")
    summary["exponent"] = {"p": p.tolist(), "pairs": sizes.tolist(),
                           "values": expo.tolist()}  # fmt: skip
    fig.suptitle(
        "Order parameter 2 — with the final readout, the locality of the ridge "
        "dictionary (rank 2^N at every p; 2^(l − S(l)) functions fit in l sites), "
        "Clifford QELM, T = 2N, K = ∞",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "ridge_order_locality.png", dpi=150)
    plt.close(fig)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--locality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "rank": figure_rank(load_transition(args.transition), args.out),
        "locality": figure_locality(json.loads(args.locality.read_text()), args.out),
    }
    (args.out / "summary_ridge_order.json").write_text(json.dumps(summary))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
