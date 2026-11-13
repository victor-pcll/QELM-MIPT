#!/usr/bin/env python3
"""Figure of issue #352: trajectory entanglement against ridge locality, Haar.

Usage::

    uv run python experiments/clifford_ridge/plot_haar_order.py \\
        --traj outputs/haar_order --span outputs/haar_span \\
        --out experiments/clifford_ridge/figures

``--traj``: ``traj_<gates>_N*_s0.json`` of ``haar_order.py``. ``--span``:
``haar_span_N*_seed*.json`` of ``haar_span.py`` (#339).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P_C_HAAR = 0.29  # #322, this geometry
SIZE = {3: "#deebf7", 4: "#c6dbef", 5: "#9ecae1", 6: "#6baed6", 7: "#4292c6",
        8: "#2171b5", 10: "#08519c", 12: "#08306b", 14: "#3f007d",
        16: "#000000"}  # fmt: skip


def load_traj(root: Path, gates: str) -> dict[int, dict]:
    out = {}
    for f in sorted(root.glob(f"traj_{gates}_N*_s0.json")):
        d = json.loads(f.read_text())
        out[int(d["n"])] = d
    return out


def load_span(root: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for f in sorted(root.glob("haar_span_N*_seed*.json")):
        d = json.loads(f.read_text())
        out.setdefault(int(d["n"]), []).append(d)
    return out


def col(d: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.array([r["p"] for r in d["per_p"]])
    y = np.array([r[key] for r in d["per_p"]])
    e = np.array([r[f"{key}_sem"] for r in d["per_p"]])
    return p, y, e


def vline(ax) -> None:
    ax.axvline(P_C_HAAR, color="k", ls="--", lw=0.8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--span", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    haar = load_traj(args.traj, "haar")
    cliff = load_traj(args.traj, "clifford")
    span = load_span(args.span)
    big = {n: d for n, d in haar.items() if n >= 8 and n % 4 == 0}
    summary: dict = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) S(N/2) against p.
    ax = axes[0, 0]
    for n, d in sorted(big.items()):
        p, y, e = col(d, "S_half")
        ax.errorbar(
            p, y, yerr=e, fmt="o-", ms=3, color=SIZE.get(n, "k"), label=f"Haar N={n}"
        )
    for n, d in sorted(cliff.items()):
        if n >= 8 and n % 4 == 0:
            p, y, e = col(d, "S_half")
            ax.plot(p, y, "--", color=SIZE.get(n, "k"), lw=1, label=f"Clifford N={n}")
    vline(ax)
    ax.set(xlabel="p", ylabel="S(N/2) of psi_m (bits)",
           title="(a) half-chain entanglement of the effects")  # fmt: skip
    ax.legend(fontsize=7, ncol=2)

    # (b) growth exponent between successive sizes (Haar).
    ax = axes[0, 1]
    sizes = sorted(big)
    expo = {}
    pa = np.array([])
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        pa, ya, _ = col(big[a], "S_half")
        _, yb, _ = col(big[b], "S_half")
        ex = np.log(yb / ya) / np.log(b / a)
        expo[f"{a}-{b}"] = ex.tolist()
        ax.plot(pa, ex, "o-", ms=3, color=SIZE.get(b, "k"), label=f"N = {a} → {b}")
    summary["haar_growth_exponent"] = {"p": pa.tolist(), **expo}
    vline(ax)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set(xlabel="p", ylabel="d log S(N/2) / d log N",
           title="(b) growth exponent of S(N/2), Haar")  # fmt: skip
    ax.legend(fontsize=7)

    # (c) tripartite information (4 | N), Haar.
    ax = axes[0, 2]
    for n, d in sorted(big.items()):
        if "I3" in d["per_p"][0]:
            p, y, e = col(d, "I3")
            ax.errorbar(
                p, y, yerr=e, fmt="o-", ms=3, color=SIZE.get(n, "k"), label=f"N={n}"
            )
            summary[f"haar_I3_N{n}"] = y.tolist()
    vline(ax)
    ax.set(xlabel="p", ylabel="I3 (bits)",
           title="(c) tripartite information of psi_m, Haar")  # fmt: skip
    ax.legend(fontsize=8)

    # (d) local fraction of the window functions, as in Clifford: 1 - S(l)/l.
    ax = axes[1, 0]
    n_prof = max(big or haar)
    d = haar[n_prof]
    for p_show, c in ((0.05, "#1f77b4"), (0.15, "#6baed6"), (0.3, "#555555"),
                      (0.5, "#d62728")):  # fmt: skip
        r = min(d["per_p"], key=lambda row: abs(row["p"] - p_show))
        s = np.array(r["S_profile"])[1 : n_prof // 2 + 1]
        ell = np.arange(1, s.size + 1)
        ax.plot(ell, 1 - s / ell, "o-", color=c, label=f"p = {r['p']}")
    ax.set(xlabel="block l (left end)", ylabel="1 − S(l)/l", ylim=(0, 1.02),
           title=f"(d) Clifford-style locality profile, Haar N={n_prof}")  # fmt: skip
    ax.legend(fontsize=8)

    def capture(runs: list[dict], kind: str) -> tuple[list, list]:
        ps = runs[0]["p"]
        cap = [np.mean([np.mean(r["per_kind"][kind][j]["full"]["window"]["2"])
                        for r in runs]) for j in range(len(ps))]  # fmt: skip
        return ps, cap

    # (e) same N: locality of one trajectory against locality of the ridge.
    ax = axes[1, 1]
    n_cmp = 6 if 6 in span else max(span)
    for kind, c in (("haar", "#08519c"), ("clifford", "#d94801")):
        if n_cmp in span:
            ps, cap = capture(span[n_cmp], kind)
            ax.plot(ps, cap, "o-", ms=3, color=c,
                    label=f"{kind}: ridge, dim(Span ∩ V_2)/9")  # fmt: skip
            summary[f"capture2_{kind}_N{n_cmp}"] = {"p": ps, "value": cap}
        traj = haar if kind == "haar" else cliff
        if n_cmp in traj:
            d = traj[n_cmp]
            p = [r["p"] for r in d["per_p"]]
            loc = [1 - r["S_profile"][2] / 2 for r in d["per_p"]]
            ax.plot(p, loc, "s--", ms=3, color=c,
                    label=f"{kind}: one psi_m, 1 − S(2)/2")  # fmt: skip
            summary[f"traj_local2_{kind}_N{n_cmp}"] = {"p": p, "value": loc}
    vline(ax)
    ax.set(xlabel="p", ylabel="local fraction, 2-site window", ylim=(0, 1.02),
           title=f"(e) trajectory locality vs ridge locality, N={n_cmp}")  # fmt: skip
    ax.legend(fontsize=7)

    # (f) Haar ridge: local capture follows the size of the span.
    ax = axes[1, 2]
    for n, runs in sorted(span.items()):
        ps, cap = capture(runs, "haar")
        ang = [np.mean([r["per_kind"]["haar"][j]["full"]["angle"] for r in runs]) / 3**n
               for j in range(len(ps))]  # fmt: skip
        c = SIZE.get(n, "k")
        ax.plot(ps, cap, "o-", ms=3, color=c, label=f"N={n}: dim(Span ∩ V_2)/9")
        ax.plot(ps, ang, ":", color=c, label=f"N={n}: rank / 3^N")
        summary[f"angle_rank_haar_N{n}"] = {"p": ps, "value": ang}
        summary[f"capture2_haar_N{n}"] = {"p": ps, "value": cap}
    vline(ax)
    ax.set(xlabel="p", ylabel="fraction", ylim=(0, 1.02),
           title="(f) Haar ridge: local capture tracks the rank")  # fmt: skip
    ax.legend(fontsize=6, ncol=2)

    fig.suptitle(
        "Haar QELM, open brickwork, T = 2N, full final readout, K = ∞. Top: "
        "entanglement of the rank-one effects E_m = |psi_m><psi_m| (Born records). "
        "Bottom: the exact ridge span (#339). Dashed line: Haar p_c ≈ 0.29.",
        fontsize=9,
    )
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "haar_order.png", dpi=140)
    plt.close(fig)
    (args.out / "summary_haar_order.json").write_text(json.dumps(summary))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
