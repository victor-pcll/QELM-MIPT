#!/usr/bin/env python3
"""Figures of the Fourier filter of the record features (issue #341).

Reads ``transfer_N*_s*[_tag].json`` (chunks merged per ``N``) and
``validate_N*_s*.json`` from the output folder of ``fourier_filter.py``,
writes ``figures/fourier_filter.png`` and ``figures/summary_fourier_filter.json``.

Usage::

    uv run python experiments/clifford_ridge/plot_fourier_filter.py \
        outputs/fourier_filter
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
PICKS = (0.04, 0.1, 0.16, 0.22, 0.3, 0.5)


def load(folder: Path) -> tuple[dict, list]:
    chunks: dict = defaultdict(list)
    for path in sorted(folder.glob("transfer_N*_s*.json")):
        d = json.loads(path.read_text())
        chunks[d["n"]].append(d)
    runs = {}
    for n, parts in chunks.items():
        run = dict(parts[0])
        rows = {r["p"]: r for part in parts for r in part["per_p"]}
        run["p"] = sorted(rows)
        run["per_p"] = [rows[p] for p in run["p"]]
        runs[n] = run
    val = [
        json.loads(p.read_text()) for p in sorted(folder.glob("validate_N*_s*.json"))
    ]
    return runs, val


def leaf(run: dict, *path: str) -> tuple[np.ndarray, np.ndarray]:
    m, e = [], []
    for row in run["per_p"]:
        x = row
        for k in path:
            x = x[k]
        m.append(x["mean"])
        e.append(x["sem"])
    return np.asarray(m, dtype=float), np.asarray(e, dtype=float)


def palette(count: int, cmap: str = "viridis") -> list:
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.1, 0.9, max(count, 1))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    runs, val = load(args.folder)
    if not runs:
        raise SystemExit("No transfer run.")
    out = args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ns = sorted(runs)
    big = runs[ns[-1]]
    small = runs[ns[0]]
    cols = dict(zip(ns, palette(len(ns)), strict=True))
    summary: dict = {"N": ns}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Number of visible modes per site of each readout.
    ax = axes[0, 0]
    for n in ns:
        run = runs[n]
        m, e = leaf(run, "L1", "modes_per_site")
        ax.errorbar(
            run["p"], m, e, color=cols[n], marker="o", ms=3, label=f"bits, N={n}"
        )
        summary[f"L1_modes_N{n}"] = m.tolist()
        if "L12" in run["per_p"][0]:
            m, e = leaf(run, "L12", "modes_per_site")
            ax.errorbar(run["p"], m, e, color=cols[n], ls="--", marker="s", ms=3,
                        label=f"bits + pairs, N={n}")  # fmt: skip
            summary[f"L12_modes_N{n}"] = m.tolist()
        if "G_modes_per_site" in run["per_p"][0]:
            m, e = leaf(run, "G_modes_per_site")
            ax.errorbar(run["p"], m, e, color=cols[n], ls=":", marker="^", ms=3,
                        label=f"full record G, N={n}")  # fmt: skip
            summary[f"G_modes_N{n}"] = m.tolist()
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("distinct visible Fourier modes per site")
    ax.set_title(
        "Size of the dictionary: features of 1 bit, of 2 bits, all", fontsize=10
    )
    ax.legend(fontsize=6, ncol=2)

    # (b) Transfer function: mode density over (weight, p), bit features.
    ax = axes[0, 1]
    hist = np.array(leaf(big, "L1", "weight_hist")[0])
    wmax = int(np.max(np.nonzero(hist.sum(axis=0) > 1e-4)[0])) + 1
    mesh = ax.pcolormesh(big["p"], np.arange(wmax), np.log10(hist[:, :wmax].T + 1e-4),
                         shading="nearest", cmap="magma")  # fmt: skip
    fig.colorbar(mesh, ax=ax, label="log10 modes per site")
    ax.axvline(P_C, color="w", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("weight |a| (eigenvalue 2^-|a|)")
    ax.set_title(
        f"N = {big['n']}: transfer function C(p) of the bit features", fontsize=10
    )

    # (c) Weight distribution: bit features vs bits + pairs vs G (smallest N).
    ax = axes[0, 2]
    picks = [p for p in PICKS if p in small["p"]]
    for p, c in zip(picks, palette(len(picks), "coolwarm"), strict=True):
        row = small["per_p"][small["p"].index(p)]
        for key, ls in (("L1", "-"), ("L12", "--"), ("G_weight_hist", ":")):
            if key not in row:
                continue
            h = np.asarray(row[key]["weight_hist"]["mean"] if key != "G_weight_hist"
                           else row[key]["mean"])  # fmt: skip
            w = np.arange(h.size)
            keep = h > 0
            ax.semilogy(w[keep], h[keep], color=c, ls=ls, marker="o", ms=2,
                        label=f"p={p}" if key == "L1" else None)  # fmt: skip
    ax.set_xlabel("weight |a|")
    ax.set_ylabel("modes per site")
    ax.set_title(
        f"N = {small['n']}: weights of the modes (bits: solid, + pairs: dashed, "
        "G: dotted)", fontsize=9
    )  # fmt: skip
    ax.legend(fontsize=7)

    # (d) E_perp of targets: bit features vs full record, largest N.
    ax = axes[1, 0]
    summary["e_perp"] = {}
    for fam, c in zip(("Z", "X", "ZZ_1", "ZX_1"), palette(4, "tab10"), strict=True):
        m1, e1 = leaf(big, "L1", "member", fam)
        mg, eg = leaf(big, "G_member", fam)
        ax.errorbar(
            big["p"], 1 - m1, e1, color=c, marker="o", ms=3, label=f"{fam}: bits"
        )
        ax.errorbar(big["p"], 1 - mg, eg, color=c, ls=":", label=f"{fam}: full record")
        summary["e_perp"][f"{fam}_L1"] = (1 - m1).tolist()
        summary["e_perp"][f"{fam}_G"] = (1 - mg).tolist()
    ax.axvline(P_C, color="0.4", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("E_perp (K = inf, lambda -> 0+)")
    ax.set_title(f"N = {big['n']}: targets on one or two sites", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # (e) Modes transmitted at K shots, and the best p for each K.
    ax = axes[1, 1]
    k_grid = big["K"]
    nk = np.array(leaf(big, "L1", "n_K")[0])
    for j, c in zip(range(len(k_grid)), palette(len(k_grid), "plasma"), strict=True):
        ax.plot(big["p"], nk[:, j], color=c, marker="o", ms=3, label=f"K={k_grid[j]}")
    best = [float(big["p"][int(np.argmax(nk[:, j]))]) for j in range(len(k_grid))]
    summary["p_star_K"] = dict(zip(map(str, k_grid), best, strict=True))
    ax.axvline(P_C, color="0.4", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("n(K) = sum_a K w / (1 + (K - 1) w), per site")
    ax.set_title(
        f"N = {big['n']}: modes transmitted at K shots (bits); "
        f"argmax p: {', '.join(f'{b:g}' for b in best)}", fontsize=9
    )  # fmt: skip
    ax.legend(fontsize=7, ncol=2)

    # (f) Validation of the filter at finite K.
    ax = axes[1, 2]
    pts = [pt for v in val for pt in v["points"]]
    errs = []
    for pt in pts:
        c = plt.get_cmap("viridis")(min(pt["weight"], 6) / 6)
        mk = "o" if pt["m"] == 1 else "x"
        ax.scatter(pt["pred_single"], pt["r2"], color=c, marker=mk, s=12)
        if pt["m"] == 1:
            errs += list(np.subtract(pt["r2"], pt["pred_single"]))
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1)
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    summary["validation"] = {"points": len(pts), "rms_single": rms}
    ax.set_xlabel("filter: K w / (1 + (K - 1) w)")
    ax.set_ylabel("empirical population R2 (stim)")
    ax.set_title(
        f"Validation, N = 16 and 32: {len(pts)} modes x K; rms (m = 1) = {rms:.3f}\n"
        "colour: weight 1 (dark) to 6+; x: mode carried by several bits", fontsize=9
    )  # fmt: skip

    fig.tight_layout()
    fig.savefig(out / "fourier_filter.png", dpi=150)
    plt.close(fig)
    (out / "summary_fourier_filter.json").write_text(json.dumps(summary))
    print(f"wrote {out / 'fourier_filter.png'}")


if __name__ == "__main__":
    main()
