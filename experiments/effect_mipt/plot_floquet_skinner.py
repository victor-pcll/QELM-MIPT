"""Figure: dense Fourier weight against half-chain S_2, Skinner Floquet circuit (#333).

Reads ``floquet_N*.json`` (both probes on the same measurement patterns) and
``floquet_pure_N*.json`` (pure state only, both central cuts, larger ``N``)
from ``floquet_skinner.py`` and writes ``floquet_skinner.png`` and
``floquet_skinner_summary.json``:

- top row, the missing Fourier weight ``Delta = N - log2(1 + W)`` after
  ``2N`` half-layers: ``Delta / N``, ``Delta``, the pairwise crossings and a
  free collapse ``Delta`` against ``(p - p_c) N^(1/nu)``;
- bottom row, the half-chain ``S_2``: on the same patterns (one cut), on the
  larger pure-state runs (mean of the two central cuts, after ``4N``
  half-layers), the collapse of ``S_2(p, N) - S_2(p_c, N)`` used by Skinner
  et al., and every ``p_c`` next to the values of the paper.

Errors on fitted ``p_c`` are the spread of the fits leaving one size out
(jackknife).

Usage::

    uv run python experiments/effect_mipt/plot_floquet_skinner.py \
        outputs/effect_mipt --out experiments/effect_mipt/figures
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

# Skinner, Ruhman & Nahum, PRX 9, 031009 (2019), Table 1, n = 2.
PAPER = {"Floquet": (0.22, 0.05), "Haar": (0.27, 0.03)}

plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)


def load(folder: Path, prefix: str) -> dict[int, dict]:
    """Rows pooled by ``N`` and ``p`` (weighted by samples)."""
    acc: dict[int, dict] = {}
    for path in sorted(folder.glob(f"{prefix}_N*_s*.json")):
        d = json.loads(path.read_text())
        n, s = int(d["n"]), int(d["samples"])
        block = acc.setdefault(n, {"checkpoints": d["checkpoints"], "rows": {}})
        if (
            block["checkpoints"][: len(d["checkpoints"])]
            != d["checkpoints"][: len(block["checkpoints"])]
        ):
            raise ValueError(f"checkpoints differ for N={n}")
        for row in d["rows"]:
            r = block["rows"].setdefault(round(row["p"], 3), {})
            for key in ("delta", "s2"):
                m = np.array(row[f"{key}_mean"])
                e = np.array(row[f"{key}_sem"])
                k = len(block["checkpoints"])
                m, e = m[:k], e[:k]
                r.setdefault(key, []).append((s, m, e))
    out: dict[int, dict] = {}
    for n, block in sorted(acc.items()):
        ps = sorted(block["rows"])
        res = {"p": np.array(ps), "checkpoints": block["checkpoints"]}
        for key in ("delta", "s2"):
            means, sems = [], []
            for p in ps:
                parts = block["rows"][p][key]
                total = sum(s for s, _, _ in parts)
                means.append(sum(s * m for s, m, _ in parts) / total)
                sems.append(np.sqrt(sum((s * e) ** 2 for s, _, e in parts)) / total)
            res[key] = np.array(means)
            res[f"{key}_sem"] = np.array(sems)
        res["samples"] = {
            float(p): sum(s for s, _, _ in block["rows"][p]["delta"]) for p in ps
        }
        out[n] = res
    return out


def crossing(p, small, large, err):
    """Zero of a weighted line through ``large - small`` near its last + to - change."""
    d = large - small
    down = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
    if len(down) == 0:
        return None
    i = int(down[-1])
    lo, hi = max(0, i - 1), min(len(p), i + 3)
    w = 1 / np.maximum(err[lo:hi], 1e-6)
    (a, b), cov = np.polyfit(p[lo:hi], d[lo:hi], 1, w=w, cov="unscaled")
    if a >= 0:
        return None
    grad = np.array([b / a**2, -1 / a])
    return {"p": float(-b / a), "err": float(np.sqrt(grad @ cov @ grad))}


def _cost(x, y, e, degree=3) -> float:
    keep = np.abs(x) < 1.5
    if keep.sum() < 2 * degree + 2:
        return 1e9
    coef = np.polyfit(x[keep], y[keep], degree, w=1 / e[keep])
    return float(np.mean(((np.polyval(coef, x[keep]) - y[keep]) / e[keep]) ** 2))


def collapse(curves: dict[int, dict], subtract: bool) -> dict:
    """Free collapse of ``y`` (minus ``y(p_c)`` if ``subtract``), scaled ``p``."""

    def cost(theta, sizes):
        pc, nu = theta
        if not 0.02 < pc < 0.48 or not 0.3 < nu < 6:
            return 1e9
        xs, ys, es = [], [], []
        for n in sizes:
            c = curves[n]
            y = c["y"] - (np.interp(pc, c["p"], c["y"]) if subtract else 0.0)
            xs.append((c["p"] - pc) * n ** (1 / nu))
            ys.append(y)
            es.append(np.maximum(c["e"], 1e-2))
        return _cost(np.concatenate(xs), np.concatenate(ys), np.concatenate(es))

    sizes = list(curves)
    best = min(
        (minimize(cost, [pc0, nu0], args=(sizes,), method="Nelder-Mead")
         for pc0 in (0.1, 0.2, 0.3) for nu0 in (1.0, 2.0)),
        key=lambda r: r.fun,
    )  # fmt: skip
    jack = []
    if len(sizes) > 2:
        for drop in sizes:
            rest = [n for n in sizes if n != drop]
            r = minimize(cost, best.x, args=(rest,), method="Nelder-Mead")
            jack.append(r.x[0])
    k = len(jack)
    err = (
        float(np.sqrt((k - 1) / k * np.sum((np.array(jack) - np.mean(jack)) ** 2)))
        if k
        else 0.0
    )
    return {"p_c": float(best.x[0]), "nu": float(best.x[1]), "p_c_err": err}


def blues(sizes):
    return dict(
        zip(sizes, plt.get_cmap("Blues")(np.linspace(0.4, 1, len(sizes))), strict=True)
    )


def figure(same: dict[int, dict], pure: dict[int, dict], path: Path) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), constrained_layout=True)
    summary: dict = {"paper_n2": PAPER}
    lo, hi = (
        PAPER["Floquet"][0] - PAPER["Floquet"][1],
        PAPER["Floquet"][0] + PAPER["Floquet"][1],
    )

    # Top row: Delta after 2N half-layers, same patterns.
    sizes = list(same)
    col = blues(sizes)
    curves = {
        n: {"p": d["p"], "y": d["delta"][:, 0], "e": d["delta_sem"][:, 0]}
        for n, d in same.items()
    }
    ax = axes[0, 0]
    for n, c in curves.items():
        ax.errorbar(
            c["p"],
            c["y"] / n,
            c["e"] / n,
            color=col[n],
            marker="o",
            ms=3.5,
            lw=1.5,
            label=f"N = {n}",
        )
    ax.axvspan(lo, hi, color="#ffedd5", zorder=-1)
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"$(N - \log_2(1 + W))\,/\,N$")
    ax.set_title(
        r"Missing Fourier weight per site, $\Delta / N$", loc="left", fontsize=10
    )
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for n, c in curves.items():
        ax.errorbar(
            c["p"],
            np.maximum(c["y"], 1e-3),
            c["e"],
            color=col[n],
            marker="o",
            ms=3.5,
            lw=1.5,
        )
    ax.axvspan(lo, hi, color="#ffedd5", zorder=-1)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"$\Delta$ (bits)")
    ax.set_title(
        r"$\Delta$; shaded: Skinner et al. $p_c$ = 0.22 ± 0.05", loc="left", fontsize=10
    )
    ax = axes[0, 2]
    pairs = []
    for n1, n2 in combinations(sizes, 2):
        c1, c2 = curves[n1], curves[n2]
        ps = np.intersect1d(c1["p"], c2["p"])
        i1, i2 = np.isin(c1["p"], ps), np.isin(c2["p"], ps)
        err = np.hypot(c1["e"][i1], c2["e"][i2])
        cr = crossing(ps, c1["y"][i1], c2["y"][i2], err)
        if cr:
            pairs.append({"n1": n1, "n2": n2, **cr})
            x = 1 / np.sqrt(n1 * n2)
            ax.errorbar(
                x, cr["p"], cr["err"], color=col[n2], marker="o", ms=6, capsize=2
            )
            ax.annotate(
                f"{n1}, {n2}",
                (x, cr["p"]),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=8,
                color="0.35",
            )
    ax.axhspan(lo, hi, color="#ffedd5", zorder=-1)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel(r"$1/\sqrt{N N'}$")
    ax.set_ylabel(r"crossing of $\Delta$ for the pair (N, N')")
    ax.set_title("Pairwise crossings of $\\Delta$", loc="left", fontsize=10)
    fit_delta = collapse(curves, subtract=False)
    ax = axes[0, 3]
    for n, c in curves.items():
        ax.errorbar(
            (c["p"] - fit_delta["p_c"]) * n ** (1 / fit_delta["nu"]),
            np.maximum(c["y"], 1e-3),
            c["e"],
            color=col[n],
            marker="o",
            ms=3.5,
            ls="none",
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel(rf"$(p - {fit_delta['p_c']:.3f})\, N^{{1/{fit_delta['nu']:.2f}}}$")
    ax.set_ylabel(r"$\Delta$ (bits)")
    ax.set_title("Free collapse of $\\Delta$", loc="left", fontsize=10)
    summary["delta"] = {"crossings": pairs, "collapse": fit_delta}

    # Bottom row: half-chain S_2.
    ax = axes[1, 0]
    for n, d in same.items():
        ax.errorbar(
            d["p"],
            d["s2"][:, 0],
            d["s2_sem"][:, 0],
            color=col[n],
            marker="o",
            ms=3.5,
            lw=1.5,
            label=f"N = {n}",
        )
    ax.axvspan(lo, hi, color="#ffedd5", zorder=-1)
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"half-chain $S_2$ (bits)")
    ax.set_title("$S_2$ on the same patterns (one cut, 2N)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    summary["s2_same"] = {
        str(n): {"p": d["p"].tolist(), "s2": d["s2"][:, 0].tolist()}
        for n, d in same.items()
    }

    if not pure:  # pure-state runs not there yet
        pure = {n: dict(d, s2=d["s2"][:, :1]) for n, d in same.items()}
    psizes = list(pure)
    pcol = dict(
        zip(
            psizes,
            plt.get_cmap("Greens")(np.linspace(0.4, 1, len(psizes))),
            strict=True,
        )
    )
    last = {n: d["s2"].shape[1] - 1 for n, d in pure.items()}
    pcurves = {
        n: {"p": d["p"], "y": d["s2"][:, last[n]], "e": d["s2_sem"][:, last[n]]}
        for n, d in pure.items()
    }
    ax = axes[1, 1]
    for n, c in pcurves.items():
        ax.errorbar(
            c["p"],
            c["y"],
            c["e"],
            color=pcol[n],
            marker="o",
            ms=3.5,
            lw=1.5,
            label=f"N = {n}",
        )
    ax.axvspan(lo, hi, color="#ffedd5", zorder=-1)
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"half-chain $S_2$ (bits)")
    ax.set_title("$S_2$, mean of the two central cuts (4N)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fit_s2 = collapse(pcurves, subtract=True)
    ax = axes[1, 2]
    for n, c in pcurves.items():
        y = c["y"] - np.interp(fit_s2["p_c"], c["p"], c["y"])
        ax.errorbar(
            (c["p"] - fit_s2["p_c"]) * n ** (1 / fit_s2["nu"]),
            y,
            c["e"],
            color=pcol[n],
            marker="o",
            ms=3.5,
            ls="none",
        )
    ax.set_xlabel(rf"$(p - {fit_s2['p_c']:.3f})\, N^{{1/{fit_s2['nu']:.2f}}}$")
    ax.set_ylabel(r"$S_2(p, N) - S_2(p_c, N)$")
    ax.set_title("Collapse of $S_2$ (procedure of the paper)", loc="left", fontsize=10)
    summary["s2_pure"] = {
        "collapse": fit_s2,
        "sizes": {
            str(n): {"p": c["p"].tolist(), "s2": c["y"].tolist()}
            for n, c in pcurves.items()
        },
    }

    ax = axes[1, 3]
    w = (
        np.array([1 / max(c["err"], 1e-3) ** 2 for c in pairs])
        if pairs
        else np.array([])
    )
    mean_cross = (
        float(np.sum(w * [c["p"] for c in pairs]) / w.sum()) if pairs else np.nan
    )
    spread = float(np.std([c["p"] for c in pairs])) if pairs else 0.0
    # Where S_2 stops growing with N: last p with S_2(N_max) - S_2(N_min) > 2 sigma
    # and the next grid point.
    small, large = pcurves[psizes[0]], pcurves[psizes[-1]]
    diff = large["y"] - small["y"]
    sig = np.hypot(large["e"], small["e"])
    grows = np.flatnonzero(diff > 2 * sig)
    i = int(grows[-1]) if len(grows) else 0
    lo_s, hi_s = (
        float(large["p"][i]),
        float(large["p"][min(i + 1, len(large["p"]) - 1)]),
    )
    summary["s2_pure"]["size_dependence_ends"] = [lo_s, hi_s]
    rows = [
        ("Delta, crossings (this work)", mean_cross, spread, "#1d4ed8"),
        (
            "Delta, free collapse (this work)",
            fit_delta["p_c"],
            fit_delta["p_c_err"],
            "#1d4ed8",
        ),
        ("S2, collapse (this work)", fit_s2["p_c"], fit_s2["p_c_err"], "#15803d"),
        (
            f"S2 stops growing with N (N = {psizes[0]} vs {psizes[-1]})",
            (lo_s + hi_s) / 2,
            (hi_s - lo_s) / 2,
            "#15803d",
        ),
        ("S2, Floquet (Skinner et al.)", *PAPER["Floquet"], "#c2410c"),
        ("S2, Haar (Skinner et al.)", *PAPER["Haar"], "0.5"),
    ]
    for i, (label, v, e, c) in enumerate(rows):
        ax.errorbar(v, -i, xerr=e, color=c, marker="o", ms=7, capsize=3)
        ax.text(
            0.01,
            -i + 0.28,
            label,
            fontsize=8.5,
            color="0.25",
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks([])
    ax.set_ylim(-len(rows) + 0.4, 0.8)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("$p_c$")
    ax.set_title("Critical points compared", loc="left", fontsize=10)
    summary["compared"] = [{"label": r[0], "p_c": r[1], "err": r[2]} for r in rows]

    fig.suptitle(
        "Floquet circuit of Skinner, Ruhman & Nahum (2019): "
        "dense Fourier weight W(m) against half-chain $S_2$",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/effect_mipt/figures")
    )
    args = parser.parse_args()
    same = load(args.folder, "floquet")
    pure = load(args.folder, "floquet_pure")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = figure(same, pure, args.out / "floquet_skinner.png")
    (args.out / "floquet_skinner_summary.json").write_text(
        json.dumps(summary, indent=1, default=float)
    )
    print(json.dumps({k: summary[k] for k in ("compared",)}, indent=1, default=float))
    print(json.dumps(summary["delta"]["crossings"], indent=0))


if __name__ == "__main__":
    main()
