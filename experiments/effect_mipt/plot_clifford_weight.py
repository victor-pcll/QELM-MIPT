"""Figure: the Fourier weight of one encoded qubit crosses at the Clifford MIPT (#325).

Reads the ``clifford_weight_N*.json`` files of ``clifford_weight.py`` (files
of the same ``N`` are pooled) and writes ``clifford_weight.png``:

- the mean weight ``<W>`` against ``p`` for every size, with the literature
  ``p_c``;
- the crossings of every pair of sizes;
- the collapse against ``(p - p_c) N^(1/nu)`` with the literature values
  (nothing fitted).

``clifford_weight_summary.json`` holds the plotted numbers, the crossings,
and a two-parameter collapse fit given as a check.

Usage::

    uv run python experiments/effect_mipt/plot_clifford_weight.py \
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

# Gullans & Huse, PRL 125, 070606 (2020); Li, Chen & Fisher, PRB 100, 134306 (2019).
PC_LIT = 0.16
NU_LIT = 1.3

plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)


def load(folder: Path) -> dict[int, dict]:
    """Pooled mean and standard error of ``W`` by ``N`` (same ``p`` grid required)."""
    pooled: dict[int, dict] = {}
    for path in sorted(folder.glob("clifford_weight_N*.json")):
        d = json.loads(path.read_text())
        n, s = int(d["n"]), int(d["samples"])
        m, e = np.array(d["w_mean"]), np.array(d["w_sem"])
        if n not in pooled:
            pooled[n] = {
                "p": np.array(d["p"]),
                "sum": s * m,
                "var": (s * e) ** 2,
                "samples": s,
            }
            continue
        block = pooled[n]
        if not np.allclose(block["p"], d["p"]):
            raise ValueError(f"p grids differ for N={n}")
        block["sum"] += s * m
        block["var"] += (s * e) ** 2
        block["samples"] += s
    out = {}
    for n, b in sorted(pooled.items()):
        out[n] = {
            "p": b["p"],
            "w": b["sum"] / b["samples"],
            "sem": np.sqrt(b["var"]) / b["samples"],
            "samples": b["samples"],
        }
    return out


def crossing(p: np.ndarray, small: dict, large: dict, half: int = 2) -> dict | None:
    """Crossing of two sizes: zero of a weighted straight line through the difference.

    The difference (large minus small) is fitted on the ``2 * half`` grid
    points around its last change of sign from negative to positive; the
    error comes from the fit covariance.
    """
    d = large["w"] - small["w"]
    e = np.sqrt(large["sem"] ** 2 + small["sem"] ** 2)
    up = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    if len(up) == 0:
        return None
    i = int(up[-1])
    lo, hi = max(0, i - half + 1), min(len(p), i + half + 1)
    coef, cov = np.polyfit(p[lo:hi], d[lo:hi], 1, w=1 / e[lo:hi], cov="unscaled")
    a, b = coef
    if a <= 0:
        return None
    grad = np.array([b / a**2, -1 / a])
    return {"p": float(-b / a), "err": float(np.sqrt(grad @ cov @ grad))}


def collapse_cost(theta: np.ndarray, data: dict[int, dict], degree: int = 5) -> float:
    """Chi-square per point of one polynomial through all sizes (scaled ``p``)."""
    pc, nu = theta
    x = np.concatenate([(b["p"] - pc) * n ** (1 / nu) for n, b in data.items()])
    y = np.concatenate([b["w"] for b in data.values()])
    e = np.concatenate([np.maximum(b["sem"], 1e-3) for b in data.values()])
    keep = np.abs(x) < 2.0
    if keep.sum() < 3 * degree:
        return 1e9
    coef = np.polyfit(x[keep], y[keep], degree, w=1 / e[keep])
    return float(np.mean(((np.polyval(coef, x[keep]) - y[keep]) / e[keep]) ** 2))


def figure(data: dict[int, dict], path: Path) -> dict:
    sizes = list(data)
    colors = dict(
        zip(sizes, plt.get_cmap("Blues")(np.linspace(0.4, 1, len(sizes))), strict=True)
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.axvline(PC_LIT, color="0.6", lw=1, ls="--", zorder=0)
    ax.text(PC_LIT + 0.003, 0.03, f"literature $p_c$ = {PC_LIT}", color="0.35")
    for n in sizes:
        b = data[n]
        ax.errorbar(
            b["p"],
            b["w"],
            b["sem"],
            color=colors[n],
            marker="o",
            ms=4,
            lw=1.6,
            label=f"N = {n}",
        )
    ax.set_xticks(np.arange(0.08, 0.25, 0.04))
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"mean Fourier weight of the encoded qubit, $\langle W \rangle$")
    ax.set_title(
        "Larger sizes carry less weight below $p_c$, more above",
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    pairs = []
    for n1, n2 in combinations(sizes, 2):
        c = crossing(data[n1]["p"], data[n1], data[n2])
        if c is None:
            continue
        pairs.append({"n1": n1, "n2": n2, **c})
        ax.errorbar(
            1 / np.sqrt(n1 * n2),
            c["p"],
            c["err"],
            color=colors[n2],
            marker="o",
            ms=6,
            capsize=2,
        )
        ax.annotate(
            f"{n1}, {n2}",
            (1 / np.sqrt(n1 * n2), c["p"]),
            xytext=(6, 4 if c["p"] >= PC_LIT else -10),
            textcoords="offset points",
            fontsize=8,
            color="0.35",
        )
    ax.axhline(PC_LIT, color="0.6", lw=1, ls="--", zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(PC_LIT - 0.015, PC_LIT + 0.015)
    ax.set_xlabel(r"$1/\sqrt{N N'}$")
    ax.set_ylabel("crossing of the pair (N, N')")
    ax.set_title(
        "Pairwise crossings against the literature $p_c$", loc="left", fontsize=10
    )

    ax = axes[2]
    for n in sizes:
        b = data[n]
        ax.errorbar(
            (b["p"] - PC_LIT) * n ** (1 / NU_LIT),
            b["w"],
            b["sem"],
            color=colors[n],
            marker="o",
            ms=4,
            ls="none",
        )
    ax.set_xlim(-3, 3)
    ax.set_xlabel(rf"$(p - {PC_LIT})\, N^{{1/{NU_LIT}}}$")
    ax.set_ylabel(r"$\langle W \rangle$")
    ax.set_title(
        "Collapse with the literature $p_c$ and $\\nu$, nothing fitted",
        loc="left",
        fontsize=10,
    )

    fig.suptitle(
        "Clifford brickwork, one encoded qubit: $W = |r_m|^2 = 1 - S_2$(reference) "
        "crosses at the measurement-induced transition",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)

    fit = minimize(collapse_cost, [PC_LIT, NU_LIT], args=(data,), method="Nelder-Mead")
    summary = {
        "literature": {"p_c": PC_LIT, "nu": NU_LIT},
        "sizes": {
            str(n): {
                "samples": b["samples"],
                "p": b["p"].tolist(),
                "w_mean": b["w"].tolist(),
                "w_sem": b["sem"].tolist(),
            }
            for n, b in data.items()
        },
        "crossings": pairs,
        "collapse": {
            "cost_literature": collapse_cost(np.array([PC_LIT, NU_LIT]), data),
            "fit_p_c": float(fit.x[0]),
            "fit_nu": float(fit.x[1]),
            "cost_fit": float(fit.fun),
        },
    }
    if pairs:
        w = np.array([1 / c["err"] ** 2 for c in pairs])
        summary["crossing_mean"] = float(np.sum(w * [c["p"] for c in pairs]) / w.sum())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/effect_mipt/figures")
    )
    args = parser.parse_args()
    data = load(args.folder)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = figure(data, args.out / "clifford_weight.png")
    summary_path = args.out / "clifford_weight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in summary if k != "sizes"}, indent=1))


if __name__ == "__main__":
    main()
