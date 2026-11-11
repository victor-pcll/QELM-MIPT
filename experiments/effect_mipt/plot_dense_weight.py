"""Figures of the dense-encoding Fourier weight of Clifford circuits (#327).

Reads the ``dense_*.json`` files of ``dense_weight.py`` (files of the same
convention and ``N`` are pooled) and writes, per convention and depth
(``2N`` and ``4N`` half-layers), ``dense_<conv>_<depth>.png``:

- ``log2(1 + W) / N`` against ``p``: the share of the dense input that one
  record fixes;
- ``Delta = N - log2(1 + W)`` (Renyi-2 purification entropy) against ``p``;
- the crossings of ``Delta`` for every pair of sizes;
- the collapse against ``(p - p_c) N^(1/nu)``, nothing fitted.

``dense_summary.json`` holds the plotted numbers, the crossings, and a free
two-parameter collapse as a check.

Usage::

    uv run python experiments/effect_mipt/plot_dense_weight.py \
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

# Literature circuit: Gullans & Huse, PRL 125, 070606 (2020). Open
# convention of #314: p_c from the reference probes of #324; nu is the
# literature value, assumed (same universality class).
REFERENCE = {
    "periodic": {"p_c": 0.16, "nu": 1.3, "source": "literature"},
    "open": {"p_c": 0.29, "nu": 1.3, "source": "#324 (p_c +- 0.02), literature (nu)"},
}
TITLES = {
    "periodic": "literature circuit (periodic, measurement after every half-layer)",
    "open": "convention of #314 (open, measurement after every layer)",
}

plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)


def load(folder: Path) -> dict[str, dict[int, dict]]:
    """Pooled mean and standard error of ``k`` by convention and ``N``."""
    acc: dict[str, dict[int, dict]] = {}
    for path in sorted(folder.glob("dense_*_N*.json")):
        d = json.loads(path.read_text())
        n, s = int(d["n"]), int(d["samples"])
        m, e = np.array(d["k_mean"]), np.array(d["k_sem"])
        block = acc.setdefault(d["conv"], {}).get(n)
        if block is None:
            acc[d["conv"]][n] = {
                "p": np.array(d["p"]),
                "checkpoints": d["checkpoints"],
                "sum": s * m,
                "var": (s * e) ** 2,
                "samples": s,
            }
            continue
        if not np.allclose(block["p"], d["p"]):
            raise ValueError(f"p grids differ for {d['conv']} N={n}")
        block["sum"] = block["sum"] + s * m
        block["var"] = block["var"] + (s * e) ** 2
        block["samples"] += s
    out: dict[str, dict[int, dict]] = {}
    for conv, by_n in acc.items():
        out[conv] = {}
        for n, b in sorted(by_n.items()):
            out[conv][n] = {
                "p": b["p"],
                "checkpoints": b["checkpoints"],
                "k": b["sum"] / b["samples"],
                "sem": np.sqrt(b["var"]) / b["samples"],
                "samples": b["samples"],
            }
    return out


def crossing(p: np.ndarray, small: np.ndarray, large: np.ndarray, err: np.ndarray):
    """Crossing of ``Delta`` for two sizes (larger above below ``p_c``).

    Zero of a weighted straight line through the difference on the four grid
    points around its last change of sign from positive to negative; the
    error comes from the fit covariance.
    """
    d = large - small
    down = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
    if len(down) == 0:
        return None
    i = int(down[-1])
    lo, hi = max(0, i - 1), min(len(p), i + 3)
    w = 1 / np.maximum(err[lo:hi], 1e-6)
    coef, cov = np.polyfit(p[lo:hi], d[lo:hi], 1, w=w, cov="unscaled")
    a, b = coef
    if a >= 0:
        return None
    grad = np.array([b / a**2, -1 / a])
    return {"p": float(-b / a), "err": float(np.sqrt(grad @ cov @ grad))}


def collapse_cost(theta, curves, degree: int = 5) -> float:
    """Chi-square per point of one polynomial through all sizes (scaled ``p``)."""
    pc, nu = theta
    x = np.concatenate([(c["p"] - pc) * n ** (1 / nu) for n, c in curves.items()])
    y = np.concatenate([c["delta"] for c in curves.values()])
    e = np.concatenate([np.maximum(c["sem"], 1e-2) for c in curves.values()])
    keep = np.abs(x) < 2.0
    if keep.sum() < 3 * degree:
        return 1e9
    coef = np.polyfit(x[keep], y[keep], degree, w=1 / e[keep])
    return float(np.mean(((np.polyval(coef, x[keep]) - y[keep]) / e[keep]) ** 2))


def figure(conv: str, data: dict[int, dict], depth_index: int, path: Path) -> dict:
    ref = REFERENCE[conv]
    pc, nu = ref["p_c"], ref["nu"]
    sizes = list(data)
    colors = dict(
        zip(sizes, plt.get_cmap("Blues")(np.linspace(0.4, 1, len(sizes))), strict=True)
    )
    curves = {}
    for n, b in data.items():
        k = b["k"][:, depth_index]
        curves[n] = {
            "p": b["p"],
            "k": k,
            "delta": n - k,
            "sem": b["sem"][:, depth_index],
        }
    depth = data[sizes[0]]["checkpoints"][depth_index] // sizes[0]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    ax = axes[0]
    for n, c in curves.items():
        ax.errorbar(
            c["p"], c["k"] / n, c["sem"] / n, color=colors[n], marker="o", ms=3.5,
            lw=1.5, label=f"N = {n}",
        )  # fmt: skip
    ax.axvline(pc, color="0.6", lw=1, ls="--", zorder=0)
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"$\log_2(1 + W)\,/\,N$")
    ax.set_title(
        "Share of the dense input fixed by one record", loc="left", fontsize=10
    )
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for n, c in curves.items():
        ax.errorbar(
            c["p"], c["delta"], c["sem"], color=colors[n], marker="o", ms=3.5, lw=1.5
        )
    ax.axvline(pc, color="0.6", lw=1, ls="--", zorder=0)
    ax.text(pc, 0.98, f" $p_c$ = {pc}", transform=ax.get_xaxis_transform(),
            va="top", color="0.35")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"$\Delta = N - \log_2(1 + W)$ (bits)")
    ax.set_title("Missing Fourier weight", loc="left", fontsize=10)

    ax = axes[2]
    pairs = []
    for n1, n2 in combinations(sizes, 2):
        c1, c2 = curves[n1], curves[n2]
        cr = crossing(c1["p"], c1["delta"], c2["delta"], np.hypot(c1["sem"], c2["sem"]))
        if cr is None:
            continue
        pairs.append({"n1": n1, "n2": n2, **cr})
        x = 1 / np.sqrt(n1 * n2)
        ax.errorbar(
            x, cr["p"], cr["err"], color=colors[n2], marker="o", ms=6, capsize=2
        )
        below = any(np.isclose(x, 1 / np.sqrt(c["n1"] * c["n2"])) for c in pairs[:-1])
        ax.annotate(f"{n1}, {n2}", (x, cr["p"]), xytext=(6, -10 if below else 4),
                    textcoords="offset points", fontsize=8, color="0.35")  # fmt: skip
    ax.axhline(pc, color="0.6", lw=1, ls="--", zorder=0)
    if conv == "open":
        ax.axhspan(pc - 0.02, pc + 0.02, color="0.9", zorder=-1)
    ax.set_xlim(left=0)
    ax.set_ylim(pc - 0.03, pc + 0.03)
    ax.set_xlabel(r"$1/\sqrt{N N'}$")
    ax.set_ylabel(r"crossing of $\Delta$ for the pair (N, N')")
    ax.set_title(f"Pairwise crossings against $p_c$ = {pc}", loc="left", fontsize=10)

    pc_collapse, label = pc, r"Collapse with $p_c$ and $\nu$ above, nothing fitted"
    if conv == "open" and pairs:
        w = np.array([1 / max(c["err"], 1e-4) ** 2 for c in pairs])
        pc_collapse = round(float(np.sum(w * [c["p"] for c in pairs]) / w.sum()), 3)
        label = r"Collapse with $p_c$ = mean crossing, $\nu$ = 1.3"
    ax = axes[3]
    for n, c in curves.items():
        ax.errorbar((c["p"] - pc_collapse) * n ** (1 / nu), c["delta"], c["sem"],
                    color=colors[n], marker="o", ms=3.5, ls="none")  # fmt: skip
    ax.set_xlim(-3, 3)
    ax.set_yscale("log")
    ax.set_xlabel(rf"$(p - {pc_collapse})\, N^{{1/{nu}}}$")
    ax.set_ylabel(r"$\Delta$ (bits)")
    ax.set_title(label, loc="left", fontsize=10)
    fig.suptitle(
        f"Dense-encoding Fourier weight W(m), Clifford, {TITLES[conv]}, "
        f"after {depth}N half-layers",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.savefig(path, dpi=200)
    plt.close(fig)

    fit = minimize(collapse_cost, [pc, nu], args=(curves,), method="Nelder-Mead")
    summary = {
        "reference": ref,
        "depth_half_layers_over_N": depth,
        "crossings": pairs,
        "collapse_p_c_plotted": pc_collapse,
        "collapse_fit": {"p_c": float(fit.x[0]), "nu": float(fit.x[1])},
        "sizes": {
            str(n): {
                "samples": data[n]["samples"],
                "p": c["p"].tolist(),
                "log2_1_plus_w": c["k"].tolist(),
                "delta": c["delta"].tolist(),
                "sem": c["sem"].tolist(),
            }
            for n, c in curves.items()
        },
    }
    if pairs:
        w = np.array([1 / max(c["err"], 1e-4) ** 2 for c in pairs])
        summary["crossing_mean"] = float(np.sum(w * [c["p"] for c in pairs]) / w.sum())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/effect_mipt/figures")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for conv, data in load(args.folder).items():
        checkpoints = data[min(data)]["checkpoints"]
        for i, c in enumerate(checkpoints):
            depth = c // min(data)
            if depth == 1:
                continue
            path = args.out / f"dense_{conv}_{depth}N.png"
            summary[f"{conv}_{depth}N"] = figure(conv, data, i, path)
    (args.out / "dense_summary.json").write_text(json.dumps(summary, indent=1))
    for key, s in summary.items():
        print(key, s.get("crossing_mean"), s["collapse_fit"], len(s["crossings"]))


if __name__ == "__main__":
    main()
