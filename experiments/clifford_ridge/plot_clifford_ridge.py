#!/usr/bin/env python3
"""Figures of the Clifford QELM: transition, then ridge (issue #337).

Reads ``transition_N*_T*_s*.json`` and ``ridge_N*_T*_seed*.json`` from the
output folder of ``clifford_ridge.py`` and writes the figures and
``summary.json`` (every plotted value, the crossings) in ``figures/``.

Usage::

    uv run python experiments/clifford_ridge/plot_clifford_ridge.py \
        outputs/clifford_ridge
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P_C_LIT = 0.160
LABELS = {
    "z_mid": "Z of middle qubit",
    "x_mid": "X of middle qubit",
    "zz_mid": "ZZ of two neighbours",
    "xx_mid": "XX of two neighbours",
    "deg1": "degree 1",
}


def palette(count: int, cmap: str = "viridis") -> list:
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.1, 0.9, max(count, 1))]


def crossing(p: np.ndarray, a, ea, b, eb) -> tuple[float, bool]:
    """First sign change of ``a - b``, linear interpolation; resolved if
    the difference exceeds two standard errors on the grid points on both
    sides."""
    d = np.asarray(a) - np.asarray(b)
    e = np.sqrt(np.asarray(ea) ** 2 + np.asarray(eb) ** 2)
    for i in range(len(p) - 1):
        if d[i] == 0 or d[i] * d[i + 1] < 0:
            x = p[i] - d[i] * (p[i + 1] - p[i]) / (d[i + 1] - d[i])
            ok = abs(d[i]) > 2 * e[i] and abs(d[i + 1]) > 2 * e[i + 1]
            return float(x), bool(ok)
    return float("nan"), False


def load_transition(folder: Path) -> dict:
    runs: dict = {}
    for path in sorted(folder.glob("transition_N*_T*_s*.json")):
        d = json.loads(path.read_text())
        runs[(d["n"], d["t"] // d["n"])] = d
    return runs


def load_ridge(folder: Path) -> dict:
    runs: dict = defaultdict(list)
    for path in sorted(folder.glob("ridge_N*_T*_seed*.json")):
        if re.match(r"ridge_N\d+_T\d+_seed\d+\.json", path.name):
            d = json.loads(path.read_text())
            runs[d["n"]].append(d)
    return runs


def fig_transition(runs: dict, out: Path, summary: dict) -> float:
    """Delta/N, S_mid and the single-shot R2 of two monomials; crossings."""
    ratios = sorted({r for (_, r) in runs})
    if not ratios:
        return P_C_LIT
    fig, axes = plt.subplots(
        len(ratios), 4, figsize=(18, 4.2 * len(ratios)), squeeze=False
    )
    summary["transition"] = {}
    pcs = []
    for row, ratio in enumerate(ratios):
        sizes = sorted(n for (n, r) in runs if r == ratio)
        cols = palette(len(sizes))
        series: dict = {}
        for n, c in zip(sizes, cols, strict=True):
            d = runs[(n, ratio)]
            p = np.array([r["p"] for r in d["per_p"]])
            quantities = {
                "delta_per_site": [r["delta_per_site"] for r in d["per_p"]],
                "s_mid": [r["s_mid"] for r in d["per_p"]],
                "r2_full_z_mid": [r["r2_single_full"]["z_mid"] for r in d["per_p"]],
                "r2_mid_z_mid": [r["r2_single_mid"]["z_mid"] for r in d["per_p"]],
            }
            for col, (key, vals) in enumerate(quantities.items()):
                m, e = np.array(vals).T
                axes[row, col].errorbar(
                    p, m, e, color=c, marker="o", ms=2.5, capsize=1.5, label=f"N={n}"
                )
                series.setdefault(key, {})[n] = (p, m, e)
                summary["transition"][f"T{ratio}N_N{n}_{key}"] = m.tolist()
        for col, (key, title) in enumerate(
            (
                ("delta_per_site", r"$\Delta / N$ (unread input bits per site)"),
                ("s_mid", "entropy of the middle reference"),
                ("r2_full_z_mid", "single-shot R2, Z middle, full record"),
                ("r2_mid_z_mid", "single-shot R2, Z middle, mid record"),
            )
        ):
            ax = axes[row, col]
            ax.axvline(P_C_LIT, color="0.5", ls="--", lw=1)
            ax.set_xlabel("p")
            ax.set_title(f"T = {ratio}N: {title}", fontsize=10)
            crosses = []
            for a, b in zip(sizes, sizes[1:], strict=False):
                pa, ma, ea = series[key][a]
                _, mb, eb = series[key][b]
                x, ok = crossing(pa, ma, ea, mb, eb)
                crosses.append({"pair": [a, b], "p": x, "resolved": ok})
                if np.isfinite(x):
                    ax.axvline(x, color="r" if ok else "orange", lw=0.8, alpha=0.7)
            summary["transition"][f"T{ratio}N_{key}_crossings"] = crosses
            if key == "delta_per_site":
                pcs += [c["p"] for c in crosses if c["resolved"]]
        axes[row, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "transition.png", dpi=150)
    plt.close(fig)
    p_c = float(np.mean(pcs)) if pcs else P_C_LIT
    summary["p_c_used"] = p_c
    return p_c


def _stat(runs: list[dict], fn) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[fn(row) for row in r["per_p"]] for r in runs], dtype=float)
    sem = a.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0 * a[0]
    return a.mean(axis=0), sem


def fig_ridge(runs: dict, out: Path, summary: dict, p_c: float) -> None:
    """Test error at lambda* against p, per N, one panel per K and target."""
    if not runs:
        return
    sizes = sorted(runs)
    ks = [str(k) for k in runs[sizes[0]][0]["K"]]
    names = ("z_mid", "x_mid", "zz_mid", "deg1")
    fig, axes = plt.subplots(
        len(names), len(ks), figsize=(4.4 * len(ks), 3.5 * len(names)), squeeze=False
    )
    summary["ridge"] = {}
    for row, name in enumerate(names):
        for col, K in enumerate(ks):
            ax = axes[row, col]
            for n, c in zip(sizes, palette(len(sizes)), strict=True):
                p = np.array([r["p"] for r in runs[n][0]["per_p"]])
                m, e = _stat(
                    runs[n],
                    lambda r, K=K, name=name: r["targets"][name]["ridge"][K][
                        "test_mse"
                    ],
                )
                ax.errorbar(
                    p, 1 - m, e, color=c, marker="o", ms=3, capsize=2, label=f"N={n}"
                )
                summary["ridge"][f"N{n}_{name}_K{K}_r2"] = (1 - m).tolist()
            ax.axvline(p_c, color="0.4", ls="--", lw=1)
            ax.set_xlabel("p")
            ax.set_ylabel("test R2 at lambda*")
            ax.set_title(f"{LABELS[name]}, K = {K}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ridge_r2.png", dpi=150)
    plt.close(fig)


def fig_bias_variance(runs: dict, out: Path, summary: dict, p_c: float) -> None:
    """Bias^2, variance and lambda* against p (target Z middle and degree 1)."""
    if not runs:
        return
    sizes = sorted(runs)
    ks = [str(k) for k in runs[sizes[0]][0]["K"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    summary["bias_variance"] = {}
    for row, name in enumerate(("z_mid", "deg1")):
        for n, ls in zip(sizes, ("-", "--", ":", "-."), strict=False):
            p = np.array([r["p"] for r in runs[n][0]["per_p"]])
            for K, c in zip(ks, palette(len(ks), "plasma"), strict=True):
                b, _ = _stat(
                    runs[n], lambda r, K=K: r["targets"][name]["ridge"][K]["bias2"]
                )
                v, _ = _stat(
                    runs[n], lambda r, K=K: r["targets"][name]["ridge"][K]["variance"]
                )
                lam = np.exp(
                    _stat(
                        runs[n],
                        lambda r, K=K: np.log(
                            r["targets"][name]["ridge"][K]["lambda_star"]
                        ),
                    )[0]
                )
                lab = f"K={K}" if n == sizes[0] else None
                axes[row, 0].plot(p, b, color=c, ls=ls, marker="o", ms=2.5, label=lab)
                axes[row, 1].plot(p, v, color=c, ls=ls, marker="o", ms=2.5)
                axes[row, 2].semilogy(p, lam, color=c, ls=ls, marker="o", ms=2.5)
                summary["bias_variance"][f"N{n}_{name}_K{K}"] = {
                    "bias2": b.tolist(),
                    "variance": v.tolist(),
                    "lambda_star": lam.tolist(),
                }
        for col, title in enumerate(
            ("bias² at lambda*", "variance at lambda*", "lambda* (relative)")
        ):
            ax = axes[row, col]
            ax.axvline(p_c, color="0.4", ls="--", lw=1)
            ax.set_xlabel("p")
            ax.set_title(f"{LABELS[name]}: {title}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].text(
        0.02,
        0.03,
        "line style: N = " + ", ".join(map(str, sizes)) + " (solid, dashed, dotted)",
        transform=axes[0, 0].transAxes,
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out / "ridge_bias_variance.png", dpi=150)
    plt.close(fig)


def fig_spectrum(runs: dict, out: Path, summary: dict, p_c: float) -> None:
    """n_learn against p and p*(K), per N."""
    if not runs:
        return
    sizes = sorted(runs)
    kpop = runs[sizes[0]][0]["K_population"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    summary["spectrum"] = {}
    for n, c in zip(sizes, palette(len(sizes)), strict=True):
        p = np.array([r["p"] for r in runs[n][0]["per_p"]])
        for ax, K in zip(axes[:2], (16, 256), strict=True):
            m, e = _stat(runs[n], lambda r, K=K: r["n_learn"][kpop.index(K)])
            ax.errorbar(p, m, e, color=c, marker="o", ms=3, capsize=2, label=f"N={n}")
            summary["spectrum"][f"N{n}_nlearn_K{K}"] = m.tolist()
        ks = [str(k) for k in runs[n][0]["K"]]
        best = []
        for K in ks:
            m, _ = _stat(
                runs[n], lambda r, K=K: r["targets"]["z_mid"]["ridge"][K]["test_mse"]
            )
            best.append(float(p[int(np.argmin(m))]))
        axes[2].plot([int(k) for k in ks], best, color=c, marker="o", label=f"N={n}")
        summary["spectrum"][f"N{n}_p_star_z_mid"] = best
    for ax, K in zip(axes[:2], (16, 256), strict=True):
        ax.axvline(p_c, color="0.4", ls="--", lw=1)
        ax.set_xlabel("p")
        ax.set_title(f"n_learn (local features), K = {K}")
        ax.legend(fontsize=8)
    axes[2].axhline(p_c, color="0.4", ls="--", lw=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("K")
    axes[2].set_title("p*(K), minimum of the test error, Z middle")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ridge_spectrum.png", dpi=150)
    plt.close(fig)


def fig_double_descent(runs: dict, out: Path, summary: dict) -> None:
    """Test error against S/d at lambda -> 0 and lambda*, per p (smallest N)."""
    with_dd = [n for n in sorted(runs) if "double_descent" in runs[n][0]["per_p"][0]]
    if not with_dd:
        return
    n = with_dd[0]
    rs = runs[n]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    summary["double_descent"] = {"N": n}
    ps = [r["p"] for r in rs[0]["per_p"]]
    for i, (p, c) in enumerate(zip(ps, palette(len(ps), "coolwarm"), strict=True)):
        dd = [r["per_p"][i]["double_descent"] for r in rs]
        # The sizes are a prefix of the same S/d grid; it is cut where S
        # exceeds the training set, which depends on d, so keep the common part.
        n_common = min(len(x["S"]) for x in dd)
        s = np.mean([np.array(x["S"][:n_common]) / x["d"] for x in dd], axis=0)
        for ax, key in zip(axes, ("mse_lambda_min", "mse_lambda_star"), strict=True):
            m = np.mean([x["targets"]["z_mid"][key][:n_common] for x in dd], axis=0)
            ax.semilogy(s, m, color=c, marker="o", ms=3, label=f"p={p}")
            summary["double_descent"][f"p{p}_{key}"] = m.tolist()
    for ax, title in zip(axes, ("lambda = 1e-8", "lambda*"), strict=True):
        ax.axvline(1.0, color="0.4", ls=":")
        ax.set_xlabel("S / d (training inputs per feature)")
        ax.set_ylabel("test error (1 - R2), Z middle")
        ax.set_title(
            f"N = {n}, K = {rs[0]['per_p'][0]['double_descent']['K']}: {title}"
        )
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "double_descent.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    out = args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    p_c = fig_transition(load_transition(args.folder), out, summary)
    ridge = load_ridge(args.folder)
    fig_ridge(ridge, out, summary, p_c)
    fig_bias_variance(ridge, out, summary, p_c)
    fig_spectrum(ridge, out, summary, p_c)
    fig_double_descent(ridge, out, summary)
    (out / "summary.json").write_text(json.dumps(summary))
    print(f"wrote figures in {out}")


if __name__ == "__main__":
    main()
