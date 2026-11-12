#!/usr/bin/env python3
"""Figures of the effect spectrum and ridge bias/variance (issue #334).

Reads ``level1_N*_seed*.json`` and ``level2_N*_seed*.json`` from the output
folder of ``ridge_spectrum.py`` and writes, next to them in ``figures/``,
the six figures of the README and ``summary.json`` with every plotted value.

Usage::

    uv run python experiments/ridge_spectrum/plot_ridge_spectrum.py \
        outputs/ridge_spectrum
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

P_C = 0.29
P_C_ERR = 0.02
TARGETS = ("k1", "k2", "k3", "x_mid", "z_mid")
LABELS = {
    "k1": "degree 1",
    "k2": "degree 2",
    "k3": "degree 3",
    "x_mid": "X of middle qubit",
    "z_mid": "Z of middle qubit",
}


def load(folder: Path) -> dict[tuple[int, int], list[dict]]:
    """Runs keyed by ``(level, N)``."""
    runs: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(folder.glob("level*_N*_seed*.json")):
        m = re.match(r"level(\d)_N(\d+)_seed\d+\.json", path.name)
        if m:
            runs[(int(m.group(1)), int(m.group(2)))].append(
                json.loads(path.read_text())
            )
    return runs


def stat(values: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over seeds (axis 0)."""
    a = np.asarray(values, dtype=float)
    sem = a.std(axis=0, ddof=1) / np.sqrt(a.shape[0]) if a.shape[0] > 1 else 0 * a[0]
    return a.mean(axis=0), sem


def p_values(runs: list[dict]) -> np.ndarray:
    return np.array([row["p"] for row in runs[0]["per_p"]])


def mark_pc(ax: plt.Axes) -> None:
    ax.axvspan(P_C - P_C_ERR, P_C + P_C_ERR, color="0.85", zorder=0)
    ax.axvline(P_C, color="0.4", ls="--", lw=1, zorder=1)


def colors(count: int, cmap: str = "viridis") -> list:
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.1, 0.9, max(count, 1))]


def fig_spectrum(runs: dict, out: Path, summary: dict) -> None:
    """Figure 1: sorted mu_k of F (level I, largest N) per p, noise floors."""
    sizes = sorted(n for (lvl, n) in runs if lvl == 1)
    if not sizes:
        return
    n = sizes[-1]
    rs = runs[(1, n)]
    ps = p_values(rs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    summary["fig1"] = {"N": n, "p": ps.tolist()}
    for ax, variant in zip(axes[:2], ("mid", "full"), strict=True):
        for p_idx, (p, c) in enumerate(
            zip(ps, colors(len(ps), "coolwarm"), strict=True)
        ):
            mus = [np.sort(r["per_p"][p_idx][variant]["mu"])[::-1] for r in rs]
            k = min(len(m) for m in mus)
            mean = np.mean([m[:k] for m in mus], axis=0)
            ax.loglog(
                np.arange(1, k + 1), np.maximum(mean, 1e-12), color=c, label=f"p={p}"
            )
            summary["fig1"].setdefault(variant, []).append(mean[:200].tolist())
        for K in (1, 16, 256, 4096):
            ax.axhline(1 / (K + 1), color="k", ls=":", lw=0.8)
            ax.text(1.1, 1 / (K + 1) * 1.15, f"1/(K+1), K={K}", fontsize=8)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("index k")
        ax.set_ylabel(r"$\mu_k$ (squared canonical correlation)")
        ax.set_title(f"Level I, record {variant}, N={n}")
    axes[1].legend(fontsize=7, ncol=2)
    sizes2 = sorted(n2 for (lvl, n2) in runs if lvl == 2)
    if sizes2:
        n2 = sizes2[min(1, len(sizes2) - 1)]
        rs2 = runs[(2, n2)]
        ps2 = p_values(rs2)
        ax = axes[2]
        for p_idx, (p, c) in enumerate(
            zip(ps2, colors(len(ps2), "coolwarm"), strict=True)
        ):
            rho = [np.sort(r["per_p"][p_idx]["rho2"])[::-1] for r in rs2]
            k = min(len(m) for m in rho)
            mean = np.mean([m[:k] for m in rho], axis=0)
            ax.loglog(np.arange(1, k + 1), np.maximum(mean, 1e-12), color=c)
        for K in (1, 16, 256):
            ax.axhline(1 / (K + 1), color="k", ls=":", lw=0.8)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("index k")
        ax.set_ylabel(r"$\rho_k^2$")
        ax.set_title(f"Level II, local features, N={n2}")
    fig.tight_layout()
    fig.savefig(out / "spectrum.png", dpi=150)
    plt.close(fig)


def fig_nlearn(runs: dict, out: Path, summary: dict) -> None:
    """Figure 2: n_learn(p, K) per level and N, span rank, Delta."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    summary["fig2"] = {}
    for col, variant in enumerate(("mid", "full")):
        for (lvl, n), rs in sorted(runs.items()):
            if lvl != 1:
                continue
            ps = p_values(rs)
            ks = rs[0]["K"]
            for row_ax, ki in (
                (axes[0, col], ks.index("16")),
                (axes[1, col], ks.index("4096")),
            ):
                vals = [[row[variant]["n_learn"][ki] for row in r["per_p"]] for r in rs]
                m, e = stat(vals)
                dim = 3**n - 1
                row_ax.errorbar(
                    ps, m / dim, e / dim, marker="o", ms=3, capsize=2, label=f"N={n}"
                )
                summary["fig2"][f"L1_{variant}_N{n}_K{ks[ki]}"] = (m / dim).tolist()
            vals = [[row[variant]["span_rank"] - 1 for row in r["per_p"]] for r in rs]
            m, _ = stat(vals)
            axes[1, col].plot(ps, m / (3**n - 1), ls="--", color="0.5")
        for row, K in ((0, 16), (1, 4096)):
            ax = axes[row, col]
            mark_pc(ax)
            ax.set_title(f"Level I, {variant}: n_learn / (3^N - 1), K = {K}")
            ax.set_xlabel("p")
            ax.legend(fontsize=8)
        axes[1, col].text(
            0.02,
            0.92,
            "dashed: span rank",
            transform=axes[1, col].transAxes,
            fontsize=8,
        )
    ax = axes[0, 2]
    for (lvl, n), rs in sorted(runs.items()):
        if lvl != 1:
            continue
        ps = p_values(rs)
        vals = [[row["mid"]["delta_dense_mean"] for row in r["per_p"]] for r in rs]
        m, e = stat(vals)
        ax.errorbar(ps, m, e, marker="o", ms=3, capsize=2, label=f"N={n}")
        summary["fig2"][f"delta_N{n}"] = m.tolist()
    mark_pc(ax)
    ax.set_title(r"Level I, mid record: mean $\Delta$ (dense, bits)")
    ax.set_xlabel("p")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for (lvl, n), rs in sorted(runs.items()):
        if lvl != 2:
            continue
        ps = p_values(rs)
        kp = rs[0]["K_population"]
        ki = kp.index(64)
        vals = [[row["n_learn"][ki] for row in r["per_p"]] for r in rs]
        m, e = stat(vals)
        ax.errorbar(ps, m, e, marker="o", ms=3, capsize=2, label=f"N={n}")
        summary["fig2"][f"L2_N{n}_K64"] = m.tolist()
    mark_pc(ax)
    ax.set_title("Level II, local features: n_learn, K = 64")
    ax.set_xlabel("p")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "n_learn.png", dpi=150)
    plt.close(fig)


def fig_level1_bias(runs: dict, out: Path, summary: dict) -> None:
    """Figure 3: level I, E_perp and the noise-limited error 1 - R2(K) - E_perp."""
    sizes = sorted(n for (lvl, n) in runs if lvl == 1)
    if not sizes:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    summary["fig3"] = {}
    for col, name in enumerate(("k1", "x_mid", "k3")):
        for row, variant in enumerate(("mid", "full")):
            ax = axes[row, col]
            for n, ls in zip(sizes, ("-", "--", ":"), strict=False):
                rs = runs[(1, n)]
                ps = p_values(rs)
                ks = rs[0]["K"]
                ep, ee = stat(
                    [
                        [
                            r_["e_perp"]
                            for r_ in (
                                row_[variant]["targets"][name] for row_ in r["per_p"]
                            )
                        ]
                        for r in rs
                    ]
                )
                ax.errorbar(
                    ps,
                    ep,
                    ee,
                    color="k",
                    ls=ls,
                    marker="s",
                    ms=3,
                    capsize=2,
                    label=f"E_perp N={n}",
                )
                summary["fig3"][f"{variant}_{name}_N{n}_eperp"] = ep.tolist()
                for K, c in zip(
                    ("1", "16", "256", "4096"), colors(4, "plasma"), strict=True
                ):
                    ki = ks.index(K)
                    tot, _ = stat(
                        [
                            [
                                1 - row_[variant]["targets"][name]["r2_K"][ki]
                                for row_ in r["per_p"]
                            ]
                            for r in rs
                        ]
                    )
                    ax.plot(
                        ps,
                        tot,
                        color=c,
                        ls=ls,
                        marker="o",
                        ms=3,
                        label=f"1-R2, K={K}" if n == sizes[-1] else None,
                    )
                    summary["fig3"][f"{variant}_{name}_N{n}_K{K}"] = tot.tolist()
            mark_pc(ax)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("p")
            ax.set_title(f"Level I, {variant}, target {LABELS[name]}")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "level1_error.png", dpi=150)
    plt.close(fig)


def fig_level2_bias(runs: dict, out: Path, summary: dict) -> None:
    """Figure 4: level II, bias^2 and variance at lambda* against p, per K."""
    sizes = sorted(n for (lvl, n) in runs if lvl == 2)
    if not sizes:
        return
    names = ("k1", "z_mid", "x_mid")
    fig, axes = plt.subplots(
        len(names),
        len(sizes),
        figsize=(4.3 * len(sizes), 3.6 * len(names)),
        squeeze=False,
    )
    summary["fig4"] = {}
    for col, n in enumerate(sizes):
        rs = runs[(2, n)]
        ps = p_values(rs)
        ks = [str(k) for k in rs[0]["K"]] + ["inf"]
        for row, name in enumerate(names):
            ax = axes[row, col]
            for K, c in zip(ks, colors(len(ks), "plasma"), strict=True):
                b, _ = stat(
                    [
                        [
                            row_["targets"][name]["ridge"][K]["bias2"]
                            for row_ in r["per_p"]
                        ]
                        for r in rs
                    ]
                )
                v, _ = stat(
                    [
                        [
                            row_["targets"][name]["ridge"][K]["variance"]
                            for row_ in r["per_p"]
                        ]
                        for r in rs
                    ]
                )
                ax.plot(ps, b, color=c, marker="o", ms=3, label=f"bias², K={K}")
                ax.plot(ps, v, color=c, ls=":", marker="x", ms=3)
                summary["fig4"][f"N{n}_{name}_K{K}"] = {
                    "bias2": b.tolist(),
                    "variance": v.tolist(),
                }
            ep, _ = stat(
                [
                    [row_["targets"][name]["e_perp_local"] for row_ in r["per_p"]]
                    for r in rs
                ]
            )
            ax.plot(ps, ep, color="k", ls="--", lw=1.5, label="E_perp (local)")
            mark_pc(ax)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("p")
            ax.set_title(f"N={n}, {LABELS[name]}")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].text(
        0.02,
        0.02,
        "solid: bias², dotted: variance",
        transform=axes[0, 0].transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / "level2_bias_variance.png", dpi=150)
    plt.close(fig)


def fig_lambda(runs: dict, out: Path, summary: dict) -> None:
    """Figure 5: lambda* against p, per K and N (level II, target degree 1)."""
    sizes = sorted(n for (lvl, n) in runs if lvl == 2)
    if not sizes:
        return
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(4.3 * len(sizes), 3.8), squeeze=False
    )
    summary["fig5"] = {}
    for col, n in enumerate(sizes):
        rs = runs[(2, n)]
        ps = p_values(rs)
        ks = [str(k) for k in rs[0]["K"]] + ["inf"]
        ax = axes[0, col]
        for K, c in zip(ks, colors(len(ks), "plasma"), strict=True):
            lam = np.exp(
                np.mean(
                    [
                        [
                            np.log(row_["targets"]["k1"]["ridge"][K]["lambda_star"])
                            for row_ in r["per_p"]
                        ]
                        for r in rs
                    ],
                    axis=0,
                )
            )
            ax.semilogy(ps, lam, color=c, marker="o", ms=3, label=f"K={K}")
            summary["fig5"][f"N{n}_K{K}"] = lam.tolist()
        mark_pc(ax)
        ax.set_xlabel("p")
        ax.set_ylabel(r"$\lambda^*$ (relative, geometric mean)")
        ax.set_title(f"N={n}, target degree 1")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "lambda_star.png", dpi=150)
    plt.close(fig)


def fig_optimum(runs: dict, out: Path, summary: dict) -> None:
    """Figure 6: error map 1 - R2 in (p, K) with the minimum p*(K)."""
    sizes = sorted(n for (lvl, n) in runs if lvl == 1)
    if not sizes:
        return
    n = sizes[-1]
    rs = runs[(1, n)]
    ps = p_values(rs)
    ks = [k for k in rs[0]["K"] if k != "inf"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    summary["fig6"] = {"N": n, "p": ps.tolist(), "K": ks}
    for ax, name in zip(axes[:2], ("k1", "x_mid"), strict=True):
        err = np.mean(
            [
                [
                    [
                        1 - row["full"]["targets"][name]["r2_K"][rs[0]["K"].index(K)]
                        for row in r["per_p"]
                    ]
                    for K in ks
                ]
                for r in rs
            ],
            axis=0,
        )
        im = ax.pcolormesh(
            ps, np.arange(len(ks)), err, shading="nearest", cmap="magma", vmin=0, vmax=1
        )
        best = ps[np.argmin(err, axis=1)]
        ax.plot(best, np.arange(len(ks)), "c-o", ms=4, label="p*(K)")
        ax.axvline(P_C, color="w", ls="--")
        ax.set_yticks(np.arange(len(ks)), ks)
        ax.set_xlabel("p")
        ax.set_ylabel("K (shots per input)")
        ax.set_title(f"Level I full, N={n}: 1 - R2, {LABELS[name]}")
        fig.colorbar(im, ax=ax)
        ax.legend(fontsize=8)
        summary["fig6"][name] = {"error": err.tolist(), "p_star": best.tolist()}
    ax = axes[2]
    for (lvl, n2), rs2 in sorted(runs.items()):
        if lvl != 2:
            continue
        ps2 = p_values(rs2)
        ks2 = [str(k) for k in rs2[0]["K"]]
        best = []
        for K in ks2:
            err, _ = stat(
                [
                    [row["targets"]["k1"]["ridge"][K]["test_mse"] for row in r["per_p"]]
                    for r in rs2
                ]
            )
            best.append(ps2[int(np.argmin(err))])
        ax.plot([int(k) for k in ks2], best, marker="o", label=f"Level II N={n2}")
        summary["fig6"][f"L2_N{n2}_p_star"] = best
    for n1 in sizes:
        rs1 = runs[(1, n1)]
        err = np.mean(
            [
                [
                    [
                        1 - row["full"]["targets"]["k1"]["r2_K"][rs1[0]["K"].index(K)]
                        for row in r["per_p"]
                    ]
                    for K in ks
                ]
                for r in rs1
            ],
            axis=0,
        )
        ax.plot(
            [int(k) for k in ks],
            p_values(rs1)[np.argmin(err, axis=1)],
            ls="--",
            marker="s",
            label=f"Level I full N={n1}",
        )
    ax.axhspan(P_C - P_C_ERR, P_C + P_C_ERR, color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("p*(K), minimum of the error, target degree 1")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "optimum.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    runs = load(args.folder)
    out = args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"runs": {f"level{k[0]}_N{k[1]}": len(v) for k, v in runs.items()}}
    for fig in (
        fig_spectrum,
        fig_nlearn,
        fig_level1_bias,
        fig_level2_bias,
        fig_lambda,
        fig_optimum,
    ):
        fig(runs, out, summary)
    (out / "summary.json").write_text(json.dumps(summary))
    print(f"wrote figures in {out}")


if __name__ == "__main__":
    main()
