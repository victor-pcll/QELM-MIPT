#!/usr/bin/env python3
"""Figures of the readout locality study (issue #386).

Reads ``locality_N<N>[_T<T>].npz`` (``run.py``) and ``haar_N<N>.npz``
(``haar.py``) and writes:

- ``readout_profile.png``: ``E_readout(k) / (1 - E_circuit)`` against ``k / N``
  at several ``p``, one panel per ``N`` (flat prior; iid dashed);
- ``readout_vs_p.png``: the same ratio against ``p`` at ``k = 1, 2, 3`` and
  ``k = N / 2, N``, one curve per ``N``;
- ``mean_weight.png``: mean minimal parity weight ``<w> / N`` against ``p``;
  the local exponent ``d ln <w> / d ln N`` from consecutive sizes; and
  ``<w | |g|>`` against the support ``|g|`` at small and large ``p``;
- ``depth.png``: ``<w>`` against ``p`` for several depths ``T`` (if the
  ``_T`` files exist);
- ``haar_vs_clifford.png``: the ratio against ``p`` at ``k = 1 .. 4``, Haar
  and Clifford on the same masks (``haar_N<N>.npz``).

``summary.json`` holds every curve drawn.

Usage::

    uv run python experiments/readout_locality/plot.py \
        --data outputs/readout_locality --out experiments/readout_locality/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"
SIZE_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e34948", "#6250d6"]
KIND_COLORS = {"haar": "#2a78d6", "clifford": "#eb6834"}
P_C = 0.161  # Clifford, from Delta (#346)


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def mark_pc(ax: plt.Axes) -> None:
    ax.axvline(P_C, color=MUTED, lw=0.7, ls=":", zorder=0)


def mean_sem(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    m = np.nanmean(x, axis=axis)
    n = np.sum(np.isfinite(x), axis=axis)
    return m, np.nanstd(x, axis=axis) / np.sqrt(np.maximum(n - 1, 1))


class Locality:
    """Per-circuit observables of one ``locality_N<N>.npz``."""

    def __init__(self, path: Path) -> None:
        d = np.load(path)
        self.hist = d["hist"].astype(float)  # (c, p, w, s)
        self.p = d["p"]
        self.bits = d["bits"]
        cfg = json.loads(str(d["config"]))
        self.n, self.depth = int(cfg["n"]), int(cfg["depth"])
        self.w = np.arange(self.hist.shape[2])
        self.s = np.arange(self.hist.shape[3])

    def ratio(self, k: float, prior: str = "flat") -> np.ndarray:
        """``(c, p)`` relative readout error at order ``k``."""
        q = np.ones_like(self.s, dtype=float) if prior == "flat" else 0.5**self.s
        h = self.hist * q[None, None, None, :]
        tot = h.sum(axis=(2, 3))
        miss = h[:, :, self.w > k].sum(axis=(2, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, miss / tot, np.nan)

    def mean_w(self) -> np.ndarray:
        """``(c, p)`` mean of ``w`` over the visible strings of ``G``."""
        hw = self.hist.sum(axis=3)
        tot = hw.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, (hw * self.w).sum(axis=2) / tot, np.nan)

    def w_given_support(self, j: int) -> np.ndarray:
        """Pooled ``<w | |g| = s>`` at ``p[j]``, ``s = 0 .. N``."""
        hs = self.hist[:, j].sum(axis=0)
        tot = hs.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, (hs * self.w[:, None]).sum(axis=0) / tot, np.nan)


def load_all(data: Path) -> tuple[dict[int, Locality], dict[tuple[int, int], Locality]]:
    main, depth = {}, {}
    for f in sorted(data.glob("locality_N*.npz")):
        m = re.fullmatch(r"locality_N(\d+)(?:_T(\d+))?\.npz", f.name)
        if m is None:
            continue
        loc = Locality(f)
        depth[(loc.n, loc.depth)] = loc
        if m.group(2) is None:
            main[loc.n] = loc
    return dict(sorted(main.items())), depth


def fig_profile(main: dict[int, Locality], out: Path, summary: dict) -> None:
    sizes = list(main)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(2.6 * len(sizes), 2.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    p_show = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    cmap = plt.get_cmap("viridis")
    summary["profile"] = {}
    for ax, n in zip(axes, sizes, strict=True):
        loc = main[n]
        ks = np.arange(0, 3 * n + 1)
        summary["profile"][str(n)] = {}
        for i, p in enumerate(p_show):
            j = int(np.argmin(np.abs(loc.p - p)))
            col = cmap(i / (len(p_show) - 1))
            for prior, ls in (("flat", "-"), ("iid", "--")):
                y = np.array([np.nanmean(loc.ratio(k, prior)[:, j]) for k in ks])
                ax.plot(
                    ks / n,
                    y,
                    ls,
                    color=col,
                    lw=1.2 if prior == "flat" else 0.8,
                    label=f"p = {loc.p[j]:g}" if prior == "flat" else None,
                )
                summary["profile"][str(n)][f"{prior}_p{loc.p[j]:g}"] = y.tolist()
        ax.set_title(f"N = {n}, T = {loc.depth}", fontsize=8, color=INK)
        ax.set_xlabel("k / N", fontsize=8)
        ax.set_xlim(0, 3)
        style(ax)
    axes[0].set_ylabel(r"$E_{readout}(k)\,/\,(1-E_{circuit})$", fontsize=8)
    axes[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "readout_profile.png", dpi=200)
    plt.close(fig)


def fig_vs_p(main: dict[int, Locality], out: Path, summary: dict) -> None:
    orders = [
        ("1", lambda n: 1),
        ("2", lambda n: 2),
        ("3", lambda n: 3),
        ("N/2", lambda n: n / 2),
        ("N", lambda n: n),
    ]
    fig, axes = plt.subplots(
        2, len(orders), figsize=(2.3 * len(orders), 4.4), sharey="row", sharex=True
    )
    summary["vs_p"] = {}
    for r, prior in enumerate(("flat", "iid")):
        for c, (name, kf) in enumerate(orders):
            ax = axes[r, c]
            for i, (n, loc) in enumerate(main.items()):
                m, e = mean_sem(loc.ratio(kf(n), prior))
                ax.errorbar(
                    loc.p,
                    m,
                    e,
                    color=SIZE_COLORS[i % 6],
                    lw=1,
                    ms=2.5,
                    marker="o",
                    capsize=0,
                    label=f"N = {n}",
                )
                summary["vs_p"][f"{prior}_k{name}_N{n}"] = m.tolist()
            mark_pc(ax)
            style(ax)
            if r == 0:
                ax.set_title(f"k = {name}", fontsize=8, color=INK)
            if r == 1:
                ax.set_xlabel("p", fontsize=8)
        axes[r, 0].set_ylabel(f"ratio ({prior} prior)", fontsize=8)
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "readout_vs_p.png", dpi=200)
    plt.close(fig)


def peak(p: np.ndarray, per_circuit: np.ndarray, n_boot: int = 300) -> dict:
    """Position and height of the maximum of the mean over circuits.

    Parabola through the five grid points around the maximum; the error is the
    standard deviation over a bootstrap of the circuits.
    """

    def fit(m: np.ndarray) -> tuple[float, float]:
        j = int(np.argmax(m))
        sl = slice(max(j - 2, 0), j + 3)
        c = np.polyfit(p[sl], m[sl], 2)
        x = float(-c[1] / (2 * c[0])) if c[0] < 0 else float(p[j])
        return x, float(np.polyval(c, x))

    x, y = fit(np.nanmean(per_circuit, axis=0))
    rng = np.random.default_rng(0)
    n_c = per_circuit.shape[0]
    boot = [
        fit(np.nanmean(per_circuit[rng.integers(0, n_c, n_c)], axis=0))
        for _ in range(n_boot)
    ]
    b = np.array(boot)
    return {
        "p_max": x,
        "p_max_err": float(b[:, 0].std()),
        "height": y,
        "height_err": float(b[:, 1].std()),
    }


def fig_mean_weight(main: dict[int, Locality], out: Path, summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    sizes = list(main)
    summary["mean_w"] = {}
    means = {}
    for i, n in enumerate(sizes):
        loc = main[n]
        m, e = mean_sem(loc.mean_w())
        means[n] = m
        axes[0].errorbar(
            loc.p,
            m / n,
            e / n,
            color=SIZE_COLORS[i % 6],
            marker="o",
            ms=2.5,
            lw=1,
            label=f"N = {n}",
        )
        summary["mean_w"][str(n)] = {
            "mean": m.tolist(),
            "sem": e.tolist(),
            "peak": peak(loc.p, loc.mean_w()),
            "std_over_circuits": np.nanstd(loc.mean_w(), axis=0).tolist(),
        }
    axes[0].set_ylabel(r"$\langle w\rangle / N$", fontsize=8)
    axes[0].set_xlabel("p", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    summary["exponent"] = {}
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        alpha = np.log(means[b] / means[a]) / np.log(b / a)
        axes[1].plot(
            main[a].p,
            alpha,
            marker="o",
            ms=2.5,
            lw=1,
            color=SIZE_COLORS[(i + 1) % 6],
            label=f"N = {a}, {b}",
        )
        summary["exponent"][f"{a}_{b}"] = alpha.tolist()
    for y in (1, 2):
        axes[1].axhline(y, color=MUTED, lw=0.6, ls="--", zorder=0)
    axes[1].set_ylabel(r"$d\ln\langle w\rangle / d\ln N$", fontsize=8)
    axes[1].set_xlabel("p", fontsize=8)
    axes[1].legend(fontsize=6, frameon=False)
    n = sizes[-1]
    loc = main[n]
    summary["w_given_support"] = {"N": n}
    cmap = plt.get_cmap("viridis")
    p_show = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for i, p in enumerate(p_show):
        j = int(np.argmin(np.abs(loc.p - p)))
        y = loc.w_given_support(j)
        axes[2].plot(
            loc.s[1:],
            y[1:],
            marker="o",
            ms=2.5,
            lw=1,
            color=cmap(i / (len(p_show) - 1)),
            label=f"p = {loc.p[j]:g}",
        )
        summary["w_given_support"][f"p{loc.p[j]:g}"] = y.tolist()
    axes[2].set_xlabel("support |g|", fontsize=8)
    axes[2].set_ylabel(rf"$\langle w \mid |g|\rangle$, N = {n}", fontsize=8)
    axes[2].legend(fontsize=6, frameon=False)
    for ax in axes[:2]:
        mark_pc(ax)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "mean_weight.png", dpi=200)
    plt.close(fig)


def fig_depth(depth: dict[tuple[int, int], Locality], out: Path, summary: dict) -> None:
    sizes = sorted({n for n, _ in depth})
    sizes = [n for n in sizes if sum(1 for m, _ in depth if m == n) > 1]
    if not sizes:
        return
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(3.2 * len(sizes), 2.8), squeeze=False
    )
    summary["depth"] = {}
    for ax, n in zip(axes[0], sizes, strict=True):
        ts = sorted(t for m, t in depth if m == n)
        for i, t in enumerate(ts):
            loc = depth[(n, t)]
            m, e = mean_sem(loc.mean_w())
            ax.errorbar(
                loc.p,
                m,
                e,
                marker="o",
                ms=2.5,
                lw=1,
                color=SIZE_COLORS[i % 6],
                label=f"T = {t} = {t / n:g} N",
            )
            summary["depth"][f"N{n}_T{t}"] = m.tolist()
        ax.set_title(f"N = {n}", fontsize=8, color=INK)
        ax.set_xlabel("p", fontsize=8)
        ax.set_ylabel(r"$\langle w\rangle$", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        mark_pc(ax)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "depth.png", dpi=200)
    plt.close(fig)


def fig_haar(data: Path, out: Path, summary: dict) -> None:
    files = sorted(data.glob("haar_N*.npz"))
    if not files:
        return
    d = np.load(files[-1])
    cfg = json.loads(str(d["config"]))
    rank, wtr, p = d["rank"].astype(float), d["wtr"], d["p"]
    k_max = int(cfg["k_max"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    summary["haar"] = {"n": cfg["n"], "depth": cfg["depth"]}
    styles = ["-", "--", "-.", ":"]
    for ax, (prior, arr) in zip(axes, (("flat", rank), ("iid", wtr)), strict=True):
        for a, kind in enumerate(("haar", "clifford")):
            full = arr[a, :, :, k_max]
            for k in range(1, k_max + 1):
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(full > 0, (full - arr[a, :, :, k - 1]) / full, np.nan)
                m, _ = mean_sem(r)
                ax.plot(
                    p,
                    m,
                    styles[(k - 1) % 4],
                    marker="o",
                    ms=2.5,
                    lw=1,
                    color=KIND_COLORS[kind],
                    label=f"{kind}, k = {k}" if prior == "flat" else None,
                )
                summary["haar"][f"{prior}_{kind}_k{k}"] = m.tolist()
        ax.set_title(
            f"{prior} prior, N = {cfg['n']}, T = {cfg['depth']}", fontsize=8, color=INK
        )
        ax.set_xlabel("p", fontsize=8)
        mark_pc(ax)
        style(ax)
    axes[0].set_ylabel(r"$E_{readout}(k)\,/\,(1-E_{circuit})$", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "haar_vs_clifford.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("outputs/readout_locality"))
    ap.add_argument(
        "--out", type=Path, default=Path("experiments/readout_locality/figures")
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    main_runs, depth = load_all(args.data)
    summary: dict = {"sizes": list(main_runs)}
    if main_runs:
        summary["p"] = next(iter(main_runs.values())).p.tolist()
        fig_profile(main_runs, args.out, summary)
        fig_vs_p(main_runs, args.out, summary)
        fig_mean_weight(main_runs, args.out, summary)
    fig_depth(depth, args.out, summary)
    fig_haar(args.data, args.out, summary)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
