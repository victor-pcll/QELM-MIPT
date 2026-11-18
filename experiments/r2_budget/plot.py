#!/usr/bin/env python3
"""Figures of the R^2 budget against p (issue #360).

Reads the ``.npz`` / ``.json`` written by ``run.py`` and writes, per prior:

- ``budget_hist_k2_<prior>.png``: one column per ``p``, one row per term
  (``E_circuit``, ``E_readout(2)``, ``R^2_pop(2)``); Haar and Clifford on the
  same axes (same masks, same targets, ``N = 4``).
- ``budget_hist_kinf_<prior>.png``: same with the decoder on every parity
  (``k = inf``): ``E_readout = 0`` identically, so ``R^2 = 1 - E_circuit``.
- ``budget_mean_<prior>.png``: mean budget per ``p``, stacked, Haar and
  Clifford side by side, for ``k = 1``, ``k = 2`` and ``k = inf``.
- ``clifford_sizes_<prior>.png``: the Clifford route at larger ``N`` (#337
  model), histograms of the same terms at ``k = 2`` and ``k = inf``.

Usage::

    uv run python experiments/r2_budget/plot.py --data outputs/r2_budget
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"haar": "#2a78d6", "clifford": "#eb6834"}
LABELS = {"haar": "Haar", "clifford": "Clifford"}
TERM_COLORS = {"circuit": "#e34948", "readout": "#eda100", "r2": "#008300"}
SIZE_COLORS = ["#eb6834", "#6250d6", "#1baf7a"]
P_SHOW = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def terms(raw: dict, kind: str, j: int, ks: list[int], k: int | None) -> dict:
    """Samples of the three terms at decoder order ``k`` (``None`` = inf)."""
    circ = raw[f"{kind}_circuit_{j}"].astype(float)
    if k is None:
        return {"circuit": circ, "readout": np.zeros_like(circ), "r2": 1.0 - circ}
    i = ks.index(k)
    return {
        "circuit": circ,
        "readout": raw[f"{kind}_readout_{j}"][i].astype(float),
        "r2": raw[f"{kind}_r2_{j}"][i].astype(float),
    }


def term_name(term: str, k: int | None) -> str:
    kk = r"\infty" if k is None else str(k)
    return {
        "circuit": r"$E_{\rm circuit}$",
        "readout": rf"$E_{{\rm readout}}(k={kk})$",
        "r2": rf"$R^2_{{\rm pop}}(k={kk})$",
    }[term]


def pick(ps: list[float], show: list[float]) -> list[int]:
    idx = [i for i, p in enumerate(ps) if any(abs(p - s) < 1e-9 for s in show)]
    return idx or list(range(len(ps)))


def hist_figure(
    raw: dict,
    meta: dict,
    kinds: list[str],
    k: int | None,
    rows: list[str],
    path: Path,
    title: str,
    show: list[float],
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
) -> None:
    labels = labels or LABELS
    colors = colors or COLORS
    ps = meta["p"]
    cols = pick(ps, show)
    bins = np.linspace(0, 1, 26)
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(2.0 * len(cols) + 0.6, 1.8 * len(rows) + 0.9),
        squeeze=False,
        sharex=True,
    )
    for c, j in enumerate(cols):
        for r, term in enumerate(rows):
            ax = axes[r, c]
            style(ax)
            for kind in kinds:
                v = terms(raw, kind, j, meta["k"], k)[term]
                weights = np.full(v.size, 1.0 / v.size)
                ax.hist(
                    np.clip(v, 0, 1),
                    bins=bins,
                    weights=weights,
                    color=colors[kind],
                    alpha=0.45,
                    edgecolor="white",
                    linewidth=0.5,
                    label=labels[kind],
                )
                ax.axvline(v.mean(), color=colors[kind], lw=1.5)
            if r == 0:
                ax.set_title(f"p = {ps[j]:.2f}", fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(term_name(term, k) + "\nfraction", fontsize=8, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("part of Var y", fontsize=8, color=MUTED)
            ax.set_xlim(0, 1)
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, lab, loc="upper right", ncol=len(kinds), fontsize=8, frameon=False
    )
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mean_figure(raw: dict, meta: dict, path: Path, title: str) -> None:
    ps = meta["p"]
    orders: list[int | None] = [k for k in (1, 2) if k in meta["k"]] + [None]
    fig, axes = plt.subplots(
        1, len(orders), figsize=(4.2 * len(orders), 3.9), sharey=True, squeeze=False
    )
    width = 0.018
    for a, k in enumerate(orders):
        ax = axes[0, a]
        style(ax)
        for s, kind in enumerate(("haar", "clifford")):
            x = np.array(ps) + (s - 0.5) * (width + 0.002)
            bottom = np.zeros(len(ps))
            for term in ("circuit", "readout", "r2"):
                h = np.array(
                    [
                        terms(raw, kind, j, meta["k"], k)[term].mean()
                        for j in range(len(ps))
                    ]
                )
                ax.bar(
                    x,
                    h,
                    width,
                    bottom=bottom,
                    color=TERM_COLORS[term],
                    edgecolor="white",
                    linewidth=0.8,
                    hatch=None if kind == "haar" else "////",
                    label=term_name(term, k) if s == 0 else None,
                )
                bottom += h
        kk = "∞" if k is None else str(k)
        ax.set_title(
            f"decoder k = {kk}   (left: Haar, hatched: Clifford)",
            fontsize=8.5,
            color=INK,
        )
        ax.set_xlabel("p", fontsize=9, color=INK)
        ax.set_ylim(0, 1.0)
        ax.legend(
            fontsize=7,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
        )
    axes[0, 0].set_ylabel("mean share of Var y", fontsize=9, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sizes_figure(
    raw: dict, meta: dict, sizes: list[int], path: Path, title: str
) -> None:
    """Clifford at larger N: the shares are tiny, so a log axis against p."""
    ps = np.array(meta["p"])
    panels: list[tuple[str, int | None]] = [("r2", None), ("r2", 2), ("readout", 2)]
    panels = [(t, k) for t, k in panels if k is None or k in meta["k"]]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.0 * len(panels), 3.4), sharey=True, squeeze=False
    )
    for a, (term, k) in enumerate(panels):
        ax = axes[0, a]
        style(ax)
        ax.set_yscale("log")
        for i, n in enumerate(sizes):
            q = np.array(
                [
                    np.quantile(
                        terms(raw, f"N{n}", j, meta["k"], k)[term], [0.05, 0.5, 0.95]
                    )
                    for j in range(len(ps))
                ]
            )
            floor = 1e-6
            q = np.maximum(q, floor)
            color = SIZE_COLORS[i % 3]
            ax.fill_between(ps, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0)
            ax.plot(ps, q[:, 1], color=color, lw=2, marker="o", ms=4, label=f"N = {n}")
        name = term_name(term, k)
        if term == "r2" and k is None:
            name += r" $= 1 - E_{\rm circuit}$"
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlabel("p", fontsize=9, color=INK)
        ax.set_ylim(1e-6, 1)
        ax.legend(fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("share of Var y (clipped at 1e-6)", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load(path: Path) -> tuple[dict, dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as z:
        raw = {key: z[key] for key in z.files}
    return raw, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/r2_budget"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--clifford-n", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--p-show", type=float, nargs="+", default=P_SHOW)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["", "_sphere_nofinal_T8"],
        help="suffixes of the dense runs to plot (run.py dense_tag)",
    )
    args = parser.parse_args(argv)
    out = args.out or args.data / "figures"
    out.mkdir(parents=True, exist_ok=True)
    prior_text = {
        "iid": r"$\beta_a \sim N(0,1)$",
        "flat": r"isotropic $\sqrt{w_a}\,\beta_a \sim N(0,1)$",
    }
    for prior, tag in [(pr, t) for pr in ("iid", "flat") for t in args.tags]:
        dense = args.data / f"dense_N{args.n}_{prior}{tag}.json"
        if dense.exists():
            raw, meta = load(dense)
            setting = ""
            if meta.get("ensemble", "xz") == "sphere":
                setting += "dense (sphere) encoding, "
            if not meta.get("final", True):
                setting += "no final readout, "
            head = (
                f"R² budget, N = {meta['n']}, T = {meta['depth']}, {setting}"
                f"{prior_text[prior]}, K = ∞, λ → 0+  "
                f"({meta['circuits']} masks × {meta['targets']} targets per p)"
            )
            name = prior + tag
            if 2 in meta["k"]:
                hist_figure(
                    raw,
                    meta,
                    ["haar", "clifford"],
                    2,
                    ["circuit", "readout", "r2"],
                    out / f"budget_hist_k2_{name}.png",
                    head + ", decoder k = 2",
                    args.p_show,
                )
            hist_figure(
                raw,
                meta,
                ["haar", "clifford"],
                None,
                ["circuit", "r2"],
                out / f"budget_hist_kinf_{name}.png",
                head + ", decoder k = ∞ (E_readout = 0)",
                args.p_show,
            )
            mean_figure(raw, meta, out / f"budget_mean_{name}.png", head)
        # Clifford route at larger N: N on the same axes.
        sizes = [
            n
            for n in args.clifford_n
            if tag == "" and (args.data / f"clifford_N{n}_{prior}.json").exists()
        ]
        if sizes:
            merged: dict = {}
            metas = []
            for n in sizes:
                raw_n, meta_n = load(args.data / f"clifford_N{n}_{prior}.json")
                metas.append(meta_n)
                merged.update(
                    {k.replace("clifford", f"N{n}", 1): v for k, v in raw_n.items()}
                )
            meta = {"p": metas[0]["p"], "k": metas[0]["k"]}
            sizes_figure(
                merged,
                meta,
                sizes,
                out / f"clifford_sizes_{prior}.png",
                f"Clifford (#337 model, T = 2N), {prior_text[prior]}, "
                "K = ∞, λ → 0+: median and 5-95 % band over circuits × targets",
            )
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
