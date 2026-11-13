"""Min cut of the space-time graph against the effect-space rank (#321).

Two parts, both written to ``--out`` with a JSON of the plotted values:

1. ``mincut_vs_rank``: for the seeds and nested masks of ``effect_rank.py``
   (mask ``u < p``, ``u`` drawn from ``default_rng((seed, N, 1000))``), the
   mean of ``2^C`` over the seeds against the mean ranks stored in
   ``figures/effect_rank_dense.json``, for each class and ``N``. Since
   ``rank <= 2^C`` circuit by circuit, equal means imply equality for every
   circuit.
2. ``mincut_large_n``: ``C / N`` against ``p`` and against ``x = p (T - 1)``
   for ``N`` up to ``--n-max``, ``T = N`` and ``T = 2N``, and the position
   and height of the peak against ``T``.

Usage::

    uv run python experiments/effect_rank/mincut_rank.py \
        --out experiments/effect_rank/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from QELM.evaluation.effect_rank import mincut_bits  # noqa: E402

HERE = Path(__file__).parent
KINDS = ("haar", "u1", "z2", "clifford")
TITLES = {
    "haar": "Haar U(4)",
    "u1": "U(1)",
    "z2": "Z$_2$ parity",
    "clifford": "Clifford",
}


def compare(ranks_file: Path, seeds: int) -> dict:
    """Mean ``2^C`` and mean exact rank per class, ``N`` and ``p``."""
    data = json.loads(ranks_file.read_text())
    out: dict = {}
    for key, block in data.items():
        n = int(key)
        ps = block["p"]
        cuts = np.array(
            [
                [
                    2
                    ** mincut_bits(
                        np.random.default_rng((s, n, 1000)).random((n, n)) < p
                    )
                    for p in ps
                ]
                for s in range(seeds)
            ]
        )
        out[key] = {"p": ps, "mean_2C": cuts.mean(0).tolist()}
        for kind in KINDS:
            if kind in block:
                out[key][kind] = block[kind]["mean"]
    return out


def scan(sizes: list[int], ratio: int, masks: int) -> dict:
    """``C / N`` for many masks, on a grid of ``x = p (T - 1)`` and of ``p``."""
    xs = np.concatenate([np.linspace(0, 3, 25), np.linspace(3.5, 12, 18)])
    fixed = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out: dict = {}
    for n in sizes:
        depth = ratio * n
        ps = sorted({min(1.0, float(x) / (depth - 1)) for x in xs} | set(fixed))
        reps = max(6, masks // n)
        c = (
            np.array(
                [
                    [
                        mincut_bits(
                            np.random.default_rng((r, n, ratio)).random((depth, n)) < p
                        )
                        for p in ps
                    ]
                    for r in range(reps)
                ]
            )
            / n
        )
        out[str(n)] = {
            "T": depth,
            "p": ps,
            "masks": reps,
            "mean": c.mean(0).tolist(),
            "sem": (c.std(0, ddof=1) / np.sqrt(reps)).tolist(),
        }
        print(f"T = {ratio}N, N = {n}", flush=True)
    return out


def plot_compare(res: dict, path: Path) -> None:
    sizes = sorted(res, key=int)
    colours = plt.get_cmap("Blues")(np.linspace(0.4, 1, len(sizes)))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS, strict=True):
        for key, col in zip(sizes, colours, strict=True):
            r = res[key]
            if kind not in r:
                continue
            ratio = np.array(r[kind]) / np.array(r["mean_2C"])
            ax.plot(r["p"], ratio, "o-", ms=3, lw=1.5, color=col, label=f"N = {key}")
        ax.axhline(1, color="0.4", ls="--", lw=1)
        ax.set_title(TITLES[kind], fontsize=10)
        ax.set_xlabel("p")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean exact dense rank / mean 2$^C$")
    axes[0].set_yscale("log")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sizes), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scan(res: dict, path: Path) -> None:
    sizes = sorted(res["T=N"], key=int)
    colours = dict(
        zip(sizes, plt.get_cmap("Blues")(np.linspace(0.4, 1, len(sizes))), strict=True)
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for key in sizes:
        r = res["T=N"][key]
        axes[0].errorbar(
            r["p"],
            r["mean"],
            yerr=r["sem"],
            color=colours[key],
            lw=1.5,
            marker="o",
            ms=2.5,
            label=f"N = {key}",
        )
        x = np.array(r["p"]) * (r["T"] - 1)
        keep = x <= 4
        axes[1].plot(
            x[keep],
            np.array(r["mean"])[keep],
            "o-",
            ms=2.5,
            lw=1.5,
            color=colours[key],
            label=f"N = {key}",
        )
    pp = np.linspace(0, 1, 50)
    axes[0].plot(pp, 2 - pp, color="0.4", ls="--", lw=1, label="2 − p")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("p")
    axes[0].set_ylabel("C / N = log$_2$(Haar dense rank) / N")
    axes[0].set_title("T = N, against p", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="lower center")
    xx = np.linspace(0, 1, 10)
    axes[1].plot(xx, 1 + xx, color="0.4", ls="-.", lw=1, label="1 + x (records)")
    axes[1].set_xlabel("x = p (T − 1)")
    axes[1].set_title("T = N, against measurements per site", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    for name, marker, ls in (("T=N", "o", "-"), ("T=2N", "s", "--")):
        block = res[name]
        ts, ps, hs = [], [], []
        for key in sorted(block, key=int):
            r = block[key]
            i = int(np.argmax(r["mean"]))
            ts.append(r["T"])
            ps.append(r["p"][i])
            hs.append(r["mean"][i])
        face = "white" if name == "T=2N" else None
        axes[2].plot(
            ts,
            ps,
            marker=marker,
            ls=ls,
            color=plt.get_cmap("Blues")(0.8),
            mfc=face,
            label=f"p* ({name})",
        )
        axes[2].plot(
            ts,
            2 - np.array(hs),
            marker=marker,
            ls=ls,
            color="0.45",
            mfc=face,
            label=f"2 − peak height ({name})",
        )
    tt = np.array([8, 256])
    axes[2].plot(tt, 1.15 / (tt - 1), color="k", ls=":", lw=1, label="1.15 / (T − 1)")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("T")
    axes[2].set_title("Peak position and height", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ranks", type=Path, default=HERE / "figures/effect_rank_dense.json"
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=128)
    parser.add_argument("--masks", type=int, default=400, help="masks x N per size")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = compare(args.ranks, args.seeds)
    (args.out / "mincut_vs_rank.json").write_text(json.dumps(comparison, indent=1))
    plot_compare(comparison, args.out / "mincut_vs_rank.png")
    sizes = [n for n in (8, 16, 32, 64, 128, 256, 512) if n <= args.n_max]
    large = {"T=N": scan(sizes, 1, args.masks), "T=2N": scan(sizes, 2, args.masks)}
    (args.out / "mincut_large_n.json").write_text(json.dumps(large))
    plot_scan(large, args.out / "mincut_large_n.png")


if __name__ == "__main__":
    main()
