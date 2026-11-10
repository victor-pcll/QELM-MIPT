"""Figures of the effect-space rank against p, one panel per circuit class.

Reads the ``N<n>_seed<s>.json`` files of ``effect_rank.py`` and writes, for
each encoding, ``effect_rank_<encoding>.png`` (raw rank, log scale) and
``effect_rank_<encoding>_normalised.png`` (rank divided by the symmetry
bound of the class) with the summary ``effect_rank_<encoding>.json``.

Usage::

    uv run python experiments/effect_rank/plot_effect_rank.py \
        outputs/effect_rank --out outputs/effect_rank/figures
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
from matplotlib.lines import Line2D  # noqa: E402

from QELM.evaluation.effect_rank import rank_bounds  # noqa: E402

KINDS = ("haar", "u1", "z2", "clifford")
TITLES = {
    "haar": "Haar U(4)",
    "u1": "U(1)",
    "z2": "Z$_2$ parity",
    "clifford": "Clifford",
}
BOUND_KEY = {"haar": "none", "u1": "u1", "z2": "z2", "clifford": "none"}
BOUND_LABEL = {
    "angle": {"none": "$3^N$", "u1": "$(3^N+1)/2$", "z2": "$(3^N+1)/2$"},
    "dense": {"none": "$4^N$", "u1": "$\\binom{2N}{N}$", "z2": "$4^N/2$"},
}

SUPTITLE = {
    "angle": "Bound: $3^N$ (Haar, Clifford), $(3^N+1)/2$ (U(1), Z$_2$)",
    "dense": "Bound: $4^N$ (Haar, Clifford), $\\binom{2N}{N}$ (U(1)), $4^N/2$ (Z$_2$)",
}


def load(folder: Path) -> dict[int, list[dict]]:
    """Runs grouped by size ``N``, the files of one (N, seed) merged.

    A run may cover only some classes (``--kinds``): its file is
    ``N<n>_seed<s>_<kinds>.json`` and its ranks are merged into the run of
    the same size and seed.
    """
    merged: dict[tuple[int, int], dict] = {}
    for path in sorted(folder.glob("N*_seed*.json")):
        run = json.loads(path.read_text())
        key = (int(run["n"]), int(run["seed"]))
        if key in merged:
            merged[key]["ranks"].update(run["ranks"])
        else:
            merged[key] = run
    runs: dict[int, list[dict]] = defaultdict(list)
    for (n, _), run in sorted(merged.items()):
        runs[n].append(run)
    return dict(runs)


def summarise(runs: dict[int, list[dict]], encoding: str) -> dict:
    """Mean and standard error over seeds, per size and class."""
    out: dict = {}
    for n, items in runs.items():
        out[str(n)] = {"p": items[0]["p"], "bounds": rank_bounds(n)[encoding]}
        for kind in KINDS:
            have = [it for it in items if kind in it["ranks"]]
            if not have:
                continue
            r = np.array([it["ranks"][kind][encoding] for it in have], dtype=float)
            sem = r.std(axis=0, ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0 * r[0]
            out[str(n)][kind] = {
                "mean": r.mean(axis=0).tolist(),
                "sem": sem.tolist(),
                "seeds": len(r),
            }
    return out


def colours(sizes: list[int]) -> dict[int, tuple]:
    """One blue per size, light (small N) to dark (large N)."""
    cmap = plt.get_cmap("Blues")
    steps = np.linspace(0.38, 1.0, len(sizes))
    return {n: cmap(s) for n, s in zip(sizes, steps)}


def plot(summary: dict, encoding: str, out: Path, normalised: bool) -> None:
    sizes = [int(n) for n in summary]
    col = colours(sizes)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=normalised)
    for ax, kind in zip(axes, KINDS):
        key = BOUND_KEY[kind]
        for n in sizes:
            s = summary[str(n)]
            if kind not in s:
                continue
            p = np.array(s["p"])
            bound = s["bounds"][key]
            scale = bound if normalised else 1.0
            mean = np.array(s[kind]["mean"]) / scale
            sem = np.array(s[kind]["sem"]) / scale
            ax.errorbar(
                p,
                mean,
                yerr=sem,
                color=col[n],
                lw=2,
                marker="o",
                ms=3.5,
                capsize=2,
                label=f"N = {n} ({s[kind]['seeds']} seeds)",
            )
            readout = s["bounds"]["readout"] / scale
            if not normalised:
                ax.axhline(bound, color=col[n], ls="--", lw=1)
            ax.axhline(readout, color=col[n], ls=":", lw=1)
        ax.set_title(f"{TITLES[kind]}, T = N, {encoding} encoding")
        ax.set_xlabel("p")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="major")
        handles, labels = ax.get_legend_handles_labels()
        bound_txt = BOUND_LABEL[encoding][key]
        extra = [Line2D([], [], color="grey", ls=":", lw=1)]
        extra_lab = ["final readout only: $2^N$" + (" / bound" if normalised else "")]
        if not normalised:
            extra.insert(0, Line2D([], [], color="grey", ls="--", lw=1))
            extra_lab.insert(0, f"bound {bound_txt}")
        ax.legend(handles + extra, labels + extra_lab, fontsize=7.5, loc="best")
    ylabel = "rank / bound of the class" if normalised else "rank of the effect span"
    axes[0].set_ylabel(ylabel + "\n(one circuit, one fixed mask per seed)")
    if normalised:
        fig.suptitle(SUPTITLE[encoding], fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = load(args.folder)
    for encoding in ("angle", "dense"):
        summary = summarise(runs, encoding)
        (out / f"effect_rank_{encoding}.json").write_text(json.dumps(summary, indent=1))
        plot(summary, encoding, out / f"effect_rank_{encoding}.png", normalised=False)
        plot(
            summary,
            encoding,
            out / f"effect_rank_{encoding}_normalised.png",
            normalised=True,
        )
        print(f"wrote {out}/effect_rank_{encoding}*.png")


if __name__ == "__main__":
    main()
