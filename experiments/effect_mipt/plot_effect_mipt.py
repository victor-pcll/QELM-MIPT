"""Figures of the reference MIPT probes and of the frame dimension (#322).

Reads the JSON files of ``effect_mipt.py`` in a folder (several files of the
same probe and ``N`` are merged: rows by ``p``, runs by seed) and writes:

- ``clifford_mixed.png`` and ``clifford_reference.png``: Clifford, entropy
  of the initially mixed system and of the reference qubit at the last
  layer against ``p``, a zoom on the crossing, and the pairwise crossings;
- ``haar_reference.png``: the same for the Haar reference qubit;
- ``frame_dimension.png``: ``log2(D_2 / 2^N) / N`` and ``C / N`` (rank) for
  the same circuits, the ratio of the two, and the position of the maxima,
  with the ``p_c`` of the two probes.

``summary.json`` holds every plotted number, the crossings and ``p_c``.

Usage::

    uv run python experiments/effect_mipt/plot_effect_mipt.py \
        outputs/effect_mipt --out experiments/effect_mipt/figures
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

plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)


def load(folder: Path, probe: str) -> dict[int, dict]:
    """Rows of one probe by ``N``, merged across files (``p`` sorted)."""
    by_n: dict[int, dict] = {}
    for path in sorted(folder.glob(f"{probe}_N*.json")):
        d = json.loads(path.read_text())
        n = int(d["n"])
        if probe == "frame":
            block = by_n.setdefault(n, {"depth": d["depth"], "p": d["p"], "runs": []})
            block["runs"] += d["runs"]
        else:
            block = by_n.setdefault(n, {"depth": d["depth"], "rows": {}})
            for row in d["rows"]:
                old = block["rows"].get(row["p"])
                block["rows"][row["p"]] = merge_rows(old, row) if old else row
    return dict(sorted(by_n.items()))


def merge_rows(a: dict, b: dict) -> dict:
    """Pool two rows of the same ``p`` (means and standard errors)."""
    out = {"p": a["p"], "samples": a["samples"] + b["samples"]}
    for key in ("reference", "system"):
        if f"{key}_mean" not in a:
            continue
        na, nb = a["samples"], b["samples"]
        ma, mb = np.array(a[f"{key}_mean"]), np.array(b[f"{key}_mean"])
        va = (np.array(a[f"{key}_sem"]) ** 2) * na * (na - 1)
        vb = (np.array(b[f"{key}_sem"]) ** 2) * nb * (nb - 1)
        m = (na * ma + nb * mb) / (na + nb)
        ss = va + vb + na * (ma - m) ** 2 + nb * (mb - m) ** 2
        n = na + nb
        out[f"{key}_mean"] = m.tolist()
        out[f"{key}_sem"] = np.sqrt(ss / (n - 1) / n).tolist()
    return out


def last_layer(block: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ps = np.array(sorted(block["rows"]))
    m = np.array([block["rows"][p][f"{key}_mean"][-1] for p in ps])
    e = np.array([block["rows"][p][f"{key}_sem"][-1] for p in ps])
    return ps, m, e


def crossing(p, m1, e1, m2, e2) -> dict | None:
    """First sign change of ``m2 - m1`` (larger N minus smaller N).

    Returns the linear interpolation of the crossing and whether it is
    resolved: ``|z| > 2`` on a grid point on each side.
    """
    d = m2 - m1
    z = d / np.sqrt(e1**2 + e2**2 + 1e-30)
    for i in range(len(p) - 1):
        if d[i] > 0 >= d[i + 1]:
            pc = p[i] + (p[i + 1] - p[i]) * d[i] / (d[i] - d[i + 1])
            resolved = bool(np.any(z[: i + 1] > 2) and np.any(z[i + 1 :] < -2))
            return {"p": float(pc), "resolved": resolved}
    return None


def blues(sizes) -> dict:
    return dict(
        zip(sizes, plt.get_cmap("Blues")(np.linspace(0.4, 1, len(sizes))), strict=True)
    )


def reference_figure(data: dict[int, dict], key: str, title: str, path: Path) -> dict:
    """Entropy at the last layer, its pairwise crossings, and ``p_c``.

    ``p_c`` is the median of the pairwise crossings; its range is their
    spread. No collapse is fitted: with these
    sizes the fit runs to the edge of any window (see the README).
    """
    sizes = list(data)
    col = blues(sizes)
    curves = {n: last_layer(data[n], key) for n in sizes}
    pairs = []
    for n1, n2 in zip(sizes, sizes[1:], strict=False):
        p1, p2 = curves[n1][0], curves[n2][0]
        k1, k2 = np.isin(p1, p2), np.isin(p2, p1)
        c = crossing(
            p1[k1],
            curves[n1][1][k1],
            curves[n1][2][k1],
            curves[n2][1][k2],
            curves[n2][2][k2],
        )
        if c is not None:
            pairs.append({"N": [n1, n2], **c})
    found = [c["p"] for c in pairs]
    pc = float(np.median(found)) if found else float("nan")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for n in sizes:
        p, m, e = curves[n]
        for ax in axes[:2]:
            ax.errorbar(
                p,
                m,
                yerr=e,
                color=col[n],
                marker="o",
                ms=3,
                lw=1.5,
                capsize=2,
                label=f"N = {n}",
            )
    label = "S(R)" if key == "reference" else "S(ρ$_{sys}$)"
    axes[0].set_ylabel(f"{label} at the last layer (bits)")
    axes[0].set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlim(pc - 0.08, pc + 0.08)
    axes[1].set_yscale("log")
    axes[1].set_ylim(3e-3, 3)
    axes[1].set_title("Zoom on the crossing (log scale)", fontsize=10)
    for ax in axes[:2]:
        ax.axvline(pc, color="0.4", ls="--", lw=1)
        ax.set_xlabel("p")
    for c in pairs:
        n1, n2 = c["N"]
        mk = "o" if c["resolved"] else "x"
        axes[2].plot(
            1 / np.sqrt(n1 * n2), c["p"], mk, color=plt.get_cmap("Blues")(0.8), ms=7
        )
    axes[2].axhline(pc, color="0.4", ls="--", lw=1, label=f"median: {pc:.3f}")
    axes[2].set_xlabel("1 / sqrt(N N')")
    axes[2].set_ylabel("crossing p of N and N'")
    axes[2].set_title("Pairwise crossings (o resolved, x not)", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].set_xlim(left=0)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "quantity": key,
        "crossings": pairs,
        "p_c": pc,
        "p_c_range": [min(found), max(found)] if found else [],
        "curves": {str(n): [a.tolist() for a in curves[n]] for n in sizes},
    }


def frame_figure(data: dict[int, dict], pcs: dict[str, float], path: Path) -> dict:
    sizes = list(data)
    col = blues(sizes)
    out: dict = {}
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4))
    for n in sizes:
        runs = data[n]["runs"]
        p = np.array(data[n]["p"])
        d2 = (np.array([r["log2_d2"] for r in runs]) - n) / n
        cut = (np.array([r["cut"] for r in runs]) - n) / n
        k = len(runs)
        m, e = d2.mean(0), d2.std(0, ddof=1) / np.sqrt(k)
        mc, ec = cut.mean(0), cut.std(0, ddof=1) / np.sqrt(k)
        ratio = m / np.where(mc > 0, mc, np.nan)
        axes[0].errorbar(
            p,
            mc,
            yerr=ec,
            color=col[n],
            marker="o",
            ms=3,
            lw=1.5,
            capsize=2,
            label=f"N = {n} ({k} circuits)",
        )
        axes[1].errorbar(
            p,
            m,
            yerr=e,
            color=col[n],
            marker="o",
            ms=3,
            lw=1.5,
            capsize=2,
            label=f"N = {n}",
        )
        axes[2].plot(p, ratio, "o-", color=col[n], ms=3, lw=1.5, label=f"N = {n}")
        i, j = int(np.argmax(m)), int(np.argmax(mc))
        axes[3].plot(n, p[i], "o", color=col[n], ms=7)
        axes[3].plot(n, p[j], "s", color=col[n], ms=7, mfc="white")
        out[str(n)] = {
            "p": p.tolist(),
            "circuits": k,
            "d2_excess": m.tolist(),
            "d2_sem": e.tolist(),
            "rank_excess": mc.tolist(),
            "rank_sem": ec.tolist(),
            "p_max_d2": float(p[i]),
            "p_max_rank": float(p[j]),
        }
    styles = {"clifford": ":", "haar": "--"}
    for ax in axes[:3]:
        for name, pc in pcs.items():
            ax.axvline(pc, color="0.4", ls=styles[name], lw=1)
        ax.set_xlabel("p")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("extra bits per site")
    axes[0].set_title("Rank: log$_2$(rank / 2$^N$) / N  (min cut)", fontsize=10)
    axes[1].set_title("Born-weighted: log$_2$(D$_2$ / 2$^N$) / N", fontsize=10)
    axes[2].set_title("Ratio of the two", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    for name, pc in pcs.items():
        axes[3].axhline(
            pc, color="0.4", ls=styles[name], lw=1, label=f"p_c {name} = {pc:.3f}"
        )
    axes[3].plot([], [], "o", color="0.3", label="max of D$_2$")
    axes[3].plot([], [], "s", color="0.3", mfc="white", label="max of the rank")
    axes[3].set_xlabel("N")
    axes[3].set_ylabel("p of the maximum")
    axes[3].set_title("Position of the maxima (grid points)", fontsize=10)
    axes[3].legend(fontsize=8, frameon=False)
    axes[3].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/effect_mipt/figures")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = defaultdict(dict)
    pcs = {}
    cliff = load(args.folder, "clifford")
    if cliff:
        summary["clifford"] = reference_figure(
            cliff,
            "system",
            "Clifford, mixed start, T = 4N",
            args.out / "clifford_mixed.png",
        )
        summary["clifford_reference"] = reference_figure(
            cliff,
            "reference",
            "Clifford, reference qubit, T = 4N",
            args.out / "clifford_reference.png",
        )
        pcs["clifford"] = summary["clifford"]["p_c"]
    haar = load(args.folder, "haar")
    if haar:
        summary["haar"] = reference_figure(
            haar,
            "reference",
            "Haar, reference qubit, N then 2N layers",
            args.out / "haar_reference.png",
        )
        pcs["haar"] = summary["haar"]["p_c"]
    frame = load(args.folder, "frame")
    if frame:
        summary["frame"] = frame_figure(frame, pcs, args.out / "frame_dimension.png")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name in ("clifford", "clifford_reference", "haar"):
        if name in summary:
            print(name, summary[name]["p_c"], summary[name]["crossings"])


if __name__ == "__main__":
    main()
