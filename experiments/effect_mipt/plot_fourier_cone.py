"""Figure of the Fourier cone (#327).

Reads the ``cone_N*_p*.json`` files of ``fourier_cone.py`` and writes
``fourier_cone.png``. With ``k(l, t)`` the number of independent Fourier
modes of the record supported in an interval of length ``l``:

- ``I(l, t) = k(N) - k(l) - k(N - l)`` counts the modes that straddle the
  two ends of the interval, that is the Fourier weight that correlates
  inputs on both sides;
- ``n(l, t) = dI / dl / 2`` is the number of modes per end longer than
  ``l``: the maps show where it is non-zero (the cone);
- ``xi(t) = sum_l n(l, t) / n(0, t)`` is the mean range of the modes that
  straddle a cut (truncated at ``N / 2``).

Usage::

    uv run python experiments/effect_mipt/plot_fourier_cone.py \
        outputs/effect_mipt --out experiments/effect_mipt/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

MAPS = (0.04, 0.08, 0.16, 0.24, 0.32)
P_C = 0.16

plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)


def load(folder: Path) -> dict[float, dict]:
    """Cone runs by ``p`` (one file per ``p``)."""
    out = {}
    for path in sorted(folder.glob("cone_N*_p*.json")):
        d = json.loads(path.read_text())
        k = np.array(d["k_mean"])
        n = d["n"]
        straddle = k[:, -1:] - k - k[:, ::-1]
        half = straddle[:, : n // 2 + 1]
        longer = np.diff(half, axis=1) / 2
        # Mean range of the straddling modes (truncated at N / 2).
        xi = longer.sum(axis=1) / np.maximum(longer[:, 0], 1e-9)
        out[round(d["p"], 2)] = {
            "n": n,
            "times": np.array(d["times"]),
            "straddle": half,
            "longer": longer,
            "xi": xi,
            "samples": d["samples"],
        }
    return dict(sorted(out.items()))


def phase_colors(ps: list[float]) -> dict[float, tuple]:
    """Blues for the mixed phase (darker deeper), oranges for the pure phase."""
    below = [p for p in ps if p < P_C - 1e-9]
    above = [p for p in ps if p > P_C + 1e-9]
    out: dict[float, tuple] = {}
    blues = plt.get_cmap("Blues")(np.linspace(0.95, 0.45, max(len(below), 1)))
    oranges = plt.get_cmap("Oranges")(np.linspace(0.45, 0.95, max(len(above), 1)))
    out.update(zip(below, map(tuple, blues), strict=False))
    out.update(zip(above, map(tuple, oranges), strict=False))
    for p in ps:
        out.setdefault(p, (0.1, 0.1, 0.1, 1.0))
    return out


def figure(data: dict[float, dict], path: Path) -> dict:
    fig = plt.figure(figsize=(17, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, len(MAPS))
    norm = LogNorm(vmin=1e-3, vmax=0.3)
    image = None
    for i, p in enumerate(MAPS):
        d = data[p]
        n, times = d["n"], d["times"]
        ax = fig.add_subplot(grid[0, i])
        image = ax.pcolormesh(
            np.arange(1, n // 2 + 1) - 0.5,
            times,
            np.clip(d["longer"], 1e-3, None),
            norm=norm,
            cmap="Blues",
            shading="nearest",
        )
        ax.plot(d["xi"], times, color="#111827", lw=1.6, label=r"mean range $\xi(t)$")
        ax.set_xlim(0, n // 2)
        ax.set_xlabel("range of the Fourier modes, l (sites)")
        if i == 0:
            ax.set_ylabel("record length t (half-layers)")
            ax.legend(frameon=False, loc="upper right")
        phase = "mixed" if p < P_C else ("critical" if p == P_C else "pure")
        title_color = {"mixed": "#1d4ed8", "critical": "0.1", "pure": "#c2410c"}[phase]
        ax.set_title(f"p = {p:.2f} ({phase})", loc="left", fontsize=11,
                     color=title_color, fontweight="bold")  # fmt: skip
    fig.colorbar(image, ax=fig.axes[: len(MAPS)], shrink=0.9,
                 label="modes longer than l, per end")  # fmt: skip

    colors = phase_colors(list(data))
    ax = fig.add_subplot(grid[1, :2])
    for p, d in data.items():
        ax.plot(d["times"], d["xi"], color=colors[p], lw=1.6, label=f"p = {p:.2f}")
    ax.set_xlabel("record length t (half-layers)")
    ax.set_ylabel(r"mean range of the modes, $\xi(t)$ (sites)")
    ax.set_title(
        "Mean range of the straddling modes against time", loc="left", fontsize=10
    )
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = fig.add_subplot(grid[1, 2:4])
    for p, d in data.items():
        ax.plot(d["times"], d["straddle"][:, -1], color=colors[p], lw=1.6)
    ax.set_xlabel("record length t (half-layers)")
    ax.set_ylabel(r"$I(N/2, t)$ (modes)")
    ax.set_title("Modes straddling a cut of the input in two halves", loc="left",
                 fontsize=10)  # fmt: skip

    ax = fig.add_subplot(grid[1, 4])
    last = [d["straddle"][-1, -1] for d in data.values()]
    ax.plot(list(data), last, color="0.6", lw=1, zorder=0)
    ax.scatter(list(data), last, c=[colors[p] for p in data], s=40, zorder=1)
    ax.axvspan(0, P_C, color="#dbeafe", zorder=-1)
    ax.axvspan(P_C, 1, color="#ffedd5", zorder=-1)
    ax.set_xlim(min(data) - 0.02, max(data) + 0.02)
    ax.text(0.03, 0.05, "mixed", transform=ax.transAxes, color="#1d4ed8")
    ax.text(0.97, 0.9, "pure", transform=ax.transAxes, color="#c2410c", ha="right")
    ax.axvline(P_C, color="0.6", lw=1, ls="--")
    ax.set_xlabel("measurement rate p")
    ax.set_ylabel(r"$I(N/2, 2N)$ (modes)")
    ax.set_title("At the end of the record", loc="left", fontsize=10)

    n = data[MAPS[0]]["n"]
    fig.suptitle(
        f"Fourier cone: range of the Fourier modes of one record, dense encoding, "
        f"Clifford literature circuit, N = {n}",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        str(p): {
            "samples": d["samples"],
            "times": d["times"].tolist(),
            "xi": d["xi"].tolist(),
            "straddle_half": d["straddle"][:, -1].tolist(),
        }
        for p, d in data.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/effect_mipt/figures")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = figure(load(args.folder), args.out / "fourier_cone.png")
    (args.out / "fourier_cone_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
