#!/usr/bin/env python3
"""Characteristic p of the QELM against the space-time volume ``N T`` (issue #285).

On the four encodings of ``experiments/nonlinear_M`` (``ghz``, ``w``,
``bell``, ``charge``; fixed Haar U(4) circuits), two measurement rates
mark the curves of the QELM:

- ``p_cross``: the first ``p > 0`` where the accuracy of the QELM on the
  state and the record (``ridge_M_quantum``: ``<P_k>``, ``<P_i P_j>`` and
  ``vec(M)``) falls below the accuracy of the Bayes decoder of ``M``
  (``bayes``: ``sign(<Z_R>)``, the observer who knows the circuit). Linear
  interpolation of the difference of the two mean curves between the two
  grid points around the sign change.
- ``p_min``: the ``p > 0`` where the mean accuracy of that QELM is minimal,
  refined by the vertex of the parabola through the grid minimum and its two
  neighbours. It is **defined** only if the minimum is not on the first or
  last grid point and the curve rises after it by more than
  ``MIN_SIGNIFICANCE`` standard errors; otherwise the QELM stays on a
  plateau (at chance) after its fall and there is no dip to locate.

Errors: the standard deviation of each estimate over :data:`N_BOOT`
bootstrap resamplings of the circuits (seeds); ``*_defined_fraction`` is the
share of resamplings where the estimate exists.

Subcommands:

    # p grid of one run, dense in p N T (used by characteristic_p.sh)
    python experiments/characteristic_p/characteristic_p.py grid 8 8

    # figures and summary.json of a sweep tree <root>/<encoding>/N<N>_T<T>/seed<s>/
    uv run python experiments/characteristic_p/characteristic_p.py plot \\
        outputs/characteristic_p/<stamp> --out figs

``plot`` writes ``accuracy_vs_pNT_N<N>_T<T>.png`` for every ``(N, T)`` of
the sweep (the QELM and Bayes curves against ``p N T`` with ``p_cross`` and
``p_min`` marked), ``p_star_vs_NT.png`` when the sweep holds several
``N T`` (``p_cross`` and ``p_min`` against ``N T``, and ``p* N T``), and
``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Expected numbers of measurements p N T of the grid.
PNT_GRID = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 32, 45, 64)
# Values of p always included, so that the large-p rise is seen at any N T.
P_ANCHORS = (0.5, 1.0)
ENCODINGS = ("ghz", "w", "bell", "charge")
LABELS = {"ghz": "GHZ", "w": "W", "bell": "Bell (middle)", "charge": "charge"}
# The QELM whose crossing and minimum are located, and the reference.
QELM = "ridge_M_quantum"
BAYES = "bayes"
# Curves of the accuracy figure: (metric, legend, color, style).
CURVES = (
    (QELM, "QELM on <P_k>, <P_i P_j> and M", "#fa5703", "-"),
    ("ridge_quantum", "QELM on <P_k>, <P_i P_j>", "#1f77b4", ":"),
    ("ridge_raw", "QELM on M", "#d62728", ":"),
    (BAYES, "Bayes on M", "black", "-"),
)
N_BOOT = 1000
MIN_SIGNIFICANCE = 2.0
# Share of resamplings where an estimate must exist to enter the fit.
FIT_MIN_DEFINED = 0.9


def p_grid(N: int, T: int) -> np.ndarray:
    """Measurement probabilities of a run: ``PNT_GRID / (N T)`` and the anchors."""
    if N < 1 or T < 1:
        raise ValueError(f"N and T must be >= 1, got N={N}, T={T}.")
    values = np.asarray(PNT_GRID, dtype=float) / (N * T)
    values = np.concatenate([values[values <= 1.0], P_ANCHORS])
    return np.asarray(np.unique(np.round(values, 6)))


def crossing(p: np.ndarray, qelm: np.ndarray, bayes: np.ndarray) -> float | None:
    """First ``p > 0`` where ``qelm - bayes`` goes from positive to ``<= 0``.

    Linear interpolation between the two grid points around the change of
    sign. ``None`` if the QELM is not above Bayes at the first ``p > 0``, or
    never falls below it.
    """
    d = np.asarray(qelm, dtype=float) - np.asarray(bayes, dtype=float)
    p = np.asarray(p, dtype=float)
    keep = p > 0
    p, d = p[keep], d[keep]
    if d.size == 0 or d[0] <= 0:
        return None
    below = np.flatnonzero(d <= 0)
    if below.size == 0:
        return None
    j = int(below[0])
    p0, p1, d0, d1 = p[j - 1], p[j], d[j - 1], d[j]
    return float(p0 + (p1 - p0) * d0 / (d0 - d1))


def minimum(
    p: np.ndarray, acc: np.ndarray, sem: np.ndarray | None = None
) -> float | None:
    """``p > 0`` of the minimum of ``acc``, or ``None`` if there is no dip.

    The grid minimum is refined by the vertex of the parabola through it and
    its two neighbours (kept between them). ``None`` if the minimum is on the
    first or last grid point of ``p > 0``, or, with ``sem``, if the largest
    accuracy after it is not above it by :data:`MIN_SIGNIFICANCE` combined
    standard errors.
    """
    p = np.asarray(p, dtype=float)
    acc = np.asarray(acc, dtype=float)
    keep = p > 0
    p, acc = p[keep], acc[keep]
    j = int(np.argmin(acc))
    if j == 0 or j == p.size - 1:
        return None
    if sem is not None:
        err = np.asarray(sem, dtype=float)[keep]
        k = j + 1 + int(np.argmax(acc[j + 1 :]))
        if acc[k] - acc[j] <= MIN_SIGNIFICANCE * float(np.hypot(err[j], err[k])):
            return None
    x, y = p[j - 1 : j + 2], acc[j - 1 : j + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(p[j])
    return float(np.clip(-b / (2 * a), x[0], x[2]))


Tree = dict[str, dict[tuple[int, int], list[dict[str, float]]]]


def load_tree(root: Path) -> Tree:
    """``{encoding: {(N, T): [metrics of each seed]}}`` of a sweep tree.

    Raises:
        ValueError: If the tree holds no run.
    """
    out: Tree = {}
    for path in sorted(Path(root).glob("*/N*_T*/seed*/metrics.json")):
        encoding = path.parents[2].name
        metrics = json.loads(path.read_text())
        key = (int(metrics["N"]), int(metrics["T"]))
        out.setdefault(encoding, {}).setdefault(key, []).append(metrics)
    if not out:
        raise ValueError(f"No <encoding>/N<N>_T<T>/seed<s>/metrics.json under {root}.")
    return out


def p_values(metrics: dict[str, float]) -> np.ndarray:
    """The ``p`` of a run, from its ``accuracy_bayes_p<p>`` keys."""
    prefix = "accuracy_bayes_p"
    return np.asarray(
        sorted(float(k[len(prefix) :]) for k in metrics if k.startswith(prefix))
    )


def curves(runs: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``p`` (common to every seed) and ``(n_seeds, n_p)`` accuracy arrays.

    Raises:
        ValueError: If the seeds do not share one p grid.
    """
    ps = p_values(runs[0])
    for run in runs[1:]:
        if not np.array_equal(p_values(run), ps):
            raise ValueError("The seeds of one (N, T) have different p grids.")
    arrays = {
        name: np.asarray([[run[f"accuracy_{name}_p{v:g}"] for v in ps] for run in runs])
        for name, *_ in CURVES
    }
    return ps, arrays


def _sem(a: np.ndarray) -> np.ndarray:
    if a.shape[0] < 2:
        return np.zeros(a.shape[1])
    return np.asarray(a.std(axis=0, ddof=1) / np.sqrt(a.shape[0]))


def estimate(
    ps: np.ndarray, arrays: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, Any]:
    """``p_cross`` and ``p_min`` of the mean curves and their bootstrap errors."""
    q, b = arrays[QELM], arrays[BAYES]
    n = q.shape[0]
    out: dict[str, Any] = {
        "p_cross": crossing(ps, q.mean(0), b.mean(0)),
        "p_min": minimum(ps, q.mean(0), _sem(q)),
    }
    boot: dict[str, list[float]] = {"p_cross": [], "p_min": []}
    for _ in range(N_BOOT):
        idx = rng.integers(n, size=n)
        qs, bs = q[idx], b[idx]
        c = crossing(ps, qs.mean(0), bs.mean(0))
        m = minimum(ps, qs.mean(0), _sem(qs))
        if c is not None:
            boot["p_cross"].append(c)
        if m is not None:
            boot["p_min"].append(m)
    for name, values in boot.items():
        out[f"{name}_err"] = float(np.std(values)) if len(values) > 1 else None
        out[f"{name}_defined_fraction"] = len(values) / N_BOOT
    positive = np.where(ps > 0, q.mean(0), np.inf)
    out["accuracy_min"] = float(positive.min())
    out["p_grid_min"] = float(ps[int(np.argmin(positive))])
    return out


def _order(encoding: str) -> int:
    return ENCODINGS.index(encoding) if encoding in ENCODINGS else len(ENCODINGS)


def summarize(tree: Tree, seed: int = 0) -> dict[str, Any]:
    """Curves and estimates of every encoding and ``(N, T)`` of the tree."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for encoding in sorted(tree, key=_order):
        for (N, T), runs in sorted(tree[encoding].items()):
            ps, arrays = curves(runs)
            entry: dict[str, Any] = {
                "N": N,
                "T": T,
                "NT": N * T,
                "n_circuits": len(runs),
                "p": ps.tolist(),
            }
            for name, a in arrays.items():
                entry[f"accuracy_{name}"] = a.mean(0).tolist()
                entry[f"accuracy_{name}_sem"] = _sem(a).tolist()
            entry.update(estimate(ps, arrays, rng))
            out.setdefault(encoding, {})[f"N{N}_T{T}"] = entry
    return out


def plot_accuracy(summary: dict[str, Any], N: int, T: int, out: Path) -> Path:
    """Accuracy against ``p N T`` at one ``(N, T)``, one panel per encoding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"N{N}_T{T}"
    encodings = [e for e in summary if key in summary[e]]
    cols = min(len(encodings), 2)
    rows = (len(encodings) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(6 * cols, 4.2 * rows), squeeze=False, sharey=True
    )
    for ax, encoding in zip(axes.flat, encodings, strict=False):
        entry = summary[encoding][key]
        x = np.asarray(entry["p"]) * N * T
        for name, legend, color, style in CURVES:
            mean = np.asarray(entry[f"accuracy_{name}"])
            sem = np.asarray(entry[f"accuracy_{name}_sem"])
            width = 2.0 if name in (QELM, BAYES) else 1.2
            ax.plot(
                x, mean, style, color=color, lw=width, marker="o", ms=3, label=legend
            )
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        for name, style in (("p_cross", "-"), ("p_min", "--")):
            value, err = entry[name], entry[f"{name}_err"]
            if value is None:
                continue
            ax.axvline(value * N * T, color="#555555", ls=style, lw=1.2)
            if err:
                lo, hi = (value - err) * N * T, (value + err) * N * T
                ax.axvspan(lo, hi, color="#555555", alpha=0.12, lw=0)
            ax.annotate(
                f"{name} = {value:.3f}",
                xy=(value * N * T, 0.97),
                xycoords=("data", "axes fraction"),
                rotation=90,
                ha="right",
                va="top",
                fontsize=8,
                color="#333333",
            )
        missing = [n for n in ("p_cross", "p_min") if entry[n] is None]
        if missing:
            ax.text(
                0.98,
                0.6,
                ", ".join(missing) + ": not defined",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                color="#333333",
            )
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(0, x.max() * 1.05)
        title = f"{LABELS.get(encoding, encoding)} ({entry['n_circuits']} circuits)"
        ax.set_title(title)
        ax.set_xlabel("p N T (expected number of measurements)")
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in list(axes.flat)[len(encodings) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"QELM against Bayes on M, N = {N}, T = {T} (fixed Haar U(4))")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out / f"accuracy_vs_pNT_{key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


MARKERS = {4: "s", 6: "^", 8: "o", 10: "D", 12: "v"}


def power_law_fit(
    summary: dict[str, Any], min_defined: float = FIT_MIN_DEFINED
) -> dict[str, Any]:
    """Least-squares fit of ``log p* = log A + a log N + b log T`` per estimate.

    Only the ``(N, T)`` where the estimate exists in at least ``min_defined``
    of the bootstrap resamplings enter; three points at least, with two
    values of ``N`` and two of ``T``. If ``p*`` depended on ``N T`` only,
    ``a = b``; ``a = b = -1`` is a constant ``p* N T``.

    Returns:
        ``{encoding: {name: {A, a, b, rms_log, n_points} or None}}``.
    """
    out: dict[str, Any] = {}
    for encoding, entries in summary.items():
        out[encoding] = {}
        for name in ("p_cross", "p_min"):
            pts = [
                e
                for e in entries.values()
                if e[name] is not None and e[f"{name}_defined_fraction"] >= min_defined
            ]
            Ns = {e["N"] for e in pts}
            Ts = {e["T"] for e in pts}
            if len(pts) < 3 or len(Ns) < 2 or len(Ts) < 2:
                out[encoding][name] = None
                continue
            A = np.column_stack(
                [
                    np.ones(len(pts)),
                    np.log([e["N"] for e in pts]),
                    np.log([e["T"] for e in pts]),
                ]
            )
            y = np.log([e[name] for e in pts])
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            res = y - A @ coef
            out[encoding][name] = {
                "A": float(np.exp(coef[0])),
                "a": float(coef[1]),
                "b": float(coef[2]),
                "rms_log": float(np.sqrt(np.mean(res**2))),
                "n_points": len(pts),
            }
    return out


# Rows of the scaling figure: (y label, factor of p* as a function of N, T).
SCALING_ROWS = (
    ("p*", lambda N, T: 1.0),
    ("p* N T", lambda N, T: float(N * T)),
    ("p* T", lambda N, T: float(T)),
)


def plot_p_star(
    summary: dict[str, Any], out: Path, fits: dict[str, Any] | None = None
) -> Path:
    """``p_cross`` and ``p_min`` against ``N T``: ``p*``, ``p* N T``, ``p* T``.

    One column per encoding, one marker per ``N``; the points of one ``N``
    are joined. Estimates defined in less than :data:`FIT_MIN_DEFINED` of the
    resamplings are drawn hollow.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    encodings = list(summary)
    rows = len(SCALING_ROWS)
    fig, axes = plt.subplots(
        rows,
        len(encodings),
        figsize=(4.2 * len(encodings), 3.4 * rows),
        squeeze=False,
        sharex=True,
    )
    for col, encoding in enumerate(encodings):
        entries = sorted(summary[encoding].values(), key=lambda e: (e["N"], e["T"]))
        sizes = sorted({int(e["N"]) for e in entries})
        for name, color in (("p_cross", "black"), ("p_min", "#fa5703")):
            for N in sizes:
                pts = [e for e in entries if e["N"] == N and e[name] is not None]
                if not pts:
                    continue
                m = MARKERS.get(N, "o")
                for row, (_, factor) in enumerate(SCALING_ROWS):
                    ax = axes[row, col]
                    k = np.asarray([factor(e["N"], e["T"]) for e in pts])
                    nt = np.asarray([e["NT"] for e in pts], dtype=float)
                    val = np.asarray([e[name] for e in pts]) * k
                    err = np.asarray([e[f"{name}_err"] or 0.0 for e in pts]) * k
                    ax.plot(nt, val, "-", color=color, lw=0.8, alpha=0.5)
                    for e, x, y, dy in zip(pts, nt, val, err, strict=True):
                        solid = e[f"{name}_defined_fraction"] >= FIT_MIN_DEFINED
                        ax.errorbar(
                            x,
                            y,
                            yerr=dy,
                            fmt=m,
                            color=color,
                            mfc=color if solid else "white",
                            ms=6,
                            capsize=2,
                        )
        ticks = sorted({e["NT"] for e in entries})
        for row, (label, _) in enumerate(SCALING_ROWS):
            ax = axes[row, col]
            ax.set_xscale("log")
            if row == 0:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                ax.set_ylim(bottom=0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25, lw=0.5, which="both")
        axes[-1, col].set_xlabel("N T")
        title = LABELS.get(encoding, encoding)
        fit = (fits or {}).get(encoding, {}).get("p_cross")
        if fit:
            title += f"\np_cross ~ N^{fit['a']:.2f} T^{fit['b']:.2f}"
        axes[0, col].set_title(title, fontsize=10)
    for name, color, legend in (
        ("p_cross", "black", "p_cross (QELM = Bayes on M)"),
        ("p_min", "#fa5703", "p_min (QELM minimum)"),
    ):
        axes[0, 0].plot([], [], "-", color=color, label=legend)
    all_sizes = sorted({int(e["N"]) for enc in summary.values() for e in enc.values()})
    for N in all_sizes:
        axes[0, 0].plot([], [], MARKERS.get(N, "o"), color="grey", label=f"N = {N}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Characteristic p of the QELM against the space-time volume N T")
    fig.tight_layout()
    path = out / "p_star_vs_NT.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(value: float | None, err: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.4f}+-{err or 0.0:.4f}"


def main(argv: list[str] | None = None) -> list[Path]:
    """CLI: ``grid N T`` prints the p list; ``plot ROOT... --out DIR`` draws."""
    parser = argparse.ArgumentParser(description="Characteristic p of the QELM.")
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("grid", help="print the p grid of one run")
    grid.add_argument("N", type=int)
    grid.add_argument("T", type=int)
    plot = sub.add_parser("plot", help="figures and summary.json of sweeps")
    plot.add_argument("roots", nargs="+", type=Path)
    plot.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    if args.command == "grid":
        print("[" + ",".join(f"{v:g}" for v in p_grid(args.N, args.T)) + "]")
        return []
    tree: Tree = {}
    for root in args.roots:
        for encoding, sizes in load_tree(root).items():
            for key, runs in sizes.items():
                tree.setdefault(encoding, {}).setdefault(key, []).extend(runs)
    summary = summarize(tree)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    sizes = sorted({(e["N"], e["T"]) for enc in summary.values() for e in enc.values()})
    paths = [plot_accuracy(summary, N, T, args.out) for N, T in sizes]
    if len({N * T for N, T in sizes}) > 1:
        fits = power_law_fit(summary)
        (args.out / "fits.json").write_text(json.dumps(fits, indent=2))
        paths.append(plot_p_star(summary, args.out, fits))
        for encoding, by_name in fits.items():
            for name, fit in by_name.items():
                if fit:
                    print(
                        f"{encoding:7s} {name:7s} ~ {fit['A']:.2f} N^{fit['a']:.2f} "
                        f"T^{fit['b']:.2f} (rms log {fit['rms_log']:.3f}, "
                        f"{fit['n_points']} points)"
                    )
    for encoding, entries in summary.items():
        for key, e in entries.items():
            print(
                f"{encoding:7s} {key:8s} circuits={e['n_circuits']:2d} "
                f"p_cross={_fmt(e['p_cross'], e['p_cross_err'])} "
                f"({e['p_cross_defined_fraction']:.0%}) "
                f"p_min={_fmt(e['p_min'], e['p_min_err'])} "
                f"({e['p_min_defined_fraction']:.0%})"
            )
    return paths


if __name__ == "__main__":
    main(sys.argv[1:])
