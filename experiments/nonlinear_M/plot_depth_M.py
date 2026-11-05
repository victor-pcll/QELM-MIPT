#!/usr/bin/env python3
"""Accuracy of the QELM on M against p, one curve per depth T (issue #269).

Does what the ridge on ``vec(M)`` reads depend on the depth of the circuit?
Each argument ``T=<folder>`` names the sweep (or sweeps, comma-separated, for
extra values of ``p`` on the same circuits) of ``nonlinear_M.sh`` run with
``DEPTH=<T>``. The circuit of seed ``s`` at depth ``T`` is the first ``T``
layers of the circuit of the same seed at a larger depth, because the gates
are drawn layer by layer from the same generator.

Writes ``depth_M_<N>.png`` (and ``depth_M_<N>.json``, the curves) to
``--out``:

    uv run python experiments/nonlinear_M/plot_depth_M.py \\
        4=outputs/nonlinear_M/T4 8=outputs/nonlinear_M/T8 --label GHZ --out figs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def _plot_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "plot_nonlinear_M", HERE / "plot_nonlinear_M.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def depth_curves(pairs: list[tuple[int, list[Path]]]) -> dict[str, Any]:
    """``{N: {T: {p, accuracy, sem, n_circuits}}}`` of the ridge on ``M``."""
    plot = _plot_module()
    out: dict[str, Any] = {}
    if not pairs:
        raise ValueError("No T=<folder> given.")
    for T, folders in sorted(pairs):
        for key, entry in plot.summarize(plot.load_runs(folders)).items():
            if int(entry["T"]) != T:
                # Another N of the same sweep, with T = N or 2N.
                continue
            out.setdefault(key, {})[str(T)] = {
                "p": entry["p"],
                "accuracy": entry["accuracy_ridge_raw"],
                "sem": entry["accuracy_ridge_raw_sem"],
                "n_circuits": entry["n_circuits"],
            }
    return out


def plot_depth(curves: dict[str, Any], key: str, label: str, out: Path) -> Path:
    """One curve per ``T``, ``p`` on a linear axis."""
    by_T = curves[key]
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(by_T)))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for color, (T, c) in zip(
        colors, sorted(by_T.items(), key=lambda kv: int(kv[0])), strict=True
    ):
        ps, mean, sem = (np.asarray(c[k]) for k in ("p", "accuracy", "sem"))
        ax.plot(ps, mean, "-", color=color, marker="o", ms=4, label=f"T = {T}")
        ax.fill_between(ps, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    n_circuits = min(c["n_circuits"] for c in by_T.values())
    head = f"{label}, " if label else ""
    ax.set_title(
        f"QELM on M, {head}Haar U(4), N = {key[1:]}, {n_circuits} fixed circuits",
        fontsize=10,
    )
    ax.set_xlabel("measurement probability p")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    path = out / f"depth_M_{key}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw one figure per ``N``; return their paths."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pairs", nargs="+", help="T=<folder>[,<folder>...]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="", help="the encoding, for the title")
    args = parser.parse_args(argv)
    pairs = []
    for item in args.pairs:
        T, _, folders = item.partition("=")
        if not T.isdigit() or not folders:
            raise ValueError(f"Expected T=<folder>, got {item!r}.")
        pairs.append((int(T), [Path(f) for f in folders.split(",")]))
    args.out.mkdir(parents=True, exist_ok=True)
    curves = depth_curves(pairs)
    paths = []
    for key in sorted(curves):
        path = plot_depth(curves, key, args.label, args.out)
        path.with_suffix(".json").write_text(json.dumps(curves[key], indent=2))
        print(f"Saved {path}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
