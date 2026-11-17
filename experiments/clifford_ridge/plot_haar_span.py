#!/usr/bin/env python3
"""Haar against Clifford at K = infinity, same geometry (issue #339).

Reads ``haar_span_N*_seed*.json`` from the output folder of ``haar_span.py``
and writes ``haar_vs_clifford.png`` and ``summary_haar_span.json`` in
``figures/``.

Usage::

    uv run python experiments/clifford_ridge/plot_haar_span.py outputs/haar_span
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

P_C_HAAR = 0.29
KINDS = {"haar": ("-", "o"), "clifford": ("--", "s")}


def load(folder: Path) -> dict:
    runs: dict = defaultdict(list)
    for path in sorted(folder.glob("haar_span_N*_seed*.json")):
        d = json.loads(path.read_text())
        runs[d["n"]].append(d)
    return dict(runs)


def stat(runs: list, kind: str, getter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``p``, mean and SEM over the seeds of ``getter(row)``."""
    p = np.asarray(runs[0]["p"])
    vals = np.array([[getter(r) for r in run["per_kind"][kind]] for run in runs])
    sem = vals.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0 * p
    return p, vals.mean(axis=0), sem


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    runs = load(args.folder)
    if not runs:
        raise SystemExit("No haar_span run.")
    out = args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ns = sorted(runs)
    cols = dict(
        zip(
            ns,
            [plt.get_cmap("viridis")(x) for x in np.linspace(0.1, 0.85, len(ns))],
            strict=True,
        )
    )

    def e_perp(name: str):
        return lambda r, n: r["full"]["e_perp"][name]

    panels = [
        ("dense rank / 4^N", lambda r, n: r["full"]["dense"] / 4**n, "log"),
        ("angle rank / 3^N (no Y)", lambda r, n: r["full"]["angle"] / 3**n, "log"),
        ("E_perp of Z of the middle qubit", e_perp("z_mid"), "linear"),
        ("E_perp of a degree-1 target", e_perp("k1"), "linear"),
        ("E_perp of a degree-2 target", e_perp("k2"), "linear"),
        (
            "dim(Span ∩ V_l) / 3^l, window l = 2",
            lambda r, n: float(np.mean(r["full"]["window"]["2"])),
            "linear",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    summary: dict = {"N": ns, "seeds": {n: len(runs[n]) for n in ns}}
    for ax, (title, getter, scale) in zip(axes.ravel(), panels, strict=True):
        summary[title] = {}
        for n in ns:
            for kind, (ls, mk) in KINDS.items():
                p, m, e = stat(runs[n], kind, lambda r, n=n, g=getter: g(r, n))
                ax.errorbar(p, m, e, color=cols[n], ls=ls, marker=mk, ms=3,
                            label=f"{kind} N={n}")  # fmt: skip
                summary[title][f"{kind}_N{n}"] = {"p": p.tolist(), "mean": m.tolist()}
            if title.startswith("dense"):
                ax.axhline(2.0**n / 4**n, color=cols[n], ls=":", lw=0.8)
        ax.axvline(P_C_HAAR, color="0.4", ls="--", lw=1)
        ax.set_yscale(scale)
        ax.set_xlabel("p")
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(
        "K = inf, lambda -> 0+, full record. Haar (solid) vs Clifford (dashed), "
        "same circuits geometry and masks, T = 2N. Dotted: 2^N / 4^N.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "haar_vs_clifford.png", dpi=150)
    plt.close(fig)
    (out / "summary_haar_span.json").write_text(json.dumps(summary))
    print(f"wrote {out / 'haar_vs_clifford.png'}")


if __name__ == "__main__":
    main()
