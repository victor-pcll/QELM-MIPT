#!/usr/bin/env python3
"""Locality of the dictionary of the Clifford ridge at K = infinity (issue #339).

Reads ``locality_N*_T*_s*[_tag].json`` (chunks merged per ``N``) and writes
``figures/dictionary_locality.png`` and ``figures/summary_locality.json``.

For the full record, ``2^(l - S(l))`` strings of the group of the effect
``G`` fit in an interval of ``l`` sites, where ``S(l)`` is the entropy of
``l`` references. In the clipped gauge of the stabilizer generators,
``S(l) = (1/2) sum_g min(l, l_g)`` on average, so the density of generators
of length ``l`` (per site) is ``rho(l) = -2 S''(l)``: each generator is one
representable function, of extent ``l_g``.

Usage::

    uv run python experiments/clifford_ridge/plot_locality.py outputs/clifford_ridge
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

P_C = 0.161
PICKS = (0.04, 0.1, 0.14, 0.16, 0.18, 0.22, 0.3, 0.5)


def load(folder: Path) -> dict:
    """``{N: {p: (S, S_sem)}}`` merged over the chunks, ``T = 2N`` only."""
    out: dict = defaultdict(dict)
    for path in sorted(folder.glob("locality_N*_T*_s*.json")):
        d = json.loads(path.read_text())
        if d["t"] != 2 * d["n"]:
            continue
        for row in d["per_p"]:
            out[d["n"]][row["p"]] = (np.asarray(row["s"]), np.asarray(row["s_sem"]))
    return dict(out)


def generator_density(s: np.ndarray) -> np.ndarray:
    """``rho(l) = -2 S''(l)`` for ``l = 1 .. N/2 - 1`` (per site)."""
    n = s.size - 1
    half = s[: n // 2 + 1]
    return np.asarray(-2 * (half[2:] - 2 * half[1:-1] + half[:-2]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--p-c", type=float, default=P_C)
    args = parser.parse_args(argv)
    data = load(args.folder)
    if not data:
        raise SystemExit("No locality run with T = 2N.")
    out = args.folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    p_c = args.p_c
    n_max = max(data)
    big = data[n_max]
    ps = sorted(big)
    picks = [p for p in PICKS if p in big]
    cols = [plt.get_cmap("coolwarm")(x) for x in np.linspace(0.05, 0.95, len(picks))]
    summary: dict = {"N": n_max, "p_c": p_c}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) map of the representable fraction 1 - S(l)/l over (p, l/N).
    ax = axes[0, 0]
    ell = np.arange(1, n_max // 2 + 1)
    frac = np.array([1 - big[p][0][1 : n_max // 2 + 1] / ell for p in ps]).T
    mesh = ax.pcolormesh(
        ps, ell, frac, cmap="viridis", vmin=0, vmax=1, shading="nearest"
    )
    cs = ax.contour(ps, ell, frac, levels=[0.25, 0.5, 0.75, 0.9], colors="w",
                    linewidths=0.8)  # fmt: skip
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    ax.set_yscale("log")
    ax.axvline(p_c, color="w", ls="--", lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("window l (sites)")
    ax.set_title(
        f"N = {n_max}: k(l)/l = 1 - S(l)/l, log2 of the representable\n"
        "functions in a window, per site", fontsize=10
    )  # fmt: skip
    fig.colorbar(mesh, ax=ax)

    # (b) S(l) against l.
    ax = axes[0, 1]
    summary["S"] = {}
    for p, c in zip(picks, cols, strict=True):
        s = big[p][0][1 : n_max // 2 + 1]
        ax.loglog(ell, s, color=c, label=f"p={p}")
        summary["S"][str(p)] = s.tolist()
    ax.loglog(ell, ell / 2, color="0.5", ls=":", label="l / 2 (volume law)")
    ax.loglog(ell, np.log2(ell + 1), color="0.5", ls="--", label="log2 l")
    ax.set_xlabel("window l")
    ax.set_ylabel("S(l): input bits of the window not representable locally")
    ax.set_title(f"N = {n_max}: entropy profile of the references", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # (c) length distribution of the generators.
    ax = axes[1, 0]
    summary["rho"] = {}
    lmid = np.arange(1, n_max // 2)
    for p, c in zip(picks, cols, strict=True):
        rho = generator_density(big[p][0])
        keep = rho > 1e-4
        ax.loglog(lmid[keep], rho[keep], color=c, marker="o", ms=2, lw=1)
        summary["rho"][str(p)] = rho.tolist()
    ax.loglog(lmid, 2.0 * lmid**-2.0, color="0.4", ls="--", label="2 l^-2")
    ax.set_xlabel("generator length l")
    ax.set_ylabel("rho(l) = -2 S''(l), generators of length l per site")
    ax.set_title("Extent of the representable functions (clipped gauge)", fontsize=10)
    ax.legend(fontsize=8)

    # (d) S(N/2) against N.
    ax = axes[1, 1]
    ns = sorted(data)
    summary["s_half"] = {}
    for p, c in zip(picks, cols, strict=True):
        if not all(p in data[n] for n in ns):
            continue
        sh = np.array([data[n][p][0][n // 2] for n in ns])
        eh = np.array([data[n][p][1][n // 2] for n in ns])
        ax.errorbar(ns, sh, eh, color=c, marker="o", ms=3, label=f"p={p}")
        summary["s_half"][str(p)] = sh.tolist()
    nn = np.array(ns, dtype=float)
    ax.plot(nn, nn / 8, color="0.5", ls=":", label="slope 1")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("S(N/2)")
    ax.set_title("Volume law below p_c, area law above", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out / "dictionary_locality.png", dpi=150)
    plt.close(fig)
    summary["N_list"] = ns
    (out / "summary_locality.json").write_text(json.dumps(summary))
    print(f"wrote {out / 'dictionary_locality.png'}")


if __name__ == "__main__":
    main()
