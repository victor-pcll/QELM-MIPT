#!/usr/bin/env python3
"""Distribution of the population R^2 budget against p (issue #360).

For each ``p``, draw circuits (gates and measurement mask) and, for each
circuit, random dense targets ``y = sum_a beta_a B_a``; record
``E_circuit``, ``E_readout(k)`` and ``R^2_pop(k)`` for every pair. ``K = inf``,
infinite training set, ``lambda -> 0+``, so ``E_estimation = 0``; targets are
built on the Bloch monomials, so ``E_encoding = 0``.

Subcommands:

- ``dense``: open brickwork of :mod:`QELM.evaluation.fourier_framework`,
  ``N = 4``, ``T = N``, exact projections. Each mask is drawn once and used
  with Haar gates and with Clifford gates, so the two columns of the figure
  see the same masks and the same targets. Masks with more than
  ``--max-bits`` record bits are redrawn (the count is stored).
- ``clifford``: periodic brickwork of #337, ``T = 2N``, any ``N <= 12``, by
  counting strings in ``G`` and ``D_k``.

Output: ``<out>/<name>.npz`` (every sample) and ``<out>/<name>.json`` (config,
per-``p`` histograms on ``--bins`` bins of ``[0, 1]``, means, quantiles,
largest cross term).

Example::

    uv run python experiments/r2_budget/run.py dense --circuits 300 --targets 40
    uv run python experiments/r2_budget/run.py clifford --n 8 --circuits 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb
from QELM.evaluation.clifford_ridge import draw_circuit, with_mask

P_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def summarise(values: np.ndarray, bins: int) -> dict:
    """Histogram on ``[0, 1]``, mean, std and quantiles of a flat sample."""
    v = np.clip(np.asarray(values, dtype=float).ravel(), 0.0, 1.0)
    counts, _ = np.histogram(v, bins=bins, range=(0.0, 1.0))
    return {
        "counts": counts.tolist(),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "quantiles": np.quantile(v, QUANTILES).tolist(),
    }


def stack(budgets: list[rb.Budget]) -> dict[str, np.ndarray]:
    """Concatenate the targets of several circuits: ``(n_k, n_samples)`` arrays."""
    return {
        "circuit": np.concatenate([b.circuit for b in budgets]),
        "readout": np.concatenate([b.readout for b in budgets], axis=1),
        "r2": np.concatenate([b.r2 for b in budgets], axis=1),
        "cross": np.concatenate([b.cross for b in budgets], axis=1),
    }


def describe(arr: dict[str, np.ndarray], ks: list[int], bins: int) -> dict:
    total = arr["circuit"][None] + arr["readout"] + arr["r2"]
    return {
        "circuit": summarise(arr["circuit"], bins),
        "readout": {
            str(k): summarise(arr["readout"][j], bins) for j, k in enumerate(ks)
        },
        "r2": {str(k): summarise(arr["r2"][j], bins) for j, k in enumerate(ks)},
        "max_abs_cross": float(np.abs(arr["cross"]).max()),
        "max_abs_sum_minus_one": float(np.abs(total - 1).max()),
    }


def run_dense(args: argparse.Namespace) -> None:
    n, depth, ks = args.n, args.depth or args.n, list(args.k)
    out: dict = {
        "route": "dense",
        "n": n,
        "depth": depth,
        "k": ks,
        "prior": args.prior,
        "circuits": args.circuits,
        "targets": args.targets,
        "seed": args.seed,
        "max_bits": args.max_bits,
        "bins": args.bins,
        "p": args.p,
        "geometry": "open brickwork, layer = even bonds, odd bonds, then Z "
        "measurement of each site with probability p; final Z readout",
        "per_p": {"haar": [], "clifford": []},
    }
    raw: dict[str, np.ndarray] = {}
    for j, p in enumerate(args.p):
        t0 = time.time()
        rng = np.random.default_rng((args.seed, n, j, 360))
        res: dict[str, list[rb.Budget]] = {"haar": [], "clifford": []}
        redrawn = 0
        bits = []
        for _ in range(args.circuits):
            haar = ff.draw_circuit(n, depth, p, "haar", rng)
            while ff.n_bits(haar) > args.max_bits:
                redrawn += 1
                haar = ff.with_mask(haar, rng.random((depth, n)) < p)
            cliff = ff.with_mask(
                ff.draw_circuit(n, depth, p, "clifford", rng), haar.mask
            )
            bits.append(ff.n_bits(haar))
            g = rb.draw_whitened_targets(n, args.targets, args.prior, rng)
            for kind, circ in (("haar", haar), ("clifford", cliff)):
                res[kind].append(rb.dense_budget(circ, tuple(ks), g, args.max_bits))
        for kind, budgets in res.items():
            arr = stack(budgets)
            d = describe(arr, ks, args.bins)
            d.update(p=p, redrawn_masks=redrawn, mean_bits=float(np.mean(bits)))
            out["per_p"][kind].append(d)
            for name, a in arr.items():
                raw[f"{kind}_{name}_{j}"] = a.astype(np.float32)
        h, c = out["per_p"]["haar"][-1], out["per_p"]["clifford"][-1]
        print(
            f"dense N={n} p={p:.2f} ({time.time() - t0:.0f} s, {redrawn} redrawn): "
            f"E_circ haar {h['circuit']['mean']:.3f} clifford "
            f"{c['circuit']['mean']:.3f}; max|cross| "
            f"{max(h['max_abs_cross'], c['max_abs_cross']):.1e}",
            flush=True,
        )
    save(args, out, raw, f"dense_N{n}_{args.prior}")


def run_clifford(args: argparse.Namespace) -> None:
    n, depth, ks = args.n, args.depth or 2 * args.n, list(args.k)
    out: dict = {
        "route": "clifford",
        "n": n,
        "depth": depth,
        "k": ks,
        "prior": args.prior,
        "circuits": args.circuits,
        "targets": args.targets,
        "seed": args.seed,
        "bins": args.bins,
        "p": args.p,
        "geometry": "periodic brickwork of #337, one half-layer per step, Z "
        "measurement with probability p after every step; final Z readout",
        "per_p": {"clifford": []},
    }
    raw: dict[str, np.ndarray] = {}
    for j, p in enumerate(args.p):
        t0 = time.time()
        rng = np.random.default_rng((args.seed, n, j, 3600))
        budgets = []
        group_frac = []
        for _ in range(args.circuits):
            circ = with_mask(draw_circuit(n, depth, rng), rng.random((depth, n)), p)
            q = rb.draw_power(n, args.targets, args.prior, rng)
            in_g, in_d = rb.clifford_masks(circ, tuple(ks))
            group_frac.append(in_g.sum() / (3**n - 1))
            budgets.append(rb.counting_budget(in_g, in_d, q))
        arr = stack(budgets)
        d = describe(arr, ks, args.bins)
        d.update(p=p, visible_group_fraction=float(np.mean(group_frac)))
        out["per_p"]["clifford"].append(d)
        for name, a in arr.items():
            raw[f"clifford_{name}_{j}"] = a.astype(np.float32)
        print(
            f"clifford N={n} p={p:.2f} ({time.time() - t0:.0f} s): E_circ "
            f"{d['circuit']['mean']:.3f}, R2(k={ks[-1]}) "
            f"{d['r2'][str(ks[-1])]['mean']:.4f}",
            flush=True,
        )
    save(args, out, raw, f"clifford_N{n}_{args.prior}")


def save(args: argparse.Namespace, out: dict, raw: dict, name: str) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{name}.npz", **raw)  # type: ignore[arg-type]
    (args.out / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out / name}.json/.npz")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="route", required=True)
    for name, n_default, circ_default in (("dense", 4, 300), ("clifford", 8, 200)):
        s = sub.add_parser(name)
        s.add_argument("--n", type=int, default=n_default)
        s.add_argument("--depth", type=int, default=None)
        s.add_argument("--k", type=int, nargs="+", default=[1, 2])
        s.add_argument("--p", type=float, nargs="+", default=P_GRID)
        s.add_argument("--prior", choices=rb.PRIORS, default="iid")
        s.add_argument("--circuits", type=int, default=circ_default)
        s.add_argument("--targets", type=int, default=40)
        s.add_argument("--bins", type=int, default=40)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out", type=Path, default=Path("outputs/r2_budget"))
        if name == "dense":
            s.add_argument("--max-bits", type=int, default=16)
    args = parser.parse_args(argv)
    {"dense": run_dense, "clifford": run_clifford}[args.route](args)


if __name__ == "__main__":
    main()
