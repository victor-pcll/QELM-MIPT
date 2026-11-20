#!/usr/bin/env python3
"""Relative readout error of Haar and Clifford circuits at small N (issue #386).

Open brickwork of :mod:`QELM.evaluation.fourier_framework`, ``T = 2N``,
final ``Z`` readout, XZ inputs. For each circuit the gates (Haar, and
independently Clifford) and the uniform numbers ``u`` of the mask are drawn
once and shared by every ``p`` (nested masks ``u < p``). The Haar and Clifford
circuits share the mask.

For each circuit and ``p``, with ``P`` the projector on the full feature span
(``Pi_C``) and ``P_k`` on the parities of at most ``k`` bits (whitened
coordinates, as in :mod:`QELM.evaluation.r2_budget`), the mean over targets of
the budget terms is a trace:

- flat prior (isotropic ``g``): ``E[E_readout(k)] / E[1 - E_circuit] =
  (rank P - rank P_k) / rank P``;
- iid prior (``g_a ~ sqrt(w_a) N(0, 1)``): the same with ``sum_a w_a [.]_aa``.

Output: ``<out>/haar_N<N>.npz`` with arrays ``rank`` and ``wtr`` of shape
``(2, n_circuits, n_p, k_max + 1)`` (kinds ``haar, clifford``; last index
``k = 1 .. k_max``, then the full span), and ``bits`` ``(n_circuits, n_p)``.

Example::

    uv run python experiments/readout_locality/haar.py --n 4 --circuits 100
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb

KINDS = ("haar", "clifford")
P_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def span_stats(rows: np.ndarray | None, w: np.ndarray) -> tuple[int, float]:
    """Rank and ``sum_a w_a [P]_aa`` of the whitened span of ``rows``."""
    if rows is None or rows.shape[0] == 0:
        return 0, 0.0
    v = rb.span_basis(rows, w)
    return int(v.shape[0]), float(np.sum(w[1:] * np.sum(v**2, axis=0)))


def one_circuit(n: int, depth: int, ps: list[float], k_max: int, seed: int, i: int):
    rng = np.random.default_rng((seed, n, i, 3861))
    base = {kind: ff.draw_circuit(n, depth, 0.0, kind, rng) for kind in KINDS}
    u = rng.random((depth, n))
    w = ff.input_weights(n, "xz")
    rank = np.zeros((2, len(ps), k_max + 1), dtype=np.int64)
    wtr = np.zeros((2, len(ps), k_max + 1))
    bits = np.zeros(len(ps), dtype=np.int64)
    for j, p in enumerate(ps):
        for a, kind in enumerate(KINDS):
            circ = ff.with_mask(base[kind], u < p)
            nb = ff.n_bits(circ)
            bits[j] = nb
            subsets = ff.parity_subsets(nb, k_max)
            rows = rb.parity_operator_rows(circ, subsets)
            sizes = np.array([len(s) for s in subsets])
            for k in range(1, k_max + 1):
                rank[a, j, k - 1], wtr[a, j, k - 1] = span_stats(rows[sizes <= k], w)
            rank[a, j, k_max], wtr[a, j, k_max] = span_stats(rb.effect_basis(circ), w)
    return rank, wtr, bits


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--depth", type=int, default=0, help="T (default 2N)")
    ap.add_argument("--circuits", type=int, default=100)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--p", type=float, nargs="+", default=P_GRID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("outputs/readout_locality"))
    args = ap.parse_args(argv)
    n, depth = args.n, args.depth or 2 * args.n
    t0 = time.time()
    c = args.circuits
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    mapper = pool.map if pool is not None else map
    try:
        res = list(
            mapper(
                one_circuit,
                [n] * c,
                [depth] * c,
                [list(args.p)] * c,
                [args.k_max] * c,
                [args.seed] * c,
                range(c),
            )
        )
    finally:
        if pool is not None:
            pool.shutdown()
    rank = np.stack([r[0] for r in res], axis=1)
    wtr = np.stack([r[1] for r in res], axis=1)
    bits = np.stack([r[2] for r in res])
    config = {
        "n": n,
        "depth": depth,
        "circuits": c,
        "k_max": args.k_max,
        "p": list(args.p),
        "seed": args.seed,
        "geometry": "open brickwork of fourier_framework, nested masks u < p, "
        "final Z readout, XZ inputs; Haar and Clifford share the mask",
        "seconds": time.time() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"haar_N{n}.npz",
        rank=rank,
        wtr=wtr,
        bits=bits,
        p=np.array(args.p),
        config=json.dumps(config),
    )
    print(f"haar N={n}: {c} circuits in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
