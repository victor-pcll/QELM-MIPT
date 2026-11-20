#!/usr/bin/env python3
"""Minimal parity weight of the effects of the Clifford QELM against p and N.

For each circuit (periodic brickwork of #337, ``T = 2N`` steps, final ``Z``
readout) and each ``p``, compute the exact minimal parity weight ``w(g)`` of
every visible string of the effect group ``G``
(:func:`QELM.evaluation.readout_locality.min_parity_weights`) and store its
joint histogram with the string weight ``|g|``. The gates and the uniform
numbers ``u`` of the mask are drawn once per circuit and shared by every
``p`` (nested masks ``u < p``), so the ``p`` dependence is not blurred by
redrawing.

From the histograms, :mod:`plot` derives ``E_readout(k) / (1 - E_circuit)``
for every ``k`` (flat prior: fraction of the visible strings of ``G`` with
``w > k``; iid prior: the same with weights ``2^{-|g|}``).

Output: ``<out>/locality_N<N>.npz`` (``locality_N<N>_T<T>.npz`` when
``T != 2N``) with ``hist`` of shape
``(n_circuits, n_p, W + 1, N + 1)`` (``hist[c, j, w, s]``: number of visible
strings of ``G`` with minimal parity weight ``w`` and ``|g| = s``; ``w = W``
collects ``w >= W``), plus ``p``, ``n_bits`` and the config.

Example::

    uv run python experiments/readout_locality/run.py --n 8 --circuits 200
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from QELM.evaluation import readout_locality as rl
from QELM.evaluation.clifford_ridge import draw_circuit, with_mask

P_GRID = [round(0.025 * i, 3) for i in range(13)] + [0.35, 0.4, 0.45, 0.5]
W_MAX = 64


def one_circuit(n: int, depth: int, ps: list[float], seed: int, index: int) -> dict:
    """Histograms of one circuit at every ``p``."""
    rng = np.random.default_rng((seed, n, index, 378))
    gates = draw_circuit(n, depth, rng)
    u = rng.random((depth, n))
    vis = rl.visible_codes(n)
    size = rl.string_weights(n)[vis]
    hist = np.zeros((len(ps), W_MAX + 1, n + 1), dtype=np.int32)
    bits = np.zeros(len(ps), dtype=np.int64)
    for j, p in enumerate(ps):
        circ = with_mask(gates, u, p)
        w = rl.min_parity_weights(circ)[vis].astype(np.int64)
        keep = w < rl.INF
        np.add.at(hist[j], (np.minimum(w[keep], W_MAX), size[keep]), 1)
        bits[j] = circ.n_mid + n
    return {"hist": hist, "bits": bits}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--depth", type=int, default=0, help="T (default 2N)")
    ap.add_argument("--circuits", type=int, default=200)
    ap.add_argument("--p", type=float, nargs="+", default=P_GRID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("outputs/readout_locality"))
    args = ap.parse_args(argv)
    n, depth = args.n, args.depth or 2 * args.n
    t0 = time.time()
    ns = [n] * args.circuits
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    mapper = pool.map if pool is not None else map
    try:
        res = list(
            mapper(
                one_circuit,
                ns,
                [depth] * args.circuits,
                [list(args.p)] * args.circuits,
                [args.seed] * args.circuits,
                range(args.circuits),
            )
        )
    finally:
        if pool is not None:
            pool.shutdown()
    hist = np.stack([r["hist"] for r in res])
    bits = np.stack([r["bits"] for r in res])
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "n": n,
        "depth": depth,
        "circuits": args.circuits,
        "p": list(args.p),
        "seed": args.seed,
        "w_max": W_MAX,
        "geometry": "periodic brickwork of #337, Z measurement with probability "
        "p after every step (nested masks u < p), final Z readout",
        "seconds": time.time() - t0,
    }
    np.savez_compressed(
        args.out / f"locality_N{n}{'' if depth == 2 * n else f'_T{depth}'}.npz",
        hist=hist,
        bits=bits,
        p=np.array(args.p),
        config=json.dumps(config),
    )
    print(f"N={n}: {args.circuits} circuits in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
