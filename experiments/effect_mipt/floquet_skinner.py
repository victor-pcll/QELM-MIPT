"""Dense Fourier weight and half-chain S_2, Floquet circuit of Skinner et al. (#333).

Skinner, Ruhman & Nahum, PRX 9, 031009 (2019), Eq. 1: open brickwork of one
fixed Floquet Ising brick, ``Z`` measurement of each spin with probability
``p`` after every layer of bricks (half a period). For each measurement
pattern (:func:`~QELM.evaluation.effect_mipt.floquet_record`):

- ``Delta = N - log2(1 + W(m))`` of the dense encoding, exact Kraus operator,
  record drawn by the Born rule of the maximally mixed input;
- ``S_2`` of the left half of the pure-state trajectory from ``|0...0>``.

The file is rewritten after every ``p``. With ``--pure-only``, only the pure
trajectory is run (cheap, larger ``N``) and ``S_2`` is the mean of the two
central cuts, which removes the even/odd-bond parity effect between sizes.

Usage::

    uv run python experiments/effect_mipt/floquet_skinner.py --n 8 --samples 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_mipt import (
    floquet_half_chain_s2,
    floquet_record,
    skinner_floquet_brick,
)

P_GRID = [0.05, 0.10, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.35, 0.40, 0.50]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+", default=P_GRID)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--out", type=Path, default=Path("outputs/effect_mipt"))
    parser.add_argument(
        "--pure-only",
        action="store_true",
        help="only the pure trajectory, S_2 at both central cuts (large N)",
    )
    args = parser.parse_args()
    n = args.n
    checkpoints = [r * n for r in args.ratios]
    brick = skinner_floquet_brick()
    out: dict = {"n": n, "samples": args.samples, "seed": args.seed}
    out.update(checkpoints=checkpoints, p=[], rows=[])
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "floquet_pure" if args.pure_only else "floquet"
    path = args.out / f"{tag}_N{n}_s{args.seed}.json"
    for j, p in enumerate(args.p):
        start = time.perf_counter()
        rng = np.random.default_rng((args.seed, n, j, 51))
        delta = np.zeros((args.samples, len(checkpoints)))
        s2 = np.zeros_like(delta)
        for s in range(args.samples):
            masks = rng.random((checkpoints[-1], n)) < p
            if args.pure_only:
                cuts = floquet_half_chain_s2(n, masks, brick, rng, checkpoints)
                delta[s], s2[s] = np.nan, cuts.mean(axis=1)
                continue
            record = floquet_record(n, masks, brick, rng, checkpoints)
            delta[s], s2[s] = record.delta, record.s2_half
        root = np.sqrt(args.samples)
        out["p"].append(p)
        out["rows"].append(
            {
                "p": p,
                "delta_mean": delta.mean(0).tolist(),
                "delta_sem": (delta.std(0, ddof=1) / root).tolist(),
                "s2_mean": s2.mean(0).tolist(),
                "s2_sem": (s2.std(0, ddof=1) / root).tolist(),
            }
        )
        path.write_text(json.dumps(out))
        dt = time.perf_counter() - start
        print(
            f"N={n} p={p}: Delta = {np.round(delta.mean(0), 2)}, "
            f"S2 = {np.round(s2.mean(0), 2)} ({dt:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
