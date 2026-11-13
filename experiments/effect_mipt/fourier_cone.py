"""Fourier cone: spatial range of the Fourier modes of a record against time (#327).

Dense encoding, Clifford, literature circuit (periodic brickwork, ``Z``
measurement of each site with probability ``p`` after every half-layer).
For a record of ``t`` half-layers, the Fourier spectrum of ``x -> Pr(m | x)``
is flat on the stabilizer group ``S_m`` of the effect
(:func:`~QELM.evaluation.effect_mipt.effect_stabilizers`). For an interval
``A`` of the input, ``1 + W_A = 2^(k_A)`` is the weight of the modes
supported in ``A`` (:func:`~QELM.evaluation.effect_mipt.interval_counts`).
``k(l, t)`` is averaged over ``starts`` random positions of ``A`` and over
trajectories (one fresh circuit each).

Usage::

    uv run python experiments/effect_mipt/fourier_cone.py --n 64 --p 0.12 \
        --samples 300 --out outputs/effect_mipt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_mipt import (
    draw_periodic_half_layers,
    effect_stabilizers,
    interval_counts,
)


def run(n: int, p: float, samples: int, seed: int, every: int, starts: int) -> dict:
    """Mean and standard error of ``k(l, t)`` for ``t = every, 2 every, ..., 2N``."""
    times = list(range(every, 2 * n + 1, every))
    rng = np.random.default_rng((seed, n, int(round(1000 * p)), 41))
    k = np.zeros((samples, len(times), n + 1))
    start = time.perf_counter()
    for s in range(samples):
        circuit = draw_periodic_half_layers(n, p, 2 * n, rng)
        for i, t in enumerate(times):
            gens = effect_stabilizers(circuit, t)
            positions = rng.integers(n, size=starts)
            k[s, i] = np.mean([interval_counts(gens, n, int(a)) for a in positions], 0)
    print(f"N={n} p={p}: {time.perf_counter() - start:.0f} s", flush=True)
    return {
        "n": n,
        "p": p,
        "samples": samples,
        "seed": seed,
        "starts": starts,
        "times": times,
        "k_mean": k.mean(axis=0).tolist(),
        "k_sem": (k.std(axis=0, ddof=1) / np.sqrt(samples)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/effect_mipt"))
    args = parser.parse_args()
    result = run(args.n, args.p, args.samples, args.seed, args.every, args.starts)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"cone_N{args.n}_p{args.p:.2f}_s{args.seed}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
