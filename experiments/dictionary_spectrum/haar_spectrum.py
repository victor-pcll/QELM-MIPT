#!/usr/bin/env python3
"""Spectrum of the Haar effect dictionary across the MIPT (issue #356).

Geometry of issue #334: open brickwork, one layer = even bonds, odd bonds,
then a ``Z`` measurement of each site with probability ``p``; ``T = 2N``;
``p_c = 0.29`` for Haar gates in this convention (#322). For each seed, one
Haar circuit, one Clifford circuit and one uniform draw ``u`` (nested masks
``u < p``, shared by the two classes, as in ``clifford_ridge/haar_span.py``).

For each ``p`` and each record (mid-circuit only, full):

- Born-drawn records give ``F = E_m[v v^T]`` (:mod:`QELM.evaluation.ridge_spectrum`),
  its sorted eigenvalues ``mu_k`` and the Born profile ``N_F(l)``;
- the exact span of the effects gives its rank and the span profile
  ``P(l)`` (:mod:`QELM.evaluation.dictionary_spectrum`).

For the Clifford circuit the exact spectrum of ``F`` is ``2^-l`` with
multiplicity ``P(l)``, which checks the sampled one.

Example::

    uv run python experiments/dictionary_spectrum/haar_spectrum.py --n 6 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.dictionary_spectrum import (
    born_profile,
    effective_ranks,
    span_profile,
)
from QELM.evaluation.effect_rank import GATES, brick_pairs
from QELM.evaluation.ridge_spectrum import (
    conditional_means,
    effect_span,
    record_spectrum,
    sample_effect,
)

P_GRID = [0.04, 0.1, 0.16, 0.22, 0.3, 0.4, 0.5]
THRESHOLDS = [1e-2, 1e-3, 1e-4]


def draw_gates(kind: str, n: int, depth: int, rng: np.random.Generator) -> list:
    sampler = GATES[kind]
    return [[sampler(rng) for _ in brick_pairs(n)] for _ in range(depth)]


def one_point(
    gates: list, masks: np.ndarray, final: bool, records: int, rng: np.random.Generator
) -> dict:
    n = masks.shape[1]
    v = np.array(
        [
            conditional_means(sample_effect(gates, masks, final, rng))
            for _ in range(records)
        ]
    )
    mu = record_spectrum(v).mu
    span = effect_span(gates, masks, final)
    return {
        "mu": mu.tolist(),
        "span_rank": int(span.shape[0]),
        "span_profile": span_profile(span, n).tolist(),
        "born_profile": born_profile(v, n).tolist(),
        "trace_F": float(mu.sum()),
        "ranks": effective_ranks(mu, THRESHOLDS),
    }


def run(n: int, seed: int, p_grid: list[float], records: int) -> dict:
    depth = 2 * n
    u = np.random.default_rng((seed, n, 1000)).random((depth, n))
    out: dict = {
        "n": n,
        "depth": depth,
        "seed": seed,
        "records": records,
        "p": p_grid,
        "thresholds": THRESHOLDS,
        "per_kind": {},
    }
    for j, kind in enumerate(("haar", "clifford")):
        gates = draw_gates(kind, n, depth, np.random.default_rng((seed, n, 0, j)))
        rows = []
        for p in p_grid:
            t0 = time.time()
            masks = u < p
            row: dict = {"p": p}
            for variant, final in (("mid", False), ("full", True)):
                rng = np.random.default_rng((seed, n, 3000, j, round(1000 * p), final))
                row[variant] = one_point(gates, masks, final, records, rng)
            rows.append(row)
            f = row["full"]
            print(
                f"{kind} N={n} seed={seed} p={p}: rank={f['span_rank']} "
                f"trF={f['trace_F']:.3f} PR={f['ranks']['participation']:.1f} "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
        out["per_kind"][kind] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--records", type=int, default=None)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/dictionary_spectrum"))
    args = parser.parse_args()
    records = args.records or max(2000, 8 * 3**args.n)
    result = run(args.n, args.seed, args.p or P_GRID, records)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"haar_spectrum_N{args.n}_seed{args.seed}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
