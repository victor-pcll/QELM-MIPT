#!/usr/bin/env python3
"""Exact ``W(m)`` and ``Delta`` of the Haar MIPT for one size (issue #348).

For each ``p`` of the grid, ``--samples`` independent circuits of the
periodic Haar brickwork (:func:`QELM.evaluation.haar_mipt.haar_record`),
one Born record each, read after ``T = r N`` half-layers for each ratio
``r`` of ``--ratios``. Writes ``runs/haar_N<N>.json`` with, per ``p`` and
per ratio, the mean and standard error of ``Delta`` and of ``W``.

Run from the repository root, for example:

    uv run python experiments/haar_collapse_map/run.py --n 8 --samples 3000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from QELM.evaluation.haar_mipt import haar_record

HERE = Path(__file__).resolve().parent
P_GRID = np.round(np.arange(0.04, 0.361, 0.02), 4)


def _chunk(args: tuple[int, float, tuple[int, ...], int, int, int]) -> np.ndarray:
    """``(samples, 2, len(checkpoints))``: Delta and W of a block of circuits."""
    n, p, checkpoints, seed, first, count = args
    out = np.zeros((count, 2, len(checkpoints)))
    dtype = np.complex64 if n >= 12 else np.complex128
    for s in range(count):
        ss = np.random.SeedSequence([seed, n, round(p * 10_000), first + s])
        r = haar_record(n, p, checkpoints, np.random.default_rng(ss), dtype)
        out[s, 0], out[s, 1] = r.delta, r.weight
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--ratios", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--p", type=float, nargs="+", default=list(P_GRID))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    args = parser.parse_args(argv)

    checkpoints = tuple(r * args.n for r in args.ratios)
    jobs = []
    for p in args.p:
        for first in range(0, args.samples, args.chunk):
            count = min(args.chunk, args.samples - first)
            jobs.append((args.n, float(p), checkpoints, args.seed, first, count))
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        blocks = list(pool.map(_chunk, jobs))
    per_p = []
    for p in args.p:
        data = np.concatenate(
            [b for j, b in zip(jobs, blocks, strict=True) if j[1] == float(p)]
        )
        row: dict[str, object] = {"p": float(p), "samples": int(data.shape[0])}
        for i, r in enumerate(args.ratios):
            for k, name in enumerate(("delta", "weight")):
                x = data[:, k, i]
                err = float(x.std(ddof=1) / np.sqrt(len(x)))
                row[f"{name}_T{r}N"] = [float(x.mean()), err]
        per_p.append(row)
    result = {
        "n": args.n,
        "ratios": args.ratios,
        "seed": args.seed,
        "circuit": "periodic Haar brickwork, Z measurement with probability p "
        "after every half-layer, T = ratio * N half-layers",
        "seconds": round(time.time() - start, 1),
        "per_p": per_p,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"haar_N{args.n}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path} in {result['seconds']} s")


if __name__ == "__main__":
    main()
