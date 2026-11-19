#!/usr/bin/env python3
"""Runs without final readout: Delta, S_2(N/2), I3, S_2(one) per sample (#378).

For each ``p``, ``--samples`` independent circuits of the periodic brickwork
(Haar or Clifford gates, ``Z`` measurement of each site with probability ``p``
after every half-layer, no final readout), read after ``T = r N``
half-layers for each ``r`` of ``--ratios``. Every observable is an ``S_2`` of
the references, read from the Fourier weights ``W_A`` of the effect
(:mod:`QELM.evaluation.no_readout`).

Writes ``runs/<model>_N<N>.json`` after each ``p``: the per-sample values,
and their mean and standard error. A new launch with the same arguments
skips the ``p`` already written (resume after an interruption).

Run from the repository root, for example:

    uv run python experiments/pc_nu_no_readout/run.py --model clifford --n 64
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from QELM.evaluation.no_readout import (  # noqa: E402
    OBSERVABLES,
    clifford_sample,
    haar_sample,
)

HERE = Path(__file__).resolve().parent
P_GRID = {
    "haar": [0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.27,
             0.30],
    "clifford": [0.10, 0.12, 0.13, 0.14, 0.15, 0.155, 0.16, 0.165, 0.17, 0.18, 0.19,
                 0.20, 0.22, 0.24],
}  # fmt: skip
SAMPLES = {
    "haar": {4: 8000, 6: 8000, 8: 6000, 10: 2000, 12: 300},
    "clifford": {32: 4000, 64: 2000, 128: 1000, 256: 500},
}
MODEL_ID = {"haar": 0, "clifford": 1}


def _chunk(args: tuple[str, int, float, tuple[int, ...], int, int, int]) -> np.ndarray:
    """``(count, len(checkpoints), 4)``: the observables of a block of circuits."""
    model, n, p, checkpoints, seed, first, count = args
    out = np.zeros((count, len(checkpoints), len(OBSERVABLES)))
    for s in range(count):
        ss = np.random.SeedSequence(
            [seed, MODEL_ID[model], n, round(p * 10_000), first + s]
        )
        rng = np.random.default_rng(ss)
        if model == "haar":
            dtype = np.complex64 if n >= 12 else np.complex128
            out[s] = haar_sample(n, p, checkpoints, rng, dtype)
        else:
            out[s] = clifford_sample(n, p, checkpoints, rng)
    return out


def summarise(data: np.ndarray, ratios: list[int]) -> dict[str, object]:
    """Per-sample values and ``[mean, sem]`` of each observable and ratio."""
    row: dict[str, object] = {}
    for i, r in enumerate(ratios):
        for k, name in enumerate(OBSERVABLES):
            x = data[:, i, k]
            key = f"{name}_T{r}N"
            if np.all(np.isnan(x)):
                continue
            row[key] = [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]
            row[f"samples_{key}"] = [round(float(v), 6) for v in x]
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(P_GRID), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--ratios", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    args = parser.parse_args(argv)

    samples = args.samples or SAMPLES[args.model][args.n]
    grid = args.p or P_GRID[args.model]
    chunk = args.chunk or max(1, samples // (4 * args.workers))
    checkpoints = tuple(r * args.n for r in args.ratios)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.model}_N{args.n}.json"
    result: dict[str, object] = {
        "model": args.model,
        "n": args.n,
        "ratios": args.ratios,
        "seed": args.seed,
        "observables": list(OBSERVABLES),
        "circuit": f"periodic {args.model} brickwork, Z measurement with "
        "probability p after every half-layer, no final readout, T = ratio * N "
        "half-layers; S_2 of the references from the Fourier weights W_A",
        "per_p": [],
    }
    if path.exists():
        old = json.loads(path.read_text())
        same = all(old[k] == result[k] for k in ("model", "n", "ratios", "seed"))
        if not same:
            raise SystemExit(f"{path} holds another configuration; use --out.")
        result["per_p"] = old["per_p"]
    per_p: list[dict[str, object]] = result["per_p"]  # type: ignore[assignment]
    done = {round(float(r["p"]), 6) for r in per_p if r["samples"] == samples}

    with ProcessPoolExecutor(args.workers) as pool:
        for p in grid:
            if round(p, 6) in done:
                print(f"p = {p}: already written, skipped")
                continue
            jobs = [
                (args.model, args.n, float(p), checkpoints, args.seed, first,
                 min(chunk, samples - first))
                for first in range(0, samples, chunk)
            ]  # fmt: skip
            start = time.time()
            data = np.concatenate(list(pool.map(_chunk, jobs)))
            row: dict[str, object] = {
                "p": float(p),
                "samples": int(data.shape[0]),
                "seconds": round(time.time() - start, 1),
            }
            row.update(summarise(data, args.ratios))
            per_p[:] = [r for r in per_p if round(float(r["p"]), 6) != round(p, 6)]
            per_p.append(row)
            per_p.sort(key=lambda r: float(r["p"]))  # type: ignore[arg-type]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result) + "\n")
            tmp.replace(path)
            print(f"p = {p}: {row['seconds']} s, wrote {path}", flush=True)


if __name__ == "__main__":
    main()
