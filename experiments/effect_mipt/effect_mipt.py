"""Reference MIPT probes and Born-weighted frame dimension, convention of #314.

Three probes, on open brickworks with the layer convention of
:mod:`QELM.evaluation.effect_rank` (issue #322):

- ``clifford``: entropy of an initially maximally mixed system and of a
  reference qubit, Clifford gates, ``T = 4N`` measured layers, one fresh
  circuit and mask per sample (:func:`clifford_purification`);
- ``haar``: entropy of a reference qubit, Haar gates, ``N`` unmeasured layers
  then ``T = 2N`` measured layers, one fresh circuit, mask and trajectory per
  sample (:func:`haar_reference_entropy`);
- ``frame``: effective dimension ``D_2 = 1 / Phi_2`` of the effects, one
  circuit and one mask per seed with the seeds and nested masks of
  ``experiments/effect_rank/effect_rank.py``, so that the rank of the same
  circuit is known (:func:`frame_potential`).

Usage::

    uv run python experiments/effect_mipt/effect_mipt.py clifford --n 32 \
        --samples 100 --out outputs/effect_mipt
    uv run python experiments/effect_mipt/effect_mipt.py frame --n 8 \
        --seeds 0 1 2 --samples 600 --out outputs/effect_mipt

Each call writes one JSON file named after the probe, ``N`` and the seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_mipt import (
    clifford_purification,
    draw_haar_circuit,
    fourier_weight,
    frame_potential,
    haar_reference_entropy,
    sample_kraus,
)
from QELM.evaluation.effect_rank import mincut_bits

P_GRID = [round(0.05 * i, 2) for i in range(1, 13)]


def run_clifford(n: int, ps: list[float], samples: int, seed: int, ratio: int) -> dict:
    """Mixed-state and reference entropies per layer, mean and error."""
    depth = ratio * n
    out: dict = {"probe": "clifford", "n": n, "depth": depth, "p": ps, "rows": []}
    for j, p in enumerate(ps):
        rng = np.random.default_rng((seed, n, j, 11))
        sys_, ref = [], []
        for _ in range(samples):
            masks = rng.random((depth, n)) < p
            r = clifford_purification(n, masks, rng)
            sys_.append(r.system)
            ref.append(r.reference)
        s, q = np.array(sys_), np.array(ref)
        out["rows"].append(
            {
                "p": p,
                "samples": samples,
                "system_mean": s.mean(0).tolist(),
                "system_sem": (s.std(0, ddof=1) / np.sqrt(samples)).tolist(),
                "reference_mean": q.mean(0).tolist(),
                "reference_sem": (q.std(0, ddof=1) / np.sqrt(samples)).tolist(),
            }
        )
        print(f"clifford N={n} p={p}", flush=True)
    return out


def run_haar(n: int, ps: list[float], samples: int, seed: int, ratio: int) -> dict:
    """Reference entropy per layer, mean and error."""
    depth = ratio * n
    out: dict = {
        "probe": "haar",
        "n": n,
        "depth": depth,
        "scramble": n,
        "p": ps,
        "rows": [],
    }
    for j, p in enumerate(ps):
        rng = np.random.default_rng((seed, n, j, 12))
        q = np.array(
            [
                haar_reference_entropy(n, rng.random((depth, n)) < p, rng)
                for _ in range(samples)
            ]
        )
        out["rows"].append(
            {
                "p": p,
                "samples": samples,
                "reference_mean": q.mean(0).tolist(),
                "reference_sem": (q.std(0, ddof=1) / np.sqrt(samples)).tolist(),
            }
        )
        print(f"haar N={n} p={p}", flush=True)
    return out


def run_frame(
    n: int, ps: list[float], samples: int, seeds: list[int], ratio: int
) -> dict:
    """``log2 D_2`` and the min-cut rank for each seed and ``p``."""
    depth = ratio * n
    out: dict = {
        "probe": "frame",
        "n": n,
        "depth": depth,
        "p": ps,
        "samples": samples,
        "runs": [],
    }
    for seed in seeds:
        u = np.random.default_rng((seed, n, 1000)).random((depth, n))
        gates = draw_haar_circuit(n, depth, np.random.default_rng((seed, n, 0)))
        rng = np.random.default_rng((seed, n, 13))
        run: dict = {"seed": seed, "log2_d2": [], "sem_bits": [], "cut": [], "k": []}
        for p in ps:
            start = time.perf_counter()
            fp = frame_potential(gates, u < p, samples, rng)
            run["log2_d2"].append(fp.log2_d2)
            run["sem_bits"].append(fp.sem / fp.phi2 / np.log(2))
            run["cut"].append(mincut_bits(u < p))
            run["k"].append(int((u[:-1] < p).sum()))
            print(
                f"frame N={n} seed={seed} p={p}: {time.perf_counter() - start:.1f}s",
                flush=True,
            )
        out["runs"].append(run)
    return out


def run_weight(
    n: int, ps: list[float], samples: int, seeds: list[int], ratio: int
) -> dict:
    """Per-record Fourier weight, in bits, for each seed and ``p``.

    For each circuit and ``p``, ``samples`` mid-circuit records are drawn;
    the run stores the mean over records of ``log2(1 + W)`` for the dense
    and the angle encodings, and the min cut of the mask.
    """
    depth = ratio * n
    out: dict = {
        "probe": "weight",
        "n": n,
        "depth": depth,
        "p": ps,
        "samples": samples,
        "runs": [],
    }
    for seed in seeds:
        u = np.random.default_rng((seed, n, 1000)).random((depth, n))
        gates = draw_haar_circuit(n, depth, np.random.default_rng((seed, n, 0)))
        rng = np.random.default_rng((seed, n, 14))
        run: dict = {"seed": seed, "dense_bits": [], "angle_bits": [], "cut": []}
        for p in ps:
            w = [
                fourier_weight(sample_kraus(gates, u < p, rng)) for _ in range(samples)
            ]
            run["dense_bits"].append(float(np.mean([np.log2(1 + x.dense) for x in w])))
            run["angle_bits"].append(float(np.mean([np.log2(1 + x.angle) for x in w])))
            run["cut"].append(mincut_bits(u < p))
            print(f"weight N={n} T={depth} seed={seed} p={p}", flush=True)
        out["runs"].append(run)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", choices=("clifford", "haar", "frame", "weight"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+", default=P_GRID)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--ratio", type=int, default=None, help="T = ratio * N")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, default=Path("outputs/effect_mipt"))
    args = parser.parse_args()
    if args.probe == "clifford":
        result = run_clifford(
            args.n, args.p, args.samples, args.seeds[0], args.ratio or 4
        )
    elif args.probe == "haar":
        result = run_haar(args.n, args.p, args.samples, args.seeds[0], args.ratio or 2)
    elif args.probe == "weight":
        result = run_weight(args.n, args.p, args.samples, args.seeds, args.ratio or 1)
    else:
        result = run_frame(args.n, args.p, args.samples, args.seeds, args.ratio or 1)
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = "-".join(str(s) for s in args.seeds)
    path = (
        args.out / f"{args.probe}_N{args.n}_T{args.ratio or ''}_s{seeds}{args.tag}.json"
    )
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
