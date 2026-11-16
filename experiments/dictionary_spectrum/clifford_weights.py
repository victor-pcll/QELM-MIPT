#!/usr/bin/env python3
"""Weight histogram N(l) of the Clifford effect group across the MIPT (issue #356).

Circuit of issue #337: periodic brickwork of random two-qubit Clifford
gates, one half-layer per step, a ``Z`` measurement of each site with
probability ``p`` after every step, ``T = 2N``, ``p_c = 0.161``. For each
circuit (one gate draw and one uniform draw ``u``, nested masks ``u < p``),
the group ``G`` of the effect is enumerated with and without the final
readout, and the Hamming weights of its ``2^r`` strings are counted, all
of them and those without ``Y``.

Example::

    uv run python experiments/dictionary_spectrum/clifford_weights.py --n 16 --samples 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.clifford_ridge import (
    draw_circuit,
    effect_groups,
    purification,
    with_mask,
)
from QELM.evaluation.dictionary_spectrum import weight_enumerator

P_GRID = [0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5]


def run(n: int, samples: int, p_grid: list[float], seed: int) -> dict:
    t_steps = 2 * n
    keys = ("full_all", "full_vis", "mid_all", "mid_vis")
    hist = {k: np.zeros((len(p_grid), samples, n + 1), dtype=np.int64) for k in keys}
    delta = np.zeros((len(p_grid), samples))
    t0 = time.time()
    for s in range(samples):
        rng = np.random.default_rng((seed, n, s))
        gates = draw_circuit(n, t_steps, rng)
        u = rng.random((t_steps, n))
        for j, p in enumerate(p_grid):
            circ = with_mask(gates, u, p)
            groups = effect_groups(circ, seed=s)
            hist["full_all"][j, s], hist["full_vis"][j, s] = weight_enumerator(
                groups.full
            )
            hist["mid_all"][j, s], hist["mid_vis"][j, s] = weight_enumerator(
                groups.mid
            )
            delta[j, s] = purification(circ, {}, seed=s).delta
        if (s + 1) % max(1, samples // 10) == 0:
            print(f"N={n} {s + 1}/{samples} {time.time() - t0:.0f}s", flush=True)
    out: dict = {"n": n, "T": t_steps, "samples": samples, "seed": seed, "p": p_grid}
    out["delta_mean"] = delta.mean(axis=1).tolist()
    out["delta_sem"] = (delta.std(axis=1, ddof=1) / np.sqrt(samples)).tolist()
    for k, h in hist.items():
        out[k] = {
            "mean": h.mean(axis=1).tolist(),
            "sem": (h.std(axis=1, ddof=1) / np.sqrt(samples)).tolist(),
            "total_min": h.sum(axis=2).min(axis=1).tolist(),
            "total_max": h.sum(axis=2).max(axis=1).tolist(),
        }
    # Exact check, per circuit: sum_l N(l) = 2^r.
    assert np.all(hist["full_all"].sum(axis=2) == 2**n)
    assert np.all(hist["mid_all"].sum(axis=2) == np.rint(2.0 ** (n - delta)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/dictionary_spectrum"))
    args = parser.parse_args()
    result = run(args.n, args.samples, args.p or P_GRID, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"clifford_weights_N{args.n}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
