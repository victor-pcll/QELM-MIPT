#!/usr/bin/env python3
"""Entanglement of the backward trajectories of the Haar QELM (issue #352).

Geometry of #334/#339: open brickwork, ``T = 2N`` steps (even bonds, odd
bonds, then ``Z`` measurements with probability ``p``), full final readout.
Each sample draws a fresh circuit (gates of the class ``--gates`` and mask)
and one record with the Born rule of the maximally mixed input; the
normalised effect ``psi_m`` (:func:`backward_trajectory`) gives:

- the profile ``S(l)`` of the left blocks, ``l = 0 .. N``;
- ``S(N/2)``;
- the tripartite information ``I3`` of the four quarters (``4 | N``).

``S(l)`` is the quantity that fixes the locality of the ridge dictionary
in the Clifford case: ``2^(l - S(l))`` representable functions fit in
``l`` sites (#337, #339).

Example::

    uv run python experiments/clifford_ridge/haar_order.py --n 12 --samples 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.backward_trajectory import (
    backward_trajectory,
    cut_profile,
    tripartite_information,
)
from QELM.evaluation.effect_rank import GATES, brick_pairs

P_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.32, 0.35, 0.4, 0.5, 0.6]


def run(n: int, p_grid: list[float], samples: int, kind: str, seed: int) -> dict:
    depth = 2 * n
    sampler = GATES[kind]
    out: dict = {"n": n, "t": depth, "samples": samples, "gates": kind}
    out.update(seed=seed, p=p_grid, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, j, 3520))
        prof, i3 = [], []
        for _ in range(samples):
            gates = [[sampler(rng) for _ in brick_pairs(n)] for _ in range(depth)]
            masks = rng.random((depth, n)) < p
            psi, _, _ = backward_trajectory(gates, masks, rng)
            prof.append(cut_profile(psi, n))
            if n % 4 == 0:
                i3.append(tripartite_information(psi, n))
        a = np.array(prof)
        row = {
            "p": p,
            "S_profile": a.mean(axis=0).tolist(),
            "S_profile_sem": (a.std(axis=0, ddof=1) / np.sqrt(samples)).tolist(),
            "S_half": float(a[:, n // 2].mean()),
            "S_half_sem": float(a[:, n // 2].std(ddof=1) / np.sqrt(samples)),
        }
        if i3:
            row["I3"] = float(np.mean(i3))
            row["I3_sem"] = float(np.std(i3, ddof=1) / np.sqrt(samples))
        out["per_p"].append(row)
        extra = f" I3={row['I3']:.3f}" if i3 else ""
        print(
            f"{kind} N={n} p={p}: S(N/2)={row['S_half']:.3f}{extra} "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--gates", choices=sorted(GATES), default="haar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/haar_order"))
    args = parser.parse_args(argv)
    res = run(args.n, args.p or P_GRID, args.samples, args.gates, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"traj_{args.gates}_N{args.n}_s{args.seed}.json"
    path.write_text(json.dumps(res))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
