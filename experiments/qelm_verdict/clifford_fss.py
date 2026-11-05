#!/usr/bin/env python3
"""Calcul 6.A of the note: does the Bayes share g of the linear QELM-M survive
at large N on Clifford circuits? (issue #303)

The protocol of ``experiments/dehghani_decoder`` (#287): periodic Clifford
brickwork, one fixed circuit per seed, ``R`` in a Bell pair with the middle
site, a Z measurement of each site with probability ``p`` after each
half-layer, measured sites drawn anew per trajectory. Only the record ``M`` is
read (case 1), with the ridge only: near ``p_c`` the state features of the
full QELM are at chance from ``N = 8`` on (#287, case 3), and their exact
values cost ``O(N^2)`` stabilizer peeks per trajectory.

Two depth regimes (``--regime``):

- ``T2N``: ``2N`` measured steps from the Bell pair (the protocol of #287);
- ``scrambled``: ``2N`` steps without measurement, then ``2N`` measured steps,
  the probe whose oracle crosses at ``p_c = 0.16`` in #292.

For each ``(regime, N, seed)`` it writes ``<regime>_N<N>_seed<seed>.json``
into ``--out``, with one row per ``p``: the purified fraction ``f``, the
share ``w1`` of purified labels of parity weight ``<= 1`` (``--n-parity``
replayed trajectories), the ridge accuracy on purified trajectories
``acc_pure`` and over all of them ``acc_all``, and the Bayes ceiling
``(1 + f)/2``. Finished files are skipped, so a killed run resumes.

Run from the repository root:

    uv run python experiments/qelm_verdict/clifford_fss.py \
        --regime scrambled --N 12 16 24 32 --seeds 40 --jobs 4 \
        --out outputs/qelm_verdict/clifford_fss
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from QELM.evaluation.dehghani_decoder import (
    decode_accuracy,
    draw_fixed_circuit,
    parity_weight_one,
    simulate_records,
)

P_VALUES = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26)


def run_one(
    regime: str,
    N: int,
    seed: int,
    p_values: tuple[float, ...],
    n_traj: int,
    n_parity: int,
) -> dict:
    """Every ``p`` of one fixed circuit."""
    scramble = 2 * N if regime == "scrambled" else 0
    rng = np.random.default_rng([seed, N, scramble])
    gates = draw_fixed_circuit(N, scramble + 2 * N, rng)
    rows = []
    t0 = time.time()
    for p in p_values:
        rec, _ = simulate_records(gates, N, p, n_traj, rng, scramble=scramble)
        acc = decode_accuracy(rec, readouts=("ridge",), seed=seed)
        w = []
        for i in np.flatnonzero(rec.purified)[:n_parity]:
            mask = np.zeros((gates.shape[0], N), dtype=bool)
            mask[scramble:] = rec.M[i] != 0
            r = parity_weight_one(gates, N, mask, rng)
            if r is not None:
                w.append(r)
        rows.append(
            {
                "p": p,
                "purified": float(rec.purified.mean()),
                "weight_one": float(np.mean(w)) if w else float("nan"),
                "ridge_acc_pure": acc["ridge"]["acc_pure"],
                "ridge_acc_all": acc["ridge"]["acc_all"],
                "bayes_acc_all": acc["bayes"]["acc_all"],
                "n_test_pure": acc["ridge"]["n_test_pure"],
            }
        )
    return {
        "regime": regime,
        "N": N,
        "seed": seed,
        "T": 2 * N,
        "scramble": scramble,
        "n_traj": n_traj,
        "seconds": time.time() - t0,
        "rows": rows,
    }


def main() -> None:
    """Parse the options and run every ``(N, seed)`` in parallel."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=("T2N", "scrambled"), required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[12, 16, 24, 32])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--first-seed", type=int, default=42)
    ap.add_argument("--p", type=float, nargs="+", default=list(P_VALUES))
    ap.add_argument("--n-traj", type=int, default=10000)
    ap.add_argument("--n-parity", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/qelm_verdict/clifford_fss")
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # seeds first, so that every N gets circuits early and a partial run is usable
    jobs = [
        (N, s)
        for s in range(args.first_seed, args.first_seed + args.seeds)
        for N in sorted(args.N, reverse=True)
        if not (args.out / f"{args.regime}_N{N}_seed{s}.json").exists()
    ]
    print(f"{len(jobs)} circuits to run", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {
            pool.submit(
                run_one, args.regime, N, s, tuple(args.p), args.n_traj, args.n_parity
            ): (N, s)
            for N, s in jobs
        }
        for fut in as_completed(futs):
            N, s = futs[fut]
            res = fut.result()
            path = args.out / f"{args.regime}_N{N}_seed{s}.json"
            path.write_text(json.dumps(res))
            print(
                f"{args.regime} N={N} seed={s} done in {res['seconds']:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
