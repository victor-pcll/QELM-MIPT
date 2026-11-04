#!/usr/bin/env python3
"""CNN decoder of the record M on Clifford circuits (issue #293).

The protocol of #289 (:mod:`QELM.evaluation.dehghani_decoder`): ``R`` in a
Bell pair with the middle site, one fixed periodic Clifford brickwork per
seed, ``T = 2N`` steps, measured sites drawn anew per trajectory. For each
``N``, seed and ``p`` it samples ``--n-traj`` trajectories with stim and
scores, on the same 70 / 30 split and on the trajectories purified along
each axis:

- the ridge and the MLP of #289 (:func:`decode_accuracy`);
- the CNN of :mod:`QELM.evaluation.cnn_decoder`;
- the Bayes-optimal decoder (accuracy 1 on purified trajectories).

Two modes, from the repository root:

    # accuracy against p
    uv run python experiments/cnn_decoder/cnn_decoder.py sweep \
        --N 6 8 10 12 16 --seeds 2 --out outputs/cnn_decoder

    # accuracy against the training-set size, same test set
    uv run python experiments/cnn_decoder/cnn_decoder.py scaling \
        --N 8 12 --p 0.1 0.16 0.3 --n-traj 300000 --out outputs/cnn_decoder

One JSON per job lands in ``--out`` (``sweep_N<N>_seed<s>_p<p>.json`` or
``scaling_N<N>_seed<s>_p<p>.json``); an existing file is skipped, so a run
can be resumed. ``plot_cnn_decoder.py`` draws the figures from that folder.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

P_VALUES = (0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4)
TRAIN_SIZES = (1000, 3000, 10000, 30000, 100000, 210000)


def _data(N: int, seed: int, p: float, n_traj: int):  # type: ignore[no-untyped-def]
    """The fixed circuit of ``(N, seed)`` (as in #289) and its trajectories."""
    from QELM.evaluation.dehghani_decoder import draw_fixed_circuit, simulate_records

    gates = draw_fixed_circuit(N, 2 * N, np.random.default_rng([seed, N]))
    rng = np.random.default_rng([seed, N, round(1000 * p)])
    t0 = time.perf_counter()
    records, _ = simulate_records(gates, N, p, n_traj, rng)
    return records, time.perf_counter() - t0


def _summary(records) -> dict:  # type: ignore[no-untyped-def]
    axes = np.bincount(records.axis, minlength=4).tolist()
    return {"purified": float(records.purified.mean()), "axes": axes}


def run_sweep(N: int, seed: int, p: float, n_traj: int, out: Path) -> str:
    """Ridge, MLP, CNN and Bayes at one ``(N, seed, p)``."""
    from QELM.evaluation.cnn_decoder import decode_cnn_accuracy
    from QELM.evaluation.dehghani_decoder import decode_accuracy

    records, t_sim = _data(N, seed, p, n_traj)
    t0 = time.perf_counter()
    acc = decode_accuracy(records, seed=seed)
    t_lin = time.perf_counter() - t0
    cnn, log = decode_cnn_accuracy(records, seed=seed)
    row = {
        "N": N,
        "T": 2 * N,
        "seed": seed,
        "p": p,
        "n_traj": n_traj,
        **_summary(records),
        **acc,
        "cnn": cnn,
        "cnn_log": asdict(log),
        "seconds": {"simulate": t_sim, "ridge+mlp": t_lin, "cnn": log.seconds},
    }
    path = out / f"sweep_N{N}_seed{seed}_p{p:.3f}.json"
    path.write_text(json.dumps(row))
    return (
        f"N={N} seed={seed} p={p:.3f} f={row['purified']:.2f} "
        f"ridge={acc['ridge']['acc_pure']:.3f} mlp={acc['mlp']['acc_pure']:.3f} "
        f"cnn={cnn['acc_pure']:.3f} ({log.seconds:.0f} s)"
    )


def run_scaling(
    N: int, seed: int, p: float, n_traj: int, sizes: tuple[int, ...], out: Path
) -> str:
    """CNN accuracy against the number of training trajectories."""
    from QELM.evaluation.cnn_decoder import decode_cnn_accuracy

    records, _ = _data(N, seed, p, n_traj)
    rows = []
    for size in sizes:
        cnn, log = decode_cnn_accuracy(records, seed=seed, max_train=size)
        rows.append({"n_train": size, **cnn, "log": asdict(log)})
    row = {"N": N, "T": 2 * N, "seed": seed, "p": p, **_summary(records)}
    row["rows"] = rows
    (out / f"scaling_N{N}_seed{seed}_p{p:.3f}.json").write_text(json.dumps(row))
    return f"N={N} p={p:.3f} " + " ".join(
        f"{r['n_train']}:{r['acc_pure']:.3f}" for r in rows
    )


def _init_worker() -> None:
    """One thread per worker process: the jobs run side by side."""
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def main() -> None:
    """Parse the options and run the jobs in parallel, largest first."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("sweep", "scaling"))
    parser.add_argument("--N", type=int, nargs="+", default=[6, 8, 10, 12, 16])
    parser.add_argument("--seeds", type=int, default=2, help="circuits per N")
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--n-traj", type=int, default=100000)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/cnn_decoder"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p_values = args.p or (list(P_VALUES) if args.mode == "sweep" else [0.1, 0.16, 0.3])
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    jobs = [
        (N, s, p)
        for N in sorted(args.N, reverse=True)
        for s in seeds
        for p in p_values
        if not (args.out / f"{args.mode}_N{N}_seed{s}_p{p:.3f}.json").exists()
    ]
    with ProcessPoolExecutor(args.jobs, initializer=_init_worker) as pool:
        if args.mode == "sweep":
            futures = [
                pool.submit(run_sweep, N, s, p, args.n_traj, args.out)
                for N, s, p in jobs
            ]
        else:
            futures = [
                pool.submit(
                    run_scaling, N, s, p, args.n_traj, tuple(args.sizes), args.out
                )
                for N, s, p in jobs
            ]
        for fut in futures:
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
