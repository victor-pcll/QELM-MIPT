#!/usr/bin/env python3
"""Linear and non-linear decoders of the purified reference qubit (issue #287).

The protocol of Dehghani et al., Nat. Commun. 14, 2918 (2023), on the
periodic Clifford brickwork (:mod:`QELM.evaluation.dehghani_decoder`): ``R``
in a Bell pair with the middle site, one fixed circuit per seed, ``T = 2N``
measured steps, measured sites drawn anew per trajectory. For each ``N``,
seed and ``p``:

1. sample ``--n-traj`` trajectories (record ``M``, purification axis and
   sign of ``R``);
2. for the record alone and for the record plus a final readout, train a
   ridge (the linear QELM-M) and an MLP on ``vec(M)``, one per purification
   axis, on 70 % of the trajectories purified along that axis,
   and score them on the other 30 % (:func:`decode_accuracy`);
3. replay ``--n-parity`` purified trajectories to measure the share whose
   label is fixed by at most one outcome (:func:`parity_weight_one`).

Run from the repository root:

    uv run python experiments/dehghani_decoder/dehghani_decoder.py \
        --N 6 8 10 12 --seeds 6 --out outputs/dehghani_decoder

Two cases are decoded from the same trajectories: ``M`` (the record only,
``R`` after the last step, Dehghani et al.) and ``M_final`` (the record and
a final single-shot Z readout of every system site, ``R`` after it).

One ``N<N>_seed<seed>.json`` per circuit lands in ``--out``, with a list of
rows, one per ``p``: ``p`` and, for each case ``M`` and ``M_final``,
``purified`` (fraction of trajectories where ``R`` ends pure, ``1 - S_Q``),
``axes`` (counts of mixed, X, Y, Z), ``weight_one`` (share of weight-1
parities, NaN if none purified) and, per readout (``ridge``, ``mlp``,
``bayes``), ``acc_pure`` and ``acc_all``.
With ``--case qelm``, the third case, the full QELM: the record ``M`` and the
expectation values ``<P_k>``, ``<P_i P_j>`` of the final state, one decoder
per axis on all trajectories (``qelm_N<N>_seed<seed>.json``, rows with
``acc``, ``acc_pure``, ``acc_mixed`` per input and readout).

``plot_dehghani_decoder.py`` draws the figures from that folder.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from QELM.evaluation.dehghani_decoder import (
    AXES,
    decode_accuracy,
    decode_qelm_accuracy,
    draw_fixed_circuit,
    parity_weight_one,
    simulate_qelm_records,
    simulate_records,
)

P_VALUES = (
    0.0,
    0.02,
    0.05,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.25,
    0.3,
    0.4,
    0.6,
    1.0,
)


def run_circuit(
    N: int, seed: int, p_values: tuple[float, ...], n_traj: int, n_parity: int
) -> list[dict]:
    """Every ``p`` of one fixed circuit of ``N`` sites drawn from ``seed``."""
    rng = np.random.default_rng([seed, N])
    gates = draw_fixed_circuit(N, 2 * N, rng)
    rows = []
    for p in p_values:
        row: dict = {"p": p}
        for case, rec in zip(
            ("M", "M_final"), simulate_records(gates, N, p, n_traj, rng), strict=True
        ):
            acc = decode_accuracy(rec, seed=seed)
            pure = np.flatnonzero(rec.purified)[:n_parity]
            w = [parity_weight_one(gates, N, rec.M[i] != 0, rng) for i in pure]
            row[case] = {
                "purified": float(rec.purified.mean()),
                "axes": dict(
                    zip(AXES, np.bincount(rec.axis, minlength=4).tolist(), strict=True)
                ),
                "weight_one": float(np.mean(w)) if w else float("nan"),
                **acc,
            }
            print(
                f"N={N} seed={seed} p={p:.3f} {case:7s} "
                f"pure={row[case]['purified']:.2f} w1={row[case]['weight_one']:.2f} "
                + " ".join(f"{k}={acc[k]['acc_pure']:.3f}" for k in ("ridge", "mlp")),
                flush=True,
            )
        rows.append(row)
    return rows


def run_circuit_qelm(
    N: int, seed: int, p_values: tuple[float, ...], n_traj: int, n_parity: int
) -> list[dict]:
    """The full QELM case (``--case qelm``) for one fixed circuit.

    The same circuit as :func:`run_circuit` (same seed), with the record and
    the QELM features ``<P_k>``, ``<P_i P_j>`` of the final state
    (:func:`simulate_qelm_records`, :func:`decode_qelm_accuracy`).
    ``n_parity`` is not used.
    """
    rng = np.random.default_rng([seed, N])
    gates = draw_fixed_circuit(N, 2 * N, rng)
    rows = []
    for p in p_values:
        rec = simulate_qelm_records(gates, N, p, n_traj, rng)
        acc = decode_qelm_accuracy(rec, seed=seed)
        rows.append({"p": p, "purified": float(np.mean(rec.axis > 0)), **acc})
        print(
            f"N={N} seed={seed} p={p:.3f} qelm "
            + " ".join(f"{k}={v['acc']:.3f}" for k, v in acc.items()),
            flush=True,
        )
    return rows


def main() -> None:
    """Parse the options and run every ``(N, seed)`` in parallel."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--N", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--seeds", type=int, default=6, help="circuits per N")
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--p", type=float, nargs="+", default=list(P_VALUES))
    parser.add_argument("--n-traj", type=int, default=40000)
    parser.add_argument("--n-parity", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--case",
        choices=("record", "qelm"),
        default="record",
        help="record: M and M + final readout; qelm: M + <P_k> + <P_i P_j>",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/dehghani_decoder"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prefix, run = (
        ("qelm_", run_circuit_qelm) if args.case == "qelm" else ("", run_circuit)
    )
    jobs = [
        (N, s)
        for N in args.N
        for s in range(args.first_seed, args.first_seed + args.seeds)
        if not (args.out / f"{prefix}N{N}_seed{s}.json").exists()  # resume
    ]
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(run, N, s, tuple(args.p), args.n_traj, args.n_parity): (
                N,
                s,
            )
            for N, s in jobs
        }
        for fut, (N, s) in futures.items():
            rows = fut.result()
            path = args.out / f"{prefix}N{N}_seed{s}.json"
            path.write_text(json.dumps({"N": N, "seed": s, "T": 2 * N, "rows": rows}))


if __name__ == "__main__":
    main()
