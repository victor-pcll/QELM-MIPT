#!/usr/bin/env python
"""Entropy profile S(l) and I3 of the effect dictionary, periodic brickwork (#358).

One sample = one circuit and one record, with the full final readout. The
state is ``psi_m = A_m^dagger |m_f>`` (the references after the readout):

- ``haar``: backward trajectory of the Haar brickwork
  (:func:`QELM.evaluation.bipartite_entropy.backward_trajectory`);
- ``clifford``: the group of the effect with the full record
  (:func:`QELM.evaluation.clifford_ridge.effect_groups`), same circuit as
  #337 / #346.

Writes ``<out>/<model>_N<N>_T<T>_s<seed>.json`` with, per ``p``, the mean and
error of ``S(l)`` (``l = 0 .. N/2``) and the per-sample ``S(N/2)`` and ``I3``.

    uv run python experiments/bipartite_entropy/run.py haar --n 12 --samples 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.bipartite_entropy import (
    backward_trajectory,
    draw_haar_circuit,
    stabilizer_profile,
    state_profile,
)
from QELM.evaluation.clifford_ridge import draw_circuit, effect_groups, with_mask

P_HAAR = [
    0.04,
    0.08,
    0.10,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20,
    0.22,
    0.24,
    0.27,
    0.30,
    0.35,
    0.40,
]
P_CLIFFORD = [
    0.04,
    0.08,
    0.10,
    0.12,
    0.13,
    0.14,
    0.15,
    0.155,
    0.16,
    0.165,
    0.17,
    0.18,
    0.19,
    0.20,
    0.22,
    0.25,
    0.30,
    0.40,
    0.50,
]


def sample(model: str, n: int, t: int, p: float, rng: np.random.Generator, stride: int):
    """Profile of one circuit and one record."""
    if model == "haar":
        circuit = draw_haar_circuit(n, t, p, rng)
        psi = backward_trajectory(circuit, rng)[0]
        return state_profile(psi, n, starts=range(0, n, stride))
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
    return stabilizer_profile(effect_groups(circuit).full)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", choices=("haar", "clifford"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--t", type=int, default=None, help="half-layers (default 2N)")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Haar: start stride")
    parser.add_argument("--out", type=Path, default=Path("outputs/bipartite_entropy"))
    args = parser.parse_args()
    n = args.n
    t = args.t or 2 * n
    grid = args.p or (P_HAAR if args.model == "haar" else P_CLIFFORD)
    out: dict = {"model": args.model, "n": n, "t": t, "samples": args.samples}
    out.update(seed=args.seed, stride=args.stride, p=grid, per_p=[])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.model}_N{n}_T{t}_s{args.seed}.json"
    model_id = 0 if args.model == "haar" else 1
    for j, p in enumerate(grid):
        t0 = time.time()
        rng = np.random.default_rng((args.seed, model_id, n, t, j, 358))
        profs = [
            sample(args.model, n, t, p, rng, args.stride) for _ in range(args.samples)
        ]
        s1 = np.array([q.s1 for q in profs])
        s2 = np.array([q.s2 for q in profs])
        root = np.sqrt(args.samples)
        out["per_p"].append(
            {
                "p": p,
                "s1": s1.mean(axis=0).tolist(),
                "s1_err": (s1.std(axis=0, ddof=1) / root).tolist(),
                "s2": s2.mean(axis=0).tolist(),
                "s2_err": (s2.std(axis=0, ddof=1) / root).tolist(),
                "half_s1": s1[:, -1].tolist(),
                "half_s2": s2[:, -1].tolist(),
                "i3": [q.i3 for q in profs],
                "i3_2": [q.i3_2 for q in profs],
            }
        )
        print(
            f"{args.model} N={n} T={t} p={p}: S(N/2)={s1[:, -1].mean():.3f}"
            f" I3={np.mean([q.i3 for q in profs]):.3f} {time.time() - t0:.1f}s",
            flush=True,
        )
        path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
