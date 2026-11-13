"""Dense-encoding Fourier weight ``W(m)`` of Clifford circuits at large ``N`` (#327).

For a mid-circuit record ``m``, ``Pr(m | x) = Tr[E_m rho(x)]`` on the ``N``
input qubits (dense encoding, no ancilla), and ``W(m) = sum_{a != 0}
c_a^2 / c_0^2`` over the ``4^N`` Pauli strings. With Clifford gates and
``Z`` measurements, ``log2(1 + W(m)) = k`` is the number of independent
stabilizers of the output of the maximally mixed input
(:func:`~QELM.evaluation.effect_mipt.dense_weight_bits`), and
``Delta = N - k`` is the Renyi-2 purification entropy.

Two circuits, one fresh circuit and record pattern per trajectory, no final
readout:

- ``periodic``: the literature circuit (Gullans & Huse, PRL 125, 070606),
  periodic brickwork, ``Z`` measurement of each site with probability ``p``
  after every half-layer;
- ``open``: the convention of #314, open brickwork, measurement after every
  layer of two half-layers.

``k`` is recorded after ``N``, ``2N`` and ``4N`` half-layers.

Usage::

    uv run python experiments/effect_mipt/dense_weight.py periodic --n 64 \
        --samples 2000 --out outputs/effect_mipt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_mipt import (
    dense_weight_bits,
    draw_open_half_layers,
    draw_periodic_half_layers,
)

P_GRID = {
    "periodic": [round(0.08 + 0.01 * i, 2) for i in range(17)],
    "open": [round(0.18 + 0.01 * i, 2) for i in range(23)],
}


def run(conv: str, n: int, ps: list[float], samples: int, seed: int) -> dict:
    """Mean and standard error of ``k`` and ``Delta`` at each checkpoint, per ``p``."""
    checkpoints = [n, 2 * n, 4 * n]
    out: dict = {"conv": conv, "n": n, "checkpoints": checkpoints}
    out.update(samples=samples, seed=seed, p=ps, k_mean=[], k_sem=[])
    tag = 31 if conv == "periodic" else 32
    for j, p in enumerate(ps):
        start = time.perf_counter()
        rng = np.random.default_rng((seed, n, j, tag))
        ks = np.empty((samples, len(checkpoints)))
        for s in range(samples):
            if conv == "periodic":
                circuit = draw_periodic_half_layers(n, p, 4 * n, rng)
            else:
                circuit = draw_open_half_layers(n, rng.random((2 * n, n)) < p, rng)
            ks[s] = dense_weight_bits(circuit, checkpoints)
        out["k_mean"].append(ks.mean(axis=0).tolist())
        out["k_sem"].append((ks.std(axis=0, ddof=1) / np.sqrt(samples)).tolist())
        delta = n - ks.mean(axis=0)
        dt = time.perf_counter() - start
        print(
            f"{conv} N={n} p={p}: Delta = {np.round(delta, 2)} ({dt:.0f} s)", flush=True
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("conv", choices=("periodic", "open"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/effect_mipt"))
    args = parser.parse_args()
    ps = args.p or P_GRID[args.conv]
    result = run(args.conv, args.n, ps, args.samples, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"dense_{args.conv}_N{args.n}_s{args.seed}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
