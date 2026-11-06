"""Fourier expressivity of the monitored QELM on one fixed circuit (issue #309).

For one dynamics, one size ``N``, one depth ``T`` and one seed (one fixed
draw of the gates), compute the transfer matrix ``C(p)`` of the record
features ``M_{t,i}`` and of the final ``Z_i`` on the ``3**N`` Bloch monomials
of the angle encoding, for every ``p`` of a grid, and write its measures
(:func:`QELM.evaluation.fourier.expressivity_measures`) to
``<out>/metrics.json``. Everything is exact: no trajectory is sampled.

    uv run python experiments/fourier_expressivity/fourier_expressivity.py \
        --model haar --N 6 --T 6 --seed 42 --out outputs/fourier/one_run

The launcher ``fourier_expressivity.sh`` runs a grid of these.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.fourier import expressivity_measures, transfer_matrix
from QELM.models.circuits import (
    CliffordBrickworkCircuit,
    HaarBrickworkCircuit,
    U1BrickworkCircuit,
)
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.symmetric import Z2BrickworkCircuit

MODELS: dict[str, type[BrickworkCircuit]] = {
    "haar": HaarBrickworkCircuit,
    "u1": U1BrickworkCircuit,
    "z2": Z2BrickworkCircuit,
    "clifford": CliffordBrickworkCircuit,
}


def run(
    model: str,
    N: int,
    T: int,
    seed: int,
    p_values: list[float],
    low: float = 0.0,
    high: float = float(np.pi),
) -> dict[str, object]:
    """Return the measures of ``C(p)`` for each ``p``, on one circuit."""
    rng = np.random.default_rng(seed)
    circuit = MODELS[model]({"N": N, "depth": T, "Nr": 0}, rng)
    per_p = []
    for p in p_values:
        C = transfer_matrix(circuit.layers, N, p)
        per_p.append({"p": p, **expressivity_measures(C, N, low, high)})
    return {
        "model": model,
        "N": N,
        "T": T,
        "seed": seed,
        "angle_range": [low, high],
        "n_features": T * N + N,
        "n_monomials": 3**N,
        "per_p": per_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--p", type=float, nargs="+", default=list(np.round(np.linspace(0, 1, 21), 3))
    )
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=float(np.pi))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start = time.time()
    result = run(args.model, args.N, args.T, args.seed, args.p, args.low, args.high)
    result["seconds"] = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'p':>5} {'rank':>5} {'k_mean':>7} {'d_eff':>6} {'odd_x':>6}")
    for row in result["per_p"]:  # type: ignore[attr-defined]
        print(
            f"{row['p']:5.2f} {row['rank']:5d} {row['mean_degree']:7.3f}"
            f" {row['d_eff']:6.2f} {row['odd_x_weight']:6.3f}"
        )
    print(f"Saved to {args.out} in {result['seconds']:.1f} s")


if __name__ == "__main__":
    main()
