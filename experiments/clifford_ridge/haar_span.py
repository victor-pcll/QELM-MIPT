#!/usr/bin/env python3
"""Ridge at K = infinity: Haar against Clifford gates, same geometry (issue #339).

Geometry of issue #334: open brickwork, one layer = even bonds, odd bonds,
then a ``Z`` measurement of each site with probability ``p``; depth
``T = 2N``; final ``Z`` readout (``p_c = 0.29`` for Haar in this
convention, #322). For each seed, one Haar circuit, one Clifford circuit and
one uniform draw ``u`` (nested masks ``u < p``, shared by the two classes).

For each ``p`` and each record (mid-circuit only, full), from the exact span
of the effects (:func:`QELM.evaluation.ridge_spectrum.effect_operator_basis`):

- dense rank ``dim Span{E_m}`` and angle rank (strings without ``Y``);
- ``E_perp`` of the population ridge with ``lambda -> 0+`` for the X and Z
  of the middle qubit and for random degree-1 and degree-2 targets;
- ``dim(Span ∩ V_A) / 3^l`` for every window ``A`` of ``l`` sites
  (``l = 1 .. 3``): the fraction of the local function space of the window
  that the readout can represent.

Example::

    uv run python experiments/clifford_ridge/haar_span.py --n 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_rank import GATES, brick_pairs
from QELM.evaluation.ridge_spectrum import (
    angle_span,
    approximation_bias,
    degree_target_circle,
    effect_operator_basis,
    single_site_target,
    window_capture,
)

P_GRID = [round(0.05 * i, 2) for i in range(21)]
WINDOWS = (1, 2, 3)


def draw_gates(kind: str, n: int, depth: int, rng: np.random.Generator) -> list:
    """Unitaries ``gates[t][g]`` of an open brickwork, order of ``brick_pairs``."""
    sampler = GATES[kind]
    return [[sampler(rng) for _ in brick_pairs(n)] for _ in range(depth)]


def targets(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng((seed, n, 2000))
    out = {
        "x_mid": single_site_target(n, n // 2, "x"),
        "z_mid": single_site_target(n, n // 2, "z"),
        "k1": degree_target_circle(n, 1, rng),
    }
    if n >= 2:
        out["k2"] = degree_target_circle(n, 2, rng)
    return out


def span_row(gates: list, masks: np.ndarray, final: bool, tgt: dict) -> dict:
    n = masks.shape[1]
    basis = effect_operator_basis(gates, masks, final)
    span = angle_span(basis, n)
    row: dict = {
        "dense": int(basis.shape[0]),
        "angle": int(span.shape[0]),
        "e_perp": {k: approximation_bias(span, b) for k, b in tgt.items()},
        "window": {},
    }
    for ell in WINDOWS:
        if ell >= n:
            continue
        caps = [
            window_capture(span, n, range(s, s + ell)) / 3**ell
            for s in range(n - ell + 1)
        ]
        row["window"][str(ell)] = caps
    return row


def run(n: int, seed: int, p_grid: list[float]) -> dict:
    depth = 2 * n
    u = np.random.default_rng((seed, n, 1000)).random((depth, n))
    tgt = targets(n, seed)
    out: dict = {"n": n, "depth": depth, "seed": seed, "p": p_grid, "per_kind": {}}
    for j, kind in enumerate(("haar", "clifford")):
        gates = draw_gates(kind, n, depth, np.random.default_rng((seed, n, 0, j)))
        rows = []
        for p in p_grid:
            t0 = time.time()
            masks = u < p
            res = {
                "p": p,
                "mid": span_row(gates, masks, False, tgt),
                "full": span_row(gates, masks, True, tgt),
            }
            rows.append(res)
            f = res["full"]
            print(
                f"{kind} N={n} seed={seed} p={p}: dense={f['dense']} "
                f"angle={f['angle']}/{3**n} E_perp(z_mid)={f['e_perp']['z_mid']:.3f} "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
        out["per_kind"][kind] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/haar_span"))
    args = parser.parse_args()
    result = run(args.n, args.seed, args.p or P_GRID)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"haar_span_N{args.n}_seed{args.seed}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
