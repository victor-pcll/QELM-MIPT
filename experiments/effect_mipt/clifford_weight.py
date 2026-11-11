"""Fourier weight of one encoded qubit across the Clifford MIPT (#325).

The literature circuit, the reference-qubit probe of Gullans & Huse (PRL
125, 070606 (2020)) as implemented in :mod:`QELM.models.circuits.stabilizer`:
periodic brickwork of random two-qubit Clifford gates, one qubit ``x``
encoded in a Bell pair with the middle site, ``2N`` half-layers without
measurement, then ``2N`` half-layers with a ``Z`` measurement of every site
with probability ``p`` after **each** half-layer.

For a record ``m``, ``Pr(m | x) = c_0 (1 + r_m . b(x))`` with ``b`` the Bloch
vector of the input, so the Fourier weight outside the constant mode is
``W(m) = |r_m|^2``, and ``log2(1 + W) = 1 - S_2`` with ``S_2`` the Renyi-2
entropy of the reference qubit. For a stabilizer state ``W`` is 0 or 1 and
does not depend on the outcomes, so the mean of ``W`` is one minus the mean
entropy of the reference qubit.

Usage::

    uv run python experiments/effect_mipt/clifford_weight.py --n 64 \
        --samples 2000 --out outputs/effect_mipt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from QELM.models.circuits.stabilizer import (
    draw_probe_history,
    reference_entropy_history,
)

P_GRID = [round(0.08 + 0.01 * i, 2) for i in range(17)]


def run(n: int, ps: list[float], samples: int, seed: int) -> dict:
    """Mean Fourier weight of the encoded qubit at the last step, per ``p``."""
    out: dict = {"n": n, "scramble": 2 * n, "steps": 2 * n}
    out.update(samples=samples, seed=seed)
    out["p"], out["w_mean"], out["w_sem"] = ps, [], []
    for j, p in enumerate(ps):
        rng = np.random.default_rng((seed, n, j, 21))
        w = np.empty(samples)
        for k in range(samples):
            history = draw_probe_history(n, p, 2 * n, 2 * n, rng)
            w[k] = 1.0 - reference_entropy_history(history)[-1]
        out["w_mean"].append(float(w.mean()))
        out["w_sem"].append(float(w.std(ddof=1) / np.sqrt(samples)))
        print(f"N={n} p={p}: W = {w.mean():.3f}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+", default=P_GRID)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/effect_mipt"))
    args = parser.parse_args()
    result = run(args.n, args.p, args.samples, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"clifford_weight_N{args.n}_s{args.seed}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
