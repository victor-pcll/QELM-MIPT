"""Rank of the effect space against p, fixed mask, one panel per circuit class.

For one size ``N`` and one seed, draw one brickwork circuit of depth
``T = N`` per class (Haar U(4), U(1), Z2 parity, Clifford) and one uniform
draw ``u`` of shape ``(T, N)``. The mask of the mid-circuit measurements at
rate ``p`` is ``u < p``: the masks are nested from one ``p`` to the next,
and every class sees the same masks. For each class and each ``p``, the
script writes the exact rank of the effect space seen by the dense and by
the angle encoding (:mod:`QELM.evaluation.effect_rank`).

Usage::

    uv run python experiments/effect_rank/effect_rank.py --n 6 --seed 0 \
        --out outputs/effect_rank

writes ``outputs/effect_rank/N6_seed0.json``. The plot script
``plot_effect_rank.py`` reads the folder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_rank import draw_circuit, effect_rank

KINDS = ("haar", "u1", "z2", "clifford")


def run(
    n: int,
    seed: int,
    p_grid: list[float],
    depth: int | None = None,
    kinds: tuple[str, ...] = KINDS,
) -> dict:
    """Ranks of every class on the grid ``p_grid`` for one size and seed."""
    t = n if depth is None else depth
    u = np.random.default_rng((seed, n, 1000)).random((t, n))
    out: dict = {"n": n, "depth": t, "seed": seed, "p": p_grid, "ranks": {}}
    for kind in kinds:
        k = KINDS.index(kind)
        circuit = draw_circuit(kind, n, t, np.random.default_rng((seed, n, k)))
        dense, angle, seconds = [], [], []
        for p in p_grid:
            start = time.perf_counter()
            r = effect_rank(circuit, u < p)
            seconds.append(time.perf_counter() - start)
            dense.append(r.dense)
            angle.append(r.angle)
        out["ranks"][kind] = {"dense": dense, "angle": angle, "seconds": seconds}
        print(f"N={n} seed={seed} {kind}: {sum(seconds):.1f}s", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True, help="number of qubits")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=None, help="default T = N")
    parser.add_argument("--p-step", type=float, default=0.05)
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    parser.add_argument("--out", type=Path, default=Path("outputs/effect_rank"))
    args = parser.parse_args()
    steps = round(1 / args.p_step)
    p_grid = [round(i / steps, 6) for i in range(steps + 1)]
    result = run(args.n, args.seed, p_grid, args.depth, tuple(args.kinds))
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if tuple(args.kinds) == KINDS else "_" + "-".join(args.kinds)
    path = args.out / f"N{args.n}_seed{args.seed}{tag}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
