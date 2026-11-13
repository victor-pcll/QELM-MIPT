"""Exact small-N table of the Fourier framework: Haar against Clifford (issue #353).

For each seed, one Haar and one Clifford circuit see the same measurement
mask (open brickwork, ``T = N``). Everything is exact: every record is
enumerated, every effect is expanded on the ``4^N`` Pauli strings, and the
inputs are all the products of ``|0>, |1>, |+>, |->`` (XZ ensemble).

Usage::

    uv run python experiments/fourier_framework/small_n_table.py \
        --n 4 --seeds 10 --out outputs/fourier_framework

It writes ``small_n_table.json`` and prints a Markdown table. About 2 min at
``N = 4``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from QELM.evaluation import fourier_framework as ff


def _monomial(n: int, site: int, axis: int) -> np.ndarray:
    beta = np.zeros(4**n)
    beta[axis * 4 ** (n - 1 - site)] = 1.0
    return beta


def one_circuit(circuit: ff.Circuit, inputs: ff.Inputs) -> dict[str, float]:
    n = circuit.n
    w = ff.input_weights(n, "xz")
    vis = ff.visible_mask(n, "xz")
    full = ff.enumerate_records(circuit)
    mid = ff.enumerate_records(circuit, final=False)
    c = ff.feature_matrix(full.effects)
    cm = ff.feature_matrix(mid.effects)
    spec = ff.purification_spectrum(mid.effects)
    z_mid = _monomial(n, n // 2, 3)
    x_mid = _monomial(n, n // 2, 1)
    deg1 = sum(_monomial(n, i, a) for i in range(n) for a in (1, 3))
    c1 = ff.parity_features(c, full.bits, 1)
    c2 = ff.parity_features(c, full.bits, 2)
    phi = inputs.monomials @ c.T
    split1 = ff.two_filter_split(c1, c, w, z_mid)
    split2 = ff.two_filter_split(c2, c, w, z_mid)
    return {
        "bits": float(ff.n_bits(circuit)),
        "log2_dense_rank": float(np.log2(ff.span_rank(c))),
        "visible_rank_over_3N": ff.span_rank(c, vis) / 3**n,
        "commutator": ff.max_commutator(c),
        "log2_mid_rank": float(np.log2(max(ff.span_rank(cm), 1))),
        "renyi0": spec.renyi0,
        "renyi1": spec.renyi1,
        "renyi2": spec.renyi2,
        "eperp_z_full": ff.population_error(c, w, z_mid),
        "eperp_x_full": ff.population_error(c, w, x_mid),
        "eperp_deg1_full": ff.population_error(c, w, deg1),
        "eperp_deg1_mid": ff.population_error(cm, w, deg1),
        "readout1_z": split1.readout,
        "readout2_z": split2.readout,
        "cross1_z": split1.cross,
        "injective": float(ff.n_distinct_features(phi) == len(phi)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--max-bits", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("outputs/fourier_framework"))
    args = parser.parse_args()
    n = args.n
    inputs = ff.eigenstate_inputs(n, "xz")
    rows: list[dict[str, float | str]] = []
    for p in args.p:
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * seed + n)
            haar = ff.draw_circuit(n, n, p, "haar", rng)
            clif = ff.with_mask(ff.draw_circuit(n, n, p, "clifford", rng), haar.mask)
            if ff.n_bits(haar) > args.max_bits:
                continue
            for kind, circ in (("haar", haar), ("clifford", clif)):
                row: dict[str, float | str] = {"kind": kind, "p": p, "seed": seed}
                row.update(one_circuit(circ, inputs))
                rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "small_n_table.json").write_text(json.dumps(rows, indent=1))
    keys = [k for k in rows[0] if k not in ("kind", "p", "seed")]
    print("| kind | p | circuits | " + " | ".join(keys) + " |")
    print("|" + "---|" * (len(keys) + 3))
    for kind in ("haar", "clifford"):
        for p in args.p:
            sel = [r for r in rows if r["kind"] == kind and r["p"] == p]
            if not sel:
                continue
            means = [np.mean([float(r[k]) for r in sel]) for k in keys]
            cells = " | ".join(f"{m:.3f}" for m in means)
            print(f"| {kind} | {p} | {len(sel)} | {cells} |")


if __name__ == "__main__":
    main()
