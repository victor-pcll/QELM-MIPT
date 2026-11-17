#!/usr/bin/env python3
"""Rank structure behind the modes of the Haar E_circuit histograms (issue #362).

Replays the circuits of ``run.py dense`` (same seeds, same order of random
draws) and computes, for each circuit, :func:`QELM.evaluation.r2_budget.
rank_profile`: dense and visible ranks of the effect space, min cut ``C``,
measurement counts, mean ``R^2``. The per-circuit ``E_circuit`` comes from the
stored samples of ``run.py``; the first circuit of each ``p`` (when it has at
most ``--check-bits`` bits) is recomputed from scratch to check that the
replay sees the same circuit and targets.

Usage::

    uv run python experiments/r2_budget/rank_structure.py --data outputs/r2_budget
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/r2_budget"))
    parser.add_argument("--prior", choices=rb.PRIORS, default="iid")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--check-bits", type=int, default=12)
    args = parser.parse_args(argv)
    meta = json.loads((args.data / f"dense_N{args.n}_{args.prior}.json").read_text())
    with np.load(args.data / f"dense_N{args.n}_{args.prior}.npz") as z:
        raw = {k: z[k] for k in z.files}
    n, depth, n_t = meta["n"], meta["depth"], meta["targets"]
    out: dict = {
        "source": f"dense_N{n}_{args.prior}",
        "n": n,
        "depth": depth,
        "p": meta["p"],
        "per_p": [],
    }
    for j, p in enumerate(meta["p"]):
        rng = np.random.default_rng((meta["seed"], n, j, 360))
        rows: dict[str, dict[str, list]] = {"haar": {}, "clifford": {}}
        checked = False
        for i in range(meta["circuits"]):
            haar = ff.draw_circuit(n, depth, p, "haar", rng)
            while ff.n_bits(haar) > meta["max_bits"]:
                haar = ff.with_mask(haar, rng.random((depth, n)) < p)
            cliff = ff.with_mask(
                ff.draw_circuit(n, depth, p, "clifford", rng), haar.mask
            )
            g = rb.draw_whitened_targets(n, n_t, args.prior, rng)
            for kind, circ in (("haar", haar), ("clifford", cliff)):
                prof = rb.rank_profile(circ)
                e = raw[f"{kind}_circuit_{j}"][i * n_t : (i + 1) * n_t]
                for key, val in prof.__dict__.items():
                    rows[kind].setdefault(key, []).append(val)
                rows[kind].setdefault("e_circuit", []).append(float(e.mean()))
                if i == 0 and ff.n_bits(circ) <= args.check_bits:
                    b = rb.dense_budget(circ, (1,), g)
                    if not np.allclose(b.circuit, e, atol=1e-5):
                        raise RuntimeError(f"replay mismatch at p = {p}, {kind}")
                    checked = True
        res: dict = {"p": p, "replay_checked": checked}
        for kind, r in rows.items():
            levels = Counter(r["visible"])
            res[kind] = {
                **r,
                "levels": {str(k): v for k, v in sorted(levels.items())},
                "e_by_level": {
                    str(k): float(
                        np.mean(np.array(r["e_circuit"])[np.array(r["visible"]) == k])
                    )
                    for k in sorted(levels)
                },
            }
        hv = res["haar"]
        bound = np.minimum(n + np.array(hv["k_eff"]), 2 * n - np.array(hv["n_first"]))
        res["haar"]["two_bound_exact"] = float(np.mean(bound == np.array(hv["mincut"])))
        res["haar"]["dense_is_2_to_c"] = bool(
            np.all(np.array(hv["dense"]) == 2 ** np.array(hv["mincut"]))
        )
        out["per_p"].append(res)
        print(
            f"p={p:.2f} Haar levels {res['haar']['levels']} E_circ by level "
            f"{ {k: round(v, 3) for k, v in res['haar']['e_by_level'].items()} } "
            f"min(N+K, 2N-n1) = C: {res['haar']['two_bound_exact']:.2f}; "
            f"Clifford levels {res['clifford']['levels']}",
            flush=True,
        )
    path = args.data / f"rank_structure_N{n}_{args.prior}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
