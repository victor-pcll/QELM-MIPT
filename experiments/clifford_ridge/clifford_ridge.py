#!/usr/bin/env python3
"""Clifford QELM: the MIPT in the QELM protocol and its ridge readout (issue #337).

Two subcommands, on the periodic Clifford brickwork of Gullans & Huse with a
``Z`` measurement of each site with probability ``p`` after every half-layer,
the input encoded at ``t = 0`` and no scrambling prelude:

- ``transition``: a fresh circuit and mask per sample; entropy of the
  references (``Delta``), entropy of the middle reference, and membership of
  the target monomials in the group of the effect (mid-circuit record and
  full record), which gives the single-shot R2 ``w_a Pr[a in G_m]``.
- ``ridge``: one circuit per seed, nested masks ``u < p``, random
  Pauli-eigenstate inputs, local features averaged over ``K`` shots, ridge
  readout with bias/variance over shot repetitions; optionally the double
  descent in the number of training inputs (``--dd``).

Example::

    uv run python experiments/clifford_ridge/clifford_ridge.py transition --n 16 --t 32
    uv run python experiments/clifford_ridge/clifford_ridge.py ridge --n 16 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.clifford_ridge import (
    degree_one_values,
    draw_circuit,
    draw_inputs,
    empirical_spectrum,
    mean_and_sem,
    monomial_values,
    monomial_weight,
    population_r2,
    purification,
    sample_inputs,
    targets_for,
    with_mask,
)
from QELM.evaluation.ridge_spectrum import (
    generalized_spectrum,
    n_learn,
    ridge_sweep,
    shot_groups,
)

P_TRANSITION = [round(0.08 + 0.01 * i, 2) for i in range(23)]
P_RIDGE = [0.04, 0.08, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.3, 0.4]
K_GRID = [1, 4, 16, 64]
K_POPULATION = [1, 4, 16, 64, 256, 1024]
LAMBDAS = np.logspace(-8, 2, 31)
DD_FRACTIONS = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]


def run_transition(n: int, t: int, p_grid: list[float], samples: int, seed: int) -> dict:
    targets = targets_for(n)
    out: dict = {"probe": "transition", "n": n, "t": t, "samples": samples}
    out.update(seed=seed, p=p_grid, targets=targets, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, t, j, 337))
        rows = []
        for _ in range(samples):
            gates = draw_circuit(n, t, rng)
            u = rng.random((t, n))
            rows.append(purification(with_mask(gates, u, p), targets))
        res: dict = {"p": p}
        res["delta"] = mean_and_sem([r.delta for r in rows])
        res["delta_per_site"] = mean_and_sem([r.delta / n for r in rows])
        res["s_mid"] = mean_and_sem([r.s_mid for r in rows])
        for record in ("mid", "full"):
            res[f"r2_single_{record}"] = {}
            for name, ops in targets.items():
                member = [getattr(r, f"member_{record}")[name] for r in rows]
                m, e = mean_and_sem(member)
                w = monomial_weight(ops)
                res[f"r2_single_{record}"][name] = [w * m, w * e]
        out["per_p"].append(res)
        print(
            f"transition N={n} T={t} p={p}: Delta/N={res['delta_per_site'][0]:.3f} "
            f"S_mid={res['s_mid'][0]:.3f} {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def run_ridge(
    n: int,
    seed: int,
    p_grid: list[float],
    t: int,
    kmax: int,
    sizes: tuple[int, int, int],
    max_reps: int,
    dd: bool,
) -> dict:
    gates = draw_circuit(n, t, np.random.default_rng((seed, n, 0)))
    u = np.random.default_rng((seed, n, 1000)).random((t, n))
    n_tr, n_va, n_te = sizes
    codes = draw_inputs(n_tr + n_va + n_te, n, np.random.default_rng((seed, n, 4000)))
    split = {
        "train": slice(0, n_tr),
        "val": slice(n_tr, n_tr + n_va),
        "test": slice(n_tr + n_va, None),
    }
    monomials = targets_for(n)
    y = {name: monomial_values(codes, ops) for name, ops in monomials.items()}
    weights = np.random.default_rng((seed, n, 2000)).normal(size=(2, n))
    y["deg1"] = degree_one_values(codes, weights)
    k_values = [k for k in K_GRID if 2 * k <= kmax]
    out: dict = {"probe": "ridge", "n": n, "t": t, "seed": seed, "kmax": kmax}
    out.update(sizes=list(sizes), max_reps=max_reps, K=k_values)
    out.update(K_population=K_POPULATION, lambdas=LAMBDAS.tolist(), per_p=[])
    for p in p_grid:
        t0 = time.time()
        circuit = with_mask(gates, u, p)
        shots = sample_inputs(circuit, codes, kmax, seed=int(1e6 * p) + 7919 * seed)
        signal, noise, means = empirical_spectrum(shots[split["train"]])
        rho2 = generalized_spectrum(signal, noise)
        probe = purification(circuit, monomials)
        row: dict = {
            "p": p,
            "n_features": int(shots.shape[2]),
            "rho2": rho2[:400].tolist(),
            "n_learn": [n_learn(rho2, k) for k in [*K_POPULATION, float("inf")]],
            "delta": probe.delta,
            "member_full": probe.member_full,
            "member_mid": probe.member_mid,
            "targets": {},
        }
        groups = {
            k: {part: shot_groups(shots[sl], k, max_reps) for part, sl in split.items()}
            for k in k_values
        }
        for name, yy_all in y.items():
            yy = {part: yy_all[sl] for part, sl in split.items()}
            ytr = yy["train"]
            cov = (means - means.mean(axis=0)).T @ (ytr - ytr.mean()) / (n_tr - 1)
            res: dict = {
                "r2_population": [
                    population_r2(signal, noise, cov, k)
                    for k in [*K_POPULATION, float("inf")]
                ],
                "ridge": {},
            }
            for k, feats in groups.items():
                sweep = ridge_sweep(
                    feats["train"], ytr, feats["val"], yy["val"],
                    feats["test"], yy["test"], LAMBDAS,
                )  # fmt: skip
                best = int(np.argmin(sweep.val_mse))
                var = float(np.var(yy["test"]))
                res["ridge"][str(k)] = {
                    "reps": sweep.reps,
                    "lambda_star": float(LAMBDAS[best]),
                    "test_mse": float(sweep.test_mse[best]) / var,
                    "bias2": float(sweep.bias2[best]) / var,
                    "variance": float(sweep.variance[best]) / var,
                    "curve_test_mse": (sweep.test_mse / var).tolist(),
                }
            row["targets"][name] = res
        if dd:
            row["double_descent"] = double_descent(groups, y, split, shots.shape[2])
        out["per_p"].append(row)
        print(
            f"ridge N={n} seed={seed} p={p}: d={shots.shape[2]} "
            f"n_learn={row['n_learn'][:4]} {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def double_descent(groups: dict, y: dict, split: dict, d: int) -> dict:
    """Test error against the number of training inputs, ``K = 16``."""
    k = 16 if 16 in groups else max(groups)
    feats = groups[k]
    n_tr = feats["train"].shape[1]
    out: dict = {"K": k, "d": d, "S": [], "targets": {}}
    sizes = [int(round(f * d)) for f in DD_FRACTIONS if round(f * d) <= n_tr]
    out["S"] = sizes
    for name in ("z_mid", "deg1"):
        ytr, yva, yte = (y[name][split[part]] for part in ("train", "val", "test"))
        var = float(np.var(yte))
        tiny, best, b2, vv = [], [], [], []
        for s in sizes:
            sweep = ridge_sweep(
                feats["train"][:, :s], ytr[:s], feats["val"], yva,
                feats["test"], yte, LAMBDAS,
            )  # fmt: skip
            i = int(np.argmin(sweep.val_mse))
            tiny.append(float(sweep.test_mse[0]) / var)
            best.append(float(sweep.test_mse[i]) / var)
            b2.append(float(sweep.bias2[0]) / var)
            vv.append(float(sweep.variance[0]) / var)
        out["targets"][name] = {
            "mse_lambda_min": tiny,
            "mse_lambda_star": best,
            "bias2_lambda_min": b2,
            "variance_lambda_min": vv,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("probe", choices=("transition", "ridge"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--t", type=int, default=None, help="steps (default 2N)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--kmax", type=int, default=128)
    parser.add_argument("--sizes", type=int, nargs=3, default=(1000, 500, 500))
    parser.add_argument("--max-reps", type=int, default=8)
    parser.add_argument("--dd", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/clifford_ridge"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    t = args.t if args.t is not None else 2 * args.n
    if args.probe == "transition":
        p_grid = args.p or P_TRANSITION
        result = run_transition(args.n, t, p_grid, args.samples, args.seed)
        name = f"transition_N{args.n}_T{t}_s{args.seed}"
    else:
        p_grid = args.p or P_RIDGE
        result = run_ridge(
            args.n, args.seed, p_grid, t, args.kmax, tuple(args.sizes),
            args.max_reps, args.dd,
        )  # fmt: skip
        name = f"ridge_N{args.n}_T{t}_seed{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = args.out / f"{name}{tag}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
