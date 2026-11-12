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
    GroupBasis,
    degree_one_values,
    draw_circuit,
    draw_inputs,
    effect_groups,
    effective_dimension,
    empirical_spectrum,
    enumerate_group,
    interval,
    interval_entropies,
    local_visible_weights,
    mean_and_sem,
    monomial_values,
    monomial_weight,
    population_r2,
    purification,
    range_targets,
    sample_inputs,
    targets_for,
    visible_weights,
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
P_POPULATION = [0.02, 0.04, 0.06, *P_TRANSITION, 0.35, 0.4, 0.5]
LAMBDAS_POP = [2.0**-k for k in range(0, 17, 2)]
LOCAL_SPAN = 6
ENUMERATE_MAX_N = 16
LOCAL_STARTS = 8
DD_FRACTIONS = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]


def run_transition(
    n: int, t: int, p_grid: list[float], samples: int, seed: int
) -> dict:
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


def powers_of_two(n: int) -> list[int]:
    """1, 2, 4, ... up to ``n // 2``, and ``n // 2`` itself."""
    out, k = [], 1
    while k <= n // 2:
        out.append(k)
        k *= 2
    if out[-1] != n // 2:
        out.append(n // 2)
    return out


def population_sample(circuit, ranges: list[int], lengths: list[int]) -> dict:
    """K = infinity probes of one circuit, mid-circuit and full records."""
    n = circuit.N
    groups = effect_groups(circuit)
    targets = range_targets(n, ranges)
    starts = np.linspace(0, n, min(LOCAL_STARTS, n), endpoint=False).astype(int)
    out: dict = {}
    for record, gens in (("mid", groups.mid), ("full", groups.full)):
        basis = GroupBasis(gens)
        row: dict = {"rank": basis.rank}
        row["member"] = {k: float(basis.contains(v).mean()) for k, v in targets.items()}
        row["local_rank"] = [
            float(np.mean([basis.local_rank(interval(s, ell, n)) for s in starts]))
            for ell in lengths
        ]
        hist = np.bincount(local_visible_weights(basis, LOCAL_SPAN), minlength=n + 1)
        row["local_visible"] = hist[: LOCAL_SPAN + 1].tolist()
        if n <= ENUMERATE_MAX_N:
            w = visible_weights(enumerate_group(basis.rows))
            row["visible"] = np.bincount(w, minlength=n + 1).tolist()
            row["d_eff"] = effective_dimension(w, LAMBDAS_POP).tolist()
            # Victor's effective rank sum_k mu_k / mu_max of the covariance.
            row["rank_eff"] = float((0.5**w).sum() / 0.5 ** w.min()) if w.size else 0.0
        out[record] = row
    return out


def _mean_tree(rows: list) -> object:
    """Mean and SEM of every leaf of a list of identically shaped trees."""
    first = rows[0]
    if isinstance(first, dict):
        return {k: _mean_tree([r[k] for r in rows]) for k in first}
    a = np.asarray(rows, dtype=float)
    m = a.mean(axis=0)
    e = a.std(axis=0, ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else 0 * m
    return {"mean": np.asarray(m).tolist(), "sem": np.asarray(e).tolist()}


def run_population(
    n: int, t: int, p_grid: list[float], samples: int, seed: int
) -> dict:
    """Ridge at K = infinity, lambda -> 0+: the group of the effect."""
    ranges = powers_of_two(n)
    lengths = [*powers_of_two(n), n]
    out: dict = {"probe": "population", "n": n, "t": t, "samples": samples}
    out.update(seed=seed, p=p_grid, ranges=ranges, lengths=lengths)
    out.update(local_span=LOCAL_SPAN, lambdas=LAMBDAS_POP, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, t, j, 3370))
        rows = []
        for _ in range(samples):
            gates = draw_circuit(n, t, rng)
            u = rng.random((t, n))
            rows.append(population_sample(with_mask(gates, u, p), ranges, lengths))
        res: dict = {"p": p, **_mean_tree(rows)}  # type: ignore[dict-item]
        out["per_p"].append(res)
        mid, full = res["mid"], res["full"]
        print(
            f"population N={n} T={t} p={p}: Delta/N="
            f"{1 - mid['rank']['mean'] / n:.3f} "
            f"Z in G_full={full['member']['Z']['mean']:.3f} "
            f"Z in G_mid={mid['member']['Z']['mean']:.3f} {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def run_locality(n: int, t: int, p_grid: list[float], samples: int, seed: int) -> dict:
    """Entropy profile ``S(l)`` of the references after the full readout.

    ``2^(l - S(l))`` strings of the group of the effect fit in an interval of
    ``l`` sites; ``S`` is averaged over the ``N`` starting sites, then over
    the circuits.
    """
    out: dict = {"probe": "locality", "n": n, "t": t, "samples": samples}
    out.update(seed=seed, p=p_grid, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, t, j, 3390))
        prof = []
        for _ in range(samples):
            circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
            prof.append(interval_entropies(effect_groups(circuit).full).mean(axis=0))
        a = np.asarray(prof)
        out["per_p"].append(
            {
                "p": p,
                "s": a.mean(axis=0).tolist(),
                "s_sem": (a.std(axis=0, ddof=1) / np.sqrt(samples)).tolist(),
            }
        )
        print(
            f"locality N={n} T={t} p={p}: S(N/2)={a[:, n // 2].mean():.2f} "
            f"{time.time() - t0:.1f}s",
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
    parser.add_argument(
        "probe", choices=("transition", "population", "locality", "ridge")
    )
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
    elif args.probe == "population":
        p_grid = args.p or P_POPULATION
        result = run_population(args.n, t, p_grid, args.samples, args.seed)
        name = f"population_N{args.n}_T{t}_s{args.seed}"
    elif args.probe == "locality":
        p_grid = args.p or P_POPULATION
        result = run_locality(args.n, t, p_grid, args.samples, args.seed)
        name = f"locality_N{args.n}_T{t}_s{args.seed}"
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
