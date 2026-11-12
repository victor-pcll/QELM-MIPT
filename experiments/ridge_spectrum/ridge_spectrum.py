#!/usr/bin/env python3
"""Spectrum of the effects and ridge bias/variance across the MIPT (issue #334).

Two subcommands, on the same Haar circuits (open brickwork, convention of
``experiments/effect_mipt``: one layer is a full period, then a ``Z``
measurement of each site with probability ``p``; depth ``T = 2N``):

- ``level1``: full record, fixed mask ``u < p`` of ``effect_rank.py``.
  Born-drawn records give the spectrum ``mu_k`` of ``F``, the learnable
  directions ``n_learn(K)`` and the population ``R2(K)`` of each target; the
  exact span of the effects gives ``E_perp``. Two records: mid-circuit only
  (``mid``) and with the final readout (``full``).
- ``level2``: local features (``M_{t,i}`` and final ``Z_i``, fresh mask per
  shot) averaged over ``K`` shots, ridge readout, squared bias and variance
  over independent shot draws, ``lambda*`` on a validation set.

Example::

    uv run python experiments/ridge_spectrum/ridge_spectrum.py level1 --n 4 --seed 0
    uv run python experiments/ridge_spectrum/ridge_spectrum.py level2 --n 6 --seed 0

writes ``outputs/ridge_spectrum/level1_N4_seed0.json`` and
``outputs/ridge_spectrum/level2_N6_seed0.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.effect_mipt import draw_haar_circuit
from QELM.evaluation.fourier import bloch_monomials, transfer_matrix
from QELM.evaluation.ridge_spectrum import (
    TWO_PI,
    approximation_bias,
    circle_weights,
    conditional_means,
    degree_target_circle,
    effect_span,
    generalized_spectrum,
    layers_from_gates,
    n_learn,
    record_spectrum,
    ridge_sweep,
    sample_effect,
    sample_shots,
    shot_groups,
    single_shot_r2,
    single_site_target,
    spectral_r2,
    target_values,
)

P_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.5, 0.6]
K_GRID = [1, 4, 16, 64]
K_POPULATION = [1, 4, 16, 64, 256, 1024, 4096, 16384]
LAMBDAS = np.logspace(-8, 2, 31)


def circuit(n: int, seed: int) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Gates and uniform draw ``u`` of the seed, as in ``effect_mipt.py``."""
    depth = 2 * n
    gates = draw_haar_circuit(n, depth, np.random.default_rng((seed, n, 0)))
    u = np.random.default_rng((seed, n, 1000)).random((depth, n))
    return gates, u


def targets(n: int, seed: int) -> dict[str, np.ndarray]:
    """Degree-``k`` targets (``k = 1, 2, 3``) and the middle qubit's X and Z."""
    rng = np.random.default_rng((seed, n, 2000))
    out = {f"k{k}": degree_target_circle(n, k, rng) for k in (1, 2, 3) if k <= n}
    out["x_mid"] = single_site_target(n, n // 2, "x")
    out["z_mid"] = single_site_target(n, n // 2, "z")
    return out


def run_level1(n: int, seed: int, p_grid: list[float], records: int) -> dict:
    gates, u = circuit(n, seed)
    tgt = targets(n, seed)
    k_values: list[float] = [*K_POPULATION, float("inf")]
    out: dict = {
        "level": 1,
        "n": n,
        "depth": 2 * n,
        "seed": seed,
        "records": records,
        "K": [str(k) for k in k_values],
        "per_p": [],
    }
    for p in p_grid:
        masks = u < p
        row: dict = {"p": p}
        for variant, final in (("mid", False), ("full", True)):
            t0 = time.time()
            rng = np.random.default_rng((seed, n, 3000, int(round(1000 * p)), final))
            effects = [sample_effect(gates, masks, final, rng) for _ in range(records)]
            v = np.array([conditional_means(e) for e in effects])
            # Dense Renyi-2 deficit of each record, Delta = N - log2(1 + W).
            purity = np.array([np.sum(np.abs(e) ** 2) for e in effects])
            delta = n - np.log2(2**n * purity)
            spec = record_spectrum(v)
            half = record_spectrum(v[: records // 2])
            span = effect_span(gates, masks, final)
            res: dict = {
                "mu": spec.mu.tolist(),
                "trace_F": float(spec.mu.sum()),
                "effective_rank": float(spec.mu.sum() ** 2 / np.sum(spec.mu**2))
                if spec.mu.sum() > 0
                else 0.0,
                "span_rank": int(span.shape[0]),
                "delta_dense_mean": float(delta.mean()),
                "n_learn": [n_learn(spec.mu, k) for k in k_values],
                "n_learn_half": [n_learn(half.mu, k) for k in k_values],
                "targets": {},
            }
            for name, beta in tgt.items():
                res["targets"][name] = {
                    "e_perp": approximation_bias(span, beta),
                    "r2_single_shot": single_shot_r2(v, beta),
                    "r2_K": [spectral_r2(spec, beta, k) for k in k_values],
                    "r2_K_half": [spectral_r2(half, beta, k) for k in k_values],
                }
            row[variant] = res
            print(
                f"level1 N={n} seed={seed} p={p} {variant}: "
                f"trF={res['trace_F']:.3g} rank={res['span_rank']} "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
        out["per_p"].append(row)
    return out


def run_level2(
    n: int,
    seed: int,
    p_grid: list[float],
    kmax: int,
    sizes: tuple[int, int, int],
    max_reps: int,
) -> dict:
    gates, _ = circuit(n, seed)
    layers = layers_from_gates(gates, n)
    tgt = targets(n, seed)
    n_tr, n_va, n_te = sizes
    rng_in = np.random.default_rng((seed, n, 4000))
    theta = rng_in.uniform(0, TWO_PI, (n_tr + n_va + n_te, n))
    split = {
        "train": slice(0, n_tr),
        "val": slice(n_tr, n_tr + n_va),
        "test": slice(n_tr + n_va, None),
    }
    y = {name: target_values(beta, theta) for name, beta in tgt.items()}
    w = circle_weights(n)
    k_values = [k for k in K_GRID if 2 * k <= kmax]
    out: dict = {
        "level": 2,
        "n": n,
        "depth": 2 * n,
        "seed": seed,
        "kmax": kmax,
        "sizes": list(sizes),
        "max_reps": max_reps,
        "K": k_values,
        "K_population": K_POPULATION,
        "lambdas": LAMBDAS.tolist(),
        "per_p": [],
    }
    for p in p_grid:
        t0 = time.time()
        rng = np.random.default_rng((seed, n, 5000, int(round(1000 * p))))
        shots = sample_shots(layers, n, p, theta, kmax, rng)
        C = transfer_matrix(layers, n, p)
        phi = bloch_monomials(theta) @ C.T
        Cw = C[:, 1:] * w[1:]
        signal = Cw @ C[:, 1:].T
        tr = shots[split["train"]].astype(float)
        noise = np.mean(
            [np.cov(tr[s], rowvar=False, ddof=1) for s in range(tr.shape[0])], axis=0
        )
        rho2 = generalized_spectrum(signal, noise)
        row: dict = {
            "p": p,
            "rho2": rho2.tolist(),
            "n_learn": [n_learn(rho2, k) for k in [*K_POPULATION, float("inf")]],
            "targets": {},
        }
        groups = {
            k: {part: shot_groups(shots[sl], k, max_reps) for part, sl in split.items()}
            for k in k_values
        }
        exact = {part: phi[sl][None] for part, sl in split.items()}
        for name, beta in tgt.items():
            cov = Cw @ beta[1:]
            pinv = np.linalg.pinv(signal, rcond=1e-10, hermitian=True)
            res: dict = {
                "e_perp_local": float(1 - cov @ pinv @ cov),
                "r2_population": [
                    float(
                        cov
                        @ np.linalg.pinv(signal + noise / k, rcond=1e-10, hermitian=True)
                        @ cov
                    )
                    for k in K_POPULATION
                ],
                "ridge": {},
            }
            yy = {part: y[name][sl] for part, sl in split.items()}
            for k, feats in [*groups.items(), ("inf", exact)]:
                sweep = ridge_sweep(
                    feats["train"],
                    yy["train"],
                    feats["val"],
                    yy["val"],
                    feats["test"],
                    yy["test"],
                    LAMBDAS,
                )
                best = int(np.argmin(sweep.val_mse))
                # Errors in units of the test variance of the target: 1 - R2.
                var = float(np.var(yy["test"]))
                res["ridge"][str(k)] = {
                    "reps": sweep.reps,
                    "lambda_star": float(LAMBDAS[best]),
                    "test_mse": float(sweep.test_mse[best]) / var,
                    "bias2": float(sweep.bias2[best]) / var,
                    "variance": float(sweep.variance[best]) / var,
                    "curve_test_mse": (sweep.test_mse / var).tolist(),
                    "curve_bias2": (sweep.bias2 / var).tolist(),
                    "curve_variance": (sweep.variance / var).tolist(),
                }
            row["targets"][name] = res
        out["per_p"].append(row)
        print(
            f"level2 N={n} seed={seed} p={p}: n_learn={row['n_learn']} "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("level", choices=("level1", "level2"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+", default=P_GRID)
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--kmax", type=int, default=256)
    parser.add_argument("--sizes", type=int, nargs=3, default=(1000, 500, 500))
    parser.add_argument("--max-reps", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("outputs/ridge_spectrum"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.level == "level1":
        result = run_level1(args.n, args.seed, args.p, args.records)
    else:
        result = run_level2(
            args.n, args.seed, args.p, args.kmax, tuple(args.sizes), args.max_reps
        )
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = args.out / f"{args.level}_N{args.n}_seed{args.seed}{tag}.json"
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
