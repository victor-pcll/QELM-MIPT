#!/usr/bin/env python3
"""The ridge on record features as an exact Fourier filter (issue #341).

Clifford QELM of issue #337 (periodic brickwork, ``Z`` measurement with
probability ``p`` after every half-layer, final readout, ``T = 2N``). The
feature of a parity of measured bits is ``Tr[rho(x) O_S]``, where ``O_S`` is
one Pauli string or zero (:func:`parity_strings`). The Fourier transfer
matrix ``C`` of a readout is a selection matrix. A mode ``a`` present ``m``
times is transmitted with ``m w_a / (m w_a + lambda)`` at ``K = inf``. A mode
measured by a single feature is transmitted with ``K w / (1 + (K - 1) w)``
at ``K`` shots (checked by ``validate``).

Subcommands:

- ``transfer``: per ``p``, a fresh circuit per sample. Reports:
  - the dictionary of the local readout (one feature per bit, ``L1``);
  - with ``--pairs``, the dictionary of the bits and pairs of bits
    (``L1 + L2``);
  - both compared with the full-record group ``G`` of #337;
  - for each, the weight and range of the visible modes, the fraction of
    killed bits, the membership of the targets, and the filter
    ``d_eff(lambda)`` and ``n(K)``.
- ``validate``: stim samples. Compares the empirical population ``R2`` of
  monomial targets on the local features, from the signal and noise
  covariances, with the filter prediction.

Example::

    uv run python experiments/clifford_ridge/fourier_filter.py transfer --n 16 --pairs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.clifford_ridge import (
    GroupBasis,
    draw_circuit,
    draw_inputs,
    effect_groups,
    empirical_spectrum,
    enumerate_group,
    local_strings,
    monomial_values,
    parity_strings,
    pauli_bits,
    population_r2,
    sample_inputs,
    visible_weights,
    with_mask,
)

P_GRID = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.25,
          0.3, 0.35, 0.4, 0.5]  # fmt: skip
LAMBDAS = [2.0**-k for k in range(0, 17, 2)]
K_GRID = [1, 4, 16, 64, 256, 1024, 4096]
FAMILIES = ("Z", "X", "ZZ_1", "XX_1", "ZX_1")


def ring_range(support: np.ndarray) -> int:
    """Smallest arc of the periodic chain that holds the support."""
    sites = np.flatnonzero(support)
    if sites.size == 0:
        return 0
    n = support.size
    gaps = np.diff(np.concatenate([sites, [sites[0] + n]]))
    return int(n - gaps.max() + 1)


def family_rows(n: int) -> dict[str, np.ndarray]:
    out = {
        "Z": [pauli_bits(n, {i: "Z"}) for i in range(n)],
        "X": [pauli_bits(n, {i: "X"}) for i in range(n)],
        "ZZ_1": [pauli_bits(n, {i: "Z", (i + 1) % n: "Z"}) for i in range(n)],
        "XX_1": [pauli_bits(n, {i: "X", (i + 1) % n: "X"}) for i in range(n)],
        "ZX_1": [pauli_bits(n, {i: "Z", (i + 1) % n: "X"}) for i in range(n)],
    }
    return {k: np.stack(v) for k, v in out.items()}


def dictionary_stats(strings: np.ndarray, alive: np.ndarray, n: int) -> dict:
    """Visible modes of a readout and their filter, normalised per site."""
    s = strings[alive]
    x, z = s[:, :n], s[:, n:]
    visible = ~(x & z).any(axis=1) & (x | z).any(axis=1)
    keys, counts = np.unique(s[visible], axis=0, return_counts=True)
    support = keys[:, :n] | keys[:, n:]
    weights = support.sum(axis=1)
    ranges = np.array([ring_range(r) for r in support], dtype=int)
    w = 0.5**weights
    m = counts.astype(float)
    known = {k.tobytes() for k in keys}
    member = {
        fam: float(np.mean([r.tobytes() in known for r in rows]))
        for fam, rows in family_rows(n).items()
    }
    return {
        "alive_fraction": float(alive.mean()) if alive.size else 0.0,
        "modes_per_site": keys.shape[0] / n,
        "multiplicity": float(m.mean()) if m.size else 0.0,
        "weight_hist": (np.bincount(weights, minlength=n + 1) / n).tolist(),
        "range_hist": (np.bincount(ranges, minlength=n + 1) / n).tolist(),
        "member": member,
        "d_eff": [float(np.sum(m * w / (m * w + lam)) / n) for lam in LAMBDAS],
        "n_K": [float(np.sum(k * w / (1 + (k - 1) * w)) / n) for k in K_GRID],
    }


def transfer_sample(circuit, pairs: bool) -> dict:
    n = circuit.N
    strings, alive = local_strings(circuit)
    n_mid = circuit.n_mid
    out = {
        "L1": dictionary_stats(strings, alive, n),
        "killed_mid": float(1 - alive[:n_mid].mean()) if n_mid else 0.0,
        "killed_final": float(1 - alive[n_mid:].mean()),
    }
    if pairs:
        nb = strings.shape[0]
        sets = [[j, k] for j in range(nb) for k in range(j + 1, nb)]
        s2, a2 = parity_strings(circuit, sets)
        out["L12"] = dictionary_stats(
            np.vstack([strings, s2]), np.concatenate([alive, a2]), n
        )
    basis = GroupBasis(effect_groups(circuit).full)
    out["G_member"] = {
        fam: float(basis.contains(rows).mean()) for fam, rows in family_rows(n).items()
    }
    if n <= 16:
        w = visible_weights(enumerate_group(basis.rows))
        out["G_modes_per_site"] = w.size / n
        out["G_weight_hist"] = (np.bincount(w, minlength=n + 1) / n).tolist()
    return out


def mean_tree(rows: list) -> object:
    first = rows[0]
    if isinstance(first, dict):
        return {k: mean_tree([r[k] for r in rows]) for k in first}
    a = np.asarray(rows, dtype=float)
    sem = a.std(axis=0, ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else 0 * a[0]
    return {
        "mean": np.asarray(a.mean(axis=0)).tolist(),
        "sem": np.asarray(sem).tolist(),
    }


def run_transfer(
    n: int, p_grid: list[float], samples: int, pairs: bool, seed: int
) -> dict:
    t = 2 * n
    out: dict = {"probe": "transfer", "n": n, "t": t, "samples": samples, "seed": seed}
    out.update(p=p_grid, lambdas=LAMBDAS, K=K_GRID, pairs=pairs, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, j, 3410))
        rows = [
            transfer_sample(
                with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p), pairs
            )
            for _ in range(samples)
        ]
        res = {"p": p, **mean_tree(rows)}  # type: ignore[dict-item]
        out["per_p"].append(res)
        l1 = res["L1"]
        print(
            f"transfer N={n} p={p}: L1 modes/site={l1['modes_per_site']['mean']:.3f} "
            f"killed final={res['killed_final']['mean']:.3f} "
            f"Z in L1={l1['member']['Z']['mean']:.3f} "
            f"Z in G={res['G_member']['Z']['mean']:.3f} {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def run_validate(
    n: int, p_grid: list[float], circuits: int, inputs: int, kmax: int, seed: int
) -> dict:
    """Empirical population R2 on local features against the filter."""
    t = 2 * n
    k_values = [k for k in (1, 4, 16, 64, 256) if k <= kmax]
    out: dict = {"probe": "validate", "n": n, "t": t, "kmax": kmax, "K": k_values}
    out.update(inputs=inputs, circuits=circuits, p=p_grid, points=[])
    for j, p in enumerate(p_grid):
        for c in range(circuits):
            t0 = time.time()
            rng = np.random.default_rng((seed, n, j, c, 3411))
            circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
            strings, alive = local_strings(circuit)
            codes = draw_inputs(inputs, n, rng)
            shots = sample_inputs(circuit, codes, kmax, seed=int(rng.integers(1 << 30)))
            signal, noise, means = empirical_spectrum(shots)
            # Distinct visible modes of the local readout and their multiplicity.
            s = strings[alive]
            x, z = s[:, :n], s[:, n:]
            vis = ~(x & z).any(axis=1) & (x | z).any(axis=1)
            keys, counts = np.unique(s[vis], axis=0, return_counts=True)
            for key, m in zip(keys, counts, strict=True):
                ops = {int(q): "X" for q in np.flatnonzero(key[:n])}
                ops |= {int(q): "Z" for q in np.flatnonzero(key[n:])}
                y = monomial_values(codes, ops)
                cov = ((means - means.mean(axis=0)).T @ (y - y.mean())) / (inputs - 1)
                cov /= y.std(ddof=1)
                r2 = [population_r2(signal, noise, cov, k) for k in k_values]
                w = 0.5 ** len(ops)
                pred = [k * w / (1 + (k - 1) * w) for k in k_values]
                out["points"].append(
                    {"p": p, "circuit": c, "weight": len(ops), "m": int(m),
                     "r2": r2, "pred_single": pred}
                )  # fmt: skip
            print(
                f"validate N={n} p={p} circuit={c}: {len(keys)} modes "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("probe", choices=("transfer", "validate"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--circuits", type=int, default=6)
    parser.add_argument("--inputs", type=int, default=3000)
    parser.add_argument("--kmax", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, default=Path("outputs/fourier_filter"))
    args = parser.parse_args()
    if args.probe == "transfer":
        res = run_transfer(
            args.n, args.p or P_GRID, args.samples, args.pairs, args.seed
        )
    else:
        res = run_validate(
            args.n, args.p or [0.08, 0.16, 0.3], args.circuits, args.inputs,
            args.kmax, args.seed,
        )  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = args.out / f"{args.probe}_N{args.n}_s{args.seed}{tag}.json"
    path.write_text(json.dumps(res))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
