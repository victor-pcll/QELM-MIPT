#!/usr/bin/env python3
"""Two-filter decomposition of the ridge error (issue #349).

Clifford QELM of #337 (periodic brickwork, ``Z`` measurement with
probability ``p`` after every half-layer, final readout, ``T = 2N``). Two
readouts of the same record, both at ``K = inf``:

- ``S``: every parity of the record bits. Its features span ``Span G`` (the
  visible strings of the effect group, #344
  ``test_all_parities_reach_the_whole_group``). ``E_circuit`` is the error of
  the ridge on ``S``.
- ``F^(k)``: the parities of at most ``k`` bits (#341). ``E_total(k)`` is the
  error of the ridge on ``F^(k)``, and ``E_readout(k) = ||R_S y - R_F y||^2``.

``m`` identical feature columns are one column scaled by ``sqrt(m)`` for any
ridge, so each readout is stored as its distinct visible strings and their
multiplicities.

Subcommands:

- ``exact`` (``N <= 8``): the whole population of ``4^N`` inputs. For every
  visible monomial and for dense, degree-1 and "basis" targets, the terms
  ``total``, ``circuit``, ``readout`` and the cross term, at ``lambda = 0``
  and ``lambda > 0``, and with a finite training set. The inclusion
  ``F^(k) ⊂ S`` is checked string by string. The error of the best decoder
  of any form (conditional mean on the fibres of the exact features) is
  compared with the linear one.
- ``locality`` (``N <= 16``): no input is drawn. For a Clifford circuit and
  a monomial target ``a``, ``E_circuit = 1[a not in G]`` and
  ``E_readout(k) = 1[a in G, a not in D_k]`` exactly (``exact`` checks it),
  so the terms are counts. Reported per target family and extent, for
  isotropic targets on a window of ``l`` sites, and for targets drawn inside
  ``G`` by extent.
- ``shots`` (``N = 4``): finite ``K`` with stim, to measure the extra
  estimation term.

Example::

    uv run python experiments/clifford_ridge/two_filters.py exact --n 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.clifford_ridge import (
    GroupBasis,
    draw_circuit,
    effect_groups,
    enumerate_group,
    parity_strings,
    pauli_bits,
    sample_inputs,
    with_mask,
)
from QELM.evaluation.two_filters import (
    all_codes,
    any_decoder_error,
    decompose,
    ridge_coef,
    ridge_fit,
    string_values,
    visible,
)

P_GRID = [0.04, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4]
LAMBDAS = [0.0, 2.0**-12, 2.0**-8, 2.0**-4]
TRAIN = [64, 256, 1024]


def ring_range(support: np.ndarray) -> int:
    """Smallest arc of the periodic chain that holds the support."""
    sites = np.flatnonzero(support)
    if sites.size == 0:
        return 0
    n = support.size
    gaps = np.diff(np.concatenate([sites, [sites[0] + n]]))
    return int(n - gaps.max() + 1)


def key(row: np.ndarray) -> bytes:
    return np.packbits(np.asarray(row, dtype=bool)).tobytes()


def parity_dictionaries(
    circuit, k_max: int, chunk: int = 200_000
) -> list[dict[bytes, tuple[np.ndarray, int]]]:
    """Distinct visible strings (and multiplicities) of the parities of <= k bits.

    Returns one dict per ``k = 1 .. k_max``: key -> (string, multiplicity),
    cumulative in ``k``.
    """
    n = circuit.N
    n_bits = circuit.n_mid + n
    acc: dict[bytes, list] = {}
    out = []
    for k in range(1, k_max + 1):
        if k > n_bits:
            out.append({a: (v[0], v[1]) for a, v in acc.items()})
            continue
        combos = itertools.combinations(range(n_bits), k)
        while True:
            block = list(itertools.islice(combos, chunk))
            if not block:
                break
            strings, alive = parity_strings(circuit, block)
            s = strings[alive]
            s = s[visible(s)]
            if s.shape[0] == 0:
                continue
            rows, counts = np.unique(s, axis=0, return_counts=True)
            for row, c in zip(rows, counts, strict=True):
                h = key(row)
                if h in acc:
                    acc[h][1] += int(c)
                else:
                    acc[h] = [row, int(c)]
        out.append({a: (v[0], v[1]) for a, v in acc.items()})
    return out


def group_visible(circuit) -> np.ndarray:
    """Visible strings of the full-record group ``G`` (``N <= 22``)."""
    elems = enumerate_group(GroupBasis(effect_groups(circuit).full).rows)
    return np.asarray(elems[visible(elems)])


def all_visible_monomials(n: int) -> np.ndarray:
    """All ``3^N - 1`` Y-free non-identity strings, ``[x | z]`` rows."""
    rows = []
    for ops in itertools.product((0, 1, 2), repeat=n):
        if not any(ops):
            continue
        rows.append(
            np.concatenate([np.array(ops) == 1, np.array(ops) == 2]).astype(bool)
        )
    return np.array(rows)


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------


def features(codes: np.ndarray, dic: dict[bytes, tuple[np.ndarray, int]]) -> np.ndarray:
    """Columns ``sqrt(m) Tr[rho g]`` of a readout stored as strings + multiplicities."""
    if not dic:
        return np.zeros((codes.shape[0], 0))
    strings = np.array([v[0] for v in dic.values()])
    mult = np.array([v[1] for v in dic.values()], dtype=float)
    return string_values(codes, strings) * np.sqrt(mult)[None, :]


def exact_sample(circuit, k_max: int, rng: np.random.Generator) -> dict:
    n = circuit.N
    codes = all_codes(n)
    bx, bz = (np.asarray(a) for a in _bloch(codes))
    dics = parity_dictionaries(circuit, k_max)
    g_vis = group_visible(circuit)
    g_keys = {key(r) for r in g_vis}
    # S: all parities. Its multiplicities are not needed at lambda = 0; at
    # lambda > 0 we give each string of G multiplicity 1 (one representative
    # parity per mode). F^(k) uses the true multiplicities.
    s_dic = {key(r): (r, 1) for r in g_vis}
    phi_s = features(codes, s_dic)
    out: dict = {"n_bits": circuit.n_mid + n, "G_visible": int(g_vis.shape[0])}
    out["inclusion_violations"] = [int(sum(h not in g_keys for h in d)) for d in dics]
    out["F_size"] = [len(d) for d in dics]

    mono = all_visible_monomials(n)
    y_mono = string_values(codes, mono).astype(float)
    weight = (mono[:, :n] | mono[:, n:]).sum(axis=1)
    extent = np.array([ring_range(r[:n] | r[n:]) for r in mono])
    in_g = np.array([key(r) in g_keys for r in mono])
    # Dense, degree-1 and basis targets.
    coef = rng.normal(size=mono.shape[0])
    dense = y_mono @ (coef / np.sqrt(0.5**weight))
    deg1 = (
        y_mono[:, weight == 1] @ rng.normal(size=int((weight == 1).sum())) * np.sqrt(2)
    )
    basis = bx[:, 0] ** 2 - bz[:, 0] ** 2  # +1 on |+-> of site 0, -1 on |0>, |1>
    named = {"dense": dense, "deg1": deg1, "basis": basis}

    pred_s0 = ridge_fit(phi_s, y_mono, 0.0)
    res_k = []
    for d in dics:
        phi_f = features(codes, d)
        pred_f0 = ridge_fit(phi_f, y_mono, 0.0)
        var = np.mean(y_mono**2, axis=0)
        a = y_mono - pred_s0
        b = pred_s0 - pred_f0
        tot = np.mean((y_mono - pred_f0) ** 2, axis=0) / var
        circ = np.mean(a**2, axis=0) / var
        read = np.mean(b**2, axis=0) / var
        cross = 2 * np.mean(a * b, axis=0) / var
        in_f = np.array([key(r) in d for r in mono])
        row: dict = {
            "mono_total": float(tot.mean()),
            "mono_circuit": float(circ.mean()),
            "mono_readout": float(read.mean()),
            "mono_max_abs_cross": float(np.abs(cross).max()),
            # Monomial E_perp is the indicator of the readout predicted in #337/#341.
            "mono_max_dev_circuit_indicator": float(np.abs(circ - (~in_g)).max()),
            "mono_max_dev_total_indicator": float(np.abs(tot - (~in_f)).max()),
            "by_weight": {
                "circuit": [float(circ[weight == w].mean()) for w in range(1, n + 1)],
                "readout": [float(read[weight == w].mean()) for w in range(1, n + 1)],
            },
            "by_extent": {
                "circuit": [
                    float(circ[extent == e].mean()) if (extent == e).any() else None
                    for e in range(1, n + 1)
                ],
                "readout": [
                    float(read[extent == e].mean()) if (extent == e).any() else None
                    for e in range(1, n + 1)
                ],
            },
        }
        # Named targets at every lambda, on the population.
        for name, y in named.items():
            per_lam = []
            for lam in LAMBDAS:
                dec = decompose(ridge_fit(phi_f, y, lam), ridge_fit(phi_s, y, lam), y)
                per_lam.append(dec.__dict__)
            row[name] = per_lam
            # Finite training set, exact features, lambda = 2^-8, tested on the
            # population.
            per_n = []
            for m in TRAIN:
                idx = rng.integers(codes.shape[0], size=m)
                cf = ridge_coef(phi_f[idx], y[idx], 2.0**-8)
                cs = ridge_coef(phi_s[idx], y[idx], 2.0**-8)
                pf = phi_f @ cf if cf.size else np.zeros_like(y)
                ps = phi_s @ cs if cs.size else np.zeros_like(y)
                per_n.append(decompose(pf, ps, y).__dict__)
            row[f"{name}_train"] = per_n
        # Best decoder of any form on the exact features.
        any_f = np.array(
            [any_decoder_error(phi_f, y_mono[:, j]) for j in range(mono.shape[0])]
        )
        row["any_mono_mean"] = float(any_f.mean())
        row["lin_mono_mean"] = float(tot.mean())
        row["any_beats_lin_mono"] = float(np.mean(any_f < tot - 1e-9))
        # Group generated by the readout strings: outside it, no decoder of
        # any form can do better than 1 (Pauli-conjugation symmetry).
        if d:
            closure = GroupBasis(np.array([v[0] for v in d.values()]))
            in_closure = closure.contains(mono)
        else:
            in_closure = np.zeros(mono.shape[0], dtype=bool)
        row["closure_size_visible"] = int(in_closure.sum())
        row["any_outside_closure_min"] = (
            float(any_f[~in_closure].min()) if (~in_closure).any() else None
        )
        row["any_recovered_in_closure_not_F"] = float(
            np.sum((any_f < 1 - 1e-9) & in_closure & ~in_f)
        )
        row["in_closure_not_F"] = float(np.sum(in_closure & ~in_f))
        row["any_basis"] = any_decoder_error(phi_f, basis)
        res_k.append(row)
    out["k"] = res_k
    any_s = np.array(
        [any_decoder_error(phi_s, y_mono[:, j]) for j in range(mono.shape[0])]
    )
    out["S_any_outside_G_min"] = float(any_s[~in_g].min()) if (~in_g).any() else None
    out["S_any_basis"] = any_decoder_error(phi_s, basis)
    out["S_lin_basis"] = float(
        np.mean((basis - ridge_fit(phi_s, basis, 0.0)) ** 2) / np.var(basis)
    )
    # Same theorem for the mid-circuit record alone (group of rank N - Delta).
    mid = enumerate_group(GroupBasis(effect_groups(circuit).mid).rows)
    mid = mid[visible(mid)]
    phi_mid = (
        string_values(codes, mid) if mid.shape[0] else np.zeros((codes.shape[0], 0))
    )
    mid_keys = {key(r) for r in mid}
    out_mid = np.array([key(r) not in mid_keys for r in mono])
    any_mid = np.array(
        [any_decoder_error(phi_mid, y_mono[:, j]) for j in np.flatnonzero(out_mid)]
    )
    out["mid_any_outside_Gmid_min"] = float(any_mid.min()) if any_mid.size else None
    return out


def _bloch(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from QELM.evaluation.clifford_ridge import bloch_components

    return bloch_components(codes)


def mean_tree(rows: list) -> object:
    first = rows[0]
    if isinstance(first, dict):
        return {k: mean_tree([r[k] for r in rows]) for k in first}
    if isinstance(first, list):
        return [mean_tree([r[j] for r in rows]) for j in range(len(first))]
    if first is None or any(r is None for r in rows):
        vals = [r for r in rows if r is not None]
        if not vals:
            return None
        rows = vals
    a = np.asarray(rows, dtype=float)
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "max": float(a.max())}


def run_exact(n: int, p_grid: list[float], samples: int, k_max: int, seed: int) -> dict:
    t = 2 * n
    out: dict = {"probe": "exact", "n": n, "t": t, "samples": samples, "seed": seed}
    out.update(p=p_grid, k_max=k_max, lambdas=LAMBDAS, train=TRAIN, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, j, 3490))
        rows = []
        for _ in range(samples):
            circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
            rows.append(exact_sample(circuit, k_max, rng))
        res = {"p": p, **mean_tree(rows)}  # type: ignore[dict-item]
        out["per_p"].append(res)
        ks = res["k"]
        print(
            f"exact N={n} p={p}: max|cross| (lambda=0, monomials) k=1..{k_max}: "
            + " ".join(f"{r['mono_max_abs_cross']['max']:.1e}" for r in ks)
            + " | E_circ mono="
            + f"{ks[0]['mono_circuit']['mean']:.3f} E_read(k) mono="
            + " ".join(f"{r['mono_readout']['mean']:.3f}" for r in ks)
            + f" | viol={[v['max'] for v in res['inclusion_violations']]}"
            + f" {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# locality
# ---------------------------------------------------------------------------


def window_mask(n: int, start: int, length: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[[(start + q) % n for q in range(length)]] = True
    return m


def fixed_families(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Fixed monomial targets, ``N`` rows per family (one per first site)."""
    fam: dict[str, list] = {}
    for i in range(n):
        fam.setdefault("Z", []).append(pauli_bits(n, {i: "Z"}))
        fam.setdefault("ZZ_1", []).append(pauli_bits(n, {i: "Z", (i + 1) % n: "Z"}))
        fam.setdefault("ZZ_N/2", []).append(
            pauli_bits(n, {i: "Z", (i + n // 2) % n: "Z"})
        )
        for ell in sorted({2, 4, n // 2, n}):
            fam.setdefault(f"Z^{ell}", []).append(
                pauli_bits(n, {(i + q) % n: "Z" for q in range(ell)})
            )
            ops = rng.choice(["X", "Z"], size=ell)
            fam.setdefault(f"rand^{ell}", []).append(
                pauli_bits(n, {(i + q) % n: str(ops[q]) for q in range(ell)})
            )
    return {k: np.stack(v) for k, v in fam.items()}


def locality_sample(circuit, k_max: int, rng: np.random.Generator) -> dict:
    n = circuit.N
    dics = parity_dictionaries(circuit, k_max)
    g_vis = group_visible(circuit)
    g_keys = {key(r) for r in g_vis}
    support = g_vis[:, :n] | g_vis[:, n:]
    extent = np.array([ring_range(s) for s in support], dtype=int)
    reached = np.array([[key(r) in d for r in g_vis] for d in dics]).reshape(
        len(dics), -1
    )
    out: dict = {
        "G_visible": int(g_vis.shape[0]),
        "inclusion_violations": [int(sum(h not in g_keys for h in d)) for d in dics],
        "G_extent_hist": np.bincount(extent, minlength=n + 1)[1:].tolist(),
        # Targets drawn uniformly inside G among the strings of extent e:
        # E_circuit = 0 and E_readout(k) = fraction of them not in D_k.
        "inG_readout_by_extent": [
            [
                float(1 - reached[k][extent == e].mean())
                if (extent == e).any()
                else None
                for e in range(1, n + 1)
            ]
            for k in range(len(dics))
        ],
    }
    fams = fixed_families(n, rng)
    fam_out = {}
    for name, rows in fams.items():
        in_g = np.array([key(r) in g_keys for r in rows])
        in_d = np.array([[key(r) in d for r in rows] for d in dics])
        fam_out[name] = {
            "circuit": float(1 - in_g.mean()),
            "readout": [float((in_g & ~row).mean()) for row in in_d],
        }
    out["families"] = fam_out
    # Isotropic target on a window of l sites: every visible monomial of the
    # window with equal variance. E_circuit(l) = 1 - n_G(W) / (3^l - 1).
    lengths = [ell for ell in (1, 2, 3, 4, 6, 8, 12, 16) if ell <= n]
    iso_c, iso_r = [], [[] for _ in dics]
    for ell in lengths:
        c_vals, r_vals = [], [[] for _ in dics]
        for start in range(n):
            w = window_mask(n, start, ell)
            inside = ~(support & ~w).any(axis=1)
            total = 3**ell - 1
            n_g = int(inside.sum())
            c_vals.append(1 - n_g / total)
            for k in range(len(dics)):
                r_vals[k].append((n_g - int((inside & reached[k]).sum())) / total)
        iso_c.append(float(np.mean(c_vals)))
        for k in range(len(dics)):
            iso_r[k].append(float(np.mean(r_vals[k])))
    out["iso_lengths_circuit"] = iso_c
    out["iso_lengths_readout"] = iso_r
    return out


def run_locality(
    n: int, p_grid: list[float], samples: int, k_max: int, seed: int
) -> dict:
    t = 2 * n
    out: dict = {"probe": "locality", "n": n, "t": t, "samples": samples, "seed": seed}
    lengths = [ell for ell in (1, 2, 3, 4, 6, 8, 12, 16) if ell <= n]
    out.update(p=p_grid, k_max=k_max, iso_lengths=lengths, per_p=[])
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, j, 3491))
        rows = []
        for _ in range(samples):
            circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
            rows.append(locality_sample(circuit, k_max, rng))
        res = {"p": p, **mean_tree(rows)}  # type: ignore[dict-item]
        out["per_p"].append(res)
        fz = res["families"]["Z"]
        print(
            f"locality N={n} p={p}: Z: E_circ={fz['circuit']['mean']:.3f} "
            "E_read(k)="
            + " ".join(f"{r['mean']:.3f}" for r in fz["readout"])
            + f" | G visible={res['G_visible']['mean']:.0f}"
            + f" viol={[v['max'] for v in res['inclusion_violations']]}"
            + f" {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------


def run_shots(
    p_grid: list[float],
    samples: int,
    k: int,
    shots: list[int],
    seed: int,
    train: int = 256,
    lam: float = 2.0**-8,
) -> dict:
    """Finite K at N = 4: ridge on empirical parity means, tested on fresh shots.

    ``S`` uses one representative parity per visible string of ``G`` (the
    first subset of bits found, in order of size); ``F^(k)`` uses all the
    parities of at most ``k`` bits. Both are trained on ``train`` inputs and
    tested on all ``4^N`` inputs with fresh shots. The reference is the same
    split at ``K = inf`` with the same training inputs.
    """
    n = 4
    t = 2 * n
    out: dict = {"probe": "shots", "n": n, "t": t, "samples": samples, "seed": seed}
    out.update(p=p_grid, k=k, shots=shots, train=train, lam=lam, per_p=[])
    codes = all_codes(n)
    for j, p in enumerate(p_grid):
        t0 = time.time()
        rng = np.random.default_rng((seed, n, j, 3492))
        rows = []
        for _ in range(samples):
            circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
            n_bits = circuit.n_mid + n
            if n_bits > 18:
                continue
            sets = [
                list(c) for r in range(1, n_bits + 1)
                for c in itertools.combinations(range(n_bits), r)
            ]  # fmt: skip
            strings, alive = parity_strings(circuit, sets)
            ok = alive & visible(strings)
            reps: dict[bytes, int] = {}
            for i in np.flatnonzero(ok):
                reps.setdefault(key(strings[i]), int(i))
            s_sets = [sets[i] for i in reps.values()]
            f_sets = [sets[i] for i in np.flatnonzero(ok) if len(sets[i]) <= k]
            exact_s = string_values(codes, strings[list(reps.values())])
            f_idx = [i for i in np.flatnonzero(ok) if len(sets[i]) <= k]
            exact_f = string_values(codes, strings[f_idx])
            idx = rng.integers(codes.shape[0], size=train)
            # Dense target over all visible monomials.
            mono = all_visible_monomials(n)
            w = (mono[:, :n] | mono[:, n:]).sum(axis=1)
            c = rng.normal(size=mono.shape[0]) / np.sqrt(0.5**w)
            y = string_values(codes, mono) @ c
            row: dict = {}
            for kk in shots:
                feats = {}
                for tag, sel in (("S", s_sets), ("F", f_sets)):
                    mats = []
                    for split in (0, 1):
                        rec = sample_inputs(
                            circuit, codes, kk, seed=int(rng.integers(1 << 30)) + split
                        ).astype(np.int64)
                        par = np.stack(
                            [np.prod(rec[:, :, s], axis=2).mean(axis=1) for s in sel],
                            axis=1,
                        ) if sel else np.zeros((codes.shape[0], 0))  # fmt: skip
                        mats.append(par)
                    feats[tag] = mats
                cf = ridge_coef(feats["F"][0][idx], y[idx], lam)
                cs = ridge_coef(feats["S"][0][idx], y[idx], lam)
                pf = feats["F"][1] @ cf if cf.size else np.zeros_like(y)
                ps = feats["S"][1] @ cs if cs.size else np.zeros_like(y)
                row[f"K{kk}"] = decompose(pf, ps, y).__dict__
            cf = ridge_coef(exact_f[idx], y[idx], lam)
            cs = ridge_coef(exact_s[idx], y[idx], lam)
            pf = exact_f @ cf if cf.size else np.zeros_like(y)
            ps = exact_s @ cs if cs.size else np.zeros_like(y)
            row["Kinf"] = decompose(pf, ps, y).__dict__
            pf0 = ridge_fit(exact_f, y, 0.0)
            ps0 = ridge_fit(exact_s, y, 0.0)
            row["population"] = decompose(pf0, ps0, y).__dict__
            rows.append(row)
        res = {"p": p, "used": len(rows), **mean_tree(rows)}  # type: ignore[dict-item]
        out["per_p"].append(res)
        print(
            f"shots p={p} ({len(rows)} circuits): population total="
            f"{res['population']['total']['mean']:.3f}; "
            + " ".join(
                f"K={kk}: total={res[f'K{kk}']['total']['mean']:.3f} "
                f"cross={res[f'K{kk}']['cross']['mean']:+.3f}"
                for kk in shots
            )
            + f" {time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("probe", choices=("exact", "locality", "shots"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--p", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--shots", type=int, nargs="+", default=[4, 16, 64, 256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/two_filters"))
    args = parser.parse_args(argv)
    p_grid = args.p or P_GRID
    if args.probe == "exact":
        res = run_exact(args.n, p_grid, args.samples, args.k_max, args.seed)
        name = f"exact_N{args.n}_s{args.seed}.json"
    elif args.probe == "locality":
        res = run_locality(args.n, p_grid, args.samples, args.k_max, args.seed)
        name = f"locality_N{args.n}_s{args.seed}.json"
    else:
        res = run_shots(p_grid, args.samples, args.k_max, args.shots, args.seed)
        name = f"shots_N4_s{args.seed}.json"
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / name
    path.write_text(json.dumps(res))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
