"""Pauli-weight profile of the k-bit parity subspace S_k (#388).

Haar gates, dense inputs, no final readout, ``T = 2N``. For each circuit the
span ``S`` of every record (``k = inf``) and the span ``S_k`` of the parities
of at most ``k`` mid-circuit bits (``k = 1, 2, 3``) are built in whitened
coordinates, and each gives its weight profile
``d_m^(k) = sum_{|a| = m} [Pi_k]_aa``. Parity operators are evolved in chunks
and appended to an orthonormal basis, so memory does not grow with the number
of subsets. Each point is written to ``<out>.parts/`` as soon as it is done.

Usage (from experiments/mincut_pedagogy)::

    uv run python sk_profile_data.py figures/sk_profile_data.json
"""

import json
import os
import sys
import time
from itertools import combinations
from math import comb
from multiprocessing import Pool

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from QELM.evaluation import fourier_framework as ff  # noqa: E402
from QELM.evaluation import r2_budget as rb  # noqa: E402
from QELM.evaluation.mincut_graph import mincut  # noqa: E402

SIZES = (4, 5, 6)
CIRCUITS = {4: 40, 5: 30, 6: 20}
PS = (0.1, 0.2, 0.3, 0.5, 0.7)
KS = (1, 2, 3)
CHUNK = 1500
TOL = 1e-9


def extend(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Append to an orthonormal ``basis`` the directions of ``rows`` it misses."""
    norm = np.linalg.norm(rows, axis=1)
    rows = rows[norm > TOL]
    if not len(rows):
        return basis
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    for _ in range(2):  # twice is enough for the Gram-Schmidt step
        if len(basis):
            rows = rows - (rows @ basis.T) @ basis
    _, s, vt = np.linalg.svd(rows, full_matrices=False)
    new = vt[s > 1e-7]
    return np.vstack([basis, new]) if len(basis) else new


def parity_rows(c: ff.Circuit, subsets: list[tuple[int, ...]]) -> np.ndarray:
    """``rb.parity_operator_rows(c, subsets, final=False)``, with matmuls.

    Same backward recursion; the three-operand ``einsum`` of the library is
    replaced by two batched products, which is much faster at ``N = 6``.
    """
    n = c.n
    dim = 2**n
    mid = [(t, int(i)) for t in range(len(c.gates)) for i in np.flatnonzero(c.mask[t])]
    index = {tm: j for j, tm in enumerate(mid)}
    sets = [set(s) for s in subsets]
    ops = np.broadcast_to(np.eye(dim, dtype=complex), (len(sets), dim, dim)).copy()
    for t in reversed(range(len(c.gates))):
        for site in np.flatnonzero(c.mask[t]):
            j = index[(t, int(site))]
            bits = (np.arange(dim) >> (n - 1 - int(site))) & 1
            same = bits[:, None] == bits[None, :]
            one = (bits[:, None] == 1) & same
            flip = np.array([j in s for s in sets])
            ops *= same[None]
            ops[flip] = np.where(one[None], -ops[flip], ops[flip])
        u = np.eye(dim, dtype=complex)
        for left, gate in c.gates[t]:
            u = ff._embed(gate, left, n) @ u
        ops = u.conj().T @ ops @ u
    return np.asarray(ff.pauli_vectors(ops).real)


def sk_bases(c: ff.Circuit, w: np.ndarray, full: int) -> list[np.ndarray]:
    """Orthonormal whitened bases of ``S_1 ⊂ S_2 ⊂ S_3`` (constant removed).

    ``S_k`` extends ``S_{k-1}`` with the parities of exactly ``k`` bits, and the
    loop stops once the basis reaches ``full = dim S``.
    """
    nb = ff.n_bits(c, False)
    basis = np.zeros((0, len(w) - 1))
    out = []
    for k in KS:
        subsets = list(combinations(range(nb), k))
        for i in range(0, len(subsets), CHUNK):
            if basis.shape[0] >= full:
                break
            rows = parity_rows(c, subsets[i : i + CHUNK])
            basis = extend(basis, rows[:, 1:] * np.sqrt(w[1:]))
        out.append(basis)
    return out


def one_point(task: tuple[int, int]) -> dict:
    n, j = task
    p = PS[j]
    rng = np.random.default_rng((n, j, 3830))
    w = ff.input_weights(n, "sphere")
    weight = ff.pauli_weight(ff.pauli_codes(n))[1:]
    rows = []
    for _ in range(CIRCUITS[n]):
        c = ff.draw_circuit(n, 2 * n, p, "haar", rng)
        v_s = rb.span_basis(rb.effect_basis(c, final=False), w)
        bases = [*sk_bases(c, w, v_s.shape[0]), v_s]
        prof = []
        for v in bases:
            diag = (v**2).sum(0)
            prof.append([float(diag[weight == m].sum()) for m in range(1, n + 1)])
        rows.append(
            dict(
                C=mincut(c.mask, final=False).value,
                K=int(c.mask.sum()),
                dims=[int(v.shape[0]) for v in bases],
                d_m=prof,
            )
        )
    mult = [comb(n, m) * 3**m for m in range(1, n + 1)]
    return dict(n=n, p=p, ks=[*KS, "inf"], mult=mult, circuits=rows)


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1] + ".parts"
    os.makedirs(parts, exist_ok=True)
    tasks = [
        (n, j)
        for n in reversed(SIZES)
        for j in range(len(PS))
        if not os.path.exists(f"{parts}/N{n}_{j}.json")
    ]
    with Pool(4) as pool:
        for res in pool.imap_unordered(one_point, tasks):
            j = PS.index(res["p"])
            json.dump(res, open(f"{parts}/N{res['n']}_{j}.json", "w"))
            print(res["n"], res["p"], f"{time.time() - t0:.0f}s", flush=True)
    out: dict = {str(n): [] for n in SIZES}
    for name in sorted(os.listdir(parts)):
        res = json.load(open(f"{parts}/{name}"))
        out[str(res["n"])].append(res)
    for n in out:
        out[n].sort(key=lambda r: r["p"])
    json.dump(out, open(sys.argv[1], "w"))
