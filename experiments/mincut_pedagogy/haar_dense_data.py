"""Haar, dense inputs, no final readout: rank, min cut and E_circuit (#383).

For ``N = 4, 5, 6``, ``T = 2N`` and ``p`` from 0 to 1, each circuit gives its
exact dense rank (backward recursion), its min cut ``C`` and the circuit error
``E_circuit = 1 - R^2(inf)`` of the population ridge, averaged over 40
isotropic (flat) targets. Each circuit also gives the Pauli-weight profile
of the span, ``d_m = sum_{|a| = m} [Pi]_aa`` (``Pi`` the orthogonal projector in
whitened coordinates), from which both priors follow exactly (ratio of means):
flat, ``sum_m d_m / (4^N - 1)``; iid, ``sum_m 3^(-m) d_m / (2^N - 1)``. The min
cut alone is also computed on many more masks, independent of the circuits,
for the purely geometric prediction. Each point is written to
``<out>.parts/`` as soon as it is done, so an interrupted run can resume.

Usage (from experiments/mincut_pedagogy, about 15 minutes on 4 cores)::

    uv run python haar_dense_data.py figures/haar_dense_data.json
"""

import json
import os
import sys
import time
from math import comb
from multiprocessing import Pool

# One BLAS thread per worker: four workers with four threads each oversubscribe
# the machine and slow the eigendecompositions down by an order of magnitude.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from QELM.evaluation import fourier_framework as ff  # noqa: E402
from QELM.evaluation import r2_budget as rb  # noqa: E402
from QELM.evaluation.mincut_graph import (  # noqa: E402
    input_bound,
    mincut,
    record_bound,
)

SIZES = (4, 5, 6)
CIRCUITS = {4: 60, 5: 40, 6: 30}
GEOMETRY_MASKS = 2000
TARGETS = 40
PS = np.round(np.linspace(0, 1, 21), 3).tolist()


def one_point(task: tuple[int, int]) -> dict:
    """Every circuit of one size ``n`` and one value ``PS[j]``."""
    n, j = task
    p = PS[j]
    depth = 2 * n
    rng = np.random.default_rng((n, j, 383))
    w = ff.input_weights(n, "sphere")
    m = int(np.count_nonzero(w[1:]))
    weight = ff.pauli_weight(ff.pauli_codes(n))[1:]
    mult = np.array([comb(n, k) * 3**k for k in range(1, n + 1)], dtype=float)
    rows = []
    for _ in range(CIRCUITS[n]):
        c = ff.draw_circuit(n, depth, p, "haar", rng)
        basis = rb.effect_basis(c, final=False)
        v = rb.span_basis(basis, w)
        g = rb.draw_whitened_targets(n, TARGETS, "flat", rng, ensemble="sphere")
        r2 = ((g @ v.T) ** 2).sum(1) / (g**2).sum(1)
        diag = (v**2).sum(0)
        d_m = np.array([diag[weight == k].sum() for k in range(1, n + 1)])
        rows.append(
            dict(
                C=mincut(c.mask, final=False).value,
                rank=int(basis.shape[0]),
                d=int(v.shape[0]),
                records=record_bound(c.mask, final=False),
                input=input_bound(c.mask),
                e_circuit=float(1 - r2.mean()),
                d_m=d_m.tolist(),
                frac_m=(d_m / mult).tolist(),
                r2_flat=float(d_m.sum() / (4**n - 1)),
                r2_iid=float((3.0 ** -np.arange(1, n + 1) * d_m).sum() / (2**n - 1)),
            )
        )
    geo = np.random.default_rng((n, j, 777))
    cuts = [
        mincut(geo.random((depth, n)) < p, final=False).value
        for _ in range(GEOMETRY_MASKS)
    ]
    return dict(n=n, p=p, M=m, circuits=rows, geometry_C=cuts)


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1] + ".parts"
    os.makedirs(parts, exist_ok=True)
    tasks = [
        (n, j)
        for n in reversed(SIZES)
        for j in range(len(PS))
        if not os.path.exists(f"{parts}/N{n}_{j:02d}.json")
    ]
    with Pool(4) as pool:
        for res in pool.imap_unordered(one_point, tasks):
            j = PS.index(res["p"])
            json.dump(res, open(f"{parts}/N{res['n']}_{j:02d}.json", "w"))
            print(res["n"], res["p"], f"{time.time() - t0:.0f}s", flush=True)
    out: dict = {str(n): [] for n in SIZES}
    for name in sorted(os.listdir(parts)):
        res = json.load(open(f"{parts}/{name}"))
        out[str(res["n"])].append(res)
    for n in out:
        out[n].sort(key=lambda r: r["p"])
    json.dump(out, open(sys.argv[1], "w"))
