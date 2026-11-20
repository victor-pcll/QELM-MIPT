"""Mean min cut against p, exact Haar ranks and S_0 of SRN (issue #383).

Usage (from experiments/mincut_pedagogy)::

    uv run python data.py figures/mincut_data.json
"""

import json
import sys
import time

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb
from QELM.evaluation.mincut_graph import (
    hartley_mincut,
    input_bound,
    mincut,
    record_bound,
)

out = {"mincut": {}, "exact": {}, "s0": {}}
ps = np.round(np.linspace(0, 1, 21), 3).tolist()
t0 = time.time()
for final in (False, True):
    key = "final" if final else "nofinal"
    out["mincut"][key] = {}
    for n in (4, 6, 8, 12, 16, 24, 32):
        T = 2 * n
        reps = {4: 300, 6: 300, 8: 200, 12: 100, 16: 80, 24: 40, 32: 30}[n]
        rng = np.random.default_rng((n, 1 if final else 0))
        rows = []
        for p in ps:
            Cs = []
            cls = [0, 0, 0]
            for _ in range(reps):
                m = rng.random((T, n)) < p
                C = mincut(m, final).value
                K = record_bound(m, final)
                inp = input_bound(m)
                Cs.append(C)
                if C == K:
                    cls[0] += 1
                elif C == inp:
                    cls[1] += 1
                else:
                    cls[2] += 1
            Cs = np.array(Cs)
            rows.append(
                dict(
                    p=p,
                    mean=float(Cs.mean()),
                    sem=float(Cs.std(ddof=1) / np.sqrt(reps)),
                    records=cls[0] / reps,
                    input=cls[1] / reps,
                    inside=cls[2] / reps,
                    reps=reps,
                )
            )
        out["mincut"][key][n] = rows
        print(key, n, f"{time.time() - t0:.0f}s", flush=True)
# exact ranks, Haar, small N
pe = np.round(np.linspace(0, 1, 11), 3).tolist()
for final in (False, True):
    key = "final" if final else "nofinal"
    out["exact"][key] = {}
    for n in (3, 4, 5):
        T = 2 * n
        reps = 16 if n < 5 else 8
        rng = np.random.default_rng((n, 50 + final))
        rows = []
        for p in pe:
            lr, Cs, eq = [], [], 0
            for _ in range(reps):
                c = ff.draw_circuit(n, T, p, "haar", rng)
                r = rb.effect_basis(c, final=final).shape[0]
                C = mincut(c.mask, final).value
                lr.append(np.log2(r))
                Cs.append(C)
                eq += r == 2**C
            rows.append(
                dict(
                    p=p,
                    log2rank=float(np.mean(lr)),
                    sem=float(np.std(lr, ddof=1) / np.sqrt(reps)),
                    C=float(np.mean(Cs)),
                    equal=eq,
                    reps=reps,
                )
            )
        out["exact"][key][n] = rows
        print("exact", key, n, f"{time.time() - t0:.0f}s", flush=True)
# S0 (Skinner) for both geometries
for half in (True, False):
    key = "half" if half else "full"
    out["s0"][key] = {}
    for n in (8, 16, 32, 64):
        T = 4 * n if n <= 32 else 2 * n
        reps = {8: 200, 16: 100, 32: 40, 64: 16}[n]
        rng = np.random.default_rng((n, 90 + half))
        rows = []
        for p in ps:
            v = np.array([hartley_mincut(n, T, p, rng, half) for _ in range(reps)])
            rows.append(
                dict(
                    p=p,
                    mean=float(v.mean()),
                    sem=float(v.std(ddof=1) / np.sqrt(reps)),
                    reps=reps,
                )
            )
        out["s0"][key][n] = rows
        print("s0", key, n, f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(sys.argv[1], "w"))
