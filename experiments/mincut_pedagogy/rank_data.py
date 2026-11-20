"""Exact dense rank (Haar and Clifford, same masks) against the min cut (#383).

Two settings, as in ``ecircuit_data.py``: final readout at ``T = N`` and no
final readout at ``T = 2N``. ``N = 3, 4, 5``, ``p`` from 0 to 1.

Usage (from experiments/mincut_pedagogy, a few minutes)::

    uv run python rank_data.py figures/rank_data.json
"""

import json
import sys
import time

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb
from QELM.evaluation.mincut_graph import input_bound, mincut, record_bound

ps = np.round(np.linspace(0, 1, 11), 3).tolist()
CONF = [("final_T=N", True, 1), ("nofinal_T=2N", False, 2)]
out: dict = {}
t0 = time.time()
for name, final, ratio in CONF:
    out[name] = {}
    for n in (3, 4, 5):
        depth = ratio * n
        reps = 24 if n < 5 else 10
        rng = np.random.default_rng((n, 384, ratio))
        rows = []
        for p in ps:
            for _ in range(reps):
                haar = ff.draw_circuit(n, depth, p, "haar", rng)
                cliff = ff.with_mask(
                    ff.draw_circuit(n, depth, 0.0, "clifford", rng), haar.mask
                )
                rows.append(
                    dict(
                        p=p,
                        C=mincut(haar.mask, final).value,
                        records=record_bound(haar.mask, final),
                        input=input_bound(haar.mask),
                        haar=int(rb.effect_basis(haar, final=final).shape[0]),
                        clifford=int(rb.effect_basis(cliff, final=final).shape[0]),
                    )
                )
        out[name][n] = rows
        print(name, n, f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(sys.argv[1], "w"))
