"""E_circuit = 1 - R^2(inf) per circuit against the min cut C (issue #383).

Haar circuits, flat (isotropic) targets, 40 targets per circuit. Two settings:
final readout with XZ inputs at T = N (as in #362), and no final readout with
dense inputs at T = 2N (as in #363). N = 3, 4, 5, p = 0 to 0.5.

Usage (from experiments/mincut_pedagogy, about one minute)::

    uv run python ecircuit_data.py figures/ecircuit_data.json
"""

import json
import sys
import time

import numpy as np

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb
from QELM.evaluation.mincut_graph import mincut

out = {}
ps = np.round(np.linspace(0, 0.5, 11), 3).tolist()
CONF = [("final_xz_T=N", True, "xz", 1), ("nofinal_sphere_T=2N", False, "sphere", 2)]
t0 = time.time()
for name, final, ens, ratio in CONF:
    out[name] = {}
    for n in (3, 4, 5):
        T = ratio * n
        reps = 40 if n < 5 else 16
        rng = np.random.default_rng((n, 383, ratio))
        w = ff.input_weights(n, ens)
        M = int(np.count_nonzero(w[1:]))
        rows = []
        for p in ps:
            for _ in range(reps):
                c = ff.draw_circuit(n, T, p, "haar", rng)
                basis = rb.effect_basis(c, final=final)
                v = rb.span_basis(basis, w)
                g = rb.draw_whitened_targets(n, 40, "flat", rng, ensemble=ens)
                # g: (targets, 4^n) whitened coefficients; identity excluded upstream
                proj = g @ v.T
                r2 = (proj**2).sum(1) / (g**2).sum(1)
                rows.append(
                    dict(
                        p=p,
                        C=mincut(c.mask, final).value,
                        dense=int(basis.shape[0]),
                        visible=int(v.shape[0]),
                        e_circuit=float(1 - r2.mean()),
                        e_sd=float(r2.std()),
                        M=M,
                    )
                )
        out[name][n] = rows
        print(name, n, f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(sys.argv[1], "w"))
