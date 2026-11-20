"""Circuit drawings of the min cut, N = 4, with and without final readout (issue #383).

Usage (from experiments/mincut_pedagogy)::

    uv run python fig_circuits.py figures
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from draw import BLUE_IN, GREY, REC, RED, draw
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb

OUT = sys.argv[1]
rng = np.random.default_rng(2024)


def mk(T, n, cells):
    m = np.zeros((T, n), bool)
    for t, s in cells:
        m[t - 1, s] = True  # t = layer 1..T
    return m


def exact(m, final):
    T, n = m.shape
    c = ff.with_mask(ff.draw_circuit(n, T, 0, "haar", rng), m)
    b = rb.effect_basis(c, final=final)
    vis = rb.span_basis(b, ff.input_weights(n, "xz")).shape[0]
    return b.shape[0], vis


def legend_handles(final):
    h = [
        Line2D(
            [],
            [],
            color="#222222",
            lw=1.6,
            label="free quantum wire: cost 2 ({I,X,Y,Z})",
        ),
        Line2D(
            [],
            [],
            color="#7f7f7f",
            lw=1.0,
            ls=(0, (2, 1.5)),
            label="wire touching a Z measurement: cost 1 ({I,Z})",
        ),
        Line2D(
            [],
            [],
            color=REC,
            lw=1.2,
            marker=">",
            ms=4,
            label="record bit (click): cost 1",
        ),
        Line2D(
            [],
            [],
            color=RED,
            lw=3,
            marker="x",
            ms=8,
            mew=2,
            label="edge cut by the scissors (cost in red)",
        ),
        Patch(fc=BLUE_IN, ec="#555555", label="input side of the cut"),
        Patch(fc=GREY, ec="#555555", label="record side (or irrelevant)"),
    ]
    if not final:
        h.append(
            Line2D(
                [], [], color="#bbbbbb", ls=":", label="traced out (no final readout)"
            )
        )
    else:
        h.append(
            Line2D(
                [],
                [],
                color="#222222",
                lw=1.6,
                marker="s",
                mfc="white",
                ms=6,
                label="final Z readout: cost 1 per site",
            )
        )
    return h


summary = {}
# ---------------- Figure 1: no final readout, N = 4, T = 8 -----------------
n, T = 4, 8
cases = [
    ("(a) $p = 0$: no measurement", mk(T, n, [])),
    ("(b) $K = 1$: q2 at $t = 5$", mk(T, n, [(5, 2)])),
    ("(c) $K = 2$: q1 at $t = 2$, q3 at $t = 6$", mk(T, n, [(2, 1), (6, 3)])),
    (
        "(d) $K = 3$ but only 2 bits:\nq0, q1 at $t = 4$, then q0 at $t = 5$",
        mk(T, n, [(4, 0), (4, 1), (5, 0)]),
    ),
    ("(e) Zeno, $p = 1$: $K = 32$", np.ones((T, n), bool)),
    (
        "(f) intermediate: $K = 7$, $n_1 = 0$",
        mk(T, n, [(2, 1), (2, 2), (3, 1), (4, 0), (5, 2), (6, 2), (7, 0)]),
    ),
]
fig, axes = plt.subplots(1, 6, figsize=(19, 5.6))
summary["nofinal"] = []
for ax, (title, m) in zip(axes, cases):
    r, vis = exact(m, False)
    C, S, cut = draw(ax, m, final=False, title=title, rank=r)
    summary["nofinal"].append(
        dict(
            title=title,
            C=C,
            rank=r,
            K=int(m.sum()),
            n1=int(m[0].sum()),
            visible_xz_minus_const=vis,
        )
    )
for ax in axes[1:]:
    ax.set_ylabel("")
fig.suptitle(
    "Min cut between the input (bottom) and the records, open brickwork, "
    "$N = 4$, $T = 8$, Z measurements after each layer, no final readout",
    fontsize=11,
)
fig.legend(
    handles=legend_handles(False),
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=8.8,
    labelspacing=0.9,
)
fig.tight_layout()
fig.savefig(f"{OUT}/mincut_circuits_nofinal.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: with final readout, N = 4, T = 4 --------------
n, T = 4, 4
cases = [
    ("(a) $p = 0$: final readout only", mk(T, n, [])),
    ("(b) $K = 1$: q2 at $t = 2$", mk(T, n, [(2, 2)])),
    ("(c) $K = 2$: q1 at $t = 1$, q3 at $t = 3$", mk(T, n, [(1, 1), (3, 3)])),
    ("(d) last layer is free:\nq0 at $t = 4$ repeats the readout", mk(T, n, [(4, 0)])),
    ("(e) Zeno, $p = 1$", np.ones((T, n), bool)),
    ("(f) $C = 7$: $K = 4$", mk(T, n, [(1, 0), (2, 2), (3, 1), (3, 3)])),
]
fig, axes = plt.subplots(1, 6, figsize=(19, 4.4))
summary["final"] = []
for ax, (title, m) in zip(axes, cases):
    r, vis = exact(m, True)
    C, S, cut = draw(ax, m, final=True, title=title, rank=r)
    ax.text(
        0.5,
        -0.13,
        f"XZ rank without constant: {vis}"
        + (
            f" $= 2^{{{C}}} - 1$"
            if vis == 2**C - 1
            else rf" $< 2^{{{C}}} - 1$ (XZ cap: $3^4 - 1 = 80$)"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#08306b",
    )
    summary["final"].append(
        dict(
            title=title,
            C=C,
            rank=r,
            K=int(m.sum()),
            n1=int(m[0].sum()),
            visible_xz_minus_const=vis,
        )
    )
for ax in axes[1:]:
    ax.set_ylabel("")
fig.suptitle(
    "Same rules with the final Z readout of every site "
    "(convention of the repository), $N = 4$, $T = N = 4$",
    fontsize=11,
)
fig.legend(
    handles=legend_handles(True),
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=8.8,
    labelspacing=0.9,
)
fig.tight_layout()
fig.savefig(f"{OUT}/mincut_circuits_final.png", dpi=200, bbox_inches="tight")
plt.close(fig)
json.dump(summary, open(f"{OUT}/mincut_circuits.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
