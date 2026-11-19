"""Drawing of the min cut on the space-time graph of the brickwork (issue #383)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch, Rectangle

from QELM.evaluation.mincut_graph import mincut

BLUE_IN = "#cfe0f3"
GREY = "#d9d9d9"
RED = "#c0392b"
DARK = "#222222"
REC = "#2a7f62"


def ypos(node, nodes, T):
    nd = nodes[node]
    if nd.kind == "source":
        return -0.35
    if nd.kind == "gate":
        return nd.layer + 0.28 + 0.34 * nd.half
    if nd.kind == "meas":
        return nd.layer + 0.88
    return T + 0.35


def draw(ax, masks, final=True, title=None, rank=None, show_costs=True):
    masks = np.asarray(masks, bool)
    T, n = masks.shape
    mc = mincut(masks, final)
    C, S, nodes = mc.value, mc.source_side, mc.nodes
    edges = [(e.tail, e.head, e.bits, e.kind, e.site) for e in mc.edges]
    cut = [(e.tail, e.head, e.bits, e.kind, e.site) for e in mc.cut]
    cutset = {(a, b, k, s) for a, b, c, k, s in cut}
    # wires: per qubit, segments between successive nodes
    for a, b, c, k, s in edges:
        if k not in ("wire",):
            continue
        ya = ypos(a, nodes, T)
        yb = ypos(b, nodes, T)
        if nodes[a].kind == "source":
            ya = -0.2
        iscut = (a, b, k, s) in cutset
        ax.plot(
            [s, s],
            [ya, yb],
            color=RED if iscut else (DARK if c == 2 else "#7f7f7f"),
            lw=3.2 if iscut else (1.6 if c == 2 else 1.0),
            ls="-" if c == 2 else (0, (2, 1.5)),
            zorder=1,
        )
        if iscut and show_costs:
            ym = 0.5 * (ya + yb)
            ax.plot(s, ym, marker="x", color=RED, ms=9, mew=2.4, zorder=6)
            ax.text(
                s + 0.13,
                ym,
                str(c),
                color=RED,
                fontsize=11,
                fontweight="bold",
                va="center",
                zorder=7,
            )
    # dangling wires after the last node (no final readout)
    last = {}
    for a, b, c, k, s in edges:
        if k == "wire":
            last[s] = max(last.get(s, -1), ypos(b, nodes, T))
    if not final:
        for s in range(n):
            y0 = last.get(s, -0.2)
            ax.plot([s, s], [y0, T + 0.1], color="#bbbbbb", lw=1.0, ls=":", zorder=0)
            ax.text(
                s, T + 0.22, "Tr", ha="center", va="bottom", fontsize=7, color="#888888"
            )
    # gates
    for node, nd in enumerate(nodes):
        if nd.kind != "gate":
            continue
        left = nd.site
        y = ypos(node, nodes, T)
        fc = BLUE_IN if node in S else GREY
        ax.add_patch(
            FancyBboxPatch(
                (left - 0.18, y - 0.11),
                1.36,
                0.22,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                fc=fc,
                ec="#555555",
                lw=0.8,
                zorder=3,
            )
        )
    # measurements and records
    for a, b, c, k, s in edges:
        if k == "record":
            y = ypos(a, nodes, T)
            iscut = (a, b, k, s) in cutset
            fc = BLUE_IN if a in S else "white"
            ax.add_patch(
                Rectangle(
                    (s - 0.15, y - 0.11), 0.30, 0.22, fc=fc, ec=DARK, lw=1.0, zorder=4
                )
            )
            ax.text(s, y, "Z", ha="center", va="center", fontsize=7, zorder=5)
            ax.plot(
                [s + 0.15, s + 0.42],
                [y, y],
                color=RED if iscut else REC,
                lw=2.6 if iscut else 1.2,
                zorder=4,
            )
            ax.plot(s + 0.45, y, marker=">", color=REC, ms=3.5, zorder=4)
            if iscut and show_costs:
                ax.plot(s + 0.29, y, marker="x", color=RED, ms=8, mew=2.2, zorder=6)
                ax.text(
                    s + 0.33,
                    y + 0.16,
                    "1",
                    color=RED,
                    fontsize=11,
                    fontweight="bold",
                    zorder=7,
                )
        if k == "final":
            y = T + 0.35
            iscut = (a, b, k, s) in cutset
            ya = last.get(s, -0.2)
            ax.plot(
                [s, s],
                [ya, y - 0.11],
                color=RED if iscut else DARK,
                lw=3.2 if iscut else 1.6,
                zorder=1,
            )
            ax.add_patch(
                Rectangle(
                    (s - 0.17, y - 0.11),
                    0.34,
                    0.22,
                    fc="white",
                    ec=DARK,
                    lw=1.2,
                    zorder=4,
                )
            )
            ax.text(
                s,
                y,
                "Z",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                zorder=5,
            )
            if iscut and show_costs:
                ym = 0.5 * (ya + y)
                ax.plot(s, ym, marker="x", color=RED, ms=9, mew=2.4, zorder=6)
                ax.text(
                    s + 0.13,
                    ym,
                    "1",
                    color=RED,
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    zorder=7,
                )
    # input
    for s in range(n):
        ax.add_patch(
            Rectangle(
                (s - 0.2, -0.45), 0.4, 0.25, fc=BLUE_IN, ec="#3b6ea5", lw=1.0, zorder=4
            )
        )
        ax.text(
            s,
            -0.325,
            rf"$\rho_{{{s}}}$",
            ha="center",
            va="center",
            fontsize=7,
            zorder=5,
        )
    ax.set_xlim(-0.6, n - 0.2 + 0.55)
    ax.set_ylim(-0.65, T + 0.75)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"q{s}" for s in range(n)], fontsize=8)
    ax.set_yticks([t + 0.55 for t in range(T)])
    ax.set_yticklabels([str(t) for t in range(1, T + 1)], fontsize=7)
    for t in range(0, T + 1):
        ax.axhline(t + 0.05, color="#eeeeee", lw=0.6, zorder=-1)
    ax.set_ylabel("layer $t$", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    txt = f"$C = {C}$" + (
        ""
        if rank is None
        else f",  rank $= {rank}$"
        + (f" $= 2^{{{C}}}$" if rank == 2**C else rf" $< 2^{{{C}}}$")
    )
    if title:
        ax.set_title(title + "\n" + txt, fontsize=9.5)
    return C, S, cut
