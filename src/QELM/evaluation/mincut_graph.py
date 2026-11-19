"""Min cut of the space-time graph of a monitored brickwork, with its partition.

Companion of :func:`QELM.evaluation.effect_rank.mincut_bits` (issue #383). It
builds the same graph, with or without the final ``Z`` readout, and also
returns the cut itself, which the pedagogical drawings need.

Costs of the edges, in bits:

- a free quantum wire between two gates: 2 (the four Pauli strings
  ``{I, X, Y, Z}``);
- a wire that enters a ``Z`` measurement, or leaves it until the next gate: 1
  (only ``{I, Z}`` survives the measurement);
- the record bit of a measurement: 1;
- the final readout of a site, when there is one: 1.

The rank of the effect space is at most ``2^C``, with ``C`` the minimum cut
between the input legs (source) and the records (sink).

**Causal pruning.** Without the final readout the wires are traced out at the
end. A gate whose outputs are all traced acts as the identity
(``U^dagger I U = I``) and carries no bit. The graph is therefore restricted
to the causal past of the records: every edge whose head cannot reach a
record forward in time is removed. Without this step the cut can be too
large. Example: q0 and q1 measured at layer ``t`` and q0 again at ``t + 1``
gives ``C = 2``, not 3. With a final readout every wire ends on a record, so
the pruning changes nothing and :func:`mincut` equals ``mincut_bits``.

:func:`hartley_mincut` is the min cut of Skinner, Ruhman and Nahum (PRX 9,
031009 (2019), Sec. III): the Hartley entropy ``S_0`` of the left half of a
state, where a measurement breaks the wire (cost 0) and every other wire
costs 1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_flow

from QELM.evaluation.effect_rank import brick_pairs

SOURCE = 0
"""Node of the input legs."""
SINK = 1
"""Node of the records (mid-circuit clicks and final readout)."""


@dataclass(frozen=True)
class Node:
    """One node of the space-time graph.

    Attributes:
        kind: ``"source"``, ``"sink"``, ``"gate"`` or ``"meas"``.
        layer: Layer ``t`` (0-based) of a gate or measurement, else ``-1``.
        site: Left site of a gate, or measured site; ``-1`` otherwise.
        half: ``0`` for the even half-layer of a gate, ``1`` for the odd one.
    """

    kind: str
    layer: int = -1
    site: int = -1
    half: int = 0


@dataclass(frozen=True)
class Edge:
    """One edge, stored forward in time (``tail`` before ``head``).

    Attributes:
        tail: Earlier node.
        head: Later node (or the sink for a record).
        bits: Capacity of the edge in bits.
        kind: ``"wire"``, ``"record"`` or ``"final"``.
        site: Qubit of the edge.
    """

    tail: int
    head: int
    bits: int
    kind: str
    site: int


@dataclass(frozen=True)
class MinCut:
    """Minimum cut and its partition.

    Attributes:
        value: Capacity ``C`` of the minimum cut, in bits.
        source_side: Nodes on the input side of the cut.
        cut: Edges crossing the cut; their bits sum to ``value``.
        nodes: All nodes, indexed as in the edges.
        edges: Edges of the graph after the causal pruning.
    """

    value: int
    source_side: frozenset[int]
    cut: tuple[Edge, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


def spacetime_graph(
    masks: np.ndarray, final: bool = True
) -> tuple[list[Node], list[Edge]]:
    """Nodes and edges of the brickwork, layer convention of ``effect_rank``.

    Args:
        masks: Boolean array ``(T, n)``, ``masks[t, i]`` true when site ``i``
            is measured after the gates of layer ``t``.
        final: Add the final ``Z`` readout of every site.

    Returns:
        The nodes and the edges, before pruning.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    nodes = [Node("source"), Node("sink")]
    edges: list[Edge] = []
    leg = [SOURCE] * n
    bits = [2] * n
    n_even = len(range(0, n - 1, 2))
    for t in range(depth):
        for index, left in enumerate(brick_pairs(n)):
            gate = len(nodes)
            nodes.append(Node("gate", t, left, 0 if index < n_even else 1))
            for s in (left, left + 1):
                edges.append(Edge(leg[s], gate, bits[s], "wire", s))
                leg[s], bits[s] = gate, 2
        for site in np.flatnonzero(masks[t]).tolist():
            meas = len(nodes)
            nodes.append(Node("meas", t, site))
            edges.append(Edge(leg[site], meas, 1, "wire", site))
            edges.append(Edge(meas, SINK, 1, "record", site))
            leg[site], bits[site] = meas, 1
    if final:
        for s in range(n):
            edges.append(Edge(leg[s], SINK, 1, "final", s))
    return nodes, edges


def causal_prune(edges: list[Edge]) -> list[Edge]:
    """Keep the edges whose head lies in the causal past of a record."""
    live = {SINK}
    changed = True
    while changed:
        changed = False
        for e in edges:
            if e.head in live and e.tail not in live:
                live.add(e.tail)
                changed = True
    return [e for e in edges if e.head in live]


def _max_flow(
    edges: list[Edge], n_nodes: int, source: int, sink: int
) -> tuple[int, frozenset[int]]:
    """Value of the undirected max flow and the minimal source side."""
    capacity: dict[tuple[int, int], int] = {}
    for e in edges:
        if e.tail == e.head or e.bits == 0:
            continue
        for a, b in ((e.tail, e.head), (e.head, e.tail)):
            capacity[(a, b)] = capacity.get((a, b), 0) + e.bits
    if not capacity:
        return 0, frozenset({source})
    rows = np.array([a for a, _ in capacity], dtype=np.int64)
    cols = np.array([b for _, b in capacity], dtype=np.int64)
    values = np.array(list(capacity.values()), dtype=np.int32)
    graph = csr_array((values, (rows, cols)), shape=(n_nodes, n_nodes))
    result = maximum_flow(graph, source, sink)
    residual = (graph - result.flow).tocsr()
    side = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        start, stop = residual.indptr[u], residual.indptr[u + 1]
        for w, r in zip(residual.indices[start:stop], residual.data[start:stop]):
            if r > 0 and int(w) not in side:
                side.add(int(w))
                queue.append(int(w))
    return int(result.flow_value), frozenset(side)


def mincut(masks: np.ndarray, final: bool = True, prune: bool = True) -> MinCut:
    """Minimum cut between the input and the records, with its partition.

    Args:
        masks: Boolean array ``(T, n)`` of the mid-circuit measurements.
        final: Final ``Z`` readout of every site.
        prune: Restrict the graph to the causal past of the records (see the
            module docstring). Only ``False`` for comparison.

    Returns:
        The cut. ``source_side`` is the input side closest to the input (the
        nodes reachable from the source in the residual graph).
    """
    nodes, edges = spacetime_graph(masks, final)
    if prune:
        edges = causal_prune(edges)
    value, side = _max_flow(edges, len(nodes), SOURCE, SINK)
    cut = tuple(e for e in edges if (e.tail in side) != (e.head in side))
    return MinCut(value, side, cut, tuple(nodes), tuple(edges))


def record_bound(masks: np.ndarray, final: bool = True) -> int:
    """Cut through every record: ``K`` without readout, ``N + K_eff`` with.

    With the final readout, a measurement of the last layer repeats it and is
    not counted (``K_eff`` excludes the last layer).
    """
    masks = np.asarray(masks, dtype=bool)
    if final:
        return int(masks.shape[1] + masks[:-1].sum())
    return int(masks.sum())


def input_bound(masks: np.ndarray) -> int:
    """Cut just after the first layer of measurements: ``2N - n_1``."""
    masks = np.asarray(masks, dtype=bool)
    return int(2 * masks.shape[1] - masks[0].sum())


def hartley_mincut(
    n: int, depth: int, p: float, rng: np.random.Generator, half_layers: bool = True
) -> int:
    """``S_0`` of the left half by min cut, Skinner, Ruhman and Nahum 2019.

    State network of an open brickwork from a product state: every wire costs
    1 bit, a projective measurement breaks it (cost 0), the bottom legs
    dangle, and the cut separates the top legs of the left half ``A`` from
    those of the right half.

    Args:
        n: Number of qubits.
        depth: Number of full layers.
        p: Probability of a measurement in each slot.
        rng: Random generator of the measurement positions.
        half_layers: One measurement slot after each half-layer (the geometry
            of SRN, bond percolation at ``p_c = 1/2``); otherwise one slot
            after each full layer (the geometry of the repository).

    Returns:
        ``S_0(A)`` in bits.
    """
    nodes = 2
    edges: list[Edge] = []
    leg: list[int | None] = [None] * n
    halves = (list(range(0, n - 1, 2)), list(range(1, n - 1, 2)))
    for _ in range(depth):
        for h, bonds in enumerate(halves):
            for left in bonds:
                gate = nodes
                nodes += 1
                for s in (left, left + 1):
                    tail = leg[s]
                    if tail is not None:
                        edges.append(Edge(tail, gate, 1, "wire", s))
                    leg[s] = gate
            if half_layers or h == 1:
                for s in np.flatnonzero(rng.random(n) < p).tolist():
                    leg[s] = None
    for s in range(n):
        last = leg[s]
        if last is None:
            continue
        if s < n // 2:
            edges.append(Edge(SOURCE, last, 1, "final", s))
        else:
            edges.append(Edge(last, SINK, 1, "final", s))
    return _max_flow(edges, nodes, SOURCE, SINK)[0]
