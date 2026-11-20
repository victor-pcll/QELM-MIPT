"""Locality of the Clifford readout: minimal parity weight of each effect (#386).

In the Clifford QELM of #337 the mean of the parity ``prod_{j in S} (-1)^{m_j}``
of a set ``S`` of record bits is ``Tr[rho O_S]``, with ``O_S`` one Pauli
string (up to sign) or zero (:func:`QELM.evaluation.clifford_ridge.parity_strings`).
Both facts are linear over GF(2) in the indicator vector of ``S``:

- without the dephasing, the back-propagated string of ``S`` is the sum of the
  strings of its bits, ``B S`` (a Clifford gate acts linearly on ``[x | z]``,
  and a bit adds ``Z`` on its site);
- ``O_S`` survives iff at every mid-circuit measurement ``(t, k)`` the string
  of the later bits of ``S`` commutes with ``Z_k``, i.e. ``A S = 0`` with one
  row of ``A`` per measurement (the ``x_k`` bit of that string).

So the alive sets form the subspace ``L = ker A``, ``S -> O_S = B S`` is linear
on ``L`` and the effect group is ``G = B L``. The readout ``F^(k)`` (parities
of at most ``k`` bits) reaches the strings ``D_k = {B S : S in L, |S| <= k}``,
and the minimal parity weight of a string ``g in G``,

    w(g) = min { |S| : S in L, B S = g },

decides at which ``k`` the population ridge starts to read it:
``g in D_k  <=>  w(g) <= k``. With the flat prior the relative readout error
``E_readout(k) / (1 - E_circuit)`` is the fraction of the visible strings of
``G`` with ``w > k``; with the iid prior each string counts ``2^{-|g|}``.

:func:`min_parity_weights` computes ``w`` exactly by walking ``L`` (``2^dim L``
sets, feasible up to ``dim L`` about 26).
"""

from __future__ import annotations

import numpy as np

from QELM.evaluation.clifford_ridge import (
    CliffordQELM,
    bit_times,
    heisenberg_table,
    periodic_bonds,
)


def parity_linear_maps(circuit: CliffordQELM) -> tuple[np.ndarray, np.ndarray]:
    """GF(2) maps ``A`` (dephasing constraints) and ``B`` (string at ``t = 0``).

    Bits in the order of :func:`QELM.evaluation.clifford_ridge.body_program`
    (mid-circuit layer by layer, sites increasing, then the final readout).

    Returns:
        ``A``: ``(n_mid, n_bits)`` bool; ``B``: ``(2 N, n_bits)`` bool, rows
        ``[x | z]`` of the string. ``S`` is alive iff ``A S = 0``, and then
        ``O_S = B S`` up to sign.
    """
    table = heisenberg_table()
    n, t_steps = circuit.N, circuit.T
    steps, sites = bit_times(circuit)
    n_bits = steps.size
    code = np.zeros((n_bits, n), dtype=np.int64)  # 0 I, 1 X, 2 Z, 3 Y
    a_rows: list[np.ndarray] = []
    final = steps == t_steps
    code[final, sites[final]] = 2
    for t in reversed(range(t_steps)):
        for k in np.flatnonzero(circuit.measured[t]):
            a_rows.append((code[:, k] == 1) | (code[:, k] == 3))
        new = steps == t
        code[new, sites[new]] ^= 2  # a fresh bit: Z on an identity
        for g, (a, b) in zip(circuit.gates[t], periodic_bonds(n, t), strict=True):
            out = table[int(g), 4 * code[:, a] + code[:, b]]
            code[:, a], code[:, b] = out // 4, out % 4
    x = (code == 1) | (code == 3)
    z = code >= 2
    b_map = np.concatenate([x, z], axis=1).T
    a_map = np.array(a_rows).reshape(len(a_rows), n_bits)
    return a_map, b_map


def nullspace_gf2(m: np.ndarray) -> np.ndarray:
    """Basis of ``{v : m v = 0}`` over GF(2), rows of shape ``(d, n_cols)``."""
    a = np.array(m, dtype=bool)
    n_rows, n_cols = a.shape
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        if r == n_rows:
            break
        hit = np.flatnonzero(a[r:, c])
        if hit.size == 0:
            continue
        i = r + int(hit[0])
        a[[r, i]] = a[[i, r]]
        rows = np.flatnonzero(a[:, c])
        rows = rows[rows != r]
        a[rows] ^= a[r]
        pivots.append(c)
        r += 1
    free = [c for c in range(n_cols) if c not in set(pivots)]
    basis = np.zeros((len(free), n_cols), dtype=bool)
    for j, f in enumerate(free):
        basis[j, f] = True
        for i, c in enumerate(pivots):
            basis[j, c] = a[i, f]
    return basis


INF = np.iinfo(np.uint8).max


def min_parity_weights(circuit: CliffordQELM) -> np.ndarray:
    """Exact ``w(g)`` for every Pauli string ``g``, by dynamic programming.

    Walks the circuit backwards with a table ``D[o]`` over the ``4^N`` strings:
    the fewest bits of a set ``S`` of the later bits that is alive so far and
    whose back-propagated string is ``o``. The final readout seeds ``D[z] =
    |z|`` on the ``Z`` strings; a measurement ``(t, k)`` first drops the
    strings with ``x_k = 1`` (dephased), then lets its bit in,
    ``D[o] <- min(D[o], D[o Z_k] + 1)``; a gate permutes the table. The cost is
    ``O((n_mid + N T) 4^N)``, with no listing of records or sets.

    Returns:
        ``uint8`` array of shape ``(4,) * N`` indexed by the per-site codes
        (``0 I, 1 X, 2 Z, 3 Y``, axis ``i`` = site ``i``); :data:`INF` where
        the string is not in ``G``. ``w(I) = 0``.
    """
    table = heisenberg_table()
    n = circuit.N
    # Seed: Z strings of the final readout, cost = number of bits.
    one = np.array([0, INF, 1, INF], dtype=np.int16)
    d = one.astype(np.uint8)
    for _ in range(n - 1):
        d = np.minimum(np.add.outer(d.astype(np.int16), one), INF).astype(np.uint8)
    zflip = np.array([2, 3, 0, 1])
    for t in reversed(range(circuit.T)):
        for k in np.flatnonzero(circuit.measured[t]):
            v = np.moveaxis(d, int(k), 0)
            v[1] = INF
            v[3] = INF
            moved = np.minimum(v[zflip].astype(np.int16) + 1, INF).astype(np.uint8)
            np.minimum(v, moved, out=v)
        for g, (a, b) in zip(circuit.gates[t], periodic_bonds(n, t), strict=True):
            out = table[int(g)]
            v = np.moveaxis(d, (a, b), (0, 1))
            shape = v.shape
            flat = v.reshape(16, -1)
            new = np.empty_like(flat)
            new[out] = flat
            d = np.moveaxis(new.reshape(shape), (0, 1), (a, b)).copy()
    return d


def visible_codes(n: int) -> np.ndarray:
    """Boolean ``(4,) * n`` mask of the Y-free strings other than ``I``."""
    m = np.ones((4,) * n, dtype=bool)
    for i in range(n):
        idx: list[slice | int] = [slice(None)] * n
        idx[i] = 3
        m[tuple(idx)] = False
    m[(0,) * n] = False
    return m


def string_weights(n: int) -> np.ndarray:
    """``|g|`` (number of non-identity sites) on the ``(4,) * n`` grid."""
    w = np.zeros((4,) * n, dtype=np.int64)
    for i in range(n):
        shape = [1] * n
        shape[i] = 4
        w = w + (np.arange(4) > 0).reshape(shape)
    return w


def relative_readout(
    w: np.ndarray, ks: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """``E_readout(k) / (1 - E_circuit)`` from the minimal weights of ``G``.

    Args:
        w: Minimal parity weights of the visible strings of ``G`` (finite).
        ks: Readout orders.
        weights: Power of each string in ``Var y`` (``None``: flat prior,
            every string the same; iid prior: ``2^{-|g|}``).

    Returns:
        One value per ``k``: the power on the strings with ``w > k`` over the
        power on ``G``; ``nan`` if ``G`` has no visible string.
    """
    w = np.asarray(w)
    q = np.ones(w.shape) if weights is None else np.asarray(weights, dtype=float)
    total = q.sum()
    if total == 0:
        return np.full(len(ks), np.nan)
    return np.array([q[w > k].sum() / total for k in ks])
