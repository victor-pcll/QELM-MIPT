"""Reduced density matrix of a subset of qubits (partial trace).

Site 0 is the leftmost Kronecker factor, the most significant bit of a basis
index, as in :func:`QELM.physics.operators.get_site_operator` and
:mod:`QELM.embedding`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import scipy.sparse as sp


def state_qubits(state: np.ndarray | sp.spmatrix) -> int:
    """Return ``N`` for a state vector ``(2**N,)``, a sparse column
    ``(2**N, 1)`` or a density matrix ``(2**N, 2**N)``, dense or sparse.

    Raises:
        ValueError: If the shape is none of these, or ``2**N`` is not a power
            of two with ``N >= 1``.
    """
    shape = tuple(state.shape)
    dim = int(shape[0]) if len(shape) in (1, 2) else 0
    N = dim.bit_length() - 1
    if dim < 2 or 2**N != dim or shape not in ((dim,), (dim, 1), (dim, dim)):
        raise ValueError(
            "state must be a vector (2**N,), a column (2**N, 1) or a density "
            f"matrix (2**N, 2**N) with N >= 1, got shape {shape}."
        )
    return N


def check_sites(sites: Sequence[int], N: int, name: str = "sites") -> list[int]:
    """Return ``sites`` as a list of ints, checked to be distinct and in range.

    Raises:
        ValueError: If a site is outside ``0 .. N-1`` or appears twice.
    """
    out = [int(q) for q in sites]
    if any(not 0 <= q < N for q in out):
        raise ValueError(f"{name} must be in [0, {N - 1}], got {out}.")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must not repeat a qubit, got {out}.")
    return out


def as_vector(state: np.ndarray | sp.spmatrix) -> np.ndarray | None:
    """The state as a 1-D array if it is a vector or a column, else ``None``."""
    if sp.issparse(state):
        if state.shape[1] != 1:
            return None
        return np.asarray(cast(sp.csr_matrix, state).toarray()).ravel()
    arr = np.asarray(state)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1 and arr.shape[0] != 1:
        return arr.ravel()
    return None


def vector_matrix(psi: np.ndarray, keep: Sequence[int], N: int) -> np.ndarray:
    """Reshape ``psi`` into the ``(2**|keep|, 2**(N - |keep|))`` matrix
    ``M[a, b] = <a_keep b_rest | psi>``, so that ``rho_keep = M M^dagger``.

    The rows follow the order of ``keep``; the columns follow the remaining
    qubits in increasing order.
    """
    rest = [q for q in range(N) if q not in set(keep)]
    tensor = psi.reshape([2] * N).transpose([*keep, *rest])
    return tensor.reshape(2 ** len(keep), 2 ** len(rest))


def partial_trace(state: np.ndarray | sp.spmatrix, keep: Sequence[int]) -> np.ndarray:
    """Return ``rho_A = Tr_B(rho)`` for the qubits ``A = keep``.

    ``B`` is every other qubit. The ``2**|A| x 2**|A|`` result is dense; its
    basis follows the order of ``keep``, so ``keep=[2, 0]`` puts qubit 2 in
    the most significant bit. ``keep=[]`` gives the ``1 x 1`` matrix
    ``[[Tr rho]]``, and ``keep`` equal to every qubit in order gives ``rho``.

    Args:
        state: State vector ``(2**N,)`` (or a sparse column ``(2**N, 1)``,
            as returned by ``Embedding.state(sparse=True)``), for which
            ``rho = |psi><psi|`` is never formed, or a density matrix
            ``(2**N, 2**N)``, dense or sparse. A sparse ``rho`` is traced
            entry by entry, without a dense copy.
        keep: Qubits of ``A``, distinct, each in ``0 .. N-1``.

    Returns:
        The reduced density matrix of ``A``, complex, of shape
        ``(2**|A|, 2**|A|)``.

    Raises:
        ValueError: If ``state`` has none of the shapes above, or ``keep``
            repeats a qubit or holds one out of range.
    """
    N = state_qubits(state)
    keep = check_sites(keep, N, "keep")
    dim_A = 2 ** len(keep)

    psi = as_vector(state)
    if psi is not None:
        M = vector_matrix(psi.astype(complex), keep, N)
        return M @ M.conj().T

    traced = [q for q in range(N) if q not in set(keep)]
    if sp.issparse(state):
        coo = sp.coo_matrix(cast(sp.csr_matrix, state))
        rows, cols, data = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data

        def bits(index: np.ndarray, sites: list[int]) -> np.ndarray:
            out = np.zeros_like(index)
            for q in sites:
                out = (out << 1) | ((index >> (N - 1 - q)) & 1)
            return out

        same_B = bits(rows, traced) == bits(cols, traced)
        reduced = np.zeros((dim_A, dim_A), dtype=complex)
        np.add.at(
            reduced,
            (bits(rows[same_B], keep), bits(cols[same_B], keep)),
            data[same_B],
        )
        return reduced

    rho = np.asarray(state)
    row = list(range(N))
    col = list(range(N, 2 * N))
    for q in traced:
        col[q] = row[q]  # same label on both axes: summed, i.e. traced out
    out = [row[q] for q in keep] + [col[q] for q in keep]
    reduced = np.einsum(rho.reshape([2] * (2 * N)), row + col, out)
    return np.asarray(reduced, dtype=complex).reshape(dim_A, dim_A)
