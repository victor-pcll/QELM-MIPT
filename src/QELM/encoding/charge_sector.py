"""Charge-sector states entangled with one ancilla qubit.

The charge of a computational basis state ``|b_0 ... b_{N-1}>`` is its
number of 1 bits, ``Q = sum_i b_i``. The charge sector ``Q`` is spanned by
the ``binom(N, Q)`` basis states with that charge.

:class:`AncillaChargeSectorEmbedding` prepares, on an ancilla ``R`` and the
system ``S``::

    |psi> = ( |0>_R |Q1>_S + |1>_R |Q2>_S ) / sqrt(2)

on ``N`` qubits in total: one ancilla and ``Ns = N - 1`` system qubits.
``Q1 = Ns // 2``, ``Q2 = Ns // 2 - 1``, and ``|Q>_S`` is the uniform
superposition of the basis states of sector ``Q``. The two sectors are
orthogonal, so the reduced state of ``R`` is ``I / 2``: the ancilla is
maximally entangled with the charge of ``S``.

With ``initial: product`` the sector states are the product states of
Agrawal et al., PRX 14, 041012 (2024), Eq. (B1) (issue #137): the system
qubits alternate ``|0>, |1>, |0>, |1>, ...`` (charge ``Ns // 2``), and the
last ``|1>`` is flipped to ``|0>`` for the charge ``Ns // 2 - 1``. For
``Ns = 6``: ``|010101>`` and ``|010100>``. The paper then scrambles the
charge with random U(1) gates (:class:`QELM.models.circuits.haar.U1BrickworkCircuit`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp

from QELM.encoding.base import Embedding


def charge_sector_indices(N: int, Q: int) -> np.ndarray:
    """Return the basis indices of ``N`` qubits with exactly ``Q`` bits at 1.

    Args:
        N: Number of qubits.
        Q: Target charge, from 0 to ``N``.

    Returns:
        Sorted integer array of shape ``(binom(N, Q),)``.

    Raises:
        ValueError: If ``Q`` is not in ``[0, N]``.
    """
    if not 0 <= Q <= N:
        raise ValueError(f"Charge Q={Q} is out of range [0, {N}].")
    indices = np.arange(2**N)
    popcount = np.array([bin(i).count("1") for i in indices])
    return indices[popcount == Q]


def product_charge_index(N: int, Q: int) -> int:
    """Return the basis index of the alternating product state of charge ``Q``.

    The qubits alternate ``0, 1, 0, 1, ...`` from site 0, and only the first
    ``Q`` sites holding a 1 keep it: ``N = 6, Q = 3`` gives ``|010101>``,
    ``Q = 2`` gives ``|010100>``.

    Args:
        N: Number of qubits.
        Q: Target charge, from 0 to ``N // 2``.

    Returns:
        Integer in ``[0, 2**N)``; site 0 is the most significant bit.

    Raises:
        ValueError: If ``Q`` is not in ``[0, N // 2]``.
    """
    if not 0 <= Q <= N // 2:
        raise ValueError(f"Charge Q={Q} is out of range [0, {N // 2}].")
    ones = list(range(1, N, 2))[:Q]
    return sum(1 << (N - 1 - site) for site in ones)


@overload
def charge_sector_state(
    N: int, Q: int, sparse: Literal[False] = False
) -> np.ndarray: ...
@overload
def charge_sector_state(N: int, Q: int, sparse: Literal[True]) -> sp.csr_matrix: ...
def charge_sector_state(
    N: int, Q: int, sparse: bool = False
) -> np.ndarray | sp.csr_matrix:
    """Return the uniform superposition of the basis states of sector ``Q``.

    Args:
        N: Number of qubits.
        Q: Target charge, from 0 to ``N``.
        sparse: ``True`` for a sparse column, ``False`` (default) for a NumPy
            vector.

    Returns:
        Normalized complex state: a CSR column of shape ``(2**N, 1)`` with
        ``binom(N, Q)`` stored entries if ``sparse``, otherwise a vector of
        shape ``(2**N,)``.
    """
    idx = charge_sector_indices(N, Q)
    amp = 1.0 / np.sqrt(idx.size)
    if sparse:
        return _sparse_column(idx, np.full(idx.size, amp, dtype=complex), 2**N)
    psi = np.zeros(2**N, dtype=complex)
    psi[idx] = amp
    return psi


def _sparse_column(rows: np.ndarray, values: np.ndarray, dim: int) -> sp.csr_matrix:
    """Return a CSR column of shape ``(dim, 1)`` with ``values`` at ``rows``."""
    cols = np.zeros(rows.size, dtype=np.int64)
    return sp.csr_matrix((values, (rows, cols)), shape=(dim, 1), dtype=complex)


class AncillaChargeSectorEmbedding(Embedding):
    """``( |0>_R |Ns//2>_S + |1>_R |Ns//2 - 1>_S ) / sqrt(2)`` on ``N`` qubits.

    The ancilla ``R`` is site 0 (most significant bit), as in
    ``np.kron(ancilla, psi_S)``, and the system ``S`` is sites ``1 .. N-1``,
    so ``Ns = N - 1``. The state does not depend on the input.

    Attributes:
        Q1: Charge paired with ``|0>_R``, ``Ns // 2``.
        Q2: Charge paired with ``|1>_R``, ``Ns // 2 - 1``.
        initial: ``"uniform"`` (default) for the uniform superposition of
            each sector, ``"product"`` for the alternating product state of
            each charge (:func:`product_charge_index`).
    """

    name = "ancilla_charge_sector"
    Nr = 1

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store ``N`` and the two charges.

        Args:
            params: Parameter names mapped to scalar values. ``N``, the total
                number of qubits, is required and must be at least 3, so that
                ``Ns >= 2`` and ``Ns // 2 - 1 >= 0``. ``initial`` is optional,
                ``"uniform"`` or ``"product"``.

        Raises:
            ValueError: If ``N < 3`` or ``initial`` is unknown.
        """
        super().__init__(params)
        if self.Ns < 2:
            raise ValueError(f"N={self.N} must be at least 3.")
        self.Q1: int = self.Ns // 2
        self.Q2: int = self.Ns // 2 - 1
        self.initial: str = str(params.get("initial") or "uniform")
        if self.initial not in ("uniform", "product"):
            raise ValueError(
                f"initial must be 'uniform' or 'product', got {self.initial!r}."
            )

    @overload
    def state(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...
    @overload
    def state(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...
    def state(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return the entangled state on ``R + S``.

        Basis index ``b`` of ``|0>_R |Q1>_S`` stays ``b``; for ``|1>_R |Q2>_S``
        the ancilla bit adds ``2**Ns``. Both branches have amplitude
        ``1 / sqrt(2 binom(Ns, Q))`` for ``initial="uniform"``, and
        ``1 / sqrt(2)`` on a single basis state for ``initial="product"``.

        Args:
            x: Ignored; the state is fixed.
            sparse: ``True`` for a CSR column built without any dense vector
                of size ``2**N``, ``False`` (default) for a NumPy vector.

        Returns:
            Normalized complex state: a CSR column of shape ``(2**N, 1)`` if
            ``sparse``, otherwise a vector of shape ``(2**N,)``.
        """
        if self.initial == "product":
            idx1 = np.array([product_charge_index(self.Ns, self.Q1)])
            idx2 = np.array([product_charge_index(self.Ns, self.Q2)])
        else:
            idx1 = charge_sector_indices(self.Ns, self.Q1)
            idx2 = charge_sector_indices(self.Ns, self.Q2)
        rows = np.concatenate([idx1, idx2 + 2**self.Ns])
        values = np.concatenate(
            [
                np.full(idx1.size, 1.0 / np.sqrt(2 * idx1.size), dtype=complex),
                np.full(idx2.size, 1.0 / np.sqrt(2 * idx2.size), dtype=complex),
            ]
        )
        if sparse:
            return _sparse_column(rows, values, self.dim)
        psi = np.zeros(self.dim, dtype=complex)
        psi[rows] = values
        return psi
