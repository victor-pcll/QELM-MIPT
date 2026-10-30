"""A reference qubit in a Bell pair with one system site (issue #185).

:class:`AncillaBellPairEncoding` prepares, on the ancilla ``R`` (qubit 0)
and ``N`` system qubits (qubits ``1 .. N``)::

    |psi> = ( |0>_R |0 ... 0>_S + |1>_R |0 ... 1_site ... 0>_S ) / sqrt(2)

the Bell pair ``(|00> + |11>) / sqrt(2)`` between ``R`` and the system site
``site``, every other system qubit in ``|0>``. It is the "encoding" step of
the reference-qubit probe of Gullans & Huse, PRL 125, 070606 (2020),
arXiv:1910.00020: the reference starts maximally entangled with one site,
``S_R = 1`` bit, and the circuit then spreads that entanglement over the
chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp

from QELM.encoding.base import Encoding
from QELM.encoding.charge_sector import _sparse_column


class AncillaBellPairEncoding(Encoding):
    """Bell pair between the ancilla ``R`` and one system site, on ``N + 1`` qubits.

    The ancilla ``R`` is qubit 0 (most significant bit), and system site
    ``k`` (from 0) is qubit ``k + 1``, as in :class:`AncillaChargeSectorEncoding`.
    The state does not depend on the input. Config block
    ``ancilla_bell_pair``.

    Attributes:
        site: System site paired with ``R``, from 0 to ``N - 1``; ``N // 2``
            (the middle of the chain) by default.
    """

    name = "ancilla_bell_pair"
    Nr = 1

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store ``N`` and the paired site.

        Args:
            params: ``N``, the number of system qubits, is required.
                ``site`` is optional (``None`` or missing: ``N // 2``).

        Raises:
            ValueError: If ``site`` is not a system site.
        """
        super().__init__(params)
        site = params.get("site")
        self.site: int = self.N // 2 if site is None else int(site)
        if not 0 <= self.site < self.N:
            raise ValueError(f"site must be in [0, {self.N - 1}], got {self.site}.")

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
        """Return the Bell pair of ``R`` and ``site``, the rest in ``|0>``.

        Args:
            x: Ignored; the state is fixed.
            sparse: ``True`` for a CSR column, ``False`` (default) for a
                NumPy vector.

        Returns:
            Normalized complex state with two amplitudes ``1 / sqrt(2)``: a
            CSR column of shape ``(2**N_total, 1)`` if ``sparse``, otherwise
            a vector of shape ``(2**N_total,)``.
        """
        # Qubit q is the bit of weight 2**(N_total - 1 - q); R is qubit 0.
        r_bit = 1 << self.N
        site_bit = 1 << (self.N - 1 - self.site)
        rows = np.array([0, r_bit | site_bit])
        values = np.full(2, 1.0 / np.sqrt(2), dtype=complex)
        if sparse:
            return _sparse_column(rows, values, self.dim)
        psi = np.zeros(self.dim, dtype=complex)
        psi[rows] = values
        return psi
