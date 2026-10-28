"""Transverse-field Ising model (TFIM) with random all-to-all couplings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp

from QELM.models.base import HamiltonianModel


class TransverseFieldIsingModel(HamiltonianModel):
    """Transverse-field Ising Hamiltonian on ``N`` qubits.

    The Pauli basis of the couplings is set by ``basis``:

    - ``basis = "X"`` (default, quantum information convention):
      ``H = h * sum_i Z_i + sum_{i<j} J_ij X_i X_j``.
    - ``basis = "Z"`` (condensed matter convention):
      ``H = -h * sum_i X_i - sum_{i<j} J_ij Z_i Z_j``.

    Each ``J_ij`` is drawn once, uniformly in ``[-J/2, J/2]``, when the model
    is built. ``N``, ``J``, ``h`` and ``basis`` come from
    ``conf/models/TFIM.yaml``. Selected with ``models=TFIM``.

    The two conventions are related by a Hadamard on every qubit and the sign
    flip ``(h, J_ij) -> (-h, -J_ij)``, so they have the same spectrum up to
    ``E -> -E``. They are not interchangeable in the QELM pipeline: with
    ``basis = "X"`` the Hamiltonian conserves the Z parity, which the charge
    sectors and the Z measurements rely on; with ``basis = "Z"`` it conserves
    the X parity instead.
    """

    name = "TFIM"
    BASES = ("X", "Z")

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters and draw the disorder.

        Args:
            params: ``N`` (number of qubits), ``J`` (coupling scale), ``h``
                (transverse field) and the optional ``basis`` (``"X"`` or
                ``"Z"``, default ``"X"``), as scalars from YAML.

        Raises:
            ValueError: If ``basis`` is neither ``"X"`` nor ``"Z"``.
        """
        super().__init__(params)
        self.J = float(params["J"])
        self.h = float(params["h"])
        self.basis = str(params.get("basis", "X"))
        if self.basis not in self.BASES:
            raise ValueError(f"TFIM basis must be 'X' or 'Z', got {self.basis!r}")
        # Drawn from NumPy's global generator, seeded by ``set_seed``. Only
        # i < j entries are used.
        self.J_matrix: np.ndarray = np.random.uniform(
            -self.J / 2, self.J / 2, size=(self.N, self.N)
        )

    def sector(self, b: int) -> np.ndarray:
        """Return the basis states the evolution connects to ``|b>``.

        With ``basis = "X"`` the TFIM conserves the Z parity (``Z_i`` is
        diagonal, ``X_i X_j`` flips two bits): the basis states with the
        parity of ``|b>``. With ``basis = "Z"`` it conserves the X parity
        instead, which is not diagonal in the Z basis: every basis state.

        Args:
            b: Index of a basis state, site 0 the most significant bit.

        Returns:
            Sorted indices of basis states, ``2**(N - 1)`` or ``2**N``.
        """
        if self.basis != "X":
            return np.arange(self.dim)
        popcount = np.array([bin(c).count("1") for c in range(self.dim)])
        return np.flatnonzero(popcount % 2 == bin(int(b)).count("1") % 2)

    def hamiltonian(self) -> sp.csr_matrix:
        """Build the Hamiltonian as a sparse CSR matrix.

        Every term is a product of Pauli matrices, with one non-zero entry
        per row, so ``H`` has at most ``2**N * (1 + N(N-1)/2)`` non-zero
        entries instead of ``4**N``. Use ``H.toarray()`` for a dense copy.

        The entries are written directly from the bits of the basis states
        (issue #201), without the Kronecker products of
        :func:`QELM.physics.operators.get_site_operator`. With ``z_i = +1``
        when bit ``i`` of ``|b>`` is 0 and ``-1`` when it is 1 (site 0 the
        most significant bit): ``Z_i`` and ``Z_i Z_j`` are diagonal, ``z_i``
        and ``z_i z_j``; ``X_i`` maps ``|b>`` to ``|b XOR m_i>`` and
        ``X_i X_j`` to ``|b XOR m_ij>``, with ``m`` the mask of the flipped
        bits. Distinct sites and pairs give distinct masks, so no two terms
        share an off-diagonal entry.

        Returns:
            Complex Hermitian CSR matrix of shape ``(2**N, 2**N)``. It is
            also stored in ``self._H``.
        """
        basis = np.arange(self.dim)
        shifts = self.N - 1 - np.arange(self.N)
        z = 1 - 2 * ((basis[:, None] >> shifts[None, :]) & 1)  # (dim, N)
        pairs = [(i, j) for i in range(self.N) for j in range(i + 1, self.N)]

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        values: list[np.ndarray] = []

        def flip(mask: int, value: float) -> None:
            rows.append(basis)
            cols.append(basis ^ mask)
            values.append(np.full(self.dim, value, dtype=complex))

        if self.basis == "X":
            # h sum_i Z_i + sum_{i<j} J_ij X_i X_j
            diagonal = self.h * np.sum(z, axis=1).astype(float)
            for i, j in pairs:
                flip((1 << int(shifts[i])) | (1 << int(shifts[j])), self.J_matrix[i, j])
        else:
            # -h sum_i X_i - sum_{i<j} J_ij Z_i Z_j
            diagonal = np.zeros(self.dim)
            for i, j in pairs:
                diagonal -= self.J_matrix[i, j] * z[:, i] * z[:, j]
            for i in range(self.N):
                flip(1 << int(shifts[i]), -self.h)
        rows.append(basis)
        cols.append(basis)
        values.append(diagonal.astype(complex))

        H = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim),
            dtype=np.complex128,
        )
        self._H = H
        return H
