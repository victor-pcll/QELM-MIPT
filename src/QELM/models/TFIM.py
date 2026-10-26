"""Transverse-field Ising model (TFIM) with random all-to-all couplings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp

from QELM.models.base import HamiltonianModel


class TransverseFieldIsingModel(HamiltonianModel):
    """Transverse-field Ising Hamiltonian on ``N`` qubits.

    ``H = h * sum_i Z_i + sum_{i<j} J_ij X_i X_j``, with each ``J_ij`` drawn
    once, uniformly in ``[-J/2, J/2]``, when the model is built. ``N``, ``J``
    and ``h`` are scalars from ``conf/models/TFIM.yaml``. Selected with
    ``models=TFIM``.
    """

    name = "TFIM"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters and draw the disorder.

        Args:
            params: ``N`` (number of qubits), ``J`` (coupling scale) and
                ``h`` (transverse field), as scalars from YAML.
        """
        super().__init__(params)
        self.J = float(params["J"])
        self.h = float(params["h"])
        # Drawn from NumPy's global generator, seeded by ``set_seed``. Only
        # i < j entries are used.
        self.J_matrix: np.ndarray = np.random.uniform(
            -self.J / 2, self.J / 2, size=(self.N, self.N)
        )

    def hamiltonian(self) -> sp.csr_matrix:
        """Build the Hamiltonian as a sparse CSR matrix.

        Every term is a product of Pauli matrices, with one non-zero entry
        per row, so ``H`` has at most ``2**N * (1 + N(N-1)/2)`` non-zero
        entries instead of ``4**N``. Use ``H.toarray()`` for a dense copy.

        The entries are written directly from the bits of the basis states
        (issue #201), without the Kronecker products of
        :func:`QELM.physics.operators.get_site_operator`: ``Z_i`` is
        diagonal, ``+1`` when bit ``i`` of the state is 0 and ``-1`` when it
        is 1, and ``X_i X_j`` maps ``|b>`` to ``|b XOR m_ij>``, with ``m_ij``
        the mask of the bits ``i`` and ``j`` (site 0 the most significant
        bit). Distinct pairs give distinct masks, so no two terms share an
        off-diagonal entry.

        Returns:
            Complex Hermitian CSR matrix of shape ``(2**N, 2**N)``. It is
            also stored in ``self._H``.
        """
        basis = np.arange(self.dim)
        shifts = self.N - 1 - np.arange(self.N)
        bits = (basis[:, None] >> shifts[None, :]) & 1
        diagonal = self.h * np.sum(1 - 2 * bits, axis=1)

        rows = [basis]
        cols = [basis]
        values = [diagonal.astype(np.complex128)]
        for i in range(self.N):
            for j in range(i + 1, self.N):
                mask = (1 << int(shifts[i])) | (1 << int(shifts[j]))
                rows.append(basis)
                cols.append(basis ^ mask)
                values.append(np.full(self.dim, self.J_matrix[i, j], dtype=complex))
        H = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim),
            dtype=np.complex128,
        )
        self._H = H
        return H
