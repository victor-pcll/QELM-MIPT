"""Transverse-field Ising model (TFIM) with random all-to-all couplings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp

from QELM.models.base import HamiltonianModel
from QELM.physics.operators import X, Z, get_site_operator


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

    def hamiltonian(self) -> sp.csr_matrix:
        """Build the Hamiltonian as a sparse CSR matrix.

        Every term is a product of Pauli matrices, with one non-zero entry
        per row, so ``H`` has at most ``2**N * (1 + N(N-1)/2)`` non-zero
        entries instead of ``4**N``. Use ``H.toarray()`` for a dense copy.

        Returns:
            Complex Hermitian CSR matrix of shape ``(2**N, 2**N)``. It is
            also stored in ``self._H``.
        """
        # basis "X": field on Z, couplings on X X, signs +.
        # basis "Z": field on X, couplings on Z Z, signs -.
        if self.basis == "X":
            field_op, coupling_op, sign = Z, X, 1.0
        else:
            field_op, coupling_op, sign = X, Z, -1.0

        H: sp.csr_matrix = sp.csr_matrix((self.dim, self.dim), dtype=np.complex128)

        for i in range(self.N):
            H = H + sign * self.h * get_site_operator(field_op, i, self.N, sparse=True)

        for i in range(self.N):
            Ci = get_site_operator(coupling_op, i, self.N, sparse=True)
            for j in range(i + 1, self.N):
                Cj = get_site_operator(coupling_op, j, self.N, sparse=True)
                H = H + sign * self.J_matrix[i, j] * (Ci @ Cj)

        self._H = H.tocsr()
        return H.tocsr()
