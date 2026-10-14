"""Transverse-field Ising model (TFIM) with random all-to-all couplings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.linalg

from QELM.models.base import HamiltonianModel
from QELM.physics.operators import X, Z, get_site_operator


class TransverseFieldIsingModel(HamiltonianModel):
    """Transverse-field Ising Hamiltonian on ``N`` qubits.

    ``H = h * sum_i Z_i + sum_{i<j} J_ij X_i X_j``, with each ``J_ij`` drawn
    once, uniformly in ``[-J/2, J/2]``, when the model is built. ``N``, ``J``
    and ``h`` are scalars from ``conf/sim_type/TFIM.yaml``. Selected with
    ``sim_type=TFIM``.
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

    def hamiltonian(self) -> np.ndarray:
        """Build the dense Hamiltonian.

        Returns:
            Complex Hermitian matrix of shape ``(2**N, 2**N)``. It is also
            stored in ``self._H``.
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.N):
            H += self.h * get_site_operator(Z, i, self.N)

        for i in range(self.N):
            Xi = get_site_operator(X, i, self.N)
            for j in range(i + 1, self.N):
                Xj = get_site_operator(X, j, self.N)
                H += self.J_matrix[i, j] * (Xi @ Xj)

        self._H = H
        return H

    def evolve_density(self, *, dt, rho):
        U = scipy.linalg.expm(-1j * self.hamiltonian() * dt)
        return U @ rho @ U.conj().T
    
    def evolve_state(self, *, dt, psi):
        U = scipy.linalg.expm(-1j * self.hamiltonian() * dt)
        return U @ psi