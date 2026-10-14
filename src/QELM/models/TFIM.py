"""Transverse-field Ising model (TFIM) with random all-to-all couplings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

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

    def __init__(
        self,
        params: Mapping[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store the parameters and draw the disorder.

        Args:
            params: ``N`` (number of qubits), ``J`` (coupling scale) and
                ``h`` (transverse field), as scalars from YAML.
            rng: Generator the couplings are drawn from. It moves forward,
                so the next model built from it gets new couplings. ``None``
                uses NumPy's global generator, seeded by ``set_seed``.
        """
        super().__init__(params, rng)
        self.J = float(params["J"])
        self.h = float(params["h"])
        # Only i < j entries are used.
        low, high, shape = -self.J / 2, self.J / 2, (self.N, self.N)
        if rng is None:
            self.J_matrix: np.ndarray = np.random.uniform(low, high, size=shape)
        else:
            self.J_matrix = rng.uniform(low, high, size=shape)

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
