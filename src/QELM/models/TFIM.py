"""Ising model scaffold."""

from __future__ import annotations
import numpy as np

from QELM.models.base import HamiltonianModel
from QELM.physics.operators import X, Z, get_site_operator


class TransverseFieldIsingModel(HamiltonianModel):
    """Transverse Field Ising Hamiltonian and time evolution.

    ``N``, ``J``, and ``learning_rate`` are scalars from
    ``conf/sim_type/ising.yaml``. The operator of shape ``(dim, dim)`` is
    not implemented.
    """

    name = "Transverse Field Ising Model"

    def __init__(self, params: dict[str, float]) -> None:
        """Store scalar Hamiltonian parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
        """
        super().__init__(params)
        self.N = params["N"]
        self.J = params["J"]
        self.h = params["h"]
        self.J_matrix: np.ndarray = np.random.uniform(
            -self.J / 2, self.J / 2, size=(self.N, self.N)
        )

    def hamiltonian(self):
        H = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.N):
            H += self.h * get_site_operator(Z, i, self.N)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                Jij = self.J_matrix[i, j]
                Xi = get_site_operator(X, i, self.N)
                Xj = get_site_operator(X, j, self.N)
                H += Jij * (Xi @ Xj)

        self._H = H
        return H