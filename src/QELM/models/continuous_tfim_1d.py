"""Nearest-neighbour Ising chain in a tilted field, in continuous time (issue #241).

The Hamiltonian counterpart of the Floquet chain of Skinner et al. (2019)
(:mod:`QELM.models.circuits.floquet`, issue #239)::

    H = - sum_{i=1}^{N-1} J_i Z_i Z_{i+1} - h_x sum_i X_i - h_z sum_i Z_i

with open boundaries, ``h_x = 1.05``, ``h_z = 0.5`` and each ``J_i`` drawn
uniformly in ``[J - W/2, J + W/2] = [0.5, 1.5]``. With uniform ``J_i = 1``
these are the parameters of Kim and Huse, PRL 111, 127205 (2013), where the
chain is robustly non-integrable and thermalizes; the longitudinal field
``h_z`` breaks the integrability of the transverse-field chain, and keeping
every ``J_i >= 0.5`` never cuts the chain. Nothing is conserved: neither the
charge ``sum_i Z_i`` nor its parity.

Selected with ``models=continuous_tfim_1d``. The monitored dynamics (a
Poisson process of Z measurements at rate ``gamma`` per qubit) lives in
``experiments/QELM_S/exp_QELM_S.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp

from QELM.models.base import HamiltonianModel


class ContinuousTFIMChain(HamiltonianModel):
    """Open Ising chain of ``N`` qubits with a transverse and a longitudinal field.

    ``H = - sum_i J_i Z_i Z_{i+1} - h_x sum_i X_i - h_z sum_i Z_i``, the
    bonds ``(i, i + 1)`` for ``i = 0 .. N - 2``. Each ``J_i`` is drawn once,
    uniformly in ``[J - W/2, J + W/2]``, when the model is built (``W = 0``
    gives the clean chain). ``N``, ``J``, ``W``, ``h_x`` and ``h_z`` come from
    ``conf/models/continuous_tfim_1d.yaml``.
    """

    name = "continuous_tfim_1d"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters and draw the couplings.

        Args:
            params: ``N``, and the optional ``J`` (mean coupling, default 1),
                ``W`` (width of the uniform distribution of ``J_i``, default
                1), ``h_x`` (default 1.05) and ``h_z`` (default 0.5).

        Raises:
            ValueError: If ``N < 2``, ``W < 0``, or some ``J_i`` could be
                negative or zero (``W / 2 >= J``), which would cut the chain.
        """
        super().__init__(params)
        self.J = float(params.get("J", 1.0))
        self.W = float(params.get("W", 1.0))
        self.h_x = float(params.get("h_x", 1.05))
        self.h_z = float(params.get("h_z", 0.5))
        if self.N < 2:
            raise ValueError(f"The chain needs N >= 2, got N={self.N}.")
        if self.W < 0 or self.W / 2 >= self.J:
            raise ValueError(
                f"Need 0 <= W < 2 J so that every J_i > 0, got J={self.J}, W={self.W}."
            )
        # Drawn from NumPy's global generator, seeded by ``set_seed``.
        self.couplings: np.ndarray = (
            self.J + self.W * (np.random.uniform(size=self.N - 1) - 0.5)
            if self.W > 0
            else np.full(self.N - 1, self.J)
        )

    def bonds(self) -> list[tuple[int, int]]:
        """Return the nearest-neighbour bonds ``(i, i + 1)`` of the open chain."""
        return [(i, i + 1) for i in range(self.N - 1)]

    def hamiltonian(self) -> sp.csr_matrix:
        """Build the Hamiltonian as a sparse CSR matrix.

        Written from the bits of the basis states, as
        :meth:`QELM.models.TFIM.TransverseFieldIsingModel.hamiltonian`: with
        ``z_i = +1`` for bit ``i`` at 0 and ``-1`` at 1 (site 0 the most
        significant bit), the diagonal is ``- sum_i J_i z_i z_{i+1} - h_z
        sum_i z_i``, and ``X_i`` maps ``|b>`` to ``|b XOR m_i>``.

        Returns:
            Real symmetric CSR matrix of complex dtype, shape
            ``(2**N, 2**N)``. It is also stored in ``self._H``.
        """
        basis = np.arange(self.dim)
        shifts = self.N - 1 - np.arange(self.N)
        z = 1 - 2 * ((basis[:, None] >> shifts[None, :]) & 1)  # (dim, N)
        diagonal = -self.h_z * z.sum(axis=1).astype(float)
        for (i, j), J_i in zip(self.bonds(), self.couplings, strict=True):
            diagonal -= J_i * z[:, i] * z[:, j]
        rows = [basis] + [basis] * self.N
        cols = [basis] + [basis ^ (1 << int(s)) for s in shifts]
        values = [diagonal.astype(complex)] + [
            np.full(self.dim, -self.h_x, dtype=complex) for _ in range(self.N)
        ]
        H = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim),
            dtype=np.complex128,
        )
        self._H = H
        return H
