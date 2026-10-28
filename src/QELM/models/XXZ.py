"""Nearest-neighbour XXZ (Heisenberg) chain with a strict U(1) symmetry (issue #236)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp

from QELM.models.base import HamiltonianModel


class XXZChainModel(HamiltonianModel):
    """XXZ chain of ``N`` qubits with nearest-neighbour couplings.

    ``H = sum_<i,j> [J (X_i X_j + Y_i Y_j) + Delta Z_i Z_j] + sum_i h_i Z_i``,

    over the bonds ``<i, j> = (i, i + 1)`` of a chain, plus ``(N - 1, 0)``
    with ``boundary = "periodic"``. ``Delta = J`` is the Heisenberg chain,
    ``Delta = 0`` the XX chain (free fermions). Each field ``h_i`` is drawn
    once, uniformly in ``[-h, h]``, when the model is built: ``h = 0`` (the
    default) is the clean chain, ``h > 0`` a random-field disorder that keeps
    the symmetry. ``N``, ``J``, ``Delta``, ``h`` and ``boundary`` come from
    ``conf/models/XXZ.yaml``. Selected with ``models=XXZ``.

    ``X_i X_j + Y_i Y_j = 2 (S+_i S-_j + S-_i S+_j)`` swaps the bits of
    ``i`` and ``j`` when they differ, and ``Z_i Z_j``, ``Z_i`` are diagonal.
    So ``H`` conserves the number of qubits at ``|1>``, the charge
    ``Q = sum_i (1 - Z_i) / 2``, exactly: ``[Q, H] = 0``, unlike the TFIM,
    which conserves only its parity. The evolution of a state of charge
    ``Q`` stays in the ``C(N, Q)`` basis states of that charge
    (:meth:`sector`), and a charge can only move by one site per bond: the
    chain has a light cone, the TFIM with its all-to-all couplings has none.
    """

    name = "XXZ"
    BOUNDARIES = ("open", "periodic")

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters and draw the random fields.

        Args:
            params: ``N`` (number of qubits), ``J`` (exchange), ``Delta``
                (anisotropy, default ``J``), ``h`` (width of the random
                fields, default 0) and ``boundary`` (``"open"`` or
                ``"periodic"``, default ``"open"``), as scalars from YAML.

        Raises:
            ValueError: If ``boundary`` is unknown, ``h < 0``, or the chain
                is periodic with fewer than 3 sites (its bonds would repeat).
        """
        super().__init__(params)
        self.J = float(params["J"])
        self.Delta = float(params.get("Delta", self.J))
        self.h = float(params.get("h", 0.0))
        self.boundary = str(params.get("boundary", "open"))
        if self.boundary not in self.BOUNDARIES:
            raise ValueError(
                f"XXZ boundary must be 'open' or 'periodic', got {self.boundary!r}"
            )
        if self.h < 0:
            raise ValueError(f"XXZ h must be >= 0, got {self.h}")
        if self.boundary == "periodic" and self.N < 3:
            raise ValueError(f"A periodic XXZ chain needs N >= 3, got N={self.N}")
        # Drawn from NumPy's global generator, seeded by ``set_seed``. With
        # h = 0 nothing is drawn, so the clean chain leaves the stream as is.
        self.fields: np.ndarray = (
            np.random.uniform(-self.h, self.h, size=self.N)
            if self.h > 0
            else np.zeros(self.N)
        )

    def bonds(self) -> list[tuple[int, int]]:
        """Return the nearest-neighbour bonds ``(i, j)`` of the chain."""
        bonds = [(i, i + 1) for i in range(self.N - 1)]
        if self.boundary == "periodic":
            bonds.append((self.N - 1, 0))
        return bonds

    def charges(self) -> np.ndarray:
        """Return the charge (number of bits at 1) of every basis state, ``(dim,)``."""
        basis = np.arange(self.dim)
        shifts = self.N - 1 - np.arange(self.N)
        return np.asarray(np.sum((basis[:, None] >> shifts[None, :]) & 1, axis=1))

    def sector(self, b: int) -> np.ndarray:
        """Return the basis states the evolution connects to ``|b>``.

        Those of the same charge as ``|b>``: the evolution of a state of
        charge ``Q`` never leaves them.

        Args:
            b: Index of a basis state, site 0 the most significant bit.

        Returns:
            Sorted indices of the ``C(N, Q)`` basis states of charge ``Q``.
        """
        Q = bin(int(b)).count("1")
        return np.flatnonzero(self.charges() == Q)

    def hamiltonian(self) -> sp.csr_matrix:
        """Build the Hamiltonian as a sparse CSR matrix.

        The entries are written from the bits of the basis states, as in
        :meth:`QELM.models.TFIM.TransverseFieldIsingModel.hamiltonian`
        (issue #201), with ``z_i = +1`` when bit ``i`` of ``|b>`` is 0 and
        ``-1`` when it is 1 (site 0 the most significant bit). The diagonal
        is ``sum_<i,j> Delta z_i z_j + sum_i h_i z_i``. On a bond whose two
        bits differ, ``X_i X_j + Y_i Y_j`` maps ``|b>`` to ``|b XOR m_ij>``
        with amplitude 2 (``X X`` and ``Y Y`` each give 1 there, and cancel
        where the bits agree), so ``H`` has an entry ``2 J`` there.

        Returns:
            Real symmetric CSR matrix of complex dtype, shape
            ``(2**N, 2**N)``. It is also stored in ``self._H``.
        """
        basis = np.arange(self.dim)
        shifts = self.N - 1 - np.arange(self.N)
        z = 1 - 2 * ((basis[:, None] >> shifts[None, :]) & 1)  # (dim, N)

        diagonal = z @ self.fields.astype(float)
        rows: list[np.ndarray] = [basis]
        cols: list[np.ndarray] = [basis]
        values: list[np.ndarray] = []
        for i, j in self.bonds():
            diagonal = diagonal + self.Delta * z[:, i] * z[:, j]
            differ = basis[z[:, i] != z[:, j]]
            mask = (1 << int(shifts[i])) | (1 << int(shifts[j]))
            rows.append(differ)
            cols.append(differ ^ mask)
            values.append(np.full(differ.size, 2.0 * self.J, dtype=complex))
        values.insert(0, diagonal.astype(complex))

        H = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim),
            dtype=np.complex128,
        )
        self._H = H
        return H
