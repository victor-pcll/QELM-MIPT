"""Charge-sector states entangled with one ancilla qubit.

The charge of a computational basis state ``|b_0 ... b_{N-1}>`` is its
number of 1 bits, ``Q = sum_i b_i``. The charge sector ``Q`` is spanned by
the ``binom(N, Q)`` basis states with that charge.

:class:`AncillaChargeSectorEmbedding` prepares, on an ancilla ``R`` and the
system ``S``::

    |psi> = ( |0>_R |Q1>_S + |1>_R |Q2>_S ) / sqrt(2)

on ``N`` qubits in total: one ancilla and ``N_S = N - 1`` system qubits.
``Q1 = N_S // 2``, ``Q2 = N_S // 2 - 1``, and ``|Q>_S`` is the uniform
superposition of the basis states of sector ``Q``. The two sectors are
orthogonal, so the reduced state of ``R`` is ``I / 2``: the ancilla is
maximally entangled with the charge of ``S``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.embedding.base import Embedding


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


def charge_sector_state(N: int, Q: int) -> np.ndarray:
    """Return the uniform superposition of the basis states of sector ``Q``.

    Args:
        N: Number of qubits.
        Q: Target charge, from 0 to ``N``.

    Returns:
        Normalized complex vector of shape ``(2**N,)``.
    """
    idx = charge_sector_indices(N, Q)
    psi = np.zeros(2**N, dtype=complex)
    psi[idx] = 1.0 / np.sqrt(idx.size)
    return psi


class AncillaChargeSectorEmbedding(Embedding):
    """``( |0>_R |N_S//2>_S + |1>_R |N_S//2 - 1>_S ) / sqrt(2)`` on ``N`` qubits.

    The ancilla ``R`` is site 0 (most significant bit), as in
    ``np.kron(ancilla, psi_S)``, and the system ``S`` is sites ``1 .. N-1``,
    so ``N_S = N - 1``. The state does not depend on the input.

    Attributes:
        Q1: Charge paired with ``|0>_R``, ``N_S // 2``.
        Q2: Charge paired with ``|1>_R``, ``N_S // 2 - 1``.
    """

    name = "ancilla_charge_sector"
    n_ancilla = 1

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store ``N`` and the two charges.

        Args:
            params: Parameter names mapped to scalar values. ``N``, the total
                number of qubits, is required and must be at least 3, so that
                ``N_S >= 2`` and ``N_S // 2 - 1 >= 0``.

        Raises:
            ValueError: If ``N < 3``.
        """
        super().__init__(params)
        if self.N_S < 2:
            raise ValueError(f"N={self.N} must be at least 3.")
        self.Q1: int = self.N_S // 2
        self.Q2: int = self.N_S // 2 - 1

    def state(self, x: np.ndarray | None = None) -> np.ndarray:
        """Return the entangled state on ``R + S``.

        Args:
            x: Ignored; the state is fixed.

        Returns:
            Normalized complex vector of shape ``(2**N,)``.
        """
        ket0 = np.array([1, 0], dtype=complex)
        ket1 = np.array([0, 1], dtype=complex)
        return (
            np.kron(ket0, charge_sector_state(self.N_S, self.Q1))
            + np.kron(ket1, charge_sector_state(self.N_S, self.Q2))
        ) / np.sqrt(2)
