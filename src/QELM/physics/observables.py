"""Observables measured on the reservoir state to build QELM features.

A single observable ``O`` gives the ``N`` features ``<O_i> = Tr(rho O_i)``.
A double observable ``(A, B)`` gives the ``N (N - 1) / 2`` two-point
correlators ``<A_i B_j>`` for ``i < j``. Operators are Pauli labels
(``"X"``, ``"Y"`` or ``"Z"``) from YAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.physics.operators import X, Y, Z, get_site_operator
from QELM.physics.readout import Readout, ReadoutResult

PAULIS: dict[str, np.ndarray] = {"X": X, "Y": Y, "Z": Z}


def pauli(label: str) -> np.ndarray:
    """Return the 2x2 Pauli matrix named ``label``.

    Args:
        label: ``"X"``, ``"Y"`` or ``"Z"``.

    Returns:
        Complex matrix of shape ``(2, 2)``.

    Raises:
        ValueError: If ``label`` is not a Pauli label.
    """
    try:
        return PAULIS[label]
    except KeyError as exc:
        known = ", ".join(PAULIS)
        raise ValueError(f"Unknown observable {label!r}. Known: {known}.") from exc


def n_qubits(rho: np.ndarray) -> int:
    """Return ``N`` for a density matrix of shape ``(2**N, 2**N)``.

    Args:
        rho: Density matrix.

    Returns:
        Number of qubits.

    Raises:
        ValueError: If ``rho`` is not square with a power-of-two side.
    """
    dim = rho.shape[0]
    N = dim.bit_length() - 1
    if rho.shape != (dim, dim) or dim < 2 or 2**N != dim:
        raise ValueError(f"rho must have shape (2**N, 2**N), got {rho.shape}.")
    return N


class Observables(Readout):
    """Set of operators whose expectation values are the QELM features.

    The readout is deterministic and leaves ``rho`` unchanged.

    Attributes:
        name: Config key for this observable set; ``from_config`` reads
            ``cfg[name]``.
        params: Scalar parameters from the matching YAML block.
    """

    name: str = "base"

    def operators(self, N: int) -> list[np.ndarray]:
        """Return the measured operators on ``N`` qubits.

        Args:
            N: Number of qubits.

        Returns:
            Operators of shape ``(2**N, 2**N)``, one per feature.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} operators are not implemented.")

    def expectation(self, rho: np.ndarray) -> np.ndarray:
        """Return the features ``Tr(rho O_k)`` for every operator ``O_k``.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.

        Returns:
            Real array of shape ``(n_features,)``.
        """
        ops = self.operators(n_qubits(rho))
        # Tr(rho O) = sum_ij rho_ij O_ji; real because rho and O are Hermitian.
        return np.array([np.einsum("ij,ji->", rho, op).real for op in ops])

    def measure(
        self, rho: np.ndarray, rng: np.random.Generator | None = None
    ) -> ReadoutResult:
        """Return :meth:`expectation` as values and ``rho`` unchanged.

        Args:
            rho: Density matrix of shape ``(2**N, 2**N)``.
            rng: Ignored; expectation values are deterministic.

        Returns:
            Values of shape ``(n_features,)`` and the input ``rho``.
        """
        return ReadoutResult(values=self.expectation(rho), rho=rho)


class SingleObservable(Observables):
    """One-site expectation values ``<O_i>`` for ``i = 0 .. N-1``.

    Attributes:
        observable: Pauli label of ``O``, from ``params["observable"]``.
    """

    name: str = "single_observable"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store and check the Pauli label.

        Args:
            params: ``observable`` is required, such as ``"Z"``.
        """
        super().__init__(params)
        self.observable: str = str(params["observable"])
        pauli(self.observable)

    def operators(self, N: int) -> list[np.ndarray]:
        """Return ``[O_0, ..., O_{N-1}]``, so ``N`` features."""
        P = pauli(self.observable)
        return [get_site_operator(P, i, N) for i in range(N)]


class DoubleObservable(Observables):
    """Two-point correlators ``<A_i B_j>`` for every pair ``i < j``.

    Attributes:
        observable_1: Pauli label of ``A``, from ``params["observable_1"]``.
        observable_2: Pauli label of ``B``, from ``params["observable_2"]``.
    """

    name: str = "double_observable"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store and check the two Pauli labels.

        Args:
            params: ``observable_1`` and ``observable_2`` are required.
        """
        super().__init__(params)
        self.observable_1: str = str(params["observable_1"])
        self.observable_2: str = str(params["observable_2"])
        pauli(self.observable_1)
        pauli(self.observable_2)

    def operators(self, N: int) -> list[np.ndarray]:
        """Return ``A_i B_j`` for ``i < j`` in lexicographic order.

        That is ``N (N - 1) / 2`` features: ``(0, 1), (0, 2), ..., (N-2, N-1)``.
        """
        A = pauli(self.observable_1)
        B = pauli(self.observable_2)
        return [
            get_site_operator(A, i, N) @ get_site_operator(B, j, N)
            for i in range(N)
            for j in range(i + 1, N)
        ]
