"""Feature extraction: observables measured on the final reservoir state.

A single observable ``O`` gives the ``N`` features ``<O_i> = Tr(rho O_i)``.
A double observable ``(A, B)`` gives the ``N (N - 1) / 2`` two-point
correlators ``<A_i B_j>`` for ``i < j``. Operators are Pauli labels
(``"X"``, ``"Y"`` or ``"Z"``) from YAML.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf

from QELM.physics.operators import get_site_operator, n_qubits, pauli


class Observables:
    """Set of operators whose expectation values are the QELM features.

    Attributes:
        name: Config key for this observable set; ``from_config`` reads
            ``cfg[name]``.
        params: Scalar parameters from the matching YAML block.
    """

    name: str = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
        """
        self.params: dict[str, Any] = dict(params)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the observable set from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this class.

        Raises:
            TypeError: If the config block is not a mapping.
        """
        node = cfg[cls.name]
        raw = (
            OmegaConf.to_container(node, resolve=True)
            if OmegaConf.is_config(node)
            else node
        )
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        return cls({str(key): value for key, value in raw.items()})

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
