"""Base class for the ways of encoding an input into the reservoir state.

An embedding prepares the initial state of the reservoir: a pure state
``psi`` of shape ``(dim,)`` and its density matrix ``rho = |psi><psi|`` of
shape ``(dim, dim)``. Some embeddings add ancilla qubits to the ``N``
system qubits, so ``dim = 2**(N + n_ancilla)``.

Qubit ordering follows :func:`QELM.physics.operators.get_site_operator`:
site 0 is the leftmost factor of ``np.kron``, i.e. the most significant bit
of a basis index. Ancillas come first (sites ``0 .. n_ancilla - 1``), the
system qubits after them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf


class Embedding:
    """Base class for state embeddings configured from a Hydra block.

    Attributes:
        name: Config key for this embedding; ``from_config`` reads
            ``cfg[name]``.
        n_ancilla: Number of ancilla qubits placed before the system.
        params: Scalar parameters from the matching YAML block.
        N: Number of system qubits.
        n_qubits: Total number of qubits, ``n_ancilla + N``.
        dim: Dimension of the embedded state, ``2**n_qubits``.
    """

    name: str = "base"
    n_ancilla: int = 0

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``N``, the number of system qubits, is required.
        """
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.n_qubits: int = self.n_ancilla + self.N
        self.dim: int = 2**self.n_qubits

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the embedding from the Hydra block named ``cls.name``.

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

    def state(self, x: np.ndarray | None = None) -> np.ndarray:
        """Return the embedded pure state.

        Args:
            x: Input data to encode. ``None`` for an embedding that does not
                depend on the input.

        Returns:
            Normalized complex vector of shape ``(dim,)``.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} state is not implemented.")

    def density_matrix(self, x: np.ndarray | None = None) -> np.ndarray:
        """Return ``rho = |psi><psi|`` for ``psi = self.state(x)``.

        Args:
            x: Input data, passed to :meth:`state`.

        Returns:
            Complex density matrix of shape ``(dim, dim)``.
        """
        psi = self.state(x)
        return np.outer(psi, psi.conj())
