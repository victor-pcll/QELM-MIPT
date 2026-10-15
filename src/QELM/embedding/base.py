"""Base class for the ways of encoding an input into the reservoir state.

An embedding prepares the initial state of the reservoir: a pure state
``psi`` of shape ``(dim,)`` and its density matrix ``rho = |psi><psi|`` of
shape ``(dim, dim)``. ``N`` is the total number of qubits, so
``dim = 2**N`` matches the models built with the same ``N``. Some embeddings
use ``n_ancilla`` of them as ancillas, which leaves ``N_S = N - n_ancilla``
system qubits.

Qubit ordering follows :func:`QELM.physics.operators.get_site_operator`:
site 0 is the leftmost factor of ``np.kron``, i.e. the most significant bit
of a basis index. Ancillas come first (sites ``0 .. n_ancilla - 1``), the
``N_S`` system qubits after them.
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
        N: Total number of qubits, ancillas included.
        N_S: Number of system qubits, ``N - n_ancilla``.
        dim: Dimension of the embedded state, ``2**N``.
    """

    name: str = "base"
    n_ancilla: int = 0

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``N``, the total number of qubits, is required.

        Raises:
            ValueError: If ``N`` leaves no system qubit.
        """
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.N_S: int = self.N - self.n_ancilla
        if self.N_S < 1:
            raise ValueError(
                f"N={self.N} leaves no system qubit next to "
                f"{self.n_ancilla} ancilla(s)."
            )
        self.dim: int = 2**self.N

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the embedding from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this class.

        Raises:
            ValueError: If ``cfg`` has no ``cls.name`` block.
            TypeError: If the config block is not a mapping.
        """
        node = OmegaConf.select(cfg, cls.name)
        if node is None:
            raise ValueError(
                f"The config has no {cls.name!r} block. Compose the full config "
                f"with QELM.load_config(), or add `{cls.name}: {{N: ...}}`."
            )
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
