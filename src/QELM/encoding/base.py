"""Base class for the ways of encoding an input into the reservoir state.

An encoding prepares the initial state of the reservoir: a pure state
``psi`` of shape ``(dim,)`` and its density matrix ``rho = |psi><psi|`` of
shape ``(dim, dim)``. ``N`` is the total number of qubits, so
``dim = 2**N`` matches the models built with the same ``N``. Some encodings
use ``Nr`` of them as ancillas, which leaves ``Ns = N - Nr``
system qubits.

Qubit ordering follows :func:`QELM.physics.operators.get_site_operator`:
site 0 is the leftmost factor of ``np.kron``, i.e. the most significant bit
of a basis index. Ancillas come first (sites ``0 .. Nr - 1``), the
``Ns`` system qubits after them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self, overload

import numpy as np
import scipy.sparse as sp
from omegaconf import DictConfig, OmegaConf


class Encoding:
    """Base class for state encodings configured from a Hydra block.

    Attributes:
        name: Config key for this encoding; ``from_config`` reads
            ``cfg[name]``.
        Nr: Number of ancilla qubits placed before the system.
        params: Scalar parameters from the matching YAML block.
        N: Total number of qubits, ancillas included.
        Ns: Number of system qubits, ``N - Nr``.
        dim: Dimension of the encoded state, ``2**N``.
    """

    name: str = "base"
    Nr: int = 0

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
        self.Ns: int = self.N - self.Nr
        if self.Ns < 1:
            raise ValueError(
                f"N={self.N} leaves no system qubit next to {self.Nr} ancilla(s)."
            )
        self.dim: int = 2**self.N

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the encoding from the Hydra block named ``cls.name``.

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

    @overload
    def state(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...
    @overload
    def state(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...
    def state(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return the encoded pure state.

        Args:
            x: Input data to encode. ``None`` for an encoding that does not
                depend on the input.
            sparse: ``True`` for a CSR column, ``False`` (default) for a
                NumPy vector.

        Returns:
            Normalized complex state: a CSR column of shape ``(dim, 1)`` if
            ``sparse``, otherwise a vector of shape ``(dim,)``.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} state is not implemented.")

    @overload
    def density_matrix(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...
    @overload
    def density_matrix(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...
    def density_matrix(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return ``rho = |psi><psi|`` for ``psi = self.state(x)``.

        Args:
            x: Input data, passed to :meth:`state`.
            sparse: ``True`` for a CSR matrix built from the sparse state
                (``k**2`` stored entries for ``k`` non-zero amplitudes),
                ``False`` (default) for a NumPy array.

        Returns:
            Complex density matrix of shape ``(dim, dim)``.
        """
        if sparse:
            psi_sp = self.state(x, sparse=True)
            return sp.csr_matrix(psi_sp @ psi_sp.conj().T)
        psi = self.state(x)
        return np.outer(psi, psi.conj())

    @overload
    def rho(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...
    @overload
    def rho(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...
    def rho(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return the initial density matrix ``rho = |psi><psi|``.

        Short name for :meth:`density_matrix`, inherited by every encoding.

        Args:
            x: Input data, passed to :meth:`state`.
            sparse: ``True`` for a CSR matrix, ``False`` (default) for a
                NumPy array.

        Returns:
            Complex density matrix of shape ``(2**N, 2**N)``.
        """
        if sparse:
            return self.density_matrix(x, sparse=True)
        return self.density_matrix(x)
