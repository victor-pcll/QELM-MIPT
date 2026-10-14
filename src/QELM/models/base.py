"""Base class for Hamiltonian models selected by ``sim_type``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, TypedDict

import numpy as np
import scipy.linalg
from omegaconf import DictConfig, OmegaConf


class EvolutionState(TypedDict):
    """Record of a requested evolution. No trajectory tensor is stored.

    Attributes:
        model: Simulation name, such as ``"TFIM"``.
        dt: Scalar integration step.
        n_steps: Scalar number of requested steps.
        status: ``"scaffold"`` until integration is implemented.
    """

    model: str
    dt: float
    n_steps: int
    status: str


class HamiltonianModel:
    """Time-evolution model selected by ``sim_type``.

    Attributes:
        name: Config key for this model. Subclasses override it with their
            ``sim_type`` value; ``from_config`` reads ``cfg[name]``.
        params: Scalar parameters from the matching YAML block, such as
            ``N`` and ``J``.
        N: Number of qubits.
        dim: Hilbert-space dimension, ``2**N``.
    """

    name: str = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store scalar Hamiltonian parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``N`` is required.
        """
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.dim: int = 2**self.N
        self._H: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the model from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this model class.

        Raises:
            TypeError: If the config block is not a mapping.
        """
        node = cfg[cls.name]
        # A scalar such as ``TFIM: 3`` is not a config node, and
        # ``to_container`` would raise a ValueError on it.
        raw = (
            OmegaConf.to_container(node, resolve=True)
            if OmegaConf.is_config(node)
            else node
        )
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        params = {str(key): value for key, value in raw.items()}
        return cls(params)

    def hamiltonian(self) -> np.ndarray:
        """Return the Hamiltonian matrix for ``self.params``.

        Returns:
            Not built by this scaffold. A later implementation would return
            an operator of shape ``(dim, dim)``.

        Raises:
            NotImplementedError: Always, until the operator is defined.
        """
        raise NotImplementedError(f"{self.name} Hamiltonian is not implemented.")

    def evolve(self, *, dt: float, n_steps: int) -> EvolutionState:
        """Record a requested evolution without integrating it.

        Args:
            dt: Scalar integration step.
            n_steps: Scalar number of steps. A later implementation would
                return states of shape ``(n_steps, dim)``. This scaffold
                does not allocate that tensor.

        Returns:
            An :class:`EvolutionState` recording ``dt`` and ``n_steps``.
        """
        return {
            "model": self.name,
            "dt": dt,
            "n_steps": n_steps,
            "status": "scaffold",
        }

    def propagator(self, *, dt: float) -> np.ndarray:
        """Return the unitary ``U(dt) = exp(-i H dt)``, with ``hbar = 1``.

        Args:
            dt: Scalar evolution time.

        Returns:
            Complex unitary matrix of shape ``(dim, dim)``.
        """
        H = self._H if self._H is not None else self.hamiltonian()
        return np.asarray(scipy.linalg.expm(-1j * dt * H))

    def evolve_state(self, *, dt: float, psi: np.ndarray) -> np.ndarray:
        """Evolve a pure state: ``psi(dt) = U(dt) psi``.

        Args:
            dt: Scalar evolution time.
            psi: State vector of shape ``(dim,)``, or several states as the
                columns of an array of shape ``(dim, n_states)``.

        Returns:
            Evolved state(s), with the same shape as ``psi``.
        """
        return self.propagator(dt=dt) @ psi

    def evolve_density(self, *, dt: float, rho: np.ndarray) -> np.ndarray:
        """Evolve a density matrix: ``rho(dt) = U(dt) rho U(dt)^dagger``.

        Args:
            dt: Scalar evolution time.
            rho: Density matrix of shape ``(dim, dim)``.

        Returns:
            Evolved density matrix of shape ``(dim, dim)``.
        """
        U = self.propagator(dt=dt)
        return U @ rho @ U.conj().T
