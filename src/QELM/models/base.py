"""Base class for Hamiltonian models selected by ``sim_type``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, TypedDict

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg
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


def evolution_request(
    model: HamiltonianModel, *, dt: float, n_steps: int
) -> EvolutionState:
    """Record a requested evolution without integrating it.

    Args:
        model: Model the evolution is requested for.
        dt: Scalar integration step.
        n_steps: Scalar number of steps. No trajectory of shape
            ``(n_steps, dim)`` is allocated.

    Returns:
        An :class:`EvolutionState` recording ``dt`` and ``n_steps``.
    """
    return {
        "model": model.name,
        "dt": dt,
        "n_steps": n_steps,
        "status": "scaffold",
    }


class HamiltonianModel:
    """Time-evolution model selected by ``sim_type``.

    Attributes:
        name: Config key for this model. Subclasses override it with their
            ``sim_type`` value; ``from_config`` reads ``cfg[name]``.
        params: Scalar parameters from the matching YAML block, such as
            ``N`` and ``J``.
        N: Number of qubits.
        dim: Hilbert-space dimension, ``2**N``.
        dt: Default step used by :meth:`evolve` when no ``dt`` is passed.
            ``from_config`` sets it from ``evolution.dt``; ``None`` otherwise.
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
        self.dt: float | None = None
        self._H: sp.csr_matrix | None = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the model from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this model class, with ``dt`` set from
            ``cfg.evolution.dt`` when that key exists.

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
        model = cls(params)
        dt = OmegaConf.select(cfg, "evolution.dt")
        if dt is not None:
            model.dt = float(dt)
        return model

    def hamiltonian(self) -> sp.csr_matrix:
        """Return the Hamiltonian matrix for ``self.params``.

        Returns:
            Not built by this scaffold. A later implementation returns a
            sparse CSR matrix of shape ``(dim, dim)`` and stores it in
            ``self._H``.

        Raises:
            NotImplementedError: Always, until the operator is defined.
        """
        raise NotImplementedError(f"{self.name} Hamiltonian is not implemented.")

    def _cached_hamiltonian(self) -> sp.csr_matrix:
        """Return ``self._H``, building it on first use."""
        if self._H is None:
            self._H = sp.csr_matrix(self.hamiltonian())
        return self._H

    def propagator(self, *, dt: float) -> np.ndarray:
        """Return the unitary ``U(dt) = exp(-i H dt)``, with ``hbar = 1``.

        ``U`` is dense in general, even when ``H`` is sparse, so this builds
        a ``(dim, dim)`` matrix. :meth:`evolve` does not call it: it applies
        ``U`` to its argument directly from the sparse ``H``.

        Args:
            dt: Scalar evolution time.

        Returns:
            Complex unitary matrix of shape ``(dim, dim)``.
        """
        H = self._cached_hamiltonian().toarray()
        return np.asarray(scipy.linalg.expm(-1j * dt * H))

    def evolve(self, state: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Evolve a pure state or a density matrix by ``U(dt) = exp(-i H dt)``.

        The kind of ``state`` is read from its shape:

        - ``(dim,)``: a state vector ``psi``, returned as ``U psi``;
        - ``(dim, dim)``: a density matrix ``rho``, returned as
          ``U rho U^dagger``.

        ``U`` is never formed. ``scipy.sparse.linalg.expm_multiply``
        computes ``exp(-i H dt) v`` from products ``H @ v`` (a truncated
        Taylor series). For ``rho`` it is applied twice, as
        ``U (U rho^dagger)^dagger``.

        Args:
            state: State vector of shape ``(dim,)`` or density matrix of
                shape ``(dim, dim)``.
            dt: Scalar evolution time. ``None`` uses ``self.dt``, set from
                ``evolution.dt`` by :meth:`from_config`.

        Returns:
            The evolved state, with the same shape as ``state``.

        Raises:
            ValueError: If ``state`` has any other shape, or if ``dt`` is
                ``None`` and ``self.dt`` is not set.
        """
        if dt is None:
            if self.dt is None:
                raise ValueError(
                    "evolve needs dt: pass dt=..., or build the model with "
                    "from_config (reads evolution.dt) or set model.dt."
                )
            dt = self.dt
        state = np.asarray(state, dtype=complex)
        if state.shape == (self.dim,):
            return self._apply_propagator(dt, state)
        if state.shape == (self.dim, self.dim):
            # (U rho^dagger)^dagger = rho U^dagger, then U (rho U^dagger).
            U_rho_dag = self._apply_propagator(dt, state.conj().T)
            return self._apply_propagator(dt, U_rho_dag.conj().T)
        raise ValueError(
            f"evolve expects a state vector of shape ({self.dim},) or a "
            f"density matrix of shape ({self.dim}, {self.dim}) for N={self.N}, "
            f"got shape {state.shape}."
        )

    def _apply_propagator(self, dt: float, v: np.ndarray) -> np.ndarray:
        """Return ``exp(-i H dt) @ v`` for ``v`` of shape ``(dim,)`` or ``(dim, k)``."""
        A = (-1j * dt) * self._cached_hamiltonian()
        return np.asarray(scipy.sparse.linalg.expm_multiply(A, v))
