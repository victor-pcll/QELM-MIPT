"""Base class for Hamiltonian models selected by ``sim_type``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, TypedDict

import numpy as np
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
        rng: Generator for random parameters such as disorder, or ``None``
            to use NumPy's global generator.
    """

    name: str = "base"

    def __init__(
        self,
        params: Mapping[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store scalar Hamiltonian parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``N`` is required.
            rng: Generator the model draws its random parameters from. Pass
                the same one to every model of a run (see
                ``QELM.seed.make_rng``) to get a new, reproducible
                realisation per model. ``None`` falls back to NumPy's global
                generator, seeded by ``set_seed``.
        """
        self.rng: np.random.Generator | None = rng
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.dim: int = 2**self.N
        self._H: np.ndarray | None = None

    @classmethod
    def from_config(
        cls, cfg: DictConfig, rng: np.random.Generator | None = None
    ) -> Self:
        """Build the model from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.
            rng: Generator passed to the constructor. See ``__init__``.

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
        return cls(params, rng=rng)

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
