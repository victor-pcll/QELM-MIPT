"""Hamiltonian models. Physics is not implemented; these types are routing stubs."""

from __future__ import annotations

from typing import Any, Mapping, Self

from omegaconf import DictConfig, OmegaConf


class HamiltonianModel:
    """Time-evolution model selected by ``sim_type``."""

    name: str

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = dict(params)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the model from the Hydra block named ``cls.name``."""
        raw = OmegaConf.to_container(cfg[cls.name], resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        return cls(raw)

    def hamiltonian(self) -> object:
        """Return the Hamiltonian for ``self.params``. Not implemented."""
        raise NotImplementedError(f"{self.name} Hamiltonian is not implemented.")

    def evolve(self, *, dt: float, n_steps: int) -> dict[str, Any]:
        """Record the requested evolution. Does not integrate a trajectory."""
        return {
            "model": self.name,
            "dt": dt,
            "n_steps": n_steps,
            "status": "scaffold",
        }
