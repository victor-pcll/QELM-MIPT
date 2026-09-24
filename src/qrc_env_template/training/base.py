"""Training algorithms. Fit and predict are stubs."""

from __future__ import annotations

from typing import Self

from omegaconf import DictConfig


class Trainer:
    """Read parameters from config and expose fit/predict hooks."""

    name: str

    def __init__(self, learning_rate: float | None = None) -> None:
        self.learning_rate = learning_rate

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Read ``learning_rate`` from the selected simulation block when present."""
        sim = cfg[str(cfg.sim_type)]
        learning_rate = sim.get("learning_rate")
        if learning_rate is not None:
            learning_rate = float(learning_rate)
        return cls(learning_rate=learning_rate)

    def fit(self, features: object, targets: object) -> str:
        """Scaffold. Does not fit a model."""
        return "scaffold"

    def predict(self, features: object) -> str:
        """Scaffold. Does not emit predictions."""
        return "scaffold"
