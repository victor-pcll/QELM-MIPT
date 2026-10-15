"""Parent class of the regressions fitted on the QELM features.

Fit and predict are stubs.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from omegaconf import DictConfig


class Trainer:
    """Read parameters from config and expose fit/predict hooks.

    Attributes:
        name: Config key for this trainer.
    """

    name: str

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the trainer from the composed config.

        Trainers take no config parameter: ridge regression uses the
        closed-form solution, so there is no learning rate.

        Args:
            cfg: Composed Hydra config. Not read yet; kept so every
                component is built the same way.

        Returns:
            An instance of this trainer class.
        """
        return cls()

    def fit(
        self,
        features: np.ndarray | None,
        targets: np.ndarray | None,
    ) -> str:
        """Record a fit request without estimating weights.

        Args:
            features: Design matrix of shape ``(n_samples, n_features)``,
                or ``None`` when no dataset has been built.
            targets: Labels of shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``, or ``None``.

        Returns:
            The status string ``"scaffold"``.
        """
        return "scaffold"

    def predict(self, features: np.ndarray | None) -> str:
        """Record a predict request without emitting a tensor.

        Args:
            features: Design matrix of shape ``(n_samples, n_features)``,
                or ``None``.

        Returns:
            The status string ``"scaffold"``. A later implementation would
            return predictions of shape ``(n_samples, n_outputs)``.
        """
        return "scaffold"

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return arrays to store in a checkpoint.

        The scaffold has not estimated a weight vector. ``weights`` is a
        length-0 ``float64`` array. A later fit would store shape
        ``(n_features,)`` or ``(n_features, n_outputs)``.

        Returns:
            Mapping whose ``weights`` entry is the trainable state.
        """
        return {"weights": np.zeros((0,), dtype=np.float64)}
