"""Parent class of the regressions fitted on the QELM features.

The base ``fit`` and ``predict`` are stubs; :class:`RidgeReadout` overrides
them with the closed-form solution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf


class Trainer:
    """Read parameters from config and expose fit/predict hooks.

    Attributes:
        name: Config key for this trainer.
        params: Parameters from the ``training`` block, without ``name``.
    """

    name: str

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Store the trainer parameters.

        Args:
            params: Parameter names mapped to scalar values from the
                ``training`` YAML block. ``None`` means no parameter.
        """
        self.params: dict[str, Any] = dict(params or {})

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the trainer from the composed config.

        Reads the ``training`` block (``conf/training/<name>.yaml``), minus
        its ``name`` key. There is no learning rate: ridge regression uses
        the closed-form solution.

        Args:
            cfg: Composed Hydra config. ``cfg.training`` may be absent.

        Returns:
            An instance of this trainer class.
        """
        node = OmegaConf.select(cfg, "training")
        raw = OmegaConf.to_container(node, resolve=True) if node is not None else {}
        if not isinstance(raw, dict):
            raise TypeError("training config must be a mapping.")
        params = {str(key): value for key, value in raw.items() if key != "name"}
        return cls(params)

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

    def predict(self, features: np.ndarray | None) -> np.ndarray | str:
        """Record a predict request without emitting a tensor.

        Args:
            features: Design matrix of shape ``(n_samples, n_features)``,
                or ``None``.

        Returns:
            The status string ``"scaffold"`` here. A trainer that fits
            weights returns predictions of shape ``(n_samples,)`` or
            ``(n_samples, n_outputs)``.
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
