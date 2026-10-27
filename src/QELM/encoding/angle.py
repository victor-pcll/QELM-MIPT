"""Angle encoding: one feature per qubit, a polar angle (issue #196)."""

from __future__ import annotations

import numpy as np

from QELM.encoding.base import ProductEncoding


class AngleEncoding(ProductEncoding):
    """``|x> = ⊗_i (cos(x_i / 2) |0> + sin(x_i / 2) |1>)``, ``x_i in [0, pi]``.

    One feature per qubit, ``M = N``. The same state as
    :class:`QELM.encoding.RotationEncoding` with ``axis = "y"``
    (``R_y(x_i) |0>``), with the bounds of the feature checked. Config
    block: ``angle_encoding: {N}``.
    """

    name: str = "angle_encoding"
    features_per_qubit: int = 1
    bounds: tuple[tuple[float, float], ...] = ((0.0, np.pi),)

    def amplitudes(self, groups: np.ndarray) -> np.ndarray:
        """Return ``(cos(x/2), sin(x/2))`` per qubit, ``(n_samples, N, 2)``."""
        theta = groups[..., 0]
        return np.stack([np.cos(theta / 2), np.sin(theta / 2)], axis=-1)
