"""Dense angle encoding: two features per qubit, polar angle and phase (#196)."""

from __future__ import annotations

import numpy as np

from QELM.encoding.base import ProductEncoding


class DenseAngleEncoding(ProductEncoding):
    """``⊗_i (cos(x_{2i-1} / 2) |0> + e^{i x_{2i}} sin(x_{2i-1} / 2) |1>)``.

    Two features per qubit, ``N = ceil(M / 2)``: the first sets the polar
    angle of the Bloch vector, the second its azimuth. Both lie in
    ``[0, pi]``. Config block: ``dense_angle_encoding: {N, n_features}``.
    """

    name: str = "dense_angle_encoding"
    features_per_qubit: int = 2
    bounds: tuple[tuple[float, float], ...] = ((0.0, np.pi), (0.0, np.pi))

    def amplitudes(self, groups: np.ndarray) -> np.ndarray:
        """Return ``(cos(t/2), e^{i phi} sin(t/2))`` per qubit."""
        theta, phi = groups[..., 0], groups[..., 1]
        return np.stack(
            [np.cos(theta / 2) + 0j, np.exp(1j * phi) * np.sin(theta / 2)], axis=-1
        )
