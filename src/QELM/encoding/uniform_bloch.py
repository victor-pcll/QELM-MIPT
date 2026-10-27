"""Uniform Bloch sphere encoding: two features per qubit (issue #196)."""

from __future__ import annotations

import numpy as np

from QELM.encoding.base import ProductEncoding


class UniformBlochEncoding(ProductEncoding):
    """``⊗_i (sqrt(x_{2i-1}) |0> + e^{i x_{2i}} sqrt(1 - x_{2i-1}) |1>)``.

    Two features per qubit, ``N = ceil(M / 2)``. The first, in ``[0, 1]``,
    is the population of ``|0>``, so ``<Z> = 2 x_{2i-1} - 1`` is linear in
    it, where the angle encoding gives ``<Z> = cos(x)``: equal steps of the
    feature cut equal areas of the sphere (Archimedes). The second, in
    ``[0, pi]``, is the phase. Config block:
    ``uniform_bloch_encoding: {N, n_features}``.
    """

    name: str = "uniform_bloch_encoding"
    features_per_qubit: int = 2
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, np.pi))

    def amplitudes(self, groups: np.ndarray) -> np.ndarray:
        """Return ``(sqrt(u), e^{i phi} sqrt(1 - u))`` per qubit."""
        # clip: a feature within atol of a bound can be a hair outside it.
        u = np.clip(groups[..., 0], 0.0, 1.0)
        phi = groups[..., 1]
        return np.stack([np.sqrt(u) + 0j, np.exp(1j * phi) * np.sqrt(1 - u)], axis=-1)
