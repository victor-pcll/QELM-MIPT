"""General encoding: two features per qubit as real amplitudes (issue #196)."""

from __future__ import annotations

import numpy as np

from QELM.encoding.base import ProductEncoding


class GeneralEncoding(ProductEncoding):
    """``⊗_i (x_{2i-1} |0> + x_{2i} |1>) / sqrt(x_{2i-1}^2 + x_{2i}^2)``.

    Two features per qubit, ``N = ceil(M / 2)``, both in ``[0, 1]``. The
    state of a qubit is real and depends only on the ratio of its two
    features: it is ``R_y(theta)|0>`` with ``theta = 2 atan2(x_{2i},
    x_{2i-1})`` in ``[0, pi]``. A pair ``(0, 0)`` has no state and raises.
    Config block: ``general_encoding: {N, n_features}``.
    """

    name: str = "general_encoding"
    features_per_qubit: int = 2
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))

    def amplitudes(self, groups: np.ndarray) -> np.ndarray:
        """Return ``(x_1, x_2) / ||(x_1, x_2)||`` per qubit.

        Raises:
            ValueError: If both features of a qubit are 0.
        """
        norm = np.linalg.norm(groups, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError(
                f"{self.name}: a qubit with both features 0 has no state; "
                "shift the features away from (0, 0)."
            )
        return groups / norm
