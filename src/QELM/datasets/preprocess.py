"""From pixels to rotation angles: PCA, then a linear map to ``[0, pi]``."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA  # type: ignore[import-untyped]


class AngleEncoder:
    """PCA to ``n_components`` dimensions, each mapped linearly to an angle.

    Fitted on the training images only, so the test set never informs the
    preprocessing. Component ``k`` is mapped by
    ``theta_k = angle_range * (z_k - min_k) / (max_k - min_k)``, with the
    minimum and maximum over the training images, then clipped to
    ``[0, angle_range]`` (a test image can fall outside).

    With ``angle_range = pi`` (default) the rotation ``R_y(theta)`` sends
    the smallest value to ``|0>`` and the largest to ``|1>``, so the full
    range of each component is used and no two values share an angle.

    Attributes:
        n_components: Number of principal components, one per qubit.
        angle_range: Width of the interval of angles.
        pca: The fitted :class:`sklearn.decomposition.PCA`, or ``None``.
        low: Minimum of each component over the training images.
        high: Maximum of each component over the training images.
    """

    def __init__(self, n_components: int, angle_range: float = np.pi) -> None:
        """Store the parameters.

        Args:
            n_components: Number of principal components (>= 1).
            angle_range: Width of the interval of angles (> 0).

        Raises:
            ValueError: If a parameter is out of range.
        """
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        if not angle_range > 0:
            raise ValueError(f"angle_range must be > 0, got {angle_range}.")
        self.n_components = int(n_components)
        self.angle_range = float(angle_range)
        self.pca: PCA | None = None
        self.low: np.ndarray | None = None
        self.high: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> AngleEncoder:
        """Fit the PCA and the range of each component on ``x``.

        Args:
            x: Training images, of shape ``(n_samples, n_pixels)``.

        Returns:
            ``self``.
        """
        self.pca = PCA(n_components=self.n_components).fit(x)
        z = self.pca.transform(x)
        self.low, self.high = z.min(axis=0), z.max(axis=0)
        return self

    def components(self, x: np.ndarray) -> np.ndarray:
        """Return the principal components of ``x``, ``(n_samples, n_components)``.

        Raises:
            RuntimeError: Before :meth:`fit`.
        """
        if self.pca is None:
            raise RuntimeError("Call fit() before transform().")
        return np.asarray(self.pca.transform(x))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return the angles of ``x``, of shape ``(n_samples, n_components)``.

        Raises:
            RuntimeError: Before :meth:`fit`.
        """
        z = self.components(x)
        assert self.low is not None and self.high is not None
        span = np.where(self.high > self.low, self.high - self.low, 1.0)
        return self.angle_range * np.clip((z - self.low) / span, 0.0, 1.0)
