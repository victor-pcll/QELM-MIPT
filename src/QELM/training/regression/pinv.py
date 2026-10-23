"""Unregularized readout: the Moore-Penrose pseudo-inverse (issue #181)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.training.regression.base import Trainer


class PseudoInverseReadout(Trainer):
    """Least-squares readout ``W = Phi^+ y``, with no penalty.

    ``Phi^+`` is the Moore-Penrose pseudo-inverse (``numpy.linalg.pinv``,
    from the SVD of ``Phi``). It is the limit ``lambda -> 0`` of the ridge
    solution in both regimes:

    - more samples than features (``Phi`` of full column rank): the
      ordinary least-squares solution, ``(Phi^T Phi)^{-1} Phi^T y``;
    - fewer samples than features: the interpolating solution
      (``Phi W = y`` exactly) of smallest norm ``||W||``.

    At ``n_samples = rank(Phi)`` the smallest singular value of ``Phi`` is
    close to 0 and ``||W||`` blows up: this is the interpolation peak of the
    double descent, which a ridge penalty would hide. Selected with
    ``training/regression=pinv``. No intercept is fitted: add a constant
    column to ``Phi`` if a bias is needed.

    Attributes:
        rcond: Relative cutoff on the singular values passed to
            ``numpy.linalg.pinv`` as ``rtol``; ``None`` for NumPy's default,
            ``max(n_samples, n_features) * eps``.
        W: Fitted weights of shape ``(n_features,)`` or
            ``(n_features, n_outputs)``, or ``None`` before :meth:`fit`.
    """

    name = "pinv"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Store the parameters and read ``rcond``.

        Args:
            params: Optional ``rcond``, a non-negative cutoff or ``None``.

        Raises:
            ValueError: If ``rcond`` is negative or not a number.
        """
        super().__init__(params)
        rcond = self.params.get("rcond")
        if rcond is not None:
            if isinstance(rcond, bool) or not float(rcond) >= 0:
                raise ValueError(f"rcond must be None or >= 0, got {rcond!r}.")
            rcond = float(rcond)
        self.rcond: float | None = rcond
        self.W: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray | None,
        targets: np.ndarray | None,
    ) -> str:
        """Compute ``W = Phi^+ y``.

        Args:
            features: ``Phi``, of shape ``(n_samples, n_features)``, or
                ``None`` when no dataset has been built yet.
            targets: ``y``, of shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``, or ``None``.

        Returns:
            ``"fitted"``, or ``"scaffold"`` when ``features`` or ``targets``
            is ``None``.

        Raises:
            ValueError: If the shapes do not match.
        """
        if features is None or targets is None:
            return "scaffold"
        phi = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if phi.ndim != 2:
            raise ValueError(
                f"features must have shape (n_samples, n_features), got {phi.shape}."
            )
        if y.ndim not in (1, 2) or y.shape[0] != phi.shape[0]:
            raise ValueError(
                f"targets must have {phi.shape[0]} rows like features, "
                f"got shape {y.shape}."
            )
        pinv = (
            np.linalg.pinv(phi)
            if self.rcond is None
            else np.linalg.pinv(phi, rtol=self.rcond)
        )
        self.W = pinv @ y
        return "fitted"

    def predict(self, features: np.ndarray | None) -> np.ndarray:
        """Return ``Phi @ W``.

        Args:
            features: ``Phi`` of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.

        Raises:
            RuntimeError: If :meth:`fit` has not computed ``W`` yet.
        """
        if self.W is None or features is None:
            raise RuntimeError("The model must be fitted before calling predict().")
        return np.asarray(features, dtype=np.float64) @ self.W

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return ``W`` under ``weights``, or a length-0 array before fit."""
        if self.W is None:
            return super().state_dict()
        return {"weights": self.W}
