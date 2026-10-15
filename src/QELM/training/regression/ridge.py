"""Analytical ridge regression readout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.training.regression.base import Trainer


class RidgeReadout(Trainer):
    """Ridge regression on the reservoir features, solved in closed form.

    ``W = (Phi^T Phi + alpha I)^{-1} Phi^T y``, computed by solving the linear
    system ``(Phi^T Phi + alpha I) W = Phi^T y`` with ``numpy.linalg.solve``
    rather than inverting the matrix. No intercept is fitted: add a constant
    column to ``Phi`` if a bias is needed. Selected with ``training/regression=ridge``.

    ``alpha`` is resolved in this order: the argument of :meth:`fit`, then
    ``alpha`` in the ``training`` config block, then 1.0 (as in
    scikit-learn).

    Attributes:
        config_alpha: ``alpha`` from the config, or 1.0.
        W: Fitted weights of shape ``(n_features,)`` or
            ``(n_features, n_outputs)``, or ``None`` before :meth:`fit`.
    """

    name = "ridge"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Store the parameters and read ``alpha``.

        Args:
            params: Optional ``alpha``, a non-negative regularization
                strength (default 1.0).

        Raises:
            ValueError: If ``alpha`` is negative or not a number.
        """
        super().__init__(params)
        self.config_alpha: float = _check_alpha(self.params.get("alpha", 1.0))
        self.W: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray | None,
        targets: np.ndarray | None,
        alpha: float | None = None,
    ) -> str:
        """Compute ``W`` with the closed-form ridge solution.

        Args:
            features: ``Phi``, of shape ``(n_samples, n_features)``, or
                ``None`` when no dataset has been built yet.
            targets: ``y``, of shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``, or ``None``.
            alpha: Regularization strength. ``None`` uses ``config_alpha``.

        Returns:
            ``"fitted"``, or ``"scaffold"`` when ``features`` or ``targets``
            is ``None`` (the pipeline does not build a dataset yet).

        Raises:
            ValueError: If the shapes do not match, or ``alpha`` is invalid.
        """
        if features is None or targets is None:
            return "scaffold"
        lam = self.config_alpha if alpha is None else _check_alpha(alpha)
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
        A = phi.T @ phi + lam * np.eye(phi.shape[1])
        self.W = np.linalg.solve(A, phi.T @ y)
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


def _check_alpha(alpha: Any) -> float:
    """Return ``alpha`` as a float after checking it is a number >= 0."""
    if isinstance(alpha, bool):
        raise ValueError(f"alpha must be a non-negative number, got {alpha!r}.")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"alpha must be a non-negative number, got {alpha!r}."
        ) from exc
    if not value >= 0:
        raise ValueError(f"alpha must be a non-negative number, got {alpha!r}.")
    return value
