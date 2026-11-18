"""Analytical ridge regression readout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from QELM.training.regression.base import Trainer

# Same value as conf/training/regression/ridge.yaml. alpha=1.0 also shrinks the
# bias weight and costs R2 (issue #92).
DEFAULT_ALPHA = 1e-3

# Conventions of the penalty scale (issue #372), see ridge_penalty.
PENALTIES = ("sum", "mean", "relative")


def ridge_penalty(phi: np.ndarray, alpha: float, penalty: str = "sum") -> float:
    """Return the absolute penalty ``lam`` of ``(Phi^T Phi + lam I) W = Phi^T y``.

    The three conventions of the repository (issue #372), for ``Phi`` of
    shape ``(S, d)``:

    - ``"sum"``: ``lam = alpha``, the penalty on the sum of squares (the
      historical convention of :class:`RidgeReadout`);
    - ``"mean"``: ``lam = alpha S``, the penalty on the mean Gram,
      ``(Phi^T Phi / S + alpha I)``, the population ridge of
      :mod:`QELM.evaluation.two_filters`, :mod:`QELM.evaluation.fourier_framework`
      and :mod:`QELM.evaluation.clifford_ridge`;
    - ``"relative"``: ``lam = alpha tr(Phi^T Phi) / d``, ``alpha`` in units of
      the mean eigenvalue of the Gram matrix, invariant to the number of
      samples and to a rescaling of the features (the ``lambda`` of
      :func:`QELM.evaluation.ridge_spectrum.ridge_sweep`).

    Pass the centred ``Phi`` when the intercept is fitted apart.

    Args:
        phi: Features ``(S, d)``, as they enter the Gram matrix.
        alpha: Non-negative strength, in the units of ``penalty``.
        penalty: One of :data:`PENALTIES`.

    Returns:
        The absolute penalty, ``>= 0``.

    Raises:
        ValueError: If ``penalty`` is unknown or ``alpha`` is invalid.
    """
    alpha = _check_alpha(alpha)
    penalty = _check_penalty(penalty)
    phi = np.asarray(phi, dtype=np.float64)
    if penalty == "sum":
        return alpha
    if penalty == "mean":
        return alpha * phi.shape[0]
    d = phi.shape[1] if phi.ndim == 2 else 0
    if d == 0:
        return 0.0
    return alpha * float(np.sum(phi**2)) / d


class RidgeReadout(Trainer):
    """Ridge regression on the reservoir features, solved in closed form.

    ``W = (Phi^T Phi + alpha I)^{-1} Phi^T y``, computed by solving the linear
    system ``(Phi^T Phi + alpha I) W = Phi^T y`` with ``numpy.linalg.solve``
    rather than inverting the matrix. No intercept is fitted: add a constant
    column to ``Phi`` if a bias is needed. Selected with ``training/regression=ridge``.

    ``alpha = 0`` is allowed and gives the limit ``alpha -> 0`` of that
    solution, ``W = Phi^+ y`` with ``Phi^+`` the Moore-Penrose pseudo-inverse
    (``numpy.linalg.lstsq``): ordinary least squares when ``Phi`` has full
    column rank, and below ``n_samples = rank(Phi)`` the interpolating
    solution (``Phi W = y``) of smallest norm. This is the unregularized
    readout of the double-descent experiment (issue #181), whose test error
    peaks at ``n_samples = rank(Phi)``.

    ``alpha`` is resolved in this order: the argument of :meth:`fit`, then
    ``alpha`` in the ``training`` config block, then 1e-3 (the value of
    ``conf/training/regression/ridge.yaml``, issue #92).

    Two options set the scale of ``alpha`` and the intercept (issue #372);
    their defaults are the historical behaviour:

    - ``penalty``: ``"sum"`` (default), ``"mean"`` or ``"relative"``, the
      conventions of :func:`ridge_penalty`. The system solved is
      ``(Phi^T Phi + lam I) W = Phi^T y`` with ``lam = ridge_penalty(Phi,
      alpha, penalty)``;
    - ``fit_intercept``: ``False`` (default) or ``True``. With ``True``,
      ``Phi`` and ``y`` are centred on their training means, the ridge is
      solved on the centred data (``lam`` computed from the centred ``Phi``)
      and the intercept ``b = mean(y) - mean(Phi) W`` is not penalized.

    Attributes:
        config_alpha: ``alpha`` from the config, or 1e-3.
        penalty: Scale of ``alpha``, one of :data:`PENALTIES`.
        fit_intercept: Whether an unpenalized intercept is fitted.
        W: Fitted weights of shape ``(n_features,)`` or
            ``(n_features, n_outputs)``, or ``None`` before :meth:`fit`.
        b: Fitted intercept (0 when ``fit_intercept`` is ``False``), a float
            or an array of shape ``(n_outputs,)``.
    """

    name = "ridge"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Store the parameters and read ``alpha``.

        Args:
            params: Optional ``alpha``, a non-negative regularization
                strength (default 1e-3), ``penalty`` (default ``"sum"``) and
                ``fit_intercept`` (default ``False``).

        Raises:
            ValueError: If ``alpha`` is negative or not a number, or
                ``penalty`` is unknown.
        """
        super().__init__(params)
        self.config_alpha: float = _check_alpha(self.params.get("alpha", DEFAULT_ALPHA))
        self.penalty: str = _check_penalty(self.params.get("penalty", "sum"))
        self.fit_intercept: bool = bool(self.params.get("fit_intercept", False))
        self.W: np.ndarray | None = None
        self.b: float | np.ndarray = 0.0

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
        if self.fit_intercept:
            phi_mean, y_mean = phi.mean(axis=0), y.mean(axis=0)
            phi, y = phi - phi_mean, y - y_mean
        lam = ridge_penalty(phi, lam, self.penalty)
        if lam == 0.0:
            # No penalty: the limit alpha -> 0 of the ridge solution, the
            # least-squares fit of smallest norm (issue #181). It exists for
            # any shape of Phi, where Phi^T Phi is singular as soon as
            # n_samples < n_features.
            self.W = np.linalg.lstsq(phi, y, rcond=None)[0]
        else:
            A = phi.T @ phi + lam * np.eye(phi.shape[1])
            self.W = np.linalg.solve(A, phi.T @ y)
        self.b = y_mean - phi_mean @ self.W if self.fit_intercept else 0.0
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
        return np.asarray(features, dtype=np.float64) @ self.W + self.b

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return ``W`` under ``weights``, or a length-0 array before fit.

        With ``fit_intercept``, the intercept is stored under ``intercept``.
        """
        if self.W is None:
            return super().state_dict()
        if self.fit_intercept:
            return {"weights": self.W, "intercept": np.asarray(self.b)}
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


def _check_penalty(penalty: Any) -> str:
    """Return ``penalty`` after checking it is one of :data:`PENALTIES`."""
    if penalty not in PENALTIES:
        raise ValueError(f"penalty must be one of {PENALTIES}, got {penalty!r}.")
    return str(penalty)
