"""Angle encoding of a static input: one single-qubit rotation per feature."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp

from QELM.embedding.base import Embedding

AXES = ("x", "y")


class RotationEmbedding(Embedding):
    """Product state ``R_a(x_0) |0> ⊗ ... ⊗ R_a(x_{N-1}) |0>`` (issue #181).

    Each component ``x_i`` of the input (an angle, in radians) rotates qubit
    ``i`` about the axis ``a``, with ``R_a(theta) = exp(-i theta sigma_a / 2)``:

    - ``axis = "y"``: ``R_y(x)|0> = cos(x/2)|0> + sin(x/2)|1>``, Bloch vector
      ``(sin x, 0, cos x)``, a real state;
    - ``axis = "x"``: ``R_x(x)|0> = cos(x/2)|0> - i sin(x/2)|1>``, Bloch
      vector ``(0, -sin x, cos x)``.

    The input has exactly ``N`` components, one per qubit, so ``x`` is
    reduced to ``N`` dimensions first (PCA, :mod:`QELM.datasets`). There is
    no ancilla (``Nr = 0``). Config block: ``rotation_embedding: {N, axis}``.

    Attributes:
        axis: Rotation axis, ``"x"`` or ``"y"``.
    """

    name: str = "rotation_embedding"

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: ``N`` (number of qubits) and the optional ``axis``
                (default ``"y"``).

        Raises:
            ValueError: If ``axis`` is not ``"x"`` or ``"y"``.
        """
        super().__init__(params)
        self.axis: str = str(params.get("axis", "y")).lower()
        if self.axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {self.axis!r}.")

    def qubit_states(self, x: np.ndarray) -> np.ndarray:
        """Return the single-qubit states ``R_a(x_i)|0>``.

        Args:
            x: Angles of shape ``(N,)`` or ``(n_samples, N)``.

        Returns:
            Complex array of shape ``x.shape + (2,)``.

        Raises:
            ValueError: If the last axis of ``x`` is not ``N``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.N:
            raise ValueError(
                f"x must have shape ({self.N},) or (n_samples, {self.N}), "
                f"got {x.shape}."
            )
        down = np.sin(x / 2) if self.axis == "y" else -1j * np.sin(x / 2)
        return np.stack([np.cos(x / 2) + 0j, down + 0j], axis=-1)

    def states(self, x: np.ndarray) -> np.ndarray:
        """Return the embedded state of every sample of a batch.

        Site 0 is the leftmost factor of the Kronecker product, as in
        :func:`QELM.physics.operators.get_site_operator`.

        Args:
            x: Angles of shape ``(n_samples, N)``.

        Returns:
            Normalized states of shape ``(n_samples, 2**N)``.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        qubits = self.qubit_states(x)
        psi = np.ones((x.shape[0], 1), dtype=complex)
        for i in range(self.N):
            psi = np.einsum("na,nb->nab", psi, qubits[:, i]).reshape(x.shape[0], -1)
        return psi

    def bloch_vectors(self, x: np.ndarray) -> np.ndarray:
        """Return ``(<X_i>, <Y_i>, <Z_i>)`` of every qubit, in closed form.

        Args:
            x: Angles of shape ``(N,)`` or ``(n_samples, N)``.

        Returns:
            Real array of shape ``x.shape + (3,)``.
        """
        x = np.asarray(x, dtype=np.float64)
        zero = np.zeros_like(x)
        if self.axis == "y":
            return np.stack([np.sin(x), zero, np.cos(x)], axis=-1)
        return np.stack([zero, -np.sin(x), np.cos(x)], axis=-1)

    @overload
    def state(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...

    @overload
    def state(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...

    def state(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return the embedded state of one input.

        Args:
            x: Angles of shape ``(N,)``. Required.
            sparse: ``True`` for a CSR column of shape ``(dim, 1)``.

        Returns:
            Normalized complex state of shape ``(dim,)``, or ``(dim, 1)``
            if ``sparse``.

        Raises:
            ValueError: If ``x`` is ``None`` or does not have shape ``(N,)``.
        """
        if x is None:
            raise ValueError("RotationEmbedding.state needs the input x.")
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.N,):
            raise ValueError(f"x must have shape ({self.N},), got {x.shape}.")
        psi = self.states(x[None, :])[0]
        if sparse:
            return sp.csr_matrix(psi.reshape(-1, 1))
        return psi
