"""Base class for the ways of encoding an input into the reservoir state.

An encoding prepares the initial state of the reservoir: a pure state
``psi`` of shape ``(dim,)`` and its density matrix ``rho = |psi><psi|`` of
shape ``(dim, dim)``. ``N`` is the total number of qubits, so
``dim = 2**N`` matches the models built with the same ``N``. Some encodings
use ``Nr`` of them as ancillas, which leaves ``Ns = N - Nr``
system qubits.

Qubit ordering follows :func:`QELM.physics.operators.get_site_operator`:
site 0 is the leftmost factor of ``np.kron``, i.e. the most significant bit
of a basis index. Ancillas come first (sites ``0 .. Nr - 1``), the
``Ns`` system qubits after them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self, overload

import numpy as np
import scipy.sparse as sp
from omegaconf import DictConfig, OmegaConf


class Encoding:
    """Base class for state encodings configured from a Hydra block.

    Attributes:
        name: Config key for this encoding; ``from_config`` reads
            ``cfg[name]``.
        Nr: Number of ancilla qubits placed before the system.
        params: Scalar parameters from the matching YAML block.
        N: Total number of qubits, ancillas included.
        Ns: Number of system qubits, ``N - Nr``.
        dim: Dimension of the encoded state, ``2**N``.
    """

    name: str = "base"
    Nr: int = 0

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
                ``N``, the total number of qubits, is required.

        Raises:
            ValueError: If ``N`` leaves no system qubit.
        """
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.Ns: int = self.N - self.Nr
        if self.Ns < 1:
            raise ValueError(
                f"N={self.N} leaves no system qubit next to {self.Nr} ancilla(s)."
            )
        self.dim: int = 2**self.N

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the encoding from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this class.

        Raises:
            ValueError: If ``cfg`` has no ``cls.name`` block.
            TypeError: If the config block is not a mapping.
        """
        node = OmegaConf.select(cfg, cls.name)
        if node is None:
            raise ValueError(
                f"The config has no {cls.name!r} block. Compose the full config "
                f"with QELM.load_config(), or add `{cls.name}: {{N: ...}}`."
            )
        raw = (
            OmegaConf.to_container(node, resolve=True)
            if OmegaConf.is_config(node)
            else node
        )
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        return cls({str(key): value for key, value in raw.items()})

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
        """Return the encoded pure state.

        Args:
            x: Input data to encode. ``None`` for an encoding that does not
                depend on the input.
            sparse: ``True`` for a CSR column, ``False`` (default) for a
                NumPy vector.

        Returns:
            Normalized complex state: a CSR column of shape ``(dim, 1)`` if
            ``sparse``, otherwise a vector of shape ``(dim,)``.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} state is not implemented.")

    @overload
    def density_matrix(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...
    @overload
    def density_matrix(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...
    def density_matrix(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return ``rho = |psi><psi|`` for ``psi = self.state(x)``.

        Args:
            x: Input data, passed to :meth:`state`.
            sparse: ``True`` for a CSR matrix built from the sparse state
                (``k**2`` stored entries for ``k`` non-zero amplitudes),
                ``False`` (default) for a NumPy array.

        Returns:
            Complex density matrix of shape ``(dim, dim)``.
        """
        if sparse:
            psi_sp = self.state(x, sparse=True)
            return sp.csr_matrix(psi_sp @ psi_sp.conj().T)
        psi = self.state(x)
        return np.outer(psi, psi.conj())

    @overload
    def rho(
        self, x: np.ndarray | None = None, sparse: Literal[False] = False
    ) -> np.ndarray: ...
    @overload
    def rho(
        self, x: np.ndarray | None = None, *, sparse: Literal[True]
    ) -> sp.csr_matrix: ...
    def rho(
        self, x: np.ndarray | None = None, sparse: bool = False
    ) -> np.ndarray | sp.csr_matrix:
        """Return the initial density matrix ``rho = |psi><psi|``.

        Short name for :meth:`density_matrix`, inherited by every encoding.

        Args:
            x: Input data, passed to :meth:`state`.
            sparse: ``True`` for a CSR matrix, ``False`` (default) for a
                NumPy array.

        Returns:
            Complex density matrix of shape ``(2**N, 2**N)``.
        """
        if sparse:
            return self.density_matrix(x, sparse=True)
        return self.density_matrix(x)


class OutOfBoundsError(ValueError):
    """An input feature lies outside the interval its encoding expects."""


class ProductEncoding(Encoding):
    """Product state of ``N`` qubits, each encoding one or two features.

    The tensor-product encodings of "Harnessing quantum extreme learning
    machines for image classification", Phys. Rev. Applied (2025) (issue
    #196). The input ``x = (x_1, ..., x_M)`` is cut into groups of
    ``features_per_qubit`` consecutive features, group ``i`` setting the
    state ``a_i |0> + b_i |1>`` of qubit ``i``:

        |x> = (a_0 |0> + b_0 |1>) ⊗ ... ⊗ (a_{N-1} |0> + b_{N-1} |1>),

    so ``N = ceil(M / features_per_qubit)``. When ``M`` is odd and the
    encoding takes two features per qubit, the last group is completed
    with a 0. Feature ``k`` of a group must lie in ``bounds[k]``: an input
    outside raises :class:`OutOfBoundsError`, so a badly normalized input
    cannot go unnoticed. No ancilla (``Nr = 0``).

    Subclasses set ``name``, ``features_per_qubit``, ``bounds`` and
    implement :meth:`amplitudes`. Config block: ``<name>: {N, n_features}``.

    Attributes:
        features_per_qubit: Number of features encoded in one qubit.
        bounds: Interval ``(low, high)`` of each feature of a group.
        n_features: Number of input features ``M``.
    """

    features_per_qubit: int = 1
    bounds: tuple[tuple[float, float], ...] = ((0.0, np.pi),)
    # Rounding slack of the bounds check: a feature computed as pi * u can
    # exceed pi by one ulp.
    atol: float = 1e-12

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store the parameters.

        Args:
            params: ``N`` (number of qubits) and the optional ``n_features``
                (default ``N * features_per_qubit``).

        Raises:
            ValueError: If ``n_features`` does not need exactly ``N`` qubits.
        """
        super().__init__(params)
        k = self.features_per_qubit
        self.n_features: int = int(params.get("n_features", self.N * k))
        if -(-self.n_features // k) != self.N:
            raise ValueError(
                f"{self.name}: {self.n_features} features need "
                f"ceil({self.n_features} / {k}) qubits, got N={self.N}."
            )

    def feature_bounds(self) -> np.ndarray:
        """Return the interval of each input feature, of shape ``(M, 2)``."""
        groups = np.tile(np.array(self.bounds, dtype=np.float64), (self.N, 1))
        return groups[: self.n_features]

    def check_bounds(self, x: np.ndarray) -> None:
        """Raise if a feature of ``x`` lies outside its interval.

        Args:
            x: Features of shape ``(..., M)``.

        Raises:
            OutOfBoundsError: If a feature is outside ``bounds``.
        """
        b = self.feature_bounds()
        low, high = b[:, 0], b[:, 1]
        outside = (x < low - self.atol) | (x > high + self.atol)
        if np.any(outside):
            k = int(np.flatnonzero(outside.reshape(-1, self.n_features).any(axis=0))[0])
            raise OutOfBoundsError(
                f"{self.name}: feature {k} must lie in [{low[k]:g}, {high[k]:g}], "
                f"got values in [{x[..., k].min():g}, {x[..., k].max():g}]."
            )

    def amplitudes(self, groups: np.ndarray) -> np.ndarray:
        """Return ``(a_i, b_i)`` of every qubit from its group of features.

        Args:
            groups: Features of shape ``(n_samples, N, features_per_qubit)``,
                already checked against ``bounds``.

        Returns:
            Complex array of shape ``(n_samples, N, 2)``, each pair
            normalized.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"{self.name} amplitudes are not implemented.")

    def qubit_states(self, x: np.ndarray) -> np.ndarray:
        """Return the single-qubit states ``a_i |0> + b_i |1>``.

        Args:
            x: Features of shape ``(M,)`` or ``(n_samples, M)``.

        Returns:
            Complex array of shape ``(N, 2)`` or ``(n_samples, N, 2)``.

        Raises:
            ValueError: If the last axis of ``x`` is not ``M``.
            OutOfBoundsError: If a feature is outside ``bounds``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.n_features:
            raise ValueError(
                f"x must have shape ({self.n_features},) or "
                f"(n_samples, {self.n_features}), got {x.shape}."
            )
        self.check_bounds(x)
        batch = np.atleast_2d(x)
        padded = np.zeros((batch.shape[0], self.N * self.features_per_qubit))
        padded[:, : self.n_features] = batch
        groups = padded.reshape(batch.shape[0], self.N, self.features_per_qubit)
        qubits = self.amplitudes(groups).astype(complex)
        return qubits[0] if x.ndim == 1 else qubits

    def states(self, x: np.ndarray) -> np.ndarray:
        """Return the encoded state of every sample of a batch.

        Site 0 is the leftmost factor of the Kronecker product, as in
        :func:`QELM.physics.operators.get_site_operator`.

        Args:
            x: Features of shape ``(n_samples, M)``.

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

        For ``a |0> + b |1>``: ``<X> = 2 Re(a* b)``, ``<Y> = 2 Im(a* b)``,
        ``<Z> = |a|^2 - |b|^2``.

        Args:
            x: Features of shape ``(M,)`` or ``(n_samples, M)``.

        Returns:
            Real array of shape ``(N, 3)`` or ``(n_samples, N, 3)``.
        """
        q = self.qubit_states(x)
        a, b = q[..., 0], q[..., 1]
        cross = np.conj(a) * b
        z = np.abs(a) ** 2 - np.abs(b) ** 2
        return np.stack([2 * cross.real, 2 * cross.imag, z], axis=-1)

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
        """Return the encoded state of one input.

        Args:
            x: Features of shape ``(M,)``. Required.
            sparse: ``True`` for a CSR column of shape ``(dim, 1)``.

        Returns:
            Normalized complex state of shape ``(dim,)``, or ``(dim, 1)``
            if ``sparse``.

        Raises:
            ValueError: If ``x`` is ``None`` or does not have shape ``(M,)``.
        """
        if x is None:
            raise ValueError(f"{type(self).__name__}.state needs the input x.")
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_features,):
            raise ValueError(f"x must have shape ({self.n_features},), got {x.shape}.")
        psi = self.states(x[None, :])[0]
        if sparse:
            return sp.csr_matrix(psi.reshape(-1, 1))
        return psi
