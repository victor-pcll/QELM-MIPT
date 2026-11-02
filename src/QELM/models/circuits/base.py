"""Shared pieces of the random brickwork circuits (issue #189).

The gate and layer types, the application of a two-qubit gate to a state
vector or a density matrix, the random generator helper, and
:class:`BrickworkCircuit`, the base class that holds the geometry of the
brickwork and draws its layers. The kind of gate is set by the subclasses:
:mod:`QELM.models.circuits.haar` (U(1) and Haar gates) and
:mod:`QELM.models.circuits.clifford` (Clifford gates). See
:mod:`QELM.models.circuits` for the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from omegaconf import DictConfig, OmegaConf

# A layer is a list of gates, each gate a pair of qubits and a 4 x 4 unitary.
Gate = tuple[int, int, np.ndarray]
Layer = list[Gate]


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    """Return ``rng``, or a generator drawn from the global NumPy state.

    The global state is seeded by :func:`QELM.seed.set_seed`, so the result
    is reproducible for a given ``seed`` even when no ``rng`` is passed.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(np.random.randint(0, 2**31 - 1))


def _apply_to_rows(M: np.ndarray, gate: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return ``G_ij @ M`` for ``M`` of shape ``(2**N, k)``, without building ``G``.

    The row index of ``M`` is reshaped into ``N`` qubit axes; the axes of
    qubits ``i`` and ``j`` are contracted with ``gate``, ``i`` being the most
    significant bit of the gate's two-qubit index.
    """
    dim, k = M.shape
    N = dim.bit_length() - 1
    T = np.moveaxis(M.reshape((2,) * N + (k,)), (i, j), (0, 1))
    shape = T.shape
    T = (gate @ T.reshape(4, -1)).reshape(shape)
    return np.moveaxis(T, (0, 1), (i, j)).reshape(dim, k)


def apply_two_qubit_gate(
    state: np.ndarray, gate: np.ndarray, i: int, j: int
) -> np.ndarray:
    """Apply a two-qubit gate on qubits ``i`` and ``j`` of a state.

    Args:
        state: State vector of shape ``(2**N,)`` or density matrix of shape
            ``(2**N, 2**N)``.
        gate: ``4 x 4`` unitary in the basis ``|b_i b_j>``.
        i: First qubit, the most significant bit of the gate index.
        j: Second qubit, different from ``i``.

    Returns:
        ``G psi`` for a vector, ``G rho G^dagger`` for a density matrix, with
        the shape of ``state``.

    Raises:
        ValueError: If the shapes do not match, or ``i`` and ``j`` are equal
            or out of range.
    """
    state = np.asarray(state, dtype=complex)
    dim = state.shape[0]
    N = dim.bit_length() - 1
    if dim < 4 or 2**N != dim or state.shape not in ((dim,), (dim, dim)):
        raise ValueError(
            "state must be a vector (2**N,) or a density matrix (2**N, 2**N) "
            f"with N >= 2, got shape {state.shape}."
        )
    if gate.shape != (4, 4):
        raise ValueError(f"gate must have shape (4, 4), got {gate.shape}.")
    if i == j or not (0 <= i < N and 0 <= j < N):
        raise ValueError(f"Need two distinct qubits in [0, {N - 1}], got {i}, {j}.")
    if state.ndim == 1:
        return _apply_to_rows(state[:, None], gate, i, j)[:, 0]
    # A = G rho, then (G A^dagger)^dagger = A G^dagger = G rho G^dagger.
    A = _apply_to_rows(state, gate, i, j)
    return np.asarray(_apply_to_rows(A.conj().T, gate, i, j).conj().T)


def apply_layer(state: np.ndarray, layer: Layer) -> np.ndarray:
    """Apply every gate of ``layer`` to ``state``, in order."""
    for i, j, gate in layer:
        state = apply_two_qubit_gate(state, gate, i, j)
    return state


class BrickworkCircuit:
    """Brickwork of random two-qubit gates on the system qubits.

    The geometry and the bookkeeping shared by every circuit: which pairs
    get a gate, the scrambling layers, drawing and applying the layers. The
    kind of gate is left to the subclasses, through :meth:`draw_gate`:
    :class:`~QELM.models.circuits.haar.U1BrickworkCircuit`,
    :class:`~QELM.models.circuits.haar.HaarBrickworkCircuit` and
    :class:`~QELM.models.circuits.clifford.CliffordBrickworkCircuit`.

    One instance holds the gates of ``depth`` brickwork layers and of
    ``scrambling_layers`` all-to-all scrambling layers, drawn once. The
    paper keeps one instance of the gates for every trajectory
    (``fixed_instance``); call :meth:`redraw` to draw a new instance, for
    an average over circuits.

    Attributes:
        name: Config key of the subclass, read by :meth:`from_config`.
        N: Number of system qubits, the ones the gates act on.
        Nr: Number of ancilla qubits at the start of the chain, untouched.
        N_total: Number of qubits of the state, ``N + Nr``.
        depth: Number of brickwork layers, each a full period (even then
            odd half-layer).
        scrambling_layers: Number of all-to-all scrambling layers.
        boundary: ``"open"`` or ``"periodic"``.
        fixed_instance: ``True`` to keep the same gates for every
            trajectory, read by the experiment script.
        layers: The ``depth`` brickwork layers.
        scrambling: The scrambling layers.
    """

    name: str = "brickwork_circuit"
    # Layers in one period of the brickwork (every bond once): 1 here, where
    # a layer is a full period; 2 for FloquetBrickworkCircuit, whose layers
    # are half-layers.
    layers_per_period: int = 1

    def __init__(
        self, params: Mapping[str, Any], rng: np.random.Generator | None = None
    ) -> None:
        """Store the parameters and draw the gates.

        Args:
            params: ``N`` (number of system qubits) and ``depth`` are
                required; ``Nr`` (default 1),
                ``scrambling_layers`` (default 0), ``boundary`` (default
                ``"open"``) and ``fixed_instance`` (default ``True``) are
                optional.
            rng: Random generator for the gates. ``None`` draws one from the
                global NumPy state.

        Raises:
            ValueError: If a count is negative, ``N < 2`` or ``boundary`` is
                unknown.
        """
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.Nr: int = int(params.get("Nr", 1))
        self.N_total: int = self.N + self.Nr
        self.depth: int = int(params["depth"])
        self.scrambling_layers: int = int(params.get("scrambling_layers", 0))
        self.boundary: str = str(params.get("boundary", "open"))
        self.fixed_instance: bool = bool(params.get("fixed_instance", True))
        if self.Nr < 0 or self.depth < 0 or self.scrambling_layers < 0:
            raise ValueError(
                "Nr, depth and scrambling_layers must be >= 0, got "
                f"{self.Nr}, {self.depth}, {self.scrambling_layers}."
            )
        if self.N < 2:
            raise ValueError(f"Need at least 2 system qubits, got N={self.N}.")
        if self.boundary not in ("open", "periodic"):
            raise ValueError(
                f"boundary must be 'open' or 'periodic', got {self.boundary!r}."
            )
        self.layers: list[Layer] = []
        self.scrambling: list[Layer] = []
        self.redraw(rng)

    @classmethod
    def from_config(
        cls, cfg: DictConfig, rng: np.random.Generator | None = None
    ) -> Self:
        """Build the circuit from the ``cls.name`` block of ``cfg``.

        Args:
            cfg: Composed config with a ``cls.name`` mapping (for example
                ``U1_circuit``, ``haar_circuit`` or ``clifford_circuit``).
            rng: Random generator for the gates, as in ``__init__``.

        Returns:
            A circuit with freshly drawn gates.

        Raises:
            TypeError: If the block is not a mapping.
        """
        node = cfg[cls.name]
        raw = (
            OmegaConf.to_container(node, resolve=True)
            if OmegaConf.is_config(node)
            else node
        )
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        return cls({str(key): value for key, value in raw.items()}, rng)

    def bonds(self, half: int) -> list[tuple[int, int]]:
        """Return the qubit pairs of one half-layer of the brickwork.

        Args:
            half: 0 (even) for the half-layer starting on the first system
                qubit, 1 (odd) for the one starting on the second; only the
                parity counts.

        Returns:
            Disjoint pairs ``(i, i + 1)`` of system qubits, plus the
            wrap-around pair for the odd half-layer with periodic boundaries
            and an even ``N``.
        """
        sites = list(range(self.Nr, self.N_total))
        start = half % 2
        pairs = [(sites[k], sites[k + 1]) for k in range(start, self.N - 1, 2)]
        if self.boundary == "periodic" and start == 1 and self.N % 2 == 0:
            pairs.append((sites[-1], sites[0]))
        return pairs

    def scrambling_bonds(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Pair the system qubits at random, all to all.

        Args:
            rng: Random generator for the shuffle.

        Returns:
            ``N // 2`` disjoint pairs; one qubit is left out when ``N`` is
            odd.
        """
        order = rng.permutation(np.arange(self.Nr, self.N_total))
        return [(int(order[k]), int(order[k + 1])) for k in range(0, self.N - 1, 2)]

    def redraw(self, rng: np.random.Generator | None = None) -> None:
        """Draw new gates for every layer, scrambling layers first.

        Args:
            rng: Random generator. ``None`` draws one from the global NumPy
                state.
        """
        gen = _generator(rng)
        self.scrambling = [
            [(i, j, self.draw_gate(gen)) for i, j in self.scrambling_bonds(gen)]
            for _ in range(self.scrambling_layers)
        ]
        # One layer = the even half-layer, then the odd one.
        self.layers = [
            [(i, j, self.draw_gate(gen)) for i, j in self.bonds(0) + self.bonds(1)]
            for _ in range(self.depth)
        ]

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one ``4 x 4`` gate of the circuit.

        Subclasses override it to set the kind of gate and keep the
        brickwork.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement draw_gate.")

    def scramble(self, state: np.ndarray) -> np.ndarray:
        """Apply the scrambling layers to ``state`` (vector or density matrix)."""
        for layer in self.scrambling:
            state = apply_layer(state, layer)
        return state

    def apply_layer(self, state: np.ndarray, t: int) -> np.ndarray:
        """Apply brickwork layer ``t`` to ``state`` (vector or density matrix)."""
        return apply_layer(state, self.layers[t])
