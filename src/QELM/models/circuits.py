"""Random circuits of charge-conserving (U(1)) two-qubit gates.

This is the circuit model of Agrawal et al., PRX 14, 041012 (2024), used for
their charge-sharpening benchmark (issue #137). Each gate acts on two qubits
in the basis ``|00>, |01>, |10>, |11>`` as::

    u = e^{i phi_0}  (+)  U_2  (+)  e^{i phi_3}

a phase on ``|00>``, a Haar-random ``2 x 2`` unitary ``U_2`` on the
one-particle block ``{|01>, |10>}`` and a phase on ``|11>``. The gate never
changes the number of 1 bits, so the charge ``Q = sum_i b_i`` of the system
is conserved. The paper parametrizes ``U_2`` by angles ``phi_{0..3}`` and
``theta`` drawn with the measure ``sin(theta)``, which is the Haar measure
on ``U(2)``; here ``U_2`` is drawn directly from the Haar measure.

Gates act on the ``Ns`` system qubits ``Nr .. N-1`` only, never on the
``Nr`` ancillas at the start of the chain. Two kinds of layers are used:

- brickwork layers: each layer is one full period of the brickwork, an even
  half-layer on the pairs ``(s_0, s_1), (s_2, s_3), ...`` followed by an odd
  half-layer on ``(s_1, s_2), (s_3, s_4), ...``, so that every bond gets a
  gate once per layer. Boundaries are open by default
  (``boundary="periodic"`` adds the pair ``(s_{Ns-1}, s_0)`` to the odd
  half-layer when ``Ns`` is even). Taking a layer as a full period is what
  reproduces the paper's Fig. 7 (issue #137);
- scrambling layers: the system qubits are shuffled and paired at random,
  all to all, which the paper applies 5 times to the product initial state
  to spread the charge before the monitored dynamics.

Qubit 0 is the leftmost factor of the tensor product, as in
:func:`QELM.physics.operators.get_site_operator`.
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


def haar_unitary(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw an ``n x n`` unitary from the Haar measure.

    QR decomposition of a complex Gaussian matrix, with the phases of the
    diagonal of ``R`` moved into ``Q`` so that the distribution is exactly
    Haar (Mezzadri, Notices AMS 54, 592 (2007)).

    Args:
        n: Matrix size, at least 1.
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex unitary matrix of shape ``(n, n)``.
    """
    gen = _generator(rng)
    z = (gen.standard_normal((n, n)) + 1j * gen.standard_normal((n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return np.asarray(q * (d / np.abs(d)))


def random_u1_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a random charge-conserving two-qubit gate.

    Args:
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex ``4 x 4`` unitary in the basis ``|00>, |01>, |10>, |11>``:
        uniform phases on ``|00>`` and ``|11>`` and a Haar ``2 x 2`` block
        on ``{|01>, |10>}``.
    """
    gen = _generator(rng)
    phi_0, phi_3 = gen.uniform(0.0, 2 * np.pi, size=2)
    u = np.zeros((4, 4), dtype=complex)
    u[0, 0] = np.exp(1j * phi_0)
    u[1:3, 1:3] = haar_unitary(2, gen)
    u[3, 3] = np.exp(1j * phi_3)
    return u


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


class U1BrickworkCircuit:
    """Brickwork circuit of random U(1) gates on the system qubits.

    One instance holds the gates of ``depth`` brickwork layers and of
    ``scrambling_layers`` all-to-all scrambling layers, drawn once. The
    paper keeps one instance of the gates for every trajectory
    (``fixed_instance``); call :meth:`redraw` to draw a new instance, for
    an average over circuits.

    Attributes:
        name: Config key, ``"U1_circuit"``.
        N: Total number of qubits, ancillas included.
        Nr: Number of ancilla qubits at the start of the chain, untouched.
        Ns: Number of system qubits, ``N - Nr``.
        depth: Number of brickwork layers, each a full period (even then
            odd half-layer).
        scrambling_layers: Number of all-to-all scrambling layers.
        boundary: ``"open"`` or ``"periodic"``.
        fixed_instance: ``True`` to keep the same gates for every
            trajectory, read by the experiment script.
        layers: The ``depth`` brickwork layers.
        scrambling: The scrambling layers.
    """

    name: str = "U1_circuit"

    def __init__(
        self, params: Mapping[str, Any], rng: np.random.Generator | None = None
    ) -> None:
        """Store the parameters and draw the gates.

        Args:
            params: ``N`` and ``depth`` are required; ``Nr`` (default 1),
                ``scrambling_layers`` (default 0), ``boundary`` (default
                ``"open"``) and ``fixed_instance`` (default ``True``) are
                optional.
            rng: Random generator for the gates. ``None`` draws one from the
                global NumPy state.

        Raises:
            ValueError: If a count is negative, ``Ns < 2`` or ``boundary`` is
                unknown.
        """
        self.params: dict[str, Any] = dict(params)
        self.N: int = int(params["N"])
        self.Nr: int = int(params.get("Nr", 1))
        self.Ns: int = self.N - self.Nr
        self.depth: int = int(params["depth"])
        self.scrambling_layers: int = int(params.get("scrambling_layers", 0))
        self.boundary: str = str(params.get("boundary", "open"))
        self.fixed_instance: bool = bool(params.get("fixed_instance", True))
        if self.Nr < 0 or self.depth < 0 or self.scrambling_layers < 0:
            raise ValueError(
                "Nr, depth and scrambling_layers must be >= 0, got "
                f"{self.Nr}, {self.depth}, {self.scrambling_layers}."
            )
        if self.Ns < 2:
            raise ValueError(f"Need at least 2 system qubits, got Ns={self.Ns}.")
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
        """Build the circuit from the ``U1_circuit`` block of ``cfg``.

        Args:
            cfg: Composed config with a ``U1_circuit`` mapping.
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
            and an even ``Ns``.
        """
        sites = list(range(self.Nr, self.N))
        start = half % 2
        pairs = [(sites[k], sites[k + 1]) for k in range(start, self.Ns - 1, 2)]
        if self.boundary == "periodic" and start == 1 and self.Ns % 2 == 0:
            pairs.append((sites[-1], sites[0]))
        return pairs

    def scrambling_bonds(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Pair the system qubits at random, all to all.

        Args:
            rng: Random generator for the shuffle.

        Returns:
            ``Ns // 2`` disjoint pairs; one qubit is left out when ``Ns`` is
            odd.
        """
        order = rng.permutation(np.arange(self.Nr, self.N))
        return [(int(order[k]), int(order[k + 1])) for k in range(0, self.Ns - 1, 2)]

    def redraw(self, rng: np.random.Generator | None = None) -> None:
        """Draw new gates for every layer, scrambling layers first.

        Args:
            rng: Random generator. ``None`` draws one from the global NumPy
                state.
        """
        gen = _generator(rng)
        self.scrambling = [
            [(i, j, random_u1_gate(gen)) for i, j in self.scrambling_bonds(gen)]
            for _ in range(self.scrambling_layers)
        ]
        # One layer = the even half-layer, then the odd one.
        self.layers = [
            [(i, j, random_u1_gate(gen)) for i, j in self.bonds(0) + self.bonds(1)]
            for _ in range(self.depth)
        ]

    def scramble(self, state: np.ndarray) -> np.ndarray:
        """Apply the scrambling layers to ``state`` (vector or density matrix)."""
        for layer in self.scrambling:
            state = apply_layer(state, layer)
        return state

    def apply_layer(self, state: np.ndarray, t: int) -> np.ndarray:
        """Apply brickwork layer ``t`` to ``state`` (vector or density matrix)."""
        return apply_layer(state, self.layers[t])
