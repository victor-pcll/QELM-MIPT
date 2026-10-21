"""Random brickwork circuits of two-qubit gates: U(1), Haar and Clifford.

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

The same brickwork carries two other kinds of gates (issue #171), used as
the unitary dynamics of the QELM experiment in place of the TFIM:

- :class:`HaarBrickworkCircuit`: each gate is a Haar-random ``4 x 4``
  unitary (:func:`haar_unitary` with ``n = 4``), with no conserved charge;
- :class:`CliffordBrickworkCircuit`: each gate is drawn uniformly from the
  two-qubit Clifford group (:func:`random_clifford_gate`), the 11 520
  elements of the group up to a global phase. A Clifford gate maps Pauli
  strings to Pauli strings, the dynamics of the stabilizer-circuit studies
  of the measurement-induced transition.

Qubit 0 is the leftmost factor of the tensor product, as in
:func:`QELM.physics.operators.get_site_operator`.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
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


def haar_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a Haar-random two-qubit gate, a ``4 x 4`` unitary of U(4).

    Args:
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex ``4 x 4`` unitary in the basis ``|00>, |01>, |10>, |11>``.
    """
    return haar_unitary(4, rng)


def _phase_key(u: np.ndarray) -> bytes:
    """Hashable key of ``u`` up to a global phase.

    ``u`` is divided by the phase of its first non-zero entry (row-major
    order) and rounded, so two matrices that differ by a global phase share
    the key.
    """
    flat = u.ravel()
    first = flat[np.argmax(np.abs(flat) > 1e-9)]
    v = np.round(flat * (abs(first) / first), 9) + 0.0  # + 0.0: no -0.0
    return v.tobytes()


@cache
def clifford_group_2q() -> tuple[np.ndarray, ...]:
    """Return the two-qubit Clifford group, one matrix per element.

    The group is generated by ``H`` and ``S`` on each qubit and the ``CNOT``
    from qubit 0 to qubit 1. It is enumerated once, by a breadth-first
    search from the identity, and cached. Two matrices that differ by a
    global phase are the same element, so the group has
    ``|C_2| = 11 520`` elements.

    Returns:
        The ``11 520`` complex ``4 x 4`` unitaries, in the basis
        ``|00>, |01>, |10>, |11>``, in a fixed order.
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.diag([1, 1j])
    I2 = np.eye(2, dtype=complex)
    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    generators = [np.kron(H, I2), np.kron(I2, H), np.kron(S, I2), np.kron(I2, S), CNOT]
    identity = np.eye(4, dtype=complex)
    elements = [identity]
    seen = {_phase_key(identity)}
    frontier = [identity]
    while frontier:
        nxt = []
        for u in frontier:
            for g in generators:
                v = g @ u
                key = _phase_key(v)
                if key not in seen:
                    seen.add(key)
                    elements.append(v)
                    nxt.append(v)
        frontier = nxt
    return tuple(elements)


def random_clifford_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw a two-qubit Clifford gate uniformly from the group.

    Args:
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        Complex ``4 x 4`` unitary, one of the ``11 520`` elements of
        :func:`clifford_group_2q`, each with probability ``1 / 11 520``.
    """
    group = clifford_group_2q()
    return group[int(_generator(rng).integers(len(group)))].copy()


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
            [(i, j, self.draw_gate(gen)) for i, j in self.scrambling_bonds(gen)]
            for _ in range(self.scrambling_layers)
        ]
        # One layer = the even half-layer, then the odd one.
        self.layers = [
            [(i, j, self.draw_gate(gen)) for i, j in self.bonds(0) + self.bonds(1)]
            for _ in range(self.depth)
        ]

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one gate of the circuit, here a random U(1) gate.

        Subclasses override it to change the kind of gate and keep the
        brickwork.
        """
        return random_u1_gate(rng)

    def scramble(self, state: np.ndarray) -> np.ndarray:
        """Apply the scrambling layers to ``state`` (vector or density matrix)."""
        for layer in self.scrambling:
            state = apply_layer(state, layer)
        return state

    def apply_layer(self, state: np.ndarray, t: int) -> np.ndarray:
        """Apply brickwork layer ``t`` to ``state`` (vector or density matrix)."""
        return apply_layer(state, self.layers[t])


class HaarBrickworkCircuit(U1BrickworkCircuit):
    """Brickwork circuit of Haar-random two-qubit gates on the system qubits.

    Same geometry and parameters as :class:`U1BrickworkCircuit`, with each
    gate drawn from the Haar measure on U(4) (:func:`haar_gate`): no charge
    is conserved. Config block ``haar_circuit`` (issue #171).
    """

    name: str = "haar_circuit"

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a Haar-random ``4 x 4`` unitary."""
        return haar_gate(rng)


class CliffordBrickworkCircuit(U1BrickworkCircuit):
    """Brickwork circuit of random two-qubit Clifford gates on the system qubits.

    Same geometry and parameters as :class:`U1BrickworkCircuit`, with each
    gate drawn uniformly from the two-qubit Clifford group
    (:func:`random_clifford_gate`). Config block ``clifford_circuit``
    (issue #171).
    """

    name: str = "clifford_circuit"

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a uniformly random two-qubit Clifford gate."""
        return random_clifford_gate(rng)
