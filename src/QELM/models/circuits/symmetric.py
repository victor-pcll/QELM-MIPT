"""Brickwork circuits with a symmetry that commutes with Z measurements (#288).

Each circuit conserves a quantity diagonal in the Z basis, so that a Z
measurement leaves it intact too (see
``documentation/notes/early-layers-proof.md``, section 4):

- :class:`Z2BrickworkCircuit`: the parity ``prod_i Z_i``. Each gate is a
  Haar ``2 x 2`` block on ``{|00>, |11>}`` and another on ``{|01>, |10>}``.
- :class:`DiagonalBrickworkCircuit`: every ``Z_i``. Each gate is
  ``exp(-i theta (Z_i + Z_j) / 2)``, the evolution under ``H = sum_i h_i
  Z_i``. Any gate diagonal in the Z basis gives the same law of the Z
  record, because it never changes the diagonal of the state.
- :class:`DipoleCircuit`: the charge ``Q = sum_i n_i`` and the dipole
  (polarization) ``P = sum_i i n_i``, with ``n_i = (1 - Z_i) / 2``. On
  qubits, the first gates that move charge while keeping ``Q`` and ``P`` act
  on four sites: ``|0110> <-> |1001>``. Each gate carries a Haar ``2 x 2``
  block on that pair and random phases on the other 14 basis states. A
  round is four sub-layers of such gates, starting on sites 0, 1, 2 and 3
  (Feldmeier, Sala, De Tomasi, Pollmann, Knap, PRL 125, 245303 (2020)).

The U(1) circuit is :class:`QELM.models.circuits.haar.U1BrickworkCircuit`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from QELM.models.circuits.base import BrickworkCircuit, _generator
from QELM.models.circuits.haar import haar_unitary

# A gate on any number of qubits: the qubits, then the unitary.
GeneralLayer = list[tuple[tuple[int, ...], np.ndarray]]


def apply_gate(
    state: np.ndarray, gate: np.ndarray, qubits: Sequence[int]
) -> np.ndarray:
    """Apply a gate on ``qubits`` to a state vector or a density matrix.

    Args:
        state: ``(2**n,)`` or ``(2**n, 2**n)``.
        gate: ``(2**k, 2**k)`` unitary, ``qubits[0]`` the most significant
            bit of its index.
        qubits: ``k`` distinct qubits.

    Returns:
        ``G psi`` or ``G rho G^dagger``.

    Raises:
        ValueError: If the shapes or the qubits do not match.
    """
    state = np.asarray(state, dtype=complex)
    dim = state.shape[0]
    n = dim.bit_length() - 1
    k = len(qubits)
    if 2**n != dim or state.shape not in ((dim,), (dim, dim)):
        raise ValueError(f"state must be (2**n,) or (2**n, 2**n), got {state.shape}.")
    if gate.shape != (2**k, 2**k):
        raise ValueError(
            f"gate must be {(2**k, 2**k)} for {k} qubits, got {gate.shape}."
        )
    if len(set(qubits)) != k or not all(0 <= q < n for q in qubits):
        raise ValueError(f"Need {k} distinct qubits in [0, {n - 1}], got {qubits}.")

    def rows(M: np.ndarray) -> np.ndarray:
        cols = M.shape[1]
        T = np.moveaxis(M.reshape((2,) * n + (cols,)), list(qubits), list(range(k)))
        shape = T.shape
        T = (gate @ T.reshape(2**k, -1)).reshape(shape)
        return np.moveaxis(T, list(range(k)), list(qubits)).reshape(dim, cols)

    if state.ndim == 1:
        return rows(state[:, None])[:, 0]
    A = rows(state)
    return np.asarray(rows(A.conj().T).conj().T)


def z2_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """A random two-qubit gate that conserves the parity ``Z x Z``.

    Haar ``2 x 2`` blocks on ``{|00>, |11>}`` and on ``{|01>, |10>}``.
    """
    gen = _generator(rng)
    u = np.zeros((4, 4), dtype=complex)
    even, odd = [0, 3], [1, 2]
    u[np.ix_(even, even)] = haar_unitary(2, gen)
    u[np.ix_(odd, odd)] = haar_unitary(2, gen)
    return u


def diagonal_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """``exp(-i theta (Z x 1 + 1 x Z) / 2)``, ``theta`` uniform in ``[0, 2 pi)``."""
    gen = _generator(rng)
    theta = gen.uniform(0.0, 2.0 * np.pi)
    z_sum = np.array([2.0, 0.0, 0.0, -2.0])  # z_i + z_j on |00>, |01>, |10>, |11>
    return np.diag(np.exp(-0.5j * theta * z_sum))


def dipole_gate(rng: np.random.Generator | None = None) -> np.ndarray:
    """A random four-site gate that conserves ``sum n_i`` and ``sum i n_i``.

    In the basis ``|b_0 b_1 b_2 b_3>`` (``b_0`` most significant), only
    ``|0110>`` (index 6) and ``|1001>`` (index 9) share both charges; they
    get a Haar ``2 x 2`` block, every other basis state a random phase.
    """
    gen = _generator(rng)
    u = np.diag(np.exp(1j * gen.uniform(0.0, 2.0 * np.pi, size=16))).astype(complex)
    pair = [6, 9]
    u[np.ix_(pair, pair)] = haar_unitary(2, gen)
    return u


class Z2BrickworkCircuit(BrickworkCircuit):
    """Brickwork of parity-conserving gates (:func:`z2_gate`)."""

    name: str = "Z2_circuit"

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a random parity-conserving gate."""
        return z2_gate(rng)


class DiagonalBrickworkCircuit(BrickworkCircuit):
    """Brickwork of gates diagonal in Z (:func:`diagonal_gate`), ``H = sum Z_i``."""

    name: str = "diagonal_circuit"

    def draw_gate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a random diagonal gate."""
        return diagonal_gate(rng)


class DipoleCircuit:
    """Four-site gates that conserve the charge and the dipole (:func:`dipole_gate`).

    Same interface as :class:`BrickworkCircuit` for the experiments: ``N``,
    ``Nr``, ``N_total``, ``depth``, ``layers`` and :meth:`apply_layer`.
    Round ``t`` applies four sub-layers, the gates on sites ``(s, s + 1, s +
    2, s + 3)`` for ``s = o, o + 4, ...`` with ``o = 0, 1, 2, 3``, on an open
    chain.
    """

    name: str = "dipole_circuit"

    def __init__(self, params: dict[str, Any], rng: np.random.Generator | None = None):
        self.N: int = int(params["N"])
        self.Nr: int = int(params.get("Nr", 1))
        self.N_total: int = self.N + self.Nr
        self.depth: int = int(params["depth"])
        if self.N < 4:
            raise ValueError(f"DipoleCircuit needs N >= 4, got N={self.N}.")
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}.")
        self.layers: list[GeneralLayer] = []
        self.redraw(rng)

    def blocks(self) -> list[tuple[int, ...]]:
        """The qubits of the gates of one round, in order."""
        out: list[tuple[int, ...]] = []
        for offset in range(4):
            for s in range(offset, self.N - 3, 4):
                out.append(tuple(self.Nr + s + k for k in range(4)))
        return out

    def redraw(self, rng: np.random.Generator | None = None) -> None:
        """Draw new gates for every round."""
        gen = _generator(rng)
        self.layers = [
            [(qubits, dipole_gate(gen)) for qubits in self.blocks()]
            for _ in range(self.depth)
        ]

    def apply_layer(self, state: np.ndarray, t: int) -> np.ndarray:
        """Apply round ``t`` to a state vector or a density matrix."""
        for qubits, gate in self.layers[t]:
            state = apply_gate(state, gate, qubits)
        return state
