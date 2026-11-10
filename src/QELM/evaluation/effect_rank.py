"""Exact rank of the effect space of a monitored brickwork, fixed mask.

A record ``m`` of a monitored circuit (mid-circuit outcomes and the final
``Z`` readout of every site) has the probability
``Pr(m | x) = Tr[E_m rho(x)]``, with ``E_m = K_m^dagger K_m`` its effect.
For a product input ``rho(x)`` the probability is a linear combination of
the Bloch monomials of the input, and the matrix ``A`` of these
coefficients (one row per record, one column per monomial) has a rank: the
dimension of the space of functions of the input that any readout of the
record can reproduce with infinitely many shots.

The rank is computed without listing the records. The span of the effects
obeys a backward (Heisenberg) recursion, written in the real Pauli basis
``{I, X, Y, Z}^N`` of Hermitian operators:

- after the final readout, the span is the diagonal operators,
  ``{I, Z}^N`` (dimension ``2^N``);
- a ``Z`` measurement of site ``i`` maps the span ``S`` to
  ``P_0 (x) S_0 + P_1 (x) S_1``, with ``S_0 = span{A + B}`` and
  ``S_1 = span{A - B}`` when ``E = I (x) A + Z (x) B + X (x) C + Y (x) D``;
- a gate maps ``E`` to ``U^dagger E U``, a rotation of the Pauli
  coefficients that keeps an orthonormal basis orthonormal.

At the input, the dense encoding (Bloch vector on the sphere, all ``4^N``
monomials) sees the whole span, so its rank is the dimension of the span.
The angle encoding (Bloch vector in the ``XZ`` plane, ``3^N`` monomials)
sees its projection on the Pauli strings without ``Y``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import comb

import numpy as np

from QELM.models.circuits.clifford import random_clifford_gate
from QELM.models.circuits.haar import haar_gate, random_u1_gate
from QELM.models.circuits.symmetric import z2_gate

PAULIS: tuple[np.ndarray, ...] = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
"""Single-qubit Pauli matrices in the order I, X, Y, Z (indices 0 to 3)."""

GATES: dict[str, Callable[[np.random.Generator], np.ndarray]] = {
    "haar": haar_gate,
    "u1": random_u1_gate,
    "z2": z2_gate,
    "clifford": random_clifford_gate,
}
"""Two-qubit gate samplers of each circuit class, keyed by its short name."""

_PAULI_2Q = [np.kron(a, b) for a in PAULIS for b in PAULIS]


def heisenberg_transfer(gate: np.ndarray) -> np.ndarray:
    """Real ``16 x 16`` matrix of ``E -> U^dagger E U`` in the Pauli basis.

    Args:
        gate: Two-qubit unitary ``U``, ``4 x 4``, basis ``|q_i q_{i+1}>``.

    Returns:
        Orthogonal matrix ``R`` with ``R[a, b] = Tr[s_a U^dagger s_b U] / 4``,
        so that the coefficients ``e`` of ``E`` become ``R @ e``. The index
        of ``s_a (x) s_b`` is ``4 a + b``.
    """
    u = np.asarray(gate, dtype=complex)
    conj = [u.conj().T @ s @ u for s in _PAULI_2Q]
    r = np.array([[np.trace(sa @ cb) for cb in conj] for sa in _PAULI_2Q]) / 4
    return np.asarray(r.real)


def brick_pairs(n: int) -> list[int]:
    """Left sites ``i`` of the gates ``(i, i + 1)`` of one brickwork layer.

    One layer is a full period, the even half-layer then the odd one, with
    open boundaries: the convention of :class:`QELM.models.circuits.base.
    BrickworkCircuit`, whose ``depth`` counts these periods.
    """
    return list(range(0, n - 1, 2)) + list(range(1, n - 1, 2))


def _orthonormal_rows(m: np.ndarray, tol: float) -> np.ndarray:
    """Orthonormal basis (as rows) of the row span of ``m``.

    Singular values below ``tol`` times the largest one are dropped.
    """
    rows, cols = m.shape
    if rows == 0:
        return m
    if rows > cols:
        # The span is at most ``cols``-dimensional: diagonalise m^T m.
        lam, vec = np.linalg.eigh(m.T @ m)
        if lam[-1] <= 0.0:
            return m[:0]
        keep = lam > (tol**2) * lam[-1]
        return np.ascontiguousarray(vec[:, keep].T)
    # Gram matrix of the rows: its eigenvalues are the squared singular values.
    lam, vec = np.linalg.eigh(m @ m.T)
    if lam[-1] <= 0.0:
        return m[:0]
    keep = lam > (tol**2) * lam[-1]
    q = (vec[:, keep].T @ m) / np.sqrt(lam[keep])[:, None]
    # One Cholesky QR pass removes the loss of orthogonality of the Gram
    # route; q is already well conditioned, so it is stable.
    chol = np.linalg.cholesky(q @ q.T)
    return np.asarray(np.linalg.solve(chol, q))


def measure_site(basis: np.ndarray, n: int, site: int, tol: float) -> np.ndarray:
    """Span of ``{P_s E P_s}`` for ``E`` in the span of ``basis``, ``s = 0, 1``.

    Args:
        basis: Orthonormal rows, shape ``(R, 4**n)``, Pauli coefficients.
        n: Number of qubits.
        site: Measured qubit.
        tol: Relative tolerance of the rank.

    Returns:
        Orthonormal rows of the new span, ``P_0 (x) S_0 + P_1 (x) S_1``.
    """
    r = basis.shape[0]
    left, right = 4**site, 4 ** (n - site - 1)
    v = basis.reshape(r, left, 4, right)
    a, b = v[:, :, 0, :], v[:, :, 3, :]
    q0 = _orthonormal_rows((a + b).reshape(r, -1), tol)
    q1 = _orthonormal_rows((a - b).reshape(r, -1), tol)
    r0, r1 = q0.shape[0], q1.shape[0]
    out = np.zeros((r0 + r1, left, 4, right))
    h = 1 / np.sqrt(2)
    q0 = q0.reshape(r0, left, right)
    q1 = q1.reshape(r1, left, right)
    out[:r0, :, 0, :] = h * q0
    out[:r0, :, 3, :] = h * q0
    out[r0:, :, 0, :] = h * q1
    out[r0:, :, 3, :] = -h * q1
    return out.reshape(r0 + r1, 4**n)


def apply_gate(
    basis: np.ndarray, n: int, site: int, transfer: np.ndarray
) -> np.ndarray:
    """Apply ``E -> U^dagger E U`` on sites ``(site, site + 1)`` to every row."""
    r = basis.shape[0]
    left, right = 4**site, 4 ** (n - site - 2)
    v = basis.reshape(r * left, 16, right)
    return np.asarray(np.matmul(transfer, v)).reshape(r, 4**n)


def final_readout_basis(n: int) -> np.ndarray:
    """Orthonormal rows of the diagonal operators, the strings of ``{I, Z}^n``."""
    out = np.zeros((2**n, 4**n))
    for k in range(2**n):
        bits = [(k >> (n - 1 - q)) & 1 for q in range(n)]
        idx = 0
        for bit in bits:
            idx = 4 * idx + (3 if bit else 0)
        out[k, idx] = 1.0
    return out


def no_y_columns(n: int) -> np.ndarray:
    """Boolean mask of the Pauli strings without ``Y``, the angle monomials."""
    digits = np.indices((4,) * n).reshape(n, -1)
    return np.asarray(np.all(digits != 2, axis=0))


@dataclass(frozen=True)
class EffectRank:
    """Ranks of the effect space of one circuit and one mask.

    Attributes:
        dense: Rank seen by the dense encoding, the dimension of the span.
        angle: Rank seen by the angle encoding, the projection without ``Y``.
    """

    dense: int
    angle: int


def effect_rank(
    transfers: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    tol: float = 1e-6,
) -> EffectRank:
    """Exact rank of the effect space of a monitored brickwork.

    Forward order of layer ``t``: the gates of :func:`brick_pairs` in their
    order (even bonds, then odd bonds), then a ``Z`` measurement of each
    site with ``masks[t, i]`` true. The final ``Z`` readout of every site
    follows the last layer.

    Args:
        transfers: ``transfers[t][g]`` is the :func:`heisenberg_transfer`
            of gate ``g`` of layer ``t``, in the order of
            :func:`brick_pairs`.
        masks: Boolean array ``(T, n)`` of the mid-circuit measurements.
        tol: Relative tolerance of every rank.

    Returns:
        The dense and angle ranks.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    basis = final_readout_basis(n)
    for t in reversed(range(depth)):
        for measured in np.flatnonzero(masks[t]):
            basis = measure_site(basis, n, int(measured), tol)
        pairs = brick_pairs(n)
        for left, transfer in zip(reversed(pairs), reversed(transfers[t])):
            basis = apply_gate(basis, n, left, transfer)
    dense = basis.shape[0]
    angle = int(_orthonormal_rows(basis[:, no_y_columns(n)], tol).shape[0])
    return EffectRank(dense=dense, angle=angle)


def draw_circuit(
    kind: str, n: int, depth: int, rng: np.random.Generator
) -> list[list[np.ndarray]]:
    """Heisenberg transfers of a random brickwork of the class ``kind``."""
    sampler = GATES[kind]
    return [
        [heisenberg_transfer(sampler(rng)) for _ in brick_pairs(n)]
        for _ in range(depth)
    ]


def rank_bounds(n: int) -> dict[str, dict[str, float]]:
    """Reference bounds of the rank, per encoding and symmetry.

    Returns:
        ``{"angle": {...}, "dense": {...}}``, each with the keys ``none``
        (no symmetry), ``z2``, ``u1`` and ``readout`` (``2^n``, final
        readout only).
    """
    return {
        "angle": {
            "none": 3.0**n,
            "z2": (3.0**n + 1) / 2,
            "u1": (3.0**n + 1) / 2,
            "readout": 2.0**n,
        },
        "dense": {
            "none": 4.0**n,
            "z2": 4.0**n / 2,
            "u1": float(comb(2 * n, n)),
            "readout": 2.0**n,
        },
    }
