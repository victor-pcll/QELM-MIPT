"""Exact ``W(m)`` and ``Delta`` of the Haar MIPT, literature circuit (issue #348).

The circuit is the one of Zabalo et al., PRB 101, 060301(R) (2020), and of
the Clifford runs of #337: a periodic brickwork of ``N`` qubits (``N``
even), half-layer ``t`` applying independent Haar-random two-qubit gates on
the bonds of :func:`~QELM.models.circuits.stabilizer.periodic_bonds`
``(N, t)``, then a ``Z`` measurement of each site with probability ``p``.
One time step of the paper is one such half-layer.

For one circuit and one mid-circuit record ``m``, the Kraus operator
``K_m`` is computed exactly: all ``2^N`` input basis states are evolved
together (the columns of ``K``), and each outcome is drawn with the Born rule
of the maximally mixed input, ``Pr(m) = Tr[K_m^dagger K_m] / 2^N``. With
``E = K^dagger K`` the effect of the record, the dense Fourier weight of
``x -> Pr(m | x)`` outside the constant mode is (#314)

    1 + W(m) = 2^N Tr[E^2] / Tr[E]^2,

and the missing weight is ``Delta(m) = N - log2(1 + W(m))``. Since
``Tr[E^2] / Tr[E]^2`` is the purity of ``K K^dagger / Tr[K K^dagger]``, the
state reached from ``I / 2^N``, ``Delta`` is its Renyi-2 entropy in bits:
the purification order parameter of Gullans & Huse, PRX 10, 041020 (2020),
evaluated without any approximation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QELM.models.circuits.haar import haar_gate
from QELM.models.circuits.stabilizer import periodic_bonds


def apply_pair(psi: np.ndarray, n: int, a: int, b: int, gate: np.ndarray) -> np.ndarray:
    """Apply a two-qubit gate on the sites ``(a, b)`` of batched state vectors.

    Args:
        psi: State vectors, shape ``(..., 2**n)``, qubit 0 the most
            significant bit; leading axes are batched.
        n: Number of qubits.
        a: First site of the gate (most significant bit of its basis).
        b: Second site, ``b != a``; need not be adjacent.
        gate: ``4 x 4`` unitary in the basis ``|q_a q_b>``.

    Returns:
        The new state vectors, same shape.
    """
    lead = psi.shape[:-1]
    k = len(lead)
    t = psi.reshape(*lead, *([2] * n))
    t = np.moveaxis(t, (k + a, k + b), (-2, -1))
    shape = t.shape
    t = np.matmul(t.reshape(*shape[:-2], 4), gate.T).reshape(shape)
    t = np.moveaxis(t, (-2, -1), (k + a, k + b))
    return np.ascontiguousarray(t).reshape(psi.shape)


@dataclass(frozen=True)
class Record:
    """``W(m)`` and ``Delta(m)`` of one circuit and record, at the checkpoints.

    Attributes:
        weight: ``W(m) = 2^N Tr[E^2] / Tr[E]^2 - 1``.
        delta: ``N - log2(1 + W(m))``, in bits.
        measured: Number of measurements made up to each checkpoint.
    """

    weight: np.ndarray
    delta: np.ndarray
    measured: np.ndarray


def haar_record(
    n: int,
    p: float,
    checkpoints: Sequence[int],
    rng: np.random.Generator,
    dtype: type = np.complex128,
) -> Record:
    """Draw one Haar circuit and one Born record, return ``W`` and ``Delta``.

    Args:
        n: Number of qubits, even, at least 4.
        p: Measurement probability of each site after each half-layer.
        checkpoints: Increasing numbers of half-layers after which ``W`` and
            ``Delta`` are read; the circuit has ``max(checkpoints)``
            half-layers.
        rng: Random generator of the gates, the measured sites and the
            outcomes.
        dtype: Complex type of the Kraus operator (``complex64`` halves the
            memory for ``N = 12``).

    Returns:
        The :class:`Record`.

    Raises:
        ValueError: If ``n`` is odd or below 4, or the checkpoints are not
            positive and increasing.
    """
    if n < 4 or n % 2:
        raise ValueError(f"n must be even and >= 4, got {n}.")
    cps = list(checkpoints)
    if not cps or cps[0] < 1 or any(b <= a for a, b in zip(cps, cps[1:], strict=False)):
        raise ValueError(f"checkpoints must be positive and increasing, got {cps}.")
    dim = 2**n
    marks = {c: i for i, c in enumerate(cps)}
    weight = np.zeros(len(cps))
    delta = np.zeros(len(cps))
    measured = np.zeros(len(cps), dtype=int)
    count = 0
    # k is K / sqrt(2^N), output qubits first (rows), input basis states as
    # columns: Tr[k^dagger k] = Tr[E] / 2^N = 1. Every gate is then a matmul
    # over a long trailing axis. The output qubits are stored cyclically
    # rotated: physical qubit ``j`` is logical site ``(j + shift) % n``; a
    # rotation by one site before the odd half-layers turns the bond
    # ``(n - 1, 0)`` into a physical neighbour pair.
    k = np.eye(dim, dtype=dtype) / np.sqrt(dim)
    spare = np.empty_like(k)  # ping-pong buffer: no allocation per gate
    shift = 0
    for t in range(cps[-1]):
        if shift != t % 2:
            _rotate_one(k, spare, forward=shift == 0)
            k, spare = spare, k
            shift = t % 2
        for a, _ in periodic_bonds(n, t):
            gate: np.ndarray = haar_gate(rng).astype(dtype)
            phys = (a - shift) % n
            np.matmul(
                gate, k.reshape(2**phys, 4, -1), out=spare.reshape(2**phys, 4, -1)
            )
            k, spare = spare, k
        for site in np.flatnonzero(rng.random(n) < p).tolist():
            phys = (site - shift) % n
            v = k.reshape(2**phys, 2, -1)
            p0 = float(np.linalg.norm(v[:, 0, :]) ** 2)
            total = float(np.linalg.norm(k) ** 2)
            o = 0 if rng.random() * total < p0 else 1
            v[:, 1 - o, :] = 0.0
            k /= np.sqrt(p0 if o == 0 else total - p0)
            count += 1
        if t + 1 in marks:
            i = marks[t + 1]
            purity = _purity(k)
            weight[i] = dim * purity - 1.0
            delta[i] = n - np.log2(dim * purity)
            measured[i] = count
    return Record(weight=weight, delta=delta, measured=measured)


def _rotate_one(k: np.ndarray, out: np.ndarray, forward: bool) -> None:
    """Write in ``out`` the rows of ``k`` with the output qubits rotated by one.

    ``forward``: the new physical qubit ``j`` is the old ``j + 1`` (and the
    new ``n - 1`` the old ``0``); otherwise the new ``j`` is the old
    ``j - 1``.
    """
    dim = k.shape[0]
    shape = (2, dim // 2, dim) if forward else (dim // 2, 2, dim)
    out.reshape(shape[1], shape[0], dim)[...] = k.reshape(shape).transpose(1, 0, 2)


def _purity(k: np.ndarray) -> float:
    """Purity ``Tr[G^2] / Tr[G]^2`` of ``G = k k^dagger``, the state from ``I / 2^N``.

    ``G`` has the spectrum of ``E / 2^N``. The rows that the last
    measurements set to zero are removed first, and the Gram matrix is taken
    on the smaller side.
    """
    m = k[np.flatnonzero(np.any(k != 0, axis=1))]
    small = m @ m.conj().T if m.shape[0] <= m.shape[1] else m.conj().T @ m
    norm = float(np.trace(small).real)
    return float(np.vdot(small, small).real) / norm**2
