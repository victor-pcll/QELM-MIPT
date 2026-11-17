"""Bipartite entanglement of the effect dictionary, periodic brickwork (issue #358).

Circuit of #337 / #346 / #348: a periodic brickwork of ``N`` qubits (``N``
even), half-layer ``t`` applying independent two-qubit gates on the bonds of
:func:`~QELM.models.circuits.stabilizer.periodic_bonds` ``(N, t)``, then a
``Z`` measurement of each site with probability ``p``; ``T`` half-layers,
then a ``Z`` readout of every site.

With the full final readout every effect has rank one,
``E_m = |phi_m><phi_m|`` with ``phi_m = A_m^dagger |m_f>`` (#352), and
``psi_m = phi_m / ||phi_m||`` is the state of the ``N`` reference qubits
after the readout (up to complex conjugation). Its entropy profile ``S(l)``
counts the functions of a window that the dictionary can represent
(``2^(l - S(l))`` of them in Clifford, #339).

``psi_m`` is an ordinary monitored trajectory run backwards: start from the
bit string ``m_f``, then for ``t = T - 1 .. 0`` project the sites measured
after half-layer ``t`` and apply the adjoint gates of half-layer ``t``.
Drawing ``m_f`` uniformly and each projection by the Born rule of the
backward state gives the record the probability ``Tr(E_m) / 2^N``: the
record distribution of the maximally mixed input. A Haar trajectory then
costs a state vector of ``N`` qubits, instead of the ``2^N x 2^N`` Kraus
operator of :func:`QELM.evaluation.haar_mipt.haar_record`.

This module provides that backward trajectory for Haar gates, the entropy
profile averaged over the positions of the interval (von Neumann and
Renyi-2, in bits), and the tripartite information of the four quarters,
for state vectors and for Clifford stabilizer groups.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QELM.evaluation.clifford_ridge import GroupBasis, interval_entropies
from QELM.evaluation.haar_mipt import apply_pair
from QELM.models.circuits.haar import haar_gate
from QELM.models.circuits.stabilizer import periodic_bonds


@dataclass(frozen=True)
class HaarCircuit:
    """Gates and measurement mask of one periodic Haar brickwork.

    Attributes:
        gates: ``gates[t, k]``, the ``4 x 4`` gate on bond ``k`` of
            ``periodic_bonds(N, t)``; shape ``(T, N // 2, 4, 4)``.
        measured: ``True`` where a site is measured after half-layer ``t``,
            shape ``(T, N)``.
    """

    gates: np.ndarray
    measured: np.ndarray

    @property
    def n(self) -> int:
        return int(self.measured.shape[1])

    @property
    def depth(self) -> int:
        return int(self.measured.shape[0])


def draw_haar_circuit(
    n: int, depth: int, p: float, rng: np.random.Generator
) -> HaarCircuit:
    """Draw the gates (Haar on U(4)) and the mask (each site with probability ``p``)."""
    if n < 4 or n % 2:
        raise ValueError(f"n must be even and >= 4, got {n}.")
    gates = np.array([[haar_gate(rng) for _ in range(n // 2)] for _ in range(depth)])
    return HaarCircuit(gates=gates, measured=rng.random((depth, n)) < p)


def _apply(psi: np.ndarray, n: int, a: int, b: int, gate: np.ndarray) -> np.ndarray:
    """Apply ``gate`` on ``(a, b)``; fast path for the bonds ``(a, a + 1)``."""
    if b == a + 1:
        t = psi.reshape(2**a, 4, -1)
        return np.matmul(gate, t).reshape(-1)
    return apply_pair(psi, n, a, b, gate)


def backward_trajectory(
    circuit: HaarCircuit,
    rng: np.random.Generator,
    outcomes: dict[tuple[int, int], int] | None = None,
    final: Sequence[int] | None = None,
) -> tuple[np.ndarray, float, dict[tuple[int, int], int], np.ndarray]:
    """Draw a record of the circuit and return its normalised ``psi_m``.

    Args:
        circuit: Gates and mask.
        rng: Random generator of the final bits and of the outcomes.
        outcomes: Optional forced mid-circuit outcomes ``{(t, site): bit}``
            (for tests); the others are drawn by the Born rule.
        final: Optional forced final bit string.

    Returns:
        ``(psi, log2_prob, outcomes, final)``: the state (``2^N``, qubit 0
        the most significant bit), ``log2`` of the record probability
        ``Tr(E_m) / 2^N``, the mid-circuit outcomes and the final bits.
    """
    n, depth = circuit.n, circuit.depth
    bits = np.asarray(
        rng.integers(2, size=n) if final is None else final, dtype=np.int64
    )
    psi = np.zeros(2**n, dtype=complex)
    psi[int(sum(int(b) << (n - 1 - k) for k, b in enumerate(bits)))] = 1.0
    log2_prob = -float(n)
    record: dict[tuple[int, int], int] = {}
    for t in reversed(range(depth)):
        for site in np.flatnonzero(circuit.measured[t]).tolist():
            v = psi.reshape(2**site, 2, -1)
            p0 = float(np.sum(np.abs(v[:, 0, :]) ** 2))
            if outcomes is not None and (t, site) in outcomes:
                o = outcomes[(t, site)]
            else:
                o = int(rng.random() >= p0)
            prob = p0 if o == 0 else 1.0 - p0
            if prob <= 0.0:
                raise ValueError(f"Outcome {o} at {(t, site)} has probability 0.")
            v[:, 1 - o, :] = 0.0
            psi /= np.sqrt(prob)
            log2_prob += float(np.log2(prob))
            record[(t, site)] = o
        for (a, b), gate in zip(periodic_bonds(n, t), circuit.gates[t], strict=True):
            psi = _apply(psi, n, a, b, gate.conj().T)
    return psi, log2_prob, record, bits


def _rotate(psi: np.ndarray, n: int, shift: int) -> np.ndarray:
    """Qubit ``j`` of the result is qubit ``(j + shift) % n`` of ``psi``."""
    return np.ascontiguousarray(psi.reshape(2**shift, 2 ** (n - shift)).T).reshape(-1)


def _spectrum(psi: np.ndarray, n_left: int) -> np.ndarray:
    """Schmidt probabilities of the cut after the first ``n_left`` qubits."""
    m = psi.reshape(2**n_left, -1)
    if m.shape[0] > m.shape[1]:
        m = m.T
    lam = np.linalg.eigvalsh(m @ m.conj().T)
    lam = np.clip(lam.real, 0.0, None)
    return lam / lam.sum()


def _entropies(lam: np.ndarray) -> tuple[float, float]:
    """Von Neumann and Renyi-2 entropies (bits) of a probability vector."""
    q = lam[lam > 1e-15]
    return float(-np.sum(q * np.log2(q))), float(-np.log2(np.sum(lam**2)))


@dataclass(frozen=True)
class Profile:
    """Entropy profile of one pure state, averaged over interval positions.

    Attributes:
        s1: Von Neumann ``S(l)``, ``l = 0 .. N // 2``, bits.
        s2: Renyi-2 ``S_2(l)``, same shape.
        i3: Tripartite information of the quarters (von Neumann), ``nan``
            when ``N`` is not a multiple of 4.
        i3_2: Same with Renyi-2.
    """

    s1: np.ndarray
    s2: np.ndarray
    i3: float
    i3_2: float


def state_profile(
    psi: np.ndarray, n: int, starts: Sequence[int] | None = None
) -> Profile:
    """Entropy profile of a pure state of ``n`` qubits on a ring.

    ``S(l)`` is averaged over the starting sites ``starts`` (default every
    site), with the interval ``[s, s + l)`` taken modulo ``n``. ``I3`` of the
    quarters ``A, B, C, D`` is ``S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC``,
    averaged over the same starts.

    Args:
        psi: Normalised state, length ``2**n``.
        n: Number of qubits.
        starts: Starting sites.

    Returns:
        The :class:`Profile`.
    """
    starts = list(range(n)) if starts is None else list(starts)
    half = n // 2
    s1 = np.zeros(half + 1)
    s2 = np.zeros(half + 1)
    i3: list[float] = []
    i3_2: list[float] = []
    for s in starts:
        rot = _rotate(psi, n, s % n) if s % n else psi
        cut: dict[int, tuple[float, float]] = {}
        for ell in range(1, half + 1):
            cut[ell] = _entropies(_spectrum(rot, ell))
            s1[ell] += cut[ell][0]
            s2[ell] += cut[ell][1]
        if n % 4 == 0:
            q = n // 4
            # S_B and S_C come from the rotations by q and 2q: average over
            # starts makes them equal to S_A on average, but take them
            # exactly from this state.
            sb = _entropies(_spectrum(_rotate(rot, n, q), q))
            sc = _entropies(_spectrum(_rotate(rot, n, 2 * q), q))
            sbc = _entropies(_spectrum(_rotate(rot, n, q), 2 * q))
            # S_AC: bring A next to C by moving B to the end.
            perm = (
                list(range(0, q))
                + list(range(2 * q, 3 * q))
                + list(range(q, 2 * q))
                + list(range(3 * q, n))
            )
            mixed = np.ascontiguousarray(rot.reshape((2,) * n).transpose(perm)).reshape(
                -1
            )
            sac = _entropies(_spectrum(mixed, 2 * q))
            sabc = _entropies(_spectrum(rot, 3 * q))
            sa, sab = cut[q], cut[2 * q]
            for k, out in ((0, i3), (1, i3_2)):
                out.append(sa[k] + sb[k] + sc[k] - sab[k] - sbc[k] - sac[k] + sabc[k])
    s1 /= len(starts)
    s2 /= len(starts)
    nan = float("nan")
    return Profile(
        s1=s1,
        s2=s2,
        i3=float(np.mean(i3)) if i3 else nan,
        i3_2=float(np.mean(i3_2)) if i3_2 else nan,
    )


def stabilizer_entropy(basis: GroupBasis, sites: Sequence[int]) -> int:
    """Entropy (bits) of ``sites`` for the pure stabilizer state of ``basis``.

    The elements supported on ``A`` form a subgroup of rank ``|A| - S_A``
    (#337), so ``S_A = |A| - k_A``.
    """
    return len(set(sites)) - basis.local_rank(sorted(set(sites)))


def stabilizer_profile(generators: np.ndarray) -> Profile:
    """:class:`Profile` of a pure stabilizer state of ``N`` qubits on a ring.

    Every entropy of a stabilizer state is flat, so ``S_2 = S_1``. ``S(l)``
    is averaged over every starting site (:func:`interval_entropies`);
    ``I3`` over the starts ``0 .. N/4 - 1``.

    Args:
        generators: ``N`` independent ``[x | z]`` rows.
    """
    g = np.asarray(generators, dtype=bool)
    n = g.shape[1] // 2
    half = n // 2
    s = interval_entropies(g, max_len=half).mean(axis=0)
    i3 = float("nan")
    if n % 4 == 0:
        q = n // 4
        basis = GroupBasis(g)
        vals = []
        for start in range(q):
            a, b, c = ([(start + k * q + j) % n for j in range(q)] for k in range(3))
            vals.append(
                stabilizer_entropy(basis, a)
                + stabilizer_entropy(basis, b)
                + stabilizer_entropy(basis, c)
                - stabilizer_entropy(basis, a + b)
                - stabilizer_entropy(basis, b + c)
                - stabilizer_entropy(basis, a + c)
                + stabilizer_entropy(basis, a + b + c)
            )
        i3 = float(np.mean(vals))
    return Profile(s1=s, s2=s.copy(), i3=i3, i3_2=i3)


def growth_exponent(
    sizes: Sequence[int], values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete ``alpha = d log S / d log N`` between neighbouring sizes.

    Returns:
        ``(N_mid, alpha)``: geometric means of neighbouring sizes and the
        slopes ``log(S_b / S_a) / log(N_b / N_a)``.
    """
    n = np.asarray(sizes, dtype=float)
    v = np.asarray(values, dtype=float)
    mid = np.sqrt(n[1:] * n[:-1])
    return mid, np.log(v[1:] / v[:-1]) / np.log(n[1:] / n[:-1])
