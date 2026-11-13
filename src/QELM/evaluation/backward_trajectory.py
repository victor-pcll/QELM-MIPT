"""Backward trajectories of a monitored brickwork with a full final readout (#352).

Geometry of #334/#339: open brickwork, one step = the gates of
:func:`QELM.evaluation.effect_rank.brick_pairs` (even bonds, then odd), then
a ``Z`` measurement of the masked sites; ``T`` steps; final ``Z`` readout of
every site. Qubit 0 is the most significant bit, and a gate on
``(i, i + 1)`` has ``i`` as its first tensor factor.

With the final readout, the Kraus operator of a record is
``K_m = |m_f><m_f| A_m`` and its effect ``E_m = K_m^dagger K_m`` has rank one:
``E_m = |phi_m><phi_m|`` with ``phi_m = A_m^dagger |m_f>``. Going backwards,
``phi_m`` is the state of an ordinary monitored trajectory: start from the
bit string ``m_f``, then for ``t = T - 1 .. 0`` project the sites measured
at step ``t`` and apply ``U_t^dagger`` (gates in reverse order).

Sampling ``m_f`` uniformly and every projection with the Born rule of the
backward state draws the record with probability ``Tr(E_m) / 2^N``, the
record distribution of the maximally mixed input. The entanglement of the
normalised ``psi_m = phi_m / ||phi_m||`` is the entanglement of the
reference qubits after the full readout, the quantity whose profile ``S(l)``
gives the locality of the ridge dictionary in the Clifford case (#337, #339).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from QELM.evaluation.effect_rank import brick_pairs


def _apply_two(psi: np.ndarray, gate: np.ndarray, site: int, n: int) -> np.ndarray:
    """Apply a ``4 x 4`` gate on sites ``(site, site + 1)``."""
    t = psi.reshape(2**site, 4, 2 ** (n - site - 2))
    return np.asarray(np.einsum("ab,ibj->iaj", gate, t)).reshape(-1)


def _measure(psi: np.ndarray, site: int, n: int, rng: np.random.Generator) -> tuple:
    """Born-rule ``Z`` measurement of one site; returns ``(state, outcome)``."""
    t = psi.reshape(2**site, 2, 2 ** (n - site - 1))
    p0 = float(np.sum(np.abs(t[:, 0, :]) ** 2))
    outcome = int(rng.random() >= p0)
    out = np.zeros_like(t)
    out[:, outcome, :] = t[:, outcome, :]
    norm = np.sqrt(p0 if outcome == 0 else 1.0 - p0)
    return (out / norm).reshape(-1), outcome


def backward_trajectory(
    gates: Sequence[Sequence[np.ndarray]],
    masks: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Draw a record and its normalised backward state ``psi_m``.

    Args:
        gates: ``gates[t][g]``, the ``4 x 4`` unitaries of step ``t`` in the
            order of :func:`brick_pairs`.
        masks: ``True`` where a site is measured after step ``t``, shape
            ``(T, N)``.
        rng: Random generator (final bits and Born outcomes).

    Returns:
        ``(psi, final_bits, mid)``: the state (length ``2^N``), the final
        bit string, and the mid-circuit outcomes as ``(t, site, bit)``.
    """
    masks = np.asarray(masks, dtype=bool)
    depth, n = masks.shape
    final = np.asarray(rng.integers(2, size=n), dtype=np.int64)
    psi = np.zeros(2**n, dtype=complex)
    psi[int("".join(str(int(b)) for b in final), 2)] = 1.0
    pairs = brick_pairs(n)
    mid: list[tuple[int, int, int]] = []
    for t in reversed(range(depth)):
        for site in np.flatnonzero(masks[t]):
            psi, bit = _measure(psi, int(site), n, rng)
            mid.append((t, int(site), bit))
        for left, gate in zip(reversed(pairs), reversed(gates[t]), strict=True):
            psi = _apply_two(psi, np.conj(gate).T, left, n)
    return psi, final, mid


def entropy(psi: np.ndarray, n: int, sites: Sequence[int]) -> float:
    """Von Neumann entropy (bits) of the sites ``sites`` of a pure state."""
    a = sorted(int(s) for s in sites)
    if not a or len(a) == n:
        return 0.0
    rest = [q for q in range(n) if q not in a]
    m = psi.reshape((2,) * n).transpose(a + rest).reshape(2 ** len(a), -1)
    sv = np.linalg.svd(m, compute_uv=False) ** 2
    sv = sv[sv > 1e-14]
    return float(-np.sum(sv * np.log2(sv)))


def cut_profile(psi: np.ndarray, n: int) -> np.ndarray:
    """``S(l)`` of the left block ``[0, l)`` for ``l = 0 .. n``."""
    return np.array([entropy(psi, n, range(ell)) for ell in range(n + 1)])


def tripartite_information(psi: np.ndarray, n: int) -> float:
    """``I3(A:B:C)`` for the four contiguous quarters ``A, B, C, D`` (``4 | n``)."""
    if n % 4:
        raise ValueError("I3 needs N divisible by 4.")
    q = n // 4
    a, b, c = (list(range(k * q, (k + 1) * q)) for k in range(3))

    def s(*parts: list[int]) -> float:
        return entropy(psi, n, [x for p in parts for x in p])

    return s(a) + s(b) + s(c) - s(a, b) - s(a, c) - s(b, c) + s(a, b, c)
