"""Per-trajectory charge variance of a monitored U(1) circuit (issue #299).

The charge-sharpening transition (Agrawal et al., PRX 12, 041002 (2022);
Barratt et al., PRL 129, 120604 (2022)) is a statement about the variance
of the global charge ``Q = sum_i b_i`` in the state of one trajectory,
``Var(Q) = <Q^2> - <Q>^2``: it stays finite in the charge-fuzzy phase and
goes to 0 in the sharp phase. With ``Z_i = 1 - 2 b_i``,

    Var(Q) = (1/4) [ sum_ij <Z_i Z_j> - (sum_i <Z_i>)^2 ],

linear in the two-point functions and quadratic in the one-point functions.
This module simulates the trajectories and computes ``Var(Q)`` and the
QELM features ``<Z_i>``, ``<Z_i Z_j>`` of their final state, exactly or
from a finite number of shots.

The circuit is the one of :mod:`QELM.evaluation.learnability` (Barratt et
al. 2022): a chain of ``L`` qubits, ``T`` steps of a random U(1) brickwork,
every site measured in Z with probability ``p`` after each layer (or each
step). A state of definite charge has ``Var(Q) = 0`` at every ``p``, so the
chain starts in the superposition

    (|Q0> + |Q1>) / sqrt(2)

of the uniform states of two charges (:func:`superposition_state`). The
gates conserve ``Q`` and the measurements are diagonal, so the state stays
``sqrt(w) |psi_0> + sqrt(1 - w) |psi_1>`` with ``|psi_q>`` in sector ``q``,
and, for ``Q1 = Q0 - 1``,

    Var(Q) = w (1 - w),

``w`` the weight left in ``Q0``: ``1/4`` while the record says nothing about
the charge, 0 once it has revealed it. ``w`` is the posterior of ``Q0``
given the record for an observer who knows the gates; the entropy of a
reference qubit entangled with the two sectors (#137) is ``H(w)``.

The cross terms between sectors never enter the Born probabilities, so a
record has probability ``(P(m | Q0) + P(m | Q1)) / 2``: the records are the
ones of :func:`QELM.evaluation.learnability.sample_records` with a charge
drawn at random, and the gate-ignorant (SEP) posterior of that module
applies to them unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QELM.evaluation.learnability import (
    UNMEASURED,
    _generator,
    apply_gates,
    charge_state,
    default_charges,
    measurement_rounds,
    random_u1_gates,
    site_bits,
)


def superposition_state(L: int, charges: Sequence[int]) -> np.ndarray:
    """Return the equal superposition of the uniform states of ``charges``.

    Args:
        L: Number of sites.
        charges: Distinct charges, each in ``[0, L]``.

    Returns:
        Normalized real state vector of shape ``(2**L,)``.

    Raises:
        ValueError: If the charges are not distinct or out of range.
    """
    qs = [int(q) for q in charges]
    if len(qs) < 1 or len(set(qs)) != len(qs):
        raise ValueError(f"Need distinct charges, got {qs}.")
    psi = sum(charge_state(L, q) for q in qs)
    return np.asarray(psi / np.sqrt(len(qs)))


def pair_indices(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs ``i < j`` of the two-point features, row-major."""
    i, j = np.triu_indices(L, k=1)
    return np.asarray(i), np.asarray(j)


def z_signs(L: int) -> np.ndarray:
    """Return ``Z_i`` (+1 or -1) of site ``i`` in each configuration, ``(2**L, L)``."""
    return np.asarray(1 - 2 * site_bits(L), dtype=float)


def exact_features(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``<Z_i>`` and ``<Z_i Z_j>`` (``i < j``) of Born distributions.

    Args:
        probs: ``|psi|^2`` of each state, shape ``(n, 2**L)``.

    Returns:
        ``(z, zz)`` of shapes ``(n, L)`` and ``(n, L (L - 1) / 2)``.
    """
    probs = np.asarray(probs, dtype=float)
    L = probs.shape[1].bit_length() - 1
    s = z_signs(L)
    i, j = pair_indices(L)
    return probs @ s, probs @ (s[:, i] * s[:, j])


def charge_moments(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``<Q>`` and ``Var(Q)`` of Born distributions, ``Q = sum_i b_i``.

    Args:
        probs: Shape ``(n, 2**L)``.

    Returns:
        ``(mean, var)``, each of shape ``(n,)``; ``var`` is clipped at 0.
    """
    probs = np.asarray(probs, dtype=float)
    L = probs.shape[1].bit_length() - 1
    Q = site_bits(L).sum(axis=1).astype(float)
    mean = probs @ Q
    var = probs @ Q**2 - mean**2
    return mean, np.maximum(var, 0.0)


def variance_from_features(z: np.ndarray, zz: np.ndarray) -> np.ndarray:
    """``Var(Q) = (1/4) [L + 2 sum_{i<j} <Z_i Z_j> - (sum_i <Z_i>)^2]``.

    The identity that the readouts have to learn; exact on exact features.

    Args:
        z: ``<Z_i>``, shape ``(n, L)``.
        zz: ``<Z_i Z_j>`` for ``i < j``, shape ``(n, L (L - 1) / 2)``.

    Returns:
        Shape ``(n,)``.
    """
    L = z.shape[1]
    return np.asarray((L + 2 * zz.sum(axis=1) - z.sum(axis=1) ** 2) / 4)


@dataclass(frozen=True)
class ShotEstimates:
    """Features and charge variance estimated from ``n_shots`` bitstrings.

    Attributes:
        z: Sample means of ``Z_i``, shape ``(n, L)``.
        zz: Sample means of ``Z_i Z_j``, ``i < j``, shape ``(n, L (L-1)/2)``.
        var_direct: Unbiased sample variance (``ddof=1``) of the Hamming
            weight ``Q`` of the shots, shape ``(n,)``: the direct estimator,
            with no readout.
    """

    z: np.ndarray
    zz: np.ndarray
    var_direct: np.ndarray


def shot_estimates(
    probs: np.ndarray, n_shots: int, rng: np.random.Generator
) -> ShotEstimates:
    """Draw ``n_shots`` Z-basis bitstrings of each state and estimate from them.

    Args:
        probs: ``|psi|^2`` of each state, shape ``(n, 2**L)``.
        n_shots: Bitstrings per state, at least 2.
        rng: Random generator.

    Returns:
        The :class:`ShotEstimates`.

    Raises:
        ValueError: If ``n_shots < 2``.
    """
    if n_shots < 2:
        raise ValueError(f"n_shots must be >= 2, got {n_shots}.")
    probs = np.asarray(probs, dtype=float)
    n, dim = probs.shape
    L = dim.bit_length() - 1
    cdf = np.cumsum(probs, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random((n, n_shots))
    idx = np.minimum(
        np.stack([np.searchsorted(cdf[k], u[k], side="right") for k in range(n)]),
        dim - 1,
    )
    # Empirical distribution of each state: the features are its moments.
    counts = np.zeros((n, dim))
    np.add.at(counts, (np.repeat(np.arange(n), n_shots), idx.ravel()), 1.0)
    empirical = counts / n_shots
    z, zz = exact_features(empirical)
    Q = site_bits(L).sum(axis=1)[idx].astype(float)
    return ShotEstimates(z=z, zz=zz, var_direct=np.var(Q, axis=1, ddof=1))


@dataclass(frozen=True)
class Trajectories:
    """Final states of monitored U(1) trajectories, as Born distributions.

    Attributes:
        probs: ``|psi|^2`` of the final state of each trajectory, shape
            ``(n, 2**L)``. Every feature of :func:`exact_features` and the
            charge moments are functions of it.
        outcomes: Record of each trajectory, shape ``(n, rounds, L)``, 0 or
            1 for a measured site, ``UNMEASURED`` (-1) otherwise.
        n_measured: Number of measurements of each trajectory, ``(n,)``.
        leakage: Largest weight outside the starting sectors over the
            evolution, ``(n,)``; exactly 0 when the charge is conserved
            (issue #134).
        charges: The two starting charges.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.
    """

    probs: np.ndarray
    outcomes: np.ndarray
    n_measured: np.ndarray
    leakage: np.ndarray
    charges: tuple[int, ...]
    boundary: str = "open"
    measure_every: str = "layer"


def _outside_weight(probs: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    return np.asarray(probs[:, ~allowed].sum(axis=1))


def sample_trajectories(
    L: int,
    p: float,
    n: int,
    *,
    depth: int | None = None,
    charges: Sequence[int] | None = None,
    boundary: str = "open",
    measure_every: str = "layer",
    rng: np.random.Generator | None = None,
) -> Trajectories:
    """Simulate ``n`` trajectories from :func:`superposition_state`.

    Same circuit and measurements as
    :func:`QELM.evaluation.learnability.sample_records` (a new random U(1)
    brickwork per trajectory, Born-rule outcomes), but every trajectory
    starts in the superposition of the two charges and the final state is
    kept.

    Args:
        L: Number of sites, at least 2.
        p: Measurement probability per site and per round, in ``[0, 1]``.
        n: Number of trajectories, at least 1.
        depth: Time steps ``T``; ``None`` for ``T = L``.
        charges: The two charges; ``None`` for ``(L // 2, L // 2 - 1)``.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.
        rng: Random generator; ``None`` draws one from the global NumPy state.

    Returns:
        The :class:`Trajectories`.

    Raises:
        ValueError: If an argument is out of range.
    """
    if L < 2:
        raise ValueError(f"Need L >= 2, got {L}.")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    T = L if depth is None else int(depth)
    if T < 0:
        raise ValueError(f"depth must be >= 0, got {T}.")
    qs = tuple(default_charges(L) if charges is None else (int(q) for q in charges))
    if len(qs) != 2 or qs[0] == qs[1]:
        raise ValueError(f"Need two distinct charges, got {qs}.")
    rounds = measurement_rounds(L, T, measure_every, boundary)
    gen = _generator(rng)
    bits = site_bits(L)
    allowed = np.isin(bits.sum(axis=1), qs)

    psi = np.tile(superposition_state(L, qs).astype(complex), (n, 1))
    outcomes = np.full((n, len(rounds), L), UNMEASURED, dtype=np.int8)
    leakage = _outside_weight(np.abs(psi) ** 2, allowed)
    for t, bonds in enumerate(rounds):
        for i, j in bonds:
            psi = apply_gates(psi, random_u1_gates(n, gen), i, j)
        measured = gen.random((n, L)) < p
        draws = gen.random((n, L))
        for site in range(L):
            rows = np.flatnonzero(measured[:, site])
            if rows.size == 0:
                continue
            weight = np.abs(psi[rows]) ** 2
            p_one = weight[:, bits[:, site] == 1].sum(axis=1)
            outcome = (draws[rows, site] < p_one).astype(np.int8)
            keep = bits[None, :, site] == outcome[:, None]
            kept = psi[rows] * keep
            kept /= np.linalg.norm(kept, axis=1, keepdims=True)
            psi[rows] = kept
            outcomes[rows, t, site] = outcome
        leakage = np.maximum(leakage, _outside_weight(np.abs(psi) ** 2, allowed))
    probs = np.abs(psi) ** 2
    return Trajectories(
        probs=probs / probs.sum(axis=1, keepdims=True),
        outcomes=outcomes,
        n_measured=(outcomes != UNMEASURED).sum(axis=(1, 2)),
        leakage=leakage,
        charges=qs,
        boundary=boundary,
        measure_every=measure_every,
    )


def sharpening_ratio(var: np.ndarray) -> float:
    """Return ``1 - <Var^2> / (3 <Var>^2)``, a Binder-like ratio of ``Var(Q)``.

    It is 2/3 when every trajectory has the same variance (deep in the
    fuzzy phase, or ``p = 0``), and decreases as the distribution of
    ``Var(Q)`` over trajectories becomes broad (part of them sharp, part
    fuzzy). ``nan`` when every variance is 0.
    """
    v = np.asarray(var, dtype=float)
    m1 = float(np.mean(v))
    if m1 <= 0:
        return float("nan")
    return float(1.0 - np.mean(v**2) / (3.0 * m1**2))
