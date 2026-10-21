"""Global charge of a qubit state, and checks that it is conserved (issue #134).

The charge of a computational basis state ``|b_0 ... b_{N-1}>`` is its number
of 1 bits, ``Q = sum_i b_i`` (optionally restricted to a subset of sites). The
charge operator ``Q = sum_i (1 - Z_i) / 2`` is diagonal in that basis, so the
probability of each charge value is a sum of ``|psi_x|**2`` (or of ``rho_xx``)
over the basis states ``x`` of that charge.

A U(1)-symmetric dynamics (the gates of :mod:`QELM.models.circuits`, Z-basis
measurements, the weak measurement of
:mod:`QELM.training.measurements.weak`) keeps a state of sharp charge ``Q``
inside the sector ``Q``. :func:`charge_leakage` measures how much weight
left that sector: 0 when the charge is conserved, 1 when the whole state
moved out. Because the gates have exact zeros outside their charge blocks,
the leakage of a conserving run is exactly 0 in floating point, not just
small; any positive value points to a bug or to a non-conserving step.

Site 0 is the most significant bit of a basis index, as in
:func:`QELM.physics.operators.get_site_operator`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from QELM.physics.partial_trace import check_sites, state_qubits


def charge_values(N: int, sites: Sequence[int] | None = None) -> np.ndarray:
    """Return the charge of every basis state of ``N`` qubits.

    Args:
        N: Number of qubits, at least 1.
        sites: Sites that carry the charge; ``None`` (default) for all ``N``.

    Returns:
        Integer array of shape ``(2**N,)``: entry ``x`` is the number of 1
        bits of basis state ``x`` on ``sites``.

    Raises:
        ValueError: If ``N < 1`` or a site is out of range or repeated.
    """
    if N < 1:
        raise ValueError(f"Need N >= 1, got {N}.")
    chosen = list(range(N)) if sites is None else check_sites(sites, N)
    x = np.arange(2**N)[:, None]
    shifts = N - 1 - np.asarray(chosen, dtype=np.int64)[None, :]
    return np.asarray(((x >> shifts) & 1).sum(axis=1), dtype=np.int64)


def _populations(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the basis-state weights ``|psi_x|**2`` (or ``rho_xx``) and ``N``.

    Raises:
        ValueError: If ``state`` is not a dense vector, column or density
            matrix of ``N`` qubits, or has zero norm.
    """
    state = np.asarray(state)
    N = state_qubits(state)
    if state.ndim == 2 and state.shape[1] == state.shape[0]:
        weights = np.real(np.diagonal(state)).astype(float)
    else:
        weights = np.abs(state.reshape(-1)) ** 2
    if not np.sum(weights) > 0:
        raise ValueError("state has zero norm.")
    return weights, N


def charge_distribution(
    state: np.ndarray, sites: Sequence[int] | None = None
) -> np.ndarray:
    """Return the probability ``P(Q)`` of each charge value.

    Args:
        state: State vector ``(2**N,)``, column ``(2**N, 1)`` or density
            matrix ``(2**N, 2**N)``, dense. It need not be normalized.
        sites: Sites that carry the charge; ``None`` for all of them.

    Returns:
        Array of shape ``(n_sites + 1,)`` that sums to 1; entry ``Q`` is the
        probability of charge ``Q``.
    """
    weights, N = _populations(state)
    q = charge_values(N, sites)
    n_sites = N if sites is None else len(sites)
    dist = np.bincount(q, weights=weights, minlength=n_sites + 1)
    return np.asarray(dist / np.sum(weights))


def total_charge(state: np.ndarray, sites: Sequence[int] | None = None) -> float:
    """Return the mean charge ``<Q> = sum_Q Q P(Q)`` of ``state``.

    Args:
        state: As in :func:`charge_distribution`.
        sites: Sites that carry the charge; ``None`` for all of them.

    Returns:
        ``<Q>``, between 0 and the number of sites.
    """
    dist = charge_distribution(state, sites)
    return float(np.dot(np.arange(dist.size), dist))


def sharp_charge(
    state: np.ndarray, sites: Sequence[int] | None = None, atol: float = 1e-10
) -> int:
    """Return the charge ``Q`` of a state that lies in a single sector.

    Args:
        state: As in :func:`charge_distribution`.
        sites: Sites that carry the charge; ``None`` for all of them.
        atol: Largest weight allowed outside the most likely sector.

    Returns:
        The charge of the sector that holds the state.

    Raises:
        ValueError: If more than ``atol`` of the weight lies outside that
            sector: the charge is not sharp, so there is nothing to conserve.
    """
    dist = charge_distribution(state, sites)
    Q = int(np.argmax(dist))
    if 1.0 - dist[Q] > atol:
        raise ValueError(
            f"The state has no sharp charge: P(Q={Q}) = {dist[Q]:.6g} < 1 - {atol}."
        )
    return Q


def charge_leakage(
    state: np.ndarray, Q: int, sites: Sequence[int] | None = None
) -> float:
    """Return the weight of ``state`` outside the charge sector ``Q``.

    The weight is summed over the basis states of charge different from
    ``Q`` (rather than computed as ``1 - P(Q)``), so that a state exactly in
    the sector gives exactly 0, with no rounding.

    Args:
        state: As in :func:`charge_distribution`.
        Q: Expected charge.
        sites: Sites that carry the charge; ``None`` for all of them.

    Returns:
        ``1 - P(Q)``, in ``[0, 1]``: 0 when the charge is ``Q``.
    """
    weights, N = _populations(state)
    outside = charge_values(N, sites) != int(Q)
    return float(np.sum(weights[outside]) / np.sum(weights))


def batch_charge_leakage(
    psi: np.ndarray, Q: np.ndarray, sites: Sequence[int] | None = None
) -> np.ndarray:
    """Return :func:`charge_leakage` for a batch of state vectors at once.

    Args:
        psi: State vectors, shape ``(n, 2**N)``.
        Q: Expected charge of each state, shape ``(n,)``.
        sites: Sites that carry the charge; ``None`` for all of them.

    Returns:
        Leakage of each state, shape ``(n,)``.

    Raises:
        ValueError: If the shapes do not match, or a state has zero norm.
    """
    psi = np.asarray(psi)
    Q = np.asarray(Q)
    if psi.ndim != 2 or Q.shape != (psi.shape[0],):
        raise ValueError(
            f"Need psi of shape (n, 2**N) and Q of shape (n,), got {psi.shape} "
            f"and {Q.shape}."
        )
    N = state_qubits(psi[0]) if psi.shape[0] else psi.shape[1].bit_length() - 1
    weights = np.abs(psi) ** 2
    norm = weights.sum(axis=1)
    if np.any(norm <= 0):
        raise ValueError("A state of the batch has zero norm.")
    outside = charge_values(N, sites)[None, :] != Q[:, None]
    return np.asarray((weights * outside).sum(axis=1) / norm)
