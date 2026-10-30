"""Rényi and von Neumann entanglement entropies of a subsystem ``A``.

For a state ``rho`` on ``N`` qubits and a set of qubits ``A`` (``sector_A``,
all ``N`` qubits when omitted), with ``rho_A = Tr_B(rho)`` and ``l_i`` the
eigenvalues of ``rho_A``:

- Rényi entropy of order ``n``: ``S_n = log(sum_i l_i**n) / (1 - n)``;
- ``n = 0``, Hartley entropy: ``S_0 = log(rank rho_A)``;
- ``n = 1``, von Neumann entropy: ``S_1 = -sum_i l_i log(l_i)``, the limit
  of ``S_n`` when ``n -> 1``;
- ``n = 2``, collision entropy: ``S_2 = -log Tr(rho_A**2)``, minus the log
  of the purity;
- ``n = inf``, min-entropy: ``S_inf = -log(max_i l_i)``.

``S_n`` decreases with ``n``. For a pure global state, ``S_n(A) = S_n(B)``.
These are the diagnostics of the measurement-induced phase transition: a
volume law (``S_n`` growing with ``|A|``) at low measurement rate, an area
law (``S_n`` saturating) at high rate (issue #93).

Logarithms are in base 2 by default (bits), as in
:func:`QELM.evaluation.metrics.von_neumann_entropy`; pass ``base=np.e`` for
nats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from QELM.physics.partial_trace import (
    as_vector,
    check_sites,
    partial_trace,
    state_qubits,
    vector_matrix,
)


def _check_base(base: float) -> float:
    if not (base > 0 and base != 1):
        raise ValueError(f"base must be positive and different from 1, got {base}.")
    return float(base)


def entanglement_spectrum(
    state: np.ndarray | sp.spmatrix,
    sector_A: Sequence[int] | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the eigenvalues of ``rho_A`` above ``eps``, in decreasing order.

    For a state vector the values are the squared singular values of the
    ``2**|A| x 2**|B|`` matrix of amplitudes (Schmidt decomposition), so
    neither ``rho`` nor ``rho_A`` is built. For a density matrix they are the
    eigenvalues of :func:`~QELM.physics.partial_trace.partial_trace`.

    Args:
        state: State vector ``(2**N,)``, sparse column ``(2**N, 1)``, or
            density matrix ``(2**N, 2**N)``, dense or sparse.
        sector_A: Qubits of ``A``, distinct, each in ``0 .. N-1``. Empty for
            ``A = {}``, whose only eigenvalue is ``Tr rho``. ``None``
            (default) for the whole system, ``A = {0, ..., N-1}``, with
            ``N`` read from the shape of ``state``: the spectrum is then
            that of ``rho`` itself (issue #116).
        eps: Eigenvalues ``<= eps`` are dropped, so that rounding noise such
            as ``-1e-16`` never reaches a logarithm.

    Returns:
        1-D float array, possibly empty for a zero state.

    Raises:
        ValueError: If the shape of ``state`` or ``sector_A`` is invalid.
    """
    N = state_qubits(state)
    if sector_A is None:
        sector_A = range(N)
    keep = check_sites(sector_A, N, "sector_A")
    psi = as_vector(state)
    if psi is not None:
        M = vector_matrix(psi.astype(complex), keep, N)
        values = np.linalg.svd(M, compute_uv=False) ** 2
    else:
        values = np.linalg.eigvalsh(partial_trace(state, keep))
    values = np.sort(np.real(values))[::-1]
    return values[values > eps]


def von_neumann_entropy(
    state: np.ndarray | sp.spmatrix,
    sector_A: Sequence[int] | None = None,
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Return ``S_1(rho_A) = -sum_i l_i log(l_i)``.

    Args:
        state: See :func:`entanglement_spectrum`.
        sector_A: Qubits of ``A``; ``None`` (default) for all the qubits,
            the entropy of the whole state (0 for a pure state).
        base: Logarithm base: 2 for bits (default), ``np.e`` for nats.
        eps: Eigenvalues ``<= eps`` are dropped.

    Returns:
        The entropy, between 0 and ``|A| log_base(2)``.

    Raises:
        ValueError: If ``state``, ``sector_A`` or ``base`` is invalid.
    """
    base = _check_base(base)
    values = entanglement_spectrum(state, sector_A, eps)
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)) / math.log(base))


def renyi_entropy(
    state: np.ndarray | sp.spmatrix,
    n: float,
    sector_A: Sequence[int] | None = None,
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Return the Rényi entropy ``S_n(rho_A) = log(sum_i l_i**n) / (1 - n)``.

    ``n = 0`` gives the Hartley entropy ``log(rank rho_A)`` (rank counted
    above ``eps``), ``n = 1`` the von Neumann entropy (routed to
    :func:`von_neumann_entropy`), ``n = 2`` ``-log Tr(rho_A**2)`` and
    ``n = inf`` the min-entropy ``-log(l_max)``. Any other ``n >= 0`` uses
    the general formula; it tends to the von Neumann entropy as ``n -> 1``.

    Args:
        state: See :func:`entanglement_spectrum`.
        n: Order, a number ``>= 0`` or ``float("inf")``.
        sector_A: Qubits of ``A``; ``None`` (default) for all the qubits,
            the entropy of the whole state (0 for a pure state).
        base: Logarithm base: 2 for bits (default), ``np.e`` for nats.
        eps: Eigenvalues ``<= eps`` are dropped.

    Returns:
        ``S_n(rho_A)``, between 0 and ``|A| log_base(2)``.

    Raises:
        ValueError: If ``n`` is negative or not a number, or if ``state``,
            ``sector_A`` or ``base`` is invalid.
    """
    n = float(n)
    if math.isnan(n) or n < 0:
        raise ValueError(f"n must be a number >= 0 (or inf), got {n}.")
    if n == 1.0:
        return von_neumann_entropy(state, sector_A, base, eps)
    log_base = math.log(_check_base(base))
    values = entanglement_spectrum(state, sector_A, eps)
    if values.size == 0:
        return 0.0
    if n == 0.0:
        return math.log(values.size) / log_base
    if math.isinf(n):
        return float(-np.log(values[0]) / log_base)
    return float(np.log(np.sum(values**n)) / ((1.0 - n) * log_base))


def mean_single_qubit_entropy(
    state: np.ndarray | sp.spmatrix,
    sites: Sequence[int],
    n: float = 1.0,
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Return the mean over ``k`` in ``sites`` of ``S_n(rho_k)``.

    ``rho_k`` is the 2 x 2 reduced state of qubit ``k`` alone. Each term is
    between 0 (qubit ``k`` in a pure state, a product with the rest) and
    ``log_base(2)`` (qubit ``k`` maximally mixed), so the mean is too. Unlike
    the entropy of the whole sector, it does not see the correlations
    between the qubits of ``sites``: it is the local entanglement of each
    qubit with everything else.

    Args:
        state: See :func:`entanglement_spectrum`.
        sites: Qubits to average over, distinct, at least one.
        n: Rényi order, as in :func:`renyi_entropy` (1 for von Neumann).
        base: Logarithm base: 2 for bits (default), ``np.e`` for nats.
        eps: Eigenvalues ``<= eps`` are dropped.

    Returns:
        ``(1 / |sites|) sum_k S_n(rho_k)``.

    Raises:
        ValueError: If ``sites`` is empty, or if ``state``, a site, ``n`` or
            ``base`` is invalid.
    """
    sites = check_sites(sites, state_qubits(state), "sites")
    if not sites:
        raise ValueError("sites must hold at least one qubit.")
    return float(np.mean([renyi_entropy(state, n, [k], base, eps) for k in sites]))


def tripartite_mutual_information(
    state: np.ndarray | sp.spmatrix,
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Return ``I_3(A:B:C)`` of the four contiguous quarters of the chain.

    The qubits ``0 .. N-1`` are cut into ``A, B, C, D`` of ``N / 4`` qubits
    each, in order, and ``I_3 = S_A + S_B + S_C - S_AB - S_AC - S_BC +
    S_ABC``, with von Neumann entropies. For a pure state ``S_ABC = S_D``.
    ``I_3`` is negative in the volume-law phase of a monitored circuit and
    tends to 0 in the area-law phase; its curves for several ``N`` cross
    near ``p_c`` with small finite-size drift, so the crossing locates the
    transition without fitting an exponent (Zabalo et al., PRB 101, 060301
    (2020)). Issue #251.

    Args:
        state: State or density matrix on ``N`` qubits, ``N`` a multiple of
            4; see :func:`entanglement_spectrum`.
        base: Logarithm base: 2 for bits (default), ``np.e`` for nats.
        eps: Eigenvalues ``<= eps`` are dropped.

    Returns:
        ``I_3``, in units of ``log_base(2)``.

    Raises:
        ValueError: If ``N`` is not a positive multiple of 4, or if
            ``state`` or ``base`` is invalid.
    """
    N = state_qubits(state)
    if N < 4 or N % 4:
        raise ValueError(f"I_3 of four quarters needs N a multiple of 4, got N={N}.")
    q = N // 4
    A, B, C, D = (list(range(k * q, (k + 1) * q)) for k in range(4))

    def S(sites: list[int]) -> float:
        return von_neumann_entropy(state, sites, base, eps)

    return S(A) + S(B) + S(C) - S(A + B) - S(A + C) - S(B + C) + S(A + B + C)
