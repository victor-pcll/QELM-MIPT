"""Fourier expressivity of the monitored QELM (issue #309).

The QELM encodes the input once, as a product of single-qubit states with
Bloch vectors in the x-z plane (:class:`QELM.encoding.AngleEncoding`):

    rho(theta) = prod_i (I + sin(theta_i) X_i + cos(theta_i) Z_i) / 2.

Every mean the readout can see, ``E[g(record) | theta]``, is linear in
``rho(theta)``, hence a combination of the ``3**N`` Bloch monomials

    B_alpha(theta) = prod_{alpha_i = x} sin(theta_i) prod_{alpha_i = z} cos(theta_i),

with ``alpha`` in ``{0, x, z}**N``: a trigonometric polynomial of degree at
most one in each angle, whatever the measurements, ``p`` or the symmetry of
the gates. These only set the coefficients. This module computes them for
the features of the QELM on the measurement record ``M``:

- ``M_{t,i}``: the Z outcome of site ``i`` after brickwork layer ``t``, ``+-1``
  if measured (probability ``p``), ``0`` otherwise; its mean over records and
  masks is ``p Tr[Z_i Phi_t(rho)]``, with ``Phi_t`` the layers interleaved
  with the Z-dephasing of probability ``p`` on every site (the average over
  the masks and outcomes of the earlier measurements);
- the final ``Z_i`` read after the last layer, mean ``Tr[Z_i Phi_T(rho)]``.

Each feature mean is ``Tr[O_k rho(theta)]`` with ``O_k`` the Heisenberg
image of ``Z_i``, so its coefficient on ``B_alpha`` is ``Tr[O_k
sigma_alpha] / 2**N``: the rows of the transfer matrix ``C(p)``. The
measures below read ``C(p)``: its rank (the dimension of the accessible
function space), the spread of its weight over the degrees ``|alpha|`` (the
number of angles a monomial involves, the Pauli weight of ``sigma_alpha``),
and the covariance of the features under a distribution of the inputs.

Monomial order: ``alpha`` is read as a base-3 number with site 0 the most
significant digit, digits ``0, 1, 2`` for ``I, X, Z``, the same order as
:func:`bloch_monomials`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from QELM.models.circuits.base import Layer, apply_two_qubit_gate

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
# The three Pauli matrices of the x-z plane, indexed by the digits of alpha.
PAULI_XZ = np.stack([_I, _X, _Z])


def bloch_monomials(theta: np.ndarray) -> np.ndarray:
    """Return the ``3**N`` Bloch monomials ``B_alpha(theta)``.

    Args:
        theta: Angles of shape ``(N,)`` or ``(n, N)``.

    Returns:
        Shape ``(3**N,)`` or ``(n, 3**N)``, in the order of the module.
    """
    theta = np.asarray(theta, dtype=float)
    single = theta.ndim == 1
    theta = np.atleast_2d(theta)
    n, N = theta.shape
    out = np.ones((n, 1))
    for i in range(N):
        site = np.stack([np.ones(n), np.sin(theta[:, i]), np.cos(theta[:, i])], axis=1)
        out = (out[:, :, None] * site[:, None, :]).reshape(n, -1)
    return out[0] if single else out


def monomial_degrees(N: int) -> np.ndarray:
    """Return ``|alpha|``, the number of non-identity sites, per monomial."""
    digits = np.array(np.unravel_index(np.arange(3**N), (3,) * N))
    return np.asarray((digits > 0).sum(axis=0), dtype=int)


def odd_x_mask(N: int) -> np.ndarray:
    """Return ``True`` for the monomials with an odd number of ``sin`` factors.

    A Z-diagonal symmetry (the parity ``prod_i Z_i``, or the charge
    ``sum_i Z_i``) flips every ``sin(theta_i)`` at once, so these monomials
    are out of reach of circuits and measurements that respect it.
    """
    digits = np.array(np.unravel_index(np.arange(3**N), (3,) * N))
    return np.asarray((digits == 1).sum(axis=0) % 2 == 1)


def pauli_xz_coefficients(obs: np.ndarray) -> np.ndarray:
    """Return ``Tr[obs sigma_alpha] / 2**N`` for every ``alpha`` in ``{I, X, Z}**N``.

    Args:
        obs: Hermitian operator of shape ``(2**N, 2**N)``, qubit 0 the most
            significant bit.

    Returns:
        Real array of shape ``(3**N,)``: ``Tr[obs rho(theta)] = sum_alpha c_alpha
        B_alpha(theta)``.
    """
    dim = obs.shape[0]
    N = dim.bit_length() - 1
    if obs.shape != (dim, dim) or 2**N != dim:
        raise ValueError(f"obs must be (2**N, 2**N), got shape {obs.shape}.")
    T = np.asarray(obs, dtype=complex).reshape((2,) * (2 * N))
    # Axes of T: rows of the remaining sites, then their columns, then the
    # Pauli indices of the sites already contracted.
    for i in range(N):
        remaining = N - i
        # Rows of site i are axis 0, its columns axis `remaining`.
        T = np.tensordot(PAULI_XZ, T, axes=([2, 1], [0, remaining]))
        T = np.moveaxis(T, 0, -1)
    return np.asarray(T.real.reshape(-1) / dim)


def _site_operator(op: np.ndarray, site: int, N: int) -> np.ndarray:
    return np.asarray(
        np.kron(np.kron(np.eye(2**site), op), np.eye(2 ** (N - site - 1)))
    )


def heisenberg_layer(obs: np.ndarray, layer: Layer) -> np.ndarray:
    """Return ``U^dagger obs U`` for the layer ``U = g_n ... g_1``."""
    for i, j, gate in reversed(layer):
        obs = apply_two_qubit_gate(obs, gate.conj().T, i, j)
    return obs


def dephase(obs: np.ndarray, p: float) -> np.ndarray:
    """Z-dephase every site with probability ``p`` (self-adjoint channel).

    The entries of ``obs`` whose row and column bits differ on a site are
    multiplied by ``1 - p`` for that site.
    """
    if p == 0.0:
        return obs
    N = obs.shape[0].bit_length() - 1
    return np.asarray(obs * (1.0 - p) ** _bit_flips(N))


@lru_cache(maxsize=8)
def _bit_flips(N: int) -> np.ndarray:
    """Return the number of sites where the row and column bits differ."""
    dim = 2**N
    bits = (np.arange(dim)[:, None] >> np.arange(N - 1, -1, -1)) & 1
    return np.asarray((bits[:, None, :] != bits[None, :, :]).sum(axis=2))


def transfer_matrix(
    layers: Sequence[Layer],
    N: int,
    p: float,
    final: bool = True,
) -> np.ndarray:
    """Return the transfer matrix ``C(p)`` of the record features.

    Rows: ``M_{t,i}`` for ``t = 0..T-1``, ``i = 0..N-1`` (row ``t N + i``),
    then, if ``final``, the final ``Z_i``. Columns: the ``3**N`` monomials.

    Args:
        layers: The ``T`` brickwork layers, applied in order, each followed
            by the Z measurements of probability ``p``. The gates act on
            qubits ``0..N-1`` (no ancilla). At least one layer.
        N: Number of qubits.
        p: Measurement probability.
        final: Add the ``N`` rows of the final Z readout.

    Returns:
        Array of shape ``(T N + N, 3**N)`` (``(T N, 3**N)`` without
        ``final``).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}.")
    T = len(layers)
    if T < 1:
        raise ValueError("Need at least one layer.")
    rows: list[np.ndarray] = []
    for t in range(T):
        for i in range(N):
            obs = heisenberg_layer(_site_operator(_Z, i, N).astype(complex), layers[t])
            for s in range(t - 1, -1, -1):
                obs = heisenberg_layer(dephase(obs, p), layers[s])
            rows.append(pauli_xz_coefficients(obs))
    C = np.array(rows).reshape(T * N, 3**N)
    if final:
        # The final Z_i is M_{T-1,i} without the factor p.
        return np.vstack([p * C, C[(T - 1) * N :]])
    return p * C


def accessible_rank(C: np.ndarray, rtol: float = 1e-8) -> int:
    """Return the rank of ``C``: the dimension of the accessible functions."""
    if C.size == 0:
        return 0
    s = np.linalg.svd(C, compute_uv=False)
    if s[0] == 0.0:
        return 0
    return int((s > rtol * s[0]).sum())


def torus_weights(N: int) -> np.ndarray:
    """Return ``||B_alpha||^2 = 2**-|alpha|`` for uniform angles on the circle."""
    return np.asarray(2.0 ** -monomial_degrees(N))


def degree_profile(C: np.ndarray, N: int) -> np.ndarray:
    """Return ``W_k``, the share of the non-constant weight at degree ``k``.

    The weight of monomial ``alpha`` is ``sum_k C_{k alpha}^2 ||B_alpha||^2``
    with the norm of uniform angles on the circle, where the monomials are
    orthogonal. Entry 0 is 0; the others sum to 1 (all 0 if ``C`` only
    holds constants).
    """
    deg = monomial_degrees(N)
    weight = (C**2).sum(axis=0) * torus_weights(N)
    W = np.array([weight[deg == k].sum() for k in range(N + 1)])
    W[0] = 0.0
    total = W.sum()
    return W / total if total > 0 else W


def mean_degree(W: np.ndarray) -> float:
    """Return ``sum_k k W_k / N`` for a profile of :func:`degree_profile`."""
    N = len(W) - 1
    return float((np.arange(N + 1) * W).sum() / N)


def degree_accessibility(C: np.ndarray, N: int, rtol: float = 1e-8) -> np.ndarray:
    """Return ``A_k``, the share of the degree-``k`` functions in reach.

    ``A_k`` is the mean over the monomials of degree ``k`` (normalized on the
    circle) of the squared norm of their projection on the row space of
    ``C``: 1 when every degree-``k`` monomial is a combination of features, 0
    when none is touched.
    """
    deg = monomial_degrees(N)
    Cn = C * np.sqrt(torus_weights(N))
    _, s, Vh = np.linalg.svd(Cn, full_matrices=False)
    keep = s > rtol * s[0] if s.size and s[0] > 0 else np.zeros_like(s, dtype=bool)
    proj = (Vh[keep] ** 2).sum(axis=0)
    return np.array([proj[deg == k].mean() for k in range(N + 1)])


def odd_x_weight(C: np.ndarray, N: int) -> float:
    """Return the share of the weight of ``C`` on the odd-``sin`` monomials."""
    weight = (C**2).sum(axis=0) * torus_weights(N)
    total = weight.sum()
    return float(weight[odd_x_mask(N)].sum() / total) if total > 0 else 0.0


def angle_moments(
    low: float = 0.0, high: float = np.pi, nodes: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean ``m`` and second moments ``G`` of ``(1, sin, cos)``.

    For one angle uniform on ``[low, high]``: ``m_a = E[f_a]``, ``G_ab =
    E[f_a f_b]``, by Gauss-Legendre quadrature (exact for these functions to
    machine precision).
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    theta = 0.5 * (high - low) * x + 0.5 * (high + low)
    w = w / w.sum()
    f = np.stack([np.ones_like(theta), np.sin(theta), np.cos(theta)])
    return f @ w, (f * w) @ f.T


def _contract_sites(C: np.ndarray, N: int, M: np.ndarray) -> np.ndarray:
    """Apply the 3 x 3 (or 3,) site matrix ``M`` on every site axis of ``C``."""
    K = C.shape[0]
    T = C.reshape((K,) + (3,) * N)
    for i in range(N):
        T = np.moveaxis(np.tensordot(T, M, axes=([1 + i], [0])), -1, 1 + i)
    return T.reshape(K, -1)


def feature_covariance(
    C: np.ndarray, N: int, low: float = 0.0, high: float = np.pi
) -> np.ndarray:
    """Return the covariance of the feature means for i.i.d. uniform angles.

    ``Sigma = C G C^T - (C m)(C m)^T``, with ``G = tensor_i G_i`` and ``m =
    tensor_i m_i`` from :func:`angle_moments`, contracted site by site.
    """
    m, G = angle_moments(low, high)
    CG = _contract_sites(C, N, G)
    K = C.shape[0]
    mean = C.reshape((K,) + (3,) * N)
    for _ in range(N):
        mean = np.tensordot(mean, m, axes=([1], [0]))
    mean = mean.reshape(K)
    return np.asarray(CG @ C.T - np.outer(mean, mean))


def effective_dimension(Sigma: np.ndarray) -> float:
    """Return the participation ratio ``(tr Sigma)^2 / tr Sigma^2``."""
    tr = float(np.trace(Sigma))
    tr2 = float(np.sum(Sigma * Sigma.T))
    return tr * tr / tr2 if tr2 > 0 else 0.0


def expressivity_measures(
    C: np.ndarray, N: int, low: float = 0.0, high: float = np.pi
) -> dict[str, float | list[float]]:
    """Return every measure of one transfer matrix, JSON-ready.

    ``C`` is laid out as :func:`transfer_matrix` with ``final=True``: ``T``
    blocks of ``N`` record rows, then ``N`` final rows. Besides the measures
    of the whole matrix, ``final_*`` are those of the final ``Z_i`` alone and
    ``mean_degree_by_time`` is the mean degree of each block of record rows
    (``nan`` for an all-zero block, at ``p = 0``).
    """
    W = degree_profile(C, N)
    Sigma = feature_covariance(C, N, low, high)
    final = C[-N:]
    W_final = degree_profile(final, N)
    T = C.shape[0] // N - 1
    by_time = []
    for t in range(T):
        block = C[t * N : (t + 1) * N]
        by_time.append(
            mean_degree(degree_profile(block, N)) if np.any(block) else float("nan")
        )
    return {
        "final_rank": accessible_rank(final),
        "final_degree_profile": W_final.tolist(),
        "final_mean_degree": mean_degree(W_final),
        "mean_degree_by_time": by_time,
        "rank": accessible_rank(C),
        "rank_fraction": accessible_rank(C) / 3**N,
        "degree_profile": W.tolist(),
        "mean_degree": mean_degree(W),
        "degree_accessibility": degree_accessibility(C, N).tolist(),
        "odd_x_weight": odd_x_weight(C, N),
        "d_eff": effective_dimension(Sigma),
        "feature_variance": float(np.trace(Sigma)),
    }
