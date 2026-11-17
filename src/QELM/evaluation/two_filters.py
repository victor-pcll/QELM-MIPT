"""Two-filter decomposition of the ridge error of a Clifford QELM (issue #349).

Claim under test: the incompressible error of the ridge splits as

    E_perp = E_perp,circuit + E_perp,readout(k),

with ``E_perp,circuit`` what the circuit loses (the target is not in
``Span{E_m} = Span G``) and ``E_perp,readout(k)`` what a readout by parities of
at most ``k`` bits misses inside ``Span G``.

Setting (as in #337 and #341): inputs are products of Pauli eigenstates
``|0>, |1>, |+>, |->``, drawn uniformly, so the population is the ``4^N``
inputs with equal weight. A feature is the exact mean ``Tr[rho(x) O]`` of a
record parity, ``O`` one Pauli string (:func:`parity_strings`). Its value on a
product input is ``prod_i b_i`` with ``b_i`` the Bloch component of site ``i``
along ``O_i`` (``0`` if ``O_i = Y``), so features are integers in
``{-1, 0, 1}`` (up to a sign per column, which no ridge sees).

The decomposition is an identity of orthogonal projections: if
``F ⊂ S ⊂ L^2(population)`` and ``P_F``, ``P_S`` are the orthogonal projectors
of the same inner product, then ``f - P_F f = (f - P_S f) + (P_S f - P_F f)``,
the two terms are orthogonal, and Pythagoras gives the sum. This module
computes each term separately, so that the identity, and the size of the
cross term when it fails (``lambda > 0``, finite training set, finite shots),
are measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from QELM.evaluation.clifford_ridge import bloch_components


def all_codes(N: int) -> np.ndarray:
    """All ``4^N`` product inputs, codes ``0-3`` per site, shape ``(4^N, N)``."""
    idx = np.arange(4**N)
    return np.asarray((idx[:, None] // 4 ** np.arange(N)[None, :]) % 4)


def string_values(codes: np.ndarray, strings: np.ndarray) -> np.ndarray:
    """``Tr[rho(x) g]`` for each input and each ``[x | z]`` string ``g``.

    Args:
        codes: Input codes, shape ``(S, N)``.
        strings: ``[x | z]`` rows, shape ``(d, 2 N)``.

    Returns:
        Integer array ``(S, d)`` with entries in ``{-1, 0, 1}``.
    """
    codes = np.asarray(codes)
    s = np.asarray(strings, dtype=bool)
    n = codes.shape[1]
    bx, bz = bloch_components(codes)
    gx, gz = s[:, :n], s[:, n:]
    # Per site: X -> bx, Z -> bz, Y -> 0, I -> 1.
    out = np.ones((codes.shape[0], s.shape[0]), dtype=np.int64)
    for q in range(n):
        col = np.ones((codes.shape[0], s.shape[0]), dtype=np.int64)
        only_x = gx[:, q] & ~gz[:, q]
        only_z = gz[:, q] & ~gx[:, q]
        both = gx[:, q] & gz[:, q]
        col[:, only_x] = bx[:, q, None].astype(np.int64)
        col[:, only_z] = bz[:, q, None].astype(np.int64)
        col[:, both] = 0
        out *= col
    return out


def visible(strings: np.ndarray) -> np.ndarray:
    """Mask of the strings without ``Y`` and not the identity."""
    s = np.asarray(strings, dtype=bool)
    n = s.shape[1] // 2
    x, z = s[:, :n], s[:, n:]
    return np.asarray(~(x & z).any(axis=1) & (x | z).any(axis=1))


def ridge_fit(phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge predictions ``phi (phi^T phi / S + lam)^-1 phi^T y / S`` on the same rows.

    ``lam = 0`` is the orthogonal projection (pseudo-inverse). Rows carry
    equal weight, so on the full population this is the population ridge.
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    if phi.shape[1] == 0:
        return np.zeros_like(y)
    s = phi.shape[0]
    gram = phi.T @ phi / s
    rhs = phi.T @ y / s
    if lam == 0:
        coef = np.linalg.lstsq(phi, y, rcond=None)[0]
    else:
        coef = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)
    return np.asarray(phi @ coef)


def ridge_coef(phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge coefficients of :func:`ridge_fit` (minimum-norm at ``lam = 0``)."""
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    if phi.shape[1] == 0:
        return np.zeros(0)
    if lam == 0:
        return np.asarray(np.linalg.lstsq(phi, y, rcond=None)[0])
    s = phi.shape[0]
    gram = phi.T @ phi / s
    reg = gram + lam * np.eye(gram.shape[0])
    return np.asarray(np.linalg.solve(reg, phi.T @ y / s))


def any_decoder_error(phi: np.ndarray, y: np.ndarray) -> float:
    """Error of the best decoder of any form, ``E[Var(y | phi)] / Var(y)``.

    The features are exact (``K = inf``), so the best decoder of the feature
    vector is the conditional mean of ``y`` on each fibre
    ``{x : phi(x) = v}``. The population is uniform over the rows.
    """
    phi = np.asarray(phi)
    y = np.asarray(y, dtype=float)
    var = float(np.var(y))
    if var == 0:
        return 0.0
    if phi.shape[1] == 0:
        return 1.0
    _, inv = np.unique(phi, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.bincount(inv, weights=y)
    counts = np.bincount(inv)
    cond = sums[inv] / counts[inv]
    return float(np.mean((y - cond) ** 2) / var)


@dataclass(frozen=True)
class Decomposition:
    """Terms of the two-filter split for one target, normalised by ``Var(y)``.

    Attributes:
        total: ``||y - R_F y||^2``, readout ``F`` (the parities of at most k bits).
        circuit: ``||y - R_S y||^2``, full span ``S = Span G``.
        readout: ``||R_S y - R_F y||^2``.
        cross: ``2 <y - R_S y, R_S y - R_F y>``; ``total = circuit + readout +
            cross`` always holds, and the claim is ``cross = 0``.
    """

    total: float
    circuit: float
    readout: float
    cross: float


def decompose(pred_f: np.ndarray, pred_s: np.ndarray, y: np.ndarray) -> Decomposition:
    """The two-filter terms from the predictions of the two readouts."""
    y = np.asarray(y, dtype=float)
    var = float(np.mean(y**2))
    a = y - pred_s
    b = pred_s - pred_f
    return Decomposition(
        total=float(np.mean((y - pred_f) ** 2) / var),
        circuit=float(np.mean(a**2) / var),
        readout=float(np.mean(b**2) / var),
        cross=float(2 * np.mean(a * b) / var),
    )
