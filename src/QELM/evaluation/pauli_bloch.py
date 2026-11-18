"""Pauli strings as Bloch/Fourier monomials in the dense encoding (issue #364).

The dense encoding (:class:`QELM.encoding.DenseAngleEncoding`) prepares each
qubit in ``cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>``, whose Bloch vector is

    n(theta, phi) = (sin theta cos phi, sin theta sin phi, cos theta).

For the product input ``rho(x) = prod_i (I + n_i . sigma) / 2`` and any
operator ``O``,

    Tr[O rho(x)] = sum_a c_a B_a(x),   c_a = Tr[O P_a] / 2^N,
    B_a(x) = prod_i n_i^{a_i}  (n^I = 1),

so the Pauli string ``P_a`` and the Bloch monomial ``B_a`` are the same
object seen from the operator side and from the function side. In the
angles, ``B_a`` is a single product of the real trigonometric functions
``{1, cos theta, sin theta} x {1, cos phi, sin phi}`` of each qubit:

    I -> 1,   X -> sin theta cos phi,   Y -> sin theta sin phi,   Z -> cos theta,

so the change of basis from the ``4^N`` Pauli strings to the ``9^N`` real
trigonometric monomials is a partial permutation. The Pauli weight ``|a|``
is the total frequency in the polar angles, and ``#X + #Y`` the total
frequency in the azimuths.

Conventions are those of :mod:`QELM.evaluation.fourier_framework`: qubit 0
the most significant bit, Pauli strings indexed by base-4 digits ``I, X, Y,
Z = 0, 1, 2, 3`` with site 0 first (:func:`pauli_codes`), inputs ``x = (theta_0,
phi_0, theta_1, phi_1, ...)`` as for :class:`DenseAngleEncoding`.

Trigonometric basis of one qubit: index ``3 u + v`` for the function
``f_u(theta) g_v(phi)``, ``f = (1, cos, sin)``, ``g = (1, cos, sin)``; site 0
is again the most significant digit (base 9).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from QELM.evaluation.fourier import dephase, heisenberg_layer
from QELM.evaluation.fourier_framework import pauli_codes, pauli_vectors
from QELM.models.circuits.base import Layer
from QELM.models.circuits.haar import haar_gate

PAULI_LETTERS = "IXYZ"
# (u, v) of the trigonometric function of each Pauli: f_u(theta) g_v(phi).
PAULI_TRIG: tuple[tuple[int, int], ...] = ((0, 0), (2, 1), (2, 2), (1, 0))
THETA_NAMES = ("1", "cos θ", "sin θ")
PHI_NAMES = ("1", "cos φ", "sin φ")


def pauli_label(code: Sequence[int]) -> str:
    """Return the string of letters of a Pauli code, e.g. ``(1, 0, 3) -> "XIZ"``."""
    return "".join(PAULI_LETTERS[int(c)] for c in code)


def pauli_labels(n: int) -> list[str]:
    """Return the labels of the ``4^n`` strings in the order of :func:`pauli_codes`."""
    return [pauli_label(c) for c in pauli_codes(n)]


def trig_label(index: int, n: int) -> str:
    """Return the formula of trigonometric monomial ``index`` of ``n`` qubits."""
    digits = np.unravel_index(index, (9,) * n)
    factors = []
    for i, d in enumerate(digits):
        u, v = divmod(int(d), 3)
        for name, k in ((THETA_NAMES[u], u), (PHI_NAMES[v], v)):
            if k:
                factors.append(name.replace("θ", f"θ{i}").replace("φ", f"φ{i}"))
    return " ".join(factors) if factors else "1"


def _groups(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Split inputs ``(n, 2N)`` or ``(2N,)`` into ``theta, phi`` of shape ``(n, N)``."""
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    if x.shape[1] % 2:
        raise ValueError(f"Need two angles per qubit, got {x.shape[1]} features.")
    return x[:, 0::2], x[:, 1::2], single


def bloch_vectors(x: np.ndarray) -> np.ndarray:
    """Return the Bloch vectors ``n_i`` of the dense encoding, shape ``(n, N, 3)``."""
    theta, phi, single = _groups(x)
    out = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    )
    return out[0] if single else out


def _site_products(site: np.ndarray) -> np.ndarray:
    """Tensor the per-site values ``(n, N, d)`` into ``(n, d^N)``, site 0 first."""
    n, N, d = site.shape
    out = np.ones((n, 1))
    for i in range(N):
        out = (out[:, :, None] * site[:, i, None, :]).reshape(n, -1)
    return out


def dense_monomials(x: np.ndarray) -> np.ndarray:
    """Return the ``4^N`` Bloch monomials ``B_a(x)`` of the dense encoding.

    Args:
        x: Inputs ``(theta_0, phi_0, theta_1, ...)`` of shape ``(2N,)`` or
            ``(n, 2N)``.

    Returns:
        Shape ``(4^N,)`` or ``(n, 4^N)``, in the order of :func:`pauli_codes`.
    """
    nvec = bloch_vectors(np.atleast_2d(x))
    site = np.concatenate([np.ones(nvec.shape[:2] + (1,)), nvec], axis=-1)
    out = _site_products(site)
    return out[0] if np.asarray(x).ndim == 1 else out


def trig_monomials(x: np.ndarray) -> np.ndarray:
    """Return the ``9^N`` trigonometric monomials ``prod_i f_u(theta_i) g_v(phi_i)``."""
    theta, phi, single = _groups(x)
    f = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=-1)
    g = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi)], axis=-1)
    site = (f[..., :, None] * g[..., None, :]).reshape(f.shape[:2] + (9,))
    out = _site_products(site)
    return out[0] if single else out


def trig_index(code: Sequence[int]) -> int:
    """Return the index of the trigonometric monomial equal to ``B_a``."""
    index = 0
    for c in code:
        u, v = PAULI_TRIG[int(c)]
        index = 9 * index + 3 * u + v
    return index


def passage_matrix(n: int) -> np.ndarray:
    """Return ``M`` of shape ``(4^n, 9^n)`` with ``B = M @ trig``: one 1 per row.

    Row ``a`` has its single 1 at :func:`trig_index` of ``a``: the change of
    basis from Pauli strings to trigonometric monomials is a partial
    permutation (``4^n`` of the ``9^n`` monomials are reached).
    """
    codes = pauli_codes(n)
    M = np.zeros((4**n, 9**n))
    M[np.arange(4**n), [trig_index(c) for c in codes]] = 1.0
    return M


def frequencies(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the total frequencies in the thetas and in the phis of each ``B_a``.

    The first is the Pauli weight ``|a|``, the second ``#X + #Y``.
    """
    codes = pauli_codes(n)
    k_theta = np.count_nonzero(codes, axis=1)
    k_phi = np.count_nonzero((codes == 1) | (codes == 2), axis=1)
    return k_theta, k_phi


def site_moments(ensemble: str = "sphere", nodes: int = 64) -> np.ndarray:
    """Return ``G[a, b] = E[n^a n^b]`` of one qubit (``n^I = 1``), a ``4 x 4`` matrix.

    Args:
        ensemble: ``"sphere"``: Bloch vector uniform on the sphere (or the six
            Pauli eigenstates), ``G = diag(1, 1/3, 1/3, 1/3)``; ``"box"``:
            ``theta`` and ``phi`` independent and uniform in ``[0, pi]``, the
            bounds of :class:`DenseAngleEncoding` (Gauss-Legendre quadrature,
            exact to machine precision for these functions).
        nodes: Quadrature nodes per angle for ``"box"``.
    """
    if ensemble == "sphere":
        return np.diag([1.0, 1 / 3, 1 / 3, 1 / 3])
    if ensemble != "box":
        raise ValueError(f"unknown ensemble {ensemble!r}")
    t, w = np.polynomial.legendre.leggauss(nodes)
    ang = 0.5 * np.pi * (t + 1)
    w = w / w.sum()
    th, ph = np.meshgrid(ang, ang, indexing="ij")
    ww = np.outer(w, w)
    vals = np.stack(
        [
            np.ones_like(th),
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th),
        ]
    )
    return np.einsum("aij,bij,ij->ab", vals, vals, ww)


def gram_matrix(n: int, ensemble: str = "sphere") -> np.ndarray:
    """Return ``E[B_a B_b]`` of ``n`` i.i.d. qubits, the tensor power of the moments."""
    G1 = site_moments(ensemble)
    G = np.ones((1, 1))
    for _ in range(n):
        G = np.kron(G, G1)
    return G


def sphere_weights(n: int) -> np.ndarray:
    """Return ``w_a = 3^{-|a|}``, the diagonal of the sphere Gram matrix."""
    return 3.0 ** -np.count_nonzero(pauli_codes(n), axis=1).astype(float)


def brickwork_layers(n: int, depth: int, rng: np.random.Generator) -> list[Layer]:
    """Draw ``depth`` Haar layers of an open brickwork: even bonds, then odd bonds."""
    bonds = list(range(0, n - 1, 2)) + list(range(1, n - 1, 2))
    return [[(b, b + 1, haar_gate(rng)) for b in bonds] for _ in range(depth)]


def heisenberg_z(layers: Sequence[Layer], n: int, site: int, p: float) -> np.ndarray:
    """Return the Pauli coefficients of the feature ``E[Z_site at the end]``.

    The forward circuit applies the layers in order, each followed by a Z
    measurement of every qubit with probability ``p`` whose outcome is
    forgotten (Z-dephasing). Its mean final ``Z_site`` is ``Tr[O rho(x)]``
    with ``O`` the Heisenberg image of ``Z_site``; the result is
    ``c_a = Tr[O P_a] / 2^n``, so the feature is ``sum_a c_a B_a(x)``.
    """
    Z = np.diag([1.0, -1.0]).astype(complex)
    obs = np.kron(np.kron(np.eye(2**site), Z), np.eye(2 ** (n - site - 1)))
    for layer in reversed(layers):
        obs = heisenberg_layer(dephase(obs, p), layer)
    return np.asarray(pauli_vectors(obs).real)


def weight_profile(c: np.ndarray, n: int, w: np.ndarray | None = None) -> np.ndarray:
    """Return ``P_k = sum_{|a| = k} w_a c_a^2`` for ``k = 0..n``.

    With the sphere weights (default), ``P_0`` is the squared mean and
    ``sum_{k >= 1} P_k`` the variance of the feature over the sphere.
    """
    w = sphere_weights(n) if w is None else w
    k, _ = frequencies(n)
    power = w * np.asarray(c) ** 2
    return np.array([power[k == j].sum() for j in range(n + 1)])


def reconstruction_order(c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the strings by decreasing power ``w_a c_a^2``, the identity first."""
    power = w * np.asarray(c) ** 2
    rest = np.argsort(-power[1:], kind="stable") + 1
    return np.concatenate([[0], rest])


def residual_variance(c: np.ndarray, w: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Return the share of the variance left after the first ``m`` strings of ``order``.

    Entry ``m`` (``m = 1..len(order)``, entry 0 for ``m = 0`` is 1): by
    Parseval for orthogonal monomials, ``sum_{a not kept, a != I} w_a c_a^2``
    over ``sum_{a != I} w_a c_a^2``.
    """
    power = w * np.asarray(c) ** 2
    power[0] = 0.0
    total = power.sum()
    kept = np.concatenate([[0.0], np.cumsum(power[order])])
    return np.asarray(1.0 - kept / total) if total > 0 else np.zeros(len(kept))
