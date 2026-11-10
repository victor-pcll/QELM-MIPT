"""Operator light cones of monitored Haar U(4) circuits (issue #315).

Three views of how a local operator ``sigma_{x0}`` grows in a brickwork
circuit of Haar U(4) gates on an open chain of ``N`` qubits, each qubit
measured in Z with probability ``p`` after every half-layer (one time step
``t`` = one half-layer, so the Lieb-Robinson velocity is ``v_LR = 1`` site per
step):

**A1. Outcome-averaged Pauli weight** (:func:`pauli_weight_chain`). Write
``O(t) = sum_P c_P(t) P`` in the Pauli basis. U(4) is a 2-design, so the
Haar average of the squared coefficients ``|c_P|^2`` evolves by a Markov
chain on Pauli strings (Nahum, Vijay and Haah, PRX 8, 021014 (2018); von
Keyserlingk et al., PRX 8, 021013 (2018)): a gate leaves ``II`` alone and
sends any other pair to one of the 15 non-identity pairs, uniformly. A Z
measurement averaged over its outcomes deletes the ``X`` and ``Y`` components
on the measured site; averaged over where the measurements fall, it
multiplies ``|c_P|^2`` by ``1 - p`` per ``X`` or ``Y`` site. The chain is
sampled by a weighted population of strings, resampled when the weights
degenerate, so ``N = 41`` costs nothing. The output is the Pauli-weight
density ``rho(x, t) = sum_P |c_P|^2 [P_x != I] / sum_P |c_P|^2`` and the
surviving norm ``sum_P |c_P|^2``. This quantity is **linear** in the
outcome-averaged dynamics, a dephasing channel: it cannot see the MIPT,
which lives in quantities nonlinear in the trajectory state.

**A2. Trajectory-resolved Pauli weight** (:func:`trajectory_cones`). A
reference qubit ``R`` is maximally entangled with ``x0`` and the other
sites are maximally mixed: the Choi state of the circuit, infinite
temperature as for the usual OTOC. Write the joint state of the system and
``R`` as ``rho_SR = (1/2) sum_mu M_mu (x) sigma_mu^T`` with ``M_mu(0) =
sigma_mu,x0 / 2**N``. A trajectory (gates, measured sites, Born outcomes)
maps every ``M_mu`` to ``K M_mu K^dagger``, renormalized so that ``Tr M_0 =
1``: ``M_0`` is the conditional state of the system, ``M_a`` the operator
images of the Pauli ``sigma_a`` of ``R``. At ``p = 0``, ``M_a = U sigma_a,x0
U^dagger / 2**N``: the usual operator spreading. With measurements, the part
of ``M_a`` parallel to ``M_0`` is what the record has learned about ``R``
(``Tr M_a`` is the Bloch vector of ``R``); the connected operators ``O_a =
M_a - Tr(M_a) M_0`` (traceless) carry what is still quantum. Their
Pauli-weight density ``rho_m(x, t) = 1 - sum_a ||Tr_x O_a||^2 / (2 sum_a
||O_a||^2)`` (the fraction of ``O_a`` that depolarizing site ``x`` destroys)
and their weight ``w_m = sum_a ||O_a||^2 / (3 ||M_0||^2)`` (Hilbert-Schmidt
norms, ``w = 1`` at ``p = 0``) are nonlinear in the trajectory. The
entropy of ``R``, ``S_R``, is the purification diagnostic of Gullans and
Huse, PRX 10, 041020 (2020). Exact density matrices: ``N <= 10``.

**B. Krylov complexity** (:func:`arnoldi`, :func:`krylov_wavefunction`,
:func:`floquet_heisenberg_channel`). Parker et al., PRX 9, 041017 (2019)
build the Krylov chain of ``O`` under a time-independent Liouvillian
(:func:`liouvillian_lanczos`, the reference check of their ``b_n ~ alpha
n``). A Krylov chain needs a time-independent generator, so the circuit is
made Floquet: one draw of the even and odd half-layers, repeated. One period
of the Heisenberg dynamics is the channel ``Phi`` (gates, then outcome- and
location-averaged measurements, ``O -> (1 - p) O + p sum_s Pi_s O Pi_s`` per
site). Arnoldi on ``Phi`` gives an orthonormal basis ``|O_n)``, ``O(t) =
Phi^t(O) = sum_n phi_n(t) |O_n)`` with ``phi_n(t) = 0`` for ``n > t``, and
the Hessenberg coefficients ``h_{n+1,n}`` (``b_n`` of the unitary case).
``K(t) = sum_n n |phi_n(t)|^2 / sum_n |phi_n(t)|^2``, so ``K(t) <= t``: a
discrete-time circuit has no exponential growth of ``K``, the maximally
chaotic case is ``K(t) = t``. ``Phi`` contracts every traceless operator for
``p > 0``, so ``O(t)`` converges to the slowest mode of ``Phi`` and ``K(t)``
saturates: the truncation of the Krylov chain by the measurements. ``Phi``
is the outcome-averaged dynamics again: linear, blind to the MIPT.

Conventions as in :mod:`QELM.evaluation.chaos_locality`: site 0 is the most
significant bit of a basis index, Hilbert-Schmidt inner product ``(A|B) =
Tr(A^dagger B) / 2**N`` for the Krylov basis, entropies in bits.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import scipy.sparse as sp

from QELM.evaluation.chaos_locality import brickwork_bonds
from QELM.models.circuits.haar import haar_gate

# Pauli letters of the chain A1.
I_, X_, Y_, Z_ = 0, 1, 2, 3

_PAULIS = {
    0: np.eye(2, dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),
    3: np.array([[1, 0], [0, -1]], dtype=complex),
}

# The 15 non-identity pairs (a, b), a gate output of the chain.
_NON_IDENTITY_PAIRS = np.array(
    [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)], dtype=np.int8
)


def bonds_of_step(N: int, t: int) -> list[tuple[int, int]]:
    """Bonds of half-layer ``t >= 1``: even bonds for odd ``t``, then odd ones."""
    return brickwork_bonds(N, (t - 1) % 2)


# ---------------------------------------------------------------------------
# A1: outcome-averaged Pauli weight, Haar-averaged Markov chain


@dataclass
class ChainResult:
    """Output of :func:`pauli_weight_chain`.

    Attributes:
        density: ``rho(x, t)``, shape ``(T + 1, N)``, Pauli-weight density of
            the surviving operator (normalized by the surviving norm).
        survival: ``sum_P E|c_P|^2``, shape ``(T + 1,)``, 1 at ``t = 0``.
        size: mean operator size ``sum_x rho(x, t)``, shape ``(T + 1,)``.
        ess: effective sample size of the population after each step.
    """

    density: np.ndarray
    survival: np.ndarray
    size: np.ndarray
    ess: np.ndarray


def pauli_weight_chain(
    N: int,
    T: int,
    p: float,
    rng: np.random.Generator,
    x0: int | None = None,
    population: int = 20000,
    initial: int = Z_,
) -> ChainResult:
    """Sample the Haar-averaged Pauli-weight dynamics with measurements.

    Every string of the population starts as ``initial`` on ``x0`` (default
    the middle site). At each half-layer ``t = 1..T``, every gate replaces a
    non-identity pair by a uniform non-identity pair, then every string is
    multiplied by ``(1 - p)`` per ``X`` or ``Y`` site (measurement of
    probability ``p``, averaged over locations and outcomes). The population
    is resampled (systematic resampling) when its effective size falls below
    half, and the lost mean weight is kept in ``survival``.

    Returns:
        :class:`ChainResult`.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if x0 is None:
        x0 = N // 2
    strings = np.zeros((population, N), dtype=np.int8)
    strings[:, x0] = initial
    weights = np.ones(population)
    log_norm = 0.0  # log of the mean weight lost to resampling
    density = np.zeros((T + 1, N))
    survival = np.zeros(T + 1)
    ess = np.zeros(T + 1)
    density[0] = (strings != I_).mean(axis=0)
    survival[0] = 1.0
    ess[0] = population
    keep = 1.0 - p
    for t in range(1, T + 1):
        for i, j in bonds_of_step(N, t):
            active = (strings[:, i] != I_) | (strings[:, j] != I_)
            n_active = int(active.sum())
            if n_active:
                draw = _NON_IDENTITY_PAIRS[rng.integers(0, 15, size=n_active)]
                strings[active, i] = draw[:, 0]
                strings[active, j] = draw[:, 1]
        if p > 0:
            n_xy = ((strings == X_) | (strings == Y_)).sum(axis=1)
            weights = weights * keep**n_xy
        total = weights.sum()
        if total <= 0.0:
            survival[t:] = 0.0
            density[t:] = np.nan
            break
        survival[t] = math.exp(log_norm) * total / population
        density[t] = (weights[:, None] * (strings != I_)).sum(axis=0) / total
        ess[t] = total**2 / float((weights**2).sum())
        if ess[t] < population / 2:
            mean_w = total / population
            log_norm += math.log(mean_w)
            positions = (rng.random() + np.arange(population)) / population
            cumulative = np.cumsum(weights) / total
            idx = np.minimum(np.searchsorted(cumulative, positions), population - 1)
            strings = strings[idx]
            weights = np.ones(population)
    return ChainResult(
        density=density, survival=survival, size=density.sum(axis=1), ess=ess
    )


# ---------------------------------------------------------------------------
# Light-cone measures


def cone_width(density_row: np.ndarray, threshold: float) -> int:
    """Number of sites where ``rho(x, t) >= threshold`` (0 for a NaN row)."""
    row = np.nan_to_num(np.asarray(density_row, dtype=float), nan=0.0)
    return int((row >= threshold).sum())


def front_positions(
    density: np.ndarray, x0: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Right and left fronts: the farthest sites from ``x0`` above threshold.

    Returns:
        ``(right, left)``, shape ``(T + 1,)`` each, ``x - x0`` of the
        rightmost site and ``x0 - x`` of the leftmost site with ``rho >=
        threshold``, NaN when no site is above it.
    """
    D = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    right = np.full(D.shape[0], np.nan)
    left = np.full(D.shape[0], np.nan)
    for t, row in enumerate(D):
        above = np.flatnonzero(row >= threshold)
        if above.size:
            right[t] = above.max() - x0
            left[t] = x0 - above.min()
    return right, left


def front_velocity(front: np.ndarray, t_min: int, t_max: int) -> float:
    """Least-squares slope of a front between steps ``t_min`` and ``t_max``."""
    t = np.arange(len(front))
    sel = (t >= t_min) & (t <= t_max) & np.isfinite(front)
    if sel.sum() < 2:
        return float("nan")
    return float(np.polyfit(t[sel], front[sel], 1)[0])


# ---------------------------------------------------------------------------
# Dense operators: gates on both sides, partial traces


def _apply_axes(T: np.ndarray, U: np.ndarray, a1: int, a2: int) -> np.ndarray:
    """Contract the ``4 x 4`` matrix ``U`` with the two qubit axes ``a1, a2``."""
    T = np.moveaxis(T, (a1, a2), (0, 1))
    shape = T.shape
    T = (U @ T.reshape(4, -1)).reshape(shape)
    return np.moveaxis(T, (0, 1), (a1, a2))


def conjugate_gate(
    M: np.ndarray, gate: np.ndarray, i: int, j: int, N: int, adjoint: bool = False
) -> np.ndarray:
    """Return ``G M G^dagger`` (or ``G^dagger M G``) for a stack of operators.

    Args:
        M: Operators of shape ``(k, 2**N, 2**N)``.
        gate: ``4 x 4`` gate on the qubits ``(i, j)``.
        i, j: Qubits, site 0 the most significant bit.
        N: Number of qubits.
        adjoint: ``True`` for the Heisenberg map ``G^dagger M G``.
    """
    k = M.shape[0]
    G = gate.conj().T if adjoint else gate
    if j == i + 1:
        # Adjacent qubits: contiguous reshapes and batched matmuls, no copy.
        D = 2**N
        a, c = 2**i, 2 ** (N - j - 1)
        L = np.matmul(G, M.reshape(k * a, 4, c * D))
        R = np.matmul(G.conj(), L.reshape(k * D * a, 4, c))
        return R.reshape(M.shape)
    T = M.reshape((k,) + (2,) * (2 * N))
    T = _apply_axes(T, G, 1 + i, 1 + j)
    T = _apply_axes(T, G.conj(), 1 + N + i, 1 + N + j)
    return T.reshape(M.shape)


def partial_trace_site(M: np.ndarray, x: int, N: int) -> np.ndarray:
    """Trace out site ``x`` of a stack ``(k, 2**N, 2**N)``."""
    k = M.shape[0]
    a, c = 2**x, 2 ** (N - x - 1)
    T = M.reshape(k, a, 2, c, a, 2, c)
    out = T[:, :, 0, :, :, 0, :] + T[:, :, 1, :, :, 1, :]
    return out.reshape(k, a * c, a * c)


def _hs2(M: np.ndarray) -> np.ndarray:
    """Squared Hilbert-Schmidt norm ``Tr(M^dagger M)`` of each operator of a stack."""
    flat = M.reshape(M.shape[0], -1)
    return np.asarray(np.einsum("ki,ki->k", flat.conj(), flat).real)


def site_operator(N: int, x: int, pauli: int) -> np.ndarray:
    """Dense ``sigma_pauli`` on site ``x`` of ``N`` qubits."""
    out = np.ones((1, 1), dtype=complex)
    for site in range(N):
        out = np.kron(out, _PAULIS[pauli] if site == x else _PAULIS[0])
    return out


def _bit_masks(N: int) -> np.ndarray:
    """``bits[x, b]``: bit of site ``x`` in the basis index ``b``."""
    b = np.arange(2**N)
    return np.array([(b >> (N - 1 - x)) & 1 for x in range(N)])


def binary_entropy(r: float) -> float:
    """Entropy in bits of a qubit with Bloch vector length ``r``."""
    r = min(max(r, 0.0), 1.0)
    out = 0.0
    for q in ((1 + r) / 2, (1 - r) / 2):
        if q > 1e-15:
            out -= q * math.log2(q)
    return out


# ---------------------------------------------------------------------------
# A2: trajectory-resolved Pauli weight of the operator image of R


@dataclass
class TrajectoryResult:
    """Output of :func:`trajectory_cones`, averaged over trajectories.

    Attributes:
        weighted_density: ``E[w_m rho_m(x, t)]``, shape ``(T + 1, N)``: the
            Pauli weight of the quantum (connected) part of the image of
            ``R``, relative to the background; ``rho(x, t)`` of the usual
            OTOC at ``p = 0``.
        weight: ``E[w_m(t)]``, shape ``(T + 1,)``.
        density: ``weighted_density / weight``: the normalized profile.
        entropy_R: ``E[S_R(t)]`` in bits, shape ``(T + 1,)``.
        n_traj: number of trajectories.
        per_traj_entropy: ``S_R(t)`` of each trajectory, ``(n_traj, T + 1)``.
        per_traj_system_entropy: ``S(M_0)(t)``, entropy in bits of the
            conditional state of the system, ``(n_traj, T + 1)``: the order
            parameter of the purification transition of Gullans and Huse,
            PRX 10, 041020 (2020): ``N`` bits at ``t = 0``, kept for times
            exponential in ``N`` in the mixed phase ``p < p_c``, lost in a
            time ``O(1)`` in the pure phase.
    """

    weighted_density: np.ndarray
    weight: np.ndarray
    density: np.ndarray
    entropy_R: np.ndarray
    n_traj: int
    per_traj_entropy: np.ndarray = field(repr=False)
    per_traj_system_entropy: np.ndarray = field(repr=False)


def von_neumann_bits(rho: np.ndarray) -> float:
    """Entropy in bits of a density matrix."""
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return float(-(ev * np.log2(ev)).sum())


def _trajectory_observables(
    M: np.ndarray, N: int
) -> tuple[np.ndarray, float, float, float]:
    """Return ``w rho(x)``, ``w``, ``S_R`` and ``S(M_0)`` of ``M = (M_0..M_3)``."""
    traces = np.einsum("kii->k", M)
    bloch = np.array([traces[1].real, -traces[2].real, traces[3].real])
    s_R = binary_entropy(float(np.linalg.norm(bloch)))
    op = M[1:] - traces[1:, None, None] * M[0][None]
    norm_O = float(_hs2(op).sum())
    norm_B = float(_hs2(M[:1])[0])
    w = norm_O / (3.0 * norm_B)
    wrho = np.zeros(N)
    if norm_O > 1e-14 * norm_B:
        for x in range(N):
            kept = float(_hs2(partial_trace_site(op, x, N)).sum()) / 2.0
            wrho[x] = (norm_O - kept) / (3.0 * norm_B)
    M0 = M[0]
    s_sys = von_neumann_bits((M0 + M0.conj().T) / 2)
    return wrho, w, s_R, s_sys


def trajectory_cones(
    N: int,
    T: int,
    p: float,
    n_traj: int,
    rng: np.random.Generator,
    x0: int | None = None,
    draw_gate: Callable[[np.random.Generator], np.ndarray] = haar_gate,
) -> TrajectoryResult:
    """Exact trajectories of the Choi state of a monitored brickwork circuit.

    Each trajectory draws new gates on every bond of every half-layer (a
    circuit random in space and time), measures each site with probability
    ``p`` after each half-layer and draws the outcome with the Born rule of
    the conditional state ``M_0``. See the module docstring for ``w`` and
    ``rho``.

    Returns:
        :class:`TrajectoryResult`.
    """
    if x0 is None:
        x0 = N // 2
    D = 2**N
    bits = _bit_masks(N)
    wrho_sum = np.zeros((T + 1, N))
    w_sum = np.zeros(T + 1)
    s_all = np.zeros((n_traj, T + 1))
    sys_all = np.zeros((n_traj, T + 1))
    M0 = np.stack([site_operator(N, x0, mu) / D for mu in range(4)])
    for m in range(n_traj):
        M = M0.copy()
        for t in range(T + 1):
            if t > 0:
                for i, j in bonds_of_step(N, t):
                    M = conjugate_gate(M, draw_gate(rng), i, j, N)
                for x in np.flatnonzero(rng.random(N) < p):
                    diag = np.real(np.diagonal(M[0]))
                    p0 = float(diag[bits[x] == 0].sum())
                    s = 0 if rng.random() < p0 else 1
                    keep = bits[x] == s
                    M = M * (keep[:, None] & keep[None, :])[None]
                    M = M / (p0 if s == 0 else 1.0 - p0)
            wrho, w, s_R, s_sys = _trajectory_observables(M, N)
            sys_all[m, t] = s_sys
            wrho_sum[t] += wrho
            w_sum[t] += w
            s_all[m, t] = s_R
    weighted = wrho_sum / n_traj
    weight = w_sum / n_traj
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(weight[:, None] > 1e-12, weighted / weight[:, None], np.nan)
    return TrajectoryResult(
        weighted_density=weighted,
        weight=weight,
        density=density,
        entropy_R=s_all.mean(axis=0),
        n_traj=n_traj,
        per_traj_entropy=s_all,
        per_traj_system_entropy=sys_all,
    )


# ---------------------------------------------------------------------------
# B: Krylov chain of the Floquet channel, and Parker's Lanczos


def dephasing_factors(N: int, p: float) -> np.ndarray:
    """``F[r, c] = (1 - p)^{popcount(r xor c)}``: every site dephased with ``p``.

    ``O -> (1 - p) O + p (Pi_0 O Pi_0 + Pi_1 O Pi_1)`` on each site
    multiplies the matrix element ``O[r, c]`` by ``1 - p`` per site where
    the bits of ``r`` and ``c`` differ: an elementwise product.
    """
    b = np.arange(2**N)
    x = np.asarray(b[:, None] ^ b[None, :], dtype=np.int64)
    popcount = np.zeros(x.shape, dtype=np.int64)
    for site in range(N):
        popcount += (x >> site) & 1
    return np.asarray((1.0 - p) ** popcount, dtype=float)


def floquet_heisenberg_channel(
    N: int, p: float, rng: np.random.Generator
) -> Callable[[np.ndarray], np.ndarray]:
    """Heisenberg channel of one period of a monitored Floquet Haar brickwork.

    One draw of the gates of the even and of the odd half-layer. In the
    Schrodinger picture a period is: even gates, dephasing ``p``, odd gates,
    dephasing ``p``. The returned map is its adjoint, applied to a
    ``2**N x 2**N`` operator: dephasing, ``U_odd^dagger O U_odd``,
    dephasing, ``U_even^dagger O U_even``. The output is projected on the
    traceless operators: ``Phi`` is unital and keeps the trace, and the
    projection removes the identity component that rounding would otherwise
    create and that, never decaying, dominates ``Phi^t(O)`` at long times.
    """
    layers = [
        [(i, j, haar_gate(rng)) for i, j in brickwork_bonds(N, half)] for half in (0, 1)
    ]
    F = dephasing_factors(N, p) if p > 0 else None

    def apply(op: np.ndarray) -> np.ndarray:
        M = op[None]
        for half in (1, 0):
            if F is not None:
                M = M * F[None]
            for i, j, g in layers[half]:
                M = conjugate_gate(M, g, i, j, N, adjoint=True)
        out = M[0]
        trace = np.trace(out) / out.shape[0]
        return np.asarray(out - trace * np.eye(out.shape[0]))

    return apply


@dataclass
class ArnoldiResult:
    """Output of :func:`arnoldi`.

    Attributes:
        basis: orthonormal Krylov operators, shape ``(n, dim)`` (flattened).
        hessenberg: ``h[i, j] = (O_i | Phi O_j)``, shape ``(n, n)``.
        subdiagonal: ``h_{n+1, n}``, shape ``(n - 1,)``: the hopping
            amplitudes of the Krylov chain, ``b_n`` of the unitary case.
    """

    basis: np.ndarray
    hessenberg: np.ndarray
    subdiagonal: np.ndarray


def arnoldi(
    apply: Callable[[np.ndarray], np.ndarray],
    O0: np.ndarray,
    n_max: int,
    tol: float = 1e-10,
) -> ArnoldiResult:
    """Arnoldi iteration of a linear map on operators, ``(A|B) = Tr(A^+ B)/dim``.

    Classical Gram-Schmidt applied twice (re-orthogonalization), stopped at
    ``n_max`` vectors or when the new vector has norm below ``tol``.
    """
    shape = O0.shape
    dim = shape[0]
    scale = 1.0 / dim

    def inner(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.asarray(A.conj() @ B) * scale

    v = O0.reshape(-1).astype(complex)
    v = v / math.sqrt(float(inner(v, v).real))
    Q = np.zeros((n_max, v.size), dtype=complex)
    H = np.zeros((n_max, n_max), dtype=complex)
    Q[0] = v
    n = 1
    for k in range(n_max):
        w = apply(Q[k].reshape(shape)).reshape(-1)
        coeffs = np.zeros(k + 1, dtype=complex)
        for _ in range(2):
            c = inner(Q[: k + 1], w)
            w = w - c @ Q[: k + 1]
            coeffs += c
        H[: k + 1, k] = coeffs
        if k + 1 >= n_max:
            break
        h = math.sqrt(max(float(inner(w, w).real), 0.0))
        H[k + 1, k] = h
        if h < tol:
            break
        Q[k + 1] = w / h
        n = k + 2
    H = H[:n, :n]
    return ArnoldiResult(
        basis=Q[:n], hessenberg=H, subdiagonal=np.abs(np.diag(H, -1)).real
    )


@dataclass
class KrylovWavefunction:
    """Output of :func:`krylov_wavefunction`.

    Attributes:
        prob: ``|phi_n(t)|^2 / ||O(t)||^2``, shape ``(T + 1, n)``.
        complexity: ``K(t) = sum_n n prob[t, n]``, shape ``(T + 1,)``.
        norm: ``||O(t)||^2 = (O(t)|O(t))``, shape ``(T + 1,)``.
        leak: ``1 - sum_n prob``: the part of ``O(t)`` outside the basis
            (zero up to rounding while ``t < n``).
    """

    prob: np.ndarray
    complexity: np.ndarray
    norm: np.ndarray
    leak: np.ndarray


def krylov_wavefunction(
    apply: Callable[[np.ndarray], np.ndarray],
    O0: np.ndarray,
    basis: np.ndarray,
    T: int,
) -> KrylovWavefunction:
    """Project ``O(t) = Phi^t(O0)`` on the Krylov basis, ``t = 0..T``."""
    shape = O0.shape
    scale = 1.0 / shape[0]
    n = basis.shape[0]
    op = O0.astype(complex) / math.sqrt(float(np.vdot(O0, O0).real) * scale)
    prob = np.zeros((T + 1, n))
    norm = np.zeros(T + 1)
    leak = np.zeros(T + 1)
    for t in range(T + 1):
        if t > 0:
            op = apply(op)
        v = op.reshape(-1)
        nrm = float(np.vdot(v, v).real) * scale
        phi = (basis.conj() @ v) * scale
        norm[t] = nrm
        if nrm > 0:
            prob[t] = np.abs(phi) ** 2 / nrm
        leak[t] = 1.0 - prob[t].sum()
    complexity = prob @ np.arange(n)
    return KrylovWavefunction(prob=prob, complexity=complexity, norm=norm, leak=leak)


def slowest_mode_modulus(hessenberg: np.ndarray) -> float:
    """Largest ``|lambda|`` of the Ritz values of the Hessenberg matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(hessenberg))))


def liouvillian_lanczos(
    H: sp.spmatrix | np.ndarray, O0: np.ndarray, n_max: int, tol: float = 1e-10
) -> np.ndarray:
    """Lanczos coefficients ``b_1..b_n`` of ``L = [H, .]`` from ``O0`` (Parker).

    ``(A|B) = Tr(A^dagger B) / dim``, with full re-orthogonalization (the
    recursion ``L O_n = b_n O_{n-1} + b_{n+1} O_{n+1}`` alone loses
    orthogonality after a few tens of steps).
    """
    Hs = cast(sp.csr_matrix, sp.csr_matrix(cast(np.ndarray, H)))
    dim = O0.shape[0]

    def L(A: np.ndarray) -> np.ndarray:
        return np.asarray(Hs @ A - (Hs.T @ A.T).T)

    def inner(A: np.ndarray, B: np.ndarray) -> complex:
        return complex(np.vdot(A, B)) / dim

    basis: list[np.ndarray] = []
    op = O0.astype(complex)
    op = op / math.sqrt(inner(op, op).real)
    basis.append(op)
    b: list[float] = []
    for _ in range(n_max):
        A = L(basis[-1])
        for _ in range(2):
            for Q in basis:
                A = A - inner(Q, A) * Q
        bn = math.sqrt(max(inner(A, A).real, 0.0))
        if bn < tol:
            break
        b.append(bn)
        basis.append(A / bn)
    return np.asarray(b)


def free_fermion_lanczos(
    N: int, J: float, h_x: float, x: int, n_max: int, tol: float = 1e-10
) -> np.ndarray:
    """Lanczos ``b_n`` of ``X_x`` under the transverse-field chain, exactly.

    ``H = - J sum_i Z_i Z_{i+1} - h_x sum_i X_i`` maps by Jordan-Wigner
    (``gamma_{2i} = (prod_{j<i} X_j) Z_i``, ``gamma_{2i+1} = (prod_{j<i} X_j)
    Y_i``) to ``H = (i/4) sum_ab h_ab gamma_a gamma_b`` with ``h_{2i,2i+1} =
    -2 h_x``, ``h_{2i+1,2i+2} = -2 J``. A bilinear ``O = sum_ab A_ab gamma_a
    gamma_b`` evolves by ``L: A -> -i [h, A]`` and ``(O|O)`` is proportional
    to ``Tr(A^dagger A)``, so the ``b_n`` are those of a ``2N x 2N`` matrix
    problem: exact for any ``N``. ``X_x = i gamma_{2x} gamma_{2x+1}``. On the
    ``4**N`` operator space, rounding errors outside the bilinear sector are
    amplified by the larger norm of ``L`` there, and :func:`liouvillian_lanczos`
    loses the integrable ``b_n`` after about 20 steps.
    """
    h = np.zeros((2 * N, 2 * N))
    for i in range(N):
        h[2 * i, 2 * i + 1] = -2.0 * h_x
        if i < N - 1:
            h[2 * i + 1, 2 * i + 2] = -2.0 * J
    h = h - h.T
    A = np.zeros((2 * N, 2 * N), dtype=complex)
    A[2 * x, 2 * x + 1] = 1j
    A[2 * x + 1, 2 * x] = -1j

    def L(B: np.ndarray) -> np.ndarray:
        return np.asarray(-1j * (h @ B - B @ h))

    basis = [A / np.linalg.norm(A)]
    b: list[float] = []
    for _ in range(n_max):
        W = L(basis[-1])
        for _ in range(2):
            for Q in basis:
                W = W - np.vdot(Q, W) * Q
        bn = float(np.linalg.norm(W))
        if bn < tol:
            break
        b.append(bn)
        basis.append(W / bn)
    return np.asarray(b)


def mean_over(results: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over a list of equal-shape arrays."""
    A = np.asarray(results, dtype=float)
    mean = np.nanmean(A, axis=0)
    if A.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, np.nanstd(A, axis=0, ddof=1) / math.sqrt(A.shape[0])
