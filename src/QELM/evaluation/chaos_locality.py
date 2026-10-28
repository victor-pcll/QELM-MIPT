"""Chaos and locality diagnostics of the reservoir dynamics (issue #243).

A measurement-induced phase transition (MIPT) needs two things from the
unitary part of the dynamics, and this module measures both on a
Hamiltonian ``H`` or on the unitaries of a circuit:

**Chaos (part A).** A reservoir that keeps a volume law at ``p > 0``
thermalizes, and its energy levels repel. Integrable (or localized) models
have uncorrelated levels. The statistic is read in the central third of the
spectrum of each symmetry sector (:func:`sector_spectra`; mixing two sectors
superposes two independent spectra and fakes Poisson statistics):

- the spacings ``s`` of the unfolded levels (:func:`unfold`, mean spacing
  1), whose distribution is ``P(s) = exp(-s)`` (:func:`poisson_pdf`) for an
  integrable model and the Wigner surmise of the GOE ``P(s) = (pi/2) s
  exp(-pi s^2 / 4)`` (:func:`goe_pdf`) for a chaotic one;
- the ratios of adjacent spacings ``r_n = min(d_n, d_{n+1}) / max(d_n,
  d_{n+1})`` (:func:`gap_ratios`), which need no unfolding. Their mean is
  ``R_POISSON = 2 ln 2 - 1 = 0.386`` for Poisson levels and ``R_GOE = 0.531``
  for the GOE (``R_GUE = 0.600`` without time-reversal symmetry,
  :func:`gue_pdf`), the
  large-matrix values of Atas et al., PRL 110, 084101 (2013).

For a circuit, the levels are the quasi-energies, the phases of the
eigenvalues of the unitary of one period (:func:`quasienergies`); they are
uniform on the circle, so the unfolding is a division by ``2 pi / dim``.

**Locality (part B).** A local dynamics has a Lieb-Robinson light cone:

1. from a product state at ``p = 0``, the half-chain entropy grows linearly
   with a rate that does not depend on ``N`` (:func:`entanglement_growth`);
   with all-to-all couplings the rate grows with the number of bonds
   crossing the cut;
2. the out-of-time-order commutator ``C(x, t) = (1/2) <|[Z_x(t), Z_x0]|^2>``
   at infinite temperature (:func:`otoc_map`) is zero outside a cone ``|x -
   x0| > v t``: a "V" in the plane ``(x, t)``;
3. a reference qubit ``R`` maximally entangled with the site ``x0``
   (Gullans and Huse, PRL 125, 070606 (2020)) is purified by the
   measurements: :func:`reference_qubit_trajectories` returns ``<S_Q(t)>``
   and the mean change ``<Delta S_Q(x, t)>`` that a measurement at ``(x,
   t)`` would cause, zero outside the light cone of ``x0``;
   :func:`reference_qubit_cutoff` keeps only the outcomes of the
   measurements at ``|x - x0| <= l`` and shows how far from ``x0`` the
   purification of the area-law phase reaches.

The dynamics is a list of dense unitaries ``steps`` on the ``N`` system
qubits, applied in turn and cycled (one ``exp(-i H tau)`` for a Hamiltonian,
the even and the odd half-layers for a brickwork circuit), with a round of
Z measurements of probability ``p`` per qubit after each step. Site 0 is the
most significant bit of a basis index, ``|0>`` the ``Z = +1`` state, as in
the rest of the package. Entropies are in bits.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, expm_multiply

from QELM.evaluation.entropy import von_neumann_entropy

# A time step: a dense unitary or an operator applied by ``@`` (sparse
# propagator, layer of gates).
Step = np.ndarray | LinearOperator
# A two-qubit gate on the sites (i, j), most significant bit i.
Gate = tuple[int, int, np.ndarray]

# Mean gap ratio <r> for large matrices (Atas et al., PRL 110, 084101 (2013)).
R_POISSON = 2 * math.log(2) - 1
R_GOE = 0.5307
R_GUE = 0.5996

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _dense(A: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Return ``A`` as a dense complex array."""
    if sp.issparse(A):
        return np.asarray(cast(sp.csr_matrix, A).toarray(), dtype=complex)
    return np.asarray(A, dtype=complex)


# ---------------------------------------------------------------------------
# Reference distributions


def poisson_pdf(s: np.ndarray) -> np.ndarray:
    """Return ``P(s) = exp(-s)``, the spacings of uncorrelated levels."""
    s = np.asarray(s, dtype=float)
    return np.exp(-s)


def goe_pdf(s: np.ndarray) -> np.ndarray:
    """Return the Wigner surmise of the GOE, ``(pi/2) s exp(-pi s^2 / 4)``."""
    s = np.asarray(s, dtype=float)
    return np.pi / 2 * s * np.exp(-np.pi * s**2 / 4)


def gue_pdf(s: np.ndarray) -> np.ndarray:
    """Return the Wigner surmise of the GUE, ``(32/pi^2) s^2 exp(-4 s^2 / pi)``.

    The reference of a chaotic dynamics without time-reversal symmetry, such
    as a Floquet circuit whose period has no antiunitary symmetry.
    """
    s = np.asarray(s, dtype=float)
    return 32 / np.pi**2 * s**2 * np.exp(-4 * s**2 / np.pi)


# ---------------------------------------------------------------------------
# Bits, states and operators


def _bits(N: int) -> np.ndarray:
    """Return the ``(2**N, N)`` array of the bits of every basis index."""
    shifts = N - 1 - np.arange(N)
    return (np.arange(2**N)[:, None] >> shifts[None, :]) & 1


def z_values(N: int) -> np.ndarray:
    """Return ``z[b, i] = +1`` or ``-1``, the eigenvalue of ``Z_i`` on ``|b>``."""
    return 1 - 2 * _bits(N)


def neel_state(N: int) -> np.ndarray:
    """Return ``|0101...>`` on ``N`` qubits as a state vector."""
    index = int("".join("01"[i % 2] for i in range(N)), 2)
    psi = np.zeros(2**N, dtype=complex)
    psi[index] = 1.0
    return psi


def y_polarized_state(N: int) -> np.ndarray:
    """Return ``|+y>^N``, ``|+y> = (|0> + i |1>) / sqrt 2``.

    ``<X_i> = <Z_i> = 0`` and ``<P_i P_j> = 0`` for ``P`` in ``{X, Z}``: the
    energy of every Ising and TFIM Hamiltonian here is zero, the middle of
    the spectrum (infinite temperature), so the entropy saturates at the
    Page value. The Néel state is an extreme of ``- sum J_i Z_i Z_{i+1}``.
    """
    single = np.array([1, 1j], dtype=complex) / math.sqrt(2)
    psi = np.ones(1, dtype=complex)
    for _ in range(N):
        psi = np.kron(psi, single)
    return psi


def charge_operator(N: int) -> sp.csr_matrix:
    """Return ``Q = sum_i (1 - Z_i) / 2``, the number of qubits at ``|1>``."""
    return sp.diags(_bits(N).sum(axis=1).astype(complex), format="csr")


def parity_operator(N: int, basis: str = "Z") -> sp.csr_matrix:
    """Return ``prod_i Z_i`` (``basis="Z"``) or ``prod_i X_i`` (``basis="X"``).

    Raises:
        ValueError: If ``basis`` is neither ``"X"`` nor ``"Z"``.
    """
    dim = 2**N
    if basis == "Z":
        return sp.diags(np.prod(z_values(N), axis=1).astype(complex), format="csr")
    if basis == "X":
        index = np.arange(dim)
        return sp.csr_matrix(
            (np.ones(dim, dtype=complex), (index, index ^ (dim - 1))),
            shape=(dim, dim),
        )
    raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")


def reflection_operator(N: int) -> sp.csr_matrix:
    """Return the permutation of the sites ``i -> N - 1 - i``."""
    bits = _bits(N)
    weights = 2 ** np.arange(N)  # reversed bit order
    target = (bits * weights[None, :]).sum(axis=1)
    dim = 2**N
    return sp.csr_matrix(
        (np.ones(dim, dtype=complex), (target, np.arange(dim))), shape=(dim, dim)
    )


# ---------------------------------------------------------------------------
# Reference one-dimensional reservoirs


def ising_chain_hamiltonian(
    couplings: Sequence[float] | np.ndarray, h_x: float, h_z: float
) -> sp.csr_matrix:
    """Open Ising chain in a tilted field (the reservoir of issue #241).

    ``H = - sum_i J_i Z_i Z_{i+1} - h_x sum_i X_i - h_z sum_i Z_i``, with
    ``N = len(couplings) + 1`` sites. ``h_z = 0`` is the transverse-field
    chain, which maps to free fermions (integrable) and conserves
    ``prod_i X_i``; ``h_z != 0`` breaks both (Kim and Huse, PRL 111, 127205
    (2013): ``h_x = 1.05``, ``h_z = 0.5`` is robustly chaotic).

    Returns:
        Complex Hermitian CSR matrix of shape ``(2**N, 2**N)``.
    """
    J = np.asarray(couplings, dtype=float)
    N = J.size + 1
    dim = 2**N
    z = z_values(N)
    basis = np.arange(dim)
    diagonal = -h_z * z.sum(axis=1).astype(float)
    for i, J_i in enumerate(J):
        diagonal -= J_i * z[:, i] * z[:, i + 1]
    rows = [basis] * (N + 1)
    cols = [basis] + [basis ^ (1 << (N - 1 - i)) for i in range(N)]
    values = [diagonal.astype(complex)] + [
        np.full(dim, -h_x, dtype=complex) for _ in range(N)
    ]
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    )


def xxz_chain_hamiltonian(
    N: int,
    J: float = 1.0,
    Delta: float = 1.0,
    fields: Sequence[float] | np.ndarray | None = None,
    boundary: str = "open",
) -> sp.csr_matrix:
    """XXZ chain of issue #236, ``sum_<ij> [J (XX + YY) + Delta ZZ] + sum_i h_i Z_i``.

    ``X_i X_j + Y_i Y_j`` swaps the bits ``i`` and ``j`` when they differ
    (amplitude ``2 J``), ``Z_i Z_j`` and ``Z_i`` are diagonal. Without fields
    the chain is integrable by the Bethe ansatz.

    Raises:
        ValueError: If ``boundary`` is unknown.
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary!r}")
    h = np.zeros(N) if fields is None else np.asarray(fields, dtype=float)
    bonds = [(i, i + 1) for i in range(N - 1)]
    if boundary == "periodic" and N > 2:
        bonds.append((N - 1, 0))
    dim = 2**N
    z = z_values(N)
    basis = np.arange(dim)
    diagonal = (z * h[None, :]).sum(axis=1).astype(float)
    rows, cols, values = [basis], [basis], [np.zeros(dim, dtype=complex)]
    for i, j in bonds:
        diagonal += Delta * z[:, i] * z[:, j]
        differ = z[:, i] != z[:, j]
        mask = (1 << (N - 1 - i)) | (1 << (N - 1 - j))
        rows.append(basis[differ])
        cols.append(basis[differ] ^ mask)
        values.append(np.full(int(differ.sum()), 2 * J, dtype=complex))
    values[0] = diagonal.astype(complex)
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    )


def floquet_ising_gate(
    phi_xx: float = 0.3, phi_zz: float = 0.6, theta_x: float = 0.2, theta_z: float = 0.4
) -> np.ndarray:
    """Brick of the Floquet Ising chain of Skinner et al., PRX 9, 031009 (2019).

    ``exp{-i [phi_xx XX + theta_x (X1 + X2)]} exp{-i [phi_zz ZZ + theta_z (Z1
    + Z2)]}``, the right factor first (issue #239). Defaults: the paper's
    values.
    """
    u_x = scipy.linalg.expm(
        -1j
        * (phi_xx * np.kron(_X, _X) + theta_x * (np.kron(_X, _I2) + np.kron(_I2, _X)))
    )
    u_z = scipy.linalg.expm(
        -1j
        * (phi_zz * np.kron(_Z, _Z) + theta_z * (np.kron(_Z, _I2) + np.kron(_I2, _Z)))
    )
    return np.asarray(u_x @ u_z)


def brickwork_layers(N: int, gate: np.ndarray) -> list[np.ndarray]:
    """Return the even and the odd half-layers of an open brickwork chain.

    The even layer applies ``gate`` on the bonds ``(0, 1), (2, 3), ...``, the
    odd one on ``(1, 2), (3, 4), ...``: the two time steps of one Floquet
    period, ``U_period = U_odd @ U_even``.
    """
    return [
        dense_operator(
            gate_layer_operator(N, [(i, j, gate) for i, j in brickwork_bonds(N, half)])
        )
        for half in (0, 1)
    ]


def _apply_gate_rows(M: np.ndarray, gate: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return ``G_ij @ M`` for ``M`` of shape ``(2**N, k)``, without building ``G``."""
    dim, k = M.shape
    N = dim.bit_length() - 1
    T = np.moveaxis(M.reshape((2,) * N + (k,)), (i, j), (0, 1))
    shape = T.shape
    T = (gate @ T.reshape(4, -1)).reshape(shape)
    return np.moveaxis(T, (0, 1), (i, j)).reshape(dim, k)


def gate_layer_operator(N: int, gates: Sequence[Gate]) -> LinearOperator:
    """Return a layer of two-qubit gates as an operator on ``2**N`` amplitudes.

    ``op @ M`` applies the gates ``(i, j, G)`` in order to the rows of ``M``
    (a vector or a ``(2**N, k)`` matrix) by reshaping, in ``O(2**N k)`` per
    gate: no ``2**N x 2**N`` matrix is formed.
    """
    dim = 2**N
    layer = [(int(i), int(j), np.asarray(g, dtype=complex)) for i, j, g in gates]

    def matmat(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=complex)
        for i, j, gate in layer:
            M = _apply_gate_rows(M, gate, i, j)
        return M

    def matvec(v: np.ndarray) -> np.ndarray:
        return matmat(np.asarray(v).reshape(dim, 1))[:, 0]

    return LinearOperator((dim, dim), matvec=matvec, matmat=matmat, dtype=complex)


def dense_operator(op: Step) -> np.ndarray:
    """Return a step (dense or operator) as a dense matrix."""
    if isinstance(op, np.ndarray):
        return op
    return np.asarray(op @ np.eye(op.shape[0], dtype=complex))


def brickwork_bonds(N: int, half: int) -> list[tuple[int, int]]:
    """Bonds ``(i, i + 1)`` of the even (``half = 0``) or odd half-layer, open."""
    return [(i, i + 1) for i in range(half % 2, N - 1, 2)]


def random_brickwork_steps(
    N: int,
    n_steps: int,
    draw_gate: Callable[[np.random.Generator], np.ndarray],
    rng: np.random.Generator,
) -> list[LinearOperator]:
    """Return ``n_steps`` half-layers of a random circuit, new gates everywhere.

    Half-layer ``t`` acts on the even bonds for even ``t`` and on the odd ones
    for odd ``t``, as the brickwork of :mod:`QELM.models.circuits`, every gate
    drawn independently by ``draw_gate`` (Haar or Clifford): a circuit random
    in space and time.
    """
    return [
        gate_layer_operator(
            N, [(i, j, draw_gate(rng)) for i, j in brickwork_bonds(N, t)]
        )
        for t in range(n_steps)
    ]


def random_floquet_layers(
    N: int,
    draw_gate: Callable[[np.random.Generator], np.ndarray],
    rng: np.random.Generator,
) -> list[LinearOperator]:
    """Return the two half-layers of a random Floquet circuit.

    One draw of the gates of a period (even then odd half-layer), repeated
    in time: the period ``U_odd U_even`` has quasi-energies, which a circuit
    with new gates at every step has not.
    """
    return random_brickwork_steps(N, 2, draw_gate, rng)


def step_unitary(H: sp.spmatrix | np.ndarray, tau: float) -> np.ndarray:
    """Return the dense unitary ``exp(-i H tau)``."""
    H_dense = _dense(H)
    return np.asarray(scipy.linalg.expm(-1j * tau * H_dense))


def propagator_operator(H: sp.spmatrix | np.ndarray, tau: float) -> LinearOperator:
    """Return ``exp(-i H tau)`` as an operator applied from the sparse ``H``.

    ``U @ psi`` calls ``scipy.sparse.linalg.expm_multiply``: no ``2**N x
    2**N`` matrix is formed, for the state evolutions of large ``N``.
    """
    A = (-1j * tau) * sp.csr_matrix(cast(sp.csr_matrix, H))

    def apply(v: np.ndarray) -> np.ndarray:
        return np.asarray(expm_multiply(A, v))

    return LinearOperator(A.shape, matvec=apply, matmat=apply, dtype=complex)


# ---------------------------------------------------------------------------
# Part A: level statistics


def symmetry_blocks(
    dim: int, symmetries: Sequence[sp.spmatrix | np.ndarray], tol: float = 1e-8
) -> list[np.ndarray]:
    """Split the Hilbert space into the joint eigenspaces of ``symmetries``.

    The symmetries are Hermitian operators commuting with the dynamics
    (parities, charges, reflections), applied in order: each block of the
    previous ones is split by the eigenvalues of the next symmetry
    restricted to it. A symmetry that does not leave a block invariant (the
    spin flip of the XXZ chain maps the charge ``Q`` to ``N - Q``) is skipped
    in that block only: the restricted matrix is then not unitary.

    Args:
        dim: Hilbert-space dimension.
        symmetries: Hermitian ``(dim, dim)`` operators, dense or sparse.
        tol: Tolerance to group equal eigenvalues and to test unitarity.

    Returns:
        Orthonormal bases ``V`` of shape ``(dim, d_k)``, one per block, whose
        dimensions add up to ``dim``.
    """
    blocks = [np.eye(dim, dtype=complex)]
    for S in symmetries:
        if len(blocks) == 1 and blocks[0].shape[1] == dim:
            # First split: V = 1, no product is needed.
            permuted = _permutation_eigenspaces(S)
            blocks = (
                permuted if permuted is not None else _eigenspaces(_dense(S), None, tol)
            )
            continue
        # Sparse times dense: the symmetries are diagonal or permutations.
        S_op = sp.csr_matrix(cast(sp.csr_matrix, S))
        split: list[np.ndarray] = []
        for V in blocks:
            SV = np.asarray(S_op @ V)
            S_block = V.conj().T @ SV
            if not np.allclose(SV, V @ S_block, atol=tol):
                split.append(V)  # S does not leave this block invariant
                continue
            split.extend(_eigenspaces(S_block, V, tol))
        blocks = split
    return blocks


def _permutation_eigenspaces(S: sp.spmatrix | np.ndarray) -> list[np.ndarray] | None:
    """Eigenspaces of a permutation ``S`` with ``S^2 = 1``, built directly.

    A fixed basis state ``|b>`` has eigenvalue +1, a pair ``|b> <-> |c>``
    gives ``(|b> + |c>) / sqrt 2`` (+1) and ``(|b> - |c>) / sqrt 2`` (-1).
    This avoids the ``eigh`` of a ``2**N`` matrix for the spin flip and the
    reflection.

    Returns:
        The two bases (+1 then -1, empty ones dropped), or ``None`` when
        ``S`` is not such a permutation.
    """
    coo = sp.coo_matrix(cast(sp.csr_matrix, S))
    dim = coo.shape[0]
    if coo.nnz != dim or not np.allclose(coo.data, 1):
        return None
    image = np.full(dim, -1)
    image[coo.col] = coo.row
    if np.any(image < 0) or np.any(image[image] != np.arange(dim)):
        return None
    index = np.arange(dim)
    fixed = index[image == index]
    first = index[image > index]
    second = image[first]
    plus = np.zeros((dim, fixed.size + first.size))
    plus[fixed, np.arange(fixed.size)] = 1
    columns = fixed.size + np.arange(first.size)
    plus[first, columns] = plus[second, columns] = 1 / math.sqrt(2)
    minus = np.zeros((dim, first.size))
    minus[first, np.arange(first.size)] = 1 / math.sqrt(2)
    minus[second, np.arange(first.size)] = -1 / math.sqrt(2)
    return [V for V in (plus, minus) if V.shape[1]]


def _eigh(A: np.ndarray, vectors: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``eigh`` of a Hermitian matrix, in real arithmetic when ``A`` is real.

    The models here are real symmetric in the computational basis, and a
    real ``eigh`` is about four times faster than a complex one.
    """
    if not np.any(np.abs(A.imag) > 1e-12):
        A = A.real
    if vectors:
        values, vecs = np.linalg.eigh(A)
        return values, vecs
    return np.linalg.eigvalsh(A), np.empty((0, 0))


def _eigenspaces(
    S_block: np.ndarray, V: np.ndarray | None, tol: float
) -> list[np.ndarray]:
    """Split the columns of ``V`` by the eigenvalues of ``S_block = V^+ S V``.

    ``V = None`` stands for the identity.
    """
    S_block = (S_block + S_block.conj().T) / 2
    off_diagonal = S_block - np.diag(np.diag(S_block))
    if not np.any(np.abs(off_diagonal) > tol):
        # Diagonal (a charge, the Z parity): the basis vectors already split.
        order = np.argsort(np.diag(S_block).real, kind="stable")
        values = np.diag(S_block).real[order]
        vectors = np.eye(S_block.shape[0])[:, order]
    else:
        values, vectors = _eigh(S_block)
    # Group the eigenvalues that are equal up to ``tol``.
    starts = np.flatnonzero(np.diff(values, prepend=-np.inf) > tol)
    return [
        vectors[:, lo:hi] if V is None else V @ vectors[:, lo:hi]
        for lo, hi in zip(starts, [*starts[1:], values.size], strict=True)
    ]


def sector_spectra(
    H: sp.spmatrix | np.ndarray,
    symmetries: Sequence[sp.spmatrix | np.ndarray] = (),
    blocks: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Return the sorted energies of ``H`` in each symmetry sector.

    Args:
        H: Hermitian Hamiltonian, dense or sparse.
        symmetries: Hermitian operators commuting with ``H``
            (:func:`symmetry_blocks`).
        blocks: The output of :func:`symmetry_blocks`, to reuse it over
            disorder realisations; ``symmetries`` is then ignored.

    Returns:
        One sorted 1-D array per sector.
    """
    H_op = sp.csr_matrix(cast(sp.csr_matrix, H))
    if blocks is None:
        blocks = symmetry_blocks(H_op.shape[0], symmetries)
    spectra = []
    for V in blocks:
        block = V.conj().T @ np.asarray(H_op @ V)
        spectra.append(_eigh((block + block.conj().T) / 2, vectors=False)[0])
    return spectra


def quasienergies(
    U: np.ndarray,
    symmetries: Sequence[sp.spmatrix | np.ndarray] = (),
    blocks: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Return the sorted quasi-energies ``theta`` in ``[-pi, pi)`` per sector.

    The eigenvalues of the period unitary are ``exp(-i theta)``.

    Args:
        U: Unitary of one period, ``(dim, dim)``.
        symmetries: Hermitian operators commuting with ``U``.
        blocks: As for :func:`sector_spectra`.
    """
    if blocks is None:
        blocks = symmetry_blocks(U.shape[0], symmetries)
    spectra = []
    for V in blocks:
        block = V.conj().T @ U @ V
        phases = -np.angle(np.linalg.eigvals(block))
        spectra.append(np.sort(phases))
    return spectra


def central_window(levels: np.ndarray, fraction: float = 1 / 3) -> np.ndarray:
    """Return the central ``fraction`` of the sorted ``levels``."""
    levels = np.sort(np.asarray(levels, dtype=float))
    n = levels.size
    keep = int(round(fraction * n))
    start = (n - keep) // 2
    return levels[start : start + keep]


def gap_ratios(levels: np.ndarray) -> np.ndarray:
    """Return ``r_n = min(d_n, d_{n+1}) / max(d_n, d_{n+1})`` of sorted levels.

    Ratios ``0 / 0`` of exact degeneracies are dropped.
    """
    gaps = np.diff(np.sort(np.asarray(levels, dtype=float)))
    low = np.minimum(gaps[:-1], gaps[1:])
    high = np.maximum(gaps[:-1], gaps[1:])
    keep = high > 0
    return np.asarray(low[keep] / high[keep])


def unfold(levels: np.ndarray, degree: int = 7) -> np.ndarray:
    """Return the unfolded spacings, of mean 1, of sorted ``levels``.

    The staircase ``n(E)`` (number of levels below ``E``) is fitted by a
    polynomial of degree ``degree`` in ``E``, the smooth density of states;
    the unfolded levels are ``n_smooth(E_k)`` and their spacings are divided
    by their mean.
    """
    E = np.sort(np.asarray(levels, dtype=float))
    if E.size < 3:
        return np.empty(0)
    x = (E - E.mean()) / (E.std() or 1.0)
    degree = min(degree, E.size - 1)
    smooth = np.polynomial.Polynomial.fit(x, np.arange(E.size), degree)(x)
    spacings = np.diff(smooth)
    return np.asarray(spacings / spacings.mean())


@dataclass
class LevelStatistics:
    """Pooled level statistics of a model.

    Attributes:
        spacings: Unfolded spacings of the central windows of every sector
            and disorder realisation, mean 1 in each.
        ratios: Gap ratios ``r_n`` of the same windows.
        mean_r: Mean of ``ratios``.
        sector_dims: Dimension of every sector of the first realisation.
    """

    spacings: np.ndarray
    ratios: np.ndarray
    mean_r: float
    sector_dims: list[int]


def level_statistics(
    spectra: Sequence[Sequence[np.ndarray]],
    *,
    fraction: float = 1 / 3,
    degree: int = 7,
    circular: bool = False,
    min_levels: int = 20,
) -> LevelStatistics:
    """Pool the spacings and gap ratios of several sectors and realisations.

    Args:
        spectra: ``spectra[k]`` holds the sorted levels of each sector of the
            realisation ``k`` (:func:`sector_spectra` or :func:`quasienergies`).
        fraction: Central part of each sector that is kept (1/3 by default):
            the edges of a spectrum are sparse and less chaotic.
        degree: Degree of the unfolding polynomial (:func:`unfold`).
        circular: Quasi-energies: the whole circle is kept and the spacings
            are divided by their mean, the density being uniform.
        min_levels: Sectors with fewer levels are skipped.

    Returns:
        A :class:`LevelStatistics`.
    """
    spacings: list[np.ndarray] = []
    ratios: list[np.ndarray] = []
    for sectors in spectra:
        for levels in sectors:
            if len(levels) < min_levels:
                continue
            if circular:
                theta = np.sort(np.asarray(levels, dtype=float))
                gaps = np.diff(np.concatenate([theta, [theta[0] + 2 * np.pi]]))
                spacings.append(gaps / gaps.mean())
                ratios.append(
                    gap_ratios(np.concatenate([theta, theta[:2] + 2 * np.pi]))
                )
            else:
                window = central_window(np.asarray(levels), fraction)
                spacings.append(unfold(window, degree))
                ratios.append(gap_ratios(window))
    s = np.concatenate(spacings) if spacings else np.empty(0)
    r = np.concatenate(ratios) if ratios else np.empty(0)
    dims = [len(levels) for levels in spectra[0]] if spectra else []
    return LevelStatistics(
        spacings=s,
        ratios=r,
        mean_r=float(r.mean()) if r.size else float("nan"),
        sector_dims=dims,
    )


# ---------------------------------------------------------------------------
# Part B1: entanglement growth at p = 0


def entanglement_growth(
    steps: Sequence[Step],
    psi0: np.ndarray,
    n_steps: int,
    sector_A: Sequence[int] | None = None,
) -> np.ndarray:
    """Return ``S_A`` (bits) after ``0, 1, ..., n_steps`` unitary steps.

    Args:
        steps: Unitaries applied in turn and cycled, dense or as
            :func:`propagator_operator`.
        psi0: Initial state vector, ``(2**N,)``.
        n_steps: Number of steps.
        sector_A: Qubits of ``A``; the left half ``0 .. N//2 - 1`` by
            default.

    Returns:
        Array of shape ``(n_steps + 1,)``.
    """
    psi = np.asarray(psi0, dtype=complex)
    N = int(round(math.log2(psi.size)))
    A = list(range(N // 2)) if sector_A is None else list(sector_A)
    out = [von_neumann_entropy(psi, A)]
    for t in range(n_steps):
        psi = steps[t % len(steps)] @ psi
        out.append(von_neumann_entropy(psi, A))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Part B2: OTOC


def otoc_map(steps: Sequence[Step], x0: int, n_steps: int) -> np.ndarray:
    """Return ``C(x, t) = (1/2) <|[Z_x(t), Z_x0]|^2>`` at infinite temperature.

    With ``A = Z_x(t) = W^dagger Z_x W`` (``W`` the evolution operator up to
    step ``t``) and ``B = Z_x0`` diagonal with entries ``b_i``, ``C = 1 -
    Re Tr(A B A B) / D = 1 - sum_ij b_i b_j |A_ij|^2 / D``: zero while
    ``Z_x(t)`` commutes with ``Z_x0``, close to 1 once the operator has
    spread over ``x0``.

    Args:
        steps: Dense unitaries applied in turn and cycled.
        x0: Site of the fixed operator ``Z_x0``.
        n_steps: Number of steps.

    Returns:
        Array of shape ``(n_steps + 1, N)``, ``C[t, x]``.
    """
    dim = steps[0].shape[0]
    N = int(round(math.log2(dim)))
    z = z_values(N).astype(float)
    b = z[:, x0]
    W = np.eye(dim, dtype=complex)
    out = np.zeros((n_steps + 1, N))
    for t in range(n_steps + 1):
        if t > 0:
            W = steps[(t - 1) % len(steps)] @ W
        for x in range(N):
            A = W.conj().T @ (z[:, x][:, None] * W)
            out[t, x] = 1.0 - float(b @ (np.abs(A) ** 2) @ b) / dim
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# Part B3: reference qubit (Gullans and Huse)


def bell_reference_state(N: int, x0: int) -> np.ndarray:
    """Return the Néel state with site ``x0`` in a Bell pair with a reference.

    The returned matrix ``M`` of shape ``(2**N, 2)`` holds the amplitudes
    ``<b, r|psi>`` of the system ``b`` and the reference ``r``: ``(|0>_x0
    |0>_R + |1>_x0 |1>_R) / sqrt 2``, the other sites in ``|0101...>``, so
    ``S_Q = 1`` bit.
    """
    M = np.zeros((2**N, 2), dtype=complex)
    neel = int(np.argmax(np.abs(neel_state(N))))
    mask = 1 << (N - 1 - x0)
    for r in (0, 1):
        index = (neel & ~mask) | (mask if r else 0)
        M[index, r] = 1 / math.sqrt(2)
    return M


def _entropy_2x2(rho: np.ndarray) -> float:
    """Von Neumann entropy (bits) of a 2 x 2 density matrix."""
    values = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    values = values[values > 1e-14]
    return float(-np.sum(values * np.log2(values)))


def reference_entropy(M: np.ndarray) -> float:
    """Return ``S_Q`` (bits) of the reference qubit of ``M``, ``(2**N, 2)``."""
    rho = M.conj().T @ M
    return _entropy_2x2(rho / np.trace(rho).real)


def expected_entropy_after(M: np.ndarray, bit_x: np.ndarray) -> float:
    """Return ``sum_m p_m S_Q(m)`` after a Z measurement of the site ``x``.

    Args:
        M: Amplitudes ``(2**N, 2)``, normalized.
        bit_x: The bit of site ``x`` of every basis index, ``(2**N,)``.
    """
    total = 0.0
    for m in (0, 1):
        M_m = M[bit_x == m]
        p_m = float(np.sum(np.abs(M_m) ** 2))
        if p_m > 1e-14:
            total += p_m * _entropy_2x2(M_m.conj().T @ M_m / p_m)
    return total


@dataclass
class ReferenceQubitResult:
    """Averages over the trajectories of :func:`reference_qubit_trajectories`.

    Attributes:
        S_Q: ``<S_Q(t)>`` after the measurements of step ``t``, ``t = 0 ..
            n_steps`` (``S_Q(0) = 1``), shape ``(n_steps + 1,)``.
        delta_S_Q: ``<Delta S_Q(x, t)>``, shape ``(n_steps + 1, N)``: the mean
            change of ``S_Q`` that a Z measurement of ``x`` would cause after
            the unitary of step ``t`` (``t = 0``: the initial state).
        n_trajectories: Number of trajectories.
    """

    S_Q: np.ndarray
    delta_S_Q: np.ndarray
    n_trajectories: int


def reference_qubit_trajectories(
    steps: Sequence[Step],
    x0: int,
    p: float,
    n_steps: int,
    n_trajectories: int,
    rng: np.random.Generator,
) -> ReferenceQubitResult:
    """Run monitored trajectories with a reference qubit at ``x0``.

    Each step applies the next unitary of ``steps`` to the system, then
    measures every site in Z with probability ``p``, the outcome drawn with
    the Born rule. Before the measurements of step ``t`` (and on the initial
    state for ``t = 0``), ``Delta S_Q(x, t) = sum_m p_m S_Q(m) - S_Q`` is the
    mean change that one extra measurement of ``x`` would cause, averaged
    over its outcome ``m`` and not applied: the trajectory is unbiased. As
    long as no information on ``R`` has reached ``x`` (``x`` outside the
    light cone of ``x0`` at time ``t``), ``rho_{R x}`` is a product and
    ``Delta S_Q(x, t) = 0``.

    Args:
        steps: Dense unitaries on the ``N`` system qubits, cycled.
        x0: Site entangled with the reference (:func:`bell_reference_state`).
        p: Probability to measure each qubit after each step.
        n_steps: Number of steps.
        n_trajectories: Number of trajectories.
        rng: Generator of the measurement locations and outcomes.

    Returns:
        A :class:`ReferenceQubitResult`.
    """
    dim = steps[0].shape[0]
    N = int(round(math.log2(dim)))
    bits = _bits(N)
    S_Q = np.zeros(n_steps + 1)
    delta = np.zeros((n_steps + 1, N))
    for _ in range(n_trajectories):
        M = bell_reference_state(N, x0)
        for t in range(n_steps + 1):
            if t > 0:
                M = steps[(t - 1) % len(steps)] @ M
            before = reference_entropy(M)
            for x in range(N):
                delta[t, x] += expected_entropy_after(M, bits[:, x]) - before
            if t > 0:
                M = _measure(M, bits, rng.random(N) < p, rng)
            S_Q[t] += reference_entropy(M)
    return ReferenceQubitResult(
        S_Q=S_Q / n_trajectories,
        delta_S_Q=delta / n_trajectories,
        n_trajectories=n_trajectories,
    )


def _measure(
    M: np.ndarray, bits: np.ndarray, measured: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Measure the sites ``measured`` of ``M`` in Z, with Born outcomes."""
    for x in np.flatnonzero(measured):
        weight1 = float(np.sum(np.abs(M[bits[:, x] == 1]) ** 2))
        m = int(rng.random() < weight1)
        M = np.where((bits[:, x] == m)[:, None], M, 0)
        M = M / math.sqrt(weight1 if m else 1 - weight1)
    return M


def reference_qubit_cutoff(
    steps: Sequence[Step],
    x0: int,
    p: float,
    n_steps: int,
    cutoffs: Sequence[int],
    n_trajectories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``<S_Q(t)>`` when only the outcomes at ``|x - x0| <= l`` are kept.

    Every qubit is measured with probability ``p`` after each step, as in
    :func:`reference_qubit_trajectories`, but the outcome of a measurement at
    ``|x - x0| > l`` is not recorded: the state is averaged over it (a
    dephasing of ``x``), and ``S_Q`` is the entropy of ``rho_R``
    conditioned on the recorded outcomes only. Forgetting outcomes can only
    raise ``<S_Q>``; if the purification of the area-law phase is local, a
    small ``l`` already gives the curve of ``l = N``. The locations of the
    measurements are the same for every ``l`` (common random numbers).

    The state of the system and reference is a density matrix of dimension
    ``2**(N+1)``, the reference the last (least significant) qubit.

    Args:
        steps: Dense unitaries on the ``N`` system qubits, cycled.
        x0: Site entangled with the reference.
        p: Probability to measure each qubit after each step.
        n_steps: Number of steps.
        cutoffs: Values of ``l``; ``l < 0`` records nothing.
        n_trajectories: Number of trajectories per cutoff.
        rng: Generator of the measurement locations and outcomes.

    Returns:
        Array of shape ``(len(cutoffs), n_steps + 1)``.
    """
    dim = steps[0].shape[0]
    N = int(round(math.log2(dim)))
    # System bits of the 2**(N+1) indices (the reference is the last bit).
    sys_bits = np.repeat(_bits(N), 2, axis=0)
    out = np.zeros((len(cutoffs), n_steps + 1))
    for _ in range(n_trajectories):
        measured = rng.random((n_steps, N)) < p
        seeds = rng.integers(2**63, size=len(cutoffs))
        for k, cutoff in enumerate(cutoffs):
            local = np.random.default_rng(seeds[k])
            psi = bell_reference_state(N, x0).ravel()
            rho = np.outer(psi, psi.conj())
            out[k, 0] += 1.0
            for t in range(n_steps):
                rho = _evolve_with_reference(steps[t % len(steps)], rho)
                for x in np.flatnonzero(measured[t]):
                    b = sys_bits[:, x]
                    if abs(x - x0) <= cutoff:
                        weight1 = float(np.real(np.sum(np.diag(rho)[b == 1])))
                        m = int(local.random() < weight1)
                        keep = b == m
                        rho = rho * np.outer(keep, keep)
                        rho = rho / (weight1 if m else 1 - weight1)
                    else:
                        rho = rho * (b[:, None] == b[None, :])
                out[k, t + 1] += _reference_entropy_rho(rho)
    return out / n_trajectories


def _evolve_with_reference(U: Step, rho: np.ndarray) -> np.ndarray:
    """Return ``(U x 1) rho (U x 1)^+``, the reference the last qubit of ``rho``.

    The rows of ``rho`` are ``(system, reference)``, so ``(U x 1) X`` is ``U``
    applied to ``X`` reshaped to ``(2**N, 2 k)``.
    """
    d2 = rho.shape[0]
    d = d2 // 2

    def left(X: np.ndarray) -> np.ndarray:
        return np.asarray(U @ X.reshape(d, 2 * X.shape[1])).reshape(d2, X.shape[1])

    A = left(rho)
    return left(A.conj().T).conj().T


def _reference_entropy_rho(rho: np.ndarray) -> float:
    """``S_Q`` (bits) of the last qubit of a density matrix."""
    d = rho.shape[0] // 2
    blocks = rho.reshape(d, 2, d, 2)
    rho_R = np.einsum("iaib->ab", blocks)
    return _entropy_2x2(rho_R / np.trace(rho_R).real)
