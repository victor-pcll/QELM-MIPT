"""Exact small-N checks of the general Fourier (Pauli-mode) framework (issue #353).

This module backs the note ``documentation/notes/cadre-fourier-general.md``.
Everything is dense and exact, so it is meant for ``N <= 4``: it enumerates
every record of a monitored brickwork circuit, builds every effect
``E_m = K_m^dagger K_m`` and expands it on the ``4^N`` Pauli strings. Nothing
here assumes Clifford gates: the same functions run on Haar and on Clifford
circuits, which is the point of the note (general statements first, then the
Clifford reduction).

Conventions (those of ``effect_rank`` and ``ridge_spectrum``):

- ``N`` qubits on an open chain, qubit 0 the most significant bit;
- layer ``t`` applies the gates on the even bonds, then on the odd bonds, then
  a ``Z`` measurement of each site ``i`` with ``mask[t, i]``; with
  ``final=True`` every site is read in ``Z`` after the last layer;
- a Pauli string is indexed by its base-4 digits, ``I, X, Y, Z = 0, 1, 2, 3``,
  site 0 first;
- an input is a product state ``rho(x) = prod_i (I + n_i . sigma) / 2``; the
  monomial of string ``a`` is ``B_a(x) = prod_i n_i^{a_i}`` (``n^I = 1``), and
  ``Pr(m | x) = sum_a e_a(m) B_a(x)`` with ``e_a = Tr[E_m P_a] / 2^N``.

Input ensembles (both are invariant under conjugation by any Pauli string):

- ``"xz"``: the four states ``|0>, |1>, |+>, |->`` per site (same first and
  second moments as an angle uniform on the XZ circle):
  ``E[B_a B_b] = 2^{-|a|} delta_ab`` if ``a`` has no ``Y``, and ``B_a = 0``
  otherwise;
- ``"sphere"``: the six Pauli eigenstates per site (same moments as the
  uniform Bloch sphere): ``E[B_a B_b] = 3^{-|a|} delta_ab``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from QELM.models.circuits.clifford import random_clifford_gate
from QELM.models.circuits.haar import haar_gate

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
# Singular values below this are round-off: features are probabilities, so a
# genuine direction of C (or C W^{1/2}) is far above it at N <= 4.
_ABS_TOL = 1e-12

PAULIS: tuple[np.ndarray, ...] = (_I2, _X, _Y, _Z)

_BLOCH_XZ = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]], dtype=float)
_BLOCH_SPHERE = np.array(
    [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Pauli strings
# ---------------------------------------------------------------------------


def pauli_codes(n: int) -> np.ndarray:
    """All ``4^n`` Pauli strings as an ``(4^n, n)`` array of codes ``0..3``."""
    return np.array(list(product(range(4), repeat=n)), dtype=np.int8).reshape(-1, n)


def pauli_matrix(code: Sequence[int]) -> np.ndarray:
    """Dense matrix of one Pauli string (qubit 0 most significant)."""
    out = np.ones((1, 1), dtype=complex)
    for c in code:
        out = np.kron(out, PAULIS[int(c)])
    return out


def pauli_vectors(ops: np.ndarray) -> np.ndarray:
    """Coefficients ``Tr[op P_a] / 2^n`` of a batch of operators.

    Args:
        ops: ``(m, 2^n, 2^n)`` complex array (or a single ``(2^n, 2^n)``).

    Returns:
        Complex ``(m, 4^n)`` array (real for Hermitian ``ops``), strings in the
        order of :func:`pauli_codes`.
    """
    ops = np.asarray(ops, dtype=complex)
    single = ops.ndim == 2
    if single:
        ops = ops[None]
    m, dim, _ = ops.shape
    n = dim.bit_length() - 1
    # T[row bits..., col bits...] -> pairs (row_i, col_i) per site.
    t = ops.reshape((m,) + (2,) * (2 * n))
    order = [0] + [1 + k for site in range(n) for k in (site, n + site)]
    t = t.transpose(order).reshape((m,) + (4,) * n)
    # Tr[op sigma] = sum_{r,c} op[r, c] sigma[c, r]; per-site map (4 <- 4).
    to_pauli = np.array([s.T.reshape(-1) for s in PAULIS]) / 2.0
    for site in range(n):
        t = np.moveaxis(np.tensordot(to_pauli, t, axes=(1, 1 + site)), 0, 1 + site)
    out = t.reshape(m, 4**n)
    return out[0] if single else out


def pauli_weight(codes: np.ndarray) -> np.ndarray:
    """Number of non-identity sites of each string."""
    return np.count_nonzero(np.atleast_2d(codes), axis=1)


def visible_mask(n: int, ensemble: str = "xz") -> np.ndarray:
    """Strings with a non-zero monomial for the ensemble (no ``Y`` for ``xz``)."""
    codes = pauli_codes(n)
    if ensemble == "xz":
        return np.all(codes != 2, axis=1)
    if ensemble == "sphere":
        return np.ones(len(codes), dtype=bool)
    raise ValueError(f"unknown ensemble {ensemble!r}")


def input_weights(n: int, ensemble: str = "xz") -> np.ndarray:
    """Weights ``w_a = E[B_a^2]`` of the monomials, ``w_I = 1``."""
    base = {"xz": 2.0, "sphere": 3.0}[ensemble]
    w = base ** (-pauli_weight(pauli_codes(n)).astype(float))
    return np.where(visible_mask(n, ensemble), w, 0.0)


def pauli_transfer_matrix(u: np.ndarray) -> np.ndarray:
    """Heisenberg Pauli transfer matrix ``R[a, b] = Tr[P_b U^dag P_a U] / 2^n``.

    ``U^dag P_a U = sum_b R[a, b] P_b``. ``R`` is real orthogonal for every
    unitary; it is a signed permutation exactly when ``U`` is Clifford.
    """
    n = u.shape[0].bit_length() - 1
    paulis = np.array([pauli_matrix(c) for c in pauli_codes(n)])
    conj = np.einsum("ji,ajk,kl->ail", u.conj(), paulis, u)
    return np.asarray(pauli_vectors(conj).real)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """A finite, exactly uniform input ensemble.

    Attributes:
        bloch: ``(n_inputs, n, 3)`` Bloch vectors.
        monomials: ``(n_inputs, 4^n)`` values ``B_a(x)``.
        ensemble: ``"xz"`` or ``"sphere"``.
    """

    bloch: np.ndarray
    monomials: np.ndarray
    ensemble: str


def eigenstate_inputs(n: int, ensemble: str = "xz") -> Inputs:
    """Every product of Pauli eigenstates of the ensemble (``4^n`` or ``6^n``)."""
    states = {"xz": _BLOCH_XZ, "sphere": _BLOCH_SPHERE}[ensemble]
    idx = np.array(list(product(range(len(states)), repeat=n))).reshape(-1, n)
    bloch = states[idx]  # (n_inputs, n, 3)
    ext = np.concatenate([np.ones(bloch.shape[:2] + (1,)), bloch], axis=2)
    codes = pauli_codes(n)
    mono = np.ones((len(idx), len(codes)))
    for site in range(n):
        mono *= ext[:, site, codes[:, site]]
    return Inputs(bloch=bloch, monomials=mono, ensemble=ensemble)


def density_matrices(inputs: Inputs) -> np.ndarray:
    """``(n_inputs, 2^n, 2^n)`` product states ``rho(x)``."""
    out = []
    for vecs in inputs.bloch:
        rho = np.ones((1, 1), dtype=complex)
        for v in vecs:
            rho = np.kron(rho, (_I2 + v[0] * _X + v[1] * _Y + v[2] * _Z) / 2)
        out.append(rho)
    return np.array(out)


# ---------------------------------------------------------------------------
# Circuits, records and effects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Circuit:
    """Open brickwork with a fixed measurement mask.

    Attributes:
        n: Number of qubits.
        gates: ``gates[t]`` is the list of ``(site, 4 x 4 gate)`` of layer
            ``t`` (even bonds first, then odd bonds); the gate acts on
            ``(site, site + 1)``.
        mask: ``(T, n)`` boolean, ``mask[t, i]``: site ``i`` measured after
            layer ``t``.
        kind: ``"haar"`` or ``"clifford"``.
    """

    n: int
    gates: tuple[tuple[tuple[int, np.ndarray], ...], ...]
    mask: np.ndarray
    kind: str


def draw_circuit(
    n: int, depth: int, p: float, kind: str, rng: np.random.Generator
) -> Circuit:
    """Draw gates (Haar or uniform two-qubit Clifford) and a mask ``u < p``."""
    draw = {"haar": haar_gate, "clifford": random_clifford_gate}[kind]
    bonds = list(range(0, n - 1, 2)) + list(range(1, n - 1, 2))
    gates = tuple(tuple((b, draw(rng)) for b in bonds) for _ in range(depth))
    mask = rng.random((depth, n)) < p
    return Circuit(n=n, gates=gates, mask=mask, kind=kind)


def with_mask(circuit: Circuit, mask: np.ndarray) -> Circuit:
    """Same gates, another mask."""
    return Circuit(circuit.n, circuit.gates, np.asarray(mask, dtype=bool), circuit.kind)


def _embed(gate: np.ndarray, site: int, n: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2**site), gate), np.eye(2 ** (n - site - 2)))


def _projector(site: int, outcome: int, n: int) -> np.ndarray:
    diag = np.ones(2**n)
    bits = (np.arange(2**n) >> (n - 1 - site)) & 1
    diag[bits != outcome] = 0.0
    return np.diag(diag).astype(complex)


def n_bits(circuit: Circuit, final: bool = True) -> int:
    """Number of measured bits in a record."""
    return int(circuit.mask.sum()) + (circuit.n if final else 0)


@dataclass(frozen=True)
class Records:
    """Every record of a circuit with its Kraus operator.

    Attributes:
        bits: ``(n_records, n_bits)`` outcomes ``0/1`` in time order (mid bits
            layer by layer, then the final readout if any).
        kraus: ``(n_records, 2^n, 2^n)`` operators ``K_m`` with
            ``Pr(m | rho) = Tr[K_m rho K_m^dag]``.
    """

    bits: np.ndarray
    kraus: np.ndarray

    @property
    def effects(self) -> np.ndarray:
        """``E_m = K_m^dag K_m``."""
        return np.einsum("mji,mjk->mik", self.kraus.conj(), self.kraus)


def enumerate_records(
    circuit: Circuit, final: bool = True, max_bits: int = 16
) -> Records:
    """Enumerate the ``2^(n_bits)`` records and their Kraus operators.

    Args:
        circuit: The circuit.
        final: Append the final ``Z`` readout of every site.
        max_bits: Refuse circuits with more bits (memory guard).
    """
    n = circuit.n
    nb = n_bits(circuit, final)
    if nb > max_bits:
        raise ValueError(f"{nb} bits > max_bits = {max_bits}")
    branches: list[tuple[tuple[int, ...], np.ndarray]] = [
        ((), np.eye(2**n, dtype=complex))
    ]
    for t, layer in enumerate(circuit.gates):
        u = np.eye(2**n, dtype=complex)
        for site, gate in layer:
            u = _embed(gate, site, n) @ u
        branches = [(b, u @ k) for b, k in branches]
        for site in (int(s) for s in np.flatnonzero(circuit.mask[t])):
            proj = [_projector(int(site), o, n) for o in (0, 1)]
            branches = [(b + (o,), proj[o] @ k) for b, k in branches for o in (0, 1)]
    if final:
        for site in range(n):
            proj = [_projector(site, o, n) for o in (0, 1)]
            branches = [(b + (o,), proj[o] @ k) for b, k in branches for o in (0, 1)]
    bits = np.array([b for b, _ in branches], dtype=np.int8).reshape(len(branches), nb)
    kraus = np.array([k for _, k in branches])
    return Records(bits=bits, kraus=kraus)


def feature_matrix(effects: np.ndarray) -> np.ndarray:
    """Real ``(n_records, 4^n)`` matrix ``C[m, a] = Tr[E_m P_a] / 2^n``.

    ``Pr(m | x) = C[m] @ B(x)``: row ``m`` is the Fourier (Pauli) transform of
    the feature ``x -> Pr(m | x)``.
    """
    return np.asarray(pauli_vectors(effects).real)


# ---------------------------------------------------------------------------
# Record parities (Walsh modes) and readouts of order k
# ---------------------------------------------------------------------------


def parity_subsets(n_bits_: int, k_max: int | None = None) -> list[tuple[int, ...]]:
    """Non-empty subsets of the bits, of size at most ``k_max``."""
    k_max = n_bits_ if k_max is None else min(k_max, n_bits_)
    return [s for k in range(1, k_max + 1) for s in combinations(range(n_bits_), k)]


def characters(bits: np.ndarray, subsets: Sequence[tuple[int, ...]]) -> np.ndarray:
    """``(n_subsets, n_records)`` Walsh characters ``(-1)^{sum_{i in S} m_i}``."""
    out = np.empty((len(subsets), len(bits)))
    for j, s in enumerate(subsets):
        out[j] = (-1.0) ** bits[:, list(s)].sum(axis=1)
    return out


def walsh_operators(records: Records, subsets: Sequence[tuple[int, ...]]) -> np.ndarray:
    """``O_S = sum_m (-1)^{m_S} E_m``, so ``E[(-1)^{m_S} | x] = Tr[rho(x) O_S]``."""
    chi = characters(records.bits, subsets)
    return np.einsum("sm,mij->sij", chi, records.effects)


def parity_features(c: np.ndarray, bits: np.ndarray, k_max: int | None) -> np.ndarray:
    """Pauli rows of the parity means of at most ``k_max`` bits (``K = inf``)."""
    subsets = parity_subsets(bits.shape[1], k_max)
    return characters(bits, subsets) @ c


# ---------------------------------------------------------------------------
# Population ridge in the Fourier basis
# ---------------------------------------------------------------------------


def target_variance(beta: np.ndarray, w: np.ndarray) -> float:
    """``Var y`` of ``y = sum_{a != I} beta_a B_a``: ``sum_a w_a beta_a^2``."""
    return float(np.sum(w[1:] * beta[1:] ** 2))


def population_covariance(c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Centred covariance ``Sigma = C W C^T`` of the features (``a != I``)."""
    return np.asarray((c[:, 1:] * w[1:]) @ c[:, 1:].T)


def _whitened_filter(
    c: np.ndarray, w: np.ndarray, beta: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Target ``g = W^{1/2} beta`` and its ridge image, in whitened Pauli space.

    With ``A = C W^{1/2} = U s V^T`` (columns ``a != I``), ``Sigma = A A^T``
    and the population ridge maps ``g`` to ``V s^2 / (s^2 + lam) V^T g``;
    ``lam = 0`` is the ``lambda -> 0+`` limit, the projection ``V V^T``. The
    SVD keeps the cost at ``(rank) x 4^n`` whatever the number of records.
    """
    sw = np.sqrt(w[1:])
    a = c[:, 1:] * sw
    g = sw * beta[1:]
    if not np.any(a):
        return g, np.zeros_like(g)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    keep = s > max(1e-10 * s[0], _ABS_TOL)
    s, vt = s[keep], vt[keep]
    gain = np.ones_like(s) if lam == 0.0 else s**2 / (s**2 + lam)
    return g, vt.T @ (gain * (vt @ g))


def population_error(
    c: np.ndarray, w: np.ndarray, beta: np.ndarray, lam: float = 0.0
) -> float:
    """Population error ``1 - R^2`` of the ridge on features ``C`` (``K = inf``).

    With ``lam = 0`` this is the ``lambda -> 0+`` limit, the weighted
    projection of the target on the span of the feature functions: ``E_perp``.
    The intercept is free (unpenalised). Divided by ``Var y``.
    """
    g, img = _whitened_filter(c, w, beta, lam)
    return float(np.sum((g - img) ** 2) / np.sum(g**2))


def population_prediction(
    c: np.ndarray, w: np.ndarray, beta: np.ndarray, lam: float = 0.0
) -> np.ndarray:
    """Pauli coefficients (``a != I``) of the population-ridge prediction."""
    _, img = _whitened_filter(c, w, beta, lam)
    sw = np.sqrt(w[1:])
    out = np.zeros_like(img)
    np.divide(img, sw, out=out, where=sw > 0)
    return out


@dataclass(frozen=True)
class TwoFilters:
    """Population split of the error of a readout ``F`` nested in ``S``.

    All terms divided by ``Var y``; ``total = circuit + readout + cross``.
    """

    total: float
    circuit: float
    readout: float
    cross: float


def two_filter_split(
    c_small: np.ndarray,
    c_big: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    lam: float = 0.0,
) -> TwoFilters:
    """Split the population error of the ridge on ``c_small`` against ``c_big``.

    ``circuit = ||y - R_S y||^2``, ``readout = ||R_S y - R_F y||^2``,
    ``cross = 2 <y - R_S y, R_S y - R_F y>``, in ``L^2`` of the input ensemble
    (weights ``w`` on the Pauli coefficients).
    """
    var = target_variance(beta, w)
    y = beta[1:]
    pf = population_prediction(c_small, w, beta, lam)
    ps = population_prediction(c_big, w, beta, lam)
    wv = w[1:]

    def ip(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.sum(wv * u * v) / var)

    return TwoFilters(
        total=ip(y - pf, y - pf),
        circuit=ip(y - ps, y - ps),
        readout=ip(ps - pf, ps - pf),
        cross=2 * ip(y - ps, ps - pf),
    )


# ---------------------------------------------------------------------------
# Finite shots: the canonical-correlation filter
# ---------------------------------------------------------------------------


def _one_shot_noise(c: np.ndarray) -> np.ndarray:
    """Covariance ``M = diag(fbar) - fbar fbar^T`` of one record indicator."""
    fbar = c[:, 0]
    return np.asarray(np.diag(fbar) - np.outer(fbar, fbar))


def histogram_r2(c: np.ndarray, w: np.ndarray, beta: np.ndarray, shots: float) -> float:
    """Population ``R^2`` of the best linear readout of the ``K``-shot histogram.

    ``Cov(h_K) = (1 - 1/K) Sigma + M / K`` and ``Cov(h_K, y) = C W beta``
    (infinitely many training inputs, ``lambda -> 0+``).
    """
    sigma = population_covariance(c, w)
    cov = (1 - 1 / shots) * sigma + _one_shot_noise(c) / shots
    b = c[:, 1:] @ (w[1:] * beta[1:])
    r = b @ np.linalg.pinv(cov, rcond=1e-10, hermitian=True) @ b
    return float(r / target_variance(beta, w))


@dataclass(frozen=True)
class CanonicalSpectrum:
    """Squared canonical correlations ``mu_k`` of one shot with the input.

    Attributes:
        mu: Eigenvalues in ``[0, 1]``, decreasing.
        directions: Matching eigenvectors in the whitened feature space.
        whitener: Map ``M^{+1/2}`` (pseudo-inverse square root of ``M``).
    """

    mu: np.ndarray
    directions: np.ndarray
    whitener: np.ndarray


def canonical_spectrum(
    c: np.ndarray, w: np.ndarray, tol: float = 1e-12
) -> CanonicalSpectrum:
    """Eigen-decomposition of ``M^{+1/2} Sigma M^{+1/2}``."""
    noise = _one_shot_noise(c)
    val, vec = np.linalg.eigh(noise)
    keep = val > tol * max(val.max(), 1.0)
    whit = (vec[:, keep] / np.sqrt(val[keep])) @ vec[:, keep].T
    sig = whit @ population_covariance(c, w) @ whit
    mu, dirs = np.linalg.eigh((sig + sig.T) / 2)
    order = np.argsort(mu)[::-1]
    return CanonicalSpectrum(
        mu=np.clip(mu[order], 0.0, None), directions=dirs[:, order], whitener=whit
    )


def spectral_r2(c: np.ndarray, w: np.ndarray, beta: np.ndarray, shots: float) -> float:
    """Spectral form ``R^2(K) = sum_k z_k^2 K mu_k / (1 + (K - 1) mu_k)``."""
    spec = canonical_spectrum(c, w)
    b = spec.whitener @ (c[:, 1:] @ (w[1:] * beta[1:]))
    proj = (spec.directions.T @ b) ** 2
    good = spec.mu > 1e-12
    z2 = proj[good] / spec.mu[good]
    mu = spec.mu[good]
    r = np.sum(z2 * shots * mu / (1 + (shots - 1) * mu))
    return float(r / target_variance(beta, w))


# ---------------------------------------------------------------------------
# Structure of the effects: span, commutation, purification spectrum
# ---------------------------------------------------------------------------


def span_rank(
    c: np.ndarray, columns: np.ndarray | None = None, tol: float = 1e-9
) -> int:
    """Rank of the Pauli rows (dense rank), optionally restricted to ``columns``."""
    m = c if columns is None else c[:, columns]
    if m.size == 0:
        return 0
    s = np.linalg.svd(m, compute_uv=False)
    return int(np.sum(s > max(tol * s[0], _ABS_TOL)))


def span_operators(c: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal Hermitian basis ``(r, 2^n, 2^n)`` of ``Span{E_m}``."""
    n = c.shape[1].bit_length() // 2
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    basis = vt[s > max(tol * s[0], _ABS_TOL)]
    paulis = np.array([pauli_matrix(code) for code in pauli_codes(n)])
    return np.asarray(np.einsum("ra,aij->rij", basis, paulis) / np.sqrt(2**n))


def max_commutator(c: np.ndarray) -> float:
    """``max ||[A, B]||_F`` over an orthonormal basis of ``Span{E_m}``.

    Zero exactly when all the effects commute (the span is then a
    commutative algebra). ``c`` is the :func:`feature_matrix` of the effects.
    """
    ops = span_operators(c)
    worst = 0.0
    for a in ops:
        prod_ = np.einsum("ij,bjk->bik", a, ops)
        comm = prod_ - np.einsum("bij,jk->bik", ops, a)
        worst = max(worst, float(np.max(np.linalg.norm(comm, axis=(1, 2)))))
    return worst


def participation(c_row: np.ndarray) -> float:
    """Inverse participation ratio ``(sum c^2)^2 / sum c^4`` of one Pauli row.

    1 for a single Pauli string, up to ``4^n`` for a flat spread.
    """
    c2 = c_row**2
    s = c2.sum()
    return float(s**2 / np.sum(c2**2)) if s > 0 else 0.0


@dataclass(frozen=True)
class PurificationSpectrum:
    """Born-averaged Renyi entropies of the mid-record state ``E_m / Tr E_m``.

    ``E_m / Tr E_m`` has the spectrum of the state ``K_m K_m^dag / Tr`` reached
    from the maximally mixed input, so these are the purification entropies
    of the record: ``renyi0 = log2 rank`` (Hartley), ``renyi1`` (von Neumann),
    ``renyi2`` (collision). All three coincide when the spectrum is flat.
    """

    renyi0: float
    renyi1: float
    renyi2: float


def purification_spectrum(
    effects: np.ndarray, tol: float = 1e-10
) -> PurificationSpectrum:
    """Born weights ``Tr E_m / 2^n`` (maximally mixed input) and Renyi entropies."""
    dim = effects.shape[1]
    r0 = r1 = r2 = 0.0
    for e in effects:
        tr = float(np.trace(e).real)
        if tr <= tol:
            continue
        lam = np.clip(np.linalg.eigvalsh(e / tr), 0.0, None)
        lam = lam[lam > tol]
        weight = tr / dim
        r0 += weight * np.log2(len(lam))
        r1 += weight * float(-np.sum(lam * np.log2(lam)))
        r2 += weight * float(-np.log2(np.sum(lam**2)))
    return PurificationSpectrum(renyi0=r0, renyi1=r1, renyi2=r2)


# ---------------------------------------------------------------------------
# Information beyond the linear readout
# ---------------------------------------------------------------------------


def fibre_error(features: np.ndarray, y: np.ndarray, decimals: int = 9) -> float:
    """Error ``1 - R^2`` of the best decoder of any form on a finite ensemble.

    The best decoder of ``y`` from exact (``K = inf``) features is the
    conditional mean of ``y`` on each fibre (set of inputs with the same
    feature vector). Divided by ``Var y``.
    """
    keys = np.round(features, decimals) + 0.0
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    sums = np.bincount(inv, weights=y)
    counts = np.bincount(inv)
    pred = (sums / counts)[inv]
    return float(np.mean((y - pred) ** 2) / np.var(y))


def n_distinct_features(features: np.ndarray, decimals: int = 9) -> int:
    """Number of distinct feature vectors over a finite ensemble."""
    keys = np.round(features, decimals) + 0.0
    return int(len(np.unique(keys, axis=0)))


def anticommuting_centralizer_element(
    c: np.ndarray, target: int, n: int, tol: float = 1e-9
) -> np.ndarray | None:
    """A Pauli string commuting with every effect and anticommuting with ``target``.

    It exists, for a Clifford circuit, exactly when ``target`` is outside the
    effect group ``G`` (double centralizer). Returns its codes or ``None``.
    """
    codes = pauli_codes(n)
    support = np.flatnonzero(np.any(np.abs(c) > tol, axis=0))

    def anticommute(a: np.ndarray, b: np.ndarray) -> bool:
        clash = (a != 0) & (b != 0) & (a != b)
        return bool(np.count_nonzero(clash) % 2)

    t = codes[target]
    for q in codes:
        if not anticommute(q, t):
            continue
        if all(not anticommute(q, codes[s]) for s in support):
            return q
    return None


def conjugate_inputs(inputs: Inputs, q: Sequence[int]) -> np.ndarray:
    """Index map ``x -> Q x Q`` on a Pauli-eigenstate ensemble.

    Conjugation by ``X`` flips the ``Y`` and ``Z`` Bloch components, by ``Y``
    flips ``X`` and ``Z``, by ``Z`` flips ``X`` and ``Y``.
    """
    flips = {0: (1, 1, 1), 1: (1, -1, -1), 2: (-1, 1, -1), 3: (-1, -1, 1)}
    new = inputs.bloch * np.array([flips[int(c)] for c in q])[None]
    flat_old = {
        tuple(np.round(b, 9).ravel() + 0.0): i for i, b in enumerate(inputs.bloch)
    }
    return np.array([flat_old[tuple(np.round(b, 9).ravel() + 0.0)] for b in new])
