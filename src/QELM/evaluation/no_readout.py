"""Delta, S_2(N/2), I3 without final readout, from the Fourier weights (#378).

Circuit: the periodic brickwork of :mod:`QELM.evaluation.haar_mipt` (Haar
gates) or of :mod:`QELM.evaluation.clifford_ridge` (Clifford gates), a ``Z``
measurement of each site with probability ``p`` after every half-layer, and
**no final readout**. Each site ``i`` carries one reference (input) qubit.

For one circuit and one mid-circuit record ``m``, ``E = K_m^dagger K_m`` is
the effect of the record and ``rho = E / Tr E`` the state of the
references (up to a transposition, which leaves every purity unchanged).
On the Pauli strings ``P`` of the references,

    rho = 2^-N sum_P c_P P,    c_I = 1,

and, for a region ``A`` of references,

    1 + W_A = sum_{supp(P) in A} c_P^2,    S_2(A) = |A| - log2(1 + W_A).

``W = W_all`` is the Fourier weight of #314 and ``Delta = S_2(all)``. Every
observable below is an ``S_2`` read from these weights, so only the Renyi-2
entropy is available.

Per circuit and checkpoint, :data:`OBSERVABLES`:

- ``delta``: ``S_2`` of all the references;
- ``s_half``: ``S_2`` of the ``N / 2`` consecutive references starting at
  each site, averaged over the ``N`` starts;
- ``i3``: ``S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC`` of the quarters
  ``A, B, C`` starting at each site (``D`` the rest), averaged over the
  ``N`` starts (``nan`` unless ``4 | N``). The state is mixed, so the
  quarter left out matters and all ``N`` starts are kept, not ``N / 4``;
- ``s_one``: ``S_2`` of one reference, averaged over the sites, a variant
  of the reference-qubit probe ``S_Q`` of Gullans & Huse, PRL 2020 (there,
  the other inputs are pure and the circuit starts with ``2L`` unmeasured
  steps).

**Haar.** The ``4^N`` coefficients ``c_P`` are computed exactly: for a
string ``X^x Z^z``, ``|c| = |sum_i (-1)^{z.i} rho[i, i xor x]|``, a
Walsh-Hadamard transform for each ``x``. The ``c_P^2`` are summed by support
(``x | z``), then a subset-sum (zeta) transform gives ``1 + W_A`` for all
``2^N`` regions at once.

**Clifford.** ``rho`` is a normalised stabilizer projector: ``c_P = +-1`` on
the group ``G`` of the effect (rank ``r = N - Delta``) and 0 elsewhere. So
``1 + W_A = 2^{k_A}`` with ``k_A`` the rank of the subgroup supported in
``A``, ``k_A = r - rank(G restricted to the complement of A)`` over GF(2).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import stim  # type: ignore[import-untyped]

from QELM.evaluation.clifford_ridge import (
    CliffordQELM,
    _reference_group,
    body_program,
    draw_circuit,
)
from QELM.evaluation.haar_mipt import evolve_kraus
from QELM.models.circuits.stabilizer import stabilizer_matrix

OBSERVABLES = ("delta", "s_half", "i3", "s_one")

Region = tuple[int, ...]


def interval(start: int, length: int, n: int) -> Region:
    """Sites ``start, start + 1, ...`` (``length`` of them) on the ring."""
    return tuple(sorted((start + k) % n for k in range(length)))


def observables(n: int, entropy: Callable[[Region], float]) -> np.ndarray:
    """``(delta, s_half, i3, s_one)`` from the ``S_2`` of each region.

    Args:
        n: Number of sites, even.
        entropy: ``S_2`` of a region (sorted tuple of sites), in bits.
    """
    delta = entropy(tuple(range(n)))
    s_half = float(np.mean([entropy(interval(s, n // 2, n)) for s in range(n)]))
    s_one = float(np.mean([entropy((s,)) for s in range(n)]))
    i3 = float("nan")
    if n % 4 == 0:
        q = n // 4
        vals = []
        for s in range(n):
            a, b, c = (interval(s + k * q, q, n) for k in range(3))
            vals.append(
                entropy(a)
                + entropy(b)
                + entropy(c)
                - entropy(interval(s, 2 * q, n))
                - entropy(tuple(sorted(a + c)))
                - entropy(interval(s + q, 2 * q, n))
                + entropy(interval(s, 3 * q, n))
            )
        i3 = float(np.mean(vals))
    return np.array([delta, s_half, i3, s_one])


# ---------------------------------------------------------------------------
# Haar: the exact Pauli spectrum of rho
# ---------------------------------------------------------------------------


def _walsh_hadamard(a: np.ndarray) -> None:
    """In place, unnormalised Walsh-Hadamard transform along the last axis."""
    dim = a.shape[-1]
    h = 1
    while h < dim:
        v = a.reshape(*a.shape[:-1], dim // (2 * h), 2, h)
        x = v[..., 0, :].copy()
        v[..., 0, :] += v[..., 1, :]
        v[..., 1, :] *= -1
        v[..., 1, :] += x
        h *= 2


def region_weights(rho: np.ndarray, n: int) -> np.ndarray:
    """``1 + W_A`` for every region ``A``, from a density matrix.

    Args:
        rho: ``2^N x 2^N`` matrix of trace 1 (any Hermitian matrix of the
            references, or its transpose), qubit 0 the most significant bit.
        n: Number of qubits.

    Returns:
        Real array of length ``2^N``: entry ``mask`` is ``1 + W_A`` for the
        region ``A`` of the sites ``s`` with bit ``N - 1 - s`` set in
        ``mask``. Entry 0 is 1, the last entry ``1 + W``.
    """
    dim = 2**n
    i = np.arange(dim)
    # f[x, i] = rho[i, i xor x]; its transform over i gives c(x, z).
    f = rho[i[None, :], i[None, :] ^ i[:, None]]
    _walsh_hadamard(f)
    c2 = np.abs(f).astype(np.float64) ** 2
    support = (i[:, None] | i[None, :]).ravel()
    w = np.bincount(support, weights=c2.ravel(), minlength=dim)
    for bit in range(n):  # zeta transform: sum over the subsets of each mask
        v = w.reshape(-1, 2, 2**bit)
        v[:, 1, :] += v[:, 0, :]
    return w


def mask_of(region: Region, n: int) -> int:
    """Bit mask of a region in the convention of :func:`region_weights`."""
    return sum(1 << (n - 1 - s) for s in region)


def entropies_from_weights(weights: np.ndarray, n: int) -> Callable[[Region], float]:
    """``S_2(A) = |A| - log2(1 + W_A)`` as a function of the region."""

    def entropy(region: Region) -> float:
        return float(len(region) - np.log2(weights[mask_of(region, n)]))

    return entropy


def haar_sample(
    n: int,
    p: float,
    checkpoints: Sequence[int],
    rng: np.random.Generator,
    dtype: type = np.complex128,
) -> np.ndarray:
    """One Haar circuit and Born record: :data:`OBSERVABLES` at each checkpoint.

    The circuit and record are those of
    :func:`QELM.evaluation.haar_mipt.haar_record` with the same ``rng``.

    Returns:
        Array ``(len(checkpoints), 4)``.
    """
    if n < 4 or n % 2:
        raise ValueError(f"n must be even and >= 4, got {n}.")
    out = np.zeros((len(checkpoints), len(OBSERVABLES)))
    for i, k, _ in evolve_kraus(n, p, checkpoints, rng, dtype):
        m = k[np.flatnonzero(np.any(k != 0, axis=1))]
        e = m.conj().T @ m  # E / 2^N up to the norm: rho of the inputs (^T)
        e /= np.trace(e).real
        weights = region_weights(e, n)
        out[i] = observables(n, entropies_from_weights(weights, n))
    return out


# ---------------------------------------------------------------------------
# Clifford: ranks over GF(2) of the group of the effect
# ---------------------------------------------------------------------------


def _columns(group: np.ndarray, n: int) -> list[tuple[int, int]]:
    """X and Z columns of each site, one bit per generator, as Python ints."""
    weights = [1 << k for k in range(group.shape[0])]

    def column(c: int) -> int:
        return sum(w for w, bit in zip(weights, group[:, c], strict=True) if bit)

    return [(column(q), column(n + q)) for q in range(n)]


def _insert(pivots: dict[int, int], v: int) -> None:
    """Add ``v`` to an XOR basis kept by leading bit."""
    while v:
        top = v.bit_length() - 1
        if top not in pivots:
            pivots[top] = v
            return
        v ^= pivots[top]


def clifford_entropies(group: np.ndarray, n: int) -> dict[Region, float]:
    """``S_2`` of every region used by :func:`observables`.

    Args:
        group: ``[x | z]`` generators of the group ``G`` of the effect,
            shape ``(r, 2 N)``, independent.
        n: Number of sites.

    Returns:
        ``{region: |A| - k_A}``, with ``k_A = r - rank(G restricted to the
        complement of A)``. Intervals come from one pass per start of the
        complement, the rank being updated site by site; ``A + C`` of
        :func:`observables` (complement ``B + D``) is reduced on its own.
    """
    r = group.shape[0]
    cols = _columns(group, n)
    out: dict[Region, float] = {tuple(range(n)): float(n - r)}
    lengths = {n // 2, n - 1}
    if n % 4 == 0:
        lengths |= {n // 4, 3 * n // 4}
    for u in range(n):  # the complement is the interval [u, u + ell)
        pivots: dict[int, int] = {}
        for ell in range(1, n):
            if len(pivots) < r:  # once the rank is r, it stays r
                for v in cols[(u + ell - 1) % n]:
                    _insert(pivots, v)
            if ell in lengths:
                region = interval(u + ell, n - ell, n)
                out[region] = float(len(region) - (r - len(pivots)))
    if n % 4 == 0:
        q = n // 4
        for s in range(n // 2):  # A + C at s and at s + 2q is the same set
            a = interval(s, q, n)
            c = interval(s + 2 * q, q, n)
            region = tuple(sorted(a + c))
            pivots = {}
            for site in interval(s + q, q, n) + interval(s + 3 * q, q, n):
                if len(pivots) == r:
                    break
                for v in cols[site]:
                    _insert(pivots, v)
            out[region] = float(len(region) - (r - len(pivots)))
    return out


def clifford_sample(
    n: int, p: float, checkpoints: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    """One Clifford circuit: :data:`OBSERVABLES` at each checkpoint.

    Gates from :func:`~QELM.evaluation.clifford_ridge.draw_circuit`, each
    site measured after each half-layer with probability ``p``, references
    in Bell pairs with the sites at ``t = 0``. The groups, hence every
    observable, do not depend on the outcomes.

    Returns:
        Array ``(len(checkpoints), 4)``.
    """
    cps = list(checkpoints)
    if any(c % 2 for c in cps):
        raise ValueError(f"Checkpoints must be even (bond parity), got {cps}.")
    t_max = cps[-1]
    gates = draw_circuit(n, t_max, rng)
    measured = rng.random((t_max, n)) < p
    sim = stim.TableauSimulator(seed=int(rng.integers(2**31)))
    prep = "\n".join(f"H {n + i}\nCX {n + i} {i}" for i in range(n))
    sim.do_circuit(stim.Circuit(prep))
    out = np.zeros((len(cps), len(OBSERVABLES)))
    t0 = 0
    for i, t1 in enumerate(cps):
        seg = CliffordQELM(gates=gates[t0:t1], measured=measured[t0:t1])
        sim.do_circuit(stim.Circuit(body_program(seg, final=False)))
        group = _reference_group(stabilizer_matrix(sim), n)
        table = clifford_entropies(group, n)
        out[i] = observables(n, table.__getitem__)
        t0 = t1
    return out
