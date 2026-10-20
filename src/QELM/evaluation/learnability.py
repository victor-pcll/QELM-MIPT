"""Learning the global charge from local measurements (Barratt et al. 2022).

This is the setup of F. Barratt, U. Agrawal, A. C. Potter, S. Gopalakrishnan
and R. Vasseur, "Transitions in the learnability of global charges from
local measurements", Phys. Rev. Lett. 129, 200602 (2022), arXiv:2206.12429,
used for the benchmarks of their Fig. 2b (issue #113) and Fig. 2c (issue
#115).

A chain of ``L`` qubits starts in the uniform superposition of the
bitstrings of charge ``Q = sum_i b_i``, with ``Q`` one of two values
(``Q0 = L // 2`` and ``Q1 = L // 2 - 1`` by default). It evolves for ``T``
time steps (``T = L`` by default). One step is a full period of a
brickwork of random U(1) gates (:mod:`QELM.models.circuits`, even then odd
bonds), after which every site is measured in the computational basis with
probability ``p``. Each record is drawn with its Born probability, for a
fresh circuit; half of the records start at ``Q0``, half at ``Q1``.

An eavesdropper, Eve, sees the record ``m`` (which sites were measured and
the outcomes) but not the gates, and guesses ``Q``. Averaged over the Haar
gates, the Born probability of a record is a classical symmetric exclusion
process (SEP): each gate becomes the stochastic matrix that keeps a
configuration with probability 1/2 and swaps the two bits with probability
1/2 (a no-op on ``00`` and ``11``), and each measurement the projector on
the configurations that agree with its outcome. So::

    P(m | Q) = (1| prod_t T(m_t) |Q),

with ``|Q)`` the uniform distribution over the bitstrings of charge ``Q``,
and Eve's optimal (Bayes) guess uses the posterior
``P(Q | m) = P(m | Q) / sum_Q' P(m | Q')`` (Eqs. 2-3 of the paper). The
statistic of a record is ``P_corr = P(Q* | m)``, the posterior of the true
charge ``Q*``. The observables of Fig. 2 are:

- the accuracy ``P(P_corr > 1/2)`` (Fig. 2b), with the Binder ratio of
  ``P_corr``;
- the mean entropy ``E[H(P_corr)]`` of the binary distribution
  ``{P_corr, 1 - P_corr}`` (Fig. 2c), with its Binder ratio.

The Haar average is exact: the dynamics of the averaged density matrix is
diagonal after the first step, so the records drawn from the quantum
circuits are distributed exactly as the SEP predicts (checked in the
tests). The SEP is evolved densely on the ``2**L`` configurations, which is
fine up to ``L ~ 14``; the paper uses matrix product states for ``L = 20``.

Qubit 0 is the leftmost factor of the tensor product, as in
:mod:`QELM.models.circuits`; configuration index ``x`` has bit
``(x >> (L - 1 - i)) & 1`` on site ``i``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from QELM.evaluation.charge_sharpening import binder_ratio, central_moment
from QELM.models.circuits import U1BrickworkCircuit

UNMEASURED = -1


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    """Return ``rng``, or a generator drawn from the global NumPy state."""
    if rng is not None:
        return rng
    return np.random.default_rng(np.random.randint(0, 2**31 - 1))


def site_bits(L: int) -> np.ndarray:
    """Return the bit of every site in every configuration.

    Args:
        L: Number of sites, at least 1.

    Returns:
        Integer array of shape ``(2**L, L)``; entry ``[x, i]`` is the bit of
        site ``i`` in configuration ``x``, site 0 being the most significant.
    """
    x = np.arange(2**L)[:, None]
    shifts = L - 1 - np.arange(L)[None, :]
    return np.asarray((x >> shifts) & 1)


def brickwork_bonds(L: int, boundary: str = "open") -> list[tuple[int, int]]:
    """Return the bonds of one time step, even half-layer first.

    The bonds are those of :class:`~QELM.models.circuits.U1BrickworkCircuit`
    with no ancilla, so the quantum circuit and the SEP see the same
    geometry.

    Args:
        L: Number of sites, at least 2.
        boundary: ``"open"`` or ``"periodic"``.

    Returns:
        The pairs ``(i, j)`` in the order the gates are applied.
    """
    geometry = U1BrickworkCircuit({"N": L, "Nr": 0, "depth": 0, "boundary": boundary})
    return geometry.bonds(0) + geometry.bonds(1)


def charge_state(L: int, Q: int) -> np.ndarray:
    """Return the uniform superposition of the bitstrings of charge ``Q``.

    Args:
        L: Number of sites.
        Q: Charge, ``0 <= Q <= L``.

    Returns:
        Normalized real state vector of shape ``(2**L,)``.

    Raises:
        ValueError: If ``Q`` is out of range.
    """
    if not 0 <= Q <= L:
        raise ValueError(f"Q must be in [0, {L}], got {Q}.")
    in_sector = site_bits(L).sum(axis=1) == Q
    return in_sector / math.sqrt(math.comb(L, Q))


def default_charges(L: int) -> tuple[int, int]:
    """Return the two charges of the paper, ``(L // 2, L // 2 - 1)``."""
    return (L // 2, L // 2 - 1)


def random_u1_gates(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` random U(1) gates at once.

    Same distribution as :func:`QELM.models.circuits.random_u1_gate`:
    uniform phases on ``|00>`` and ``|11>``, a Haar ``2 x 2`` block on
    ``{|01>, |10>}``.

    Args:
        n: Number of gates.
        rng: Random generator.

    Returns:
        Complex array of shape ``(n, 4, 4)``.
    """
    z = (rng.standard_normal((n, 2, 2)) + 1j * rng.standard_normal((n, 2, 2))) / (
        np.sqrt(2)
    )
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=1, axis2=2)
    block = q * (d / np.abs(d))[:, None, :]
    phases = np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=(n, 2)))
    u = np.zeros((n, 4, 4), dtype=complex)
    u[:, 0, 0] = phases[:, 0]
    u[:, 1:3, 1:3] = block
    u[:, 3, 3] = phases[:, 1]
    return u


def apply_gates(psi: np.ndarray, gates: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply one gate per state on qubits ``i`` and ``j``.

    Args:
        psi: Batch of state vectors, shape ``(n, 2**L)``.
        gates: One ``4 x 4`` gate per state, shape ``(n, 4, 4)``, ``i`` being
            the most significant bit of the gate index.
        i: First qubit.
        j: Second qubit, different from ``i``.

    Returns:
        The batch ``G_k psi_k``, shape ``(n, 2**L)``.
    """
    n, dim = psi.shape
    L = dim.bit_length() - 1
    T = np.moveaxis(psi.reshape((n,) + (2,) * L), (i + 1, j + 1), (1, 2))
    shape = T.shape
    T = np.einsum("nab,nbk->nak", gates, T.reshape(n, 4, -1)).reshape(shape)
    return np.moveaxis(T, (1, 2), (i + 1, j + 1)).reshape(n, dim)


@dataclass(frozen=True)
class Records:
    """Measurement records and the charges they were drawn from.

    Attributes:
        outcomes: Integer array of shape ``(n, T, L)``: the outcome (0 or 1)
            of site ``i`` at step ``t``, or ``UNMEASURED`` (-1).
        charge: True charge of each record, shape ``(n,)``.
        charges: The candidate charges, the hypotheses of Eve.
        boundary: Boundary of the brickwork, ``"open"`` or ``"periodic"``.
    """

    outcomes: np.ndarray
    charge: np.ndarray
    charges: tuple[int, ...]
    boundary: str = "open"

    @property
    def L(self) -> int:
        """Number of sites."""
        return int(self.outcomes.shape[2])


def sample_records(
    L: int,
    p: float,
    n_records: int,
    *,
    depth: int | None = None,
    charges: Sequence[int] | None = None,
    boundary: str = "open",
    rng: np.random.Generator | None = None,
) -> Records:
    """Draw measurement records from random U(1) circuits, by the Born rule.

    Record ``k`` starts from :func:`charge_state` at ``charges[k % 2]`` (so
    the charges alternate, half each for an even ``n_records``), with its
    own random circuit. At each of the ``depth`` steps the even then odd
    bonds get a gate, then every site is measured with probability ``p``;
    the outcome is drawn with its Born probability and the state projected.

    Args:
        L: Number of sites, at least 2.
        p: Measurement probability per site and per step, in ``[0, 1]``.
        n_records: Number of records, at least 1.
        depth: Number of time steps ``T``; ``None`` (default) for ``T = L``.
        charges: Candidate charges; ``None`` for :func:`default_charges`.
            Records cycle through them.
        boundary: ``"open"`` (default) or ``"periodic"``.
        rng: Random generator. ``None`` draws one from the global NumPy
            state, seeded by :func:`QELM.seed.set_seed`.

    Returns:
        The :class:`Records`.

    Raises:
        ValueError: If an argument is out of range.
    """
    if L < 2:
        raise ValueError(f"Need L >= 2, got {L}.")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}.")
    T = L if depth is None else int(depth)
    if T < 0:
        raise ValueError(f"depth must be >= 0, got {T}.")
    qs = tuple(default_charges(L) if charges is None else (int(q) for q in charges))
    if len(qs) < 2 or len(set(qs)) != len(qs):
        raise ValueError(f"Need at least two distinct charges, got {qs}.")
    gen = _generator(rng)
    bonds = brickwork_bonds(L, boundary)
    bits = site_bits(L)

    charge = np.array([qs[k % len(qs)] for k in range(n_records)])
    psi = np.stack([charge_state(L, q) for q in charge]).astype(complex)
    outcomes = np.full((n_records, T, L), UNMEASURED, dtype=np.int8)
    for t in range(T):
        for i, j in bonds:
            psi = apply_gates(psi, random_u1_gates(n_records, gen), i, j)
        measured = gen.random((n_records, L)) < p
        draws = gen.random((n_records, L))
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
    return Records(outcomes=outcomes, charge=charge, charges=qs, boundary=boundary)


def sep_log_likelihood(
    outcomes: np.ndarray, Q: int, boundary: str = "open"
) -> np.ndarray:
    """Return ``log P(m | Q)`` of each record under the Haar-averaged dynamics.

    The SEP distribution starts uniform on the configurations of charge
    ``Q``. A gate on ``(i, j)`` maps ``v`` to ``(v + v o swap_ij) / 2``; a
    measurement zeroes the configurations that disagree with its outcome.
    The total weight left after the last step is ``P(m | Q)``; it is
    renormalized after each step and the logarithms summed, so long records
    do not underflow.

    Args:
        outcomes: Records of shape ``(n, T, L)``, as in :class:`Records`.
        Q: Charge of the hypothesis.
        boundary: ``"open"`` or ``"periodic"``, as used to draw the records.

    Returns:
        Array of shape ``(n,)``; ``-inf`` where the record is impossible for
        ``Q`` (an outcome needs more or fewer particles than ``Q``).
    """
    outcomes = np.asarray(outcomes)
    n, T, L = outcomes.shape
    bits = site_bits(L)
    configs = np.arange(2**L)
    swaps = []
    for i, j in brickwork_bonds(L, boundary):
        flip = (1 << (L - 1 - i)) | (1 << (L - 1 - j))
        differ = bits[:, i] != bits[:, j]
        swaps.append(np.where(differ, configs ^ flip, configs))

    v = np.tile(charge_state(L, Q) ** 2, (n, 1))
    log_p = np.zeros(n)
    for t in range(T):
        for perm in swaps:
            v = 0.5 * (v + v[:, perm])
        m = outcomes[:, t, :]
        for site in range(L):
            measured = m[:, site] != UNMEASURED
            if not measured.any():
                continue
            agree = bits[None, :, site] == m[:, site, None]
            v = np.where(measured[:, None] & ~agree, 0.0, v)
        total = v.sum(axis=1)
        with np.errstate(divide="ignore"):
            log_p += np.log(total)
        v = np.divide(v, total[:, None], out=np.zeros_like(v), where=total[:, None] > 0)
    return log_p


def sep_posterior(
    outcomes: np.ndarray, charges: Sequence[int], boundary: str = "open"
) -> np.ndarray:
    """Return Eve's posterior ``P(Q | m)`` for each record, flat prior.

    Args:
        outcomes: Records of shape ``(n, T, L)``.
        charges: The candidate charges.
        boundary: ``"open"`` or ``"periodic"``.

    Returns:
        Array of shape ``(n, len(charges))`` whose rows sum to 1.

    Raises:
        ValueError: If a record is impossible under every charge.
    """
    log_p = np.stack([sep_log_likelihood(outcomes, q, boundary) for q in charges], 1)
    top = log_p.max(axis=1, keepdims=True)
    if not np.all(np.isfinite(top)):
        raise ValueError("A record has zero probability under every charge.")
    w = np.exp(log_p - top)
    return np.asarray(w / w.sum(axis=1, keepdims=True))


def correct_probability(records: Records) -> np.ndarray:
    """Return ``P_corr = P(Q* | m)`` for each record, ``Q*`` its true charge.

    Args:
        records: Records from :func:`sample_records`.

    Returns:
        Array of shape ``(n,)`` with values in ``[0, 1]``.
    """
    post = sep_posterior(records.outcomes, records.charges, records.boundary)
    column = np.array([records.charges.index(int(q)) for q in records.charge])
    return np.asarray(post[np.arange(post.shape[0]), column])


def binary_entropy(P: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Return ``H(P) = -P log P - (1 - P) log(1 - P)``, elementwise.

    Args:
        P: Probabilities in ``[0, 1]``.
        base: Logarithm base, 2 (default) for bits, ``np.e`` for nats.

    Returns:
        Array of the shape of ``P``, 0 at ``P = 0`` and ``P = 1``.
    """
    P = np.clip(np.asarray(P, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -np.where(P > 0, P * np.log(P), 0.0) - np.where(
            P < 1, (1 - P) * np.log1p(-P), 0.0
        )
    return np.asarray(terms / math.log(base))


def learnability_statistics(
    P_corr: np.ndarray, base: float = 2.0, tie_tol: float = 1e-12
) -> dict[str, float]:
    """Return the observables of Fig. 2b and 2c for Born-sampled records.

    Args:
        P_corr: Posterior of the true charge, one per record, shape ``(n,)``.
        base: Logarithm base of the entropy, 2 (default) for bits.
        tie_tol: ``|P_corr - 1/2| <= tie_tol`` counts as a tie.

    Returns:
        - ``accuracy``: ``P(P_corr > 1/2)``, ties counting 1/2, so that a
          blind guess (``p = 0``) scores exactly 1/2;
        - ``accuracy_strict``: ``P(P_corr > 1/2)``, ties counting 0;
        - ``P_corr_mean`` and ``P_corr_binder``: mean and Binder ratio of
          ``P_corr`` (inset of Fig. 2b);
        - ``entropy_mean``, ``entropy_var`` and ``entropy_binder``: mean,
          variance and Binder ratio of ``H(P_corr)`` (Fig. 2c and inset).
    """
    P = np.asarray(P_corr, dtype=float)
    tie = np.abs(P - 0.5) <= tie_tol
    win = (P > 0.5) & ~tie
    H = binary_entropy(P, base)
    return {
        "accuracy": float(np.mean(win + 0.5 * tie)),
        "accuracy_strict": float(np.mean(win)),
        "P_corr_mean": float(np.mean(P)),
        "P_corr_binder": binder_ratio(P),
        "entropy_mean": float(np.mean(H)),
        "entropy_var": central_moment(H, 2),
        "entropy_binder": binder_ratio(H),
    }


def learnability_bootstrap_errors(
    P_corr: np.ndarray,
    n_boot: int = 200,
    base: float = 2.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Return bootstrap standard errors of :func:`learnability_statistics`.

    Args:
        P_corr: Posterior of the true charge, one per record.
        n_boot: Number of resamples, at least 2.
        base: Logarithm base of the entropy.
        rng: Random generator. ``None`` draws one from the global NumPy state.

    Returns:
        One ``<key>_err`` per key of :func:`learnability_statistics`, the
        standard deviation (``ddof=1``) over the resamples, ``nan`` values
        (an undefined Binder ratio) skipped.

    Raises:
        ValueError: If ``n_boot < 2``.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}.")
    P = np.asarray(P_corr, dtype=float)
    gen = _generator(rng)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        resample = P[gen.integers(0, P.size, size=P.size)]
        for key, value in learnability_statistics(resample, base).items():
            samples.setdefault(key, []).append(value)
    out: dict[str, float] = {}
    for key, values in samples.items():
        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)]
        out[f"{key}_err"] = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return out
