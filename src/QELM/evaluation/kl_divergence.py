"""Separability of ``P(M | Q)`` for the two charges: KL, Jensen-Shannon, Bayes.

Issue #270. In the learnability setup of Barratt et al. (2022)
(:mod:`QELM.evaluation.learnability`), an observer sees the measurement
record ``M`` of a monitored U(1) brickwork and guesses which of two charges
``Q+`` and ``Q-`` the chain started in. How well any observer can do depends
only on how far apart the two conditional laws

    P+(M) = P(M | Q = Q+)    and    P-(M) = P(M | Q = Q-)

are. This module estimates their distances by Monte Carlo, from records
drawn with their Born probability, through the log-likelihood ratio
``l(M) = ln P+(M) / P-(M)`` of each record (the sum over the ``3**(rounds L)``
records is out of reach):

- the Kullback-Leibler divergences
  ``D(P+ || P-) = E_{M ~ P+}[l(M)]`` and ``D(P- || P+) = -E_{M ~ P-}[l(M)]``
  and the Jeffreys distance, their mean;
- the Jensen-Shannon divergence ``JS = I(Q; M)``, the mutual information
  between the charge (uniform prior) and the record, in nats, at most
  ``ln 2``. It is related to the mean entropy of Fig. 2c of Barratt et al.
  by ``E[H(P(Q | M))] = ln 2 - JS``;
- the Bhattacharyya coefficient ``BC = sum_M sqrt(P+ P-)``;
- the Bayes error ``P_e = (1/2) sum_M min(P+, P-)``, the smallest error of
  any classifier with a flat prior, and the Bayes accuracy ``1 - P_e``,
  whose empirical counterpart is the accuracy of Fig. 2b (the "blue
  curve").

**The KL divergence is infinite at every** ``p > 0``. A record whose
measurements, within one round, find more particles than ``Q-`` (or more
holes than ``L - Q+``) is impossible under ``Q-`` but not under ``Q+``: its
``l(M) = +inf``. Such a record needs about ``L / 2`` sites measured in the
same round, which has a probability of order ``p**(L / 2)``: tiny at small
``p`` but never zero. So ``D(P+ || P-) = +inf`` exactly, and the
Bretagnolle-Huber (Le Cam) bound ``P_e >= exp(-D) / 4`` says nothing. The
estimator below therefore splits the divergence into

- the **infinite mass** ``pi_inf = P+(l = +inf)``, the probability that a
  record identifies the charge with certainty;
- the **finite part**, the KL divergence between the two laws restricted to
  the records that both allow (``S``) and renormalized,
  ``E_{P+}[l | S] + ln(P-(S) / P+(S))``;

and reports the bounded distances (JS, BC, ``P_e``), which stay finite and
bracket the Bayes error (Lin 1991; Kailath 1967):

    (1/4) (1 - JS / ln 2)**2 <= P_e <= (1/2) (1 - JS / ln 2),
    (1/2) (1 - sqrt(1 - BC**2)) <= P_e <= BC / 2.

Two likelihoods are available, two observers:

- :func:`sep_log_likelihood_ratio`, the observer who does not know the gates
  (Eve of Barratt et al.): ``P(M | Q)`` is averaged over the Haar gates, a
  symmetric exclusion process
  (:func:`~QELM.evaluation.learnability.sep_log_likelihood`).
  Her Bayes accuracy is the blue curve.
- :func:`circuit_log_likelihood`, the observer who knows the circuit ``U``:
  ``P(M | Q, U)`` from the state vector, for one fixed draw of the gates
  (:func:`draw_circuit`, records from :func:`sample_circuit_records`).
  Averaging its divergences over ``N_c`` circuits gives
  ``<D>_circuits`` and its spread. Knowing the gates can only help: its
  divergences are larger than Eve's.

In both, the probability that a given set of sites is measured,
``p**k (1 - p)**(L rounds - k)``, does not depend on ``Q`` and cancels in
``l(M)``; it is left out of the log-likelihoods.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from QELM.evaluation.learnability import (
    UNMEASURED,
    _generator,
    apply_gates,
    charge_state,
    default_charges,
    measurement_rounds,
    random_u1_gates,
    sep_log_likelihood,
    site_bits,
)

LN2 = math.log(2.0)

# Keys of divergence_statistics, each also written with an "_err" suffix (its
# Monte Carlo standard error) except the bounds, computed from the others.
STATISTICS: tuple[str, ...] = (
    "kl_plus_minus",
    "kl_minus_plus",
    "jeffreys",
    "kl_plus_minus_finite",
    "kl_minus_plus_finite",
    "jeffreys_finite",
    "infinite_mass_plus",
    "infinite_mass_minus",
    "infinite_mass",
    "js",
    "js_bits",
    "bhattacharyya",
    "hellinger2",
    "bayes_error",
    "bayes_accuracy",
    "accuracy",
)
BOUNDS: tuple[str, ...] = (
    "bound_le_cam",
    "bound_js_lower",
    "bound_js_upper",
    "bound_hellinger_lower",
    "bound_hellinger_upper",
)


def sep_log_likelihood_ratio(
    outcomes: np.ndarray,
    charges: Sequence[int],
    boundary: str = "open",
    measure_every: str = "layer",
) -> np.ndarray:
    """Return ``l(M) = ln P(M | Q+) - ln P(M | Q-)`` with the Haar-averaged law.

    Args:
        outcomes: Records of shape ``(n, rounds, L)``, as in
            :class:`~QELM.evaluation.learnability.Records`.
        charges: ``(Q+, Q-)``.
        boundary: ``"open"`` or ``"periodic"``, as used to draw the records.
        measure_every: ``"layer"`` or ``"period"``, as used to draw them.

    Returns:
        Array of shape ``(n,)``, ``+inf`` where the record is impossible
        under ``Q-`` only, ``-inf`` where it is impossible under ``Q+`` only.

    Raises:
        ValueError: If ``charges`` is not two charges, or a record is
            impossible under both.
    """
    q_plus, q_minus = _two_charges(charges)
    log_plus = sep_log_likelihood(outcomes, q_plus, boundary, measure_every)
    log_minus = sep_log_likelihood(outcomes, q_minus, boundary, measure_every)
    return log_ratio(log_plus, log_minus)


GATES = ("u1", "u4")
ENCODINGS = ("charge", "local")


def random_gates(n: int, gates: str, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` random two-qubit gates of the gate set ``gates``.

    Args:
        n: Number of gates.
        gates: ``"u1"`` for the U(1) gates of
            :func:`~QELM.evaluation.learnability.random_u1_gates` (they
            conserve the charge), ``"u4"`` for Haar-random gates of U(4)
            (they conserve nothing).
        rng: Random generator.

    Returns:
        Complex array of shape ``(n, 4, 4)``.

    Raises:
        ValueError: If ``gates`` is unknown.
    """
    if gates == "u1":
        return random_u1_gates(n, rng)
    if gates == "u4":
        z = (rng.standard_normal((n, 4, 4)) + 1j * rng.standard_normal((n, 4, 4))) / (
            math.sqrt(2.0)
        )
        q, r = np.linalg.qr(z)
        d = np.diagonal(r, axis1=1, axis2=2)
        return np.asarray(q * (d / np.abs(d))[:, None, :])
    raise ValueError(f"gates must be one of {GATES}, got {gates!r}.")


def draw_circuit(
    L: int,
    depth: int,
    boundary: str = "open",
    measure_every: str = "layer",
    rng: np.random.Generator | None = None,
    gates: str = "u1",
) -> list[np.ndarray]:
    """Draw one random brickwork, the gates of every measurement round.

    Args:
        L: Number of sites, at least 2.
        depth: Number of time steps ``T``, each an even then an odd layer.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``, see
            :func:`~QELM.evaluation.learnability.measurement_rounds`.
        rng: Random generator; ``None`` draws one from the global NumPy
            state.
        gates: Gate set, ``"u1"`` (default) or ``"u4"``, see
            :func:`random_gates`.

    Returns:
        One complex array per round, of shape ``(n_bonds, 4, 4)``: the gates
        of the bonds of :func:`measurement_rounds`, in the same order.
    """
    gen = _generator(rng)
    rounds = measurement_rounds(L, depth, measure_every, boundary)
    return [random_gates(len(bonds), gates, gen) for bonds in rounds]


def hypothesis_states(
    L: int,
    encoding: str,
    charges: Sequence[int] | None = None,
    site: int | None = None,
) -> tuple[tuple[int, ...], list[np.ndarray]]:
    """Return the labels and the initial chain state of each hypothesis.

    - ``"charge"`` (Barratt et al.): label ``Q``, the chain in the uniform
      superposition of the bitstrings of charge ``Q``
      (:func:`~QELM.evaluation.learnability.charge_state`), ``charges``
      ``(Q+, Q-)``, by default ``(L // 2, L // 2 - 1)``.
    - ``"local"`` (Gullans and Huse): a reference qubit ``R`` in a Bell pair
      with site ``site`` (``L // 2`` by default) of the Néel state
      ``|0101...>`` (:func:`QELM.evaluation.purification.reference_bell_state`).
      ``R`` is never touched, so its final Z outcome is its initial one:
      the label is ``r = 0`` or ``1`` (prior 1/2 each), and given ``r`` the
      chain starts in the Néel state with site ``site`` set to ``r``.

    Args:
        L: Number of sites.
        encoding: ``"charge"`` or ``"local"``.
        charges: Charges of the ``"charge"`` encoding.
        site: Site paired with ``R`` in the ``"local"`` encoding.

    Returns:
        ``(labels, states)``: the two labels and the two normalized state
        vectors of shape ``(2**L,)``.

    Raises:
        ValueError: If ``encoding`` is unknown or ``site`` out of range.
    """
    if encoding == "charge":
        qs = _two_charges(default_charges(L) if charges is None else charges)
        return qs, [charge_state(L, q).astype(complex) for q in qs]
    if encoding == "local":
        x0 = L // 2 if site is None else int(site)
        if not 0 <= x0 < L:
            raise ValueError(f"site must be in [0, {L - 1}], got {x0}.")
        neel = int("".join("01"[i % 2] for i in range(L)), 2)
        mask = 1 << (L - 1 - x0)
        states = []
        for r in (0, 1):
            psi = np.zeros(2**L, dtype=complex)
            psi[(neel & ~mask) | (mask if r else 0)] = 1.0
            states.append(psi)
        return (0, 1), states
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}.")


def _apply_round(
    psi: np.ndarray, gates: np.ndarray, bonds: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Apply the same gates to every state of the batch ``psi``."""
    n = psi.shape[0]
    for gate, (i, j) in zip(gates, bonds, strict=True):
        psi = apply_gates(psi, np.broadcast_to(gate, (n, 4, 4)), i, j)
    return psi


def sample_state_records(
    L: int,
    p: float,
    n_records: int,
    states: Sequence[np.ndarray],
    *,
    circuit: Sequence[np.ndarray] | None = None,
    depth: int | None = None,
    gates: str = "u1",
    boundary: str = "open",
    measure_every: str = "layer",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw records by the Born rule, from given initial states.

    Record ``k`` starts in ``states[k % len(states)]``. With ``circuit``,
    every record goes through those gates; without it, each record gets its
    own random circuit of ``depth`` steps of the gate set ``gates`` (records
    of the Haar-averaged law). After the gates of each round, every site is
    measured in Z with probability ``p``.

    Args:
        L: Number of sites.
        p: Measurement probability per site and per round, in ``[0, 1]``.
        n_records: Number of records.
        states: Initial state vectors, shape ``(2**L,)`` each.
        circuit: Fixed gates of every round, from :func:`draw_circuit`;
            ``None`` for a fresh circuit per record.
        depth: Number of time steps without ``circuit``.
        gates: Gate set of the fresh circuits, see :func:`random_gates`.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.
        rng: Random generator; ``None`` draws one from the global NumPy
            state.

    Returns:
        ``(outcomes, label)``: the records, shape ``(n, rounds, L)``, and the
        index in ``states`` of the start of each, shape ``(n,)``.

    Raises:
        ValueError: If an argument is out of range, or neither ``circuit``
            nor ``depth`` is given.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}.")
    if circuit is not None:
        rounds = _rounds_of(L, circuit, boundary, measure_every)
    elif depth is not None:
        rounds = measurement_rounds(L, int(depth), measure_every, boundary)
    else:
        raise ValueError("Give a circuit or a depth.")
    gen = _generator(rng)
    bits = site_bits(L)
    label = np.arange(n_records) % len(states)
    psi = np.stack([np.asarray(states[int(k)], dtype=np.complex128) for k in label])
    outcomes = np.full((n_records, len(rounds), L), UNMEASURED, dtype=np.int8)
    for t, bonds in enumerate(rounds):
        if circuit is not None:
            psi = _apply_round(psi, circuit[t], bonds)
        else:
            for i, j in bonds:
                psi = apply_gates(psi, random_gates(n_records, gates, gen), i, j)
        measured = gen.random((n_records, L)) < p
        draws = gen.random((n_records, L))
        for site in range(L):
            rows = np.flatnonzero(measured[:, site])
            if rows.size == 0:
                continue
            p_one = (np.abs(psi[rows][:, bits[:, site] == 1]) ** 2).sum(axis=1)
            outcome = (draws[rows, site] < p_one).astype(np.int8)
            kept = psi[rows] * (bits[None, :, site] == outcome[:, None])
            kept /= np.linalg.norm(kept, axis=1, keepdims=True)
            psi[rows] = kept
            outcomes[rows, t, site] = outcome
    return outcomes, label


def sample_circuit_records(
    L: int,
    p: float,
    n_records: int,
    circuit: Sequence[np.ndarray],
    *,
    charges: Sequence[int],
    boundary: str = "open",
    measure_every: str = "layer",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw records of one fixed circuit from charge states, by the Born rule.

    :func:`sample_state_records` from the states
    :func:`~QELM.evaluation.learnability.charge_state` of ``charges``:
    every record goes through the same gates ``circuit``; only the measured
    sites and the outcomes are random.

    Args:
        L: Number of sites.
        p: Measurement probability per site and per round, in ``[0, 1]``.
        n_records: Number of records; record ``k`` starts at
            ``charges[k % len(charges)]``.
        circuit: Gates of every round, from :func:`draw_circuit`.
        charges: Starting charges.
        boundary: ``"open"`` or ``"periodic"``, as used to draw ``circuit``.
        measure_every: ``"layer"`` or ``"period"``, as used to draw it.
        rng: Random generator; ``None`` draws one from the global NumPy
            state.

    Returns:
        ``(outcomes, charge)``: the records, shape ``(n, rounds, L)``, and
        the charge of each, shape ``(n,)``.

    Raises:
        ValueError: If ``p`` or ``n_records`` is out of range, or the circuit
            does not have one array per round.
    """
    qs = [int(q) for q in charges]
    outcomes, label = sample_state_records(
        L,
        p,
        n_records,
        [charge_state(L, q) for q in qs],
        circuit=circuit,
        boundary=boundary,
        measure_every=measure_every,
        rng=rng,
    )
    return outcomes, np.asarray(qs)[label]


def state_log_likelihood(
    outcomes: np.ndarray,
    circuit: Sequence[np.ndarray],
    psi0: np.ndarray,
    boundary: str = "open",
    measure_every: str = "layer",
) -> np.ndarray:
    """Return ``ln P(M | psi0, U)`` of each record for the known circuit ``U``.

    The state starts at ``psi0``; each round applies the gates of ``circuit``
    then projects the measured sites on their outcomes. ``P(M | psi0, U)`` is
    the squared norm left at the end (without the factor of the measured
    positions, common to every hypothesis); the state is renormalized after
    each round and the logarithms summed.

    Args:
        outcomes: Records of shape ``(n, rounds, L)``.
        circuit: Gates of every round, from :func:`draw_circuit`.
        psi0: Normalized initial state vector, shape ``(2**L,)``.
        boundary: ``"open"`` or ``"periodic"``, as used to draw ``circuit``.
        measure_every: ``"layer"`` or ``"period"``, as used to draw it.

    Returns:
        Array of shape ``(n,)``, ``-inf`` where the record is impossible.
    """
    outcomes = np.asarray(outcomes)
    n, _, L = outcomes.shape
    rounds = _rounds_of(L, circuit, boundary, measure_every)
    if outcomes.shape[1] != len(rounds):
        raise ValueError(
            f"The records have {outcomes.shape[1]} rounds, the circuit {len(rounds)}."
        )
    bits = site_bits(L)
    psi = np.tile(np.asarray(psi0, dtype=complex), (n, 1))
    log_p = np.zeros(n)
    for t, bonds in enumerate(rounds):
        psi = _apply_round(psi, circuit[t], bonds)
        m = outcomes[:, t, :]
        for site in range(L):
            measured = m[:, site] != UNMEASURED
            if not measured.any():
                continue
            agree = bits[None, :, site] == m[:, site, None]
            psi = np.where(measured[:, None] & ~agree, 0.0, psi)
        norm2 = (np.abs(psi) ** 2).sum(axis=1)
        # Below this, the weight is round-off of an impossible record.
        alive = norm2 > 1e-24
        with np.errstate(divide="ignore"):
            log_p += np.where(alive, np.log(np.where(alive, norm2, 1.0)), -np.inf)
        scale = np.where(alive, 1.0 / np.sqrt(np.where(alive, norm2, 1.0)), 0.0)
        psi = psi * scale[:, None]
    return log_p


def reference_bloch(
    outcomes: np.ndarray,
    circuit: Sequence[np.ndarray],
    states: Sequence[np.ndarray],
    boundary: str = "open",
    measure_every: str = "layer",
) -> np.ndarray:
    """Return the Bloch vector of the reference qubit given each record.

    The ``"local"`` encoding (:func:`hypothesis_states`) is the state
    ``(|0>_R |b_0> + |1>_R |b_1>) / sqrt(2)``. The circuit and the
    projections on the record act on the chain only, as an operator ``K``,
    and leave ``|0>_R K|b_0> + |1>_R K|b_1>``: the state of ``R`` given the
    record is ``rho_rs = <K b_s | K b_r> / norm``. Both branches are evolved
    together and renormalized together after each round.

    ``z = <Z_R>`` is tied to the Bayes posterior of the label,
    ``P(r = 0 | M, U) = (1 + z) / 2``, so the Bayes accuracy of the Z label is
    ``E[(1 + |z|) / 2]``. The norm ``|r|`` gives the entropy ``S_R``
    (:func:`bloch_entropy`), whose decrease measures the purification.

    Args:
        outcomes: Records of shape ``(n, rounds, L)``.
        circuit: Gates of every round, from :func:`draw_circuit`.
        states: The two branches ``(b_0, b_1)``, normalized.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.

    Returns:
        Array of shape ``(n, 3)``: ``(<X_R>, <Y_R>, <Z_R>)`` for each record.

    Raises:
        ValueError: If a record is impossible under both branches.
    """
    outcomes = np.asarray(outcomes)
    n, _, L = outcomes.shape
    rounds = _rounds_of(L, circuit, boundary, measure_every)
    if outcomes.shape[1] != len(rounds):
        raise ValueError(
            f"The records have {outcomes.shape[1]} rounds, the circuit {len(rounds)}."
        )
    if len(states) != 2:
        raise ValueError(f"Need two branches, got {len(states)}.")
    bits = site_bits(L)
    dim = 2**L
    # Rows 2k and 2k + 1: the two branches of record k.
    psi = np.tile(
        np.stack([np.asarray(s, dtype=np.complex128) for s in states]), (n, 1)
    )
    rec = np.repeat(np.arange(n), 2)
    for t, bonds in enumerate(rounds):
        psi = _apply_round(psi, circuit[t], bonds)
        m = outcomes[rec, t, :]
        for site in range(L):
            measured = m[:, site] != UNMEASURED
            if not measured.any():
                continue
            agree = bits[None, :, site] == m[:, site, None]
            psi = np.where(measured[:, None] & ~agree, 0.0, psi)
        pair = psi.reshape(n, 2, dim)
        total = (np.abs(pair) ** 2).sum(axis=(1, 2))
        if np.any(total < 1e-24):
            raise ValueError("A record has zero probability under both branches.")
        psi = (pair / np.sqrt(total)[:, None, None]).reshape(2 * n, dim)
    pair = psi.reshape(n, 2, dim)
    w0 = (np.abs(pair[:, 0]) ** 2).sum(axis=1)
    w1 = (np.abs(pair[:, 1]) ** 2).sum(axis=1)
    rho01 = np.einsum("nd,nd->n", pair[:, 1].conj(), pair[:, 0])
    return np.stack([2 * rho01.real, -2 * rho01.imag, w0 - w1], axis=1)


def bloch_entropy(bloch: np.ndarray) -> np.ndarray:
    """Return the von Neumann entropy (bits) of qubits of Bloch vectors ``bloch``.

    Args:
        bloch: Array of shape ``(..., 3)``.

    Returns:
        ``S = h((1 + |r|) / 2)``, shape ``(...)``; 0 for a pure state, 1 for
        the maximally mixed one.
    """
    r = np.clip(np.linalg.norm(np.asarray(bloch, dtype=float), axis=-1), 0.0, 1.0)
    lam = np.stack([(1 + r) / 2, (1 - r) / 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(lam > 0, -lam * np.log2(lam), 0.0)
    return np.asarray(terms.sum(axis=0))


REFERENCE_STATISTICS: tuple[str, ...] = (
    "S_R",
    "information_R",
    "purity_R",
    "bloch_norm",
    "abs_z",
    "z_accuracy",
)


def reference_statistics(bloch: np.ndarray) -> dict[str, float]:
    """Return the purification of the reference qubit over Born-sampled records.

    Args:
        bloch: Bloch vectors of ``R`` given each record, shape ``(n, 3)``
            (:func:`reference_bloch`), the records drawn with their Born
            probability from the mixture of the two branches.

    Returns:
        Every key of :data:`REFERENCE_STATISTICS` with its Monte Carlo error
        ``<key>_err``: ``S_R`` the mean entropy of ``R`` (bits);
        ``information_R = 1 - S_R``, the information the record holds on
        ``R`` (1 bit when every record purifies it); ``purity_R`` the mean
        ``Tr rho_R^2 = (1 + |r|^2) / 2``; ``bloch_norm`` the mean ``|r|``;
        ``abs_z`` the mean ``|<Z_R>|``; and ``z_accuracy = (1 + abs_z) / 2``,
        the Bayes accuracy of the Z label, which a decoder of ``Z`` cannot
        exceed even when ``R`` is pure (3/4 for a uniform direction).
    """
    b = np.asarray(bloch, dtype=float)
    norm = np.linalg.norm(b, axis=1)
    S = bloch_entropy(b)
    out: dict[str, float] = {}
    for key, values in (
        ("S_R", S),
        ("information_R", 1.0 - S),
        ("purity_R", (1.0 + norm**2) / 2.0),
        ("bloch_norm", norm),
        ("abs_z", np.abs(b[:, 2])),
        ("z_accuracy", (1.0 + np.abs(b[:, 2])) / 2.0),
    ):
        out[key], out[f"{key}_err"] = _mean_err(values)
    return out


def averaged_log_likelihood(
    outcomes: np.ndarray,
    psi0: np.ndarray,
    gates: str = "u1",
    boundary: str = "open",
    measure_every: str = "layer",
) -> np.ndarray:
    """Return ``ln P(M | psi0)`` averaged over the random gates.

    Each gate enters the Born probability once as ``U`` and once as
    ``U^dagger``, so the average over independent gates is exact gate by
    gate. On the diagonal of the density matrix:

    - ``"u1"``: the symmetric exclusion process of
      :func:`~QELM.evaluation.learnability.sep_log_likelihood`, the
      configuration kept or the two bits swapped with probability 1/2;
    - ``"u4"``: a Haar U(4) gate maps the pair to the maximally mixed state,
      ``E[U rho U^dagger] = Tr_ij(rho) x I / 4``: the four configurations of
      the pair become equally likely.

    The start is the diagonal ``|psi0|^2``. This is exact when ``psi0`` is a
    computational basis state (the ``"local"`` encoding), or when the first
    round of gates removes the coherences, as for the charge states
    (:mod:`QELM.evaluation.learnability`).

    Args:
        outcomes: Records of shape ``(n, rounds, L)``.
        psi0: Initial state vector, shape ``(2**L,)``.
        gates: ``"u1"`` or ``"u4"``.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``; with ``"layer"`` the
            number of rounds must be even.

    Returns:
        Array of shape ``(n,)``, ``-inf`` where the record is impossible.

    Raises:
        ValueError: If ``gates`` is unknown or the rounds do not match.
    """
    if gates not in GATES:
        raise ValueError(f"gates must be one of {GATES}, got {gates!r}.")
    outcomes = np.asarray(outcomes)
    n, n_rounds, L = outcomes.shape
    per_step = 2 if measure_every == "layer" else 1
    if n_rounds % per_step:
        raise ValueError(
            f"measure_every='layer' needs an even number of rounds, got {n_rounds}."
        )
    rounds = measurement_rounds(L, n_rounds // per_step, measure_every, boundary)
    bits = site_bits(L)
    configs = np.arange(2**L)

    def partners(i: int, j: int) -> np.ndarray:
        """The 4 configurations reached by setting the bits of ``i``, ``j``."""
        mi, mj = 1 << (L - 1 - i), 1 << (L - 1 - j)
        base = configs & ~(mi | mj)
        return np.stack([base | a | b for a in (0, mi) for b in (0, mj)])

    def swap(i: int, j: int) -> np.ndarray:
        flip = (1 << (L - 1 - i)) | (1 << (L - 1 - j))
        return np.where(bits[:, i] != bits[:, j], configs ^ flip, configs)

    maps = [
        [swap(i, j) if gates == "u1" else partners(i, j) for i, j in bonds]
        for bonds in rounds
    ]
    v = np.tile(np.abs(np.asarray(psi0)) ** 2, (n, 1))
    log_p = np.zeros(n)
    for t in range(n_rounds):
        for gate_map in maps[t]:
            if gates == "u1":
                v = 0.5 * (v + v[:, gate_map])
            else:
                v = 0.25 * v[:, gate_map].sum(axis=1)
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


def circuit_log_likelihood(
    outcomes: np.ndarray,
    circuit: Sequence[np.ndarray],
    Q: int,
    boundary: str = "open",
    measure_every: str = "layer",
) -> np.ndarray:
    """Return ``ln P(M | Q, U)`` of each record for the known circuit ``U``.

    :func:`state_log_likelihood` from :func:`charge_state` ``(L, Q)``.

    Args:
        outcomes: Records of shape ``(n, rounds, L)``.
        circuit: Gates of every round, from :func:`draw_circuit`.
        Q: Charge of the hypothesis.
        boundary: ``"open"`` or ``"periodic"``, as used to draw ``circuit``.
        measure_every: ``"layer"`` or ``"period"``, as used to draw it.

    Returns:
        Array of shape ``(n,)``, ``-inf`` where the record is impossible.
    """
    L = np.shape(outcomes)[2]
    return state_log_likelihood(
        outcomes, circuit, charge_state(L, Q), boundary, measure_every
    )


def circuit_log_likelihood_ratio(
    outcomes: np.ndarray,
    circuit: Sequence[np.ndarray],
    charges: Sequence[int],
    boundary: str = "open",
    measure_every: str = "layer",
) -> np.ndarray:
    """Return ``l(M) = ln P(M | Q+, U) - ln P(M | Q-, U)`` for a known circuit.

    Args:
        outcomes: Records of shape ``(n, rounds, L)``.
        circuit: Gates of every round, from :func:`draw_circuit`.
        charges: ``(Q+, Q-)``.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.

    Returns:
        Array of shape ``(n,)``, infinite as in
        :func:`sep_log_likelihood_ratio`.
    """
    q_plus, q_minus = _two_charges(charges)
    log_plus = circuit_log_likelihood(
        outcomes, circuit, q_plus, boundary, measure_every
    )
    log_minus = circuit_log_likelihood(
        outcomes, circuit, q_minus, boundary, measure_every
    )
    return log_ratio(log_plus, log_minus)


def _two_charges(charges: Sequence[int]) -> tuple[int, int]:
    qs = tuple(int(q) for q in charges)
    if len(qs) != 2 or qs[0] == qs[1]:
        raise ValueError(f"Need two distinct charges (Q+, Q-), got {qs}.")
    return qs[0], qs[1]


def log_ratio(log_plus: np.ndarray, log_minus: np.ndarray) -> np.ndarray:
    """Return ``log_plus - log_minus``, refusing records impossible under both."""
    both = np.isneginf(log_plus) & np.isneginf(log_minus)
    if np.any(both):
        raise ValueError("A record has zero probability under both charges.")
    with np.errstate(invalid="ignore"):
        return np.asarray(log_plus - log_minus)


def _rounds_of(
    L: int, circuit: Sequence[np.ndarray], boundary: str, measure_every: str
) -> list[list[tuple[int, int]]]:
    """Return the bonds of each round of ``circuit``, checking its shape."""
    per_step = 2 if measure_every == "layer" else 1
    if len(circuit) % per_step:
        raise ValueError(
            f"measure_every='layer' needs an even number of rounds, got {len(circuit)}."
        )
    rounds = measurement_rounds(L, len(circuit) // per_step, measure_every, boundary)
    for t, (gates, bonds) in enumerate(zip(circuit, rounds, strict=True)):
        if np.shape(gates) != (len(bonds), 4, 4):
            raise ValueError(
                f"Round {t} needs gates of shape {(len(bonds), 4, 4)}, got "
                f"{np.shape(gates)}."
            )
    return rounds


def _mean_err(values: np.ndarray) -> tuple[float, float]:
    """Mean of ``values`` and its standard error, ``nan`` if empty."""
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if not np.isfinite(mean):
        return mean, float("nan")
    if values.size < 2:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / math.sqrt(values.size))


def _half_sum(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """``(x + y) / 2`` of two independent estimates with their errors."""
    return 0.5 * (a[0] + b[0]), 0.5 * math.hypot(a[1], b[1])


def _restricted_kl(
    llr_common: np.ndarray,
    mass_own: tuple[float, float],
    mass_other: tuple[float, float],
) -> tuple[float, float]:
    """KL between the two laws restricted to their common support ``S``.

    With ``P|S = P 1_S / P(S)``, ``D(P+|S || P-|S) = E_+[l | S]
    + ln(P-(S) / P+(S))``, where ``P+(S) = 1 - pi_+`` and ``P-(S) = 1 - pi_-``
    (``pi`` the infinite masses). Unlike ``E_+[l | S]`` alone, it is a true
    divergence, never negative. The error adds the errors of the three terms
    in quadrature (delta method for the logarithms).

    Args:
        llr_common: ``l(M)`` of the records of the first law that lie in
            ``S`` (finite ratio), with the sign that makes its mean a KL.
        mass_own: Infinite mass of the first law, with its error.
        mass_other: Infinite mass of the second law, with its error.
    """
    mean, err = _mean_err(llr_common)
    own, other = 1.0 - mass_own[0], 1.0 - mass_other[0]
    if own <= 0.0 or other <= 0.0 or not math.isfinite(mean):
        return float("nan"), float("nan")
    value = mean + math.log(other / own)
    error = math.sqrt(err**2 + (mass_own[1] / own) ** 2 + (mass_other[1] / other) ** 2)
    return value, error


def divergence_statistics(
    llr_plus: np.ndarray, llr_minus: np.ndarray
) -> dict[str, float]:
    """Return the divergences between ``P+`` and ``P-`` from sampled records.

    Args:
        llr_plus: ``l(M) = ln P+(M) / P-(M)`` of records drawn from ``P+``.
        llr_minus: The same ``l(M)`` (not its opposite) of records drawn from
            ``P-``.

    Returns:
        Every key of :data:`STATISTICS` with its Monte Carlo standard error
        ``<key>_err``, and the :data:`BOUNDS` on the Bayes error. In nats:

        - ``kl_plus_minus`` ``= D(P+ || P-)``, ``+inf`` as soon as one record
          of ``P+`` is impossible under ``P-``; ``kl_minus_plus`` the other
          way; ``jeffreys`` their mean;
        - ``kl_*_finite`` and ``jeffreys_finite``: the KL between the two
          laws restricted to the records possible under both and
          renormalized, the finite part (:func:`_restricted_kl`);
        - ``infinite_mass_plus`` ``= P+(l = +inf)``, ``infinite_mass_minus``
          ``= P-(l = -inf)`` and ``infinite_mass`` their mean, the
          probability that a record gives the charge for certain;
        - ``js`` the Jensen-Shannon divergence, ``I(Q; M)`` with a flat
          prior, and ``js_bits = js / ln 2`` in ``[0, 1]``;
        - ``bhattacharyya`` ``BC`` and ``hellinger2 = 1 - BC``;
        - ``bayes_error`` ``P_e`` and ``bayes_accuracy = 1 - P_e``, the
          expected accuracy of the Bayes classifier; ``accuracy`` its
          empirical accuracy on these records (``l > 0`` guesses ``Q+``, a
          tie counts 1/2), the estimator of Fig. 2b;
        - ``bound_le_cam`` ``= exp(-min(D+-, D-+)) / 4``, the
          Bretagnolle-Huber lower bound on ``P_e`` (0 when both KL are
          infinite), ``bound_js_*`` and ``bound_hellinger_*`` the lower and
          upper bounds on ``P_e`` from JS and from BC.

        The estimates that mix both samples average one estimate from each,
        so their errors add in quadrature.

    Raises:
        ValueError: If a sample is empty or holds a ``nan``.
    """
    lp = np.asarray(llr_plus, dtype=float).ravel()
    lm = np.asarray(llr_minus, dtype=float).ravel()
    if lp.size == 0 or lm.size == 0:
        raise ValueError("Need records from both charges.")
    if np.isnan(lp).any() or np.isnan(lm).any():
        raise ValueError("A log-likelihood ratio is nan.")
    out: dict[str, float] = {}

    def put(key: str, value: tuple[float, float]) -> None:
        out[key], out[f"{key}_err"] = value

    inf_p = np.isposinf(lp)
    inf_m = np.isneginf(lm)
    # D(P+ || P-) = E_+[l]; a record of P+ with l = -inf has P+ = 0, which
    # Born sampling never draws, so only l = +inf can make it infinite.
    kl_pm = (math.inf, 0.0) if inf_p.any() else _mean_err(lp)
    kl_mp = (math.inf, 0.0) if inf_m.any() else _mean_err(-lm)
    put("kl_plus_minus", kl_pm)
    put("kl_minus_plus", kl_mp)
    put("jeffreys", _half_sum(kl_pm, kl_mp))
    mass_p = _mean_err(inf_p.astype(float))
    mass_m = _mean_err(inf_m.astype(float))
    fin_pm = _restricted_kl(lp[np.isfinite(lp)], mass_p, mass_m)
    fin_mp = _restricted_kl(-lm[np.isfinite(lm)], mass_m, mass_p)
    put("kl_plus_minus_finite", fin_pm)
    put("kl_minus_plus_finite", fin_mp)
    put("jeffreys_finite", _half_sum(fin_pm, fin_mp))
    put("infinite_mass_plus", mass_p)
    put("infinite_mass_minus", mass_m)
    put("infinite_mass", _half_sum(mass_p, mass_m))

    # ln(2 P+ / (P+ + P-)) = ln 2 - ln(1 + e^{-l}), finite even at l = +inf.
    js = _half_sum(
        _mean_err(LN2 - np.logaddexp(0.0, -lp)),
        _mean_err(LN2 - np.logaddexp(0.0, lm)),
    )
    put("js", js)
    put("js_bits", (js[0] / LN2, js[1] / LN2))
    # BC = E_+[sqrt(P-/P+)] = E_-[sqrt(P+/P-)].
    bc = _half_sum(_mean_err(np.exp(-0.5 * lp)), _mean_err(np.exp(0.5 * lm)))
    put("bhattacharyya", bc)
    put("hellinger2", (1.0 - bc[0], bc[1]))
    # P_e = (1/2) E_+[min(1, P-/P+)] = (1/2) E_-[min(1, P+/P-)].
    pe = _half_sum(
        _mean_err(0.5 * np.exp(-np.maximum(lp, 0.0))),
        _mean_err(0.5 * np.exp(np.minimum(lm, 0.0))),
    )
    put("bayes_error", pe)
    put("bayes_accuracy", (1.0 - pe[0], pe[1]))
    score_p = (lp > 0) + 0.5 * (lp == 0)
    score_m = (lm < 0) + 0.5 * (lm == 0)
    put("accuracy", _mean_err(np.concatenate([score_p, score_m]).astype(float)))

    kl_min = min(kl_pm[0], kl_mp[0])
    out["bound_le_cam"] = 0.25 * math.exp(-kl_min) if math.isfinite(kl_min) else 0.0
    j = min(max(out["js_bits"], 0.0), 1.0)
    out["bound_js_lower"] = 0.25 * (1.0 - j) ** 2
    out["bound_js_upper"] = 0.5 * (1.0 - j)
    b = min(max(bc[0], 0.0), 1.0)
    out["bound_hellinger_lower"] = 0.5 * (1.0 - math.sqrt(1.0 - b * b))
    out["bound_hellinger_upper"] = 0.5 * b
    return out


def split_by_charge(
    llr: np.ndarray, charge: np.ndarray, charges: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Split the ratios of mixed records into ``(llr_plus, llr_minus)``.

    Args:
        llr: ``l(M)`` of each record.
        charge: True charge of each record.
        charges: ``(Q+, Q-)``.

    Returns:
        The ratios of the records drawn at ``Q+``, then at ``Q-``.
    """
    q_plus, q_minus = _two_charges(charges)
    llr = np.asarray(llr)
    charge = np.asarray(charge)
    return llr[charge == q_plus], llr[charge == q_minus]


def circuit_ensemble_statistics(
    per_circuit: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Average the statistics of ``N_c`` circuits.

    Args:
        per_circuit: One :func:`divergence_statistics` per circuit.

    Returns:
        For every key of :data:`STATISTICS` and :data:`BOUNDS`: its mean over
        the circuits, ``<key>_std`` the spread between circuits (``ddof=1``)
        and ``<key>_err = std / sqrt(N_c)``. A divergence infinite for one
        circuit is infinite on average (its std is then ``nan``);
        ``<key>_n_infinite`` counts the circuits where it is infinite.

    Raises:
        ValueError: If ``per_circuit`` is empty.
    """
    if not per_circuit:
        raise ValueError("Need at least one circuit.")
    out: dict[str, float] = {}
    for key in (*STATISTICS, *BOUNDS):
        values = np.array([s[key] for s in per_circuit], dtype=float)
        mean = float(np.mean(values))
        if values.size > 1 and np.all(np.isfinite(values)):
            std = float(np.std(values, ddof=1))
        else:
            std = 0.0 if values.size == 1 else float("nan")
        out[key] = mean
        out[f"{key}_std"] = std
        out[f"{key}_err"] = std / math.sqrt(values.size)
        if key.startswith(("kl_", "jeffreys")) and not key.endswith("_finite"):
            out[f"{key}_n_infinite"] = float(np.count_nonzero(np.isinf(values)))
    return out
