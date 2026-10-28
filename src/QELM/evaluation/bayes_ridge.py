"""Bayes-Ridge pipeline: the charge from the record ``M`` alone (issue #233).

Two steps, on the measurement record ``M`` of the QELM on the charge
(``experiments/QELM_Q/exp_QELM_Q.py``): ``M[r, k]`` is the outcome of system
qubit ``k`` at measurement round ``r``, ``+1`` for ``|0>``, ``-1`` for
``|1>``, and ``0`` when the qubit was not measured.

1. **Generative model.** For each candidate charge ``Q``, the log-likelihood
   ``log P(M | Q)`` of the record (:func:`scrambled_log_likelihood`). Only the
   record enters: which qubits were measured, when, and their outcomes; no
   expectation value ``<Z_k>`` of the final state.
2. **Ridge calibration.** The log-likelihoods, standardized on the training
   records, plus a bias column, are the features of a closed-form ridge
   (:class:`QELM.training.RidgeReadout`) fitted on the charge
   (:class:`BayesRidge`). Its output, mapped to ``[0, 1]``, is the
   probability of the first charge.

The generative model is the fully scrambled limit of the dynamics. For the
random U(1) circuits of Barratt et al., PRL 129, 200602 (2022), the
Haar-averaged Born probability of a record is a symmetric exclusion process
(SEP, :func:`QELM.evaluation.learnability.sep_log_likelihood`), and the SEP
relaxes, between two measurement rounds, toward its stationary distribution:
the uniform distribution on the configurations of charge ``Q``. The TFIM
reservoir, with ``h = 10 J``, almost conserves ``Q`` and has no exact SEP
(its gates are not Haar, and Eve does not know its couplings), so the model
takes that limit at every round: just before a round the configuration is
uniform on charge ``Q``, whatever was measured before. A round that measures
``k`` qubits and finds ``n`` of them at ``|1>`` then has the probability

    P(m_r | Q) = C(Ns - k, Q - n) / C(Ns, Q),

that of fixing those ``k`` sites in a uniform ``Q``-subset of ``Ns`` sites,
and ``log P(M | Q) = sum_r log P(m_r | Q)``. The model forgets the memory
the dynamics keeps between two rounds (a qubit measured twice in a row
likely gives the same outcome): it over-counts the evidence of correlated
rounds, and the ridge corrects that by rescaling the log-likelihoods and
moving the decision threshold. With the record alone, the Bayes rule of the
model is :func:`bayes_accuracy`; the ridge should match it or do better.

The TFIM at ``h = 10 J`` conserves ``Q`` only approximately: a round can
find more qubits at ``|1>`` than ``Q``, impossible under the model, which
would give ``log P(M | Q) = -inf``, an infinite feature that no linear
readout can take, and a single such round would outweigh the whole record.
So each round is a mixture: with probability ``1 - epsilon`` the scrambled
model, with probability ``epsilon`` a leak that makes the ``k`` outcomes
uniform, ``2**-k``:

    P(m_r | Q) = (1 - epsilon) C(Ns - k, Q - n) / C(Ns, Q) + epsilon 2**-k.

Every log-likelihood is then finite, and a round that contradicts ``Q``
costs about ``log(epsilon) - k log 2``. ``epsilon = 1e-3`` (the default) is
the order of the charge leakage of the TFIM over one run; at ``N = 8`` the
accuracy of the pipeline moves by less than 0.01 between ``1e-2`` and
``1e-3`` and drops by 0.02 at ``1e-6``, where the contradicting rounds
become outliers of the least-squares fit.

The scrambled model ignores where and when a qubit was measured: only the
counts ``k`` and ``n`` of each round enter, as balls drawn from an urn. For
a local reservoir with a strictly conserved charge (the XXZ chain of issue
#236) a charge moves one bond at a time, so the positions carry information:
a site found at ``|1>`` is likely still at ``|1>``, or next to a ``|1>``, one
round later. :func:`local_sep_log_likelihood` keeps that memory: a hidden
Markov model whose hidden state is the configuration of charge ``Q``, uniform
at the start, which evolves between two rounds by the continuous-time
symmetric exclusion process (each nearest-neighbour bond exchanges its two
bits at rate ``gamma``) for a time ``dt``, and which each round observes
exactly on the measured sites (with the same leak ``epsilon``, now a round
whose outcomes are uniform and that leaves the configuration as it was). It
has one parameter, ``lam = gamma * dt``, the number of exchanges per bond
between two rounds: ``lam = 0`` freezes the configuration, and
``lam -> inf`` relaxes it to the uniform distribution on charge ``Q`` before
every round, which is exactly the scrambled model.
:func:`fit_local_sep_rate` picks ``lam`` on the training records by maximum
likelihood.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.special import erf, gammaln

from QELM.training.regression.ridge import DEFAULT_ALPHA, RidgeReadout

# Probability of a leak per round, see the module docstring.
DEFAULT_EPSILON = 1e-3
# Grid of lam = gamma * dt of fit_local_sep_rate, inf the scrambled limit.
DEFAULT_LOCAL_RATES = (*np.geomspace(0.01, 10.0, 13).tolist(), np.inf)
# Largest sector whose SEP propagator is formed as a dense matrix.
_DENSE_SECTOR = 2048


def _log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Return ``log C(n, k)``, ``-inf`` where ``k < 0`` or ``k > n``."""
    n, k = np.broadcast_arrays(np.asarray(n, dtype=float), np.asarray(k, dtype=float))
    valid = (k >= 0) & (k <= n)
    safe_k = np.where(valid, k, 0.0)
    out = gammaln(n + 1) - gammaln(safe_k + 1) - gammaln(n - safe_k + 1)
    return np.asarray(np.where(valid, out, -np.inf))


def scrambled_log_likelihood(
    M: np.ndarray, Q: int, epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """Return ``log P(M | Q)`` of each record under the fully scrambled model.

    Args:
        M: Records of shape ``(n, rounds, Ns)``: ``+1`` (``|0>``), ``-1``
            (``|1>``) or ``0`` (not measured), as ``exp_QELM_Q`` saves them.
        Q: Charge of the hypothesis, the number of qubits at ``|1>``,
            ``0 <= Q <= Ns``.
        epsilon: Probability of a leak per round, in ``[0, 1)``. With 0 the
            model is the pure scrambled limit, ``-inf`` where a round finds
            more qubits at ``|1>`` than ``Q`` or more at ``|0>`` than
            ``Ns - Q``.

    Returns:
        Array of shape ``(n,)``, finite when ``epsilon > 0``.

    Raises:
        ValueError: If ``M`` is not 3-dimensional or holds a value other than
            ``-1``, ``0``, ``+1``, or ``Q`` or ``epsilon`` is out of range.
    """
    M = np.asarray(M)
    if M.ndim != 3:
        raise ValueError(f"M must have shape (n, rounds, Ns), got {M.shape}.")
    if not np.all(np.isin(M, (-1, 0, 1))):
        raise ValueError("M must hold only -1, 0 and +1.")
    Ns = M.shape[2]
    if not 0 <= Q <= Ns:
        raise ValueError(f"Q must be in [0, {Ns}], got {Q}.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}.")
    k = np.sum(M != 0, axis=2)
    ones = np.sum(M == -1, axis=2)
    log_h = _log_comb(Ns - k, Q - ones) - _log_comb(np.array(Ns), np.array(Q))
    if epsilon == 0.0:
        return np.asarray(log_h.sum(axis=1))
    per_round = np.logaddexp(
        np.log1p(-epsilon) + log_h, np.log(epsilon) - k * np.log(2)
    )
    return np.asarray(per_round.sum(axis=1))


def sector_configurations(Ns: int, Q: int) -> np.ndarray:
    """Return the configurations of charge ``Q`` on ``Ns`` sites as ``z = +-1``.

    Args:
        Ns: Number of sites.
        Q: Number of sites at ``|1>`` (``z = -1``).

    Returns:
        Array of shape ``(C(Ns, Q), Ns)``, ``+1`` for ``|0>`` and ``-1``
        for ``|1>``, in the lexicographic order of the occupied sites.
    """
    if not 0 <= Q <= Ns:
        raise ValueError(f"Q must be in [0, {Ns}], got {Q}.")
    configs = np.ones((0, Ns), dtype=int)
    rows = []
    for occupied in itertools.combinations(range(Ns), Q):
        z = np.ones(Ns, dtype=int)
        z[list(occupied)] = -1
        rows.append(z)
    return np.array(rows, dtype=int) if rows else configs


def sep_generator(configs: np.ndarray, boundary: str = "open") -> sp.csr_matrix:
    """Return the generator of the symmetric exclusion process, rate 1 per bond.

    ``G[a, b] = 1`` when ``a`` and ``b`` differ by the exchange of the two
    bits of one nearest-neighbour bond, and ``G[b, b] = -`` the number of
    such ``a``. ``G`` is symmetric and its columns sum to 0, so the
    uniform distribution is stationary and ``expm(lam G)`` is a doubly
    stochastic transition matrix.

    Args:
        configs: Configurations of one charge, shape ``(dim, Ns)``, as
            :func:`sector_configurations` returns them.
        boundary: ``"open"`` or ``"periodic"`` (adds the bond
            ``(Ns - 1, 0)`` when ``Ns >= 3``).

    Returns:
        Real CSR matrix of shape ``(dim, dim)``.
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary!r}.")
    dim, Ns = configs.shape
    bonds = [(i, i + 1) for i in range(Ns - 1)]
    if boundary == "periodic" and Ns >= 3:
        bonds.append((Ns - 1, 0))
    index = {z.tobytes(): a for a, z in enumerate(configs)}
    rows: list[int] = []
    cols: list[int] = []
    for a, z in enumerate(configs):
        for i, j in bonds:
            if z[i] != z[j]:
                swapped = z.copy()
                swapped[i], swapped[j] = z[j], z[i]
                rows.append(index[swapped.tobytes()])
                cols.append(a)
    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(dim, dim), dtype=np.float64
    )
    degree = np.asarray(A.sum(axis=0), dtype=np.float64).ravel()
    return sp.csr_matrix(A - sp.diags_array(degree))


def local_sep_log_likelihood(
    M: np.ndarray,
    Q: int,
    lam: float,
    epsilon: float = DEFAULT_EPSILON,
    boundary: str = "open",
) -> np.ndarray:
    """Return ``log P(M | Q)`` of each record under the local SEP model.

    The forward algorithm of the hidden Markov model of the module
    docstring: the distribution ``pi`` of the configuration starts uniform
    on charge ``Q``; before each round it is propagated by ``expm(lam G)``
    (:func:`sep_generator`), then the round, which finds the measured sites
    in the configurations ``c`` consistent with its outcomes, has the
    probability ``P = (1 - epsilon) sum_c pi(c) + epsilon 2**-k`` and
    updates ``pi`` to ``((1 - epsilon) pi 1_c + epsilon 2**-k pi) / P``.

    Args:
        M: Records of shape ``(n, rounds, Ns)``, ``+1``, ``-1`` or ``0``.
        Q: Charge of the hypothesis, ``0 <= Q <= Ns``.
        lam: ``gamma * dt >= 0``, exchanges per bond between two rounds;
            ``inf`` gives :func:`scrambled_log_likelihood`.
        epsilon: Probability of a leak per round, in ``[0, 1)``. With 0 a
            round inconsistent with every configuration still reachable
            gives ``-inf``.
        boundary: Bonds of the SEP, ``"open"`` or ``"periodic"``.

    Returns:
        Array of shape ``(n,)``.

    Raises:
        ValueError: If an argument is out of range.
    """
    M = np.asarray(M)
    if M.ndim != 3:
        raise ValueError(f"M must have shape (n, rounds, Ns), got {M.shape}.")
    if not np.all(np.isin(M, (-1, 0, 1))):
        raise ValueError("M must hold only -1, 0 and +1.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}.")
    if not lam >= 0:
        raise ValueError(f"lam must be >= 0, got {lam}.")
    if np.isinf(lam):
        return scrambled_log_likelihood(M, Q, epsilon)
    n, rounds, Ns = M.shape
    configs = sector_configurations(Ns, Q)
    dim = configs.shape[0]
    G = sep_generator(configs, boundary)
    if lam == 0.0:

        def propagate(pi: np.ndarray) -> np.ndarray:
            return pi

    elif dim <= _DENSE_SECTOR:
        T = scipy.linalg.expm(lam * G.toarray())

        def propagate(pi: np.ndarray) -> np.ndarray:
            return np.asarray(T @ pi)

    else:
        A = lam * G

        def propagate(pi: np.ndarray) -> np.ndarray:
            return np.asarray(expm_multiply(A, pi))

    pi = np.full((dim, n), 1.0 / dim)
    log_p = np.zeros(n)
    z = configs.astype(float)
    for r in range(rounds):
        pi = propagate(pi)
        m = M[:, r, :].astype(float)  # (n, Ns)
        k = np.sum(m != 0, axis=1)
        # Number of measured sites where configuration c disagrees with m.
        mismatches = (np.sum(np.abs(m), axis=1)[None, :] - z @ m.T) / 2
        consistent = np.isclose(mismatches, 0.0)
        p_model = np.sum(np.where(consistent, pi, 0.0), axis=0)
        leak = epsilon * 2.0 ** (-k)
        p_round = (1.0 - epsilon) * p_model + leak
        with np.errstate(divide="ignore"):
            log_p += np.log(p_round)
        safe = np.where(p_round > 0, p_round, 1.0)
        pi = ((1.0 - epsilon) * np.where(consistent, pi, 0.0) + leak * pi) / safe
        # Rounding drifts the normalization over many rounds.
        total = pi.sum(axis=0)
        pi = pi / np.where(total > 0, total, 1.0)
    return log_p


def local_log_likelihood_features(
    M: np.ndarray,
    charges: Sequence[int],
    lam: float,
    epsilon: float = DEFAULT_EPSILON,
    boundary: str = "open",
) -> np.ndarray:
    """Return :func:`local_sep_log_likelihood` for each candidate charge.

    Returns:
        Array of shape ``(n, len(charges))``, finite when ``epsilon > 0``.
    """
    return np.stack(
        [local_sep_log_likelihood(M, int(q), lam, epsilon, boundary) for q in charges],
        axis=1,
    )


def fit_local_sep_rate(
    M: np.ndarray,
    charge: np.ndarray,
    rates: Sequence[float] = DEFAULT_LOCAL_RATES,
    epsilon: float = DEFAULT_EPSILON,
    boundary: str = "open",
) -> tuple[float, np.ndarray]:
    """Pick ``lam`` by maximum likelihood on labelled records.

    Args:
        M: Training records, shape ``(n, rounds, Ns)``.
        charge: Their true charges, shape ``(n,)``.
        rates: Candidate values of ``lam``, ``inf`` allowed.
        epsilon: Probability of a leak per round, in ``(0, 1)``.
        boundary: Bonds of the SEP.

    Returns:
        The best ``lam`` and the total log-likelihood
        ``sum_i log P(M_i | Q_i)`` at every rate, shape ``(len(rates),)``.
    """
    charge = np.asarray(charge).astype(int)
    totals = np.zeros(len(rates))
    for a, lam in enumerate(rates):
        for q in np.unique(charge):
            rows = charge == q
            totals[a] += float(
                np.sum(
                    local_sep_log_likelihood(M[rows], int(q), lam, epsilon, boundary)
                )
            )
    return float(rates[int(np.argmax(totals))]), totals


def log_likelihood_features(
    M: np.ndarray, charges: Sequence[int], epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """Return ``log P(M | Q)`` of each record for each candidate charge.

    Args:
        M: Records of shape ``(n, rounds, Ns)``, see
            :func:`scrambled_log_likelihood`.
        charges: The candidate charges.
        epsilon: Probability of a leak per round, in ``(0, 1)``, so that
            every feature is finite.

    Returns:
        Finite array of shape ``(n, len(charges))``, column ``j`` being
        ``log P(M | charges[j])``.

    Raises:
        ValueError: If ``epsilon`` is not in ``(0, 1)``.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
    return np.stack(
        [scrambled_log_likelihood(M, int(q), epsilon) for q in charges], axis=1
    )


def bayes_accuracy(
    log_likelihood: np.ndarray, charge: np.ndarray, charges: Sequence[int]
) -> float:
    """Return the accuracy of the Bayes rule with a flat prior.

    The guess is the charge of largest log-likelihood; a tie between the
    true charge and ``t - 1`` others counts ``1 / t``, as in
    :func:`QELM.evaluation.learnability.learnability_statistics`, so that a
    blind guess (``p = 0``) scores exactly 1/2 with two charges.

    Args:
        log_likelihood: Shape ``(n, len(charges))``.
        charge: True charge of each record, shape ``(n,)``.
        charges: The candidate charges, the columns of ``log_likelihood``.

    Returns:
        The accuracy, in ``[0, 1]``.
    """
    ll = np.asarray(log_likelihood, dtype=float)
    column = np.array([list(charges).index(int(q)) for q in charge])
    true = ll[np.arange(ll.shape[0]), column]
    best = ll.max(axis=1)
    ties = np.isclose(ll, best[:, None], rtol=0.0, atol=1e-9)
    won = np.isclose(true, best, rtol=0.0, atol=1e-9)
    return float(np.mean(np.where(won, 1.0 / ties.sum(axis=1), 0.0)))


def scrambled_discriminability(
    Ns: int, p: float, n_rounds: int, charges: Sequence[int]
) -> float:
    """Return the discriminability ``d'`` of two charges under the scrambled model.

    Under the model, the number ``n`` of qubits found at ``|1>`` among the
    ``k`` measured in a round is hypergeometric: mean ``k Q / Ns``, variance
    ``k rho (1 - rho) (Ns - k) / (Ns - 1)`` with ``rho = Q / Ns``. The two
    charges shift the mean by ``k |Q0 - Q1| / Ns``; in the Gaussian
    approximation, with ``k = p Ns`` and ``rho`` at the middle of the two
    charges, the rounds add up to

        d'^2 = n_rounds (k dQ / Ns)^2 (Ns - 1) / (k rho (1 - rho) (Ns - k)),

    which is ``~ 4 n_rounds p / ((1 - p) Ns)`` at half filling: the scaling
    variable of the accuracy is ``n_rounds p / ((1 - p) Ns)``.

    Args:
        Ns: Number of system qubits, at least 2.
        p: Measurement probability per qubit and round, in ``[0, 1)``.
        n_rounds: Number of measurement rounds, at least 0.
        charges: The two charges.

    Returns:
        ``d' >= 0``, 0 at ``p = 0``.

    Raises:
        ValueError: If an argument is out of range (``p = 1`` gives an
            infinite ``d'``: a fully measured round reads the charge).
    """
    if Ns < 2 or not 0.0 <= p < 1.0 or n_rounds < 0:
        raise ValueError(
            f"Need Ns >= 2, 0 <= p < 1, n_rounds >= 0; got {Ns}, {p}, {n_rounds}."
        )
    q0, q1 = (int(q) for q in charges)
    if q0 == q1:
        raise ValueError(f"Need two distinct charges, got {charges}.")
    if p == 0.0 or n_rounds == 0:
        return 0.0
    k = p * Ns
    rho = (q0 + q1) / (2 * Ns)
    d2 = n_rounds * (k * abs(q0 - q1) / Ns) ** 2 * (Ns - 1)
    d2 /= k * rho * (1 - rho) * (Ns - k)
    return float(np.sqrt(d2))


def scrambled_accuracy(
    Ns: int, p: float, n_rounds: int, charges: Sequence[int]
) -> float:
    """Return the Bayes accuracy predicted by the scrambled model, two charges.

    The Bayes rule between two equal-variance Gaussians at distance ``d'``
    (:func:`scrambled_discriminability`) is right with probability
    ``Phi(d' / 2)``. So at a fixed number of rounds the accuracy falls to
    1/2 as ``N`` grows, as ``N**-1/2``, for every ``p < 1``; it keeps a
    limit in ``(1/2, 1)`` only when ``n_rounds`` grows like ``N``
    (``t_f = N`` in Barratt et al. 2022).

    The Gaussian approximation holds within the sampling noise up to
    ``d' / 2 ~ 0.7``, against the Bayes rule on records drawn from the
    model itself (``tests/test_bayes_ridge.py``); above, it ignores the
    discreteness of the counts and the spread of ``k``, and comes out low:
    0.79 instead of 0.82 at ``Ns = 11``, ``p = 0.6``, 5 rounds.

    Args:
        Ns: Number of system qubits, at least 2.
        p: Measurement probability per qubit and round, in ``[0, 1]``.
        n_rounds: Number of measurement rounds, at least 0.
        charges: The two charges.

    Returns:
        The predicted accuracy, 1/2 at ``p = 0`` and 1 at ``p = 1``.
    """
    if p == 1.0 and n_rounds > 0:
        return 1.0
    d = scrambled_discriminability(Ns, p, n_rounds, charges)
    return float(0.5 * (1 + erf(d / (2 * np.sqrt(2)))))


class BayesRidge:
    """Ridge on the log-likelihoods of two charges, and a class probability.

    The ridge is fitted on the charge itself, like the other readouts of
    ``exp_QELM_Q``, so that its NMSE and R2 compare with theirs; with a bias
    column this is the same fit as on the indicator ``1[Q = charges[0]]`` up
    to an affine map, which :meth:`predict_proba` undoes.

    Attributes:
        charges: The two charges, ``charges[0]`` being the positive class.
        alpha: Ridge strength on the standardized features.
        mean_, scale_: Standardization of the training log-likelihoods,
            ``None`` before :meth:`fit`.
        readout: The :class:`RidgeReadout`; its weights ``W`` act on
            ``[standardized log-likelihoods, 1]``.
    """

    def __init__(self, charges: Sequence[int], alpha: float = DEFAULT_ALPHA) -> None:
        """Store the two charges and the ridge strength.

        Args:
            charges: Exactly two distinct charges.
            alpha: Non-negative ridge strength.

        Raises:
            ValueError: If ``charges`` is not two distinct values.
        """
        qs = tuple(int(q) for q in charges)
        if len(qs) != 2 or qs[0] == qs[1]:
            raise ValueError(f"Need exactly two distinct charges, got {qs}.")
        self.charges = qs
        self.alpha = float(alpha)
        self.readout = RidgeReadout({"alpha": self.alpha})
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def design(self, log_likelihood: np.ndarray) -> np.ndarray:
        """Return ``[standardized log-likelihoods, 1]``, the ridge features."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The model must be fitted first.")
        z = (np.asarray(log_likelihood, dtype=float) - self.mean_) / self.scale_
        return np.hstack([z, np.ones((z.shape[0], 1))])

    def fit(self, log_likelihood: np.ndarray, charge: np.ndarray) -> BayesRidge:
        """Fit the ridge on the training records.

        Args:
            log_likelihood: Features of shape ``(n, 2)``.
            charge: True charge of each record, shape ``(n,)``.

        Returns:
            ``self``.

        Raises:
            ValueError: If the shapes do not match.
        """
        ll = np.asarray(log_likelihood, dtype=float)
        if ll.ndim != 2 or ll.shape != (len(charge), 2):
            raise ValueError(
                f"log_likelihood must have shape ({len(charge)}, 2), got {ll.shape}."
            )
        self.mean_ = ll.mean(axis=0)
        std = ll.std(axis=0)
        # A constant column (p = 0: every log-likelihood is 0) is left as is.
        self.scale_ = np.where(std > 0, std, 1.0)
        self.readout.fit(self.design(ll), np.asarray(charge, dtype=float))
        return self

    def predict(self, log_likelihood: np.ndarray) -> np.ndarray:
        """Return the raw ridge prediction of the charge, shape ``(n,)``."""
        return np.asarray(self.readout.predict(self.design(log_likelihood)))

    def predict_proba(self, log_likelihood: np.ndarray) -> np.ndarray:
        """Return the probability of ``charges[0]``, clipped to ``[0, 1]``."""
        q0, q1 = self.charges
        raw = (self.predict(log_likelihood) - q1) / (q0 - q1)
        return np.asarray(np.clip(raw, 0.0, 1.0))
