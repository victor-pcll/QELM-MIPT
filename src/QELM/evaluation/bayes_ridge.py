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

    P(m_r | Q) = C(N - k, Q - n) / C(N, Q),

that of fixing those ``k`` sites in a uniform ``Q``-subset of ``N`` sites,
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

    P(m_r | Q) = (1 - epsilon) C(N - k, Q - n) / C(N, Q) + epsilon 2**-k.

Every log-likelihood is then finite, and a round that contradicts ``Q``
costs about ``log(epsilon) - k log 2``. ``epsilon = 1e-3`` (the default) is
the order of the charge leakage of the TFIM over one run; at ``N = 8`` the
accuracy of the pipeline moves by less than 0.01 between ``1e-2`` and
``1e-3`` and drops by 0.02 at ``1e-6``, where the contradicting rounds
become outliers of the least-squares fit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import erf, gammaln

from QELM.training.regression.ridge import DEFAULT_ALPHA, RidgeReadout

# Probability of a leak per round, see the module docstring.
DEFAULT_EPSILON = 1e-3


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
        M: Records of shape ``(n, rounds, N)``: ``+1`` (``|0>``), ``-1``
            (``|1>``) or ``0`` (not measured), as ``exp_QELM_Q`` saves them.
        Q: Charge of the hypothesis, the number of qubits at ``|1>``,
            ``0 <= Q <= N``.
        epsilon: Probability of a leak per round, in ``[0, 1)``. With 0 the
            model is the pure scrambled limit, ``-inf`` where a round finds
            more qubits at ``|1>`` than ``Q`` or more at ``|0>`` than
            ``N - Q``.

    Returns:
        Array of shape ``(n,)``, finite when ``epsilon > 0``.

    Raises:
        ValueError: If ``M`` is not 3-dimensional or holds a value other than
            ``-1``, ``0``, ``+1``, or ``Q`` or ``epsilon`` is out of range.
    """
    M = np.asarray(M)
    if M.ndim != 3:
        raise ValueError(f"M must have shape (n, rounds, N), got {M.shape}.")
    if not np.all(np.isin(M, (-1, 0, 1))):
        raise ValueError("M must hold only -1, 0 and +1.")
    N = M.shape[2]
    if not 0 <= Q <= N:
        raise ValueError(f"Q must be in [0, {N}], got {Q}.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}.")
    k = np.sum(M != 0, axis=2)
    ones = np.sum(M == -1, axis=2)
    log_h = _log_comb(N - k, Q - ones) - _log_comb(np.array(N), np.array(Q))
    if epsilon == 0.0:
        return np.asarray(log_h.sum(axis=1))
    per_round = np.logaddexp(
        np.log1p(-epsilon) + log_h, np.log(epsilon) - k * np.log(2)
    )
    return np.asarray(per_round.sum(axis=1))


def log_likelihood_features(
    M: np.ndarray, charges: Sequence[int], epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """Return ``log P(M | Q)`` of each record for each candidate charge.

    Args:
        M: Records of shape ``(n, rounds, N)``, see
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
    N: int, p: float, n_rounds: int, charges: Sequence[int]
) -> float:
    """Return the discriminability ``d'`` of two charges under the scrambled model.

    Under the model, the number ``n`` of qubits found at ``|1>`` among the
    ``k`` measured in a round is hypergeometric: mean ``k Q / N``, variance
    ``k rho (1 - rho) (N - k) / (N - 1)`` with ``rho = Q / N``. The two
    charges shift the mean by ``k |Q0 - Q1| / N``; in the Gaussian
    approximation, with ``k = p N`` and ``rho`` at the middle of the two
    charges, the rounds add up to

        d'^2 = n_rounds (k dQ / N)^2 (N - 1) / (k rho (1 - rho) (N - k)),

    which is ``~ 4 n_rounds p / ((1 - p) N)`` at half filling: the scaling
    variable of the accuracy is ``n_rounds p / ((1 - p) N)``.

    Args:
        N: Number of system qubits, at least 2.
        p: Measurement probability per qubit and round, in ``[0, 1)``.
        n_rounds: Number of measurement rounds, at least 0.
        charges: The two charges.

    Returns:
        ``d' >= 0``, 0 at ``p = 0``.

    Raises:
        ValueError: If an argument is out of range (``p = 1`` gives an
            infinite ``d'``: a fully measured round reads the charge).
    """
    if N < 2 or not 0.0 <= p < 1.0 or n_rounds < 0:
        raise ValueError(
            f"Need N >= 2, 0 <= p < 1, n_rounds >= 0; got {N}, {p}, {n_rounds}."
        )
    q0, q1 = (int(q) for q in charges)
    if q0 == q1:
        raise ValueError(f"Need two distinct charges, got {charges}.")
    if p == 0.0 or n_rounds == 0:
        return 0.0
    k = p * N
    rho = (q0 + q1) / (2 * N)
    d2 = n_rounds * (k * abs(q0 - q1) / N) ** 2 * (N - 1)
    d2 /= k * rho * (1 - rho) * (N - k)
    return float(np.sqrt(d2))


def scrambled_accuracy(
    N: int, p: float, n_rounds: int, charges: Sequence[int]
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
    0.79 instead of 0.82 at ``N = 11``, ``p = 0.6``, 5 rounds.

    Args:
        N: Number of system qubits, at least 2.
        p: Measurement probability per qubit and round, in ``[0, 1]``.
        n_rounds: Number of measurement rounds, at least 0.
        charges: The two charges.

    Returns:
        The predicted accuracy, 1/2 at ``p = 0`` and 1 at ``p = 1``.
    """
    if p == 1.0 and n_rounds > 0:
        return 1.0
    d = scrambled_discriminability(N, p, n_rounds, charges)
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
