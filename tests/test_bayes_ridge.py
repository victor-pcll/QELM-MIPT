"""The Bayes-Ridge pipeline on the record M, QELM.evaluation.bayes_ridge (#233)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from QELM.evaluation.bayes_ridge import (
    BayesRidge,
    bayes_accuracy,
    frozen_log_likelihood,
    log_likelihood_features,
    scrambled_accuracy,
    scrambled_discriminability,
    scrambled_log_likelihood,
)
from QELM.evaluation.learnability import UNMEASURED, sep_log_likelihood


def _record(*rounds: list[int]) -> np.ndarray:
    """One record, ``(1, rounds, Ns)``."""
    return np.array([list(rounds)])


def _brute_force(M: np.ndarray, Q: int) -> float:
    """log P(M | Q): each round sees a fresh uniform configuration of charge Q."""
    Ns = M.shape[-1]
    configs = [c for c in itertools.product((0, 1), repeat=Ns) if sum(c) == Q]
    total = 0.0
    for m in M:
        agree = sum(
            all(o == 0 or (o == -1) == (b == 1) for o, b in zip(m, c, strict=True))
            for c in configs
        )
        total += math.log(agree / len(configs)) if agree else -math.inf
    return total


def test_known_values_on_four_qubits() -> None:
    # Ns = 4, Q = 2: one qubit at |1> has probability C(3, 1) / C(4, 2) = 1/2.
    one = scrambled_log_likelihood(_record([-1, 0, 0, 0]), 2, epsilon=0.0)
    np.testing.assert_allclose(one, [math.log(1 / 2)])
    # Two qubits at |1>: C(2, 0) / C(4, 2) = 1/6.
    two = scrambled_log_likelihood(_record([-1, -1, 0, 0]), 2, epsilon=0.0)
    np.testing.assert_allclose(two, [math.log(1 / 6)])
    # An unmeasured round carries no information.
    none = scrambled_log_likelihood(_record([0, 0, 0, 0]), 2, epsilon=0.0)
    np.testing.assert_allclose(none, [0.0])


def test_impossible_round_is_minus_infinity_without_leak() -> None:
    M = _record([-1, -1, -1, 0])  # three qubits at |1> under Q = 2
    assert scrambled_log_likelihood(M, 2, epsilon=0.0)[0] == -math.inf
    # With a leak it costs log(epsilon) - k log 2, k = 3 measured qubits.
    eps = 1e-3
    np.testing.assert_allclose(
        scrambled_log_likelihood(M, 2, epsilon=eps), [math.log(eps / 8)]
    )


@pytest.mark.parametrize("Q", [1, 2, 3])
def test_matches_the_brute_force_enumeration(Q: int) -> None:
    rng = np.random.default_rng(0)
    M = rng.choice([-1, 0, 1], size=(6, 3, 5))
    ll = scrambled_log_likelihood(M, Q, epsilon=0.0)
    expected = [_brute_force(m, Q) for m in M]
    np.testing.assert_allclose(ll, expected)


@pytest.mark.parametrize("epsilon", [0.0, 1e-3, 0.3])
def test_each_round_is_normalized(epsilon: float) -> None:
    """Summed over the outcomes of a fixed set of measured qubits, P = 1."""
    Ns, Q, measured = 5, 2, (0, 2, 3)
    total = 0.0
    for outcomes in itertools.product((-1, 1), repeat=len(measured)):
        row = [0] * Ns
        for site, o in zip(measured, outcomes, strict=True):
            row[site] = o
        total += math.exp(scrambled_log_likelihood(_record(row), Q, epsilon)[0])
    assert total == pytest.approx(1.0)


def test_first_round_matches_the_sep() -> None:
    """The SEP starts at its stationary state, the uniform one on charge Q:
    its first measurement round is exactly the scrambled model."""
    rng = np.random.default_rng(1)
    L, n = 6, 20
    # Records of the learnability module: 0/1 outcomes, -1 unmeasured, two
    # rounds (one step measured after each layer), only the first measured.
    outcomes = np.full((n, 2, L), UNMEASURED)
    outcomes[:, 0, :] = rng.choice([UNMEASURED, 0, 1], size=(n, L))
    M = np.where(outcomes == UNMEASURED, 0, np.where(outcomes == 1, -1, 1))
    for Q in (2, 3):
        np.testing.assert_allclose(
            scrambled_log_likelihood(M, Q, epsilon=0.0),
            sep_log_likelihood(outcomes, Q),
        )


def test_features_are_finite_and_ordered_by_charge() -> None:
    M = np.array([[[-1, -1, -1, 0]], [[1, 0, 0, 0]]])
    ll = log_likelihood_features(M, (2, 1))
    assert ll.shape == (2, 2) and np.all(np.isfinite(ll))
    np.testing.assert_allclose(ll[:, 0], scrambled_log_likelihood(M, 2))
    np.testing.assert_allclose(ll[:, 1], scrambled_log_likelihood(M, 1))


@pytest.mark.parametrize(
    ("M", "Q", "epsilon"),
    [
        (np.zeros((2, 3)), 1, 1e-3),  # not 3-dimensional
        (np.full((1, 1, 3), 2), 1, 1e-3),  # not -1, 0, +1
        (np.zeros((1, 1, 3)), 4, 1e-3),  # Q > Ns
        (np.zeros((1, 1, 3)), 1, 1.0),  # epsilon out of range
    ],
)
def test_rejects_bad_inputs(M: np.ndarray, Q: int, epsilon: float) -> None:
    with pytest.raises(ValueError):
        scrambled_log_likelihood(M, Q, epsilon)


def test_features_need_a_positive_leak() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        log_likelihood_features(np.zeros((1, 1, 3)), (1, 0), epsilon=0.0)


def test_bayes_accuracy_counts_ties_one_half() -> None:
    ll = np.array([[0.0, -1.0], [-2.0, 0.0], [0.0, 0.0]])
    charge = np.array([3, 3, 2])
    # Right, wrong, tie: (1 + 0 + 1/2) / 3.
    assert bayes_accuracy(ll, charge, (3, 2)) == pytest.approx(0.5)
    # p = 0: every log-likelihood is 0, a blind guess.
    assert bayes_accuracy(np.zeros((4, 2)), np.array([3, 2, 3, 2]), (3, 2)) == 0.5


def test_bayes_ridge_separates_informative_log_likelihoods() -> None:
    rng = np.random.default_rng(2)
    charge = np.array([3, 2] * 200)
    shift = np.where(charge == 3, 1.0, -1.0)
    ll = np.column_stack([shift, -shift]) * 3 + rng.normal(0, 0.5, (400, 2))
    model = BayesRidge((3, 2)).fit(ll[:300], charge[:300])
    proba = model.predict_proba(ll[300:])
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_array_equal(proba > 0.5, charge[300:] == 3)
    assert model.readout.W is not None and model.readout.W.shape == (3,)


def test_bayes_ridge_at_p_zero_is_a_constant() -> None:
    """Constant log-likelihoods: no division by a zero spread, a constant
    prediction at the mean charge of the training set."""
    charge = np.array([3, 2] * 10)
    model = BayesRidge((3, 2)).fit(np.zeros((20, 2)), charge)
    pred = model.predict(np.zeros((4, 2)))
    np.testing.assert_allclose(pred, 2.5, atol=1e-3)


def test_bayes_ridge_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="two distinct"):
        BayesRidge((2, 2))
    with pytest.raises(ValueError, match="shape"):
        BayesRidge((3, 2)).fit(np.zeros((4, 3)), np.array([3, 2, 3, 2]))
    with pytest.raises(RuntimeError):
        BayesRidge((3, 2)).predict(np.zeros((1, 2)))


def _scrambled_records(
    Ns: int, Q: int, p: float, n_rounds: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw records from the scrambled model: a fresh uniform charge-Q
    configuration before every round, each qubit measured with probability p."""
    M = np.zeros((n, n_rounds, Ns), dtype=int)
    for i in range(n):
        for r in range(n_rounds):
            bits = np.zeros(Ns, dtype=int)
            bits[rng.choice(Ns, Q, replace=False)] = 1
            measured = rng.random(Ns) < p
            M[i, r] = np.where(measured, 1 - 2 * bits, 0)
    return M


@pytest.mark.parametrize(("Ns", "p"), [(7, 0.3), (11, 0.3), (11, 0.45)])
def test_scrambled_accuracy_matches_its_own_bayes_rule(Ns: int, p: float) -> None:
    """Records drawn from the model itself: the Bayes rule scores the
    Gaussian prediction Phi(d' / 2) within the sampling noise."""
    rng = np.random.default_rng(3)
    charges = (Ns // 2, Ns // 2 - 1)
    n = 2000
    M = np.concatenate([_scrambled_records(Ns, q, p, 5, n // 2, rng) for q in charges])
    charge = np.repeat(charges, n // 2)
    measured = bayes_accuracy(log_likelihood_features(M, charges), charge, charges)
    assert measured == pytest.approx(scrambled_accuracy(Ns, p, 5, charges), abs=0.03)


def test_scrambled_accuracy_limits_and_scaling() -> None:
    charges = (5, 4)
    assert scrambled_accuracy(11, 0.0, 5, charges) == 0.5
    assert scrambled_accuracy(11, 1.0, 5, charges) == 1.0
    assert scrambled_accuracy(11, 0.3, 0, charges) == 0.5
    # Fixed rounds: falls toward 1/2 as Ns grows.
    acc = [scrambled_accuracy(ns, 0.3, 5, (ns // 2, ns // 2 - 1)) for ns in (5, 11, 41)]
    assert acc[0] > acc[1] > acc[2] > 0.5
    # d'^2 ~ 4 R p / ((1 - p) Ns) at half filling and large Ns.
    d = scrambled_discriminability(1001, 0.3, 5, (501, 500))
    assert d**2 == pytest.approx(4 * 5 * 0.3 / (0.7 * 1001), rel=0.01)
    # Rounds growing like Ns: a limit in (1/2, 1).
    grow = [
        scrambled_accuracy(ns, 0.1, 2 * ns, (ns // 2, ns // 2 - 1)) for ns in (21, 201)
    ]
    assert grow[0] == pytest.approx(grow[1], abs=0.01)
    with pytest.raises(ValueError):
        scrambled_discriminability(11, 1.0, 5, charges)
    with pytest.raises(ValueError):
        scrambled_discriminability(11, 0.3, 5, (4, 4))


def _frozen_brute_force(M: np.ndarray, Q: int) -> float:
    """log P(M | Q): one uniform configuration of charge Q for every round."""
    Ns = M.shape[-1]
    configs = [c for c in itertools.product((0, 1), repeat=Ns) if sum(c) == Q]
    agree = sum(
        all(
            o == 0 or (o == -1) == (b == 1)
            for m in M
            for o, b in zip(m, c, strict=True)
        )
        for c in configs
    )
    return math.log(agree / len(configs)) if agree else -math.inf


@pytest.mark.parametrize("Q", [1, 2, 3])
def test_frozen_matches_the_brute_force_enumeration(Q: int) -> None:
    """One configuration for every round: only distinct qubits count (#253)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        config = rng.permutation([1] * Q + [0] * (4 - Q))
        mask = rng.random((3, 4)) < 0.5
        M = np.where(mask, np.where(config == 1, -1, 1), 0)
        got = frozen_log_likelihood(M[None], Q, epsilon=0.0)[0]
        assert got == pytest.approx(_frozen_brute_force(M, Q))
        for other in range(5):
            want = _frozen_brute_force(M, other)
            got = frozen_log_likelihood(M[None], other, epsilon=0.0)[0]
            assert got == pytest.approx(want) or got == want == -math.inf


def test_frozen_ignores_a_repeated_measurement() -> None:
    """Measuring the same qubit twice adds nothing; the scrambled model counts it."""
    once = _record([-1, 0, 0, 0], [0, 0, 0, 0])
    twice = _record([-1, 0, 0, 0], [-1, 0, 0, 0])
    np.testing.assert_allclose(
        frozen_log_likelihood(once, 2, epsilon=0.0),
        frozen_log_likelihood(twice, 2, epsilon=0.0),
    )
    assert scrambled_log_likelihood(twice, 2, epsilon=0.0)[0] == pytest.approx(
        2 * scrambled_log_likelihood(once, 2, epsilon=0.0)[0]
    )
    # Two outcomes that disagree: the last one is kept.
    flip = _record([-1, 0, 0, 0], [1, 0, 0, 0])
    np.testing.assert_allclose(
        frozen_log_likelihood(flip, 2, epsilon=0.0),
        frozen_log_likelihood(_record([1, 0, 0, 0]), 2, epsilon=0.0),
    )
