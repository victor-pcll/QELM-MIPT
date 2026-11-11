"""Tests of the anatomy of one record: Kraus operator, effect, Pauli spectrum."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from QELM.evaluation.effect_anatomy import (
    anatomy,
    kraus_operator,
    likelihood_ratio,
    pauli_labels,
    pauli_spectrum,
    pauli_string,
    product_states,
    sample_outcomes,
)
from QELM.evaluation.effect_rank import brick_pairs
from QELM.models.circuits.clifford import random_clifford_gate
from QELM.models.circuits.haar import haar_gate


def gates_of(n: int, depth: int, gate, rng: np.random.Generator) -> list:
    return [[gate(rng) for _ in brick_pairs(n)] for _ in range(depth)]


def test_pauli_spectrum_reconstructs_operator() -> None:
    rng = np.random.default_rng(0)
    n = 3
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    a = a + a.conj().T
    c = pauli_spectrum(a)
    rebuilt = sum(
        ci * pauli_string(lab) for ci, lab in zip(c, pauli_labels(n), strict=True)
    )
    np.testing.assert_allclose(rebuilt, a, atol=1e-12)
    assert pauli_labels(2)[:5] == ["II", "IX", "IY", "IZ", "XI"]


def test_kraus_operators_are_complete() -> None:
    """``sum_m K_m^dagger K_m = I`` over every record of a fixed mask."""
    rng = np.random.default_rng(1)
    n, depth = 3, 2
    gates = gates_of(n, depth, haar_gate, rng)
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    sites = np.argwhere(mask)
    total = np.zeros((8, 8), dtype=complex)
    for bits in itertools.product((0, 1), repeat=len(sites)):
        o = np.zeros((depth, n), dtype=int)
        for (t, s), b in zip(sites, bits, strict=True):
            o[t, s] = b
        k = kraus_operator(gates, mask, o)
        total += k.conj().T @ k
    np.testing.assert_allclose(total, np.eye(8), atol=1e-12)


def test_no_measurement_gives_identity_effect() -> None:
    rng = np.random.default_rng(2)
    gates = gates_of(3, 2, haar_gate, rng)
    mask = np.zeros((2, 3), dtype=bool)
    a = anatomy(kraus_operator(gates, mask, np.zeros((2, 3), dtype=int)))
    np.testing.assert_allclose(a.effect, np.eye(8), atol=1e-12)
    assert a.rank == 8 and a.n_strings == 1 and a.weight == pytest.approx(0.0)
    states = product_states(3, 50, rng)
    np.testing.assert_allclose(likelihood_ratio(a.effect, states), 1.0, atol=1e-12)


@pytest.mark.parametrize("p", [0.2, 0.5, 1.0])
def test_weight_identity_and_stabilizer_structure(p: float) -> None:
    """``1 + W = 2^N / r_2``; Clifford: flat spectrum, ``2^k`` strings of +-c_0."""
    rng = np.random.default_rng(3)
    n, depth = 4, 3
    for gate in (random_clifford_gate, haar_gate):
        for _ in range(4):
            gates = gates_of(n, depth, gate, rng)
            mask = rng.random((depth, n)) < p
            a = anatomy(kraus_operator(gates, mask, sample_outcomes(gates, mask, rng)))
            assert 1 + a.weight == pytest.approx(2**n / a.participation, rel=1e-9)
            assert 0 <= a.weight <= 2**n - 1 + 1e-9
            if gate is random_clifford_gate:
                assert a.participation == pytest.approx(a.rank)
                assert a.n_strings == pytest.approx(1 + a.weight)
                nz = a.coefficients[np.abs(a.coefficients) > 1e-9]
                np.testing.assert_allclose(np.abs(nz), 1.0, atol=1e-9)
                assert a.rank * a.n_strings == 2**n


def test_rank_one_haar_effect_spreads_over_all_strings() -> None:
    """Rank one gives ``W = 2^N - 1`` for any pure effect, Haar included."""
    rng = np.random.default_rng(4)
    n, depth = 3, 2
    gates = gates_of(n, depth, haar_gate, rng)
    mask = np.zeros((depth, n), dtype=bool)
    mask[-1] = True
    a = anatomy(kraus_operator(gates, mask, sample_outcomes(gates, mask, rng)))
    assert a.rank == 1
    assert a.weight == pytest.approx(2**n - 1)
    assert a.n_strings > 2**n


def test_likelihood_ratio_matches_born_rule() -> None:
    rng = np.random.default_rng(5)
    n, depth = 3, 2
    gates = gates_of(n, depth, haar_gate, rng)
    mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)
    o = sample_outcomes(gates, mask, rng)
    k = kraus_operator(gates, mask, o)
    a = anatomy(k)
    states = product_states(n, 20, rng)
    prob = np.sum(np.abs(states @ k.T) ** 2, axis=1)
    np.testing.assert_allclose(
        likelihood_ratio(a.effect, states), prob / a.prob, atol=1e-10
    )
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-12)
