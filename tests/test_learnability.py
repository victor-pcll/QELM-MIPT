"""Charge learnability of Barratt et al. 2022 (issues #113 and #115)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from QELM.evaluation.learnability import (
    UNMEASURED,
    Records,
    apply_gates,
    binary_entropy,
    brickwork_bonds,
    charge_state,
    correct_probability,
    default_charges,
    learnability_bootstrap_errors,
    learnability_statistics,
    random_u1_gates,
    sample_records,
    sep_log_likelihood,
    sep_posterior,
    site_bits,
)
from QELM.models.circuits import U1BrickworkCircuit, apply_two_qubit_gate


def test_site_bits_matches_the_circuit_convention() -> None:
    """Le site 0 est le bit de poids fort, comme dans models.circuits."""
    bits = site_bits(3)
    assert bits.shape == (8, 3)
    np.testing.assert_array_equal(bits[0b100], [1, 0, 0])
    np.testing.assert_array_equal(bits[0b011], [0, 1, 1])


def test_charge_state_is_the_uniform_superposition() -> None:
    psi = charge_state(4, 2)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    support = np.flatnonzero(psi)
    assert support.size == math.comb(4, 2)
    np.testing.assert_array_equal(site_bits(4)[support].sum(axis=1), 2)
    np.testing.assert_allclose(psi[support], 1 / math.sqrt(6))
    with pytest.raises(ValueError, match="Q must be"):
        charge_state(4, 5)


def test_default_charges() -> None:
    assert default_charges(6) == (3, 2)
    assert default_charges(7) == (3, 2)


def test_bonds_match_the_brickwork_circuit() -> None:
    for L, boundary in [(6, "open"), (5, "open"), (6, "periodic")]:
        c = U1BrickworkCircuit({"N": L, "Nr": 0, "depth": 1, "boundary": boundary})
        assert brickwork_bonds(L, boundary) == [(i, j) for i, j, _ in c.layers[0]]
    assert brickwork_bonds(6) == [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]


def test_random_u1_gates_are_unitary_and_conserve_charge() -> None:
    gates = random_u1_gates(50, np.random.default_rng(0))
    eye = np.broadcast_to(np.eye(4), gates.shape)
    np.testing.assert_allclose(gates @ gates.conj().transpose(0, 2, 1), eye, atol=1e-12)
    charge = np.diag([0.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(gates @ charge - charge @ gates, 0.0, atol=1e-12)


def test_random_u1_gates_haar_block_moment() -> None:
    """Bloc de Haar sur U(2) : E|U_01,01|^2 = 1/2, la probabilité de saut du SEP."""
    gates = random_u1_gates(20000, np.random.default_rng(1))
    assert abs(np.mean(np.abs(gates[:, 1, 1]) ** 2) - 0.5) < 0.01
    assert abs(np.mean(np.abs(gates[:, 1, 2]) ** 2) - 0.5) < 0.01


def test_apply_gates_matches_the_single_state_version() -> None:
    rng = np.random.default_rng(2)
    L, n = 4, 5
    psi = rng.standard_normal((n, 2**L)) + 1j * rng.standard_normal((n, 2**L))
    gates = random_u1_gates(n, rng)
    for i, j in [(0, 1), (2, 3), (1, 2), (3, 0)]:
        out = apply_gates(psi, gates, i, j)
        for k in range(n):
            ref = apply_two_qubit_gate(psi[k], gates[k], i, j)
            np.testing.assert_allclose(out[k], ref, atol=1e-12)


def test_sample_records_shape_and_charges() -> None:
    r = sample_records(4, 0.5, 7, rng=np.random.default_rng(0))
    assert isinstance(r, Records)
    assert r.outcomes.shape == (7, 4, 4)
    assert r.L == 4
    assert r.charges == (2, 1)
    np.testing.assert_array_equal(r.charge, [2, 1, 2, 1, 2, 1, 2])
    assert set(np.unique(r.outcomes)) <= {UNMEASURED, 0, 1}
    assert sample_records(4, 0.5, 3, depth=2).outcomes.shape == (3, 2, 4)


def test_sample_records_measurement_rate() -> None:
    r0 = sample_records(4, 0.0, 10, rng=np.random.default_rng(0))
    assert np.all(r0.outcomes == UNMEASURED)
    r = sample_records(6, 0.3, 2000, rng=np.random.default_rng(1))
    rate = np.mean(r.outcomes != UNMEASURED)
    assert abs(rate - 0.3) < 0.01


def test_full_measurement_reads_the_charge_at_every_step() -> None:
    """À p = 1 chaque pas mesure une configuration entière, de charge Q."""
    r = sample_records(6, 1.0, 40, rng=np.random.default_rng(3))
    assert np.all(r.outcomes.sum(axis=2) == r.charge[:, None])


def test_sample_records_is_reproducible() -> None:
    a = sample_records(4, 0.4, 20, rng=np.random.default_rng(7))
    b = sample_records(4, 0.4, 20, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.outcomes, b.outcomes)


def test_sample_records_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="L >= 2"):
        sample_records(1, 0.5, 1)
    with pytest.raises(ValueError, match="p must be"):
        sample_records(4, 1.5, 1)
    with pytest.raises(ValueError, match="n_records"):
        sample_records(4, 0.5, 0)
    with pytest.raises(ValueError, match="distinct charges"):
        sample_records(4, 0.5, 1, charges=[2, 2])


def _all_records(L: int, T: int) -> list[np.ndarray]:
    """Every record of L sites and T steps: each site unmeasured, 0 or 1."""
    values = (UNMEASURED, 0, 1)
    return [
        np.array(r, dtype=np.int8).reshape(T, L)
        for r in itertools.product(values, repeat=L * T)
    ]


def test_sep_probabilities_of_a_fixed_mask_sum_to_one() -> None:
    """Pour un masque de mesure fixé, sum_m P(m | Q) = 1 (règle de Born moyenne)."""
    L, T = 3, 2
    records = np.stack(_all_records(L, T))
    P = np.exp(sep_log_likelihood(records, 1))
    masks = records != UNMEASURED
    for mask in np.unique(masks.reshape(len(records), -1), axis=0):
        same = np.all(masks.reshape(len(records), -1) == mask, axis=1)
        assert np.isclose(P[same].sum(), 1.0)


def test_sep_two_sites_by_hand() -> None:
    """L = 2, Q = 1 : |01) et |10) équiprobables, le SEP les garde ainsi."""
    one = np.array([[[0, UNMEASURED]], [[1, UNMEASURED]], [[1, 1]], [[0, 1]]])
    P = np.exp(sep_log_likelihood(one, 1))
    np.testing.assert_allclose(P, [0.5, 0.5, 0.0, 0.5])


def test_quantum_records_follow_the_sep_distribution() -> None:
    """Moyennée sur les portes de Haar, la règle de Born est exactement le SEP.

    On tire des records à p = 1 (L = 4, T = 2, Q = 2) depuis les circuits
    quantiques et on compare leurs fréquences aux probabilités du SEP.
    """
    n = 30000
    r = sample_records(4, 1.0, n, depth=2, charges=[2, 1], rng=np.random.default_rng(5))
    outcomes = r.outcomes[r.charge == 2]
    keys, counts = np.unique(
        outcomes.reshape(len(outcomes), -1), axis=0, return_counts=True
    )
    P = np.exp(sep_log_likelihood(keys.reshape(-1, 2, 4), 2))
    assert np.isclose(P.sum(), 1.0)
    freq = counts / len(outcomes)
    sigma = np.sqrt(P * (1 - P) / len(outcomes))
    assert np.all(np.abs(freq - P) < 5 * sigma + 1e-3)


def test_posterior_limits() -> None:
    """p = 0 : aucune information, P_corr = 1/2. p = 1 : la charge est lue."""
    rng = np.random.default_rng(0)
    blind = sample_records(6, 0.0, 20, rng=rng)
    np.testing.assert_allclose(correct_probability(blind), 0.5)
    full = sample_records(6, 1.0, 20, rng=rng)
    np.testing.assert_allclose(correct_probability(full), 1.0)


def test_posterior_rows_sum_to_one() -> None:
    r = sample_records(6, 0.2, 200, rng=np.random.default_rng(4))
    post = sep_posterior(r.outcomes, r.charges)
    assert post.shape == (200, 2)
    np.testing.assert_allclose(post.sum(axis=1), 1.0)
    assert np.all((post >= 0) & (post <= 1))


def test_eve_learns_better_with_more_measurements() -> None:
    rng = np.random.default_rng(6)
    means = [
        learnability_statistics(
            correct_probability(sample_records(6, p, 3000, rng=rng))
        )["entropy_mean"]
        for p in (0.05, 0.2, 0.5)
    ]
    assert means[0] > means[1] > means[2]


def test_binary_entropy_values() -> None:
    np.testing.assert_allclose(binary_entropy(np.array([0.0, 0.5, 1.0])), [0, 1, 0])
    assert np.isclose(binary_entropy(np.array([0.5]), base=np.e)[0], math.log(2))
    H = binary_entropy(np.array([0.1, 0.9]))
    np.testing.assert_allclose(H, -0.1 * np.log2(0.1) - 0.9 * np.log2(0.9))


def test_learnability_statistics_ties_and_values() -> None:
    P = np.array([0.5, 0.5, 1.0, 0.0])
    s = learnability_statistics(P)
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["accuracy_strict"] == pytest.approx(0.25)
    assert s["P_corr_mean"] == pytest.approx(0.5)
    assert s["entropy_mean"] == pytest.approx(0.5)
    assert s["entropy_var"] == pytest.approx(0.25)
    # Deux pics égaux en 0 et 1 : B = 2/3.
    assert s["entropy_binder"] == pytest.approx(2 / 3)


def test_bootstrap_errors_have_every_key() -> None:
    P = np.random.default_rng(0).uniform(0.3, 1.0, size=500)
    err = learnability_bootstrap_errors(P, n_boot=20, rng=np.random.default_rng(1))
    assert set(err) == {f"{k}_err" for k in learnability_statistics(P)}
    assert all(np.isfinite(v) and v >= 0 for v in err.values())
    with pytest.raises(ValueError, match="n_boot"):
        learnability_bootstrap_errors(P, n_boot=1)
