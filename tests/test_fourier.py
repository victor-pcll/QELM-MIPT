"""Fourier expressivity of the monitored QELM (issue #309)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from QELM.evaluation.fourier import (
    accessible_rank,
    angle_moments,
    basic_measures,
    bloch_monomials,
    correlator_products,
    degree_accessibility,
    degree_profile,
    effective_dimension,
    expressivity_measures,
    feature_covariance,
    mean_degree,
    monomial_degrees,
    odd_x_mask,
    odd_x_weight,
    pauli_xz_coefficients,
    product_transfer_matrix,
    transfer_matrix,
)
from QELM.models.circuits import HaarBrickworkCircuit, U1BrickworkCircuit
from QELM.models.circuits.base import apply_layer
from QELM.models.circuits.symmetric import Z2BrickworkCircuit

Z = np.diag([1.0, -1.0])


def encoded_state(theta: np.ndarray) -> np.ndarray:
    """Density matrix of the angle encoding, ``cos(t/2)|0> + sin(t/2)|1>``."""
    psi = np.array([1.0])
    for t in theta:
        psi = np.kron(psi, [np.cos(t / 2), np.sin(t / 2)])
    return np.outer(psi, psi).astype(complex)


def z_on(i: int, N: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2**i), Z), np.eye(2 ** (N - i - 1)))


def projector(i: int, s: int, N: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2**i), np.diag([1 - s, s])), np.eye(2 ** (N - i - 1)))


def test_monomials_are_the_pauli_coefficients_of_the_encoded_state() -> None:
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, 3)
    coeffs = pauli_xz_coefficients(encoded_state(theta))
    np.testing.assert_allclose(coeffs * 2**3, bloch_monomials(theta), atol=1e-12)
    assert bloch_monomials(rng.uniform(size=(5, 3))).shape == (5, 27)


def test_degrees_and_odd_mask_follow_the_base_3_order() -> None:
    # Order for N = 2: II IX IZ XI XX XZ ZI ZX ZZ.
    np.testing.assert_array_equal(monomial_degrees(2), [0, 1, 1, 1, 2, 2, 1, 2, 2])
    np.testing.assert_array_equal(odd_x_mask(2), [0, 1, 0, 1, 0, 1, 0, 1, 0])
    # (3**N + 1) / 2 monomials have an even number of sin factors.
    assert (~odd_x_mask(5)).sum() == (3**5 + 1) // 2


def test_final_rows_at_p0_reproduce_the_unitary_evolution() -> None:
    rng = np.random.default_rng(1)
    N = 3
    circuit = HaarBrickworkCircuit({"N": N, "depth": 3, "Nr": 0}, rng)
    C = transfer_matrix(circuit.layers, N, 0.0)
    assert C.shape == (3 * N + N, 27)
    np.testing.assert_allclose(C[: 3 * N], 0.0)
    for _ in range(3):
        theta = rng.uniform(0, np.pi, N)
        rho = encoded_state(theta)
        for layer in circuit.layers:
            rho = apply_layer(rho, layer)
        direct = [np.trace(z_on(i, N) @ rho).real for i in range(N)]
        np.testing.assert_allclose(C[-N:] @ bloch_monomials(theta), direct, atol=1e-12)


def brute_force_record_means(
    layers: list, N: int, p: float, theta: np.ndarray
) -> np.ndarray:
    """``E[M_{t,i} | theta]`` by enumerating every mask and every outcome."""
    T = len(layers)
    means = np.zeros(T * N)
    for mask in itertools.product((0, 1), repeat=T * N):
        k = sum(mask)
        weight = p**k * (1 - p) ** (T * N - k)
        if weight == 0:
            continue
        branches = [(encoded_state(theta), np.zeros(T * N))]
        for t in range(T):
            new = []
            for rho, rec in branches:
                rho = apply_layer(rho, layers[t])
                cur = [(rho, rec)]
                for i in range(N):
                    if mask[t * N + i]:
                        nxt = []
                        for r, rc in cur:
                            for s in (0, 1):
                                P = projector(i, s, N)
                                rc2 = rc.copy()
                                rc2[t * N + i] = 1 - 2 * s
                                nxt.append((P @ r @ P, rc2))
                        cur = nxt
                new += cur
            branches = new
        for rho, rec in branches:
            means += weight * np.trace(rho).real * rec
    return means


@pytest.mark.parametrize("p", [0.3, 1.0])
def test_record_rows_match_the_enumeration_of_masks_and_outcomes(p: float) -> None:
    rng = np.random.default_rng(2)
    N = 3
    circuit = HaarBrickworkCircuit({"N": N, "depth": 2, "Nr": 0}, rng)
    C = transfer_matrix(circuit.layers, N, p, final=False)
    theta = rng.uniform(0, np.pi, N)
    expected = brute_force_record_means(circuit.layers, N, p, theta)
    np.testing.assert_allclose(C @ bloch_monomials(theta), expected, atol=1e-10)


@pytest.mark.parametrize("cls", [U1BrickworkCircuit, Z2BrickworkCircuit])
def test_z_symmetric_circuits_never_reach_odd_sin_monomials(cls: type) -> None:
    rng = np.random.default_rng(3)
    N = 4
    circuit = cls({"N": N, "depth": 4, "Nr": 0}, rng)
    for p in (0.0, 0.4, 1.0):
        C = transfer_matrix(circuit.layers, N, p)
        assert odd_x_weight(C, N) < 1e-20
        assert accessible_rank(C) <= (3**N + 1) // 2


def test_haar_circuit_reaches_odd_sin_monomials() -> None:
    rng = np.random.default_rng(4)
    circuit = HaarBrickworkCircuit({"N": 4, "depth": 4, "Nr": 0}, rng)
    assert odd_x_weight(transfer_matrix(circuit.layers, 4, 0.4), 4) > 0.05


def test_angle_moments_of_a_full_and_a_half_period() -> None:
    m, G = angle_moments(0.0, 2 * np.pi)
    np.testing.assert_allclose(m, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(G, np.diag([1, 0.5, 0.5]), atol=1e-12)
    m, G = angle_moments(0.0, np.pi)
    np.testing.assert_allclose(m, [1, 2 / np.pi, 0], atol=1e-12)
    np.testing.assert_allclose(G[0, 1], 2 / np.pi, atol=1e-12)


def test_feature_covariance_matches_sampled_inputs() -> None:
    rng = np.random.default_rng(5)
    N = 3
    circuit = HaarBrickworkCircuit({"N": N, "depth": 2, "Nr": 0}, rng)
    C = transfer_matrix(circuit.layers, N, 0.5)
    Sigma = feature_covariance(C, N, 0.0, np.pi)
    features = bloch_monomials(rng.uniform(0, np.pi, (200_000, N))) @ C.T
    np.testing.assert_allclose(Sigma, np.cov(features.T), atol=3e-3)


def test_profile_accessibility_and_effective_dimension() -> None:
    N = 3
    identity = np.eye(3**N)
    W = degree_profile(identity, N)
    assert W[0] == 0 and W.sum() == pytest.approx(1.0)
    assert 0 < mean_degree(W) <= 1
    np.testing.assert_allclose(degree_accessibility(identity, N), 1.0)
    assert effective_dimension(np.eye(5)) == pytest.approx(5.0)
    assert effective_dimension(np.zeros((2, 2))) == 0.0
    measures = expressivity_measures(identity, N)
    assert measures["rank"] == 27 and measures["rank_fraction"] == 1.0


def brute_force_outcome_moments(
    layers: list, N: int, p: float, theta: np.ndarray
) -> np.ndarray:
    """``E[F F^T | theta]`` over the records and final Z outcomes, enumerated."""
    T = len(layers)
    K = T * N + N
    second = np.zeros((K, K))
    for mask in itertools.product((0, 1), repeat=T * N):
        k = sum(mask)
        weight = p**k * (1 - p) ** (T * N - k)
        if weight == 0:
            continue
        branches = [(encoded_state(theta), np.zeros(K))]
        for t in range(T + 1):
            if t < T:
                branches = [(apply_layer(r, layers[t]), rc) for r, rc in branches]
            for i in range(N):
                if t == T or mask[t * N + i]:
                    nxt = []
                    for r, rc in branches:
                        for s in (0, 1):
                            P = projector(i, s, N)
                            rc2 = rc.copy()
                            rc2[t * N + i] = 1 - 2 * s
                            nxt.append((P @ r @ P, rc2))
                    branches = nxt
        for rho, rec in branches:
            second += weight * np.trace(rho).real * np.outer(rec, rec)
    return second


def test_single_feature_products_are_the_rows_of_the_transfer_matrix() -> None:
    rng = np.random.default_rng(6)
    N = 3
    circuit = HaarBrickworkCircuit({"N": N, "depth": 3, "Nr": 0}, rng)
    for p in (0.0, 0.35, 1.0):
        C = transfer_matrix(circuit.layers, N, p)
        P = product_transfer_matrix(circuit.layers, N, p, [(k,) for k in range(len(C))])
        np.testing.assert_allclose(P, C, atol=1e-12)


@pytest.mark.parametrize("p", [0.4, 1.0])
def test_pair_rows_match_the_enumeration_of_masks_and_outcomes(p: float) -> None:
    rng = np.random.default_rng(7)
    N = 2
    circuit = HaarBrickworkCircuit({"N": N, "depth": 2, "Nr": 0}, rng)
    pairs = correlator_products(2, N, "all")
    P = product_transfer_matrix(circuit.layers, N, p, pairs)
    theta = rng.uniform(0, np.pi, N)
    second = brute_force_outcome_moments(circuit.layers, N, p, theta)
    expected = [second[a, b] for a, b in pairs]
    np.testing.assert_allclose(P @ bloch_monomials(theta), expected, atol=1e-10)


def test_correlator_products_and_basic_measures() -> None:
    assert correlator_products(2, 3, "same-time") == [
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
        (6, 7),
        (6, 8),
        (7, 8),
    ]
    assert len(correlator_products(2, 3, "all")) == 9 * 8 // 2
    with pytest.raises(ValueError):
        correlator_products(2, 3, "none")
    measures = basic_measures(np.eye(27), 3)
    assert measures["rank"] == 27 and measures["n_features"] == 27


@pytest.mark.parametrize("cls", [U1BrickworkCircuit, Z2BrickworkCircuit])
def test_z_symmetric_correlators_never_reach_odd_sin_monomials(cls: type) -> None:
    rng = np.random.default_rng(8)
    N = 3
    circuit = cls({"N": N, "depth": 3, "Nr": 0}, rng)
    P = product_transfer_matrix(
        circuit.layers, N, 0.5, correlator_products(3, N, "all")
    )
    assert odd_x_weight(P, N) < 1e-20
