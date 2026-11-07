"""QELM scores on targets of known Fourier degree (issue #309)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.fourier import bloch_monomials, transfer_matrix
from QELM.evaluation.fourier_tasks import (
    degree_target,
    gaussian_accuracy,
    ideal_r2,
    predicted_r2,
    qelm_scores,
    sample_records,
    shot_noise_covariance,
    target_variance,
)
from QELM.models.circuits import HaarBrickworkCircuit


def circuit_matrix(N: int, p: float, seed: int) -> tuple[list, np.ndarray]:
    circuit = HaarBrickworkCircuit(
        {"N": N, "depth": 3, "Nr": 0}, np.random.default_rng(seed)
    )
    return circuit.layers, transfer_matrix(circuit.layers, N, p)


def test_degree_target_is_centered_unit_variance_and_of_one_degree() -> None:
    rng = np.random.default_rng(0)
    N = 3
    yhat, offset = degree_target(N, 2, rng)
    assert target_variance(yhat, N) == pytest.approx(1.0)
    theta = rng.uniform(0, np.pi, (200_000, N))
    y = bloch_monomials(theta) @ yhat - offset
    assert abs(y.mean()) < 0.01 and y.var() == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        degree_target(N, 4, rng)


def test_ideal_r2_is_one_for_a_feature_mean_and_matches_a_regression() -> None:
    rng = np.random.default_rng(1)
    N = 3
    _, C = circuit_matrix(N, 0.3, seed=1)
    assert ideal_r2(C, C[-1], N) == pytest.approx(1.0, abs=1e-8)
    yhat, offset = degree_target(N, 2, rng)
    theta = rng.uniform(0, np.pi, (200_000, N))
    B = bloch_monomials(theta)
    phi = np.hstack([B @ C.T, np.ones((len(B), 1))])
    y = B @ yhat - offset
    w = np.linalg.lstsq(phi, y, rcond=None)[0]
    r2 = 1 - np.mean((phi @ w - y) ** 2) / y.var()
    assert ideal_r2(C, yhat, N) == pytest.approx(r2, abs=0.01)


def test_sampled_records_average_to_the_rows_of_the_transfer_matrix() -> None:
    rng = np.random.default_rng(2)
    N, p = 3, 0.4
    layers, C = circuit_matrix(N, p, seed=2)
    theta = rng.uniform(0, np.pi, N)
    records = sample_records(layers, N, p, np.tile(theta, (20_000, 1)), rng)
    assert records.shape == (20_000, 3 * N + N)
    assert set(np.unique(records)) <= {-1.0, 0.0, 1.0}
    np.testing.assert_allclose(
        records.mean(axis=0), C @ bloch_monomials(theta), atol=0.03
    )
    empty = sample_records(layers, N, 0.0, theta[None, :], rng)
    np.testing.assert_array_equal(empty[0, : 3 * N], 0.0)
    assert np.all(empty[0, 3 * N :] != 0)


def test_predicted_r2_matches_the_simulated_single_shot_qelm() -> None:
    rng = np.random.default_rng(3)
    N, p = 3, 0.4
    layers, C = circuit_matrix(N, p, seed=3)
    yhat, offset = degree_target(N, 1, rng)
    theta = rng.uniform(0, np.pi, (20_000, N))
    records = sample_records(layers, N, p, theta, rng)
    noise = shot_noise_covariance(records, C, theta)
    y = bloch_monomials(theta) @ yhat - offset
    scores = qelm_scores(records, y, 15_000)
    pred = predicted_r2(C, yhat, noise, 1, N)
    assert scores["r2_test"] == pytest.approx(pred, abs=0.03)
    assert scores["accuracy_test"] == pytest.approx(gaussian_accuracy(pred), abs=0.03)
    # More shots can only help, up to the ceiling.
    many = [predicted_r2(C, yhat, noise, s, N) for s in (1, 10, 100, np.inf)]
    assert all(a <= b + 1e-12 for a, b in zip(many, many[1:], strict=False))
    assert many[-1] == pytest.approx(ideal_r2(C, yhat, N))


def test_gaussian_accuracy_limits() -> None:
    assert gaussian_accuracy(0.0) == pytest.approx(0.5)
    assert gaussian_accuracy(1.0) == pytest.approx(1.0)
    assert gaussian_accuracy(-0.2) == pytest.approx(0.5)
