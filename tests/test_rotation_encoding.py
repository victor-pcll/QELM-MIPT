"""Angle encoding ``RotationEmbedding`` (issue #181)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from QELM import load_config
from QELM.encoding import RotationEmbedding, get_embedding
from QELM.evaluation.double_descent import expectations, observable_operators
from QELM.physics import X, Y


def _rotation(axis: str, theta: float) -> np.ndarray:
    sigma = {"x": X, "y": Y}[axis]
    return np.asarray(scipy.linalg.expm(-0.5j * theta * sigma))


@pytest.mark.parametrize("axis", ["x", "y"])
def test_state_is_the_product_of_single_qubit_rotations(axis: str) -> None:
    theta = np.array([0.3, 1.2, 2.9])
    psi = RotationEmbedding({"N": 3, "axis": axis}).state(theta)
    expected = np.ones(1, dtype=complex)
    for value in theta:  # site 0 is the leftmost factor
        expected = np.kron(expected, _rotation(axis, value) @ np.array([1, 0]))
    np.testing.assert_allclose(psi, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(psi), 1.0)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_batch_states_and_bloch_vectors(axis: str) -> None:
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, size=(7, 4))
    emb = RotationEmbedding({"N": 4, "axis": axis})
    psi = emb.states(theta)
    assert psi.shape == (7, 16)
    for n in range(7):
        np.testing.assert_allclose(psi[n], emb.state(theta[n]), atol=1e-12)
    # <X_i>, <Y_i>, <Z_i> of the simulated states against the closed form.
    _, ops = observable_operators(4, ["X", "Y", "Z"])
    measured = expectations(psi, ops).reshape(7, 3, 4).transpose(0, 2, 1)
    np.testing.assert_allclose(measured, emb.bloch_vectors(theta), atol=1e-12)


def test_y_axis_gives_a_real_state() -> None:
    psi = RotationEmbedding({"N": 3}).state(np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(psi.imag, 0.0)
    # theta = 0 leaves |0>, theta = pi gives |1>.
    np.testing.assert_allclose(
        RotationEmbedding({"N": 1}).state(np.array([np.pi])), [0, 1], atol=1e-15
    )


def test_registry_config_and_rho() -> None:
    assert get_embedding("rotation_embedding") is RotationEmbedding
    cfg = load_config(config_name="double_descent")
    emb = RotationEmbedding.from_config(cfg)
    assert (emb.N, emb.Nr, emb.axis) == (8, 0, "y")
    small = RotationEmbedding({"N": 2})
    rho = small.rho(np.array([0.5, 1.0]))
    np.testing.assert_allclose(np.trace(rho), 1.0)
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-12)
    assert small.state(np.array([0.5, 1.0]), sparse=True).shape == (4, 1)


def test_errors() -> None:
    with pytest.raises(ValueError, match="axis"):
        RotationEmbedding({"N": 2, "axis": "z"})
    emb = RotationEmbedding({"N": 2})
    with pytest.raises(ValueError, match="needs the input"):
        emb.state()
    with pytest.raises(ValueError, match="shape"):
        emb.state(np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        emb.states(np.zeros((4, 3)))
