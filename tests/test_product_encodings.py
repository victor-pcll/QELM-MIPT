"""Tensor-product encodings of issue #196 (``QELM.encoding.ProductEncoding``)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM import load_config
from QELM.encoding import (
    AngleEncoding,
    DenseAngleEncoding,
    GeneralEncoding,
    OutOfBoundsError,
    ProductEncoding,
    RotationEncoding,
    UniformBlochEncoding,
    get_encoding,
)
from QELM.evaluation.double_descent import expectations, observable_operators

ENCODINGS: tuple[type[ProductEncoding], ...] = (
    AngleEncoding,
    DenseAngleEncoding,
    UniformBlochEncoding,
    GeneralEncoding,
)


def _kron(qubits: list[np.ndarray]) -> np.ndarray:
    """Product state, site 0 the leftmost factor."""
    psi = np.ones(1, dtype=complex)
    for q in qubits:
        psi = np.kron(psi, q)
    return psi


def _inside(cls: type[ProductEncoding], N: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` random inputs inside the bounds of ``cls``, shape ``(n, M)``."""
    enc = cls({"N": N})
    b = enc.feature_bounds()
    rng = np.random.default_rng(seed)
    return b[:, 0] + (b[:, 1] - b[:, 0]) * rng.uniform(0.05, 0.95, (n, len(b)))


# A dummy input, the same for every encoding: inside every interval.
DUMMY = np.array([0.3, 0.8, 0.5, 0.1])


def test_angle_encoding_matches_its_formula() -> None:
    psi = AngleEncoding({"N": 4}).state(DUMMY)
    expected = _kron([np.array([np.cos(x / 2), np.sin(x / 2)]) for x in DUMMY])
    np.testing.assert_allclose(psi, expected, atol=1e-14)


def test_dense_angle_encoding_matches_its_formula() -> None:
    psi = DenseAngleEncoding({"N": 2}).state(DUMMY)
    expected = _kron(
        [
            np.array([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)])
            for t, p in (DUMMY[0:2], DUMMY[2:4])
        ]
    )
    np.testing.assert_allclose(psi, expected, atol=1e-14)


def test_uniform_bloch_encoding_matches_its_formula() -> None:
    psi = UniformBlochEncoding({"N": 2}).state(DUMMY)
    expected = _kron(
        [
            np.array([np.sqrt(u), np.exp(1j * p) * np.sqrt(1 - u)])
            for u, p in (DUMMY[0:2], DUMMY[2:4])
        ]
    )
    np.testing.assert_allclose(psi, expected, atol=1e-14)


def test_general_encoding_matches_its_formula() -> None:
    psi = GeneralEncoding({"N": 2}).state(DUMMY)
    expected = _kron(
        [np.array([a, b]) / np.hypot(a, b) for a, b in (DUMMY[0:2], DUMMY[2:4])]
    )
    np.testing.assert_allclose(psi, expected, atol=1e-14)


def test_angle_encoding_is_the_y_rotation() -> None:
    x = _inside(AngleEncoding, 3, 6)
    np.testing.assert_allclose(
        AngleEncoding({"N": 3}).states(x),
        RotationEncoding({"N": 3, "axis": "y"}).states(x),
        atol=1e-14,
    )


@pytest.mark.parametrize("cls", ENCODINGS)
def test_states_are_normalized_and_match_the_bloch_vectors(
    cls: type[ProductEncoding],
) -> None:
    N = 3
    enc = cls({"N": N})
    x = _inside(cls, N, 7)
    psi = enc.states(x)
    assert psi.shape == (7, 2**N)
    np.testing.assert_allclose(np.linalg.norm(psi, axis=1), 1.0, atol=1e-12)
    for n in range(7):
        np.testing.assert_allclose(psi[n], enc.state(x[n]), atol=1e-14)
    _, ops = observable_operators(N, ["X", "Y", "Z"])
    measured = expectations(psi, ops).reshape(7, 3, N).transpose(0, 2, 1)
    np.testing.assert_allclose(measured, enc.bloch_vectors(x), atol=1e-12)
    rho = enc.rho(x[0])
    np.testing.assert_allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho).real, 1.0)
    np.testing.assert_allclose(enc.state(x[0], sparse=True).toarray()[:, 0], psi[0])


@pytest.mark.parametrize("cls", ENCODINGS)
def test_out_of_bounds_inputs_raise(cls: type[ProductEncoding]) -> None:
    enc = cls({"N": 2})
    x = _inside(cls, 2, 1)[0]
    width = enc.feature_bounds()[:, 1] - enc.feature_bounds()[:, 0]
    for k in range(enc.n_features):
        for shift in (-0.01, 1.01):
            bad = x.copy()
            bad[k] = enc.feature_bounds()[k, 0] + shift * width[k]
            with pytest.raises(OutOfBoundsError, match=f"feature {k}"):
                enc.state(bad)
    # OutOfBoundsError is a ValueError, and the bounds themselves are allowed.
    assert issubclass(OutOfBoundsError, ValueError)
    enc.states(
        np.stack([enc.feature_bounds()[:, 0] + 1e-3, enc.feature_bounds()[:, 1]])
    )


def test_bounds_of_each_encoding() -> None:
    pi = np.pi
    assert AngleEncoding({"N": 2}).feature_bounds().tolist() == [[0, pi], [0, pi]]
    assert DenseAngleEncoding({"N": 1}).feature_bounds().tolist() == [[0, pi], [0, pi]]
    assert UniformBlochEncoding({"N": 1}).feature_bounds().tolist() == [[0, 1], [0, pi]]
    assert GeneralEncoding({"N": 1}).feature_bounds().tolist() == [[0, 1], [0, 1]]


def test_two_features_per_qubit_take_half_the_qubits() -> None:
    enc = DenseAngleEncoding({"N": 3})
    assert enc.n_features == 6
    # An odd number of features: the last qubit gets a phase of 0.
    odd = DenseAngleEncoding({"N": 3, "n_features": 5})
    x = np.array([0.3, 0.8, 0.5, 0.1, 2.0])
    np.testing.assert_allclose(odd.state(x), enc.state(np.append(x, 0.0)), atol=1e-14)
    with pytest.raises(ValueError, match="need"):
        DenseAngleEncoding({"N": 3, "n_features": 7})
    with pytest.raises(ValueError, match="shape"):
        enc.state(np.zeros(5))


def test_general_encoding_rejects_a_zero_pair() -> None:
    with pytest.raises(ValueError, match="both features 0"):
        GeneralEncoding({"N": 2}).state(np.array([0.5, 0.5, 0.0, 0.0]))


def test_known_states() -> None:
    # |1> for a polar angle pi; the phase then shows in <X>, <Y>.
    np.testing.assert_allclose(
        AngleEncoding({"N": 1}).state(np.array([np.pi])), [0, 1], atol=1e-15
    )
    r = DenseAngleEncoding({"N": 1}).bloch_vectors(np.array([np.pi / 2, np.pi / 2]))
    np.testing.assert_allclose(r, [[0.0, 1.0, 0.0]], atol=1e-15)
    # Uniform Bloch: <Z> = 2 x_1 - 1, linear in the first feature.
    u = np.linspace(0, 1, 5)
    x = np.stack([u, np.zeros_like(u)], axis=1)
    z = UniformBlochEncoding({"N": 1}).bloch_vectors(x)[:, 0, 2]
    np.testing.assert_allclose(z, 2 * u - 1, atol=1e-15)
    # General: a real state, R_y(2 atan2(x_2, x_1))|0>.
    x = np.array([0.2, 0.7])
    theta = 2 * np.arctan2(x[1], x[0])
    np.testing.assert_allclose(
        GeneralEncoding({"N": 1}).state(x),
        AngleEncoding({"N": 1}).state(np.array([theta])),
    )


def test_registry_and_config() -> None:
    cfg = load_config(config_name="double_descent")
    for cls in ENCODINGS:
        assert get_encoding(cls.name) is cls
        enc = cls.from_config(cfg)
        assert (enc.N, enc.Nr) == (8, 0)
        assert enc.n_features == 8 * cls.features_per_qubit
