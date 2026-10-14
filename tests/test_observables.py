"""Tests des observables simples <O_i> et doubles <A_i B_j>."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.training.readout import DoubleObservable, SingleObservable


def _product_state(single: np.ndarray, N: int) -> np.ndarray:
    """Densité de |s>^{(x) N} pour un état à un qubit ``single``."""
    psi = single
    for _ in range(N - 1):
        psi = np.kron(psi, single)
    return np.outer(psi, psi.conj())


ZERO = np.array([1, 0], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_feature_counts(N: int) -> None:
    """N features simples et N (N - 1) / 2 corrélateurs."""
    rho = _product_state(ZERO, N)
    single = SingleObservable({"observable": "Z"}).expectation(rho)
    double = DoubleObservable({"observable_1": "Z", "observable_2": "Z"}).expectation(
        rho
    )
    assert single.shape == (N,)
    assert double.shape == (N * (N - 1) // 2,)


def test_known_values_on_all_zero_state() -> None:
    """|000> : <Z_i> = 1, <X_i> = 0, <Z_i Z_j> = 1."""
    rho = _product_state(ZERO, 3)
    np.testing.assert_allclose(
        SingleObservable({"observable": "Z"}).expectation(rho), 1
    )
    np.testing.assert_allclose(
        SingleObservable({"observable": "X"}).expectation(rho), 0, atol=1e-12
    )
    zz = DoubleObservable({"observable_1": "Z", "observable_2": "Z"})
    np.testing.assert_allclose(zz.expectation(rho), 1)


def test_known_values_on_plus_state() -> None:
    """|+++> : <X_i> = 1, <Z_i> = 0, <X_i X_j> = 1."""
    rho = _product_state(PLUS, 3)
    np.testing.assert_allclose(
        SingleObservable({"observable": "X"}).expectation(rho), 1
    )
    np.testing.assert_allclose(
        SingleObservable({"observable": "Z"}).expectation(rho), 0, atol=1e-12
    )
    xx = DoubleObservable({"observable_1": "X", "observable_2": "X"})
    np.testing.assert_allclose(xx.expectation(rho), 1)


def test_bell_state_has_correlations_without_magnetisation() -> None:
    """(|00> + |11>)/sqrt(2) : <Z_i> = 0 mais <Z_0 Z_1> = 1, <X_0 X_1> = 1."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    np.testing.assert_allclose(
        SingleObservable({"observable": "Z"}).expectation(rho), 0, atol=1e-12
    )
    for label in ("Z", "X"):
        obs = DoubleObservable({"observable_1": label, "observable_2": label})
        np.testing.assert_allclose(obs.expectation(rho), [1])


def test_pair_order_is_lexicographic() -> None:
    """Pour N = 3 l'ordre est (0,1), (0,2), (1,2) : seul le qubit 2 est flippé."""
    psi = np.zeros(8, dtype=complex)
    psi[0b001] = 1  # |001> : qubit 2 dans |1>
    rho = np.outer(psi, psi.conj())
    zz = DoubleObservable({"observable_1": "Z", "observable_2": "Z"})
    np.testing.assert_allclose(zz.expectation(rho), [1, -1, -1])


def test_features_are_bounded_for_random_state() -> None:
    """Les valeurs moyennes de Paulis sont réelles et dans [-1, 1]."""
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    for obs in (
        SingleObservable({"observable": "Y"}),
        DoubleObservable({"observable_1": "X", "observable_2": "Z"}),
    ):
        values = obs.expectation(rho)
        assert values.dtype == np.float64
        assert np.all(np.abs(values) <= 1 + 1e-12)


def test_unknown_label_is_rejected() -> None:
    """Une étiquette qui n'est pas X, Y ou Z lève ValueError."""
    with pytest.raises(ValueError, match="Unknown observable"):
        SingleObservable({"observable": "W"})


def test_rho_must_be_a_qubit_density_matrix() -> None:
    """Une matrice 3x3 n'est pas un état de qubits."""
    with pytest.raises(ValueError, match="2\\*\\*N"):
        SingleObservable({"observable": "Z"}).expectation(np.eye(3))


def test_from_config_reads_its_block() -> None:
    """from_config lit cfg[name], comme les modèles."""
    cfg = OmegaConf.create(
        {
            "single_observable": {"observable": "X"},
            "double_observable": {"observable_1": "Z", "observable_2": "X"},
        }
    )
    assert SingleObservable.from_config(cfg).observable == "X"
    double = DoubleObservable.from_config(cfg)
    assert (double.observable_1, double.observable_2) == ("Z", "X")
