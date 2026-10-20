"""Weak charge measurement of QELM.training.measurements.weak (issue #137)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM import load_config
from QELM.training import WeakMeasurement, get_readout
from QELM.training.measurements import measure_weak, weak_kraus


def _plus_state(N: int) -> np.ndarray:
    return np.full(2**N, 2 ** (-N / 2), dtype=complex)


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.5, 1.0])
def test_kraus_operators_are_complete(gamma: float) -> None:
    """K_0^dagger K_0 + K_1^dagger K_1 = 1."""
    K0, K1 = weak_kraus(gamma)
    np.testing.assert_allclose(K0.T @ K0 + K1.T @ K1, np.eye(2), atol=1e-12)


def test_kraus_limits() -> None:
    """gamma = 0 : identité et pas de clic ; gamma = 1 : projecteurs."""
    K0, K1 = weak_kraus(0.0)
    np.testing.assert_allclose(K0, np.eye(2))
    np.testing.assert_allclose(K1, 0.0)
    K0, K1 = weak_kraus(1.0)
    np.testing.assert_allclose(K0, np.diag([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(K1, np.diag([0.0, 1.0]), atol=1e-12)


@pytest.mark.parametrize("gamma", [-0.1, 1.5, True, "0.5"])
def test_gamma_is_checked(gamma: object) -> None:
    with pytest.raises(ValueError):
        weak_kraus(gamma)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WeakMeasurement({"gamma": gamma})


def test_gamma_zero_leaves_the_state_unchanged() -> None:
    psi = _plus_state(3)
    rng = np.random.default_rng(0)
    for q in range(3):
        out, m = measure_weak(psi, q, 0.0, rng)
        assert m == 0
        np.testing.assert_allclose(out, psi)


def test_gamma_one_is_projective() -> None:
    """À gamma = 1, un clic projette sur |1>, sinon sur |0>."""
    rng = np.random.default_rng(1)
    psi = _plus_state(2)
    for _ in range(20):
        out, m = measure_weak(psi, 0, 1.0, rng)
        bits = (np.arange(4) >> 1) & 1
        assert np.allclose(out[bits != m], 0.0)
        assert abs(np.linalg.norm(out) - 1) < 1e-12


def test_click_rate_follows_born_rule() -> None:
    """P(clic) = sin(gamma pi / 2)**2 <n> ; ici <n> = 1/2."""
    gamma = 0.4
    rng = np.random.default_rng(2)
    psi = _plus_state(1)
    clicks = [measure_weak(psi, 0, gamma, rng)[1] for _ in range(20000)]
    expected = np.sin(gamma * np.pi / 2) ** 2 / 2
    assert abs(np.mean(clicks) - expected) < 4 * np.sqrt(expected / 20000)


def test_no_click_updates_the_odds() -> None:
    """Sans clic, |1> est atténué par cos(gamma pi / 2), puis renormalisé."""
    gamma = 0.5
    psi = _plus_state(1)
    rng = np.random.default_rng(3)
    out, m = measure_weak(psi, 0, gamma, rng)
    while m == 1:
        out, m = measure_weak(psi, 0, gamma, rng)
    c = np.cos(gamma * np.pi / 2)
    np.testing.assert_allclose(out, np.array([1.0, c]) / np.sqrt(1 + c**2))


def test_vector_and_density_matrix_agree() -> None:
    """Même graine : même résultat pour psi et rho = |psi><psi|."""
    rng = np.random.default_rng(4)
    psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    for q in range(3):
        a, ma = measure_weak(psi, q, 0.6, np.random.default_rng(q))
        b, mb = measure_weak(rho, q, 0.6, np.random.default_rng(q))
        assert ma == mb
        np.testing.assert_allclose(np.outer(a, a.conj()), b, atol=1e-12)
        assert abs(np.trace(b) - 1) < 1e-12


def test_measure_weak_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        measure_weak(np.ones(6), 0, 0.5)
    with pytest.raises(ValueError):
        measure_weak(_plus_state(2), 2, 0.5)


def test_readout_skips_the_ancilla_and_records_clicks() -> None:
    """Les qubits système sont mesurés, l'ancilla jamais ; 0/1 par qubit."""
    P = WeakMeasurement({"gamma": 1.0, "Nr": 1})
    result = P.measure(_plus_state(4), rng=np.random.default_rng(5))
    assert result.outcomes.shape == (3,)
    assert set(result.outcomes) <= {0.0, 1.0}
    # The 3 system qubits are now in a basis state; the ancilla is still |+>.
    psi = result.rho
    assert np.count_nonzero(np.abs(psi) > 1e-12) == 2
    # A p passed to measure overrides the configured gamma.
    still = P.measure(_plus_state(4), p=0.0)
    np.testing.assert_allclose(still.rho, _plus_state(4))
    np.testing.assert_array_equal(still.outcomes, 0.0)


def test_readout_from_config_and_registry() -> None:
    cfg = load_config(
        config_name="charge_sharpening", overrides=["weak_measurement.gamma=0.3"]
    )
    P = WeakMeasurement.from_config(cfg)
    assert P.gamma == 0.3
    assert P.Nr == 1
    assert P.qubits is None
    assert get_readout("weak_measurement") is WeakMeasurement
    assert WeakMeasurement({}).gamma == 1.0
