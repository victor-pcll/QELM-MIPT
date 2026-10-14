"""Tests des mesures projectives et de l'interface Readout commune."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.physics import (
    DoubleObservable,
    ProjectiveMeasurement,
    Readout,
    SingleObservable,
    get_readout,
    measure_projective_rho,
)


def _pure(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def _random_rho(N: int, seed: int) -> np.ndarray:
    """État mixte aléatoire de rang 3 sur N qubits."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2**N, 3)) + 1j * rng.normal(size=(2**N, 3))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


ZERO = _pure([1, 0])
ONE = _pure([0, 1])
PLUS = _pure([1, 1])
BELL = _pure([1, 0, 0, 1])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_collapsed_state_is_a_density_matrix(seed: int, qubit: int) -> None:
    """Après effondrement : trace 1, hermitienne, positive."""
    rho = _random_rho(3, seed)
    out, mi = measure_projective_rho(rho, qubit, 3, np.random.default_rng(seed))
    assert mi in (-1, 1)
    np.testing.assert_allclose(np.trace(out), 1, atol=1e-12)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() > -1e-12


@pytest.mark.parametrize("seed", range(3))
def test_collapse_projects_the_measured_qubit(seed: int) -> None:
    """<Z_q> vaut -mi après la mesure (outcome 0 <-> |0>, Z=+1 ; mi=-1)."""
    rho = _random_rho(3, seed)
    out, mi = measure_projective_rho(rho, 1, rng=np.random.default_rng(seed))
    z1 = SingleObservable({"observable": "Z"}).expectation(out)[1]
    np.testing.assert_allclose(z1, -mi, atol=1e-12)


def test_remeasuring_gives_the_same_outcome() -> None:
    """Mesure projective idempotente : re-mesurer redonne le même résultat."""
    rng = np.random.default_rng(1)
    rho = _random_rho(2, 1)
    out, mi = measure_projective_rho(rho, 0, rng=rng)
    for _ in range(10):
        out2, mi2 = measure_projective_rho(out, 0, rng=rng)
        assert mi2 == mi
        np.testing.assert_allclose(out2, out, atol=1e-12)


def test_basis_states_are_deterministic() -> None:
    """|0> donne toujours mi=-1 et |1> toujours mi=+1, sans changer l'état."""
    rng = np.random.default_rng(0)
    for rho, expected in ((ZERO, -1), (ONE, 1)):
        for _ in range(50):
            out, mi = measure_projective_rho(rho, 0, rng=rng)
            assert mi == expected
            np.testing.assert_allclose(out, rho)
    psi = np.zeros(8)
    psi[0b010] = 1  # |010>
    rho = _pure(psi)
    readout = ProjectiveMeasurement({})
    for _ in range(20):
        result = readout.measure(rho, rng)
        np.testing.assert_array_equal(result.values, [-1, 1, -1])
        np.testing.assert_allclose(result.rho, rho)


def _frequency_of_one(rho: np.ndarray, qubit: int, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    hits = sum(measure_projective_rho(rho, qubit, rng=rng)[1] == 1 for _ in range(n))
    return hits / n


def test_plus_state_follows_born_rule() -> None:
    """|+> : p1 = 1/2. Avec n = 20000 tirages, écart-type 0.0035 ; tolérance 5 sigma."""
    n = 20000
    sigma = np.sqrt(0.25 / n)
    assert abs(_frequency_of_one(PLUS, 0, n, seed=3) - 0.5) < 5 * sigma


def test_tilted_state_follows_born_rule() -> None:
    """cos(t)|0> + sin(t)|1> : p1 = sin(t)^2."""
    t = 0.4
    rho = _pure([np.cos(t), np.sin(t)])
    p1 = np.sin(t) ** 2
    n = 20000
    sigma = np.sqrt(p1 * (1 - p1) / n)
    assert abs(_frequency_of_one(rho, 0, n, seed=4) - p1) < 5 * sigma


def test_bell_state_outcomes_are_correlated() -> None:
    """Bell : chaque qubit 50/50, et les deux résultats toujours égaux."""
    n = 5000
    rng = np.random.default_rng(5)
    readout = ProjectiveMeasurement({})
    outcomes = np.array([readout.measure(BELL, rng).values for _ in range(n)])
    np.testing.assert_array_equal(outcomes[:, 0], outcomes[:, 1])
    frac = np.mean(outcomes[:, 0] == 1)
    assert abs(frac - 0.5) < 5 * np.sqrt(0.25 / n)


def test_bell_collapse_gives_product_state() -> None:
    """Mesurer le qubit 0 d'un Bell laisse |00> ou |11>."""
    out, mi = measure_projective_rho(BELL, 0, rng=np.random.default_rng(0))
    expected = _pure([1, 0, 0, 0]) if mi == -1 else _pure([0, 0, 0, 1])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_same_seed_gives_same_outcomes() -> None:
    """Deux Generator de même seed donnent la même suite de résultats."""
    rho = _random_rho(4, 7)
    readout = ProjectiveMeasurement({})
    a = readout.measure(rho, np.random.default_rng(42))
    b = readout.measure(rho, np.random.default_rng(42))
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.rho, b.rho)


def test_global_rng_is_used_without_generator() -> None:
    """Sans rng, np.random.seed rend les tirages reproductibles."""
    rho = _random_rho(4, 8)
    readout = ProjectiveMeasurement({})
    np.random.seed(123)
    a = readout.measure(rho).values
    np.random.seed(123)
    b = readout.measure(rho).values
    np.testing.assert_array_equal(a, b)


def test_selected_qubits_are_measured_in_order() -> None:
    """``qubits`` choisit les qubits mesurés et l'ordre des valeurs."""
    psi = np.zeros(8)
    psi[0b011] = 1  # |011>
    result = ProjectiveMeasurement({"qubits": [2, 0]}).measure(
        _pure(psi), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(result.values, [1, -1])


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="qubit_idx"):
        measure_projective_rho(BELL, 2)
    with pytest.raises(ValueError, match="does not match"):
        measure_projective_rho(BELL, 0, N=3)


def test_observables_share_the_readout_interface() -> None:
    """Les observables sont des Readout : measure renvoie <O> et rho intact."""
    obs = SingleObservable({"observable": "Z"})
    assert isinstance(obs, Readout)
    result = obs.measure(BELL)
    np.testing.assert_allclose(result.values, [0, 0], atol=1e-12)
    assert result.rho is BELL


def test_registry_and_config() -> None:
    """get_readout et from_config retrouvent chaque type de readout."""
    assert get_readout("projective_measurement") is ProjectiveMeasurement
    assert get_readout("double_observable") is DoubleObservable
    with pytest.raises(KeyError, match="Unknown readout"):
        get_readout("weak")
    cfg = OmegaConf.create({"projective_measurement": {"qubits": [1]}})
    assert ProjectiveMeasurement.from_config(cfg).qubits == [1]
