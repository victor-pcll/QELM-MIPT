"""Tests de l'évolution unitaire exp(-i H dt) des modèles hamiltoniens."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.models.TFIM import TransverseFieldIsingModel


@pytest.fixture
def model() -> TransverseFieldIsingModel:
    """TFIM sur 3 qubits, désordre fixé par la seed."""
    np.random.seed(0)
    return TransverseFieldIsingModel({"N": 3, "J": 1.0, "h": 0.5})


def _random_state(dim: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


def test_propagator_is_unitary(model: TransverseFieldIsingModel) -> None:
    """U U^dagger = 1."""
    U = model.propagator(dt=0.7)
    assert U.shape == (model.dim, model.dim)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(model.dim), atol=1e-12)


def test_propagator_at_zero_time_is_identity(
    model: TransverseFieldIsingModel,
) -> None:
    """U(0) = 1."""
    np.testing.assert_allclose(model.propagator(dt=0.0), np.eye(model.dim))


def test_propagator_composes(model: TransverseFieldIsingModel) -> None:
    """U(t1) U(t2) = U(t1 + t2), car H ne dépend pas du temps."""
    np.testing.assert_allclose(
        model.propagator(dt=0.3) @ model.propagator(dt=0.4),
        model.propagator(dt=0.7),
        atol=1e-12,
    )


def test_single_qubit_known_phase() -> None:
    """N=1 : H = h Z, donc |0> prend la phase exp(-i h dt)."""
    model = TransverseFieldIsingModel({"N": 1, "J": 1.0, "h": 0.5})
    psi = model.evolve_state(dt=2.0, psi=np.array([1, 0], dtype=complex))
    np.testing.assert_allclose(psi, [np.exp(-1j * 0.5 * 2.0), 0], atol=1e-12)


def test_evolve_state_preserves_norm_and_energy(
    model: TransverseFieldIsingModel,
) -> None:
    """La norme et <H> sont conservées par l'évolution unitaire."""
    H = model.hamiltonian()
    psi0 = _random_state(model.dim)
    psi = model.evolve_state(dt=1.3, psi=psi0)
    assert psi.shape == psi0.shape
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert np.vdot(psi, H @ psi).real == pytest.approx(np.vdot(psi0, H @ psi0).real)


def test_evolve_state_accepts_columns_of_states(
    model: TransverseFieldIsingModel,
) -> None:
    """Plusieurs états en colonnes évoluent chacun comme seuls."""
    states = np.stack([_random_state(model.dim, s) for s in (1, 2)], axis=1)
    evolved = model.evolve_state(dt=0.5, psi=states)
    for k in range(2):
        np.testing.assert_allclose(
            evolved[:, k], model.evolve_state(dt=0.5, psi=states[:, k]), atol=1e-12
        )


def test_evolve_density_matches_pure_state(
    model: TransverseFieldIsingModel,
) -> None:
    """Pour rho = |psi><psi|, U rho U^dagger = |U psi><U psi|."""
    psi0 = _random_state(model.dim)
    rho = model.evolve_density(dt=0.9, rho=np.outer(psi0, psi0.conj()))
    psi = model.evolve_state(dt=0.9, psi=psi0)
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-12)


def test_evolve_density_stays_a_density_matrix(
    model: TransverseFieldIsingModel,
) -> None:
    """Trace 1, hermitienne et positive après évolution d'un état mixte."""
    weights = [0.6, 0.4]
    rho0 = sum(
        w * np.outer(p, p.conj())
        for w, p in zip(
            weights, (_random_state(model.dim, 1), _random_state(model.dim, 2))
        )
    )
    rho = model.evolve_density(dt=1.1, rho=rho0)
    assert np.trace(rho).real == pytest.approx(1.0)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-12
