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
    psi = model.evolve(dt=2.0, state=np.array([1, 0], dtype=complex))
    np.testing.assert_allclose(psi, [np.exp(-1j * 0.5 * 2.0), 0], atol=1e-12)


def test_evolve_psi_preserves_norm_and_energy(
    model: TransverseFieldIsingModel,
) -> None:
    """La norme et <H> sont conservées par l'évolution unitaire."""
    H = model.hamiltonian()
    psi0 = _random_state(model.dim)
    psi = model.evolve(dt=1.3, state=psi0)
    assert psi.shape == psi0.shape
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert np.vdot(psi, H @ psi).real == pytest.approx(np.vdot(psi0, H @ psi0).real)


def test_evolve_rho_matches_pure_state(
    model: TransverseFieldIsingModel,
) -> None:
    """Pour rho = |psi><psi|, U rho U^dagger = |U psi><U psi|."""
    psi0 = _random_state(model.dim)
    rho = model.evolve(dt=0.9, state=np.outer(psi0, psi0.conj()))
    psi = model.evolve(dt=0.9, state=psi0)
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-12)


def test_evolve_rho_stays_a_density_matrix(
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
    rho = model.evolve(dt=1.1, state=rho0)
    assert np.trace(rho).real == pytest.approx(1.0)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_evolve_returns_the_kind_it_receives(
    model: TransverseFieldIsingModel,
) -> None:
    """Un psi de forme (dim,) donne un psi, un rho de forme (dim, dim) un rho."""
    psi0 = _random_state(model.dim)
    psi = model.evolve(dt=0.4, state=psi0)
    rho = model.evolve(dt=0.4, state=np.outer(psi0, psi0.conj()))
    assert psi.shape == (model.dim,)
    assert rho.shape == (model.dim, model.dim)
    assert np.iscomplexobj(psi) and np.iscomplexobj(rho)


def test_evolve_accepts_real_input(model: TransverseFieldIsingModel) -> None:
    """Un |0...0> réel est promu en complexe, comme un psi complexe."""
    psi0 = np.zeros(model.dim)
    psi0[0] = 1.0
    np.testing.assert_allclose(
        model.evolve(dt=0.6, state=psi0),
        model.evolve(dt=0.6, state=psi0.astype(complex)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "shape",
    [(4,), (9,), (8, 1), (8, 3), (4, 4), (8, 8, 1), ()],
)
def test_evolve_rejects_wrong_shape(
    model: TransverseFieldIsingModel, shape: tuple[int, ...]
) -> None:
    """Ni (dim,) ni (dim, dim) : erreur claire qui donne les formes attendues."""
    with pytest.raises(ValueError, match=r"\(8,\).*\(8, 8\).*got shape"):
        model.evolve(dt=0.1, state=np.zeros(shape, dtype=complex))
