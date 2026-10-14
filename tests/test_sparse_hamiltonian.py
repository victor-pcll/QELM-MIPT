"""Le hamiltonien creux (CSR) et l'évolution creuse redonnent la version dense.

La référence dense est reconstruite ici avec ``get_site_operator`` (np.kron)
et ``scipy.linalg.expm``, comme le faisait le code avant le passage au creux.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.physics.operators import (
    X,
    Y,
    Z,
    get_site_operator,
    get_site_operator_sparse,
)


def _dense_reference(model: TransverseFieldIsingModel) -> np.ndarray:
    """H dense, construit exactement comme avant le passage au creux."""
    H = np.zeros((model.dim, model.dim), dtype=complex)
    for i in range(model.N):
        H += model.h * get_site_operator(Z, i, model.N)
    for i in range(model.N):
        Xi = get_site_operator(X, i, model.N)
        for j in range(i + 1, model.N):
            H += model.J_matrix[i, j] * (Xi @ get_site_operator(X, j, model.N))
    return H


def _model(N: int, seed: int = 0) -> TransverseFieldIsingModel:
    np.random.seed(seed)
    return TransverseFieldIsingModel({"N": N, "J": 1.0, "h": 0.5})


def _random_state(dim: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("op", [X, Y, Z])
@pytest.mark.parametrize("N", [1, 2, 4])
def test_sparse_site_operator_matches_dense(op: np.ndarray, N: int) -> None:
    """Chaque opérateur à un site est identique en creux et en dense."""
    for site in range(N):
        sparse = get_site_operator_sparse(op, site, N)
        assert sp.issparse(sparse) and sparse.format == "csr"
        np.testing.assert_array_equal(sparse.toarray(), get_site_operator(op, site, N))


@pytest.mark.parametrize("N", [1, 2, 3, 5, 6])
def test_hamiltonian_is_csr_and_matches_dense(N: int) -> None:
    """H est une matrice CSR, mise en cache, égale à la référence dense."""
    model = _model(N)
    H = model.hamiltonian()
    assert sp.issparse(H) and H.format == "csr"
    assert H.shape == (model.dim, model.dim)
    assert model._H is not None and model._H.format == "csr"
    np.testing.assert_allclose(H.toarray(), _dense_reference(model), atol=1e-14)
    np.testing.assert_array_equal(model._H.toarray(), H.toarray())


def test_hamiltonian_is_actually_sparse() -> None:
    """Au plus 2^N (1 + N(N-1)/2) éléments non nuls, bien moins que 4^N."""
    model = _model(8)
    H = model.hamiltonian()
    bound = model.dim * (1 + model.N * (model.N - 1) // 2)
    assert H.nnz <= bound < model.dim**2


@pytest.mark.parametrize("N", [2, 4, 6])
def test_propagator_matches_dense(N: int) -> None:
    """U(dt) depuis le H creux = expm(-i H_dense dt)."""
    model = _model(N)
    expected = scipy.linalg.expm(-1j * 0.7 * _dense_reference(model))
    np.testing.assert_allclose(model.propagator(dt=0.7), expected, atol=1e-12)


@pytest.mark.parametrize("dt", [0.0, 0.3, 2.5])
@pytest.mark.parametrize("N", [2, 4, 6])
def test_evolve_psi_matches_dense(N: int, dt: float) -> None:
    """expm_multiply sur H creux = expm(-i H_dense dt) @ psi."""
    model = _model(N)
    U = scipy.linalg.expm(-1j * dt * _dense_reference(model))
    for seed in (1, 2, 3, 4):
        psi = _random_state(model.dim, seed)
        np.testing.assert_allclose(model.evolve(dt=dt, state=psi), U @ psi, atol=1e-12)


@pytest.mark.parametrize("N", [2, 4, 6])
def test_evolve_rho_matches_dense(N: int) -> None:
    """U rho U^dagger sans former U = la version dense, état mixte inclus."""
    model = _model(N)
    U = scipy.linalg.expm(-1j * 1.1 * _dense_reference(model))
    p1, p2 = _random_state(model.dim, 1), _random_state(model.dim, 2)
    rho = 0.6 * np.outer(p1, p1.conj()) + 0.4 * np.outer(p2, p2.conj())
    np.testing.assert_allclose(
        model.evolve(dt=1.1, state=rho), U @ rho @ U.conj().T, atol=1e-12
    )
