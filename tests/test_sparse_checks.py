"""Tests for the sparse Hermitian assertion helper."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from QELM.encoding import AncillaChargeSectorEmbedding
from QELM.models.TFIM import TransverseFieldIsingModel
from sparse_checks import assert_sparse_hermitian


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_accepts_a_hermitian_matrix(fmt: str) -> None:
    """Une matrice hermitienne complexe passe, quel que soit le format."""
    A = sp.random(50, 50, density=0.1, format=fmt, random_state=0)
    B = sp.random(50, 50, density=0.1, format=fmt, random_state=1)
    M = A + 1j * B
    assert_sparse_hermitian((M + M.conj().T).asformat(fmt))


def test_accepts_the_zero_matrix() -> None:
    """Sans élément non nul, l'erreur vaut 0."""
    assert_sparse_hermitian(sp.csr_matrix((4, 4)))


def test_rejects_a_non_hermitian_matrix() -> None:
    """Une matrice symétrique mais complexe n'est pas hermitienne."""
    H = sp.csr_matrix(np.array([[0.0, 1j], [1j, 0.0]]))
    with pytest.raises(AssertionError, match=r"not Hermitian.*2\.00e\+00.*\(2, 2\)"):
        assert_sparse_hermitian(H)


def test_atol_bounds_the_error() -> None:
    """Un écart sous atol passe, au-dessus il échoue."""
    H = sp.csr_matrix(np.array([[0.0, 1.0], [1.0 + 1e-10, 0.0]]))
    assert_sparse_hermitian(H)
    with pytest.raises(AssertionError, match="not Hermitian"):
        assert_sparse_hermitian(H, atol=1e-12)


def test_rejects_dense_and_non_square_input() -> None:
    """Un tableau dense ou une matrice rectangulaire sont refusés."""
    with pytest.raises(AssertionError, match="Expected a sparse matrix"):
        assert_sparse_hermitian(np.eye(2))
    with pytest.raises(AssertionError, match="Expected a square matrix"):
        assert_sparse_hermitian(sp.csr_matrix((2, 3)))


def test_large_matrix_without_dense_copy() -> None:
    """dim = 2**16 : une copie dense prendrait 64 Go en complex128."""
    dim = 2**16
    rng = np.random.default_rng(2)
    rows, cols = rng.integers(0, dim, size=(2, 50_000))
    values = rng.normal(size=50_000) + 1j * rng.normal(size=50_000)
    A = sp.csr_matrix((values, (rows, cols)), shape=(dim, dim))
    assert_sparse_hermitian(A + A.conj().T)
    with pytest.raises(AssertionError, match="not Hermitian"):
        assert_sparse_hermitian(A + A.T)


def test_tfim_hamiltonian_is_hermitian() -> None:
    """Le H TFIM creux est hermitien."""
    model = TransverseFieldIsingModel({"N": 6, "J": 1.0, "h": 0.5})
    assert_sparse_hermitian(model.hamiltonian(), atol=1e-12)


def test_sparse_embedding_density_matrix_is_hermitian() -> None:
    """rho creux de l'embedding est hermitien."""
    embedding = AncillaChargeSectorEmbedding({"N": 5})
    assert_sparse_hermitian(embedding.density_matrix(sparse=True), atol=1e-12)
