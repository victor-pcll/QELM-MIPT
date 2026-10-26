"""Tests for Embedding.rho, the density matrix of the initial state."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from QELM.encoding import AncillaChargeSectorEmbedding
from sparse_checks import assert_sparse_hermitian


@pytest.fixture(params=[3, 4, 5, 7])
def embedding(request: pytest.FixtureRequest) -> AncillaChargeSectorEmbedding:
    return AncillaChargeSectorEmbedding({"N": request.param})


def test_dense_rho_shape_trace_and_hermiticity(
    embedding: AncillaChargeSectorEmbedding,
) -> None:
    """rho dense : forme (2**N, 2**N), trace 1, hermitienne, pure."""
    dim = 2**embedding.N
    rho = embedding.rho()
    assert isinstance(rho, np.ndarray)
    assert rho.shape == (dim, dim)
    assert np.trace(rho) == pytest.approx(1.0)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-14)


def test_sparse_rho_shape_trace_and_hermiticity(
    embedding: AncillaChargeSectorEmbedding,
) -> None:
    """rho creux : CSR, trace 1, hermitienne sans copie dense."""
    dim = 2**embedding.N
    rho = embedding.rho(sparse=True)
    assert sp.issparse(rho) and rho.format == "csr"
    assert rho.shape == (dim, dim)
    assert rho.diagonal().sum() == pytest.approx(1.0)
    assert_sparse_hermitian(rho, atol=1e-14)


def test_sparse_and_dense_rho_agree(embedding: AncillaChargeSectorEmbedding) -> None:
    """rho(sparse=True).toarray() == rho(sparse=False)."""
    np.testing.assert_allclose(
        embedding.rho(sparse=True).toarray(), embedding.rho(), atol=1e-15
    )


def test_rho_matches_state_and_density_matrix(
    embedding: AncillaChargeSectorEmbedding,
) -> None:
    """rho = |psi><psi| = density_matrix()."""
    psi = embedding.state()
    np.testing.assert_allclose(embedding.rho(), np.outer(psi, psi.conj()))
    np.testing.assert_array_equal(embedding.rho(), embedding.density_matrix())
