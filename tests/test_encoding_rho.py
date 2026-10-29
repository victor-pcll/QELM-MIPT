"""Tests for Encoding.rho, the density matrix of the initial state."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from QELM.encoding import AncillaChargeSectorEncoding
from sparse_checks import assert_sparse_hermitian


@pytest.fixture(params=[3, 4, 5, 7])
def encoding(request: pytest.FixtureRequest) -> AncillaChargeSectorEncoding:
    return AncillaChargeSectorEncoding({"N": request.param})


def test_dense_rho_shape_trace_and_hermiticity(
    encoding: AncillaChargeSectorEncoding,
) -> None:
    """rho dense : forme (2**N_total, 2**N_total), trace 1, hermitienne, pure."""
    dim = encoding.dim
    rho = encoding.rho()
    assert isinstance(rho, np.ndarray)
    assert rho.shape == (dim, dim)
    assert np.trace(rho) == pytest.approx(1.0)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-14)


def test_sparse_rho_shape_trace_and_hermiticity(
    encoding: AncillaChargeSectorEncoding,
) -> None:
    """rho creux : CSR, trace 1, hermitienne sans copie dense."""
    dim = encoding.dim
    rho = encoding.rho(sparse=True)
    assert sp.issparse(rho) and rho.format == "csr"
    assert rho.shape == (dim, dim)
    assert rho.diagonal().sum() == pytest.approx(1.0)
    assert_sparse_hermitian(rho, atol=1e-14)


def test_sparse_and_dense_rho_agree(encoding: AncillaChargeSectorEncoding) -> None:
    """rho(sparse=True).toarray() == rho(sparse=False)."""
    np.testing.assert_allclose(
        encoding.rho(sparse=True).toarray(), encoding.rho(), atol=1e-15
    )


def test_rho_matches_state_and_density_matrix(
    encoding: AncillaChargeSectorEncoding,
) -> None:
    """rho = |psi><psi| = density_matrix()."""
    psi = encoding.state()
    np.testing.assert_allclose(encoding.rho(), np.outer(psi, psi.conj()))
    np.testing.assert_array_equal(encoding.rho(), encoding.density_matrix())
