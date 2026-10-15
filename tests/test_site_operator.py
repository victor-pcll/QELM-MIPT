"""get_site_operator : dense par défaut, creuse avec sparse=True (#27, #48)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import QELM.physics.operators as ops


def _kron_site(op: np.ndarray, site: int, N: int) -> np.ndarray:
    """Référence indépendante : produit de Kronecker dense, site 0 à gauche."""
    out = np.eye(1, dtype=complex)
    for k in range(N):
        out = np.kron(out, op if k == site else ops.I)
    return out


@pytest.mark.parametrize("N", [2, 3, 4, 6])
@pytest.mark.parametrize("site", [0, 1, -1])  # -1 : dernier site
@pytest.mark.parametrize("op_name", ["X", "Y", "Z"])
def test_dense_and_sparse_are_equal(N: int, site: int, op_name: str) -> None:
    op = ops.PAULIS[op_name]
    target = N - 1 if site == -1 else site
    dense = ops.get_site_operator(op, target, N, sparse=False)
    sparse = ops.get_site_operator(op, target, N, sparse=True)
    assert isinstance(dense, np.ndarray) and dense.shape == (2**N, 2**N)
    assert sp.issparse(sparse) and sparse.format == "csr"
    np.testing.assert_array_equal(
        dense,
        sparse.toarray(),
        err_msg=f"Dense/sparse mismatch for {op_name} at site {target}, N={N}",
    )
    np.testing.assert_array_equal(dense, _kron_site(op, target, N))


def test_dense_is_the_default() -> None:
    out = ops.get_site_operator(ops.Z, 0, 3)
    assert isinstance(out, np.ndarray) and not sp.issparse(out)


def test_sparse_stores_one_entry_per_row() -> None:
    """Un produit de Pauli a 2**N éléments non nuls, pas 4**N."""
    N = 10
    assert ops.get_site_operator(ops.X, 4, N, sparse=True).nnz == 2**N


def test_middle_site_on_a_basis_state() -> None:
    """X sur le site 1 de |000> donne |010> (site 0 = bit de poids fort)."""
    psi = np.zeros(8)
    psi[0b000] = 1
    out = ops.get_site_operator(ops.X, 1, 3) @ psi
    assert out[0b010] == 1 and np.count_nonzero(out) == 1


@pytest.mark.parametrize(("site", "N"), [(-1, 3), (3, 3), (0, 0)])
def test_site_out_of_range(site: int, N: int) -> None:
    with pytest.raises(ValueError, match="site must be"):
        ops.get_site_operator(ops.X, site, N)


def test_op_must_be_two_by_two() -> None:
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        ops.get_site_operator(np.eye(4), 0, 2)
