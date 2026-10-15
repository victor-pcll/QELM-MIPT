"""Assertions on sparse matrices that never build a dense copy."""

from __future__ import annotations

from typing import Any

import scipy.sparse as sp


def assert_sparse_hermitian(H: Any, atol: float = 1e-8) -> None:
    """Assert that a sparse matrix is Hermitian, without ``toarray()``.

    ``H - H^dagger`` stays sparse, so the cost is ``O(nnz)`` in time and
    memory instead of ``O(dim**2)`` for a dense comparison.

    Args:
        H: Sparse matrix (any SciPy sparse format).
        atol: Largest allowed ``|H_ij - conj(H_ji)|``.

    Raises:
        AssertionError: If ``H`` is not sparse, not square, or not Hermitian
            within ``atol``. The message gives the largest error and the shape.
    """
    assert sp.issparse(H), f"Expected a sparse matrix, got {type(H).__name__}."
    assert H.shape[0] == H.shape[1], f"Expected a square matrix, got {H.shape}."
    diff = sp.csr_matrix(H - H.conj().T)
    max_err = float(abs(diff).max()) if diff.nnz > 0 else 0.0
    assert max_err <= atol, (
        f"Matrix is not Hermitian. Max absolute error: {max_err:.2e}, shape: {H.shape}."
    )
