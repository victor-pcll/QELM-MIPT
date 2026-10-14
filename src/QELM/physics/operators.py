"""Single-qubit Pauli matrices and their embedding in an N-qubit chain."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

I = np.eye(2, dtype=complex)  # noqa: E741 (standard name for the identity)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def get_site_operator(op: np.ndarray, site: int, N: int) -> np.ndarray:
    """Construit l'opérateur à 1 qubit ``op`` agissant sur le qubit ``site``.

    Les autres qubits de la chaîne à N qubits portent l'identité.

    Args:
        op: Matrice 2x2 (ex: X, Y, Z).
        site: Indice du qubit cible (0 à N-1).
        N: Nombre total de qubits dans le système.

    Returns:
        Matrice de taille (2^N, 2^N).
    """
    ops = [I] * N
    ops[site] = op
    res = ops[0]
    for o in ops[1:]:
        res = np.kron(res, o)
    return res


def get_site_operator_sparse(op: np.ndarray, site: int, N: int) -> sp.csr_matrix:
    """Version creuse (CSR) de :func:`get_site_operator`.

    Un produit de Kronecker de matrices de Pauli n'a qu'un élément non nul
    par ligne, soit ``2**N`` éléments au lieu de ``4**N`` : le stockage
    creux évite de construire la matrice dense.

    Args:
        op: Matrice 2x2 (ex: X, Y, Z).
        site: Indice du qubit cible (0 à N-1).
        N: Nombre total de qubits dans le système.

    Returns:
        Matrice creuse CSR de taille (2^N, 2^N), égale à
        ``get_site_operator(op, site, N)``.
    """
    left = sp.identity(2**site, dtype=complex, format="csr")
    right = sp.identity(2 ** (N - site - 1), dtype=complex, format="csr")
    return sp.csr_matrix(sp.kron(sp.kron(left, sp.csr_matrix(op)), right, "csr"))
