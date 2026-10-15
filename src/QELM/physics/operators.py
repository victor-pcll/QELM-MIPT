"""Single-qubit Pauli matrices and their embedding in an N-qubit chain."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import scipy.sparse as sp

I = np.eye(2, dtype=complex)  # noqa: E741 (standard name for the identity)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


@overload
def get_site_operator(
    op: np.ndarray, site: int, N: int, dense: Literal[False] = False
) -> sp.csr_matrix: ...


@overload
def get_site_operator(
    op: np.ndarray, site: int, N: int, dense: Literal[True]
) -> np.ndarray: ...


def get_site_operator(
    op: np.ndarray, site: int, N: int, dense: bool = False
) -> sp.csr_matrix | np.ndarray:
    """Construit l'opérateur à 1 qubit ``op`` agissant sur le qubit ``site``.

    Les autres qubits de la chaîne à N qubits portent l'identité :
    ``I_(2**site) ⊗ op ⊗ I_(2**(N - site - 1))``, le site 0 étant le facteur
    le plus à gauche.

    Par défaut la matrice est creuse (CSR). Un produit de Kronecker de
    matrices de Pauli n'a qu'un élément non nul par ligne, soit ``2**N``
    éléments stockés au lieu de ``4**N`` : cela évite de saturer la mémoire
    pour les grands ``N``. ``dense=True`` renvoie le tableau numpy.

    Args:
        op: Matrice 2x2 (ex: X, Y, Z).
        site: Indice du qubit cible (0 à N-1).
        N: Nombre total de qubits dans le système.
        dense: ``True`` pour un ``np.ndarray`` dense, ``False`` (défaut) pour
            une matrice creuse CSR.

    Returns:
        Matrice de taille (2^N, 2^N), CSR ou dense selon ``dense``.

    Raises:
        ValueError: Si ``op`` n'est pas 2x2 ou si ``site`` n'est pas dans
            ``[0, N - 1]``.
    """
    if np.shape(op) != (2, 2):
        raise ValueError(f"op must have shape (2, 2), got {np.shape(op)}.")
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    left = sp.identity(2**site, dtype=complex, format="csr")
    right = sp.identity(2 ** (N - site - 1), dtype=complex, format="csr")
    out = sp.csr_matrix(sp.kron(sp.kron(left, sp.csr_matrix(op)), right, "csr"))
    # sp.kron passe par des blocs 2x2 (BSR) et garde les zéros de ``op`` :
    # sans ce nettoyage, X stockerait 2 * 2**N éléments au lieu de 2**N.
    out.eliminate_zeros()
    return out.toarray() if dense else out


PAULIS: dict[str, np.ndarray] = {"X": X, "Y": Y, "Z": Z}


def pauli(label: str) -> np.ndarray:
    """Return the 2x2 Pauli matrix named ``label``.

    Args:
        label: ``"X"``, ``"Y"`` or ``"Z"``.

    Returns:
        Complex matrix of shape ``(2, 2)``.

    Raises:
        ValueError: If ``label`` is not a Pauli label.
    """
    try:
        return PAULIS[label]
    except KeyError as exc:
        known = ", ".join(PAULIS)
        raise ValueError(f"Unknown observable {label!r}. Known: {known}.") from exc


def n_qubits(rho: np.ndarray) -> int:
    """Return ``N`` for a density matrix of shape ``(2**N, 2**N)``.

    Args:
        rho: Density matrix.

    Returns:
        Number of qubits.

    Raises:
        ValueError: If ``rho`` is not square with a power-of-two side.
    """
    dim = rho.shape[0]
    N = dim.bit_length() - 1
    if rho.shape != (dim, dim) or dim < 2 or 2**N != dim:
        raise ValueError(f"rho must have shape (2**N, 2**N), got {rho.shape}.")
    return N
