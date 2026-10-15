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
    op: np.ndarray, site: int, N: int, sparse: Literal[False] = False
) -> np.ndarray: ...


@overload
def get_site_operator(
    op: np.ndarray, site: int, N: int, sparse: Literal[True]
) -> sp.csr_matrix: ...


def get_site_operator(
    op: np.ndarray, site: int, N: int, sparse: bool = False
) -> np.ndarray | sp.csr_matrix:
    """Construit l'opérateur à 1 qubit ``op`` agissant sur le qubit ``site``.

    Les autres qubits de la chaîne à N qubits portent l'identité :
    ``I_(2**site) ⊗ op ⊗ I_(2**(N - site - 1))``, le site 0 étant le facteur
    le plus à gauche.

    Par défaut la matrice est un tableau numpy dense. ``sparse=True`` la
    renvoie creuse (CSR) : un produit de Kronecker de matrices de Pauli n'a
    qu'un élément non nul par ligne, soit ``2**N`` éléments stockés au lieu
    de ``4**N``, ce qui évite de saturer la mémoire pour les grands ``N``.

    Args:
        op: Matrice 2x2 (ex: X, Y, Z).
        site: Indice du qubit cible (0 à N-1).
        N: Nombre total de qubits dans le système.
        sparse: ``True`` pour une matrice creuse CSR, ``False`` (défaut)
            pour un ``np.ndarray`` dense.

    Returns:
        Matrice de taille (2^N, 2^N), CSR ou dense selon ``sparse``.

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
    return out if sparse else out.toarray()


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


def n_qubits(rho: np.ndarray, n_ancilla: int = 0) -> tuple[int, int, int]:
    """Return the qubit counts of a density matrix of shape ``(2**N, 2**N)``.

    Ancillas are the first sites, as in :mod:`QELM.embedding`, so the
    system is the last ``N - n_ancilla`` qubits.

    Args:
        rho: Density matrix.
        n_ancilla: Number of ancilla qubits included in ``rho``.

    Returns:
        ``(N, N_S, n_ancilla)``: total, system and ancilla qubit counts, with
        ``N = N_S + n_ancilla``.

    Raises:
        ValueError: If ``rho`` is not square with a power-of-two side, if
            ``n_ancilla`` is negative, or if ``rho`` has fewer qubits than
            ``n_ancilla``.
    """
    dim = rho.shape[0]
    N = dim.bit_length() - 1
    if rho.shape != (dim, dim) or dim < 2 or 2**N != dim:
        raise ValueError(f"rho must have shape (2**N, 2**N), got {rho.shape}.")
    if n_ancilla < 0:
        raise ValueError(f"n_ancilla must be >= 0, got {n_ancilla}.")
    N_S = N - n_ancilla
    if N_S < 0:
        raise ValueError(
            f"Total qubits ({N}) cannot be less than ancilla qubits ({n_ancilla})."
        )
    return N, N_S, n_ancilla
