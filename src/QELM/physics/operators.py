"""Single-qubit Pauli matrices and their embedding in an N-qubit chain.

The helpers of :mod:`QELM.physics` take ``N`` as the number of qubits of
the operator or state they act on, whatever its role. In the experiments,
``N`` is the number of system qubits and the simulated state has
``N_total = N + Nr`` qubits, ancillas included (issue #235); see
:func:`qubit_counts`.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, overload

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


class QubitCounts(NamedTuple):
    """Qubit counts of a state with ancillas (issue #235).

    Attributes:
        N: Number of system qubits, the size of the studied system.
        Nr: Number of ancilla (reference) qubits, the first sites.
        N_total: Number of qubits of the simulated state, ``N + Nr``.
    """

    N: int
    Nr: int
    N_total: int


def qubit_counts(rho: np.ndarray, Nr: int = 0) -> QubitCounts:
    """Return the qubit counts of a density matrix with ``Nr`` ancillas.

    Ancillas are the first sites, as in :mod:`QELM.encoding`, so the
    system is the last ``N = N_total - Nr`` qubits of a matrix of shape
    ``(2**N_total, 2**N_total)``.

    Args:
        rho: Density matrix.
        Nr: Number of ancilla qubits included in ``rho``.

    Returns:
        ``(N, Nr, N_total)``: system, ancilla and total qubit counts, with
        ``N_total = N + Nr``.

    Raises:
        ValueError: If ``rho`` is not square with a power-of-two side, if
            ``Nr`` is negative, or if ``rho`` has fewer qubits than
            ``Nr``.
    """
    dim = rho.shape[0]
    N_total = dim.bit_length() - 1
    if rho.shape != (dim, dim) or dim < 2 or 2**N_total != dim:
        raise ValueError(
            f"rho must have shape (2**N_total, 2**N_total), got {rho.shape}."
        )
    if Nr < 0:
        raise ValueError(f"Nr must be >= 0, got {Nr}.")
    N = N_total - Nr
    if N < 0:
        raise ValueError(
            f"Total qubits ({N_total}) cannot be less than ancilla qubits ({Nr})."
        )
    return QubitCounts(N, Nr, N_total)


def local_expectation(state: np.ndarray, op: np.ndarray, site: int) -> float:
    """Return ``<op_site>`` for a one-qubit operator, without building it on N qubits.

    Equal to ``np.real(np.trace(get_site_operator(op, site, N) @ rho))`` for a
    density matrix and to ``np.real(psi.conj() @ get_site_operator(...) @ psi)``
    for a state vector, but the ``2**N x 2**N`` product is never formed. The
    state is reshaped so that the qubit ``site`` has its own axis, the other
    qubits are traced out, and ``op`` acts on the remaining ``2 x 2`` reduced
    density matrix: ``O(2**N)`` operations for ``rho`` instead of ``O(8**N)``
    for the dense product (issue #95).

    Args:
        state: Density matrix of shape ``(2**N, 2**N)`` or state vector of
            shape ``(2**N,)``, site 0 being the leftmost Kronecker factor as
            in :func:`get_site_operator`.
        op: Hermitian ``2 x 2`` operator (e.g. ``Z``).
        site: Qubit it acts on, from 0 to ``N - 1``.

    Returns:
        The real part of the expectation value (the imaginary part vanishes
        for a Hermitian ``op``).

    Raises:
        ValueError: If ``op`` is not ``2 x 2``, ``state`` is neither a qubit
            state vector nor a qubit density matrix, or ``site`` is out of
            range.
    """
    if np.shape(op) != (2, 2):
        raise ValueError(f"op must have shape (2, 2), got {np.shape(op)}.")
    state = np.asarray(state)
    dim = state.shape[0] if state.ndim in (1, 2) else 0
    N = dim.bit_length() - 1
    if dim < 2 or 2**N != dim or state.shape not in ((dim,), (dim, dim)):
        raise ValueError(
            f"state must have shape (2**N,) or (2**N, 2**N), got {state.shape}."
        )
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    left, right = 2**site, 2 ** (N - site - 1)
    if state.ndim == 1:
        psi = state.reshape(left, 2, right)
        reduced = np.einsum("aib,ajb->ij", psi, psi.conj())
    else:
        rho = state.reshape(left, 2, right, left, 2, right)
        reduced = np.einsum("aibajb->ij", rho)
    return float(np.real(np.einsum("ij,ji->", op, reduced)))
