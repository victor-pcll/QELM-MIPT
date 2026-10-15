"""Tests de n_qubits avec ancillas (issue #26)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.physics import n_qubits


@pytest.mark.parametrize("N", [1, 2, 5])
def test_default_has_no_ancilla(N: int) -> None:
    assert n_qubits(np.eye(2**N)) == (N, N, 0)


@pytest.mark.parametrize(("N", "n_ancilla"), [(2, 1), (5, 1), (5, 2), (3, 3)])
def test_counts_add_up(N: int, n_ancilla: int) -> None:
    total, system, ancilla = n_qubits(np.eye(2**N), n_ancilla)
    assert (total, system, ancilla) == (N, N - n_ancilla, n_ancilla)
    assert total == system + ancilla


def test_matches_the_ancilla_embedding() -> None:
    """Sur l'état de l'embedding, (N, N_S, 1) comme emb.N, emb.N_S."""
    emb = AncillaChargeSectorEmbedding({"N": 5})
    rho = emb.density_matrix()
    assert n_qubits(rho, emb.n_ancilla) == (emb.N, emb.N_S, emb.n_ancilla)


def test_more_ancillas_than_qubits_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be less than ancilla"):
        n_qubits(np.eye(4), n_ancilla=3)
    with pytest.raises(ValueError, match="n_ancilla must be"):
        n_qubits(np.eye(4), n_ancilla=-1)


@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (1, 1)])
def test_bad_shape_is_rejected(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="2\\*\\*N"):
        n_qubits(np.zeros(shape))
