"""Tests de qubit_counts avec ancillas (issue #26)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.encoding import AncillaChargeSectorEmbedding
from QELM.physics import qubit_counts


@pytest.mark.parametrize("N", [1, 2, 5])
def test_default_has_no_ancilla(N: int) -> None:
    assert qubit_counts(np.eye(2**N)) == (N, N, 0)


@pytest.mark.parametrize(("N", "Nr"), [(2, 1), (5, 1), (5, 2), (3, 3)])
def test_counts_add_up(N: int, Nr: int) -> None:
    total, system, ancilla = qubit_counts(np.eye(2**N), Nr)
    assert (total, system, ancilla) == (N, N - Nr, Nr)
    assert total == system + ancilla


def test_matches_the_ancilla_embedding() -> None:
    """Sur l'état de l'embedding, (N, Ns, 1) comme emb.N, emb.Ns."""
    emb = AncillaChargeSectorEmbedding({"N": 5})
    rho = emb.density_matrix()
    assert qubit_counts(rho, emb.Nr) == (emb.N, emb.Ns, emb.Nr)


def test_more_ancillas_than_qubits_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be less than ancilla"):
        qubit_counts(np.eye(4), Nr=3)
    with pytest.raises(ValueError, match="Nr must be"):
        qubit_counts(np.eye(4), Nr=-1)


@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (1, 1)])
def test_bad_shape_is_rejected(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="2\\*\\*N"):
        qubit_counts(np.zeros(shape))
