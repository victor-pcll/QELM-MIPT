"""Tests de qubit_counts avec ancillas (issues #26 et #235)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.encoding import AncillaChargeSectorEncoding
from QELM.physics import qubit_counts


@pytest.mark.parametrize("N", [1, 2, 5])
def test_default_has_no_ancilla(N: int) -> None:
    assert qubit_counts(np.eye(2**N)) == (N, 0, N)


@pytest.mark.parametrize(("N_total", "Nr"), [(2, 1), (5, 1), (5, 2), (3, 3)])
def test_counts_add_up(N_total: int, Nr: int) -> None:
    counts = qubit_counts(np.eye(2**N_total), Nr)
    assert counts == (N_total - Nr, Nr, N_total)
    assert (counts.N, counts.Nr, counts.N_total) == tuple(counts)
    assert counts.N_total == counts.N + counts.Nr


def test_matches_the_ancilla_encoding() -> None:
    """Sur l'état de l'encoding, (N, 1, N + 1) comme emb.N, emb.N_total."""
    emb = AncillaChargeSectorEncoding({"N": 5})
    rho = emb.density_matrix()
    assert qubit_counts(rho, emb.Nr) == (emb.N, emb.Nr, emb.N_total) == (5, 1, 6)


def test_more_ancillas_than_qubits_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be less than ancilla"):
        qubit_counts(np.eye(4), Nr=3)
    with pytest.raises(ValueError, match="Nr must be"):
        qubit_counts(np.eye(4), Nr=-1)


@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (1, 1)])
def test_bad_shape_is_rejected(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="2\\*\\*N_total"):
        qubit_counts(np.zeros(shape))
