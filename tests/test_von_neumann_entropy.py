"""Entropie de Von Neumann de l'ancilla R (issue #36)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.evaluation.metrics import von_neumann_entropy


def _rho(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


BELL = _rho([1, 0, 0, 1])


def test_bell_state_is_one_bit() -> None:
    assert von_neumann_entropy(BELL, 2) == pytest.approx(1.0, abs=1e-12)


def test_separable_pure_state_is_zero() -> None:
    plus = np.array([1, 1]) / np.sqrt(2)
    psi = np.kron(plus, np.kron([1, 0], [0, 1]))
    assert von_neumann_entropy(_rho(psi), 3) == pytest.approx(0.0, abs=1e-12)


def test_negative_rounding_noise_does_not_crash() -> None:
    """Valeurs propres poussées à ~ -1e-16 : pas de log d'un négatif."""
    rho = _rho([1, 0, 0, 0]).copy()
    rho[3, 3] = -1e-16
    with np.errstate(all="raise"):
        assert von_neumann_entropy(rho, 2) == pytest.approx(0.0, abs=1e-12)


def test_base_e_gives_ln2() -> None:
    assert von_neumann_entropy(BELL, 2, base=np.e) == pytest.approx(np.log(2))


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_initial_embedding_is_maximally_entangled(N: int) -> None:
    """L'état initial (|0>|Q1> + |1>|Q2>)/sqrt(2) vaut 1 bit, dense ou creux."""
    emb = AncillaChargeSectorEmbedding({"N": N})
    assert von_neumann_entropy(emb.density_matrix(), N) == pytest.approx(1.0)
    rho_sp = emb.density_matrix(sparse=True)
    assert von_neumann_entropy(rho_sp, N) == pytest.approx(1.0)


def test_mixed_ancilla_on_product_state() -> None:
    """rho = (I/2) x |0><0| : l'ancilla seule est mélangée, S = 1 bit."""
    rho = np.kron(np.eye(2) / 2, _rho([1, 0]))
    assert von_neumann_entropy(rho, 2) == pytest.approx(1.0)
    assert von_neumann_entropy(sp.csr_matrix(rho), 2) == pytest.approx(1.0)


def test_partial_entanglement_matches_formula() -> None:
    """cos t |00> + sin t |11> : S = -c² log2 c² - s² log2 s²."""
    t = 0.3
    c2, s2 = np.cos(t) ** 2, np.sin(t) ** 2
    rho = _rho([np.cos(t), 0, 0, np.sin(t)])
    expected = -c2 * np.log2(c2) - s2 * np.log2(s2)
    assert von_neumann_entropy(rho, 2) == pytest.approx(expected)


def test_single_qubit() -> None:
    assert von_neumann_entropy(np.eye(2) / 2, 1) == pytest.approx(1.0)
    assert von_neumann_entropy(_rho([1, 0]), 1) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("rho", "N", "base", "match"),
    [
        (BELL, 3, 2.0, "shape"),
        (BELL, 0, 2.0, "at least 1"),
        (BELL, 2, 1.0, "base"),
        (BELL, 2, -2.0, "base"),
    ],
)
def test_invalid_input(rho: np.ndarray, N: int, base: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        von_neumann_entropy(rho, N, base=base)
