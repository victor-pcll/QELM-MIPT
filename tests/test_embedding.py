"""Tests des embeddings d'état et de la classe Embedding commune."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
import scipy.sparse as sp
from omegaconf import OmegaConf
from scipy.sparse.linalg import norm as sparse_norm

from QELM.embedding import (
    AncillaChargeSectorEmbedding,
    Embedding,
    charge_sector_indices,
    charge_sector_state,
    get_embedding,
)
from QELM.models import TransverseFieldIsingModel

NS = [2, 3, 4, 5, 6]


# N total : 1 ancilla + N_S = N - 1 qubits de système.
NS_TOTAL = [3, 4, 5, 6, 7]


def _rho_R(rho: np.ndarray, N_S: int) -> np.ndarray:
    """Trace partielle sur S : état réduit de l'ancilla (site 0)."""
    return np.einsum("aibi->ab", rho.reshape(2, 2**N_S, 2, 2**N_S))


def _rho_S(rho: np.ndarray, N_S: int) -> np.ndarray:
    """Trace partielle sur R : état réduit du système."""
    return np.einsum("aiaj->ij", rho.reshape(2, 2**N_S, 2, 2**N_S))


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("Q", range(7))
def test_charge_sector_indices(N: int, Q: int) -> None:
    if Q > N:
        with pytest.raises(ValueError):
            charge_sector_indices(N, Q)
        return
    idx = charge_sector_indices(N, Q)
    assert idx.size == comb(N, Q)
    assert all(bin(int(i)).count("1") == Q for i in idx)


def test_charge_sector_negative() -> None:
    with pytest.raises(ValueError):
        charge_sector_indices(3, -1)


@pytest.mark.parametrize("N", NS)
def test_charge_sector_state_normalized(N: int) -> None:
    assert np.isclose(np.linalg.norm(charge_sector_state(N, N // 2)), 1.0)


@pytest.mark.parametrize("N", NS_TOTAL)
def test_ancilla_state_physical(N: int) -> None:
    emb = AncillaChargeSectorEmbedding({"N": N})
    psi = emb.state()
    rho = emb.density_matrix()
    assert emb.N_S == N - 1
    assert emb.dim == 2**N
    assert psi.shape == (2**N,)
    assert rho.shape == (2**N, 2**N)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho @ rho).real, 1.0)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-12)


@pytest.mark.parametrize("N", NS_TOTAL)
def test_ancilla_components_in_right_sector(N: int) -> None:
    N_S = N - 1
    psi = AncillaChargeSectorEmbedding({"N": N}).state()
    nonzero = np.flatnonzero(np.abs(psi) > 1e-12)
    for i in nonzero:
        ancilla, system = divmod(int(i), 2**N_S)
        expected = N_S // 2 if ancilla == 0 else N_S // 2 - 1
        assert bin(system).count("1") == expected
    # Chaque branche porte exactement le poids 1/2.
    assert np.isclose(np.sum(np.abs(psi[: 2**N_S]) ** 2), 0.5)
    assert np.isclose(np.sum(np.abs(psi[2**N_S :]) ** 2), 0.5)
    assert nonzero.size == comb(N_S, N_S // 2) + comb(N_S, N_S // 2 - 1)


@pytest.mark.parametrize("N", NS_TOTAL)
def test_ancilla_reduced_states(N: int) -> None:
    N_S = N - 1
    rho = AncillaChargeSectorEmbedding({"N": N}).density_matrix()
    assert np.allclose(_rho_R(rho, N_S), np.eye(2) / 2)
    q1 = charge_sector_state(N_S, N_S // 2)
    q2 = charge_sector_state(N_S, N_S // 2 - 1)
    expected_S = (np.outer(q1, q1.conj()) + np.outer(q2, q2.conj())) / 2
    assert np.allclose(_rho_S(rho, N_S), expected_S)


def test_ancilla_known_values_N3() -> None:
    # N = 3 : ancilla + 2 qubits de système.
    # (|0>(|01> + |10>)/sqrt(2) + |1>|00>) / sqrt(2)
    psi = AncillaChargeSectorEmbedding({"N": 3}).state()
    expected = np.zeros(8, dtype=complex)
    expected[[0b001, 0b010]] = 0.5
    expected[0b100] = 1 / np.sqrt(2)
    assert np.allclose(psi, expected)


def test_ancilla_odd_system_charges() -> None:
    # N = 6 : N_S = 5, charges 2 et 1.
    emb = AncillaChargeSectorEmbedding({"N": 6})
    assert (emb.N_S, emb.Q1, emb.Q2) == (5, 2, 1)


@pytest.mark.parametrize("N", [0, 1, 2])
def test_ancilla_rejects_small_N(N: int) -> None:
    with pytest.raises(ValueError):
        AncillaChargeSectorEmbedding({"N": N})


def test_ancilla_deterministic() -> None:
    emb = AncillaChargeSectorEmbedding({"N": 4})
    assert np.array_equal(emb.state(), emb.state())


def test_from_config_and_registry() -> None:
    cfg = OmegaConf.create({"ancilla_charge_sector": {"N": 4}})
    cls = get_embedding("ancilla_charge_sector")
    assert cls is AncillaChargeSectorEmbedding
    emb = cls.from_config(cfg)
    assert isinstance(emb, Embedding)
    assert (emb.N, emb.N_S, emb.dim) == (4, 3, 16)


def test_from_config_rejects_scalar() -> None:
    cfg = OmegaConf.create({"ancilla_charge_sector": 3})
    with pytest.raises(TypeError):
        AncillaChargeSectorEmbedding.from_config(cfg)


def test_unknown_embedding() -> None:
    with pytest.raises(KeyError, match="Known"):
        get_embedding("nope")


def test_base_without_ancilla() -> None:
    emb = Embedding({"N": 2})
    assert (emb.N, emb.N_S, emb.dim) == (2, 2, 4)
    with pytest.raises(NotImplementedError):
        emb.state()


def test_same_dimension_as_model() -> None:
    # Le même N dans la config donne la même dimension que le modèle.
    model = TransverseFieldIsingModel({"N": 4, "J": 1.0, "h": 0.5})
    emb = AncillaChargeSectorEmbedding({"N": model.N})
    assert emb.dim == model.dim
    assert model.hamiltonian().shape == emb.density_matrix().shape


# --- Config composée (issue #22) ---


def test_from_full_config_follows_the_model() -> None:
    """La config par défaut contient le bloc, et N suit le modèle choisi.

    Le N par défaut est lu dans la config plutôt que codé en dur, pour que
    le test reste vrai si on change ``TFIM.N`` dans ``conf/`` (issue #54).
    """
    from QELM import load_config

    default_N = int(load_config().TFIM.N)
    for overrides, N in (
        ([], default_N),
        (["TFIM.N=5"], 5),
        (["sim_type=kitaev"], 4),
    ):
        cfg = load_config(overrides)
        emb = AncillaChargeSectorEmbedding.from_config(cfg)
        assert emb.N == N
        assert emb.state().shape == (2**N,)


def test_from_config_missing_block_is_explicit() -> None:
    """Sans bloc, l'erreur dit quoi ajouter au lieu d'un ConfigKeyError."""
    cfg = OmegaConf.create({"seed": 0})
    OmegaConf.set_struct(cfg, True)
    with pytest.raises(ValueError, match="no 'ancilla_charge_sector' block"):
        AncillaChargeSectorEmbedding.from_config(cfg)


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_sparse_state_matches_dense(N: int) -> None:
    """state(sparse=True) est une colonne CSR (2**N, 1) égale à state()."""
    emb = AncillaChargeSectorEmbedding({"N": N})
    psi = emb.state()
    psi_sp = emb.state(sparse=True)
    assert isinstance(psi_sp, sp.csr_matrix)
    assert psi.shape == (2**N,)
    assert psi_sp.shape == (2**N, 1)
    assert psi_sp.dtype == complex
    np.testing.assert_allclose(psi_sp.toarray().ravel(), psi, atol=1e-15)
    # Seules les amplitudes non nulles sont stockées.
    assert psi_sp.nnz == np.count_nonzero(psi)


@pytest.mark.parametrize(("N", "Q"), [(3, 0), (4, 2), (5, 3)])
def test_sparse_charge_sector_state_matches_dense(N: int, Q: int) -> None:
    """charge_sector_state(sparse=True) donne le même vecteur en CSR."""
    psi_sp = charge_sector_state(N, Q, sparse=True)
    assert isinstance(psi_sp, sp.csr_matrix)
    assert psi_sp.shape == (2**N, 1)
    np.testing.assert_allclose(
        psi_sp.toarray().ravel(), charge_sector_state(N, Q), atol=1e-15
    )
    assert np.isclose(sparse_norm(psi_sp), 1.0)


@pytest.mark.parametrize("N", [3, 4, 5])
def test_sparse_density_matrix_matches_dense(N: int) -> None:
    """density_matrix(sparse=True) est la CSR de density_matrix()."""
    emb = AncillaChargeSectorEmbedding({"N": N})
    rho = emb.density_matrix()
    rho_sp = emb.density_matrix(sparse=True)
    assert isinstance(rho, np.ndarray) and rho.shape == (2**N, 2**N)
    assert isinstance(rho_sp, sp.csr_matrix) and rho_sp.shape == (2**N, 2**N)
    np.testing.assert_allclose(rho_sp.toarray(), rho, atol=1e-15)
    assert rho_sp.nnz == np.count_nonzero(rho)


def test_dense_is_the_default_state() -> None:
    """Sans argument, state() et density_matrix() sont des tableaux numpy."""
    emb = AncillaChargeSectorEmbedding({"N": 3})
    assert isinstance(emb.state(), np.ndarray)
    assert isinstance(emb.density_matrix(), np.ndarray)
    assert isinstance(charge_sector_state(3, 1), np.ndarray)
