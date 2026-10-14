"""Tests des embeddings d'état et de la classe Embedding commune."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.embedding import (
    AncillaChargeSectorEmbedding,
    Embedding,
    charge_sector_indices,
    charge_sector_state,
    get_embedding,
)

NS = [2, 3, 4, 5, 6]


def _rho_R(rho: np.ndarray, N: int) -> np.ndarray:
    """Trace partielle sur S : état réduit de l'ancilla (site 0)."""
    return np.einsum("aibi->ab", rho.reshape(2, 2**N, 2, 2**N))


def _rho_S(rho: np.ndarray, N: int) -> np.ndarray:
    """Trace partielle sur R : état réduit du système."""
    return np.einsum("aiaj->ij", rho.reshape(2, 2**N, 2, 2**N))


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


@pytest.mark.parametrize("N", NS)
def test_ancilla_state_physical(N: int) -> None:
    emb = AncillaChargeSectorEmbedding({"N": N})
    psi = emb.state()
    rho = emb.density_matrix()
    assert emb.n_qubits == N + 1
    assert psi.shape == (2 ** (N + 1),)
    assert rho.shape == (2 ** (N + 1), 2 ** (N + 1))
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho @ rho).real, 1.0)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-12)


@pytest.mark.parametrize("N", NS)
def test_ancilla_components_in_right_sector(N: int) -> None:
    emb = AncillaChargeSectorEmbedding({"N": N})
    psi = emb.state()
    nonzero = np.flatnonzero(np.abs(psi) > 1e-12)
    for i in nonzero:
        ancilla, system = divmod(int(i), 2**N)
        expected = N // 2 if ancilla == 0 else N // 2 - 1
        assert bin(system).count("1") == expected
    # Chaque branche porte exactement le poids 1/2.
    assert np.isclose(np.sum(np.abs(psi[: 2**N]) ** 2), 0.5)
    assert np.isclose(np.sum(np.abs(psi[2**N :]) ** 2), 0.5)
    assert nonzero.size == comb(N, N // 2) + comb(N, N // 2 - 1)


@pytest.mark.parametrize("N", NS)
def test_ancilla_reduced_states(N: int) -> None:
    emb = AncillaChargeSectorEmbedding({"N": N})
    rho = emb.density_matrix()
    assert np.allclose(_rho_R(rho, N), np.eye(2) / 2)
    q1 = charge_sector_state(N, N // 2)
    q2 = charge_sector_state(N, N // 2 - 1)
    expected_S = (np.outer(q1, q1.conj()) + np.outer(q2, q2.conj())) / 2
    assert np.allclose(_rho_S(rho, N), expected_S)


def test_ancilla_known_values_N2() -> None:
    # (|0>(|01> + |10>)/sqrt(2) + |1>|00>) / sqrt(2)
    psi = AncillaChargeSectorEmbedding({"N": 2}).state()
    expected = np.zeros(8, dtype=complex)
    expected[[0b001, 0b010]] = 0.5
    expected[0b100] = 1 / np.sqrt(2)
    assert np.allclose(psi, expected)


def test_ancilla_odd_N_charges() -> None:
    emb = AncillaChargeSectorEmbedding({"N": 5})
    assert (emb.Q1, emb.Q2) == (2, 1)


@pytest.mark.parametrize("N", [0, 1])
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
    assert emb.N == 4 and emb.dim == 32


def test_from_config_rejects_scalar() -> None:
    cfg = OmegaConf.create({"ancilla_charge_sector": 3})
    with pytest.raises(TypeError):
        AncillaChargeSectorEmbedding.from_config(cfg)


def test_unknown_embedding() -> None:
    with pytest.raises(KeyError, match="Known"):
        get_embedding("nope")


def test_base_state_not_implemented() -> None:
    emb = Embedding({"N": 2})
    assert emb.n_qubits == 2 and emb.dim == 4
    with pytest.raises(NotImplementedError):
        emb.state()
