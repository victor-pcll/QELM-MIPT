"""Tests des encodings d'état et de la classe Encoding commune."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
import scipy.sparse as sp
from omegaconf import OmegaConf
from scipy.sparse.linalg import norm as sparse_norm

from QELM.encoding import (
    AncillaChargeSectorEncoding,
    Encoding,
    charge_sector_indices,
    charge_sector_state,
    get_encoding,
)
from QELM.models import TransverseFieldIsingModel

NS = [2, 3, 4, 5, 6]


# N qubits de système (issue #235) : l'état a N_total = N + 1 qubits avec
# l'ancilla.
NS_SYSTEM = [2, 3, 4, 5, 6]


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


@pytest.mark.parametrize("N", NS_SYSTEM)
def test_ancilla_state_physical(N: int) -> None:
    """N qubits de système, N + 1 qubits alloués avec l'ancilla (issue #235)."""
    emb = AncillaChargeSectorEncoding({"N": N})
    psi = emb.state()
    rho = emb.density_matrix()
    assert (emb.N, emb.Nr, emb.N_total) == (N, 1, N + 1)
    assert not hasattr(emb, "Ns")
    assert emb.dim == 2 ** (N + 1)
    assert psi.shape == (2 ** (N + 1),)
    assert rho.shape == (2 ** (N + 1), 2 ** (N + 1))
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho @ rho).real, 1.0)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-12)


@pytest.mark.parametrize("N", NS_SYSTEM)
def test_ancilla_components_in_right_sector(N: int) -> None:
    psi = AncillaChargeSectorEncoding({"N": N}).state()
    nonzero = np.flatnonzero(np.abs(psi) > 1e-12)
    for i in nonzero:
        ancilla, system = divmod(int(i), 2**N)
        expected = N // 2 if ancilla == 0 else N // 2 - 1
        assert bin(system).count("1") == expected
    # Chaque branche porte exactement le poids 1/2.
    assert np.isclose(np.sum(np.abs(psi[: 2**N]) ** 2), 0.5)
    assert np.isclose(np.sum(np.abs(psi[2**N :]) ** 2), 0.5)
    assert nonzero.size == comb(N, N // 2) + comb(N, N // 2 - 1)


@pytest.mark.parametrize("N", NS_SYSTEM)
def test_ancilla_reduced_states(N: int) -> None:
    rho = AncillaChargeSectorEncoding({"N": N}).density_matrix()
    assert np.allclose(_rho_R(rho, N), np.eye(2) / 2)
    q1 = charge_sector_state(N, N // 2)
    q2 = charge_sector_state(N, N // 2 - 1)
    expected_S = (np.outer(q1, q1.conj()) + np.outer(q2, q2.conj())) / 2
    assert np.allclose(_rho_S(rho, N), expected_S)


def test_ancilla_known_values_N2() -> None:
    # N = 2 qubits de système + l'ancilla, 3 qubits en tout.
    # (|0>(|01> + |10>)/sqrt(2) + |1>|00>) / sqrt(2)
    psi = AncillaChargeSectorEncoding({"N": 2}).state()
    expected = np.zeros(8, dtype=complex)
    expected[[0b001, 0b010]] = 0.5
    expected[0b100] = 1 / np.sqrt(2)
    assert np.allclose(psi, expected)


def test_ancilla_odd_system_charges() -> None:
    # N = 5 qubits de système : charges 2 et 1.
    emb = AncillaChargeSectorEncoding({"N": 5})
    assert (emb.N, emb.Q1, emb.Q2) == (5, 2, 1)


@pytest.mark.parametrize("N", [0, 1])
def test_ancilla_rejects_small_N(N: int) -> None:
    with pytest.raises(ValueError):
        AncillaChargeSectorEncoding({"N": N})


def test_ancilla_deterministic() -> None:
    emb = AncillaChargeSectorEncoding({"N": 4})
    assert np.array_equal(emb.state(), emb.state())


def test_from_config_and_registry() -> None:
    cfg = OmegaConf.create({"ancilla_charge_sector": {"N": 4}})
    cls = get_encoding("ancilla_charge_sector")
    assert cls is AncillaChargeSectorEncoding
    emb = cls.from_config(cfg)
    assert isinstance(emb, Encoding)
    assert (emb.N, emb.N_total, emb.dim) == (4, 5, 32)


def test_from_config_rejects_scalar() -> None:
    cfg = OmegaConf.create({"ancilla_charge_sector": 3})
    with pytest.raises(TypeError):
        AncillaChargeSectorEncoding.from_config(cfg)


def test_unknown_encoding() -> None:
    with pytest.raises(KeyError, match="Known"):
        get_encoding("nope")


def test_base_without_ancilla() -> None:
    emb = Encoding({"N": 2})
    assert (emb.N, emb.N_total, emb.dim) == (2, 2, 4)
    with pytest.raises(NotImplementedError):
        emb.state()


def test_same_system_as_model() -> None:
    # Le même N donne le même système : le modèle agit sur les N qubits du
    # système, l'état y ajoute l'ancilla (issue #235).
    model = TransverseFieldIsingModel({"N": 4, "J": 1.0, "h": 0.5})
    emb = AncillaChargeSectorEncoding({"N": model.N})
    assert emb.dim == 2 * model.dim
    H = np.kron(np.eye(2), model.hamiltonian().toarray())
    assert H.shape == emb.density_matrix().shape


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
        (["N=5"], 5),
        (["models=kitaev"], 4),
    ):
        cfg = load_config(overrides)
        emb = AncillaChargeSectorEncoding.from_config(cfg)
        assert emb.N == N
        assert emb.state().shape == (2 ** (N + 1),)


def test_from_config_missing_block_is_explicit() -> None:
    """Sans bloc, l'erreur dit quoi ajouter au lieu d'un ConfigKeyError."""
    cfg = OmegaConf.create({"seed": 0})
    OmegaConf.set_struct(cfg, True)
    with pytest.raises(ValueError, match="no 'ancilla_charge_sector' block"):
        AncillaChargeSectorEncoding.from_config(cfg)


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_sparse_state_matches_dense(N: int) -> None:
    """state(sparse=True) est une colonne CSR (2**N_total, 1) égale à state()."""
    emb = AncillaChargeSectorEncoding({"N": N})
    psi = emb.state()
    psi_sp = emb.state(sparse=True)
    assert isinstance(psi_sp, sp.csr_matrix)
    assert psi.shape == (emb.dim,) == (2 ** (N + 1),)
    assert psi_sp.shape == (emb.dim, 1)
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


@pytest.mark.parametrize("N", [2, 3, 4])
def test_sparse_density_matrix_matches_dense(N: int) -> None:
    """density_matrix(sparse=True) est la CSR de density_matrix()."""
    emb = AncillaChargeSectorEncoding({"N": N})
    rho = emb.density_matrix()
    rho_sp = emb.density_matrix(sparse=True)
    dim = 2 ** (N + 1)
    assert isinstance(rho, np.ndarray) and rho.shape == (dim, dim)
    assert isinstance(rho_sp, sp.csr_matrix) and rho_sp.shape == (dim, dim)
    np.testing.assert_allclose(rho_sp.toarray(), rho, atol=1e-15)
    assert rho_sp.nnz == np.count_nonzero(rho)


def test_dense_is_the_default_state() -> None:
    """Sans argument, state() et density_matrix() sont des tableaux numpy."""
    emb = AncillaChargeSectorEncoding({"N": 2})
    assert isinstance(emb.state(), np.ndarray)
    assert isinstance(emb.density_matrix(), np.ndarray)
    assert isinstance(charge_sector_state(3, 1), np.ndarray)


def test_encoding_replaces_embedding() -> None:
    """Issue #197: the initial states live in ``QELM.encoding`` and every name
    says encoding; the old ``QELM.embedding`` package and names are gone."""
    import importlib

    encoding = importlib.import_module("QELM.encoding")
    assert encoding.AncillaChargeSectorEncoding.__module__ == (
        "QELM.encoding.charge_sector"
    )
    assert encoding.RotationEncoding.__module__ == "QELM.encoding.rotation"
    assert encoding.get_encoding("rotation_encoding") is encoding.RotationEncoding
    assert issubclass(encoding.AncillaChargeSectorEncoding, encoding.Encoding)
    for old in (
        "Embedding",
        "AncillaChargeSectorEmbedding",
        "RotationEmbedding",
        "get_embedding",
    ):
        assert not hasattr(encoding, old), old
    with pytest.raises(KeyError):
        encoding.get_encoding("rotation_embedding")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("QELM." + "embedding")
