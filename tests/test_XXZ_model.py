"""The nearest-neighbour XXZ chain, QELM.models.XXZ (issue #236)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM import load_config, set_seed
from QELM.models import XXZChainModel, get_model
from QELM.physics import X, Y, Z, get_site_operator
from sparse_checks import assert_sparse_hermitian


def _dense_reference(model: XXZChainModel) -> np.ndarray:
    """The XXZ Hamiltonian from Kronecker products of Pauli matrices."""
    N = model.N
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in model.bonds():
        for P, c in ((X, model.J), (Y, model.J), (Z, model.Delta)):
            H += c * get_site_operator(P, i, N) @ get_site_operator(P, j, N)
    for i in range(N):
        H += model.fields[i] * get_site_operator(Z, i, N)
    return H


def _charge(N: int) -> np.ndarray:
    """The charge operator Q = sum_i (1 - Z_i) / 2, diagonal."""
    return np.diag(XXZChainModel({"N": N, "J": 1.0}).charges().astype(float))


@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("h", [0.0, 0.7])
def test_matches_the_kronecker_reference(boundary: str, h: float) -> None:
    """The bitwise construction is the sum of Pauli products, with fields."""
    np.random.seed(3)
    model = XXZChainModel(
        {"N": 5, "J": 0.8, "Delta": 0.3, "h": h, "boundary": boundary}
    )
    np.testing.assert_allclose(
        model.hamiltonian().toarray(), _dense_reference(model), atol=1e-12
    )


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_hamiltonian_is_hermitian(boundary: str) -> None:
    np.random.seed(0)
    model = XXZChainModel({"N": 6, "J": 1.0, "h": 1.0, "boundary": boundary})
    assert_sparse_hermitian(model.hamiltonian(), atol=1e-12)


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_commutes_with_the_charge(boundary: str) -> None:
    """[Q, H] = 0 exactly: the chain conserves the number of |1>."""
    np.random.seed(1)
    model = XXZChainModel(
        {"N": 6, "J": 1.0, "Delta": 0.4, "h": 2.0, "boundary": boundary}
    )
    H = model.hamiltonian().toarray()
    Q = _charge(6)
    assert np.max(np.abs(Q @ H - H @ Q)) == 0.0


def test_couples_only_nearest_neighbours() -> None:
    """Every off-diagonal entry swaps the two bits of one bond (i, i + 1)."""
    model = XXZChainModel({"N": 6, "J": 1.0})
    H = model.hamiltonian().tocoo()
    pairs = set()
    for a, b, v in zip(H.row, H.col, H.data, strict=True):
        if a == b:
            continue
        flipped = a ^ b
        assert bin(flipped).count("1") == 2
        assert bin(a).count("1") == bin(b).count("1")
        assert v == pytest.approx(2.0)
        sites = sorted(6 - 1 - k for k in range(6) if flipped >> k & 1)
        pairs.add(tuple(sites))
    assert pairs == {(i, i + 1) for i in range(5)}


def test_periodic_adds_the_closing_bond() -> None:
    assert XXZChainModel({"N": 4, "J": 1.0, "boundary": "periodic"}).bonds() == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ]


def test_heisenberg_dimer_spectrum() -> None:
    """J (X X + Y Y + Z Z) = J (2 SWAP - 1): triplet at J, singlet at -3 J."""
    model = XXZChainModel({"N": 2, "J": 1.5})
    E = np.linalg.eigvalsh(model.hamiltonian().toarray())
    np.testing.assert_allclose(E, [-4.5, 1.5, 1.5, 1.5], atol=1e-12)


def test_heisenberg_open_trimer_spectrum() -> None:
    """H = 4 J S_2 . (S_1 + S_3): 2 J (S = 3/2), 0 and -4 J (S = 1/2)."""
    model = XXZChainModel({"N": 3, "J": 1.0})
    E = np.linalg.eigvalsh(model.hamiltonian().toarray())
    np.testing.assert_allclose(E, [-4, -4, 0, 0, 2, 2, 2, 2], atol=1e-12)


def test_heisenberg_periodic_trimer_spectrum() -> None:
    """H = 2 J (S_tot^2 - 9 / 4): 3 J (S = 3/2), -3 J (two S = 1/2)."""
    model = XXZChainModel({"N": 3, "J": 1.0, "boundary": "periodic"})
    E = np.linalg.eigvalsh(model.hamiltonian().toarray())
    np.testing.assert_allclose(E, [-3] * 4 + [3] * 4, atol=1e-12)


def test_xx_chain_one_particle_band() -> None:
    """Delta = 0, one |1>: hopping 2 J, energies 4 J cos(pi k / (N + 1)) + const."""
    N = 6
    model = XXZChainModel({"N": N, "J": 1.0, "Delta": 0.0})
    H = model.hamiltonian().toarray()
    one = model.sector(1)
    E = np.linalg.eigvalsh(H[np.ix_(one, one)])
    k = np.arange(1, N + 1)
    np.testing.assert_allclose(np.sort(E), np.sort(4 * np.cos(np.pi * k / (N + 1))))


def test_sector_is_the_charge_of_the_state() -> None:
    model = XXZChainModel({"N": 5, "J": 1.0})
    sector = model.sector(0b01100)
    assert sector.size == 10
    assert all(bin(b).count("1") == 2 for b in sector)


def test_clean_chain_draws_nothing() -> None:
    """h = 0 leaves NumPy's stream as it is; h > 0 draws N fields in [-h, h]."""
    np.random.seed(0)
    XXZChainModel({"N": 4, "J": 1.0})
    after_clean = np.random.random()
    np.random.seed(0)
    assert np.random.random() == after_clean
    np.random.seed(0)
    model = XXZChainModel({"N": 50, "J": 1.0, "h": 0.5})
    assert np.all(np.abs(model.fields) <= 0.5) and np.any(model.fields != 0)


@pytest.mark.parametrize(
    "params",
    [
        {"N": 4, "J": 1.0, "boundary": "ring"},
        {"N": 2, "J": 1.0, "boundary": "periodic"},
        {"N": 4, "J": 1.0, "h": -1.0},
    ],
)
def test_rejects_bad_parameters(params: dict) -> None:
    with pytest.raises(ValueError):
        XXZChainModel(params)


def test_selected_by_the_config() -> None:
    """models=XXZ builds the Heisenberg chain of conf/models/XXZ.yaml."""
    cfg = load_config(["models=XXZ", "N=4"])
    set_seed(cfg)
    model = get_model(str(cfg.models)).from_config(cfg)
    assert isinstance(model, XXZChainModel)
    assert (model.N, model.J, model.Delta, model.h) == (4, 1.0, 1.0, 0.0)
    assert model.boundary == "open"
    assert model.dt == pytest.approx(float(cfg.evolution.dt))


def test_evolution_keeps_the_charge() -> None:
    """exp(-i H t) keeps a state of charge Q in its sector, to rounding."""
    np.random.seed(2)
    model = XXZChainModel({"N": 6, "J": 1.0, "h": 1.0})
    psi = np.zeros(model.dim, dtype=complex)
    psi[0b110100] = 1.0
    out = model.evolve(psi, dt=3.0)
    outside = np.setdiff1d(np.arange(model.dim), model.sector(0b110100))
    assert np.sum(np.abs(out[outside]) ** 2) < 1e-24
    assert np.linalg.norm(out) == pytest.approx(1.0)
