"""local_expectation : <op_site> sans former le produit 2**N x 2**N (issue #95)."""

from __future__ import annotations

import timeit

import numpy as np
import pytest

from QELM.physics import PAULIS, Z, get_site_operator, local_expectation


def random_psi(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    return psi / np.linalg.norm(psi)


def random_rho(N: int, rng: np.random.Generator) -> np.ndarray:
    """Matrice densité mixte aléatoire : A A^dagger normalisée."""
    A = rng.normal(size=(2**N, 2**N)) + 1j * rng.normal(size=(2**N, 2**N))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


@pytest.mark.parametrize("N", [1, 3, 5])
@pytest.mark.parametrize("label", ["X", "Y", "Z"])
def test_matches_full_trace_for_rho(N: int, label: str) -> None:
    rng = np.random.default_rng(N)
    rho = random_rho(N, rng)
    op = PAULIS[label]
    for site in range(N):
        full = np.real(np.trace(get_site_operator(op, site, N) @ rho))
        assert local_expectation(rho, op, site) == pytest.approx(full, abs=1e-12)


@pytest.mark.parametrize("N", [1, 3, 5])
@pytest.mark.parametrize("label", ["X", "Y", "Z"])
def test_matches_full_product_for_psi(N: int, label: str) -> None:
    rng = np.random.default_rng(10 + N)
    psi = random_psi(N, rng)
    op = PAULIS[label]
    for site in range(N):
        full = np.real(psi.conj() @ get_site_operator(op, site, N) @ psi)
        assert local_expectation(psi, op, site) == pytest.approx(full, abs=1e-12)
        rho = np.outer(psi, psi.conj())
        assert local_expectation(rho, op, site) == pytest.approx(full, abs=1e-12)


def test_known_values() -> None:
    """|0...0> : <Z_k> = 1 ; |1> sur le site 1 : <Z_1> = -1 ; état maximalement
    mixte : <Z_k> = 0."""
    N = 3
    psi = np.zeros(2**N, dtype=complex)
    psi[0b010] = 1.0
    assert local_expectation(psi, Z, 0) == 1.0
    assert local_expectation(psi, Z, 1) == -1.0
    assert local_expectation(psi, Z, 2) == 1.0
    mixed = np.eye(2**N) / 2**N
    for site in range(N):
        assert local_expectation(mixed, Z, site) == pytest.approx(0.0, abs=1e-15)


def test_faster_than_full_trace() -> None:
    """Pour N = 8, la contraction bat largement np.trace(Z0 @ rho)."""
    N = 8
    rho = random_rho(N, np.random.default_rng(0))
    Z0 = get_site_operator(Z, 0, N)
    full = min(timeit.repeat(lambda: np.trace(Z0 @ rho), number=3, repeat=5))
    local = min(timeit.repeat(lambda: local_expectation(rho, Z, 0), number=3, repeat=5))
    assert local < full


@pytest.mark.parametrize(
    ("state", "op", "site", "match"),
    [
        (np.eye(4), np.eye(3), 0, "op must have shape"),
        (np.eye(3), Z, 0, "state must have shape"),
        (np.ones(6), Z, 0, "state must have shape"),
        (np.ones((4, 2)), Z, 0, "state must have shape"),
        (np.ones((2, 2, 2)), Z, 0, "state must have shape"),
        (np.eye(4), Z, 2, "site must be in"),
        (np.eye(4), Z, -1, "site must be in"),
    ],
)
def test_rejects_bad_input(
    state: np.ndarray, op: np.ndarray, site: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        local_expectation(state, op, site)
