"""Partial trace and Rényi / von Neumann entropies of a subsystem A (issue #93).

Entropies are in bits (base 2) unless a test passes ``base=np.e``.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pytest
import scipy.sparse as sp

from QELM import load_config
from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.evaluation import (
    entanglement_spectrum,
    metrics,
    renyi_entropy,
    von_neumann_entropy,
)
from QELM.physics import partial_trace

ORDERS = [0.0, 0.5, 1.0, 2.0, 3.0, math.inf]


def _random_state(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    return psi / np.linalg.norm(psi)


def _random_rho(N: int, seed: int, rank: int = 3) -> np.ndarray:
    """A mixed state: an average of ``rank`` random pure states."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(rank))
    states = [_random_state(N, seed + 1 + k) for k in range(rank)]
    return sum(w * np.outer(s, s.conj()) for w, s in zip(weights, states, strict=True))


def _kron(*ops: np.ndarray) -> np.ndarray:
    return reduce(np.kron, ops)


def _ghz(N: int) -> np.ndarray:
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = psi[-1] = 1 / math.sqrt(2)
    return psi


BELL = _ghz(2)


# ---- partial trace -----------------------------------------------------------


def test_partial_trace_of_a_product_state() -> None:
    """Tr_B(rho_A x rho_B) = rho_A, for each block position."""
    rho_A = _random_rho(1, 0, rank=2)
    rho_B = _random_rho(2, 10, rank=2)
    rho = _kron(rho_A, rho_B)
    np.testing.assert_allclose(partial_trace(rho, [0]), rho_A, atol=1e-12)
    np.testing.assert_allclose(partial_trace(rho, [1, 2]), rho_B, atol=1e-12)


def test_partial_trace_matches_the_explicit_sum() -> None:
    """rho_A[a, a'] = sum_b <a b| rho |a' b>, written out with basis vectors."""
    N, keep = 3, [0, 2]
    rho = _random_rho(N, 1)
    e = np.eye(2)
    expected = np.zeros((4, 4), dtype=complex)
    for a0, a2, c0, c2, b in np.ndindex(2, 2, 2, 2, 2):
        bra = _kron(e[a0], e[b], e[a2])
        ket = _kron(e[c0], e[b], e[c2])
        expected[2 * a0 + a2, 2 * c0 + c2] += bra @ rho @ ket
    np.testing.assert_allclose(partial_trace(rho, keep), expected, atol=1e-12)


@pytest.mark.parametrize("keep", [[0], [1], [3], [0, 1], [1, 3], [2, 0], [3, 1, 0]])
def test_vector_dense_and_sparse_agree(keep: list[int]) -> None:
    """psi, |psi><psi| dense, sparse and the sparse column give the same rho_A."""
    psi = _random_state(4, 2)
    rho = np.outer(psi, psi.conj())
    expected = partial_trace(rho, keep)
    np.testing.assert_allclose(partial_trace(psi, keep), expected, atol=1e-12)
    np.testing.assert_allclose(
        partial_trace(sp.csr_matrix(rho), keep), expected, atol=1e-12
    )
    column = sp.csr_matrix(psi.reshape(-1, 1))
    np.testing.assert_allclose(partial_trace(column, keep), expected, atol=1e-12)


def test_reduced_matrix_is_a_density_matrix() -> None:
    rho_A = partial_trace(_random_rho(4, 3), [1, 2])
    np.testing.assert_allclose(rho_A, rho_A.conj().T, atol=1e-12)
    assert np.trace(rho_A) == pytest.approx(1.0)
    assert np.linalg.eigvalsh(rho_A).min() > -1e-12


def test_keep_order_sets_the_basis_order() -> None:
    """keep=[1, 0] swaps the two qubits of rho_A."""
    rho_A, rho_B = _random_rho(1, 4, rank=2), _random_rho(1, 5, rank=2)
    rho = _kron(rho_A, rho_B)
    np.testing.assert_allclose(partial_trace(rho, [1, 0]), _kron(rho_B, rho_A))


def test_keep_all_and_keep_none() -> None:
    rho = _random_rho(3, 6)
    np.testing.assert_allclose(partial_trace(rho, [0, 1, 2]), rho, atol=1e-12)
    np.testing.assert_allclose(partial_trace(rho, []), [[1.0]], atol=1e-12)


@pytest.mark.parametrize("keep", [[3], [-1], [0, 0]])
def test_bad_keep_is_rejected(keep: list[int]) -> None:
    with pytest.raises(ValueError, match="keep"):
        partial_trace(np.eye(8) / 8, keep)


@pytest.mark.parametrize("shape", [(3,), (4, 2), (1, 1), (2, 2, 2)])
def test_bad_state_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="state must be"):
        partial_trace(np.zeros(shape), [0])


# ---- entropies: acceptance criteria of issue #93 -----------------------------


@pytest.mark.parametrize("sector_A", [[0], [1, 2], [0, 3]])
def test_renyi_converges_to_von_neumann(sector_A: list[int]) -> None:
    """S_n -> S_1 when n -> 1, from both sides (L'Hôpital limit)."""
    psi = _random_state(4, 7)
    s1 = von_neumann_entropy(psi, sector_A)
    for n in (1 - 1e-5, 1 + 1e-5):
        assert np.isclose(renyi_entropy(psi, n, sector_A), s1, atol=1e-4)
    assert renyi_entropy(psi, 1, sector_A) == s1


@pytest.mark.parametrize("state", [_random_state(4, 8), _random_rho(4, 9)])
def test_renyi_decreases_with_n(state: np.ndarray) -> None:
    """S_n >= S_m for n < m on a state that is not maximally mixed on A."""
    values = [renyi_entropy(state, n, [0, 1]) for n in ORDERS]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:], strict=False))
    assert values[0] > values[-1] + 1e-3  # strictly, for this generic state


@pytest.mark.parametrize("n", ORDERS)
def test_additivity_on_product_states(n: float) -> None:
    """S_n(rho_A x rho_B) = S_n(rho_A) + S_n(rho_B)."""
    rho_A, rho_B = _random_rho(2, 11, rank=2), _random_rho(2, 12, rank=3)
    rho_AB = _kron(rho_A, rho_B)
    total = renyi_entropy(rho_AB, n, [0, 1, 2, 3])
    parts = renyi_entropy(rho_A, n, [0, 1]) + renyi_entropy(rho_B, n, [0, 1])
    assert total == pytest.approx(parts, abs=1e-10)


@pytest.mark.parametrize(
    ("N", "sector_A"), [(2, [0]), (3, [0]), (3, [1, 2]), (4, [0, 1])]
)
def test_stabilizer_states_have_a_flat_spectrum(N: int, sector_A: list[int]) -> None:
    """Bell and GHZ: S_n = 1 bit for every n."""
    psi = _ghz(N)
    for n in ORDERS:
        assert renyi_entropy(psi, n, sector_A) == pytest.approx(1.0, abs=1e-12)


def test_stabilizer_cluster_state_flat_spectrum() -> None:
    """Linear cluster state on 4 qubits: half chain gives 1 bit for every n,
    the cut {0, 2} | {1, 3} gives 2 bits (another stabilizer state)."""
    plus = np.ones(16, dtype=complex) / 4
    signs = np.ones(16)
    for idx in range(16):
        bits = [(idx >> (3 - q)) & 1 for q in range(4)]
        signs[idx] = (-1) ** sum(bits[q] * bits[q + 1] for q in range(3))
    psi = signs * plus
    for n in ORDERS:
        assert renyi_entropy(psi, n, [0, 1]) == pytest.approx(1.0, abs=1e-12)
        assert renyi_entropy(psi, n, [0, 2]) == pytest.approx(2.0, abs=1e-12)


def test_subsystem_isolation() -> None:
    """A = every qubit of a pure state: 0. Half of a maximally entangled
    state (two Bell pairs across the cut): the maximal |A| bits."""
    psi = _random_state(4, 13)
    for n in ORDERS:
        assert renyi_entropy(psi, n, [0, 1, 2, 3]) == pytest.approx(0.0, abs=1e-10)
        assert renyi_entropy(psi, n, []) == pytest.approx(0.0, abs=1e-10)
    # Qubits (0, 2) and (1, 3) form Bell pairs, so A = {0, 1} holds 2 bits.
    bell_pairs = _kron(BELL, BELL).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).ravel()
    for n in ORDERS:
        assert renyi_entropy(bell_pairs, n, [0, 1]) == pytest.approx(2.0, abs=1e-12)


# ---- entropies: other properties ---------------------------------------------


@pytest.mark.parametrize("n", ORDERS)
def test_pure_state_entropy_is_symmetric(n: float) -> None:
    """For a pure global state, S_n(A) = S_n(B)."""
    psi = _random_state(5, 14)
    assert renyi_entropy(psi, n, [0, 3]) == pytest.approx(
        renyi_entropy(psi, n, [1, 2, 4]), abs=1e-10
    )


@pytest.mark.parametrize("n", ORDERS)
def test_vector_dense_and_sparse_entropies_agree(n: float) -> None:
    psi = _random_state(4, 15)
    rho = np.outer(psi, psi.conj())
    expected = renyi_entropy(rho, n, [1, 2])
    assert renyi_entropy(psi, n, [1, 2]) == pytest.approx(expected, abs=1e-10)
    assert renyi_entropy(sp.csr_matrix(rho), n, [1, 2]) == pytest.approx(
        expected, abs=1e-10
    )


def test_known_values_of_the_limits() -> None:
    """rho_A = diag(1/2, 1/4, 1/4): closed forms of S_0, S_1, S_2, S_inf."""
    rho = np.diag([0.5, 0.25, 0.25, 0.0])  # two qubits, A = both
    A = [0, 1]
    assert renyi_entropy(rho, 0, A) == pytest.approx(math.log2(3))
    assert von_neumann_entropy(rho, A) == pytest.approx(1.5)
    assert renyi_entropy(rho, 2, A) == pytest.approx(-math.log2(0.375))
    assert renyi_entropy(rho, math.inf, A) == pytest.approx(1.0)
    assert renyi_entropy(rho, 3, A) == pytest.approx(math.log2(0.15625) / -2)


def test_base_e_gives_nats() -> None:
    assert von_neumann_entropy(BELL, [0], base=np.e) == pytest.approx(math.log(2))
    assert renyi_entropy(BELL, 2, [0], base=np.e) == pytest.approx(math.log(2))


def test_spectrum_is_sorted_and_normalized() -> None:
    values = entanglement_spectrum(_random_state(4, 16), [0, 1])
    assert np.all(np.diff(values) <= 0)
    assert values.sum() == pytest.approx(1.0)


def test_maximally_mixed_qubits() -> None:
    rho = np.eye(8) / 8
    for n in ORDERS:
        assert renyi_entropy(rho, n, [0, 2]) == pytest.approx(2.0)


@pytest.mark.parametrize("n", [-1.0, math.nan])
def test_bad_order_is_rejected(n: float) -> None:
    with pytest.raises(ValueError, match="n must be"):
        renyi_entropy(BELL, n, [0])


@pytest.mark.parametrize("base", [1.0, 0.0, -2.0])
def test_bad_base_is_rejected(base: float) -> None:
    with pytest.raises(ValueError, match="base"):
        renyi_entropy(BELL, 2, [0], base=base)
    with pytest.raises(ValueError, match="base"):
        von_neumann_entropy(BELL, [0], base=base)


def test_bad_sector_is_rejected() -> None:
    with pytest.raises(ValueError, match="sector_A"):
        renyi_entropy(BELL, 2, [2])


def test_ancilla_entropy_of_the_embedding() -> None:
    """The charge-sector embedding entangles the ancilla with S: 1 bit, and
    the ancilla metric of evaluation.metrics is the case A = {0}."""
    cfg = load_config(["N=5"])
    emb = AncillaChargeSectorEmbedding.from_config(cfg)
    rho = emb.density_matrix()
    assert von_neumann_entropy(emb.state(), [0]) == pytest.approx(1.0)
    assert metrics.von_neumann_entropy(rho, 5) == pytest.approx(
        von_neumann_entropy(rho, [0])
    )
    # Same answer for the complement, since the global state is pure.
    assert von_neumann_entropy(emb.state(), [1, 2, 3, 4]) == pytest.approx(1.0)
