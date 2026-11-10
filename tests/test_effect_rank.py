"""Tests of the exact effect-space rank (QELM.evaluation.effect_rank)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from QELM.evaluation.effect_rank import (
    GATES,
    PAULIS,
    apply_gate,
    brick_pairs,
    draw_circuit,
    effect_rank,
    final_readout_basis,
    heisenberg_transfer,
    measure_site,
    no_y_columns,
    rank_bounds,
)


def _brute_force(gates: list[list[np.ndarray]], masks: np.ndarray) -> tuple[int, int]:
    """Dense and angle ranks by listing every record and its effect."""
    depth, n = masks.shape
    dim = 2**n

    def embed(u: np.ndarray, i: int) -> np.ndarray:
        return np.kron(np.kron(np.eye(2**i), u), np.eye(2 ** (n - i - 2)))

    def proj(i: int, s: int) -> np.ndarray:
        return np.kron(
            np.kron(np.eye(2**i), np.diag([1 - s, s])), np.eye(2 ** (n - i - 1))
        )

    kraus = [np.eye(dim, dtype=complex)]
    for t in range(depth):
        layer = np.eye(dim, dtype=complex)
        for i, g in zip(brick_pairs(n), gates[t]):
            layer = embed(g, i) @ layer
        new = []
        for k in kraus:
            branch = [layer @ k]
            for i in range(n):
                if masks[t, i]:
                    branch = [proj(i, s) @ b for b in branch for s in (0, 1)]
            new += branch
        kraus = new
    strings = list(itertools.product(range(4), repeat=n))
    ops = []
    for a in strings:
        m = np.array([[1.0 + 0j]])
        for k in a:
            m = np.kron(m, PAULIS[k])
        ops.append(m)
    rows = []
    for k in kraus:
        for z in range(dim):
            kz = k[z : z + 1]
            effect = kz.conj().T @ kz
            rows.append([np.trace(effect @ s).real for s in ops])
    c = np.array(rows)
    no_y = np.array([2 not in a for a in strings])

    def rank(m: np.ndarray) -> int:
        return int(np.linalg.matrix_rank(m, tol=1e-9 * np.linalg.norm(m, 2)))

    return rank(c), rank(c[:, no_y])


@pytest.mark.parametrize("kind", sorted(GATES))
def test_transfer_is_orthogonal_and_fixes_identity(kind: str) -> None:
    r = heisenberg_transfer(GATES[kind](np.random.default_rng(3)))
    np.testing.assert_allclose(r @ r.T, np.eye(16), atol=1e-12)
    np.testing.assert_allclose(r[:, 0], np.eye(16)[0], atol=1e-12)


def test_transfer_matches_conjugation() -> None:
    rng = np.random.default_rng(4)
    u = GATES["haar"](rng)
    coeffs = rng.standard_normal(16)
    paulis2 = [np.kron(a, b) for a in PAULIS for b in PAULIS]
    e = sum(c * s for c, s in zip(coeffs, paulis2))
    new = u.conj().T @ e @ u
    expected = np.array([np.trace(s @ new).real / 4 for s in paulis2])
    np.testing.assert_allclose(heisenberg_transfer(u) @ coeffs, expected, atol=1e-12)


def test_final_readout_and_angle_columns() -> None:
    b = final_readout_basis(3)
    assert b.shape == (8, 64)
    np.testing.assert_allclose(b @ b.T, np.eye(8))
    assert no_y_columns(3).sum() == 27


def test_measure_and_gate_keep_rows_orthonormal() -> None:
    n = 3
    rng = np.random.default_rng(5)
    basis = final_readout_basis(n)
    basis = apply_gate(basis, n, 0, heisenberg_transfer(GATES["haar"](rng)))
    basis = measure_site(basis, n, 1, 1e-6)
    np.testing.assert_allclose(basis @ basis.T, np.eye(basis.shape[0]), atol=1e-10)


@pytest.mark.parametrize("kind", sorted(GATES))
@pytest.mark.parametrize("p", [0.15, 0.35])
def test_rank_matches_record_enumeration(kind: str, p: float) -> None:
    n = 3
    rng = np.random.default_rng((7, int(100 * p)))
    gates = [[GATES[kind](rng) for _ in brick_pairs(n)] for t in range(n)]
    masks = rng.random((n, n)) < p
    transfers = [[heisenberg_transfer(g) for g in layer] for layer in gates]
    r = effect_rank(transfers, masks)
    assert (r.dense, r.angle) == _brute_force(gates, masks)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_end_points_give_final_readout_rank(p: float) -> None:
    n = 4
    circuit = draw_circuit("haar", n, n, np.random.default_rng(8))
    r = effect_rank(circuit, np.full((n, n), p > 0.5))
    assert r.dense == 2**n


@pytest.mark.parametrize("kind", sorted(GATES))
def test_rank_below_symmetry_bound_and_reproducible(kind: str) -> None:
    n = 4
    masks = np.random.default_rng(9).random((n, n)) < 0.25
    first = effect_rank(draw_circuit(kind, n, n, np.random.default_rng(1)), masks)
    again = effect_rank(draw_circuit(kind, n, n, np.random.default_rng(1)), masks)
    assert first == again
    bounds = rank_bounds(n)
    key = {"haar": "none", "clifford": "none", "u1": "u1", "z2": "z2"}[kind]
    assert first.dense <= bounds["dense"][key]
    assert first.angle <= bounds["angle"][key]
    if kind == "clifford":
        assert first.dense == 2**n
