"""Random U(1) circuits of QELM.models.circuits (issue #137)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation import von_neumann_entropy
from QELM.models.circuits import (
    U1BrickworkCircuit,
    apply_two_qubit_gate,
    haar_unitary,
    random_u1_gate,
)
from QELM.physics import get_site_operator

# Number operator n = |1><1| on one qubit.
N_OP = np.diag([0.0, 1.0])


def _charge(N: int, sites: list[int]) -> np.ndarray:
    """Dense total charge sum_{i in sites} n_i on N qubits."""
    return sum(np.asarray(get_site_operator(N_OP, i, N)) for i in sites)


def _random_state(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.standard_normal(2**N) + 1j * rng.standard_normal(2**N)
    return psi / np.linalg.norm(psi)


def _dense_gate(gate: np.ndarray, i: int, j: int, N: int) -> np.ndarray:
    """Full 2**N matrix of a two-qubit gate on (i, j), built entry by entry."""
    dim = 2**N
    G = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        bi, bj = (col >> (N - 1 - i)) & 1, (col >> (N - 1 - j)) & 1
        for out in range(4):
            oi, oj = out >> 1, out & 1
            row = col
            row = row & ~(1 << (N - 1 - i)) | (oi << (N - 1 - i))
            row = row & ~(1 << (N - 1 - j)) | (oj << (N - 1 - j))
            G[row, col] += gate[out, 2 * bi + bj]
    return G


def test_haar_unitary_is_unitary() -> None:
    """U U^dagger = 1 pour plusieurs tailles."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 4):
        U = haar_unitary(n, rng)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(n), atol=1e-12)


def test_haar_unitary_first_moment() -> None:
    """Haar sur U(2) : E|U_00|^2 = 1/2 et E U_00 = 0."""
    rng = np.random.default_rng(1)
    U00 = np.array([haar_unitary(2, rng)[0, 0] for _ in range(4000)])
    assert abs(np.mean(np.abs(U00) ** 2) - 0.5) < 0.02
    assert abs(np.mean(U00)) < 0.05


def test_u1_gate_is_unitary_and_conserves_charge() -> None:
    """La porte est unitaire et ne mélange que |01> et |10>."""
    rng = np.random.default_rng(2)
    u = random_u1_gate(rng)
    np.testing.assert_allclose(u @ u.conj().T, np.eye(4), atol=1e-12)
    charge = np.diag([0, 1, 1, 2])
    np.testing.assert_allclose(u @ charge, charge @ u, atol=1e-12)
    assert abs(abs(u[0, 0]) - 1) < 1e-12 and abs(abs(u[3, 3]) - 1) < 1e-12


@pytest.mark.parametrize(("i", "j"), [(0, 1), (1, 0), (0, 2), (2, 1)])
def test_apply_gate_matches_dense_matrix(i: int, j: int) -> None:
    """Le contraction tensorielle donne G psi et G rho G^dagger."""
    rng = np.random.default_rng(3)
    N = 3
    gate = haar_unitary(4, rng)
    G = _dense_gate(gate, i, j, N)
    psi = _random_state(N, rng)
    np.testing.assert_allclose(
        apply_two_qubit_gate(psi, gate, i, j), G @ psi, atol=1e-12
    )
    rho = np.outer(psi, psi.conj())
    np.testing.assert_allclose(
        apply_two_qubit_gate(rho, gate, i, j), G @ rho @ G.conj().T, atol=1e-12
    )


def test_apply_gate_on_adjacent_qubits_is_a_kron() -> None:
    """Sur (0, 1) de 3 qubits, la porte vaut kron(gate, I_2)."""
    rng = np.random.default_rng(4)
    gate = haar_unitary(4, rng)
    psi = _random_state(3, rng)
    expected = np.kron(gate, np.eye(2)) @ psi
    np.testing.assert_allclose(
        apply_two_qubit_gate(psi, gate, 0, 1), expected, atol=1e-12
    )


def test_apply_gate_rejects_bad_input() -> None:
    gate = np.eye(4)
    with pytest.raises(ValueError):
        apply_two_qubit_gate(np.ones(8), gate, 1, 1)
    with pytest.raises(ValueError):
        apply_two_qubit_gate(np.ones(8), gate, 0, 3)
    with pytest.raises(ValueError):
        apply_two_qubit_gate(np.ones(6), gate, 0, 1)
    with pytest.raises(ValueError):
        apply_two_qubit_gate(np.ones(8), np.eye(2), 0, 1)


def test_brickwork_bonds_open_and_periodic() -> None:
    """Ns = 6 après une ancilla : couches paires et impaires, bords ouverts ou non."""
    params = {"N": 7, "Nr": 1, "depth": 2}
    circuit = U1BrickworkCircuit(params, np.random.default_rng(0))
    assert circuit.bonds(0) == [(1, 2), (3, 4), (5, 6)]
    assert circuit.bonds(1) == [(2, 3), (4, 5)]
    periodic = U1BrickworkCircuit({**params, "boundary": "periodic"})
    assert periodic.bonds(1) == [(2, 3), (4, 5), (6, 1)]
    assert periodic.bonds(2) == [(1, 2), (3, 4), (5, 6)]
    # One layer is a full period: the even half-layer, then the odd one.
    assert [(i, j) for i, j, _ in circuit.layers[0]] == [
        (1, 2),
        (3, 4),
        (5, 6),
        (2, 3),
        (4, 5),
    ]
    assert [len(layer) for layer in circuit.layers] == [5, 5]


def test_scrambling_pairs_are_disjoint_system_qubits() -> None:
    circuit = U1BrickworkCircuit({"N": 7, "depth": 1, "scrambling_layers": 5})
    assert len(circuit.scrambling) == 5
    for layer in circuit.scrambling:
        sites = [q for i, j, _ in layer for q in (i, j)]
        assert len(sites) == len(set(sites)) == 6
        assert 0 not in sites


def test_circuit_conserves_system_charge_and_ancilla_entropy() -> None:
    """Les portes U(1) sur S gardent la charge de S et l'entropie de R."""
    rng = np.random.default_rng(5)
    N = 5
    circuit = U1BrickworkCircuit(
        {"N": N, "depth": 4, "scrambling_layers": 3}, np.random.default_rng(6)
    )
    psi = _random_state(N, rng)
    Q = _charge(N, list(range(1, N)))
    out = circuit.scramble(psi)
    for t in range(circuit.depth):
        out = circuit.apply_layer(out, t)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-12)
    # Full charge distribution, not only its mean: Q commutes with every gate.
    for q in range(N):
        P = np.diag((np.diag(Q) == q).astype(float))
        assert abs(np.vdot(out, P @ out) - np.vdot(psi, P @ psi)) < 1e-12
    assert abs(von_neumann_entropy(out, [0]) - von_neumann_entropy(psi, [0])) < 1e-10


def test_same_seed_same_gates_and_redraw_changes_them() -> None:
    a = U1BrickworkCircuit({"N": 5, "depth": 2}, np.random.default_rng(7))
    b = U1BrickworkCircuit({"N": 5, "depth": 2}, np.random.default_rng(7))
    np.testing.assert_array_equal(a.layers[0][0][2], b.layers[0][0][2])
    np.random.seed(0)
    c = U1BrickworkCircuit({"N": 5, "depth": 2})
    np.random.seed(0)
    d = U1BrickworkCircuit({"N": 5, "depth": 2})
    np.testing.assert_array_equal(c.layers[1][0][2], d.layers[1][0][2])
    before = c.layers[0][0][2].copy()
    c.redraw()
    assert not np.allclose(before, c.layers[0][0][2])


@pytest.mark.parametrize(
    "params",
    [
        {"N": 2, "depth": 1},
        {"N": 5, "depth": -1},
        {"N": 5, "depth": 1, "boundary": "twisted"},
    ],
)
def test_circuit_rejects_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        U1BrickworkCircuit(params)


def test_from_config_reads_the_block() -> None:
    from QELM import load_config

    cfg = load_config(config_name="charge_sharpening")
    circuit = U1BrickworkCircuit.from_config(cfg)
    assert (circuit.N, circuit.Nr, circuit.Ns) == (7, 1, 6)
    assert circuit.depth == 3  # t = L / 2 for L = 6
    assert circuit.scrambling_layers == 5
    assert circuit.boundary == "open"
    assert circuit.fixed_instance is True
