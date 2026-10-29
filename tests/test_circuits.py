"""Random circuits of QELM.models.circuits: U(1) (issue #137), Haar and
Clifford (issue #171)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.evaluation import von_neumann_entropy
from QELM.models.circuits.base import BrickworkCircuit, apply_two_qubit_gate
from QELM.models.circuits.clifford import (
    CliffordBrickworkCircuit,
    _phase_key,
    clifford_group_2q,
    random_clifford_gate,
)
from QELM.models.circuits.haar import (
    HaarBrickworkCircuit,
    U1BrickworkCircuit,
    haar_gate,
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
    """N = 6 après une ancilla : couches paires et impaires, bords ouverts ou non."""
    params = {"N": 6, "Nr": 1, "depth": 2}
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
    circuit = U1BrickworkCircuit({"N": 6, "depth": 1, "scrambling_layers": 5})
    assert len(circuit.scrambling) == 5
    for layer in circuit.scrambling:
        sites = [q for i, j, _ in layer for q in (i, j)]
        assert len(sites) == len(set(sites)) == 6
        assert 0 not in sites


def test_circuit_conserves_system_charge_and_ancilla_entropy() -> None:
    """Les portes U(1) sur S gardent la charge de S et l'entropie de R."""
    rng = np.random.default_rng(5)
    N = 4
    circuit = U1BrickworkCircuit(
        {"N": N, "depth": 4, "scrambling_layers": 3}, np.random.default_rng(6)
    )
    # The state has the ancilla and the N system qubits (issue #235).
    psi = _random_state(N + 1, rng)
    Q = _charge(N + 1, list(range(1, N + 1)))
    out = circuit.scramble(psi)
    for t in range(circuit.depth):
        out = circuit.apply_layer(out, t)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-12)
    # Full charge distribution, not only its mean: Q commutes with every gate.
    for q in range(N + 1):
        P = np.diag((np.diag(Q) == q).astype(float))
        assert abs(np.vdot(out, P @ out) - np.vdot(psi, P @ psi)) < 1e-12
    assert abs(von_neumann_entropy(out, [0]) - von_neumann_entropy(psi, [0])) < 1e-10


def test_same_seed_same_gates_and_redraw_changes_them() -> None:
    a = U1BrickworkCircuit({"N": 4, "depth": 2}, np.random.default_rng(7))
    b = U1BrickworkCircuit({"N": 4, "depth": 2}, np.random.default_rng(7))
    np.testing.assert_array_equal(a.layers[0][0][2], b.layers[0][0][2])
    np.random.seed(0)
    c = U1BrickworkCircuit({"N": 4, "depth": 2})
    np.random.seed(0)
    d = U1BrickworkCircuit({"N": 4, "depth": 2})
    np.testing.assert_array_equal(c.layers[1][0][2], d.layers[1][0][2])
    before = c.layers[0][0][2].copy()
    c.redraw()
    assert not np.allclose(before, c.layers[0][0][2])


@pytest.mark.parametrize(
    "params",
    [
        {"N": 1, "depth": 1},
        {"N": 4, "depth": -1},
        {"N": 4, "depth": 1, "boundary": "twisted"},
    ],
)
def test_circuit_rejects_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        U1BrickworkCircuit(params)


def test_from_config_reads_the_block() -> None:
    from QELM import load_config

    cfg = load_config(config_name="charge_sharpening")
    circuit = U1BrickworkCircuit.from_config(cfg)
    assert (circuit.N, circuit.Nr, circuit.N_total) == (6, 1, 7)
    assert not hasattr(circuit, "Ns")
    assert circuit.depth == 3  # t = L / 2 for L = 6
    assert circuit.scrambling_layers == 5
    assert circuit.boundary == "open"
    assert circuit.fixed_instance is True


# Single-qubit Pauli matrices, for the Clifford checks.
PAULIS = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]]),
    np.diag([1.0 + 0j, -1.0]),
]
# The 16 two-qubit Pauli strings P_a (x) P_b.
PAULI_STRINGS = [np.kron(a, b) for a in PAULIS for b in PAULIS]


def _is_pauli_string(M: np.ndarray) -> bool:
    """True if M = +-P or +-iP for a two-qubit Pauli string P."""
    for P in PAULI_STRINGS:
        c = np.trace(P.conj().T @ M) / 4
        if abs(abs(c) - 1) < 1e-9:
            return bool(np.allclose(M, c * P, atol=1e-9))
    return False


def test_haar_gate_is_a_unitary_of_u4() -> None:
    """Haar on U(4): unitary, E|U_00|^2 = 1/4, and no conserved charge."""
    rng = np.random.default_rng(2)
    gates = [haar_gate(rng) for _ in range(4000)]
    for u in gates[:20]:
        assert u.shape == (4, 4)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(4), atol=1e-12)
    assert abs(np.mean([abs(u[0, 0]) ** 2 for u in gates]) - 0.25) < 0.02
    # A U(1) gate has u[0, 1] = 0; a Haar gate mixes |00> and |01>.
    assert np.mean([abs(u[0, 1]) ** 2 for u in gates]) == pytest.approx(0.25, abs=0.02)


def test_clifford_group_has_11520_distinct_unitaries() -> None:
    """|C_2| = 11 520 up to a global phase, every element unitary."""
    group = clifford_group_2q()
    assert len(group) == 11520
    assert len({_phase_key(u) for u in group}) == 11520
    for u in group[::97]:
        np.testing.assert_allclose(u @ u.conj().T, np.eye(4), atol=1e-12)


def test_clifford_group_is_closed_and_maps_paulis_to_paulis() -> None:
    """The product of two elements is an element, and every element maps a
    Pauli string to a Pauli string, up to a sign: the Clifford property."""
    group = clifford_group_2q()
    keys = {_phase_key(u) for u in group}
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b = (group[k] for k in rng.integers(len(group), size=2))
        assert _phase_key(a @ b) in keys
    for u in group[::53]:
        for P in PAULI_STRINGS:
            assert _is_pauli_string(u @ P @ u.conj().T)
    # A Haar gate is not a Clifford: it does not map Z (x) 1 to a Pauli string.
    u = haar_gate(np.random.default_rng(4))
    assert not _is_pauli_string(u @ PAULI_STRINGS[12] @ u.conj().T)


def test_random_clifford_gate_is_uniform_and_seeded() -> None:
    """Same seed, same gate; the draws spread over the whole group."""
    a = random_clifford_gate(np.random.default_rng(5))
    b = random_clifford_gate(np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    keys = {_phase_key(u) for u in clifford_group_2q()}
    rng = np.random.default_rng(6)
    draws = [_phase_key(random_clifford_gate(rng)) for _ in range(2000)]
    assert set(draws) <= keys
    # 2000 uniform draws among 11 520: about 1834 distinct elements.
    assert len(set(draws)) > 1750
    # Modifying a draw leaves the cached group untouched.
    a[:] = 0
    assert len({_phase_key(u) for u in clifford_group_2q()}) == 11520


@pytest.mark.parametrize(
    ("cls", "name"),
    [
        (HaarBrickworkCircuit, "haar_circuit"),
        (CliffordBrickworkCircuit, "clifford_circuit"),
    ],
)
def test_haar_and_clifford_circuits_keep_the_brickwork(
    cls: type[BrickworkCircuit], name: str
) -> None:
    """Same bonds as the U(1) circuit, gates of their own kind, the ancilla
    untouched, and the same seed gives the same gates."""
    params = {"N": 4, "Nr": 1, "depth": 3}
    circuit = cls(params, np.random.default_rng(7))
    u1 = U1BrickworkCircuit(params, np.random.default_rng(7))
    assert cls.name == name
    assert [[(i, j) for i, j, _ in layer] for layer in circuit.layers] == [
        [(i, j) for i, j, _ in layer] for layer in u1.layers
    ]
    gates = [g for layer in circuit.layers for _, _, g in layer]
    for g in gates:
        np.testing.assert_allclose(g @ g.conj().T, np.eye(4), atol=1e-12)
    # Not U(1) gates: some gate moves weight out of the one-particle block.
    assert any(abs(g[0, 1]) > 1e-6 or abs(g[1, 0]) > 1e-6 for g in gates)
    if cls is CliffordBrickworkCircuit:
        keys = {_phase_key(u) for u in clifford_group_2q()}
        assert all(_phase_key(g) in keys for g in gates)
    same = cls(params, np.random.default_rng(7))
    for la, lb in zip(circuit.layers, same.layers, strict=True):
        for (_, _, ga), (_, _, gb) in zip(la, lb, strict=True):
            np.testing.assert_array_equal(ga, gb)
    # The gates act on the system: the ancilla's reduced state stays I / 2
    # on a state maximally entangling it with the system.
    rng = np.random.default_rng(8)
    psi = np.concatenate([_random_state(4, rng), _random_state(4, rng)])
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    before = von_neumann_entropy(rho, [0])
    for t in range(circuit.depth):
        rho = circuit.apply_layer(rho, t)
    assert von_neumann_entropy(rho, [0]) == pytest.approx(before, abs=1e-9)


def test_haar_and_clifford_from_config() -> None:
    """from_config reads the haar_circuit and clifford_circuit blocks; the
    QELM experiment adds the depth from the number of steps."""
    for cls in (HaarBrickworkCircuit, CliffordBrickworkCircuit):
        cfg = OmegaConf.create({cls.name: {"N": 3, "Nr": 1, "depth": 2}})
        circuit = cls.from_config(cfg)
        assert (circuit.N, circuit.N_total) == (3, 4) and len(circuit.layers) == 2


def test_circuits_package_layout() -> None:
    """Issue #189: the base class and the gate application live in
    ``circuits.base``, the Haar and U(1) gates in ``circuits.haar``, the
    Clifford gates in ``circuits.clifford``, each circuit inheriting from
    :class:`BrickworkCircuit`, and the package re-exports them all."""
    import QELM.models.circuits as circuits

    for cls, module in (
        (U1BrickworkCircuit, "haar"),
        (HaarBrickworkCircuit, "haar"),
        (CliffordBrickworkCircuit, "clifford"),
    ):
        assert cls.__module__ == f"QELM.models.circuits.{module}"
        assert issubclass(cls, BrickworkCircuit)
        assert getattr(circuits, cls.__name__) is cls
    # Haar and Clifford gates conserve no charge: they are not U(1) circuits.
    assert not issubclass(HaarBrickworkCircuit, U1BrickworkCircuit)
    assert not issubclass(CliffordBrickworkCircuit, U1BrickworkCircuit)
    for name in circuits.__all__:
        assert hasattr(circuits, name)


def test_base_brickwork_has_no_gate_of_its_own() -> None:
    """The base class only knows the geometry: drawing a gate is left to the
    subclasses, and a circuit without layers needs none."""
    geometry = BrickworkCircuit({"N": 4, "Nr": 1, "depth": 0})
    assert geometry.bonds(0) == [(1, 2), (3, 4)]
    with pytest.raises(NotImplementedError, match="draw_gate"):
        BrickworkCircuit({"N": 4, "Nr": 1, "depth": 1})
