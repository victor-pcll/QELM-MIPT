"""The reference-qubit probe on stabilizer tableaux (issue #185).

The stabilizer simulation is checked against the state vector: the same
gates of the library's Clifford group, on the same bonds, with the same
measured sites, must give the same entropy of R at every step.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from QELM.encoding import AncillaBellPairEncoding, get_encoding
from QELM.evaluation import von_neumann_entropy
from QELM.models.circuits import (
    apply_two_qubit_gate,
    clifford_gate_programs,
    clifford_group_2q,
    draw_probe_history,
    periodic_bonds,
    reference_entropy_history,
)
from QELM.models.circuits.stabilizer import ProbeHistory


def test_bell_pair_encoding() -> None:
    """(|0>|000> + |1>|010>) / sqrt(2) for N = 3, site 1: S_R = 1 bit."""
    emb = AncillaBellPairEncoding({"N": 3, "site": 1})
    psi = emb.state()
    expected = np.zeros(16, dtype=complex)
    expected[0b0000] = expected[0b1010] = 1 / np.sqrt(2)
    np.testing.assert_allclose(psi, expected)
    np.testing.assert_allclose(emb.state(sparse=True).toarray()[:, 0], expected)
    assert von_neumann_entropy(psi, [0]) == pytest.approx(1.0)
    # The site is pure in the pair, the other system qubits stay pure.
    assert von_neumann_entropy(psi, [2]) == pytest.approx(1.0)
    assert von_neumann_entropy(psi, [1]) == pytest.approx(0.0, abs=1e-9)
    assert AncillaBellPairEncoding({"N": 6}).site == 3
    assert get_encoding("ancilla_bell_pair") is AncillaBellPairEncoding
    with pytest.raises(ValueError):
        AncillaBellPairEncoding({"N": 3, "site": 3})


def test_periodic_bonds() -> None:
    assert periodic_bonds(6, 0) == [(0, 1), (2, 3), (4, 5)]
    assert periodic_bonds(6, 1) == [(1, 2), (3, 4), (5, 0)]
    with pytest.raises(ValueError):
        periodic_bonds(5, 0)


def test_gate_programs_are_the_library_gates() -> None:
    """Each stim program has the tableau of the library's unitary."""
    programs = clifford_gate_programs()
    group = clifford_group_2q()
    assert len(programs) == len(group) == 11520
    for k in range(0, 11520, 37):
        program = stim.Circuit(programs[k].format(a=0, b=1))
        expected = stim.Tableau.from_unitary_matrix(group[k], endian="big")
        assert program.to_tableau() == expected


def _statevector_entropies(history: ProbeHistory, rng: np.random.Generator) -> list:
    """Replay ``history`` on the state vector, with Born-rule outcomes."""
    N = history.N
    psi = AncillaBellPairEncoding({"N": N}).state()
    group = clifford_group_2q()
    out = []
    for t in range(history.T):
        for g, (a, b) in zip(history.gates[t], periodic_bonds(N, t % 2), strict=True):
            psi = apply_two_qubit_gate(psi, group[g], a + 1, b + 1)
        for k in np.flatnonzero(history.measured[t]):
            bit = (np.arange(psi.size) >> (N - 1 - k)) & 1
            p1 = float(np.sum(np.abs(psi[bit == 1]) ** 2))
            outcome = int(rng.random() < p1)
            psi = np.where(bit == outcome, psi, 0)
            psi /= np.linalg.norm(psi)
        out.append(von_neumann_entropy(psi, [0]))
    return out


@pytest.mark.parametrize("N", [2, 4, 6])
def test_stabilizer_matches_state_vector(N: int) -> None:
    rng = np.random.default_rng(N)
    for p in (0.0, 0.3, 0.7):
        history = draw_probe_history(N, p, 2 * N, 2 * N, rng)
        S_tab = reference_entropy_history(history, seed=1)
        S_vec = _statevector_entropies(history, rng)
        np.testing.assert_allclose(S_tab, S_vec, atol=1e-9)


def test_entropy_does_not_depend_on_outcomes() -> None:
    history = draw_probe_history(8, 0.2, 16, 16, np.random.default_rng(0))
    a = reference_entropy_history(history, seed=1)
    b = reference_entropy_history(history, seed=2)
    np.testing.assert_array_equal(a, b)


def test_limits() -> None:
    """p = 0: R stays mixed; p = 1: every site measured, R purified at once."""
    rng = np.random.default_rng(3)
    S0 = reference_entropy_history(draw_probe_history(16, 0.0, 32, 32, rng))
    np.testing.assert_array_equal(S0, 1.0)
    S1 = reference_entropy_history(draw_probe_history(16, 1.0, 32, 32, rng))
    np.testing.assert_array_equal(S1[:32], 1.0)
    np.testing.assert_array_equal(S1[32:], 0.0)


def test_history_draws() -> None:
    history = draw_probe_history(6, 0.5, 4, 3, np.random.default_rng(0))
    assert history.gates.shape == (7, 3) and history.measured.shape == (7, 6)
    assert not history.measured[:4].any()
    assert (history.N, history.T) == (6, 7)
    again = draw_probe_history(6, 0.5, 4, 3, np.random.default_rng(0))
    np.testing.assert_array_equal(history.gates, again.gates)
    with pytest.raises(ValueError):
        draw_probe_history(6, 1.5, 4, 3, np.random.default_rng(0))
    with pytest.raises(ValueError):
        draw_probe_history(6, 0.5, -1, 3, np.random.default_rng(0))
    with pytest.raises(ValueError):
        reference_entropy_history(history, site=6)
