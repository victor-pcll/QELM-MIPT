"""The local Floquet Ising chain of Skinner, Ruhman and Nahum (2019), issue #239."""

from __future__ import annotations

import numpy as np
import pytest

from QELM import load_config
from QELM.evaluation.entropy import von_neumann_entropy
from QELM.models.circuits import (
    SKINNER_2019,
    FloquetBrickworkCircuit,
    apply_layer,
    floquet_ising_gate,
)
from QELM.physics import X, Z, get_site_operator

I2 = np.eye(2)


def _rotation(angle: float, P: np.ndarray) -> np.ndarray:
    """``exp(-i angle P)`` for a Pauli string ``P`` (``P^2 = 1``)."""
    return np.cos(angle) * np.eye(P.shape[0]) - 1j * np.sin(angle) * P


def test_gate_is_eq_1_of_the_paper() -> None:
    """The brick equals Eq. (1) written term by term: in each factor the
    coupling and the two fields commute, so each exponential is a product of
    Pauli rotations, the Z factor acting first."""
    c = SKINNER_2019
    u_x = (
        _rotation(c["phi_xx"], np.kron(X, X))
        @ _rotation(c["theta_x"], np.kron(X, I2))
        @ _rotation(c["theta_x"], np.kron(I2, X))
    )
    u_z = (
        _rotation(c["phi_zz"], np.kron(Z, Z))
        @ _rotation(c["theta_z"], np.kron(Z, I2))
        @ _rotation(c["theta_z"], np.kron(I2, Z))
    )
    gate = floquet_ising_gate(**c)
    np.testing.assert_allclose(gate, u_x @ u_z, atol=1e-12)
    np.testing.assert_allclose(gate @ gate.conj().T, np.eye(4), atol=1e-12)
    assert SKINNER_2019 == {
        "phi_xx": 0.3,
        "phi_zz": 0.6,
        "theta_x": 0.2,
        "theta_z": 0.4,
    }


def test_longitudinal_and_coupling_terms_of_the_z_factor() -> None:
    """Without the X factor the brick is diagonal, with the phases of
    ``phi_zz z_x z_y + theta_z (z_x + z_y)`` on ``|00>, |01>, |10>, |11>``."""
    phi, theta = 0.6, 0.4
    gate = floquet_ising_gate(0.0, phi, 0.0, theta)
    energies = [phi + 2 * theta, -phi, -phi, phi - 2 * theta]
    np.testing.assert_allclose(gate, np.diag(np.exp(-1j * np.array(energies))))
    # The longitudinal field alone splits |00> from |11>.
    only_field = floquet_ising_gate(0.0, 0.0, 0.0, theta)
    np.testing.assert_allclose(
        np.diag(only_field), np.exp(-1j * np.array([2, 0, 0, -2]) * theta)
    )


def test_two_site_spectrum() -> None:
    """Without the Z factor, the brick is diagonal in the X basis, with the
    eigenphases of ``phi_xx x_1 x_2 + theta_x (x_1 + x_2)``, x = +-1."""
    phi, theta = 0.3, 0.2
    gate = floquet_ising_gate(phi, 0.0, theta, 0.0)
    expected = sorted(
        np.angle(np.exp(-1j * (phi * a * b + theta * (a + b))))
        for a in (1, -1)
        for b in (1, -1)
    )
    np.testing.assert_allclose(sorted(np.angle(np.linalg.eigvals(gate))), expected)


def test_no_charge_is_conserved() -> None:
    """The transverse field moves weight between Z sectors, even between Z
    parities: neither the charge nor its parity is conserved."""
    gate = floquet_ising_gate(**SKINNER_2019)
    # |00> (even parity) reaches |01> and |10> (odd parity).
    assert abs(gate[1, 0]) > 1e-3 and abs(gate[2, 0]) > 1e-3
    parity = np.diag([1, -1, -1, 1])
    assert np.linalg.norm(gate @ parity - parity @ gate) > 1e-3


def test_layers_are_alternating_nearest_neighbour_half_layers() -> None:
    """Layer t holds the even bonds for even t, the odd ones for odd t, every
    bond (i, i + 1) of the system qubits once per period, the ancilla never."""
    circuit = FloquetBrickworkCircuit({"N": 7, "Nr": 1, "depth": 4})
    bonds = [[(i, j) for i, j, _ in layer] for layer in circuit.layers]
    assert bonds[0] == bonds[2] == [(1, 2), (3, 4), (5, 6)]
    assert bonds[1] == bonds[3] == [(2, 3), (4, 5)]
    assert sorted(bonds[0] + bonds[1]) == [(i, i + 1) for i in range(1, 6)]
    assert circuit.layers_per_period == 2
    for layer in circuit.layers:
        for i, j, gate in layer:
            assert j == i + 1 and i >= circuit.Nr
            np.testing.assert_array_equal(gate, circuit.gate)
    periodic = FloquetBrickworkCircuit(
        {"N": 6, "Nr": 0, "depth": 2, "boundary": "periodic"}
    )
    assert [(i, j) for i, j, _ in periodic.layers[1]] == [(1, 2), (3, 4), (5, 0)]


def test_layer_matches_the_full_operator() -> None:
    """A half-layer is the product of the bricks on its bonds, written with
    full 2**N operators (a different code path from apply_layer)."""
    N = 4
    circuit = FloquetBrickworkCircuit({"N": N, "Nr": 0, "depth": 2})
    c = SKINNER_2019

    def brick(i: int) -> np.ndarray:
        def site(P: np.ndarray, k: int) -> np.ndarray:
            return np.asarray(get_site_operator(P, k, N))

        xx = site(X, i) @ site(X, i + 1)
        zz = site(Z, i) @ site(Z, i + 1)
        u_x = (
            _rotation(c["phi_xx"], xx)
            @ _rotation(c["theta_x"], site(X, i))
            @ _rotation(c["theta_x"], site(X, i + 1))
        )
        u_z = (
            _rotation(c["phi_zz"], zz)
            @ _rotation(c["theta_z"], site(Z, i))
            @ _rotation(c["theta_z"], site(Z, i + 1))
        )
        return u_x @ u_z

    rng = np.random.default_rng(0)
    psi = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    psi /= np.linalg.norm(psi)
    even = brick(0) @ brick(2)
    odd = brick(1)
    np.testing.assert_allclose(circuit.apply_layer(psi, 0), even @ psi, atol=1e-12)
    np.testing.assert_allclose(circuit.apply_layer(psi, 1), odd @ psi, atol=1e-12)


def test_deterministic_and_reads_its_config() -> None:
    """No randomness: redraw keeps the gates. The config file holds the
    values of the paper, and from_config reads them."""
    cfg = load_config(
        ["models=floquet_circuit", "N=6", "Nr=0", "+floquet_circuit.depth=3"]
    )
    assert cfg.models == "floquet_circuit"
    block = {k: cfg.floquet_circuit[k] for k in SKINNER_2019}
    assert block == SKINNER_2019
    circuit = FloquetBrickworkCircuit.from_config(cfg)
    assert circuit.Ns == 6 and len(circuit.layers) == 3
    before = [g.copy() for layer in circuit.layers for _, _, g in layer]
    circuit.redraw(np.random.default_rng(1))
    after = [g for layer in circuit.layers for _, _, g in layer]
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="scrambling"):
        FloquetBrickworkCircuit({"N": 4, "depth": 1, "scrambling_layers": 1})


def test_volume_law_without_measurements() -> None:
    """Chaotic dynamics: from the Néel state, 2 N periods without measurement
    bring the half-chain entropy close to the Page value N/2 - 1/(2 ln 2)."""
    N = 8
    circuit = FloquetBrickworkCircuit({"N": N, "Nr": 0, "depth": 4 * N})
    psi = np.zeros(2**N, dtype=complex)
    psi[int("01" * (N // 2), 2)] = 1.0
    for layer in circuit.layers:
        psi = apply_layer(psi, layer)
    page = N / 2 - 1 / (2 * np.log(2))
    S = von_neumann_entropy(psi, list(range(N // 2)))
    assert S == pytest.approx(page, abs=0.5)
