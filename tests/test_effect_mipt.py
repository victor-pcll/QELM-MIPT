"""Tests of the reference MIPT probes and the frame potential (effect_mipt)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.effect_mipt import (
    CliffordHalfLayers,
    apply_gate,
    clifford_purification,
    clifford_symplectic,
    dense_weight_bits,
    draw_haar_circuit,
    draw_open_half_layers,
    draw_periodic_half_layers,
    fourier_weight,
    frame_potential,
    frame_potential_exact,
    haar_reference_entropy,
    pauli_coefficients,
    prob_zero,
    project,
    sample_kraus,
)
from QELM.evaluation.effect_rank import PAULIS, brick_pairs
from QELM.models.circuits.base import apply_two_qubit_gate
from QELM.models.circuits.clifford import clifford_group_2q
from QELM.models.circuits.haar import haar_gate


def test_apply_gate_matches_library_convention() -> None:
    rng = np.random.default_rng(0)
    n = 4
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    gate = haar_gate(rng)
    np.testing.assert_allclose(
        apply_gate(psi, n, 1, gate), apply_two_qubit_gate(psi, gate, 1, 2), atol=1e-12
    )


def test_projectors_sum_to_identity_on_batches() -> None:
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(3, 8)) + 0j
    total = project(psi, 3, 1, 0) + project(psi, 3, 1, 1)
    np.testing.assert_allclose(total, psi)


@pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 1.0])
def test_frame_potential_sampler_matches_enumeration(p: float) -> None:
    n = 3
    rng = np.random.default_rng((2, int(10 * p)))
    gates = draw_haar_circuit(n, n, rng)
    masks = np.random.default_rng(5).random((n, n)) < p
    exact = frame_potential_exact(gates, masks)
    est = frame_potential(gates, masks, 2000, rng)
    assert abs(est.phi2 - exact) < 4 * est.sem + 1e-9


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_frame_potential_of_one_basis_is_two_to_minus_n(p: float) -> None:
    # Without useful mid-circuit records the effects are a single basis.
    n = 3
    gates = draw_haar_circuit(n, n, np.random.default_rng(3))
    masks = np.full((n, n), p > 0.5)
    assert frame_potential_exact(gates, masks) == pytest.approx(2.0**-n)


def test_frame_potential_is_reproducible() -> None:
    gates = draw_haar_circuit(3, 3, np.random.default_rng(4))
    masks = np.random.default_rng(6).random((3, 3)) < 0.4
    a = frame_potential(gates, masks, 50, np.random.default_rng(7))
    b = frame_potential(gates, masks, 50, np.random.default_rng(7))
    assert a == b


def test_clifford_limits() -> None:
    n = 6
    rng = np.random.default_rng(8)
    free = clifford_purification(n, np.zeros((3 * n, n), dtype=bool), rng)
    np.testing.assert_array_equal(free.system, n)
    np.testing.assert_array_equal(free.reference, 1)
    full = clifford_purification(n, np.ones((2, n), dtype=bool), rng)
    np.testing.assert_array_equal(full.system, 0)
    np.testing.assert_array_equal(full.reference, 0)


def test_clifford_system_entropy_matches_state_vector() -> None:
    # Replay the same gates (same generator draws) on the state vector of
    # the system and its n ancillas, outcomes drawn with the Born rule.
    n = 3
    masks = np.random.default_rng(9).random((2 * n, n)) < 0.35
    res = clifford_purification(n, masks, np.random.default_rng(10))
    group = clifford_group_2q()
    ids = np.random.default_rng(10).integers(
        len(group), size=(2 * n, len(brick_pairs(n)))
    )
    m = 2 * n
    psi = np.zeros(2**m, dtype=complex)
    for x in range(2**n):
        psi[(x << n) | x] = 2 ** (-n / 2)
    rng = np.random.default_rng(12)
    for t in range(2 * n):
        for g, site in zip(ids[t], brick_pairs(n), strict=True):
            psi = apply_gate(psi, m, site, group[g])
        for site in np.flatnonzero(masks[t]):
            p0 = prob_zero(psi, m, int(site))
            o = 0 if rng.random() < p0 else 1
            psi = project(psi, m, int(site), o)
            psi /= np.linalg.norm(psi)
        r = psi.reshape(2**n, 2**n)
        lam = np.linalg.eigvalsh(r @ r.conj().T)
        lam = lam[lam > 1e-12]
        assert -(lam * np.log2(lam)).sum() == pytest.approx(res.system[t], abs=1e-8)


def test_haar_reference_limits() -> None:
    n = 5
    rng = np.random.default_rng(11)
    free = haar_reference_entropy(n, np.zeros((4, n), dtype=bool), rng)
    np.testing.assert_allclose(free, 1.0, atol=1e-9)
    full = haar_reference_entropy(n, np.ones((2, n), dtype=bool), rng)
    np.testing.assert_allclose(full, 0.0, atol=1e-6)


def test_pauli_coefficients_match_traces() -> None:
    rng = np.random.default_rng(13)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    op = a + a.conj().T
    c = pauli_coefficients(op)
    for idx in (0, 7, 27, 38, 63):
        i, j, k = idx // 16, (idx // 4) % 4, idx % 4
        string = np.kron(np.kron(PAULIS[i], PAULIS[j]), PAULIS[k])
        assert c[idx] == pytest.approx(np.trace(string @ op).real / 8)


@pytest.mark.parametrize("p", [0.2, 0.5])
def test_dense_fourier_weight_is_the_purity_of_the_effect(p: float) -> None:
    n = 4
    rng = np.random.default_rng((14, int(10 * p)))
    gates = draw_haar_circuit(n, n, rng)
    k = sample_kraus(gates, rng.random((n, n)) < p, rng)
    effect = k.conj().T @ k
    assert np.trace(effect).real == pytest.approx(1.0)
    purity = np.trace(effect @ effect).real / np.trace(effect).real ** 2
    w = fourier_weight(k)
    assert 1 + w.dense == pytest.approx(2**n * purity)
    assert 0 <= w.angle <= w.dense + 1e-12


def test_fourier_weight_limits() -> None:
    n = 4
    rng = np.random.default_rng(15)
    gates = draw_haar_circuit(n, n, rng)
    flat = fourier_weight(sample_kraus(gates, np.zeros((n, n), dtype=bool), rng))
    assert flat.dense == pytest.approx(0.0, abs=1e-10)
    assert flat.angle == pytest.approx(0.0, abs=1e-10)
    pure = fourier_weight(sample_kraus(gates, np.ones((n, n), dtype=bool), rng))
    assert 1 + pure.dense == pytest.approx(2.0**n)


def test_sample_kraus_follows_the_born_rule() -> None:
    # One measured site after one layer: outcome 0 with probability 1/2 for
    # the maximally mixed input, and the two effects sum to the identity / 2^N.
    n = 2
    gates = draw_haar_circuit(n, 1, np.random.default_rng(16))
    masks = np.array([[True, False]])
    rng = np.random.default_rng(17)
    effects = []
    for _ in range(400):
        k = sample_kraus(gates, masks, rng)
        effects.append(k.conj().T @ k)
    first = effects[0]
    same = np.mean([np.allclose(e, first) for e in effects])
    assert abs(same - 0.5) < 0.1
    other = next(e for e in effects if not np.allclose(e, first))
    np.testing.assert_allclose((first + other) / 2, np.eye(4) / 4, atol=1e-12)


# ---------------------------------------------------------------------------
# Dense-encoding Fourier weight of Clifford circuits (#327)
# ---------------------------------------------------------------------------


def _explicit_kraus(
    circuit: CliffordHalfLayers, rng: np.random.Generator
) -> np.ndarray:
    """Kraus operator of one Born-drawn record, by dense matrices (small ``n``)."""
    n, dim = circuit.n, 2**circuit.n
    group = clifford_group_2q()
    # Axes 0..n-1: output qubits (qubit 0 most significant); last axis: input.
    t = (np.eye(dim, dtype=complex) / np.sqrt(dim)).reshape([2] * n + [dim])
    for bonds, gate_ids, sites in zip(
        circuit.bonds, circuit.gates, circuit.measured, strict=True
    ):
        for (a, b), g in zip(bonds.tolist(), gate_ids.tolist(), strict=True):
            u = group[g].reshape(2, 2, 2, 2)
            t = np.moveaxis(np.tensordot(u, t, axes=([2, 3], [a, b])), [0, 1], [a, b])
        for j in sites.tolist():
            p0 = float(np.sum(np.abs(np.take(t, 0, axis=j)) ** 2))
            o = 0 if rng.random() < p0 else 1
            index: list = [slice(None)] * (n + 1)
            index[j] = 1 - o
            t = t.copy()
            t[tuple(index)] = 0
            t = t / np.sqrt(p0 if o == 0 else 1 - p0)
    return t.reshape(dim, dim)


def test_clifford_symplectic_preserves_the_symplectic_form() -> None:
    sym = clifford_symplectic().astype(int)
    omega = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
    for k in (0, 1, 777, 11519):
        np.testing.assert_array_equal((sym[k] @ omega @ sym[k].T) % 2, omega)
    assert len({m.tobytes() for m in clifford_symplectic()}) == 720


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("p", [0.15, 0.4, 0.8])
def test_dense_weight_bits_matches_pauli_coefficients(n: int, p: float) -> None:
    rng = np.random.default_rng((n, int(100 * p)))
    circuits = [
        draw_periodic_half_layers(n, p, 6, rng),
        draw_open_half_layers(n, rng.random((3, n)) < p, rng),
    ]
    for circuit in circuits:
        k = dense_weight_bits(circuit, [len(circuit.bonds)])[0]
        for _ in range(2):  # k does not depend on the outcomes
            w = fourier_weight(_explicit_kraus(circuit, rng)).dense
            assert np.log2(1 + w) == pytest.approx(k, abs=1e-9)


def test_dense_weight_bits_limits_and_checkpoints() -> None:
    rng = np.random.default_rng(3)
    n = 8
    assert dense_weight_bits(
        draw_periodic_half_layers(n, 0.0, 10, rng), [0, 10]
    ).tolist() == [0, 0]
    assert dense_weight_bits(
        draw_periodic_half_layers(n, 1.0, 3, rng), [1, 3]
    ).tolist() == [n, n]
    ks = dense_weight_bits(draw_periodic_half_layers(n, 0.3, 40, rng), [5, 10, 20, 40])
    assert np.all((ks >= 0) & (ks <= n))
