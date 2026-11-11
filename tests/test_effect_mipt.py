"""Tests of the reference MIPT probes and the frame potential (effect_mipt)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.effect_mipt import (
    apply_gate,
    clifford_purification,
    draw_haar_circuit,
    frame_potential,
    frame_potential_exact,
    haar_reference_entropy,
    prob_zero,
    project,
)
from QELM.evaluation.effect_rank import brick_pairs
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
