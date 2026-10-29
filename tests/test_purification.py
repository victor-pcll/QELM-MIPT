"""The reference qubit of the purification task, QELM.evaluation.purification (#247)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.entropy import von_neumann_entropy
from QELM.evaluation.purification import (
    bayes_accuracy,
    entangle_reference,
    expected_accuracy,
    reference_bell_state,
    reference_bloch,
    reference_entropy,
    reference_rho,
)
from QELM.physics import X, Y, Z, get_site_operator


def _basis(N: int, bits: str) -> np.ndarray:
    psi = np.zeros(2**N, dtype=complex)
    psi[int(bits, 2)] = 1.0
    return psi


def test_bell_state_pairs_r_with_the_site() -> None:
    # System |010>, R paired with site 1: (|0>|000> + |1>|010>) / sqrt(2).
    psi = reference_bell_state(_basis(3, "010"), 1)
    assert psi.shape == (16,)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0)
    assert np.flatnonzero(psi).tolist() == [0b0000, 0b1010]
    # R is maximally mixed and maximally entangled with site 1 only (global
    # qubit 2, R being qubit 0).
    np.testing.assert_allclose(reference_rho(psi), np.eye(2) / 2, atol=1e-12)
    assert von_neumann_entropy(psi, [0]) == pytest.approx(1.0)
    assert von_neumann_entropy(psi, [0, 2]) == pytest.approx(0.0, abs=1e-9)
    assert von_neumann_entropy(psi, [2]) == pytest.approx(1.0)
    assert von_neumann_entropy(psi, [1]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("system", "site"),
    [
        (np.ones(8) / np.sqrt(8), 0),  # not a basis state
        (_basis(3, "000"), 3),  # site out of range
        (np.zeros(6), 0),  # not a power of 2
    ],
)
def test_bell_state_rejects_bad_inputs(system: np.ndarray, site: int) -> None:
    with pytest.raises(ValueError):
        reference_bell_state(system, site)


def test_bloch_vector_matches_the_pauli_operators() -> None:
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    expected = [
        float(np.real(psi.conj() @ get_site_operator(P, 0, 4) @ psi)) for P in (X, Y, Z)
    ]
    np.testing.assert_allclose(reference_bloch(psi), expected, atol=1e-12)
    rho = reference_rho(psi)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    assert np.trace(rho).real == pytest.approx(1.0)
    assert reference_entropy(reference_bloch(psi)) == pytest.approx(
        von_neumann_entropy(psi, [0]), abs=1e-9
    )


def test_bloch_vector_of_pure_reference_states() -> None:
    system = _basis(2, "01")
    for r_state, bloch in (
        (np.array([1, 0]), [0, 0, 1]),
        (np.array([0, 1]), [0, 0, -1]),
        (np.array([1, 1]) / np.sqrt(2), [1, 0, 0]),
        (np.array([1, 1j]) / np.sqrt(2), [0, 1, 0]),
    ):
        psi = np.kron(r_state, system)
        np.testing.assert_allclose(reference_bloch(psi), bloch, atol=1e-12)
        assert reference_entropy(np.asarray(bloch, dtype=float)) == pytest.approx(0.0)
    assert reference_entropy(np.zeros(3)) == pytest.approx(1.0)


def test_bloch_rejects_a_state_without_system() -> None:
    with pytest.raises(ValueError):
        reference_bloch(np.array([1.0, 0.0]))


def test_bayes_and_expected_accuracy() -> None:
    z = np.array([1.0, -1.0, 0.0, 0.5])
    assert bayes_accuracy(z) == pytest.approx((1 + 1 + 0.5 + 0.75) / 4)
    # The sign of z is the best guess; the opposite one is the worst.
    assert expected_accuracy(z, z) == pytest.approx(bayes_accuracy(z))
    assert expected_accuracy(z, -z - 1e-9) == pytest.approx(1 - bayes_accuracy(z))
    # A mixed reference: every guess is right half of the time.
    assert expected_accuracy(np.zeros(5), np.arange(5) - 2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        expected_accuracy(z, z[:2])


def test_entangle_reference_makes_a_bell_pair_with_the_site() -> None:
    rng = np.random.default_rng(3)
    system = rng.normal(size=16) + 1j * rng.normal(size=16)
    system /= np.linalg.norm(system)
    psi = np.kron([1.0, 0.0], system)
    for seed in range(6):
        out, m = entangle_reference(psi, 2, np.random.default_rng(seed))
        assert m in (1, -1)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0)
        # R maximally mixed, pure with its partner (state qubit 3), in
        # |00> + |11>: <Z_R Z_site> = +1 whatever the outcome.
        assert von_neumann_entropy(out, [0]) == pytest.approx(1.0)
        assert von_neumann_entropy(out, [0, 3]) == pytest.approx(0.0, abs=1e-9)
        ZZ = get_site_operator(Z, 0, 5) @ get_site_operator(Z, 3, 5)
        assert np.real(out.conj() @ ZZ @ out) == pytest.approx(1.0)


def test_entangle_reference_needs_r_in_zero() -> None:
    psi = np.kron([0.0, 1.0], _basis(2, "01"))
    with pytest.raises(ValueError, match="R must be in"):
        entangle_reference(psi, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="site"):
        entangle_reference(
            np.kron([1.0, 0.0], _basis(2, "01")), 2, np.random.default_rng(0)
        )
