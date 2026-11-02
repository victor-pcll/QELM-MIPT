"""The monitored Haar brickwork on the state vector (issue #257)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.models.circuits.monitored import (
    haar_reference_entropy,
    haar_steady_state_entropies,
    measure_z,
)


def test_measure_z_born_rule_and_collapse() -> None:
    # (sqrt(0.2)|0> + sqrt(0.8)|1>) |0>
    psi = np.zeros(4, dtype=complex)
    psi[0b00], psi[0b10] = np.sqrt(0.2), np.sqrt(0.8)
    rng = np.random.default_rng(0)
    ones = 0
    for _ in range(2000):
        out, outcome = measure_z(psi, 0, rng)
        ones += outcome
        assert np.linalg.norm(out) == pytest.approx(1.0)
        assert abs(out[0b10 if outcome else 0b00]) == pytest.approx(1.0)
    assert ones / 2000 == pytest.approx(0.8, abs=0.03)


def test_reference_limits() -> None:
    rng = np.random.default_rng(1)
    zero = haar_reference_entropy(6, 0.0, 12, 12, rng)
    assert zero["S_Q"] == pytest.approx(1.0) and zero["S_Q2"] == pytest.approx(1.0)
    one = haar_reference_entropy(6, 1.0, 12, 12, rng)
    assert one["S_Q"] == pytest.approx(0.0, abs=1e-9)


def test_steady_state_limits_and_renyi_order() -> None:
    rng = np.random.default_rng(2)
    one = haar_steady_state_entropies(8, 1.0, 8, rng)
    for value in one.values():
        assert value == pytest.approx(0.0, abs=1e-9)
    zero = haar_steady_state_entropies(8, 0.0, 32, rng)
    # Volume law: close to the Page value, and S_2 <= S_1.
    assert 3.0 < zero["S_half"] <= 4.0
    assert zero["S_half2"] <= zero["S_half"] + 1e-12
    assert zero["I3"] < 0
    with pytest.raises(ValueError):
        haar_steady_state_entropies(6, 0.1, 4, rng)
