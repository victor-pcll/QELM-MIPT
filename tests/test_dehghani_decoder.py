"""Decoding the purified reference qubit, QELM.evaluation.dehghani_decoder (#287)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.dehghani_decoder import (
    decode_accuracy,
    decode_qelm_accuracy,
    draw_fixed_circuit,
    local_paulis,
    parity_weight_one,
    simulate_qelm_records,
    simulate_records,
)
from QELM.models.circuits.clifford import _phase_key, clifford_group_2q


def _gate_index(u: np.ndarray) -> int:
    """Index of the Clifford gate ``u`` in :func:`clifford_group_2q`."""
    keys = [_phase_key(g) for g in clifford_group_2q()]
    return keys.index(_phase_key(np.asarray(u, dtype=complex)))


H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I2 = np.eye(2)
# CNOT with control b (second qubit) and target a (first), big endian.
CNOT_BA = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def _identity_circuit(N: int, T: int) -> np.ndarray:
    return np.full((T, N // 2), _gate_index(np.eye(4)))


def test_shapes_values_and_seed() -> None:
    N, T = 6, 12
    gates = draw_fixed_circuit(N, T, np.random.default_rng(0))
    assert gates.shape == (T, N // 2)
    a = simulate_records(gates, N, 0.2, 50, np.random.default_rng(1))
    b = simulate_records(gates, N, 0.2, 50, np.random.default_rng(1))
    without, with_readout = a
    assert without.M.shape == (50, T, N)
    assert with_readout.M.shape == (50, T + 1, N)
    assert set(np.unique(without.M)) <= {-1, 0, 1}
    # The final readout measures every site: no zero in its row.
    assert np.all(with_readout.M[:, T] != 0)
    np.testing.assert_array_equal(with_readout.M[:, :T], without.M)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.M, y.M)
        np.testing.assert_array_equal(x.sign, y.sign)
    # The sign is +-1 exactly where R is pure.
    for rec in a:
        np.testing.assert_array_equal(rec.sign != 0, rec.purified)


def test_limits_of_p() -> None:
    N, T = 6, 12
    gates = draw_fixed_circuit(N, T, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    without, with_readout = simulate_records(gates, N, 0.0, 30, rng)
    assert not without.purified.any()  # no measurement: R stays entangled
    assert with_readout.purified.all()  # the readout leaves a product state
    without, _ = simulate_records(gates, N, 1.0, 30, rng)
    assert without.purified.all()  # every site measured at every step


def test_identity_circuit_copies_the_middle_outcome() -> None:
    # Identity gates: R stays paired with site N // 2, so its Z sign is the
    # first outcome of that site, a parity of weight one.
    N, T = 4, 4
    gates = _identity_circuit(N, T)
    rng = np.random.default_rng(5)
    without, with_readout = simulate_records(gates, N, 1.0, 40, rng)
    assert np.all(without.axis == 3)
    np.testing.assert_array_equal(without.sign, without.M[:, 0, N // 2])
    without, with_readout = simulate_records(gates, N, 0.0, 40, rng)
    assert np.all(with_readout.axis == 3)
    np.testing.assert_array_equal(with_readout.sign, with_readout.M[:, T, N // 2])
    measured = np.zeros((T, N), dtype=bool)
    measured[1, N // 2] = True
    assert parity_weight_one(gates, N, measured, rng) is True
    assert parity_weight_one(gates, N, np.zeros((T, N), dtype=bool), rng) is None


def test_parity_of_two_outcomes_is_detected() -> None:
    # (I x H) then CNOT from site 3 to site 2 on R, 2, 3 = (|00> + |11>)|0>:
    # Z_R = Z_2 Z_3 afterwards, so the sign is the product of two outcomes.
    N, T = 4, 1
    gates = np.array([[_gate_index(np.eye(4)), _gate_index(CNOT_BA @ np.kron(I2, H))]])
    measured = np.zeros((T, N), dtype=bool)
    measured[0, [2, 3]] = True
    rng = np.random.default_rng(6)
    assert parity_weight_one(gates, N, measured, rng) is False
    rec, _ = simulate_records(gates, N, 1.0, 40, rng)
    np.testing.assert_array_equal(rec.sign, rec.M[:, 0, 2] * rec.M[:, 0, 3])


def test_decoders_on_a_linear_and_a_parity_task() -> None:
    N, T = 4, 2
    rng = np.random.default_rng(7)
    # Weight one: both readouts are exact.
    rec, _ = simulate_records(_identity_circuit(N, T), N, 1.0, 400, rng)
    acc = decode_accuracy(rec, seed=0)
    for readout in ("ridge", "mlp", "bayes"):
        assert acc[readout]["acc_pure"] == pytest.approx(1.0)
        assert acc[readout]["acc_all"] == pytest.approx(1.0)
    # Weight two: the ridge is at chance, the MLP learns the XOR.
    gates = np.array([[_gate_index(np.eye(4)), _gate_index(CNOT_BA @ np.kron(I2, H))]])
    rec, _ = simulate_records(gates, N, 1.0, 600, rng)
    acc = decode_accuracy(rec, seed=0)
    assert acc["ridge"]["acc_pure"] < 0.65
    assert acc["mlp"]["acc_pure"] > 0.95


def test_bayes_ceiling_and_bad_inputs() -> None:
    N, T = 6, 12
    gates = draw_fixed_circuit(N, T, np.random.default_rng(8))
    rec, _ = simulate_records(gates, N, 0.15, 300, np.random.default_rng(9))
    acc = decode_accuracy(rec, readouts=("ridge",), seed=1)
    assert 0.5 <= acc["bayes"]["acc_all"] <= 1.0
    assert acc["ridge"]["acc_all"] <= acc["bayes"]["acc_all"] + 1e-12
    with pytest.raises(ValueError):
        simulate_records(gates, N, 1.5, 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        draw_fixed_circuit(5, 3, np.random.default_rng(0))


def test_qelm_features_carry_the_label_of_an_unmeasured_bell_pair() -> None:
    # Identity gates and p = 0: R stays in a Bell pair with the middle site,
    # so measuring sigma_P on R leaves <P> = +-label on that site: the pair
    # (|00> + |11>) / sqrt(2) has <XX> = <ZZ> = 1 and <YY> = -1.
    N, T = 4, 2
    rng = np.random.default_rng(10)
    rec = simulate_qelm_records(_identity_circuit(N, T), N, 0.0, 300, rng)
    F = len(local_paulis(N))
    assert F == 3 * N + 9 * N * (N - 1) // 2
    assert rec.features.shape == (300, 3, F)
    assert not rec.M.any() and not (rec.axis > 0).any()
    for a, corr in enumerate((1, -1, 1)):  # feature 3 * site + a: P_a on site
        np.testing.assert_array_equal(
            rec.features[:, a, 3 * (N // 2) + a], corr * rec.labels[:, a]
        )
    acc = decode_qelm_accuracy(rec, readouts=("ridge",), seed=0)
    assert acc["state/ridge"]["acc"] == pytest.approx(1.0)
    assert acc["M+state/ridge"]["acc"] == pytest.approx(1.0)
    assert acc["M/ridge"]["acc"] < 0.65
    assert acc["bayes_M"]["acc"] == pytest.approx(0.5)
