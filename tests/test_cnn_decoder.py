"""The CNN decoder of the record, QELM.evaluation.cnn_decoder (#293)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from QELM.evaluation.cnn_decoder import (
    CNNConfig,
    decode_cnn_accuracy,
    forward,
    init_params,
    predict_sign,
    split,
    train_cnn,
)
from QELM.evaluation.dehghani_decoder import (
    Records,
    decode_accuracy,
    draw_fixed_circuit,
    simulate_records,
)

SMALL = CNNConfig(channels=(8, 8), hidden=16, max_epochs=40, patience=8, lr=3e-3)


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("T,N", [(12, 6), (5, 4), (1, 2)])
def test_forward_shape(T: int, N: int, pool: bool) -> None:
    config = CNNConfig(channels=(4, 4), hidden=8, pool=pool)
    params = init_params(T, N, config, jax.random.PRNGKey(0))
    x = jnp.asarray(np.random.default_rng(0).integers(-1, 2, (7, T, N)), jnp.float32)
    z = forward(params, x, pool)
    assert z.shape == (7, 3)
    assert np.all(np.isfinite(np.asarray(z)))


def test_periodic_in_space() -> None:
    """Without pooling, the trunk is equivariant to a cyclic shift of sites.

    A dense layer that only sums over sites makes the logits invariant: then
    shifting the record leaves them unchanged, which checks the periodic
    padding.
    """
    T, N = 6, 6
    config = CNNConfig(channels=(4, 4), hidden=5, pool=False)
    params = init_params(T, N, config, jax.random.PRNGKey(1))
    c = config.channels[-1]
    w1 = np.array(params[-2][0]).reshape(T, N, c, -1)
    w1[:] = w1[:, :1]  # same weights for every site
    params[-2] = (jnp.asarray(w1.reshape(T * N * c, -1)), params[-2][1])
    x = np.random.default_rng(2).integers(-1, 2, (4, T, N)).astype(np.float32)
    z = np.asarray(forward(params, jnp.asarray(x)))
    z_shift = np.asarray(forward(params, jnp.asarray(np.roll(x, 2, axis=2))))
    np.testing.assert_allclose(z, z_shift, rtol=1e-5, atol=1e-5)


def test_learns_a_two_site_parity_per_axis() -> None:
    """A synthetic XOR task: the sign is the product of two fixed outcomes,
    a different pair per axis. A linear readout is at chance; the CNN learns
    it."""
    rng = np.random.default_rng(3)
    n, T, N = 4000, 4, 4
    M = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, T, N))
    axis = rng.integers(1, 4, n).astype(np.int8)
    pairs = {1: ((0, 0), (1, 1)), 2: ((2, 3), (3, 0)), 3: ((1, 2), (2, 2))}
    sign = np.array(
        [M[i][pairs[a][0]] * M[i][pairs[a][1]] for i, a in enumerate(axis)],
        dtype=np.int8,
    )
    records = Records(M=M, axis=axis, sign=sign)
    out, log = decode_cnn_accuracy(records, seed=0, config=SMALL)
    assert out["acc_pure"] > 0.95
    assert log.n_params > 0 and log.best_epoch >= 1
    ridge = decode_accuracy(records, readouts=("ridge",), seed=0)
    assert ridge["ridge"]["acc_pure"] < 0.6


def test_reproducible_with_seed() -> None:
    N, T = 4, 8
    gates = draw_fixed_circuit(N, T, np.random.default_rng(0))
    records, _ = simulate_records(gates, N, 0.3, 300, np.random.default_rng(1))
    a, _ = decode_cnn_accuracy(records, seed=5, config=SMALL)
    b, _ = decode_cnn_accuracy(records, seed=5, config=SMALL)
    assert a == b
    assert 0.0 <= a["acc_pure"] <= 1.0
    f = records.purified[~split(300, 0.7, 5)].mean()
    assert a["acc_all"] == pytest.approx(f * a["acc_pure"] + (1 - f) / 2)


def test_same_split_as_decode_accuracy() -> None:
    """The CNN is scored on the purified test trajectories of decode_accuracy."""
    N, T = 4, 8
    gates = draw_fixed_circuit(N, T, np.random.default_rng(0))
    records, _ = simulate_records(gates, N, 0.5, 200, np.random.default_rng(2))
    cnn, _ = decode_cnn_accuracy(records, seed=7, config=SMALL)
    lin = decode_accuracy(records, readouts=("ridge",), seed=7)
    assert cnn["n_test_pure"] == lin["ridge"]["n_test_pure"]
    assert cnn["n_train_pure"] == np.sum(records.purified[split(200, 0.7, 7)])


def test_max_train_and_trivial_cases() -> None:
    N, T = 4, 8
    gates = draw_fixed_circuit(N, T, np.random.default_rng(0))
    # p = 1: every site measured at every step, R purified after the first
    # step, always with the same measured sites. For this circuit the sign
    # is a parity of two or more outcomes (the ridge is at chance), which
    # the CNN learns.
    records, _ = simulate_records(gates, N, 1.0, 2000, np.random.default_rng(3))
    assert records.purified.all()
    out, _ = decode_cnn_accuracy(records, seed=0, config=SMALL)
    assert out["acc_pure"] > 0.95
    ridge = decode_accuracy(records, readouts=("ridge",), seed=0)
    assert ridge["ridge"]["acc_pure"] < 0.6
    small, _ = decode_cnn_accuracy(records, seed=0, config=SMALL, max_train=50)
    assert small["n_train_pure"] == 50
    # p = 0: R stays mixed, nothing to train or test.
    mixed, _ = simulate_records(gates, N, 0.0, 50, np.random.default_rng(4))
    out, log = decode_cnn_accuracy(mixed, seed=0, config=SMALL)
    assert np.isnan(out["acc_pure"]) and out["acc_all"] == 0.5
    assert log.epochs == 0


def test_predict_sign_values() -> None:
    T, N = 4, 4
    params, _ = train_cnn(
        np.zeros((0, T, N), np.int8), np.zeros(0, np.int8), np.zeros(0, np.int8)
    )
    M = np.random.default_rng(0).integers(-1, 2, (10, T, N)).astype(np.int8)
    pred = predict_sign(params, M, np.full(10, 2))
    assert set(np.unique(pred)) <= {-1, 1}
