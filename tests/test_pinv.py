"""Pseudo-inverse readout ``PseudoInverseReadout`` (issue #181)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM import load_config
from QELM.training import PseudoInverseReadout, RidgeReadout, get_trainer


def _data(n: int, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=n)


def test_registered_and_selected_by_the_config() -> None:
    assert get_trainer("pinv") is PseudoInverseReadout
    cfg = load_config(["training/regression=pinv"])
    assert cfg.training.name == "pinv"
    assert cfg.training.rcond is None
    assert PseudoInverseReadout.from_config(cfg).rcond is None
    assert load_config(config_name="double_descent").training.name == "pinv"


def test_overdetermined_is_least_squares() -> None:
    phi, y = _data(50, 6)
    readout = PseudoInverseReadout()
    assert readout.fit(phi, y) == "fitted"
    expected = np.linalg.lstsq(phi, y, rcond=None)[0]
    np.testing.assert_allclose(readout.W, expected, atol=1e-12)


def test_underdetermined_interpolates_with_the_smallest_norm() -> None:
    phi, y = _data(5, 12)
    readout = PseudoInverseReadout()
    readout.fit(phi, y)
    assert readout.W is not None
    np.testing.assert_allclose(readout.predict(phi), y, atol=1e-10)
    # The minimum-norm solution lies in the row space of phi: adding any
    # null-space vector to it only makes it longer.
    null = np.linalg.svd(phi)[2][5:]
    np.testing.assert_allclose(null @ readout.W, 0.0, atol=1e-10)
    other = readout.W + null[0]
    np.testing.assert_allclose(phi @ other, y, atol=1e-10)
    assert np.linalg.norm(other) > np.linalg.norm(readout.W)


@pytest.mark.parametrize("n", [5, 50])
def test_limit_of_ridge_as_alpha_goes_to_zero(n: int) -> None:
    phi, y = _data(n, 12)
    pinv = PseudoInverseReadout()
    pinv.fit(phi, y)
    ridge = RidgeReadout()
    ridge.fit(phi, y, alpha=1e-9)
    np.testing.assert_allclose(ridge.W, pinv.W, atol=1e-5)


def test_rcond_cuts_small_singular_values() -> None:
    phi, y = _data(30, 4)
    noise = 1e-9 * np.random.default_rng(1).normal(size=(30, 1))
    phi = np.hstack([phi, phi[:, :1] + noise])  # nearly dependent column
    loose = PseudoInverseReadout({"rcond": 1e-6})
    loose.fit(phi, y)
    exact = PseudoInverseReadout()
    exact.fit(phi, y)
    assert np.linalg.norm(loose.W) < np.linalg.norm(exact.W)


def test_multiple_outputs_and_state_dict() -> None:
    phi, _ = _data(20, 3)
    Y = np.random.default_rng(1).normal(size=(20, 2))
    readout = PseudoInverseReadout()
    assert readout.state_dict()["weights"].shape == (0,)
    readout.fit(phi, Y)
    assert readout.predict(phi).shape == (20, 2)
    assert readout.state_dict()["weights"].shape == (3, 2)


def test_errors() -> None:
    with pytest.raises(ValueError, match="rcond"):
        PseudoInverseReadout({"rcond": -1})
    readout = PseudoInverseReadout()
    assert readout.fit(None, None) == "scaffold"
    with pytest.raises(RuntimeError, match="fitted"):
        readout.predict(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        readout.fit(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="rows"):
        readout.fit(np.ones((3, 2)), np.ones(4))
