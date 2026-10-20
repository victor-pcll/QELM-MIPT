"""Moments and Binder ratio of QELM.evaluation.charge_sharpening (issue #137)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation import (
    binder_ratio,
    bootstrap_errors,
    central_moment,
    sharpening_statistics,
)


def test_central_moments_match_numpy() -> None:
    rng = np.random.default_rng(0)
    S = rng.random(500)
    assert abs(central_moment(S, 1)) < 1e-12
    assert central_moment(S, 2) == pytest.approx(np.var(S))
    assert central_moment(S, 4) == pytest.approx(np.mean((S - S.mean()) ** 4))


def test_weights_equal_repeated_samples() -> None:
    """Un poids 2 vaut un échantillon répété deux fois."""
    S = np.array([0.0, 0.5, 1.0])
    w = np.array([2.0, 1.0, 1.0])
    repeated = np.array([0.0, 0.0, 0.5, 1.0])
    for k in (2, 3, 4):
        assert central_moment(S, k, w) == pytest.approx(central_moment(repeated, k))
    assert binder_ratio(S, w) == pytest.approx(binder_ratio(repeated))


def test_binder_known_values() -> None:
    """B = 2/3 pour deux pics égaux, ~0 pour une gaussienne, nan si constant."""
    assert binder_ratio(np.array([0.0, 1.0] * 50)) == pytest.approx(2 / 3)
    rng = np.random.default_rng(1)
    assert abs(binder_ratio(rng.standard_normal(200000))) < 0.02
    # Uniform distribution: mu4 / mu2**2 = 9/5, so B = 1 - 3/5 = 2/5.
    assert binder_ratio(rng.random(200000)) == pytest.approx(0.4, abs=0.01)
    assert np.isnan(binder_ratio(np.ones(10)))


def test_bad_input_raises() -> None:
    with pytest.raises(ValueError):
        central_moment(np.array([]), 2)
    with pytest.raises(ValueError):
        central_moment(np.ones((2, 2)), 2)
    with pytest.raises(ValueError):
        central_moment(np.ones(3), 0)
    with pytest.raises(ValueError):
        central_moment(np.ones(3), 2, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        bootstrap_errors(np.ones(3), n_boot=1)


def test_statistics_and_bootstrap() -> None:
    rng = np.random.default_rng(2)
    S = rng.random(2000)
    stats = sharpening_statistics(S)
    assert stats["S_R_mean"] == pytest.approx(np.mean(S))
    assert stats["S_R_var"] == pytest.approx(np.var(S))
    errors = bootstrap_errors(S, n_boot=300, rng=np.random.default_rng(3))
    # The bootstrap error of the mean is the standard error, std / sqrt(n).
    assert errors["S_R_mean_err"] == pytest.approx(np.std(S) / np.sqrt(S.size), rel=0.2)
    assert errors["S_R_var_err"] > 0 and errors["binder_err"] > 0
