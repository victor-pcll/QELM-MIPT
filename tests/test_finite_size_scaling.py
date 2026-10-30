"""Crossings and data collapse of QELM.evaluation.finite_size_scaling (#185)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.finite_size_scaling import (
    collapse_quality,
    crossing,
    crossing_errors,
    crossings,
    fit_collapse,
)


def _synthetic(p_c: float, nu: float, noise: float = 0.0) -> dict:
    """O = (1 - tanh(x)) / 2 with x = (p - p_c) N**(1/nu), exact scaling."""
    rng = np.random.default_rng(0)
    p = np.linspace(0.10, 0.22, 13)
    curves = {}
    for N in (16, 32, 64):
        y = 0.5 * (1 - np.tanh((p - p_c) * N ** (1 / nu)))
        e = np.full_like(p, max(noise, 1e-3))
        curves[N] = (p, y + noise * rng.normal(size=p.size), e)
    return curves


def test_crossing() -> None:
    x = np.array([0.0, 1.0, 2.0])
    assert crossing(x, np.array([1.0, 0.0, -1.0]), np.zeros(3)) == pytest.approx(1.0)
    assert crossing(x, np.array([1.0, 0.5, 0.0]), np.full(3, 0.25)) == pytest.approx(
        1.5
    )
    assert crossing(x, np.ones(3), np.zeros(3)) is None


def test_crossings_of_exact_scaling_are_at_p_c() -> None:
    out = crossings(_synthetic(0.16, 1.3))
    assert set(out) == {(16, 32), (16, 64), (32, 64)}
    for value in out.values():
        assert value == pytest.approx(0.16, abs=2e-3)


def test_collapse_recovers_p_c_and_nu() -> None:
    curves = _synthetic(0.16, 1.3, noise=0.005)
    assert collapse_quality(curves, 0.16, 1.3) < collapse_quality(curves, 0.18, 1.3)
    assert collapse_quality(curves, 0.16, -1.0) == np.inf
    fit = fit_collapse(curves, 0.15, 1.0, n_boot=5)
    assert fit.p_c == pytest.approx(0.16, abs=3e-3)
    assert fit.nu == pytest.approx(1.3, abs=0.15)
    assert fit.quality < 3
    assert fit.p_c_err > 0
    with pytest.raises(ValueError):
        fit_collapse({16: curves[16]}, 0.16, 1.3)


def test_crossing_errors_grow_with_the_noise() -> None:
    small = crossing_errors(_synthetic(0.16, 1.3, noise=0.002), n_boot=50)
    large = crossing_errors(_synthetic(0.16, 1.3, noise=0.02), n_boot=50)
    assert set(small) == {(16, 32), (16, 64), (32, 64)}
    for pair in small:
        assert 0 < small[pair] < large[pair]
