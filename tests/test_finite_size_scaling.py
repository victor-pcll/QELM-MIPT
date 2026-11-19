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
    hh_quality,
    hh_terms,
    quality_map,
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


def test_subtracted_collapse_of_a_log_growing_order_parameter() -> None:
    """S = log2(N) + G(x): only the subtracted collapse recovers (p_c, nu)."""
    p = np.linspace(0.10, 0.22, 13)
    curves = {}
    for N in (16, 32, 64):
        x = (p - 0.16) * N ** (1 / 1.3)
        y = np.log2(N) - 2 * np.tanh(x)
        curves[N] = (p, y, np.full_like(p, 0.01))
    assert collapse_quality(curves, 0.16, 1.3, subtract=True) < 0.5
    assert collapse_quality(curves, 0.16, 1.3) > 10
    fit = fit_collapse(curves, 0.15, 1.0, n_boot=0, subtract=True)
    assert fit.p_c == pytest.approx(0.16, abs=2e-3)
    assert fit.nu == pytest.approx(1.3, abs=0.1)


def test_quality_map_is_collapse_quality_on_the_grid() -> None:
    curves = _synthetic(0.16, 1.3, noise=0.005)
    p_c = np.array([0.14, 0.16, 0.18])
    nu = np.array([0.8, 1.3, 2.0])
    q = quality_map(curves, p_c, nu)
    assert q.shape == (3, 3)
    assert q[0, 2] == collapse_quality(curves, 0.14, 2.0)
    assert np.unravel_index(np.argmin(q), q.shape) == (1, 1)


def test_hh_terms_match_a_hand_computed_case() -> None:
    # nu = 1 and p of size 2 halved: both sizes sit at x = 0, 1, 2. With two
    # sizes, the master curve of a point is the line through the two points
    # of the other size that bracket it, so every term is known by hand.
    y1, y2 = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.5, 2.0])
    curves = {
        1: (np.array([0.0, 1.0, 2.0]), y1, np.full(3, 0.1)),
        2: (np.array([0.0, 0.5, 1.0]), y2, np.full(3, 0.1)),
    }
    terms = hh_terms(curves, 0.0, 1.0)
    # x = 0 is not bracketed (no point strictly to its left). At x = 1 the
    # line through (0, 0) and (1, y_other) gives Y = y_other, dY = 0.1:
    # term (1 - 1.5)^2 / 0.02 = 12.5. At x = 2, Y = 2 and the term is 0.
    assert sorted(terms) == pytest.approx([0.0, 0.0, 12.5, 12.5])
    assert hh_quality(curves, 0.0, 1.0) == pytest.approx(6.25)


def test_hh_quality_is_about_one_for_noisy_exact_scaling() -> None:
    noise = 0.01
    rng = np.random.default_rng(1)
    p = np.linspace(0.10, 0.22, 25)
    values = []
    for _ in range(20):
        curves = {}
        for N in (16, 32, 64):
            y = 0.5 * (1 - np.tanh((p - 0.16) * N ** (1 / 1.3)))
            curves[N] = (p, y + noise * rng.normal(size=p.size), np.full_like(p, noise))
        values.append(hh_quality(curves, 0.16, 1.3))
    assert 0.6 < float(np.mean(values)) < 1.6


def test_hh_quality_is_minimal_near_the_true_pair() -> None:
    curves = _synthetic(0.16, 1.3)
    # Exact scaling: only the linear interpolation of tanh is left at the
    # true pair, far below the quality at a wrong p_c or nu.
    best = hh_quality(curves, 0.16, 1.3)
    assert best < 5
    assert hh_quality(curves, 0.17, 1.3) > 50 * best
    assert hh_quality(curves, 0.16, 2.0) > 50 * best
    assert hh_quality(curves, 0.16, 0.0) == float("inf")
