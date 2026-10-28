"""Crossings and data collapse of the entropy curves (issue #239)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.finite_size import (
    collapse_cost,
    curve_crossings,
    fit_collapse,
    rescale,
)

P = np.linspace(0.0, 0.6, 25)


def _scaling_curves(p_c: float, nu: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``S = log2 N + F((p - p_c) N^(1/nu))``: the log of the critical point
    plus a scaling function, which the collapse must find exactly."""
    return {
        N: (P, np.log2(N) - 2.0 * np.tanh((P - p_c) * N ** (1 / nu)))
        for N in (6, 8, 10, 12)
    }


def test_curve_crossings() -> None:
    p = [0.0, 0.1, 0.2, 0.3]
    assert curve_crossings(p, [0, 0, 0, 0], [1, 0.5, -0.5, -1]) == pytest.approx([0.15])
    assert curve_crossings(p, [0, 0, 0, 0], [1, 0, -1, -2]) == [0.1]
    assert curve_crossings(p, [0, 0, 0, 0], [1, 2, 3, 4]) == []


def test_rescale_subtracts_the_critical_value() -> None:
    curves = _scaling_curves(0.2, 1.5)
    scaled = rescale(curves, 0.2, 1.5)
    for N, (x, y) in scaled.items():
        np.testing.assert_allclose(x, (P - 0.2) * N ** (1 / 1.5))
        np.testing.assert_allclose(y, -2.0 * np.tanh(x), atol=1e-12)


def test_collapse_cost_is_zero_at_the_true_parameters() -> None:
    curves = _scaling_curves(0.2, 1.5)
    exact = collapse_cost(curves, 0.2, 1.5)
    assert exact < 1e-4
    assert collapse_cost(curves, 0.35, 1.5) > 100 * exact
    assert collapse_cost(curves, 0.2, 4.0) > 100 * exact


def test_fit_collapse_recovers_p_c_and_nu() -> None:
    fit = fit_collapse(_scaling_curves(0.21, 1.8), p_range=(0.05, 0.45))
    assert fit["p_c"] == pytest.approx(0.21, abs=0.01)
    assert fit["nu"] == pytest.approx(1.8, rel=0.1)
    near = fit_collapse(_scaling_curves(0.21, 1.8), p_range=(0.05, 0.45), window=0.2)
    assert near["p_c"] == pytest.approx(0.21, abs=0.01)
    assert near["nu"] == pytest.approx(1.8, rel=0.1)
    with pytest.raises(ValueError, match="two sizes"):
        fit_collapse({6: (P, P)}, p_range=(0.1, 0.3))
