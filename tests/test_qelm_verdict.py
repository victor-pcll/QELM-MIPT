"""Tests of experiments/qelm_verdict (issue #303): the crossing estimators,
the Bayes share and the figures drawn from the saved summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "qelm_verdict" / "plot_verdict.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_verdict", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sign_changes_interpolates_the_zero() -> None:
    m = _load()
    p = np.array([0.0, 0.1, 0.2, 0.3])
    d = np.array([-1.0, -0.5, 0.5, 1.0])
    assert m.sign_changes(p, d, 0.0, 1.0) == [0.15000000000000002]
    assert m.sign_changes(p, d, 0.25, 1.0) == []


def test_resolved_crossing_needs_two_sigma_on_both_sides() -> None:
    m = _load()
    p = np.linspace(0.05, 0.5, 10)
    a = p.copy()
    b = 0.55 - p  # crosses a at p = 0.275
    small, large = np.full(10, 0.01), np.full(10, 1.0)
    r = m.crossing(p, a, small, b, small)
    assert r["resolved"] is not None
    assert abs(r["resolved"]["p_star"] - 0.275) < 1e-9
    assert r["share_with_crossing"] > 0.99
    # same means, errors far larger than the gap: a crossing, not resolved
    r = m.crossing(p, a, large, b, large)
    assert r["mean_curve_crossings"] and r["resolved"] is None


def test_parallel_curves_have_no_crossing() -> None:
    m = _load()
    p = np.linspace(0.05, 0.5, 10)
    e = np.full(10, 1e-3)
    r = m.crossing(p, p, e, p + 0.1, e)
    assert r["mean_curve_crossings"] == []
    assert r["resolved"] is None
    assert r["share_with_crossing"] == 0.0


def test_skill_is_the_share_of_the_bayes_advantage() -> None:
    m = _load()
    acc = np.array([0.5, 0.6, 0.75])
    bayes = np.array([0.505, 0.7, 1.0])
    g, ge = m.skill(acc, np.zeros(3), bayes, np.zeros(3))
    assert np.isnan(g[0])  # Bayes within 0.01 of chance
    np.testing.assert_allclose(g[1:], [0.5, 0.5])
    np.testing.assert_allclose(ge[1:], [0.0, 0.0])


def test_figures_and_summary(tmp_path: Path) -> None:
    m = _load()
    import sys

    argv = sys.argv
    sys.argv = ["plot_verdict.py", "--out", str(tmp_path)]
    try:
        m.main()
    finally:
        sys.argv = argv
    for name in (
        "fig1_meme_circuit",
        "fig2_controles_u1",
        "fig3_retrecissement",
        "fig4_critere_fss",
    ):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
    s = json.loads((tmp_path / "verdict_summary.json").read_text())
    # the oracle S_R of the scrambled Clifford probe crosses near p_c = 0.16
    for r in s["clifford_state_readout"]["oracle_crossings"].values():
        assert r["resolved"] is not None
        assert 0.14 < r["resolved"]["p_star"] < 0.19
    # the linear pooled readout of the charge variance never crosses (flat)
    lin = [v for k, v in s["u1_charge_variance"].items() if "linear" in k][0]
    assert all(r["mean_curve_crossings"] == [] for r in lin.values())
    # the shrinkage slope of the pooled readout is well below 1
    for r in s["clifford_state_readout"]["shrinkage"].values():
        assert 0.5 < r["lambda"] < 0.9
