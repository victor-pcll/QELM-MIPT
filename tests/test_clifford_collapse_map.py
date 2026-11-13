"""Collapse-quality map of the Clifford transition runs (issue #346)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "clifford_collapse_map" / "collapse_map.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("collapse_map", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["collapse_map"] = module
    spec.loader.exec_module(module)
    return module


def _write_run(folder: Path, n: int, ratio: int, p_c: float, nu: float) -> None:
    """A transition file whose Delta scales exactly with (p_c, nu)."""
    p = np.round(np.arange(0.08, 0.301, 0.01), 2)
    x = (p - p_c) * n ** (1 / nu)
    delta = 2.0 * np.exp(-1.5 * x)
    delta[-1] = 0.0  # a point with no error bar, to be dropped
    rows = [
        {"p": float(pi), "delta": [float(d), 0.01 * float(d)]}
        for pi, d in zip(p, delta, strict=True)
    ]
    (folder / f"transition_N{n}_T{ratio * n}_s0.json").write_text(
        json.dumps({"per_p": rows})
    )


def test_load_curves_drops_zero_points_and_takes_the_log(tmp_path: Path) -> None:
    mod = _load()
    _write_run(tmp_path, 16, 2, 0.16, 1.3)
    case = mod.Case("t", "t", 2, (16,))
    p, y, e = mod.load_curves(tmp_path, case)[16]
    assert p.size == 22 and np.all(y > 0)
    log_case = mod.Case("t", "t", 2, (16,), log=True, p_window=(0.10, 0.20))
    p2, y2, e2 = mod.load_curves(tmp_path, log_case)[16]
    assert p2[0] == pytest.approx(0.10) and p2[-1] == pytest.approx(0.20)
    assert np.allclose(e2, 0.01)
    assert np.allclose(np.exp(y2), y[(p >= 0.0999) & (p <= 0.2001)])


def test_grid_minimum_recovers_synthetic_p_c_and_nu(tmp_path: Path) -> None:
    mod = _load()
    for n in (16, 32, 64):
        _write_run(tmp_path, n, 2, 0.16, 1.3)
    curves = mod.load_curves(tmp_path, mod.Case("t", "t", 2, (16, 32, 64)))
    from QELM.evaluation.finite_size_scaling import quality_map

    p_c = np.linspace(0.15, 0.17, 21)
    nu = np.linspace(1.0, 1.6, 31)
    best = mod.grid_minimum(quality_map(curves, p_c, nu), p_c, nu)
    assert best["p_c"] == pytest.approx(0.16, abs=1e-3)
    assert best["nu"] == pytest.approx(1.3, abs=0.03)
    # Linear interpolation of the exponential costs about half an error bar.
    assert best["quality"] < 1.0


def test_sources_and_committed_summary_agree() -> None:
    mod = _load()
    summary = json.loads((SCRIPT.parent / "figures" / "summary.json").read_text())
    for case in (mod.MAIN, *mod.VARIANTS):
        curves = mod.load_curves(SCRIPT.parent / "sources", case)
        assert sorted(curves) == list(case.sizes)
        fit = summary[case.name]["fit"]
        assert fit["quality"] == pytest.approx(
            mod.collapse_quality(curves, fit["p_c"], fit["nu"])
        )
        assert 0.155 < fit["p_c"] < 0.167 and 1.0 < fit["nu"] < 1.7
