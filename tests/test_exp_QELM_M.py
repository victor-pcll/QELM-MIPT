"""The QELM experiment on the measurement record, experiments/QELM_M/exp_QELM_M.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M.py"
L, N = 50, 4
OVERRIDES = ["models=TFIM", f"TFIM.N={N}", f"samples.L={L}", "training.alpha=1e-3"]


def _load_script() -> ModuleType:
    """Import experiments/QELM_M/exp_QELM_M.py, which is not a package."""
    spec = importlib.util.spec_from_file_location("exp_QELM_M", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_qubits_measured_at_p_1() -> None:
    """p = 1: every feature is an outcome +-1, plus the bias column."""
    out = _load_script().run(load_config(OVERRIDES))
    features = out["X"][:, :-1]
    assert out["X"].shape == (L, (N - 1) * (N - 1) + 1)
    np.testing.assert_array_equal(out["X"][:, -1], 1.0)
    assert set(np.unique(features)) <= {-1.0, 1.0}
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


@pytest.mark.parametrize("p", [0.5, 0.1])
def test_unmeasured_qubits_give_zero_features_not_nan(p: float) -> None:
    """p < 1: skipped qubits are 0 in X, so the fit and the metrics stay
    finite (issue #91), and the share of zeros follows 1 - p."""
    out = _load_script().run(load_config([*OVERRIDES, f"projective_measurement.p={p}"]))
    features = out["X"][:, :-1]
    assert np.all(np.isfinite(out["X"]))
    assert set(np.unique(features)) <= {-1.0, 0.0, 1.0}
    # One draw per sample and system qubit; the rows of M repeat it N - 1 times.
    draws = features[:, : N - 1]
    assert np.mean(draws == 0.0) == pytest.approx(1 - p, abs=0.15)
    assert np.all(np.isfinite(out["W"]))
    assert np.all(np.isfinite(out["y_pred"]))
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])
