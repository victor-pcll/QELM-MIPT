"""The QELM experiment on the measurement record, experiments/QELM_M/exp_QELM_M.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config
from QELM.embedding import AncillaChargeSectorEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M.py"
L, N = 50, 4
OVERRIDES = ["models=TFIM", f"N={N}", f"samples.L={L}", "training.alpha=1e-3"]


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
    # One draw per sample, measurement step and system qubit.
    assert np.mean(features == 0.0) == pytest.approx(1 - p, abs=0.1)
    assert np.all(np.isfinite(out["W"]))
    assert np.all(np.isfinite(out["y_pred"]))
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


def test_every_sample_starts_from_the_embedded_state(monkeypatch) -> None:
    """Each sample evolves the embedded state of AncillaChargeSectorEmbedding,
    not the collapsed state left by the previous sample (issue #101)."""
    module = _load_script()
    cfg = load_config(OVERRIDES)
    rho0 = AncillaChargeSectorEmbedding.from_config(cfg).rho()
    seen: list[np.ndarray] = []
    evolve = module.TransverseFieldIsingModel.evolve

    def spy(self, rho, *args, **kwargs):
        seen.append(np.array(rho, copy=True))
        return evolve(self, rho, *args, **kwargs)

    monkeypatch.setattr(module.TransverseFieldIsingModel, "evolve", spy)
    module.run(cfg)
    n_meas = N - 1
    assert len(seen) == L * n_meas
    # The first step of each sample evolves the embedded state.
    for rho in seen[::n_meas]:
        np.testing.assert_allclose(rho, rho0, atol=1e-12)


def test_record_has_one_row_per_measurement_step(monkeypatch) -> None:
    """M[i] holds n_meas separate measurements, each after a step of
    t / n_meas, not one measurement copied N - 1 times (issue #102)."""
    module = _load_script()
    n_meas = 2
    cfg = load_config([*OVERRIDES, f"+experiment.n_measurements={n_meas}"])
    steps: list[float] = []
    evolve = module.TransverseFieldIsingModel.evolve

    def spy(self, rho, *args, dt=None, **kwargs):
        steps.append(dt)
        return evolve(self, rho, *args, dt=dt, **kwargs)

    monkeypatch.setattr(module.TransverseFieldIsingModel, "evolve", spy)
    out = module.run(cfg)
    assert out["M"].shape == (L, n_meas, N - 1)
    assert out["X"].shape == (L, n_meas * (N - 1) + 1)
    np.testing.assert_array_equal(out["X"][:, :-1], out["M"].reshape(L, -1))
    assert steps == [pytest.approx(float(cfg.evolution.t) / n_meas)] * (L * n_meas)
    # Evolution between the measurements changes the outcomes for some samples.
    assert np.any(out["M"][:, 0] != out["M"][:, 1])


def test_n_measurements_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_measurements"):
        _load_script().run(load_config([*OVERRIDES, "+experiment.n_measurements=0"]))
