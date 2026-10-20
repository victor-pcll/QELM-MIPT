"""The QELM experiment on the measurement record, experiments/QELM_M/exp_QELM_M.py."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import TransverseFieldIsingModel, load_config
from QELM.embedding import AncillaChargeSectorEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M.py"
BASH_SCRIPT = SCRIPT.with_suffix(".sh")
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


def _spy_run(monkeypatch, cfg):
    """Run the script and record each propagator U(dt) and each state
    handed to the projective measurement, in call order."""
    module = _load_script()
    propagators: list[tuple[float, np.ndarray]] = []
    measured: list[np.ndarray] = []
    propagator = module.TransverseFieldIsingModel.propagator
    measure = module.ProjectiveMeasurement.measure

    def spy_propagator(self, *, dt):
        U = propagator(self, dt=dt)
        propagators.append((dt, U))
        return U

    def spy_measure(self, rho, *args, **kwargs):
        measured.append(np.array(rho, copy=True))
        return measure(self, rho, *args, **kwargs)

    monkeypatch.setattr(module.TransverseFieldIsingModel, "propagator", spy_propagator)
    monkeypatch.setattr(module.ProjectiveMeasurement, "measure", spy_measure)
    return module.run(cfg), propagators, measured


def test_every_sample_starts_from_the_embedded_state(monkeypatch) -> None:
    """Each sample evolves the embedded state of AncillaChargeSectorEmbedding,
    not the collapsed state left by the previous sample (issue #101)."""
    cfg = load_config(OVERRIDES)
    rho0 = AncillaChargeSectorEmbedding.from_config(cfg).rho()
    _, propagators, measured = _spy_run(monkeypatch, cfg)
    n_meas = N - 1
    assert len(propagators) == L and len(measured) == L * n_meas
    # The first measurement of sample i sees U_i rho0 U_i^dagger.
    for (_, U), rho in zip(propagators, measured[::n_meas], strict=True):
        np.testing.assert_allclose(rho, U @ rho0 @ U.conj().T, atol=1e-12)


def test_record_has_one_row_per_measurement_step(monkeypatch) -> None:
    """M[i] holds n_meas separate measurements, each after a step of
    t / n_meas, not one measurement copied N - 1 times (issue #102)."""
    n_meas = 2
    cfg = load_config([*OVERRIDES, f"+experiment.n_measurements={n_meas}"])
    out, propagators, measured = _spy_run(monkeypatch, cfg)
    assert out["M"].shape == (L, n_meas, N - 1)
    assert out["X"].shape == (L, n_meas * (N - 1) + 1)
    np.testing.assert_array_equal(out["X"][:, :-1], out["M"].reshape(L, -1))
    # One U(t / n_meas) per sample, applied before each of the n_meas
    # measurements.
    dt = float(cfg.evolution.t) / n_meas
    assert [step for step, _ in propagators] == [pytest.approx(dt)] * L
    assert len(measured) == L * n_meas
    # Evolution between the measurements changes the outcomes for some samples.
    assert np.any(out["M"][:, 0] != out["M"][:, 1])


def test_cached_propagator_matches_evolve() -> None:
    """U rho U^dagger with U = propagator(dt) is what model.evolve(rho, dt)
    computes, so caching U per sample changes no result."""
    cfg = load_config(OVERRIDES)
    model = TransverseFieldIsingModel.from_config(cfg)
    rho = AncillaChargeSectorEmbedding.from_config(cfg).rho()
    dt = float(cfg.evolution.t) / (N - 1)
    U = model.propagator(dt=dt)
    np.testing.assert_allclose(
        U @ rho @ U.conj().T, model.evolve(rho, dt=dt), atol=1e-10
    )


def test_target_is_the_ancilla_expectation(monkeypatch) -> None:
    """Y[i] = Tr(Z_0 rho) on the final state of sample i (issue #95 only
    changes how it is computed)."""
    out, _, _ = _spy_run(monkeypatch, load_config(OVERRIDES))
    assert np.all(np.abs(out["Y"]) <= 1 + 1e-12)
    # At p = 1 the measured charge fixes the ancilla: <Z_0> = +-1 (issue #104).
    np.testing.assert_allclose(np.abs(out["Y"]), 1.0, atol=1e-10)


def test_p_override_sets_the_share_of_measured_qubits() -> None:
    """projective_measurement.p from the command line reaches the
    measurement: the share of non-zero features grows with p."""
    module = _load_script()
    shares = []
    for p in (0.2, 0.5, 0.9):
        cfg = load_config(
            [*OVERRIDES, "samples.L=200", f"projective_measurement.p={p}"]
        )
        assert cfg.projective_measurement.p == p
        features = module.run(cfg)["X"][:, :-1]
        shares.append(np.mean(features != 0.0))
        assert shares[-1] == pytest.approx(p, abs=0.06)
    assert shares == sorted(shares)


def test_reproducible_with_the_seed() -> None:
    module = _load_script()
    a = module.run(load_config([*OVERRIDES, "projective_measurement.p=0.5"]))
    b = module.run(load_config([*OVERRIDES, "projective_measurement.p=0.5"]))
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["Y"], b["Y"])


def test_bash_launcher_dry_run(tmp_path: Path) -> None:
    """Issue #94: the launcher runs end to end with L=10, N=3 and three
    probabilities, with another project's virtualenv active, and writes one
    run folder per p."""
    assert os.access(BASH_SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "10",
            "N_VALUES": "3",
            "P_VALUES": "0.2 0.5 1.0",
            "VIRTUAL_ENV": str(tmp_path / "other-project" / ".venv"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 1 x 1 x 3 runs" in result.stdout
    assert "does not match the project environment" not in result.stderr
    runs = sorted(tmp_path.glob("exp_QELM_M/*/*"))
    assert [run.name for run in runs] == [
        "N3_seed42_p0.2",
        "N3_seed42_p0.5",
        "N3_seed42_p1.0",
    ]
    for run in runs:
        assert (run / "metrics.json").is_file()
        assert (run / "exp_QELM_M.npz").is_file()
        with np.load(run / "exp_QELM_M.npz") as data:
            assert data["X"].shape == (10, 2 * 2 + 1)


def test_n_measurements_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_measurements"):
        _load_script().run(load_config([*OVERRIDES, "+experiment.n_measurements=0"]))
