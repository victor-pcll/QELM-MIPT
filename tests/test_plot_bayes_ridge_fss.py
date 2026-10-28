"""The finite-size scaling figures of the Bayes-Ridge pipeline (issue #233)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_Q" / "plot_bayes_ridge_fss.py"
EXP_Q = REPO_ROOT / "experiments" / "QELM_Q" / "exp_QELM_Q.py"
EXP_SEP = REPO_ROOT / "experiments" / "learnability" / "learnability.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_inverse_size_recovers_an_exact_law() -> None:
    N = np.array([6.0, 8.0, 10.0, 12.0])
    fit = _load().fit_inverse_size(N, 0.9 - 0.6 / N, [0.01] * 4)
    assert fit["A_inf"] == pytest.approx(0.9)
    assert fit["c"] == pytest.approx(-0.6)
    assert fit["chi2_dof"] == pytest.approx(0.0, abs=1e-12)
    # Zero errors (an exact accuracy of 1) are floored, not divided by.
    flat = _load().fit_inverse_size(N, [1.0] * 4, [0.0] * 4)
    assert flat["A_inf"] == pytest.approx(1.0)


def test_max_slope_and_crossings() -> None:
    module = _load()
    p = [0.0, 0.1, 0.2, 0.3]
    slope = module.max_slope(p, [0.5, 0.55, 0.8, 0.85])
    assert slope["slope"] == pytest.approx(2.5)
    assert slope["p_mid"] == pytest.approx(0.15)
    # The larger size is below, then above: one crossing between 0.1 and 0.2.
    small, large = [0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.9, 1.0]
    (cross,) = module.crossings(p, small, large, [0.01] * 4)
    assert cross == pytest.approx(0.15)
    # Within the errors: no crossing.
    assert module.crossings(p, small, large, [0.2] * 4) == []


def _run(script: Path, *overrides: str) -> None:
    subprocess.run(
        [sys.executable, str(script), *overrides], check=True, capture_output=True
    )


def test_figures_from_small_sweeps(tmp_path: Path) -> None:
    for N in (4, 5):
        for p in (0.0, 0.5, 1.0):
            _run(
                EXP_Q,
                f"N={N}",
                "samples.L=40",
                "samples.balanced=true",
                "projective_measurement.final_measurement=true",
                f"projective_measurement.p={p}",
                f"hydra.run.dir={tmp_path / 'tfim' / f'N{N}_p{p}'}",
            )
            _run(
                EXP_SEP,
                f"N={N - 1}",
                "samples.L=40",
                "learnability.n_boot=5",
                "learnability.batch=20",
                "learnability.depth=5",
                "learnability.measure_every=period",
                f"learnability.p={p}",
                f"hydra.run.dir={tmp_path / 'sep' / f'N{N - 1}_p{p}'}",
            )
    out = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "tfim"),
            "--sep",
            str(tmp_path / "sep"),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    for name in ("bayes_ridge_fss", "bayes_ridge_asymptotics"):
        assert (out / f"{name}.png").is_file()
    data = json.loads((out / "bayes_ridge_asymptotics.json").read_text())["extra"]
    tfim = data["TFIM Bayes-Ridge"]
    assert tfim["metric"] == "accuracy_M_bayes_ridge"
    assert tfim["N"] == [4, 5] and tfim["p"] == [0.0, 0.5, 1.0]
    # The SEP sweep is labeled by the TFIM size N = Ns + 1.
    (sep,) = [v for k, v in data.items() if k.startswith("SEP")]
    assert sep["N"] == [4, 5]


def test_refuses_a_sweep_without_the_final_measurement(tmp_path: Path) -> None:
    _run(EXP_Q, "N=4", "samples.L=20", f"hydra.run.dir={tmp_path / 'run'}")
    with pytest.raises(ValueError, match="final_measurement"):
        _load().tfim_summary(tmp_path)


def test_xxz_sweep_with_the_local_readout(tmp_path: Path) -> None:
    """models=XXZ names the panels; the local readout gets its own (#236)."""
    for N in (4, 5):
        for p in (0.0, 0.5, 1.0):
            _run(
                EXP_Q,
                "models=XXZ",
                f"N={N}",
                "samples.L=40",
                "samples.balanced=true",
                "projective_measurement.final_measurement=true",
                f"projective_measurement.p={p}",
                f"hydra.run.dir={tmp_path / 'xxz' / f'N{N}_p{p}'}",
            )
    module = _load()
    rows = module.tfim_summary(tmp_path / "xxz")
    assert {row["model"] for row in rows} == {"XXZ"}
    assert "accuracy_M_local_bayes_ridge_mean" in rows[0]
    # The averaged rows draw the local curve too.
    assert module.LOCAL_CURVE in module.reservoir_curves(rows)
    out = tmp_path / "figures"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "xxz"), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    data = json.loads((out / "bayes_ridge_asymptotics.json").read_text())["extra"]
    assert set(data) == {"XXZ Bayes-Ridge", "XXZ local Bayes-Ridge"}
    assert data["XXZ local Bayes-Ridge"]["metric"] == "accuracy_M_local_bayes_ridge"


def test_reservoir_curves_skip_a_missing_local_readout() -> None:
    module = _load()
    assert module.reservoir_curves([{"accuracy_M_local_bayes_ridge": 0.6}, {}]) == (
        module.TFIM_CURVES
    )
    assert module.reservoir_name([{"model": None}]) == "TFIM"
    with pytest.raises(ValueError, match="mixes"):
        module.reservoir_name([{"model": "TFIM"}, {"model": "XXZ"}])
