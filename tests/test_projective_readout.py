"""The projective-readout experiment in experiments/projective/projective_readout.py."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "projective" / "projective_readout.py"
BASH_SCRIPT = REPO_ROOT / "experiments" / "projective" / "projective_readout.sh"
# Virtualenv of another project, active in the shell that launches the script
# (issue #79): uv must neither use it nor warn about it.
FOREIGN_VENV = REPO_ROOT.parent / "other-project" / ".venv"
OVERRIDES = ["models=TFIM", "N=4", "samples.L=10"]


def _load_script() -> ModuleType:
    """Importe experiments/projective/projective_readout.py, qui n'est pas un paquet."""
    spec = importlib.util.spec_from_file_location("projective_readout", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_shapes_and_features() -> None:
    """Formes de Phi, Y, M, X ; mesures en +-1 ; dernière ligne de Phi = X."""
    cfg = load_config(OVERRIDES)
    out = _load_script().run(cfg)
    L, n_steps, N = 10, int(cfg.evolution.n_steps), 4
    assert out["Phi"].shape == (L, n_steps, N - 1)
    assert out["M"].shape == (L, n_steps - 1, N - 1)
    assert out["X"].shape == (L, N - 1)
    assert out["Y"].shape == (L,)
    assert set(np.unique(out["M"])) <= {-1.0, 1.0}
    np.testing.assert_array_equal(out["Phi"][:, :-1], out["M"])
    np.testing.assert_array_equal(out["Phi"][:, -1], out["X"])
    assert np.all(np.abs(out["Y"]) <= 1 + 1e-9)
    assert np.all(np.abs(out["X"]) <= 1 + 1e-9)
    n_test = L - int(cfg.samples.train_size)
    assert out["y_pred"].shape == out["y_test"].shape == (n_test,)
    assert out["W"].shape == (n_steps * (N - 1),)
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


def test_backaction_collapses_measured_qubits() -> None:
    """+experiment.backaction=false garde rho non mesuré : la cible Y change."""
    module = _load_script()
    with_ba = module.run(load_config(OVERRIDES))
    without = module.run(load_config([*OVERRIDES, "+experiment.backaction=false"]))
    assert not np.allclose(with_ba["Y"], without["Y"])


def test_run_is_reproducible_with_the_seed() -> None:
    """Même seed, mêmes résultats ; seed différente, mesures différentes."""
    module = _load_script()
    a = module.run(load_config(OVERRIDES))
    b = module.run(load_config(OVERRIDES))
    c = module.run(load_config([*OVERRIDES, "seed=1"]))
    np.testing.assert_array_equal(a["Phi"], b["Phi"])
    np.testing.assert_array_equal(a["Y"], b["Y"])
    assert a["nmse"] == b["nmse"]
    assert not np.array_equal(a["M"], c["M"])


def test_script_saves_outputs(tmp_path: Path) -> None:
    """Le script Hydra écrit le .npz, metrics.json et la config du run."""
    subprocess.run(
        [sys.executable, str(SCRIPT), *OVERRIDES, f"hydra.run.dir={tmp_path}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    data = np.load(tmp_path / "projective_readout.npz")
    assert {"Phi", "Y", "M", "X", "y_test", "y_pred", "W"} <= set(data.files)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert {"nmse", "r2"} <= set(metrics)
    assert metrics["n_samples"] == 10
    assert (tmp_path / ".hydra" / "config.yaml").is_file()


def test_bash_launcher_runs(tmp_path: Path) -> None:
    """Le lanceur bash écrit un run par N, sous OUT_ROOT/projective_readout/."""
    assert os.access(BASH_SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "5",
            "VIRTUAL_ENV": str(FOREIGN_VENV),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 2 x 1 runs" in result.stdout
    assert "does not match the project environment" not in result.stderr
    runs = sorted(tmp_path.glob("projective_readout/*/*"))
    assert [run.name for run in runs] == ["N4_seed0", "N6_seed0"]
    for run in runs:
        assert (run / "metrics.json").is_file()


def test_unmeasured_qubits_give_zero_features_not_nan() -> None:
    """p < 1 : M garde NaN pour un qubit non mesuré, Phi met 0, et la
    régression et les métriques restent finies (issue #91)."""
    out = _load_script().run(
        load_config([*OVERRIDES, "samples.L=30", "projective_measurement.p=0.5"])
    )
    assert np.isnan(out["M"]).any()
    assert np.all(np.isfinite(out["Phi"]))
    skipped = np.isnan(out["M"])
    np.testing.assert_array_equal(out["Phi"][:, : out["M"].shape[1]][skipped], 0.0)
    assert np.all(np.isfinite(out["y_pred"]))
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])
