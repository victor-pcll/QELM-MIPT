"""The simple QELM experiment in experiments/QELM_Z/exp_QELM_Z.py."""

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
SCRIPT = REPO_ROOT / "experiments" / "QELM_Z" / "exp_QELM_Z.py"
BASH_SCRIPT = REPO_ROOT / "experiments" / "QELM_Z" / "exp_QELM_Z.sh"
# Virtualenv of another project, active in the shell that launches the script
# (issue #79): uv must neither use it nor warn about it.
FOREIGN_VENV = REPO_ROOT.parent / "other-project" / ".venv"
# N = 3 system qubits and the ancilla: 4 qubits in all (issue #235).
OVERRIDES = ["models=TFIM", "N=3", "samples.L=10"]


def _load_script() -> ModuleType:
    """Import experiments/QELM_Z/exp_QELM_Z.py, which is not a package."""
    spec = importlib.util.spec_from_file_location("exp_QELM_Z", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_shapes_and_bias_column() -> None:
    """X holds 3 observables x N system sites plus a column of ones."""
    out = _load_script().run(load_config(OVERRIDES))
    L, N = 10, 3
    assert out["X"].shape == (L, 3 * N + 1)
    np.testing.assert_array_equal(out["X"][:, -1], 1.0)
    assert out["Y"].shape == (L,)
    assert np.all(np.abs(out["X"]) <= 1 + 1e-9)
    assert np.all(np.abs(out["Y"]) <= 1 + 1e-9)
    n_test = out["y_test"].shape[0]
    assert out["y_pred"].shape == (n_test,)
    assert out["W"].shape == (3 * N + 1,)
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


def test_features_match_the_density_matrix() -> None:
    """Evolving psi gives the same <Z_k> as evolving rho = |psi><psi|."""
    from omegaconf import OmegaConf

    from QELM import TransverseFieldIsingModel, set_seed
    from QELM.encoding import AncillaChargeSectorEncoding
    from QELM.training import SingleObservable

    cfg = load_config([*OVERRIDES, "samples.L=1", "+experiment.observables=[Z]"])
    out = _load_script().run(cfg)
    set_seed(cfg)
    sector = AncillaChargeSectorEncoding.from_config(cfg)
    # The TFIM of exp_QELM_Z acts on every qubit, the ancilla included.
    tfim_cfg = OmegaConf.merge(cfg, {"TFIM": {"N": int(cfg.N_total)}})
    rho = TransverseFieldIsingModel.from_config(tfim_cfg).evolve(
        sector.rho(), dt=float(cfg.evolution.t)
    )
    Z = SingleObservable({"n_qubits": 4, "observables": ["Z"]}).operators()
    expected = [np.real(np.trace(Z[k] @ rho)) for k in range(1, 4)]
    np.testing.assert_allclose(out["X"][0, :-1], expected, atol=1e-8)
    np.testing.assert_allclose(out["Y"][0], np.real(np.trace(Z[0] @ rho)), atol=1e-8)


def test_qelm_learns_the_ancilla() -> None:
    """With enough samples the readout predicts <Z_0> almost perfectly, with the
    default alpha of the config and no override (issue #92)."""
    cfg = load_config(["models=TFIM", "N=3", "samples.L=200"])
    out = _load_script().run(cfg)
    assert out["r2"] > 0.95


def test_run_is_reproducible_with_the_seed() -> None:
    """Same seed, same results; another seed, other disorder."""
    module = _load_script()
    a = module.run(load_config(OVERRIDES))
    b = module.run(load_config(OVERRIDES))
    c = module.run(load_config([*OVERRIDES, "seed=1"]))
    np.testing.assert_array_equal(a["X"], b["X"])
    assert a["nmse"] == b["nmse"]
    assert not np.array_equal(a["Y"], c["Y"])


def test_script_saves_outputs(tmp_path: Path) -> None:
    """The Hydra script writes the .npz, metrics.json and the run config,
    raw and resolved."""
    subprocess.run(
        [sys.executable, str(SCRIPT), *OVERRIDES, f"hydra.run.dir={tmp_path}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    data = np.load(tmp_path / "exp_QELM_Z.npz")
    assert {"X", "Y", "y_test", "y_pred", "W"} <= set(data.files)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert {"nmse", "r2"} <= set(metrics)
    assert metrics["n_samples"] == 10
    assert (tmp_path / ".hydra" / "config.yaml").is_file()
    # The config with its interpolations evaluated (issue #85).
    resolved = json.loads((tmp_path / "resolved_config.json").read_text())
    assert resolved["samples"]["L"] == 10
    assert resolved["samples"]["train_size"] == 7  # ${int:${mul:0.7,10}}
    assert resolved["N_total"] == resolved["N"] + resolved["Nr"]
    assert "Ns" not in resolved
    assert "${" not in (tmp_path / "resolved_config.json").read_text()


def test_bash_launcher_runs(tmp_path: Path) -> None:
    """The bash launcher writes one run per N (4 and 6) under OUT_ROOT/exp_QELM_Z/."""
    assert os.access(BASH_SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "10",
            "VIRTUAL_ENV": str(FOREIGN_VENV),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 2 x 1 runs" in result.stdout
    assert "does not match the project environment" not in result.stderr
    runs = sorted(tmp_path.glob("exp_QELM_Z/*/*"))
    assert [run.name for run in runs] == ["N4_seed42", "N6_seed42"]
    for run in runs:
        assert (run / "metrics.json").is_file()
