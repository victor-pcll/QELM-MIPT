"""experiments/QELM_M/exp_QELM_M.py with Haar and Clifford circuits (issue #171).

``models=haar_circuit`` and ``models=clifford_circuit`` replace the TFIM by
a brickwork of random two-qubit gates on the system qubits; the initial
state, the measurements, the record, the target and the scores stay those of
the TFIM run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM import load_config
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.clifford import _phase_key, clifford_group_2q

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M.py"
RATIO_LAUNCHER = SCRIPT.with_name("exp_QELM_M_ratio.sh")
L, N = 40, 4
CIRCUITS = ["haar_circuit", "clifford_circuit"]


def _load_script() -> ModuleType:
    """Import experiments/QELM_M/exp_QELM_M.py, which is not a package."""
    spec = importlib.util.spec_from_file_location("exp_QELM_M", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overrides(model: str, *extra: str) -> list[str]:
    return [f"models={model}", f"N={N}", f"samples.L={L}", *extra]


def _spy_layers(monkeypatch, module: ModuleType) -> list[tuple[int, list]]:
    """Record (t, gates) of every brickwork layer the run applies."""
    applied: list[tuple[int, list]] = []
    apply_layer = BrickworkCircuit.apply_layer

    def spy(self, state, t):
        applied.append((t, [g for _, _, g in self.layers[t]]))
        return apply_layer(self, state, t)

    monkeypatch.setattr(BrickworkCircuit, "apply_layer", spy)
    return applied


@pytest.mark.parametrize("model", CIRCUITS)
def test_config_composes_with_the_circuit_block(model: str) -> None:
    cfg = load_config([f"models={model}", "N=6"])
    assert cfg.models == model
    assert cfg[model].N == 6 and cfg[model].Nr == cfg.Nr == 1
    assert cfg[model].layers_per_step == 1
    assert cfg[model].fixed_instance is False


@pytest.mark.parametrize("model", CIRCUITS)
def test_same_features_and_scores_as_the_tfim_run(model: str) -> None:
    """p = 1: the record has the TFIM shape, outcomes +-1 and a bias column,
    the label is +-1 and every score is finite."""
    out = _load_script().run(load_config(_overrides(model)))
    assert out["M"].shape == (L, 5, N - 1)
    assert out["X"].shape == (L, 5 * (N - 1) + 1)
    np.testing.assert_array_equal(out["X"][:, -1], 1.0)
    assert set(np.unique(out["X"][:, :-1])) <= {-1.0, 1.0}
    assert set(np.unique(out["Y"])) <= {-1.0, 1.0}
    for key in ("nmse", "r2", "accuracy", "balanced_accuracy", "accuracy_optimal"):
        assert np.isfinite(out[key])
    # Every system qubit measured at the last step: the ancilla is pure.
    np.testing.assert_allclose(out["S_vN"], 0.0, atol=1e-8)


@pytest.mark.parametrize("model", CIRCUITS)
def test_no_measurement_leaves_the_reference_maximally_mixed(model: str) -> None:
    """p = 0: the gates act on the system only, so rho_R stays I / 2."""
    out = _load_script().run(
        load_config(_overrides(model, "projective_measurement.p=0"))
    )
    np.testing.assert_allclose(out["Z0"], 0.0, atol=1e-10)
    np.testing.assert_allclose(out["S_vN"], 1.0, atol=1e-9)
    assert out["accuracy_optimal"] == pytest.approx(0.5)


@pytest.mark.parametrize("model", CIRCUITS)
@pytest.mark.parametrize("per_step", [1, 3])
def test_layers_per_step_and_new_gates_per_sample(
    monkeypatch, model: str, per_step: int
) -> None:
    """Step j applies layers j * k .. (j + 1) * k - 1, and every sample gets
    new gates (fixed_instance: false), as the TFIM gets a new disorder."""
    module = _load_script()
    applied = _spy_layers(monkeypatch, module)
    steps = 2
    module.run(
        load_config(
            _overrides(
                model,
                f"projective_measurement.nbr_time_evolution={steps}",
                f"{model}.layers_per_step={per_step}",
            )
        )
    )
    depth = steps * per_step
    assert [t for t, _ in applied] == list(range(depth)) * L
    first = [applied[k * depth][1][0] for k in range(L)]
    keys = {_phase_key(g) for g in first}
    assert len(keys) > L // 2  # new draws, not one gate repeated
    if model == "clifford_circuit":
        group = {_phase_key(u) for u in clifford_group_2q()}
        assert keys <= group


@pytest.mark.parametrize("model", CIRCUITS)
def test_fixed_instance_keeps_the_gates(monkeypatch, model: str) -> None:
    module = _load_script()
    applied = _spy_layers(monkeypatch, module)
    module.run(load_config(_overrides(model, f"{model}.fixed_instance=true")))
    depth = 5
    for k in range(1, L):
        for t in range(depth):
            for a, b in zip(applied[t][1], applied[k * depth + t][1], strict=True):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("model", CIRCUITS)
def test_reproducible_with_the_seed(model: str) -> None:
    module = _load_script()
    cfg = _overrides(model, "projective_measurement.p=0.5")
    a, b = module.run(load_config(cfg)), module.run(load_config(cfg))
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["Y"], b["Y"])
    c = module.run(load_config([*cfg, "seed=7"]))
    assert not np.array_equal(a["X"], c["X"])


def test_layers_per_step_must_be_positive() -> None:
    with pytest.raises(ValueError, match="layers_per_step"):
        _load_script().run(
            load_config(_overrides("haar_circuit", "haar_circuit.layers_per_step=0"))
        )


def test_unsupported_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        _load_script().run(load_config(["models=kitaev", "samples.L=4"]))


def test_ratio_launcher_runs_a_circuit(tmp_path: Path) -> None:
    """MODEL=clifford_circuit and EXTRA_OVERRIDES reach every run of the
    ratio sweep."""
    env = {
        **os.environ,
        "MODEL": "clifford_circuit",
        "EXTRA_OVERRIDES": "clifford_circuit.fixed_instance=true",
        "N_VALUES": "3",
        "P_VALUES": "1.0",
        "ALPHA_RATIOS": "2.0",
        "N_TEST": "10",
        "OUT_DIR": str(tmp_path),
        "TQDM_DISABLE": "1",
    }
    result = subprocess.run(
        ["bash", str(RATIO_LAUNCHER)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "models=clifford_circuit" in result.stdout
    run = tmp_path / "N3_seed42_p1.0_a2.0"
    used = OmegaConf.load(run / "config_used.yaml")
    assert used.models == "clifford_circuit"
    assert used.clifford_circuit.fixed_instance is True
    assert json.loads((run / "metrics.json").read_text())["n_train"] == 22
