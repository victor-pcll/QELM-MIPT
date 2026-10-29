"""The purification-decoding experiment, experiments/QELM_R/exp_QELM_R.py (#247)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config
from QELM.evaluation.chaos_locality import floquet_ising_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "QELM_R"
SCRIPT = FOLDER / "exp_QELM_R.py"
BASH_SCRIPT = FOLDER / "exp_QELM_R.sh"
PLOT_SCRIPT = FOLDER / "plot_QELM_R.py"
N = 4
P_VALUES = "[0.0,0.3,1.0]"
PER_P = 16
# Couplings of Skinner et al. (2019), the defaults of conf/QELM_R.yaml.
SKINNER = {"phi_xx": 0.3, "phi_zz": 0.6, "theta_x": 0.2, "theta_z": 0.4}


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/QELM_R, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*extra: str) -> dict:
    cfg = load_config(
        [
            f"N={N}",
            f"purification_task.p_values={P_VALUES}",
            f"purification_task.samples_per_p={PER_P}",
            *extra,
        ],
        config_name="QELM_R",
    )
    return _load(SCRIPT).run(cfg)


def test_measure_system_never_touches_the_reference() -> None:
    exp = _load(SCRIPT)
    # R in |+>, system |00>: measuring the system leaves R in |+>.
    psi = np.kron(np.array([1, 1]) / np.sqrt(2), np.array([1, 0, 0, 0]))
    rng = np.random.default_rng(0)
    same, outcomes = exp.measure_system(psi, 0.0, rng)
    np.testing.assert_array_equal(same, psi)
    np.testing.assert_array_equal(outcomes, [0, 0])
    after, outcomes = exp.measure_system(psi.astype(complex), 1.0, rng)
    np.testing.assert_array_equal(outcomes, [1, 1])
    np.testing.assert_allclose(after, psi, atol=1e-12)


def test_half_layers_alternate_the_bonds() -> None:
    exp = _load(SCRIPT)
    # System qubit k is qubit k + 1 of the state, R being qubit 0.
    assert exp.half_layer_bonds(5, 0) == [(1, 2), (3, 4)]
    assert exp.half_layer_bonds(5, 1) == [(2, 3), (4, 5)]
    assert exp.half_layer_bonds(4, 2) == [(1, 2), (3, 4)]


def test_circuits_of_the_two_dynamics() -> None:
    exp = _load(SCRIPT)
    rng = np.random.default_rng(0)
    floquet = exp.circuit_sampler("floquet", 4, 3, False, dict(SKINNER), rng)
    a, b = floquet(), floquet()
    assert len(a) == 3 and [len(layer) for layer in a] == [2, 1, 2]
    gate = floquet_ising_gate(**SKINNER)
    for layer in a:
        for _, _, g in layer:
            np.testing.assert_array_equal(g, gate)
    assert a is b
    np.testing.assert_allclose(gate.conj().T @ gate, np.eye(4), atol=1e-12)
    fixed = exp.circuit_sampler("haar", 4, 3, True, dict(SKINNER), rng)
    assert fixed() is fixed()
    redrawn = exp.circuit_sampler("haar", 4, 3, False, dict(SKINNER), rng)
    assert not np.allclose(redrawn()[0][0][2], redrawn()[0][0][2])
    with pytest.raises(ValueError, match="dynamics"):
        exp.circuit_sampler("TFIM", 4, 3, True, dict(SKINNER), rng)


def test_default_floquet_couplings_are_the_paper_values() -> None:
    cfg = load_config([], config_name="QELM_R")
    assert dict(cfg.purification_task.floquet) == SKINNER
    assert cfg.models == "haar_circuit" and cfg.Nr == 1
    assert cfg.training.alpha is None


@pytest.mark.parametrize("dynamics", ["haar", "floquet"])
def test_shapes_and_physics_of_the_outputs(dynamics: str) -> None:
    out = _run(f"purification_task.dynamics={dynamics}")
    L, n_rounds = 3 * PER_P, 2 * N
    p = out["p"]
    assert out["M"].shape == (L, n_rounds, N)
    assert out["bloch"].shape == (L, 3)
    assert out["z"].shape == out["S_R"].shape == out["shot"].shape == p.shape == (L,)
    assert out["X"].shape == (L, n_rounds * N + 1)
    assert set(np.unique(out["M"])) <= {-1.0, 0.0, 1.0}
    assert set(np.unique(out["shot"])) <= {-1.0, 1.0}
    assert np.all(np.linalg.norm(out["bloch"], axis=1) <= 1 + 1e-9)
    # p = 0: the unitary on the system cannot change R, maximally mixed.
    np.testing.assert_array_equal(out["M"][p == 0.0], 0.0)
    np.testing.assert_allclose(out["bloch"][p == 0.0], 0.0, atol=1e-9)
    np.testing.assert_allclose(out["S_R"][p == 0.0], 1.0, atol=1e-9)
    # p = 1: the first round collapses the system to a product state, R is
    # purified.
    assert np.all(np.abs(out["M"][p == 1.0]) == 1.0)
    np.testing.assert_allclose(out["S_R"][p == 1.0], 0.0, atol=1e-6)
    assert out["purified_p1"] == 1.0 and out["purified_p0"] == 0.0
    assert out["S_R_mean_p0.3"] == pytest.approx(out["S_R"][p == 0.3].mean())
    assert out["train"].size + out["test"].size == L


def test_scores_per_p() -> None:
    exp = _load(SCRIPT)
    out = _run("purification_task.dynamics=floquet")
    for level in ("0", "0.3", "1"):
        for name in exp.PER_P_SCORES:
            assert f"{name}_p{level}" in out
        for name in ("accuracy", "accuracy_expected", "majority_accuracy"):
            assert 0.0 <= out[f"{name}_p{level}"] <= 1.0
        # No decoder beats the Bayes ceiling in expectation.
        assert out[f"accuracy_expected_p{level}"] <= (
            out[f"bayes_accuracy_p{level}"] + 1e-12
        )
        assert out[f"bayes_accuracy_p{level}"] <= (
            out[f"bloch_accuracy_p{level}"] + 1e-12
        )
        # alpha chosen by cross-validation among the candidates.
        for name in ("alpha_shot", "alpha_z"):
            assert np.isclose(out[f"{name}_p{level}"], exp.ALPHAS).any()
    # A mixed reference: chance level, and R2 of <Z_R> = 0 undefined.
    assert out["accuracy_expected_p0"] == pytest.approx(0.5)
    assert out["bayes_accuracy_p0"] == pytest.approx(0.5)
    assert np.isnan(out["r2_z_p0"])
    # A purified reference: the ceiling along its Bloch vector is 1.
    assert out["bloch_accuracy_p1"] == pytest.approx(1.0)
    assert out["y_pred"].shape == out["y_test"].shape == out["shot_pred"].shape


def test_a_fixed_alpha_is_used() -> None:
    out = _run("purification_task.dynamics=floquet", "training.alpha=2.5")
    assert out["alpha_shot_p0.3"] == 2.5 and out["alpha_z_p1"] == 2.5


def test_the_decoder_reads_the_floquet_reference_at_p_one() -> None:
    # Floquet, p = 1: the first half-layer and its measurements fix R, a
    # function of a few outcomes that a linear readout mostly learns.
    out = _run(
        "purification_task.dynamics=floquet", "purification_task.samples_per_p=120"
    )
    assert out["bloch_accuracy_p1"] == pytest.approx(1.0)
    assert out["accuracy_expected_p1"] > 0.7


def test_reproducible_with_the_seed() -> None:
    a = _run("seed=3")
    b = _run("seed=3")
    c = _run("seed=4")
    np.testing.assert_array_equal(a["M"], b["M"])
    np.testing.assert_array_equal(a["bloch"], b["bloch"])
    np.testing.assert_array_equal(a["shot"], b["shot"])
    assert not np.array_equal(a["bloch"], c["bloch"])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("Nr=0", "Nr=1"),
        ("purification_task.p_values=[0.2,1.5]", "in \\[0, 1\\]"),
        ("purification_task.p_values=[0.2,0.2]", "distinct"),
        ("purification_task.samples_per_p=3", "samples_per_p"),
        ("purification_task.n_rounds=0", "n_rounds"),
        ("purification_task.dynamics=clifford", "dynamics"),
    ],
)
def test_rejects_bad_settings(override: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _run(override)


def test_script_writes_metrics(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "purification_task.dynamics=floquet",
            f"N={N}",
            f"purification_task.p_values={P_VALUES}",
            f"purification_task.samples_per_p={PER_P}",
            f"hydra.run.dir={tmp_path}",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for key in ("r2", "accuracy_p0.3", "bayes_accuracy_p1", "S_R_mean_p0", "n_rounds"):
        assert key in metrics
    data = np.load(tmp_path / "exp_QELM_R.npz")
    assert data["z"].shape == (3 * PER_P,)


def test_bash_sweep_and_figures(tmp_path: Path) -> None:
    assert os.access(BASH_SCRIPT, os.X_OK)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "DYNAMICS": "haar floquet",
        "N_VALUES": "3",
        "SEEDS": "1 2",
        "PER_P": "8",
        "P_VALUES": "0 0.5",
        "MAX_JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
    )
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == [
        "floquet_N3_seed1",
        "floquet_N3_seed2",
        "haar_N3_seed1",
        "haar_N3_seed2",
    ]
    cfg = (tmp_path / "haar_N3_seed1" / ".hydra" / "config.yaml").read_text()
    assert "p_c: 0.17" in cfg
    cfg = (tmp_path / "floquet_N3_seed1" / ".hydra" / "config.yaml").read_text()
    assert "dynamics: floquet" in cfg and "p_c: null" in cfg
    _load(PLOT_SCRIPT).main([str(tmp_path)])
    for name in ("purification_vs_p", "accuracy_vs_p", "r2_z_vs_p"):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.json").is_file()


def test_bash_sweep_refuses_a_bad_max_jobs(tmp_path: Path) -> None:
    env = {**os.environ, "OUT_DIR": str(tmp_path), "MAX_JOBS": "0"}
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 2
