"""The entropy-prediction experiment, experiments/QELM_S/exp_QELM_S.py (#230)."""

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
from QELM.physics import Z, get_site_operator

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "QELM_S"
SCRIPT = FOLDER / "exp_QELM_S.py"
BASH_SCRIPT = FOLDER / "exp_QELM_S.sh"
PLOT_SCRIPT = FOLDER / "plot_QELM_S.py"
N = 4
P_VALUES = "[0.0,0.3,1.0]"
PER_P = 12


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/QELM_S, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*extra: str):
    return load_config(
        [
            f"N={N}",
            f"entropy_task.p_values={P_VALUES}",
            f"entropy_task.samples_per_p={PER_P}",
            *extra,
        ],
        config_name="QELM_S",
    )


def _run(*extra: str) -> dict:
    return _load(SCRIPT).run(_cfg(*extra))


def test_neel_state_and_its_observables() -> None:
    exp = _load(SCRIPT)
    psi = exp.neel_state(5)
    assert np.flatnonzero(psi).tolist() == [0b01010]
    np.testing.assert_allclose(exp.z_expectations(psi), [1, -1, 1, -1, 1])
    shot = exp.single_shot(psi, np.random.default_rng(0))
    np.testing.assert_array_equal(shot, [1, -1, 1, -1, 1])


def test_z_expectations_match_the_full_operators() -> None:
    exp = _load(SCRIPT)
    rng = np.random.default_rng(2)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    expected = [
        float(np.real(psi.conj() @ get_site_operator(Z, k, 4) @ psi)) for k in range(4)
    ]
    np.testing.assert_allclose(exp.z_expectations(psi), expected, atol=1e-12)


def test_single_shots_follow_the_born_rule() -> None:
    exp = _load(SCRIPT)
    # (|00> + |11>) / sqrt(2): the two qubits always agree, each +-1 half
    # of the time.
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rng = np.random.default_rng(0)
    shots = np.array([exp.single_shot(psi, rng) for _ in range(400)])
    np.testing.assert_array_equal(shots[:, 0], shots[:, 1])
    assert 0.4 < np.mean(shots[:, 0] == 1) < 0.6


def test_measure_round_at_p_zero_and_one() -> None:
    exp = _load(SCRIPT)
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rng = np.random.default_rng(0)
    same, outcomes = exp.measure_round(psi, 0.0, rng)
    np.testing.assert_array_equal(same, psi)
    np.testing.assert_array_equal(outcomes, [0, 0])
    collapsed, outcomes = exp.measure_round(psi, 1.0, rng)
    assert set(np.abs(outcomes)) == {1.0}
    assert outcomes[0] == outcomes[1]
    assert np.count_nonzero(np.abs(collapsed) > 1e-12) == 1


@pytest.mark.parametrize("model", ["haar_circuit", "clifford_circuit", "TFIM"])
def test_shapes_and_range_of_the_outputs(model: str) -> None:
    out = _run(f"models={model}")
    L, n_rounds = 3 * PER_P, 2 * N
    assert out["M"].shape == (L, n_rounds, N)
    assert out["Z"].shape == out["Zshot"].shape == (L, N)
    assert out["S"].shape == out["p"].shape == (L,)
    assert np.all(out["S"] >= -1e-12) and np.all(out["S"] <= N // 2 + 1e-9)
    assert set(np.unique(out["Zshot"])) <= {-1.0, 1.0}
    assert np.all(np.abs(out["Z"]) <= 1 + 1e-12)
    assert set(np.unique(out["M"])) <= {-1.0, 0.0, 1.0}
    assert out["train"].size + out["test"].size == L
    assert out["X"].shape == (L, n_rounds * N + N + 1)
    assert out["S_max"] == N // 2 and out["n_rounds"] == n_rounds


def test_the_record_follows_p() -> None:
    out = _run("models=haar_circuit")
    p, M, S = out["p"], out["M"], out["S"]
    # p = 0: nothing measured; p = 1: everything measured, a product state.
    np.testing.assert_array_equal(M[p == 0.0], 0.0)
    assert np.all(np.abs(M[p == 1.0]) == 1.0)
    np.testing.assert_allclose(S[p == 1.0], 0.0, atol=1e-9)
    # Without measurements, the Haar brickwork entangles the two halves.
    assert S[p == 0.0].mean() > 1.0
    for level in (0.0, 0.3, 1.0):
        assert out[f"S_mean_p{level:g}"] == pytest.approx(S[p == level].mean())


def test_scores_of_every_feature_set() -> None:
    exp = _load(SCRIPT)
    out = _run("models=haar_circuit", "entropy_task.p_c=0.17")
    for name in exp.FEATURE_SETS:
        for key in (f"r2_{name}", f"nmse_{name}", f"r2_within_{name}"):
            assert np.isfinite(out[key]), key
        assert out[f"y_pred_{name}"].shape == out["y_test"].shape
        assert 0.0 <= out[f"accuracy_phase_{name}"] <= 1.0
        if name != "p":
            assert np.isfinite(out[f"r2_knn_{name}"])
    # The p baseline predicts the training mean at each p: it knows the phase
    # label, and nothing about the spread at fixed p.
    assert out["accuracy_phase_p"] == 1.0
    assert out["r2_within_p"] <= 1e-9
    # S has no spread at p = 1, so the R2 there is NaN.
    assert np.isnan(out["r2_within_p_p1"])
    # Per-p R2 only for the plotted sets.
    assert "r2_within_M_Z_p0.3" in out
    assert "r2_within_absM_p0.3" not in out
    np.testing.assert_array_equal(out["y_pred"], out["y_pred_M_Z"])
    assert out["r2"] == out["r2_M_Z"]


def test_without_p_c_or_knn() -> None:
    out = _run("models=haar_circuit", "entropy_task.knn=false")
    assert not any(key.startswith(("accuracy_phase_", "r2_knn_")) for key in out)


def test_reproducible_with_the_seed() -> None:
    a = _run("models=haar_circuit", "seed=3")
    b = _run("models=haar_circuit", "seed=3")
    c = _run("models=haar_circuit", "seed=4")
    np.testing.assert_array_equal(a["M"], b["M"])
    np.testing.assert_array_equal(a["S"], b["S"])
    assert not np.array_equal(a["S"], c["S"])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("Nr=1", "without ancilla"),
        ("entropy_task.p_values=[0.2,1.5]", "in \\[0, 1\\]"),
        ("entropy_task.p_values=[0.2,0.2]", "distinct"),
        ("entropy_task.samples_per_p=1", "samples_per_p"),
        ("entropy_task.n_rounds=0", "n_rounds"),
        ("models=kitaev", "not supported"),
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
            "models=haar_circuit",
            f"N={N}",
            f"entropy_task.p_values={P_VALUES}",
            f"entropy_task.samples_per_p={PER_P}",
            f"hydra.run.dir={tmp_path}",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for key in ("r2", "r2_within_M_Z", "r2_p", "r2_knn_M_Z_even", "S_mean_p0.3"):
        assert key in metrics
    data = np.load(tmp_path / "exp_QELM_S.npz")
    assert data["S"].shape == (3 * PER_P,)


def test_bash_sweep_and_figures(tmp_path: Path) -> None:
    assert os.access(BASH_SCRIPT, os.X_OK)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "MODELS": "haar_circuit TFIM",
        "N_VALUES": "4",
        "SEEDS": "1 2",
        "PER_P": "6",
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
        "TFIM_N4_seed1",
        "TFIM_N4_seed2",
        "haar_circuit_N4_seed1",
        "haar_circuit_N4_seed2",
    ]
    cfg = (tmp_path / "haar_circuit_N4_seed1" / ".hydra" / "config.yaml").read_text()
    assert "p_c: 0.17" in cfg
    _load(PLOT_SCRIPT).main([str(tmp_path)])
    for name in ("entropy_vs_p", "r2_summary", "r2_within_vs_p", "predicted_vs_exact"):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.json").is_file()


def test_bash_sweep_refuses_a_bad_max_jobs(tmp_path: Path) -> None:
    env = {**os.environ, "OUT_DIR": str(tmp_path), "MAX_JOBS": "0"}
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 2
