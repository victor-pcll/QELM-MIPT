"""Depth T of the monitored Haar circuit read by the QELM (issue #268)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "haar_readouts"
SCRIPT = FOLDER / "haar_readouts.py"
BASH_SCRIPT = FOLDER / "depth_scaling.sh"
PLOT_SCRIPT = FOLDER / "plot_depth_scaling.py"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/haar_readouts, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hr() -> ModuleType:
    return _load(SCRIPT)


@pytest.fixture(scope="module")
def plot() -> ModuleType:
    return _load(PLOT_SCRIPT)


def _run(hr: ModuleType, *extra: str) -> dict:
    cfg = load_config(
        [
            "N=4",
            "readouts_task.p_values=[0.0,0.3]",
            "readouts_task.samples_per_p=40",
            "readouts_task.readouts=[ridge]",
            "readouts_task.features=[all]",
            "haar_circuit.fixed_instance=true",
            *extra,
        ],
        config_name="haar_readouts",
    )
    return hr.run(cfg)


@pytest.mark.parametrize(
    ("rule", "N", "depth"),
    [("10", 6, 10), (10, 12, 10), ("N", 6, 6), ("2N", 6, 12), ("3N", 4, 12)],
)
def test_depth_from_rule(hr: ModuleType, rule: str | int, N: int, depth: int) -> None:
    assert hr.depth_from_rule(rule, N) == depth


@pytest.mark.parametrize("rule", ["", "0", "0N", "N2", "1.5N", "x"])
def test_depth_from_rule_refuses_bad_rules(hr: ModuleType, rule: str) -> None:
    with pytest.raises(ValueError):
        hr.depth_from_rule(rule, 6)


def test_depth_rule_sets_the_rounds(hr: ModuleType) -> None:
    out = _run(hr, "readouts_task.depth_rule=2N")
    assert out["metrics"]["n_rounds"] == 8
    assert out["M"].shape == (80, 8, 4)
    # Without a rule, n_rounds is read as before.
    assert _run(hr)["metrics"]["n_rounds"] == 5


def test_depth_metrics_and_reproducibility(hr: ModuleType) -> None:
    a = _run(hr, "readouts_task.depth_rule=N")
    b = _run(hr, "readouts_task.depth_rule=N")
    # The timings stay out of the metrics, which repeat exactly.
    assert a["metrics"] == b["metrics"]
    assert set(a["timing"]) == {
        "time_simulation_s",
        "time_per_trajectory_s",
        "time_fit_s",
    }
    assert a["timing"]["time_simulation_s"] > 0.0
    m = a["metrics"]
    for key in ("p0", "p0.3"):
        assert 0.0 <= m[f"log10_cond_all_{key}"] <= 16.0
        for k in range(4):
            assert m[f"var_Z_{k}_{key}"] == pytest.approx(
                a["features_Z"][a["p"] == float(key[1:]), k].var()
            )
    # At p = 0 one fixed circuit maps each sector to one state: the QELM
    # reads it exactly, and <Z_k> takes two values only.
    assert m["accuracy_ridge_all_p0"] == 1.0


def test_log10_condition(hr: ModuleType) -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((50, 3)))
    q -= q.mean(axis=0)
    q, _ = np.linalg.qr(q)
    # Orthonormal centered columns: X^T X is the identity.
    assert hr.log10_condition(q) == pytest.approx(0.0, abs=1e-6)
    # Columns of scales 1 and 10: condition number 100.
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    y -= x * (x @ y) / (x @ x)
    X = np.column_stack([x / np.linalg.norm(x - x.mean()), 10 * y / np.linalg.norm(y)])
    assert hr.log10_condition(X) == pytest.approx(2.0, abs=0.05)
    # Collinear columns, or fewer samples than features: singular.
    assert hr.log10_condition(np.column_stack([x, 2 * x])) == 16.0
    assert hr.log10_condition(rng.standard_normal((3, 5))) == 16.0
    # Constant features: nothing to invert.
    assert hr.log10_condition(np.ones((10, 3))) == 0.0


def test_crossing_and_sigmoid_fit(plot: ModuleType) -> None:
    p = np.linspace(0.0, 0.4, 17)
    # Two lines that cross at p = 0.2.
    a, b = 1.0 - p, 0.9 - 0.5 * p
    assert plot.crossing(p, a, b) == pytest.approx(0.2)
    assert plot.crossing(p, a, a) is None
    # Below the floor, no crossing is searched.
    assert plot.crossing(p, a, b, floor=0.95) is None
    truth = plot.sigmoid(p, 0.15, 0.02, 0.45)
    fit = plot.fit_sigmoid(p, truth)
    assert fit["p0"] == pytest.approx(0.15, abs=1e-3)
    assert fit["w"] == pytest.approx(0.02, rel=1e-2)
    assert fit["slope"] == pytest.approx(0.45 / 0.08, rel=1e-2)


def test_bash_sweep_and_figures(tmp_path: Path, plot: ModuleType) -> None:
    assert os.access(BASH_SCRIPT, os.X_OK)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "3 4",
        "DEPTHS": "2 N",
        "SEEDS": "1 2",
        "PER_P": "20",
        "P_VALUES": "0 0.5 1",
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
        f"haar_fixed_N{n}_T{d}_seed{s}"
        for n in (3, 4)
        for d in ("2", "N")
        for s in (1, 2)
    ]
    run = tmp_path / "haar_fixed_N4_TN_seed1"
    # SKIP_DONE=1 resumes a sweep: the runs already done are not rerun.
    done = (run / "metrics.json").stat().st_mtime_ns
    (tmp_path / "haar_fixed_N3_T2_seed2" / "metrics.json").unlink()
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "SKIP_DONE": "1"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("already done, skipped") == 7
    assert (run / "metrics.json").stat().st_mtime_ns == done
    assert (tmp_path / "haar_fixed_N3_T2_seed2" / "metrics.json").is_file()
    assert not (run / "haar_readouts.npz").exists()
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["n_rounds"] == 4
    assert metrics["time_simulation_s"] > 0.0
    assert "accuracy_ridge_all_p0.5" in metrics
    plot.main([str(tmp_path)])
    for name in (
        "depth_profiles",
        "depth_crossings",
        "depth_cost",
        "depth_collapse",
        "depth_edges",
        "depth_conditioning",
    ):
        assert (tmp_path / f"{name}.png").is_file(), name
        assert (tmp_path / f"{name}.json").is_file(), name
    summary = json.loads((tmp_path / "depth_summary.json").read_text())
    assert set(summary["rules"]) == {"2", "N"}
    assert summary["rules"]["N"]["sizes"]["3"]["T"] == 3
    assert summary["rules"]["2"]["sizes"]["4"]["n_seeds"] == 2


def test_bash_sweep_refuses_bad_settings(tmp_path: Path) -> None:
    for key, value in (("MAX_JOBS", "0"), ("DEPTHS", "N2"), ("DEPTHS", "x")):
        result = subprocess.run(
            ["bash", str(BASH_SCRIPT)],
            env={**os.environ, "OUT_DIR": str(tmp_path), key: value},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (key, value)
