"""The double-descent experiment in experiments/double_descent/ (issue #181).

Every test uses scikit-learn's 8x8 digits (``double_descent.dataset=digits``),
bundled with scikit-learn, so that nothing is downloaded.
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

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "double_descent"
SCRIPT = FOLDER / "double_descent.py"
PLOT = FOLDER / "plot_double_descent.py"
BASH_SCRIPT = FOLDER / "double_descent.sh"

RATIOS = "[0.5,0.8,1.0,1.2,2.0,4.0]"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*overrides: str):  # noqa: ANN202
    return load_config(
        [
            "N=3",
            "double_descent.dataset=digits",
            "double_descent.n_pool=200",
            "double_descent.n_repeats=5",
            f"double_descent.alpha_ratios={RATIOS}",
            *overrides,
        ],
        config_name="double_descent",
    )


def test_config_composes() -> None:
    cfg = load_config(config_name="double_descent")
    assert (cfg.N, cfg.Ns, cfg.Nr) == (8, 8, 0)
    assert cfg.training.name == "pinv"
    assert cfg.TFIM.h == cfg.TFIM.J == 1.0
    assert cfg.evolution.t == 10.0
    assert list(cfg.double_descent.digits) == [3, 5]
    assert 1.0 in list(cfg.double_descent.alpha_ratios)
    assert list(cfg.double_descent.models) == ["qelm", "qelm_t0", "elm", "linear"]


def test_run_gives_the_three_checks() -> None:
    results = _load(SCRIPT).run(_cfg())
    # 3 qubits: 9 one-site observables and 9 correlators, plus the bias.
    assert results["n_columns_qelm"] == 19
    assert results["observables"][0] == "X_0"
    assert results["observables"][-1] == "bias"
    assert results["n_features_qelm"] == 19
    # t = 0: the simulation is the closed form, and the <Y> columns vanish.
    assert results["t0_max_deviation"] < 1e-12
    assert results["n_features_qelm_t0"] == 19 - 3 - 3
    # The classical ELM has as many neurons as the QELM has observables.
    assert results["n_columns_elm"] == 19
    assert results["n_features_elm"] == 19
    assert results["n_features_linear"] == 4
    for model in ("qelm", "qelm_t0", "elm", "linear"):
        assert results[f"{model}_test_mse"].shape == (6, 5)
        assert 0.9 < results[f"accuracy_{model}"] <= 1.0
    # The peak at alpha = 1. Not for linear: with 4 features its smallest
    # training set holds 2 images, already a poor fit.
    for model in ("qelm", "qelm_t0", "elm"):
        assert results[f"peak_alpha_{model}"] == 1.0
    np.testing.assert_array_equal(results["qelm_n_train"], [10, 15, 19, 23, 38, 76])
    assert results["accuracy_pixels"] > 0.9
    assert results["test_mse_at_threshold_qelm"] > np.median(
        results["qelm_test_mse"][-1]
    )


def test_same_seed_same_run_and_seeds_differ() -> None:
    run = _load(SCRIPT).run
    first, again = run(_cfg("seed=3")), run(_cfg("seed=3"))
    other = run(_cfg("seed=4"))
    np.testing.assert_array_equal(first["qelm_test_mse"], again["qelm_test_mse"])
    np.testing.assert_array_equal(first["J_matrix"], again["J_matrix"])
    assert not np.array_equal(first["J_matrix"], other["J_matrix"])


def test_run_errors() -> None:
    run = _load(SCRIPT).run
    with pytest.raises(ValueError, match="Unknown models"):
        run(_cfg("double_descent.models=[qelm,svm]"))
    with pytest.raises(ValueError, match="no ancilla"):
        run(_cfg("Nr=1"))
    with pytest.raises(ValueError, match="exceeds the pool"):
        run(_cfg("double_descent.alpha_ratios=[20.0]"))


def test_launcher_runs_the_seeds_and_plots(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "N": "3",
        "SEEDS": "0 1",
        "DATASET": "digits",
        "N_POOL": "200",
        "N_REPEATS": "3",
        "OUT_DIR": str(tmp_path / "sweep"),
        "EXTRA_OVERRIDES": f"double_descent.alpha_ratios={RATIOS}",
    }
    out = subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    sweep = tmp_path / "sweep"
    assert sorted(p.name for p in sweep.iterdir() if p.is_dir()) == [
        "N3_seed0",
        "N3_seed1",
    ]
    run_dir = sweep / "N3_seed0"
    for name in ("double_descent.npz", "metrics.json", "resolved_config.json"):
        assert (run_dir / name).is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["models"] == ["qelm", "qelm_t0", "elm", "linear"]
    assert metrics["peak_alpha_qelm"] == 1.0
    # The figure, its data and the summary in the sweep folder.
    assert (sweep / "double_descent.png").is_file()
    data = json.loads((sweep / "double_descent.json").read_text())
    assert data["extra"]["summary"]["n_runs"] == 2
    summary = json.loads((sweep / "summary.json").read_text())
    assert summary["models"]["qelm"]["peak_alpha_of_mean"] == 1.0
    assert summary["models"]["qelm"]["n_features"] == [19, 19]


def test_plot_helpers(tmp_path: Path) -> None:
    plot = _load(PLOT)
    script = _load(SCRIPT)
    for seed in (0, 1):
        script.save(script.run(_cfg(f"seed={seed}")), tmp_path / f"N3_seed{seed}")
    runs = plot.load_runs(tmp_path)
    alpha, mean, sem = plot.curve(runs, "qelm", "test_error", "mean")
    assert alpha.shape == mean.shape == sem.shape == (6,)
    assert np.all(np.isfinite(sem))
    fig, axes = plot.draw(runs)
    assert axes.shape == (2, 2)
    for ax in axes.ravel():
        assert ax.get_xscale() == "log"
        assert any(
            line.get_linestyle() == "--" and set(line.get_xdata()) == {1.0}
            for line in ax.lines
        )
    assert axes[0, 0].get_yscale() == "log"
    _, _, single = plot.curve(runs[:1], "qelm", "test_mse", "median")
    assert np.all(np.isnan(single))
    with pytest.raises(FileNotFoundError):
        plot.load_runs(tmp_path / "N3_seed0")
