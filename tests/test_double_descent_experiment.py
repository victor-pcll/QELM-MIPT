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
    assert (cfg.N, cfg.Nr, cfg.N_total) == (8, 0, 8)
    assert cfg.training.name == "ridge"
    assert cfg.training.alpha == 0.0
    assert cfg.TFIM.h == cfg.TFIM.J == 1.0
    assert cfg.evolution.t == 10.0
    assert list(cfg.double_descent.classes) == [3, 5]
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
    # Binary task: two classes, one-hot targets, a 2 x 2 confusion matrix
    # whose rows count the test images of each digit.
    assert results["n_classes"] == 2
    np.testing.assert_array_equal(results["classes"], [3, 5])
    assert set(np.unique(results["y_pool"])) == {0, 1}
    for model in ("qelm", "qelm_t0", "elm", "linear", "pixels"):
        confusion = results[f"confusion_{model}"]
        assert confusion.shape == (2, 2)
        np.testing.assert_array_equal(
            confusion.sum(axis=1), np.bincount(results["y_test"], minlength=2)
        )
        assert np.trace(confusion) / confusion.sum() == pytest.approx(
            results[f"accuracy_{model}"]
        )
    assert results["ridge_alpha"] == 0.0


def test_run_with_more_digits() -> None:
    """Multi-class: three digits, three outputs, a 3 x 3 confusion matrix."""
    results = _load(SCRIPT).run(_cfg("double_descent.classes=[3,5,8]"))
    assert results["n_classes"] == 3
    assert set(np.unique(results["y_test"])) == {0, 1, 2}
    assert results["confusion_qelm"].shape == (3, 3)
    assert results["accuracy_qelm"] > 0.8
    assert results["peak_alpha_qelm"] == 1.0


def test_ridge_penalty_removes_the_peak() -> None:
    run = _load(SCRIPT).run
    free = run(_cfg())
    ridge = run(_cfg("training.alpha=0.1"))
    assert ridge["ridge_alpha"] == 0.1
    assert ridge["test_mse_at_threshold_qelm"] < free["test_mse_at_threshold_qelm"]


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
    with pytest.raises(ValueError, match="Unknown encoding"):
        run(_cfg("double_descent.encoding=amplitude_encoding"))


@pytest.mark.parametrize(
    ("encoding", "n_inputs", "rank_t0"),
    [
        ("angle_encoding", 3, 13),  # real state: <Y_i> = <Y_i Y_j> = 0
        ("dense_angle_encoding", 6, 19),
        ("uniform_bloch_encoding", 6, 19),
        ("general_encoding", 6, 13),  # real state
    ],
)
def test_run_with_each_encoding(encoding: str, n_inputs: int, rank_t0: int) -> None:
    """Issue #196: the same reservoir and observables, another input state."""
    results = _load(SCRIPT).run(_cfg(f"double_descent.encoding={encoding}"))
    assert results["encoding"] == encoding
    assert results["n_inputs"] == n_inputs
    assert results["theta_pool"].shape == (200, n_inputs)
    assert results["t0_max_deviation"] < 1e-12
    assert results["n_columns_qelm"] == 19  # N = 3 qubits in every case
    assert results["n_features_qelm_t0"] == rank_t0
    assert results["n_features_linear"] == n_inputs + 1
    assert results["accuracy_qelm"] > 0.9
    assert results["peak_alpha_qelm"] == 1.0


def test_encoding_inputs_follow_the_bounds() -> None:
    script = _load(SCRIPT)
    cfg = _cfg("double_descent.encoding=uniform_bloch_encoding")
    bounds = script.input_bounds(script.build_encoding(cfg))
    np.testing.assert_allclose(bounds, [[0, 1], [0, np.pi]] * 3)
    u = np.array([[0.0] * 6, [0.5] * 6, [1.0] * 6])
    np.testing.assert_allclose(
        script.encoding_inputs(u, bounds),
        [[0] * 6, [0.5, np.pi / 2] * 3, [1, np.pi] * 3],
    )
    # The default encoding keeps the angles of issue #181, in [0, pi].
    default = script.build_encoding(_cfg())
    assert default.name == "rotation_encoding"
    np.testing.assert_array_equal(script.input_bounds(default), [[0, np.pi]] * 3)
    # rotation_encoding and angle_encoding give the same features.
    run = script.run
    rotation, angle = run(_cfg()), run(_cfg("double_descent.encoding=angle_encoding"))
    np.testing.assert_array_equal(rotation["theta_pool"], angle["theta_pool"])
    np.testing.assert_allclose(
        rotation["features_qelm_pool"], angle["features_qelm_pool"], atol=1e-12
    )


def test_launcher_runs_the_seeds_and_plots(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "N": "3",
        "SEEDS": "0 1",
        "DATASET": "digits",
        "N_POOL": "200",
        "N_REPEATS": "3",
        "CLASSES": "3 5",
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
        "N3_c35_seed0",
        "N3_c35_seed1",
    ]
    run_dir = sweep / "N3_c35_seed0"
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
    assert summary["classes"] == [3, 5]
    # The confusion matrices, summed over the two seeds.
    assert (sweep / "confusion_matrices.png").is_file()
    confusion = json.loads((sweep / "confusion_matrices.json").read_text())
    assert set(confusion["extra"]["confusion"]) == {
        "qelm",
        "qelm_t0",
        "elm",
        "linear",
        "pixels",
    }


def test_plot_helpers(tmp_path: Path) -> None:
    plot = _load(PLOT)
    script = _load(SCRIPT)
    for seed in (0, 1):
        script.save(script.run(_cfg(f"seed={seed}")), tmp_path / f"N3_c35_seed{seed}")
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
    # Seeds whose ranks differ by one: the points are matched by index, at
    # the mean ratio (a sweep of three digits hit this with the ELM).
    shifted = [runs[0], {**runs[1], "arrays": dict(runs[1]["arrays"])}]
    shifted[1]["arrays"]["elm_alpha"] = runs[1]["arrays"]["elm_alpha"] * 1.01
    x, _, _ = plot.curve(shifted, "elm", "test_mse", "median")
    np.testing.assert_allclose(x, runs[0]["arrays"]["elm_alpha"] * 1.005)
    shifted[1]["arrays"]["elm_alpha"] = runs[1]["arrays"]["elm_alpha"][:-1]
    with pytest.raises(ValueError, match="ratios"):
        plot.curve(shifted, "elm", "test_mse", "median")
    with pytest.raises(FileNotFoundError):
        plot.load_runs(tmp_path / "N3_c35_seed0")
    # One confusion panel per model and the pixel baseline, rows the true
    # digit; the counts are summed over the seeds.
    fig, axes = plot.draw_confusion(runs)
    assert len(axes) == 5
    assert [t.get_text() for t in axes[0].get_yticklabels()] == ["3", "5"]
    summed = plot.confusion(runs, "qelm")
    np.testing.assert_array_equal(
        summed,
        runs[0]["arrays"]["confusion_qelm"] + runs[1]["arrays"]["confusion_qelm"],
    )


def test_plot_encodings(tmp_path: Path) -> None:
    """Issue #196: one sweep per encoding, compared on one figure."""
    plot = _load(PLOT)
    script = _load(SCRIPT)
    for encoding in ("angle_encoding", "dense_angle_encoding"):
        for seed in (0, 1):
            cfg = _cfg(f"seed={seed}", f"double_descent.encoding={encoding}")
            script.save(script.run(cfg), tmp_path / encoding / f"N3_c35_seed{seed}")
    (tmp_path / "empty").mkdir()  # a folder without runs is skipped
    by_encoding = plot.load_encodings(tmp_path)
    assert list(by_encoding) == ["angle_encoding", "dense_angle_encoding"]
    table = plot.plot_encodings(tmp_path)
    assert table["dense_angle_encoding"]["n_inputs"] == 6
    assert table["angle_encoding"]["models"]["qelm"]["peak_alpha_of_mean"] == 1.0
    assert (tmp_path / "encodings.png").is_file()
    data = json.loads((tmp_path / "encodings.json").read_text())
    assert data["extra"]["model"] == "qelm"
    saved = json.loads((tmp_path / "encodings_summary.json").read_text())
    assert set(saved) == {"angle_encoding", "dense_angle_encoding"}
    fig, axes = plot.draw_encodings(by_encoding)
    assert len(axes) == 2
    assert [t.get_text() for t in axes[0].get_legend().get_texts()] == [
        "angle",
        "dense angle",
    ]
    plot.main(["--encodings", str(tmp_path)])  # the command line
    # Two seeds of one sweep with different encodings cannot be compared.
    mixed = tmp_path / "mixed"
    for name in ("angle_encoding", "dense_angle_encoding"):
        for run in (tmp_path / name).iterdir():
            (mixed / f"{name}_{run.name}").mkdir(parents=True)
            for f in run.iterdir():
                (mixed / f"{name}_{run.name}" / f.name).write_bytes(f.read_bytes())
    with pytest.raises(ValueError, match="mixes"):
        plot.load_encodings(tmp_path)
    with pytest.raises(FileNotFoundError):
        plot.load_encodings(tmp_path / "empty")
