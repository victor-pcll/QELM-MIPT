"""The sample-to-parameter ratio sweep, experiments/QELM_M/exp_QELM_M_ratio.py (#96)."""

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
from QELM.training import RidgeReadout

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M_ratio.py"
BASH_SCRIPT = SCRIPT.with_suffix(".sh")
N, N_TEST = 4, 20
# 5 measurement steps (the default) x (N - 1) system qubits + the bias.
N_PARAMS = 5 * (N - 1) + 1
OVERRIDES = ["models=TFIM", f"N={N}", f"experiment.n_test={N_TEST}"]


def _load_script() -> ModuleType:
    """Import experiments/QELM_M/exp_QELM_M_ratio.py, which is not a package."""
    spec = importlib.util.spec_from_file_location("exp_QELM_M_ratio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*overrides: str):
    return load_config([*OVERRIDES, *overrides], config_name="QELM_M_ratio")


@pytest.mark.parametrize(
    ("ratio", "train"), [(0.5, 8), (1.0, 16), (1.5, 24), (10.0, 160), (0.01, 1)]
)
def test_sample_sizes(ratio: float, train: int) -> None:
    """n_train = round(ratio * n_params), at least 1; the test set is added."""
    assert _load_script().sample_sizes(ratio, 16, 200) == (train + 200, train)


@pytest.mark.parametrize(
    ("ratio", "n_features", "n_test"), [(0.0, 16, 10), (-1.0, 16, 10), (1.0, 0, 10)]
)
def test_sample_sizes_reject_bad_input(
    ratio: float, n_features: int, n_test: int
) -> None:
    with pytest.raises(ValueError):
        _load_script().sample_sizes(ratio, n_features, n_test)


@pytest.mark.parametrize(("n", "steps"), [(3, 5), (4, 5), (4, 2)])
def test_n_params_matches_the_features(n: int, steps: int) -> None:
    """The weight count computed from the config is the column count of X."""
    module = _load_script()
    cfg = load_config(
        [
            "models=TFIM",
            f"N={n}",
            "experiment.n_test=5",
            "experiment.alpha_ratio=1.0",
            f"projective_measurement.nbr_time_evolution={steps}",
        ],
        config_name="QELM_M_ratio",
    )
    expected = steps * (n - 1) + 1
    assert module.n_params(cfg) == expected
    out = module.run(cfg)
    assert out["X"].shape == (expected + 5, expected)
    assert out["n_params"] == expected


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_run_sets_the_sample_counts(ratio: float) -> None:
    """samples.L and samples.train_size follow the ratio, whatever the
    default samples.L, and the test set keeps n_test samples."""
    cfg = _cfg(f"experiment.alpha_ratio={ratio}")
    out = _load_script().run(cfg)
    train = round(ratio * N_PARAMS)
    assert cfg.samples.train_size == train
    assert cfg.samples.L == train + N_TEST
    assert out["X"].shape == (train + N_TEST, N_PARAMS)
    assert out["y_test"].shape == (N_TEST,)
    assert out["n_train"] == train
    assert out["train_ratio"] == pytest.approx(train / N_PARAMS)
    assert out["alpha_ratio"] == ratio


@pytest.mark.parametrize("p", [1.0, 0.5])
def test_underdetermined_regime_fits(p: float) -> None:
    """alpha_ratio < 1: fewer training samples than weights, and the ridge
    penalty still gives finite weights and predictions (issue #96)."""
    out = _load_script().run(
        _cfg(
            "experiment.alpha_ratio=0.5",
            "training.alpha=1e-3",
            f"projective_measurement.p={p}",
        )
    )
    assert out["n_train"] < out["n_params"]
    assert np.all(np.isfinite(out["W"]))
    assert np.all(np.isfinite(out["y_pred"]))
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


def test_ridge_solves_fewer_samples_than_features() -> None:
    """RidgeReadout on n < d: Phi^T Phi has rank n < d, yet the penalised
    system is solved, the weights are the minimum-norm interpolant in the
    limit alpha -> 0 and they fit the training data."""
    rng = np.random.default_rng(0)
    phi = rng.choice([-1.0, 1.0], size=(5, 12))
    y = rng.normal(size=5)
    assert np.linalg.matrix_rank(phi.T @ phi) == 5
    solver = RidgeReadout({"alpha": 1e-8})
    solver.fit(phi, y)
    assert solver.W is not None
    np.testing.assert_allclose(solver.W, np.linalg.pinv(phi) @ y, atol=1e-6)
    np.testing.assert_allclose(solver.predict(phi), y, atol=1e-6)


def test_underdetermined_without_penalty_is_rejected() -> None:
    """training.alpha = 0 with fewer samples than weights has no unique
    solution: the script stops with a clear error instead of a singular
    solve."""
    with pytest.raises(ValueError, match="training.alpha > 0"):
        _load_script().run(_cfg("experiment.alpha_ratio=0.5", "training.alpha=0.0"))


def test_ratio_and_ridge_penalty_are_independent() -> None:
    """experiment.alpha_ratio sets the sample count and training.alpha the
    penalty; changing one leaves the other unchanged (issue #96)."""
    a = _cfg("experiment.alpha_ratio=2.0", "training.alpha=1e-3")
    b = _cfg("experiment.alpha_ratio=2.0", "training.alpha=0.5")
    c = _cfg("experiment.alpha_ratio=0.5", "training.alpha=1e-3")
    module = _load_script()
    out_a, out_b, out_c = module.run(a), module.run(b), module.run(c)
    assert a.samples.L == b.samples.L != c.samples.L
    assert a.training.alpha == c.training.alpha == 1e-3
    assert b.training.alpha == 0.5
    # Same samples (same seed and sizes), different penalties.
    np.testing.assert_array_equal(out_a["X"], out_b["X"])
    assert not np.allclose(out_a["W"], out_b["W"])
    assert out_c["n_train"] == 8


def test_ratio_in_the_training_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="experiment.alpha_ratio"):
        _load_script().run(_cfg("+training.alpha_ratio=0.5"))


def test_no_ratio_keeps_samples_L() -> None:
    """experiment.alpha_ratio=null runs exp_QELM_M with samples.L."""
    cfg = _cfg("experiment.alpha_ratio=null", "samples.L=30")
    out = _load_script().run(cfg)
    assert out["X"].shape == (30, N_PARAMS)
    assert out["n_train"] == 21  # int(0.7 * 30)
    assert np.isnan(out["alpha_ratio"])


def test_feature_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run stops if X does not have the computed number of weights."""
    module = _load_script()
    run = module.base.run

    def one_column_short(cfg):
        out = run(cfg)
        return {**out, "X": out["X"][:, 1:]}

    monkeypatch.setattr(module.base, "run", one_column_short)
    with pytest.raises(RuntimeError, match="features"):
        module.run(_cfg("experiment.alpha_ratio=1.0"))


def test_reproducible_with_the_seed() -> None:
    module = _load_script()
    a = module.run(_cfg("experiment.alpha_ratio=1.5", "projective_measurement.p=0.5"))
    b = module.run(_cfg("experiment.alpha_ratio=1.5", "projective_measurement.p=0.5"))
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["Y"], b["Y"])
    assert a["nmse"] == b["nmse"]


def test_bash_launcher_dry_run(tmp_path: Path) -> None:
    """The launcher writes one run folder per ratio, with the sizes used in
    metrics.json and config_used.yaml."""
    assert os.access(BASH_SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "N_VALUES": "3",
            "N_TEST": "10",
            "ALPHA_RATIOS": "0.5 2.0",
            "VIRTUAL_ENV": str(tmp_path / "other-project" / ".venv"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 1 x 1 x 1 x 2 runs" in result.stdout
    runs = sorted(tmp_path.glob("exp_QELM_M_ratio/*/*"))
    assert [run.name for run in runs] == [
        "N3_seed42_p1.0_a0.5",
        "N3_seed42_p1.0_a2.0",
    ]
    # N = 3: 5 steps x 2 system qubits + 1 = 11 weights.
    for run, train in zip(runs, (6, 22), strict=True):
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["n_features"] == metrics["n_params"] == 11
        assert metrics["n_train"] == train
        assert metrics["n_samples"] == train + 10
        assert metrics["n_test"] == 10
        used = OmegaConf.load(run / "config_used.yaml")
        assert used.samples.L == train + 10
        assert used.training.alpha == 1e-3
        assert (run / "exp_QELM_M_ratio.npz").is_file()
