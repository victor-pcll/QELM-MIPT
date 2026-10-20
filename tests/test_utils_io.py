"""QELM.utils.save_results, the save shared by the experiment scripts (issue #87)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from QELM.utils import save_results


def _results(features: str = "X", shape: tuple[int, ...] = (10, 4)) -> dict:
    return {
        features: np.ones(shape),
        "y_test": np.zeros(3),
        "W": np.arange(4.0),
        "nmse": 0.25,
        "r2": 0.5,
        "label": "not an array",
    }


def test_npz_is_named_after_the_experiment(tmp_path: Path) -> None:
    """Two experiments in the same folder write two different .npz files."""
    a = save_results(_results(), tmp_path, "exp_A")
    b = save_results(_results(), tmp_path, "exp_B")
    assert a == tmp_path / "exp_A.npz" and b == tmp_path / "exp_B.npz"
    assert a.exists() and b.exists()


def test_arrays_and_metrics_are_written(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "run"
    path = save_results(_results(), out, "exp")
    with np.load(path) as data:
        assert sorted(data.files) == ["W", "X", "y_test"]
        np.testing.assert_array_equal(data["W"], np.arange(4.0))
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics == {
        "nmse": 0.25,
        "r2": 0.5,
        "n_samples": 10,
        "n_test": 3,
        "n_features": 4,
    }


def test_features_key_and_multi_axis_features(tmp_path: Path) -> None:
    """projective_readout passes Phi of shape (L, n_steps, N - 1)."""
    save_results(_results("Phi", (10, 3, 2)), tmp_path, "proj", features="Phi")
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_samples"] == 10 and metrics["n_features"] == 6


def test_extra_metrics_are_written(tmp_path: Path) -> None:
    """exp_QELM_M adds the mean ancilla entropies to metrics.json."""
    results = {**_results(), "S_vN_mean": np.float64(0.5), "S_renyi_1_mean": 0.5}
    save_results(
        results, tmp_path, "exp", extra_metrics=["S_vN_mean", "S_renyi_1_mean"]
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["S_vN_mean"] == 0.5 and metrics["S_renyi_1_mean"] == 0.5
    assert {"nmse", "r2", "n_samples", "n_test", "n_features"} <= set(metrics)


def test_missing_extra_metric_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="S_vN_mean"):
        save_results(_results(), tmp_path, "exp", extra_metrics=["S_vN_mean"])


@pytest.mark.parametrize("name", ["", "exp.npz", "sub/exp", "../exp"])
def test_bad_exp_name_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="exp_name"):
        save_results(_results(), tmp_path, name)
