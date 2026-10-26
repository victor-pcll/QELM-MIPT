"""Bayes ceilings in exp_QELM_Q and their figures (issue #202)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM import load_config
from QELM.evaluation.bayes_optimal import evaluate_bayes_optimal_limit
from QELM.utils import save_results

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "QELM_Q"
L = 60


def _load(name: str) -> ModuleType:
    """Import a script of experiments/QELM_Q/, which is not a package."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overrides(N: int, p: float) -> list[str]:
    return [
        "models=TFIM",
        f"N={N}",
        f"samples.L={L}",
        "TFIM.h=1",
        "evolution.t=10",
        f"projective_measurement.p={p}",
    ]


def test_run_logs_the_ceilings_on_z() -> None:
    """The ceilings are those of evaluate_bayes_optimal_limit on <Z_k>."""
    exp = _load("exp_QELM_Q")
    out = exp.run(load_config(_overrides(4, 0.5)))
    for key in ("bayes_estimate_qda", "bayes_estimate_knn"):
        assert key in exp.EXTRA_METRICS
        assert 0.0 <= out[key] <= 1.0
    train = L - len(out["y_test"])
    Z = out["X"][:, :-1]
    expected = evaluate_bayes_optimal_limit(
        Z[:train], out["Y"][:train], Z[train:], out["Y"][train:]
    )
    assert out["bayes_estimate_qda"] == expected["bayes_estimate_qda"]
    assert out["bayes_estimate_knn"] == expected["bayes_estimate_knn"]


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sweep of 2 sizes x 2 values of p, written as exp_QELM_Q.sh does."""
    exp = _load("exp_QELM_Q")
    root = tmp_path_factory.mktemp("sweep")
    for N in (4, 5):
        for p in (0.0, 1.0):
            cfg = load_config(_overrides(N, p))
            run_dir = root / f"N{N}_p{p}"
            save_results(
                exp.run(cfg), run_dir, "exp_QELM_Q", extra_metrics=exp.EXTRA_METRICS
            )
            (run_dir / ".hydra").mkdir()
            OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def test_plot_writes_the_ceilings_and_the_contours(sweep: Path) -> None:
    plot = _load("plot_bayes_ceiling")
    accuracy_png, contours_png = plot.plot_sweep(sweep)
    assert accuracy_png.is_file() and contours_png.is_file()

    data = json.loads(accuracy_png.with_suffix(".json").read_text())
    curves = {c["label"]: c for c in data["panels"][0]["curves"]}
    metrics = json.loads((sweep / "N4_p1.0" / "metrics.json").read_text())
    for name, key in (
        ("ridge", "accuracy"),
        ("QDA", "bayes_estimate_qda"),
        ("K-NN", "bayes_estimate_knn"),
    ):
        assert curves[f"N = 4, {name}"]["x"] == [0.0, 1.0]
        assert curves[f"N = 4, {name}"]["y"][1] == pytest.approx(metrics[key])
    assert data["extra"] == {"h": 1.0, "t": 10.0, "final": False}

    contours = json.loads(contours_png.with_suffix(".json").read_text())
    assert len(contours["panels"]) == 4
    assert set(contours["extra"]) == {"N4_p0", "N4_p1", "N5_p0", "N5_p1"}
    labels = {c["label"] for c in contours["panels"][0]["curves"]}
    assert {"Q = 0", "Q = 1", "Q = 1, QDA 1 sigma", "Q = 0, QDA 2 sigma"} <= labels


def test_contours_refit_the_qda_of_the_run(sweep: Path) -> None:
    """The refitted QDA scores bayes_estimate_qda of the run on its test set."""
    plot = _load("plot_bayes_ceiling")
    qda, X_test, y_test = plot.fit_run(sweep / "N5_p1.0")
    metrics = json.loads((sweep / "N5_p1.0" / "metrics.json").read_text())
    assert qda.score(X_test, y_test) == pytest.approx(metrics["bayes_estimate_qda"])
    assert X_test.shape[1] == 5 - 1


def test_plot_without_ceilings_and_with_a_choice_of_p(sweep: Path) -> None:
    plot = _load("plot_bayes_ceiling")
    rows, values = plot.select_runs(plot.load_sweep(sweep, keys=plot.KEYS))
    fig = plot.draw_ceilings(rows, values, ceilings=False)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert not any("QDA" in label or "K-NN" in label for label in labels)
    fig, extra = plot.draw_contours(sweep, rows, values, p_values=[1.0])
    assert set(extra) == {"N4_p1", "N5_p1"}


def test_select_runs_needs_one_field() -> None:
    plot = _load("plot_bayes_ceiling")
    rows = [
        {"h": 1.0, "t": 10.0, "final": False},
        {"h": 2.0, "t": 10.0, "final": False},
    ]
    with pytest.raises(ValueError, match="several h"):
        plot.select_runs(rows)
    kept, values = plot.select_runs(rows, h=2.0)
    assert kept == [rows[1]] and values["h"] == 2.0
    with pytest.raises(ValueError, match="No run"):
        plot.select_runs(rows, h=3.0)
    np.testing.assert_equal(len(kept), 1)
