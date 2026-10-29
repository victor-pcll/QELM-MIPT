"""Bayes ceilings in exp_QELM_Q and their figures (issue #202)."""

from __future__ import annotations

import importlib.util
import json
import sys
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
    # Registered first: a dataclass looks its module up in sys.modules.
    sys.modules[name] = module
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
    out = exp.run(load_config(_overrides(3, 0.5)))
    for key in ("bayes_estimate_qda", "bayes_estimate_gmm", "bayes_estimate_knn"):
        assert key in exp.EXTRA_METRICS
        assert 0.0 <= out[key] <= 1.0
    train = L - len(out["y_test"])
    Z = out["X"][:, :-1]
    expected = evaluate_bayes_optimal_limit(
        Z[:train], out["Y"][:train], Z[train:], out["Y"][train:], seed=42
    )
    for key in ("bayes_estimate_qda", "bayes_estimate_gmm", "bayes_estimate_knn"):
        assert out[key] == expected[key]


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sweep of 2 sizes x 2 values of p, written as exp_QELM_Q.sh does."""
    exp = _load("exp_QELM_Q")
    root = tmp_path_factory.mktemp("sweep")
    for N in (3, 4):  # system qubits (issue #235)
        for p in (0.0, 1.0):
            cfg = load_config(_overrides(N, p))
            run_dir = root / f"N{N}_p{p}"
            save_results(
                exp.run(cfg), run_dir, "exp_QELM_Q", extra_metrics=exp.EXTRA_METRICS
            )
            (run_dir / ".hydra").mkdir()
            OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def test_plot_writes_the_ceilings_and_the_pca_figures(sweep: Path) -> None:
    plot = _load("plot_bayes_ceiling")
    pngs = plot.plot_sweep(sweep)
    assert set(pngs) == {"ceiling", "qda", "gmm", "ridge", "knn"}
    for png in pngs.values():
        assert png.is_file() and png.with_suffix(".json").is_file()

    data = json.loads(pngs["ceiling"].with_suffix(".json").read_text())
    curves = {c["label"]: c for c in data["panels"][0]["curves"]}
    metrics = json.loads((sweep / "N3_p1.0" / "metrics.json").read_text())
    for name, key in (
        ("ridge", "accuracy"),
        ("QDA", "bayes_estimate_qda"),
        ("GMM", "bayes_estimate_gmm"),
        ("K-NN", "bayes_estimate_knn"),
    ):
        assert curves[f"N = 3, {name}"]["x"] == [0.0, 1.0]
        assert curves[f"N = 3, {name}"]["y"][1] == pytest.approx(metrics[key])
    assert data["extra"] == {"h": 1.0, "t": 10.0, "final": False}

    panels = {"N3_p0", "N3_p1", "N4_p0", "N4_p1"}
    expected_labels = {
        "qda": {"Q = 0", "Q = 1", "Q = 1, QDA 1 sigma", "Q = 0, QDA 2 sigma"},
        "gmm": {"Q = 0", "Q = 1", "Q = 0, GMM components (1 sigma)"},
        "ridge": {"Q = 0", "Q = 1", "misclassified"},
        "knn": {"Q = 0", "Q = 1", "misclassified"},
    }
    for method, labels in expected_labels.items():
        figure = json.loads(pngs[method].with_suffix(".json").read_text())
        assert len(figure["panels"]) == 4
        assert set(figure["extra"]) == panels
        assert labels <= {c["label"] for c in figure["panels"][0]["curves"]}


def test_pca_figures_refit_the_models_of_the_run(sweep: Path) -> None:
    """The refitted models score the ceilings and the ridge accuracy of the run."""
    plot = _load("plot_bayes_ceiling")
    run_dir = sweep / "N4_p1.0"
    fit = plot.fit_run(run_dir, seed=42)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    X, y = fit.X_test, fit.y_test
    assert X.shape[1] == 4  # the <Z_k> of the 4 system qubits
    assert fit.qda.score(X, y) == pytest.approx(metrics["bayes_estimate_qda"])
    assert fit.gmm.score(X, y) == pytest.approx(metrics["bayes_estimate_gmm"])
    assert np.mean(fit.knn.predict(X) == y) == pytest.approx(
        metrics["bayes_estimate_knn"]
    )
    assert np.mean(fit.predict_ridge(X) == y) == pytest.approx(metrics["accuracy"])
    # The panels report the same accuracies as the regions they draw.
    figure = json.loads((sweep / "ridge_regions.json").read_text())
    assert figure["extra"]["N4_p1"]["accuracy"] == pytest.approx(metrics["accuracy"])
    figure = json.loads((sweep / "knn_regions.json").read_text())
    assert figure["extra"]["N4_p1"]["accuracy"] == pytest.approx(
        metrics["bayes_estimate_knn"]
    )


def test_plot_without_ceilings_and_with_a_choice_of_p(sweep: Path) -> None:
    plot = _load("plot_bayes_ceiling")
    rows, values = plot.select_runs(plot.load_sweep(sweep, keys=plot.KEYS))
    fig = plot.draw_ceilings(rows, values, ceilings=False)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert not any(name in label for label in labels for name in ("QDA", "K-NN"))
    fig, extra = plot.draw_plane(sweep, rows, values, "gmm", p_values=[1.0])
    assert set(extra) == {"N3_p1", "N4_p1"}


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
