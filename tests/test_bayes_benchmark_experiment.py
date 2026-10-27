"""Benchmark of the Bayes estimators on a sweep of exp_QELM_Q (issue #200)."""

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
from QELM.evaluation.bayes_optimal import benchmark_bayes_estimators
from QELM.utils import save_results

matplotlib.use("Agg")

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "QELM_Q"
L = 60


def _load(name: str) -> ModuleType:
    """Import a script of experiments/QELM_Q/, which is not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 sizes x 2 fields x 2 values of p, written as exp_QELM_Q.sh does."""
    exp = _load("exp_QELM_Q")
    root = tmp_path_factory.mktemp("sweep")
    for N in (4, 5):
        for h in (1, 2):
            for p in (0.0, 1.0):
                cfg = load_config(
                    [
                        "models=TFIM",
                        f"N={N}",
                        f"samples.L={L}",
                        f"TFIM.h={h}",
                        "evolution.t=10",
                        f"projective_measurement.p={p}",
                    ]
                )
                run_dir = root / f"N{N}_h{h}_p{p}"
                save_results(
                    exp.run(cfg),
                    run_dir,
                    "exp_QELM_Q",
                    extra_metrics=exp.EXTRA_METRICS,
                )
                (run_dir / ".hydra").mkdir()
                OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def test_every_estimator_sees_the_split_of_the_run(sweep: Path) -> None:
    """The QDA and K-NN of the benchmark are the ceilings logged by the run."""
    bench = _load("bayes_benchmark")
    run_dir = sweep / "N5_h1_p1.0"
    out = bench.benchmark_run(run_dir, seed=42)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert out["accuracy_qda"] == pytest.approx(metrics["bayes_estimate_qda"])
    assert out["accuracy_knn"] == pytest.approx(metrics["bayes_estimate_knn"])
    with np.load(run_dir / "exp_QELM_Q.npz") as data:
        X, Y, n_test = data["X"][:, :-1], data["Y"], len(data["y_test"])
    train = L - n_test
    ref = benchmark_bayes_estimators(
        X[:train], Y[:train], X[train:], Y[train:], gmm_components=(3, 5), seed=42
    )
    for name in ("qda", "gmm3", "gmm5", "svm", "knn"):
        assert out[f"accuracy_{name}"] == ref[f"accuracy_{name}"]


def test_benchmark_writes_the_table_and_the_figures(sweep: Path) -> None:
    bench = _load("bayes_benchmark")
    table, pdf, time_png = bench.benchmark_and_plot(sweep)
    assert pdf.is_file() and time_png.is_file()
    assert pdf.with_suffix(".png").is_file()

    data = json.loads(table.read_text())
    assert len(data["runs"]) == 8
    assert {(m["N"], m["h"]) for m in data["mean"]} == {
        (4, 1.0),
        (4, 2.0),
        (5, 1.0),
        (5, 2.0),
    }
    row = next(r for r in data["runs"] if r["run"] == "N4_h2_p0.0")
    metrics = json.loads((sweep / "N4_h2_p0.0" / "metrics.json").read_text())
    assert row["accuracy_ridge"] == metrics["accuracy"]

    figure = json.loads(pdf.with_suffix(".json").read_text())
    assert len(figure["panels"]) == 4
    labels = {c["label"] for c in figure["panels"][0]["curves"]}
    assert {"QDA (1 Gaussian)", "GMM (5 Gaussians)", "SVM (RBF)", "K-NN"} <= labels


def test_benchmark_needs_one_time(sweep: Path) -> None:
    bench = _load("bayes_benchmark")
    with pytest.raises(ValueError, match="No run"):
        bench.benchmark_sweep(sweep, t=3.0)
