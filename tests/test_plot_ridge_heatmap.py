"""Heatmaps of the ridge accuracy over (p, h) of exp_QELM_Q (issue #225)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import pytest
from omegaconf import OmegaConf

from QELM import load_config
from QELM.utils import save_results

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "QELM_Q"


def _load(name: str) -> ModuleType:
    """Import a script of experiments/QELM_Q/, which is not a package."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sweep of N = 4 over 2 values of p and 2 fields, 2 seeds at one point."""
    exp = _load("exp_QELM_Q")
    root = tmp_path_factory.mktemp("sweep")
    points = [(p, h, 42) for p in (0.0, 1.0) for h in (1.0, 10.0)] + [(1.0, 1.0, 43)]
    for p, h, seed in points:
        cfg = load_config(
            [
                "models=TFIM",
                "N=4",
                "samples.L=30",
                f"seed={seed}",
                f"TFIM.h={h}",
                "evolution.t=10",
                f"projective_measurement.p={p}",
            ]
        )
        run_dir = root / f"h{h}_p{p}_seed{seed}"
        save_results(
            exp.run(cfg), run_dir, "exp_QELM_Q", extra_metrics=exp.EXTRA_METRICS
        )
        (run_dir / ".hydra").mkdir()
        OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def _metric(sweep: Path, run: str, key: str) -> float:
    return float(json.loads((sweep / run / "metrics.json").read_text())[key])


def test_heatmaps_hold_the_seed_means(sweep: Path) -> None:
    plot = _load("plot_ridge_heatmap")
    png = plot.plot_sweep(sweep)
    assert png.name == "ridge_heatmap_p_h.png" and png.is_file()
    data = json.loads(png.with_suffix(".json").read_text())
    assert len(data["panels"]) >= 3
    maps = data["extra"]["maps"]
    assert set(maps) == {"accuracy_M", "accuracy_M_Z", "accuracy"}
    for metric, grid in maps.items():
        assert grid["p"] == [0.0, 1.0] and grid["h"] == [1.0, 10.0]
        # (p = 0, h = 10): one seed.
        assert grid["mean"][1][0] == pytest.approx(
            _metric(sweep, "h10.0_p0.0_seed42", metric)
        )
        # (p = 1, h = 1): mean of the two seeds.
        expected = (
            _metric(sweep, "h1.0_p1.0_seed42", metric)
            + _metric(sweep, "h1.0_p1.0_seed43", metric)
        ) / 2
        assert grid["mean"][0][1] == pytest.approx(expected)
    assert data["extra"]["N"] == 4


def test_record_alone_is_at_chance_without_measurements(sweep: Path) -> None:
    """At p = 0 the record is all zeros: [M] predicts the majority charge."""
    plot = _load("plot_ridge_heatmap")
    rows, values = plot.select_runs(plot.load_sweep(sweep, keys=plot.KEYS))
    _, extra = plot.draw_maps(rows, values, annotate=False)
    for j in range(2):
        baseline = _metric(
            sweep, f"h{extra['accuracy_M']['h'][j]}_p0.0_seed42", "accuracy_baseline"
        )
        assert extra["accuracy_M"]["mean"][j][0] == pytest.approx(baseline)


def test_select_runs_needs_one_size() -> None:
    plot = _load("plot_ridge_heatmap")
    rows = [
        {"N": 6, "t": 10.0, "final": False},
        {"N": 8, "t": 10.0, "final": False},
    ]
    with pytest.raises(ValueError, match="several N"):
        plot.select_runs(rows)
    kept, values = plot.select_runs(rows, N=8)
    assert kept == [rows[1]] and values == {"N": 8, "t": 10.0, "final": False}
    with pytest.raises(ValueError, match="No run"):
        plot.select_runs(rows, N=4)
