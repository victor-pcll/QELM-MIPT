"""The mixture ceiling over the (p, h) plane of an exp_QELM_Q sweep (issue #215)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM import load_config
from QELM.evaluation.gmm_scan import SCAN_COMPONENTS
from QELM.utils import load_sweep, save_results

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "QELM_Q"
L = 60
SEEDS = (42, 43)
HS = (1.0, 2.0)
PS = (0.0, 1.0)


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
    """N = 4, 2 fields x 2 values of p x 2 seeds, written as exp_QELM_Q.sh does."""
    exp = _load("exp_QELM_Q")
    root = tmp_path_factory.mktemp("sweep")
    for h in HS:
        for seed in SEEDS:
            for p in PS:
                cfg = load_config(
                    [
                        "models=TFIM",
                        "N=4",
                        f"samples.L={L}",
                        f"TFIM.h={h}",
                        "evolution.t=10",
                        f"seed={seed}",
                        f"projective_measurement.p={p}",
                    ]
                )
                run_dir = root / f"N4_h{h}_seed{seed}_p{p}"
                save_results(
                    exp.run(cfg),
                    run_dir,
                    "exp_QELM_Q",
                    extra_metrics=exp.EXTRA_METRICS,
                )
                (run_dir / ".hydra").mkdir()
                OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def test_scan_run_reproduces_the_logged_ceilings(sweep: Path) -> None:
    gmm = _load("gmm_phase_diagram")
    for row in load_sweep(sweep, keys=gmm.KEYS):
        scan = gmm.scan_run(sweep / row["run"], row)
        assert scan["accuracy_gmm_K3"] == row["bayes_estimate_gmm"]
        assert scan["accuracy_knn"] == row["bayes_estimate_knn"]
        assert scan["gap_gmm_ridge"] == pytest.approx(
            scan["accuracy_gmm_bic"] - row["accuracy"]
        )
        assert scan["gap_gmm_knn"] == pytest.approx(
            scan["accuracy_gmm_bic"] - scan["accuracy_knn"]
        )


def test_scan_run_stops_on_a_mismatch(sweep: Path, tmp_path: Path) -> None:
    """A refit that does not give the logged ceiling is an error, not a figure."""
    gmm = _load("gmm_phase_diagram")
    row = load_sweep(sweep, keys=gmm.KEYS)[0]
    copy = tmp_path / row["run"]
    shutil.copytree(sweep / row["run"], copy)
    row["bayes_estimate_knn"] = row["bayes_estimate_knn"] - 0.25
    with pytest.raises(RuntimeError, match="bayes_estimate_knn"):
        gmm.scan_run(copy, row)


def test_analyze_sweep_writes_the_figures_and_the_table(sweep: Path) -> None:
    gmm = _load("gmm_phase_diagram")
    out = gmm.analyze_sweep(sweep)
    for key in ("heatmaps", "components", "vs_K", "vs_p"):
        assert out[key].exists()
        assert out[key].with_suffix(".json").exists()
    table = json.loads(out["table"].read_text())
    assert table["fixed"]["N"] == 4
    summary = table["summary"]
    assert len(summary) == len(HS) * len(PS)
    assert {(r["h"], r["p"]) for r in summary} == {(h, p) for h in HS for p in PS}
    for r in summary:
        assert r["n_runs"] == len(SEEDS)
        for K in SCAN_COMPONENTS:
            assert 0.0 <= r[f"accuracy_gmm_K{K}_mean"] <= 1.0
        assert 1.0 <= r["k_bic_mean_mean"] <= 8.0
    heatmaps = json.loads(out["heatmaps"].with_suffix(".json").read_text())
    # Six maps over the 2 x 2 grid of (p, h).
    meshes = [m for panel in heatmaps["panels"] for m in panel["heatmaps"]]
    assert len(meshes) == len(gmm.MAPS)


def test_select_runs_needs_one_size(sweep: Path) -> None:
    gmm = _load("gmm_phase_diagram")
    rows = [{"N": 4, "t": 10.0, "final": False}, {"N": 6, "t": 10.0, "final": False}]
    with pytest.raises(ValueError, match="several N"):
        gmm.select_runs(rows)
    kept, values = gmm.select_runs(rows, N=6)
    assert values == {"N": 6, "t": 10.0, "final": False}
    assert len(kept) == 1
    assert np.isclose(kept[0]["t"], 10.0)
