"""PCA figure of the feature spaces [M], [M, Z], [Z] of exp_QELM_Q (issue #225)."""

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
from QELM.evaluation.pca import fit_pca
from QELM.utils import save_results

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "experiments" / "QELM_Q"
L = 40
N = 3  # system qubits (issue #235)


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
    """A sweep of N = 3 at p = 0, 0.5, 1, written as exp_QELM_Q.sh does."""
    exp = _load("exp_QELM_Q")
    root = tmp_path_factory.mktemp("sweep")
    for p in (0.0, 0.5, 1.0):
        cfg = load_config(
            [
                "models=TFIM",
                f"N={N}",
                f"samples.L={L}",
                f"projective_measurement.p={p}",
            ]
        )
        run_dir = root / f"N{N}_p{p}"
        save_results(
            exp.run(cfg), run_dir, "exp_QELM_Q", extra_metrics=exp.EXTRA_METRICS
        )
        (run_dir / ".hydra").mkdir()
        OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def test_project_run_fits_on_the_training_split(sweep: Path) -> None:
    plot = _load("plot_pca")
    run_dir = sweep / f"N{N}_p0.5"
    projected, y, pca = plot.project_run(run_dir, "M_Z")
    with np.load(run_dir / plot.NPZ) as data:
        X = data["features_M_Z"][:, :-1]
        n_test = len(data["y_test"])
        np.testing.assert_array_equal(y, data["Y"][-n_test:])
    expected = fit_pca(X[: L - n_test], 2)
    np.testing.assert_allclose(pca.components, expected.components)
    np.testing.assert_allclose(projected, expected.transform(X[L - n_test :]))
    assert projected.shape == (n_test, 2)
    every, y_all, _ = plot.project_run(run_dir, "M_Z", samples="all")
    assert every.shape == (L, 2) and len(y_all) == L


def test_record_at_p0_is_degenerate(sweep: Path) -> None:
    """Without measurements the record is all zeros (issue #225)."""
    plot = _load("plot_pca")
    projected, _, pca = plot.project_run(sweep / f"N{N}_p0.0", "M")
    assert pca.degenerate
    np.testing.assert_array_equal(projected, 0.0)
    # [M, Z] at p = 0 is [Z] padded with zeros: same variances.
    _, _, mz = plot.project_run(sweep / f"N{N}_p0.0", "M_Z")
    _, _, z = plot.project_run(sweep / f"N{N}_p0.0", "Z")
    np.testing.assert_allclose(mz.variances, z.variances)
    _, _, record = plot.project_run(sweep / f"N{N}_p1.0", "M")
    assert not record.degenerate


def test_plot_writes_the_grid(sweep: Path) -> None:
    plot = _load("plot_pca")
    pngs = plot.plot_sweep(sweep, N=N)
    assert set(pngs) == {"pca", "sum", "ridge"}
    for path in pngs.values():
        assert path.is_file() and path.with_suffix(".json").is_file()
    png = pngs["pca"]
    assert png.name == "pca_feature_spaces.png"
    data = json.loads(png.with_suffix(".json").read_text())
    # Rows: feature spaces, columns: p.
    assert len(data["panels"]) == 9
    panels = data["extra"]["panels"]
    assert set(panels) == {f"{s}_p{p}" for s in ("M", "M_Z", "Z") for p in (0, 0.5, 1)}
    assert panels["M_p0"]["degenerate"] is True
    assert panels["Z_p1"]["degenerate"] is False
    metrics = json.loads((sweep / f"N{N}_p1.0" / "metrics.json").read_text())
    assert panels["Z_p1"]["accuracy"] == pytest.approx(metrics["accuracy"])
    assert panels["M_p1"]["accuracy_M"] == pytest.approx(metrics["accuracy_M"])
    labels = {c["label"] for c in data["panels"][-1]["curves"]}
    assert labels == {"Q = 0", "Q = 1"}


def test_plot_in_3d_and_with_a_choice_of_p(sweep: Path) -> None:
    plot = _load("plot_pca")
    png = plot.plot_sweep(sweep, N=N, p_values=[1.0], n_components=3)["pca"]
    assert png.name == "pca_feature_spaces_3d.png" and png.is_file()
    data = json.loads(png.with_suffix(".json").read_text())
    assert set(data["extra"]["panels"]) == {"M_p1", "M_Z_p1", "Z_p1"}
    sectors = data["extra"]["panels"]["Z_p1"]["sectors"]
    assert all(len(point) == 3 for points in sectors.values() for point in points)


def test_select_runs_needs_one_field_and_the_size() -> None:
    plot = _load("plot_pca")
    rows = [
        {"N": 8, "h": 1.0, "t": 10.0, "final": False},
        {"N": 8, "h": 2.0, "t": 10.0, "final": False},
        {"N": 6, "h": 3.0, "t": 10.0, "final": False},
    ]
    with pytest.raises(ValueError, match="several h"):
        plot.select_runs(rows, 8)
    kept, values = plot.select_runs(rows, 8, h=2.0)
    assert kept == [rows[1]] and values == {"N": 8, "h": 2.0, "t": 10.0, "final": False}
    with pytest.raises(ValueError, match="No run"):
        plot.select_runs(rows, 4)


def test_sum_direction() -> None:
    plot = _load("plot_pca")
    u = plot.sum_direction(5, 3)
    np.testing.assert_allclose(u, [1, 1, 1, 0, 0] / np.sqrt(3.0))
    assert np.linalg.norm(u) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        plot.sum_direction(2, 3)
    with pytest.raises(ValueError):
        plot.sum_direction(2, 0)


@pytest.mark.parametrize("space", ["M", "M_Z", "Z"])
def test_refit_ridge_gives_the_logged_accuracy(sweep: Path, space: str) -> None:
    """The refitted readout is the one of the run (same split, same alpha)."""
    plot = _load("plot_pca")
    for p in (0.0, 0.5, 1.0):
        run_dir = sweep / f"N{N}_p{p}"
        fit = plot.refit_ridge(run_dir, space)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert fit["accuracy"] == pytest.approx(metrics[plot.SPACES[space][1]])
        assert fit["n_record"] == (0 if space == "Z" else metrics_record(run_dir))
    if space == "Z":
        with np.load(sweep / f"N{N}_p1.0" / plot.NPZ) as data:
            np.testing.assert_allclose(fit["W"], np.ravel(data["W"]))


def metrics_record(run_dir: Path) -> int:
    with np.load(run_dir / "exp_QELM_Q.npz") as data:
        return int(np.prod(data["M"].shape[1:]))


def test_record_sum_at_p1_is_the_charge(sweep: Path) -> None:
    """At p = 1 every round sums to N - 2 Q: one value of x @ u per sector."""
    plot = _load("plot_pca")
    fit = plot.refit_ridge(sweep / f"N{N}_p1.0", "M")
    u = plot.sum_direction(fit["X_test"].shape[1], fit["n_record"])
    n_rounds = fit["n_record"] // N
    total = fit["X_test"] @ u * np.sqrt(fit["n_record"])
    np.testing.assert_allclose(total, n_rounds * (N - 2 * fit["y_test"]))


def test_sum_and_ridge_figures(sweep: Path) -> None:
    plot = _load("plot_pca")
    rows, values = plot.select_runs(plot.load_sweep(sweep, keys=plot.KEYS), N)
    _, extra = plot.draw_sum_direction(sweep, rows, values)
    assert set(extra) == {f"{s}_p{p:g}" for s in ("M", "M_Z") for p in (0, 0.5, 1)}
    assert extra["M_p0"]["variance_share_u"] == 0.0
    assert 0.0 < extra["M_p1"]["variance_share_u"] < 1.0
    assert -1.0 <= extra["M_p1"]["cos_W_u"] <= 1.0
    _, extra = plot.draw_ridge_outputs(sweep, rows, values, p_values=[1.0])
    assert set(extra) == {"M_p1", "M_Z_p1", "Z_p1"}
    panel = extra["Z_p1"]
    outputs = [v for values in panel["outputs"].values() for v in values]
    with np.load(sweep / f"N{N}_p1.0" / plot.NPZ) as data:
        assert len(outputs) == len(data["y_test"])
