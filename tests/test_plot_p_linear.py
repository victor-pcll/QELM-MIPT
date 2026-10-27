"""The p_linear* figures from saved tables, plot_p_linear.py (#227)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/QELM_Q/plot_p_linear.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_p_linear", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tables(tmp_path: Path) -> tuple[Path, Path]:
    """The two figure JSONs of notebook 06, with made-up valleys at 0.6, 0.5."""
    metrics, binder = [], []
    for N, p0 in ((8, 0.6), (10, 0.5)):
        for h in (1, 2):
            for p in np.linspace(0.0, 1.0, 11):
                key = {"N": N, "h": h, "p": float(p), "n_runs": 5}
                metrics.append(
                    key
                    | {
                        "accuracy_M_Z_mean": 0.6 + (p - p0) ** 2,
                        "accuracy_M_Z_sem": 0.01,
                        "r2_M_Z_mean": 0.1 + (p - p0) ** 2,
                        "r2_M_Z_sem": 0.01,
                    }
                )
                binder.append(
                    key | {"binder_M_Z_mean": 0.6 - p, "binder_M_Z_sem": 0.02}
                )
    paths = tmp_path / "metrics_vs_p_N.json", tmp_path / "binder_vs_p_N.json"
    for path, table in zip(paths, (metrics, binder), strict=True):
        path.write_text(json.dumps({"extra": {"table": table}}))
    return paths


def test_plot_p_linear_saves_both_figures(
    tables: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib.use("Agg")
    script = _load_script()
    out = tmp_path / "out"
    script.main([str(tables[0]), str(tables[1]), "--out", str(out), "--n-boot", "50"])
    assert "estimator" in capsys.readouterr().out
    for name in ("p_linear_curves", "p_linear_vs_N"):
        assert (out / f"{name}.png").is_file()
        data = json.loads((out / f"{name}.json").read_text())
        estimates = data["extra"]["estimates"]
        assert len(estimates) == 2 * 2 * 3
    by = {(e["N"], e["h"], e["estimator"]): e for e in estimates}
    assert by[8, 1, "accuracy"]["p_grid"] == pytest.approx(0.6)
    assert by[10, 2, "residual"]["p_grid"] == pytest.approx(0.5)
    assert by[8, 1, "binder"]["at_edge"]
    curves = json.loads((out / "p_linear_curves.json").read_text())
    # One panel per (h, N), three curves each.
    assert len(curves["panels"]) == 4
    assert all(len(panel["curves"]) == 3 for panel in curves["panels"])


def test_plot_p_linear_without_binder_defaults_to_the_sweep_folder(
    tables: tuple[Path, Path],
) -> None:
    matplotlib.use("Agg")
    estimates = _load_script().plot_p_linear(tables[0], n_boot=20)
    assert {e["estimator"] for e in estimates} == {"accuracy", "residual"}
    assert (tables[0].parent / "p_linear_vs_N.png").is_file()
