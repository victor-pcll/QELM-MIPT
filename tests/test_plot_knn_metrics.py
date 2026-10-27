"""The K-NN comparison figures, experiments/QELM_Q/plot_knn_metrics.py (#213)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/QELM_Q/plot_knn_metrics.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_knn_metrics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, N: int, h: float, p: float, final: bool, seed: int) -> None:
    """A run folder with a config and made-up metrics that encode the run."""
    run = root / f"N{N}_h{h}_final{final}_seed{seed}_p{p}"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text(
        f"N: {N}\nseed: {seed}\nTFIM:\n  h: {h}\n"
        f"projective_measurement:\n  p: {p}\n"
        f"  final_measurement: {str(final).lower()}\n"
    )
    metrics = {
        "accuracy": 1.0 - 0.1 * p - 0.01 * seed,
        "bayes_estimate_knn": 0.9 + 0.01 * seed,
        "bayes_estimate_knn_mahalanobis": 0.8 + 0.1 * final,
        "knn_covariance_cond": 10.0 ** (1 + p + h),
    }
    (run / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def sweep(tmp_path: Path) -> Path:
    for seed in (0, 2):
        for h in (1.0, 10.0):
            for p in (0.0, 0.5, 1.0):
                _write_run(tmp_path, 8, h, p, False, seed)
            _write_run(tmp_path, 8, h, 1.0, True, seed)
    return tmp_path


def test_summary_averages_the_seeds(sweep: Path) -> None:
    summary = _load_script().load_summary(sweep)
    assert len(summary) == 2 * (3 + 1)
    row = next(r for r in summary if r["h"] == 1.0 and r["p"] == 0.5 and not r["final"])
    assert row["n_runs"] == 2
    assert row["accuracy_mean"] == pytest.approx(1.0 - 0.05 - 0.01)
    assert row["bayes_estimate_knn_mean"] == pytest.approx(0.91)
    final = next(r for r in summary if r["h"] == 10.0 and r["final"])
    assert final["bayes_estimate_knn_mahalanobis_mean"] == pytest.approx(0.9)
    assert final["knn_covariance_cond_mean"] == pytest.approx(10.0**12)


def test_rejects_several_sizes(sweep: Path) -> None:
    _write_run(sweep, 6, 1.0, 0.0, False, 0)
    with pytest.raises(ValueError, match="one N"):
        _load_script().load_summary(sweep)


def test_writes_the_three_figures(sweep: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    _load_script().main([str(sweep), "--out", str(out)])
    for name in ("knn_metrics_vs_p", "knn_metrics_vs_h", "knn_covariance_cond"):
        assert (out / f"{name}.png").is_file()
        assert (out / f"{name}.json").is_file()
    data = json.loads((out / "knn_metrics_vs_p.json").read_text())
    assert len(data["extra"]["summary"]) == 8
    # Panel h = 1: the ridge curve over p, then its final-round marker.
    curves = data["panels"][0]["curves"]
    ridge = next(c for c in curves if c["label"] == "ridge")
    np.testing.assert_allclose(ridge["x"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ridge["y"], [0.99, 0.94, 0.89])
    final = next(c for c in curves if c["label"] == "ridge, final round")
    np.testing.assert_allclose(final["x"], [1.0])
    cond = json.loads((out / "knn_covariance_cond.json").read_text())
    marker = next(
        c for c in cond["panels"][0]["curves"] if c["label"] == "h = 10 J, final round"
    )
    np.testing.assert_allclose(marker["y"], [10.0**12])
