"""The figures of the dt sweep of the Phi(d'/2) collapse (issue #253)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_Q" / "plot_dt_collapse.py"
EXP_Q = REPO_ROOT / "experiments" / "QELM_Q" / "exp_QELM_Q.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*overrides: str) -> None:
    subprocess.run(
        [sys.executable, str(EXP_Q), *overrides], check=True, capture_output=True
    )


def _sweep(root: Path, t_values: tuple[float, ...], *extra: str) -> None:
    for N in (3, 4):
        for t in t_values:
            for p in (0.0, 0.5):
                _run(
                    f"N={N}",
                    "samples.L=40",
                    "samples.balanced=true",
                    "state_representation=vector",
                    "projective_measurement.final_measurement=true",
                    f"projective_measurement.p={p}",
                    f"evolution.t={t}",
                    *extra,
                    f"hydra.run.dir={root / f'N{N}_t{t}_p{p}'}",
                )


def test_mean_gap_leaves_out_p_zero_and_one() -> None:
    points = [
        {"p": 0.0, "gap": 5.0, "gap_err": 1.0},
        {"p": 0.2, "gap": -0.1, "gap_err": 0.03},
        {"p": 0.6, "gap": 0.1, "gap_err": 0.04},
        {"p": 1.0, "gap": 5.0, "gap_err": 1.0},
    ]
    m = _load().mean_gap(points)
    assert m["gap"] == pytest.approx(0.0)
    assert m["gap_err"] == pytest.approx(0.025)
    assert m["n_p"] == 2


def test_figures_from_small_sweeps(tmp_path: Path) -> None:
    """Two steps dt = t / 5 and the Haar control give the four figures."""
    _sweep(tmp_path / "tfim", (0.5, 10.0))
    _sweep(tmp_path / "haar", (10.0,), "unitary_step=haar_u1")
    module = _load()
    by_dt = module.dt_summaries(tmp_path / "tfim")
    assert list(by_dt) == [0.1, 2.0]
    assert all(row["rounds"] == 5 for rows in by_dt.values() for row in rows)
    out = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "tfim"),
            "--haar",
            str(tmp_path / "haar"),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    for name in ("dt_accuracy_vs_p", "dt_collapse", "dt_deviation", "dt_two_rules"):
        assert (out / f"{name}.png").is_file()
    data = json.loads((out / "dt_deviation.json").read_text())["extra"]
    assert data["reference"] == "Haar"
    # One gap per (dt, N, p): 2 x 2 x 2; p = 0 is chance for both steps.
    assert len(data["points"]) == 8
    for point in data["points"]:
        if point["p"] == 0.0:
            assert point["gap"] == pytest.approx(0.0, abs=0.2)
    assert set(data["mean_gap"]) == {
        f"N={n}, dt={dt}" for n in (3, 4) for dt in ("0.1", "2")
    }
    rules = json.loads((out / "dt_two_rules.json").read_text())["extra"]
    assert set(rules) == {
        f"N={n}, dt={dt}" for n in (3, 4) for dt in ("0.1", "2", "haar_u1")
    }


def test_haar_summary_refuses_a_tfim_sweep(tmp_path: Path) -> None:
    _run(
        "N=3",
        "samples.L=20",
        "projective_measurement.final_measurement=true",
        f"hydra.run.dir={tmp_path / 'run'}",
    )
    with pytest.raises(ValueError, match="haar_u1"):
        _load().haar_summary(tmp_path)
