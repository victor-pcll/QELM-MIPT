"""Tests of experiments/characteristic_p (issue #285): the p grid, the
crossing and minimum estimators, the summary of a sweep tree, the figures
and the launcher."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "characteristic_p"
SCRIPT = FOLDER / "characteristic_p.py"
BASH_SCRIPT = FOLDER / "characteristic_p.sh"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("characteristic_p", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cp = _load()


def test_p_grid_is_dense_in_pNT() -> None:
    ps = cp.p_grid(8, 8)
    assert ps[0] == 0.0 and ps[-1] == 1.0
    assert 0.5 in ps
    assert np.all(np.diff(ps) > 0)
    # Every point of PNT_GRID that fits below p = 1 is on the grid.
    for x in cp.PNT_GRID:
        if x <= 64:
            assert np.isclose(ps, x / 64, atol=1e-6).any()
    small = cp.p_grid(2, 1)
    assert small.max() == 1.0 and small.min() == 0.0
    with pytest.raises(ValueError, match="N and T"):
        cp.p_grid(0, 4)


def test_crossing_of_two_lines() -> None:
    p = np.linspace(0, 1, 11)
    qelm = 1.0 - p  # falls
    bayes = 0.5 + 0.0 * p  # flat
    assert cp.crossing(p, qelm, bayes) == pytest.approx(0.5)
    # Between grid points: linear interpolation.
    assert cp.crossing(p, 1.0 - p, 0.45 + 0.0 * p) == pytest.approx(0.55)


def test_crossing_undefined() -> None:
    p = np.linspace(0, 1, 6)
    # Never below Bayes.
    assert cp.crossing(p, np.ones(6), 0.5 * np.ones(6)) is None
    # Below Bayes from the first p > 0: nothing falls.
    assert cp.crossing(p, 0.4 * np.ones(6), 0.5 * np.ones(6)) is None


def test_crossing_ignores_p_zero() -> None:
    p = np.array([0.0, 0.1, 0.2, 0.3])
    qelm = np.array([0.5, 0.9, 0.7, 0.5])
    bayes = np.array([0.5, 0.6, 0.6, 0.6])
    # At p = 0 both are 0.5 (d = 0): the crossing is the later one.
    assert cp.crossing(p, qelm, bayes) == pytest.approx(0.2 + 0.1 * 0.1 / 0.2)


def test_minimum_vertex_of_a_parabola() -> None:
    p = np.linspace(0, 1, 21)
    acc = 0.5 + (p - 0.33) ** 2
    assert cp.minimum(p, acc) == pytest.approx(0.33, abs=1e-9)
    sem = np.full(p.size, 1e-3)
    assert cp.minimum(p, acc, sem) == pytest.approx(0.33, abs=1e-9)


def test_minimum_undefined_on_edges_and_plateaus() -> None:
    p = np.linspace(0, 1, 11)
    # Monotone fall: the minimum is the last point.
    assert cp.minimum(p, 1.0 - p) is None
    # Monotone rise from the first p > 0.
    assert cp.minimum(p, p) is None
    # A dip of 0.005 with standard errors of 0.01: a plateau, no minimum.
    acc = np.full(p.size, 0.5)
    acc[0] = 1.0
    acc[5] = 0.495
    assert cp.minimum(p, acc) is not None
    assert cp.minimum(p, acc, np.full(p.size, 0.01)) is None


def _metrics(N: int, T: int, ps: np.ndarray, rng: np.random.Generator) -> dict:
    """Synthetic metrics.json: QELM falls then rises, Bayes rises."""
    out: dict[str, float] = {"N": float(N), "T": float(T)}
    x = ps * N * T
    curves = {
        "bayes": 0.5 + 0.3 * (1 - np.exp(-x / 10)),
        "ridge_quantum": 0.5 + 0.5 * np.exp(-x / 3),
        "ridge_raw": 0.5 + 0.2 * (1 - np.exp(-x / 20)),
    }
    curves["ridge_M_quantum"] = np.maximum(curves["ridge_quantum"], curves["ridge_raw"])
    for name, acc in curves.items():
        noisy = acc + rng.normal(0, 0.002, size=acc.size)
        for v, a in zip(ps, noisy, strict=True):
            out[f"accuracy_{name}_p{v:g}"] = float(a)
    for v in ps:
        out[f"accuracy_bayes_expected_p{v:g}"] = 0.6
    return out


def _tree(root: Path, sizes: list[tuple[int, int]], seeds: int = 4) -> None:
    rng = np.random.default_rng(0)
    for N, T in sizes:
        ps = cp.p_grid(N, T)
        for s in range(seeds):
            folder = root / "ghz" / f"N{N}_T{T}" / f"seed{s}"
            folder.mkdir(parents=True)
            metrics = _metrics(N, T, ps, rng)
            (folder / "metrics.json").write_text(json.dumps(metrics))


def test_summary_of_a_synthetic_sweep(tmp_path: Path) -> None:
    _tree(tmp_path, [(4, 4), (8, 8)])
    tree = cp.load_tree(tmp_path)
    assert set(tree) == {"ghz"} and set(tree["ghz"]) == {(4, 4), (8, 8)}
    summary = cp.summarize(tree)
    for key, NT in (("N4_T4", 16), ("N8_T8", 64)):
        entry = summary["ghz"][key]
        assert entry["n_circuits"] == 4 and entry["NT"] == NT
        # The synthetic curves depend on p N T only: p* N T is the same.
        assert entry["p_cross"] is not None and entry["p_min"] is not None
        assert entry["p_cross_defined_fraction"] == 1.0
        assert entry["p_cross_err"] is not None and entry["p_cross_err"] >= 0
    pc = [summary["ghz"][k]["p_cross"] * n for k, n in (("N4_T4", 16), ("N8_T8", 64))]
    assert pc[0] == pytest.approx(pc[1], rel=0.25)
    # p_bayes_expected is not taken for a p value.
    assert len(summary["ghz"]["N8_T8"]["p"]) == cp.p_grid(8, 8).size


def test_load_tree_refuses_empty_and_mixed_grids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No <encoding>"):
        cp.load_tree(tmp_path)
    _tree(tmp_path, [(4, 4)], seeds=2)
    path = tmp_path / "ghz" / "N4_T4" / "seed1" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics = {k: v for k, v in metrics.items() if not k.endswith("_p1")}
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="different p grids"):
        cp.summarize(cp.load_tree(tmp_path))


def test_plot_writes_the_figures(tmp_path: Path) -> None:
    _tree(tmp_path / "sweep", [(4, 4), (8, 8)])
    paths = cp.main(["plot", str(tmp_path / "sweep"), "--out", str(tmp_path / "fig")])
    names = {p.name for p in paths}
    assert names == {
        "accuracy_vs_pNT_N4_T4.png",
        "accuracy_vs_pNT_N8_T8.png",
        "p_star_vs_NT.png",
    }
    assert all(p.stat().st_size > 0 for p in paths)
    summary = json.loads((tmp_path / "fig" / "summary.json").read_text())
    assert set(summary["ghz"]) == {"N4_T4", "N8_T8"}


def test_grid_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert cp.main(["grid", "4", "2"]) == []
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("[0,") and printed.endswith(",1]")
    assert json.loads(printed) == pytest.approx(cp.p_grid(4, 2).tolist())


def test_launcher_and_plot(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    env = {
        **os.environ,
        "OUT_DIR": str(sweep),
        "ENCODINGS": "ghz",
        "SIZES": "4:2",
        "SEEDS": "1 2",
        "PER_P": "20",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=REPO_ROOT)
    run = sweep / "ghz" / "N4_T2" / "seed1" / "metrics.json"
    assert run.exists()
    metrics = json.loads(run.read_text())
    assert metrics["N"] == 4 and metrics["T"] == 2
    assert "accuracy_ridge_M_quantum_p0.125" in metrics
    stamp = run.stat().st_mtime_ns
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "SKIP_DONE": "1"},
        cwd=REPO_ROOT,
    )
    assert run.stat().st_mtime_ns == stamp
    paths = cp.main(["plot", str(sweep), "--out", str(tmp_path / "fig")])
    assert [p.name for p in paths] == ["accuracy_vs_pNT_N4_T2.png"]
