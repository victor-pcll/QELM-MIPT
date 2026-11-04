"""Readouts of S_R from the final Clifford state (issue #292)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from QELM import load_config
from QELM.models.circuits.stabilizer import (
    draw_probe_history,
    reference_entropy_history,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "clifford_state_readout"
SCRIPT = FOLDER / "clifford_state_readout.py"
BASH_SCRIPT = FOLDER / "clifford_state_readout.sh"
PLOT_SCRIPT = FOLDER / "plot_clifford_state_readout.py"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/clifford_state_readout, not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


exp = _load(SCRIPT)
plot = _load(PLOT_SCRIPT)


def test_feature_paulis_count_and_order() -> None:
    N = 4
    strings = exp.feature_paulis(N)
    assert len(strings) == exp.len_features(N) == 3 * N + 9 * N * (N - 1) // 2
    assert exp.n_one_site(N) == 12
    # One-site strings first, site-major; R (qubit 0) is never touched.
    assert str(strings[0]) == "+_X___" and str(strings[2]) == "+_Z___"
    assert str(strings[3]) == "+__X__"
    assert str(strings[12]) == "+_XX__" and str(strings[13]) == "+_XY__"
    assert all(s[0] == 0 for s in strings)
    assert exp._n1_from_features(exp.len_features(6)) == 18
    with pytest.raises(ValueError, match="No N"):
        exp._n1_from_features(7)


def test_state_features_of_known_states() -> None:
    N = 4
    sim = stim.TableauSimulator()  # |0>^(N + 1)
    sim.do_circuit(stim.Circuit(f"I {N}"))
    f = exp.state_features(sim, N)
    ones = [3 * k + 2 for k in range(N)]  # <Z_k> = 1
    np.testing.assert_array_equal(f[ones], 1.0)
    # <Z_i Z_j> = 1, every other two-site string 0.
    two = f[3 * N :].reshape(-1, 3, 3)
    np.testing.assert_array_equal(two[:, 2, 2], 1.0)
    assert np.count_nonzero(f) == N + N * (N - 1) // 2
    assert exp.reference_entropy(sim) == 0.0
    # R in a Bell pair with site 1: every feature on site 1 vanishes.
    sim = stim.TableauSimulator()
    sim.h(0)
    sim.cnot(0, 2)
    sim.do_circuit(stim.Circuit(f"I {N}"))
    f = exp.state_features(sim, N)
    np.testing.assert_array_equal(f[3:6], 0.0)
    assert exp.reference_entropy(sim) == 1.0


def test_entropy_matches_the_probe_module() -> None:
    N = 6
    for p in (0.0, 0.15, 0.3, 1.0):
        history = draw_probe_history(N, p, 0, 2 * N, np.random.default_rng(3))
        sim = exp.replay(history, 7)
        ref = reference_entropy_history(history, site=N // 2)[-1]
        assert exp.reference_entropy(sim) == ref


def test_signs_are_outcomes_but_abs_and_entropy_are_not() -> None:
    # The stabilizer group up to signs, hence |<P>| and S_R, depends only on
    # the gates and on which sites are measured, not on the outcomes.
    N = 6
    history = draw_probe_history(N, 0.3, 0, 2 * N, np.random.default_rng(11))
    feats = [exp.state_features(exp.replay(history, s), N) for s in range(6)]
    for f in feats[1:]:
        np.testing.assert_array_equal(np.abs(f), np.abs(feats[0]))
    assert any((f != feats[0]).any() for f in feats[1:])
    S = {exp.reference_entropy(exp.replay(history, s)) for s in range(6)}
    assert len(S) == 1


@pytest.mark.parametrize("p, S", [(0.0, 1.0), (1.0, 0.0)])
def test_limits_of_the_trajectory(p: float, S: float) -> None:
    N = 4
    S_R, f, n = exp.simulate_trajectory(N, p, 0, 2 * N, np.random.default_rng(0))
    assert S_R == S
    assert f.shape == (exp.len_features(N),)
    assert set(np.unique(f)) <= {-1.0, 0.0, 1.0}
    assert n == (0 if p == 0 else N * 2 * N)


def test_simulate_shapes_and_seed() -> None:
    a = exp.simulate(4, [0.1, 0.5], 5, 0, 8, seed=1)
    b = exp.simulate(4, [0.1, 0.5], 5, 0, 8, seed=1)
    assert a["S_R"].shape == (2, 5)
    assert a["features"].shape == (2, 5, exp.len_features(4))
    assert a["n_measured"].shape == (2, 5)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_quadratic_features() -> None:
    X = np.array([[1.0, -1.0, 0.0, 1.0]])
    Q = exp.quadratic_features(X, 2)
    np.testing.assert_array_equal(Q, [[1, -1, 0, 1, 1, 1, 0, 1, -1]])


def test_fit_predict_readouts() -> None:
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 0.0, 1.0], size=(400, 6))
    y = (np.abs(X[:, 0]) > 0).astype(float)  # depends on |x_0| only
    n = rng.integers(0, 10, size=400)
    tr, te = slice(0, 300), slice(300, None)
    pred = exp.fit_predict("abs", X[tr], y[tr], X[te], 3, 0)
    assert exp.scores(y[te], pred)["R2"] > 0.99
    lin = exp.fit_predict("linear", X[tr], y[tr], X[te], 3, 0)
    assert exp.scores(y[te], lin)["R2"] < 0.1  # even in x_0: invisible
    quad = exp.fit_predict("quadratic", X[tr], y[tr], X[te], 3, 0)
    assert exp.scores(y[te], quad)["R2"] > 0.99
    mlp = exp.fit_predict("mlp", X[tr], y[tr], X[te], 3, 0)
    assert np.all((mlp >= 0) & (mlp <= 1))
    chance = exp.fit_predict("chance", X[tr], y[tr], X[te], 3, 0)
    np.testing.assert_allclose(chance, y[tr].mean())
    y_n = (n > 4).astype(float)
    pn = exp.fit_predict("n_meas", X[tr], y_n[tr], X[te], 3, 0, n[tr], n[te])
    assert exp.scores(y_n[te], pn)["R2"] > 0.5
    with pytest.raises(ValueError, match="n_meas needs"):
        exp.fit_predict("n_meas", X[tr], y[tr], X[te], 3, 0)
    with pytest.raises(ValueError, match="Unknown readout"):
        exp.fit_predict("knn", X[tr], y[tr], X[te], 3, 0)
    # One class only: the constant.
    const = exp.fit_predict("linear", X[tr], np.ones(300), X[te], 3, 0)
    np.testing.assert_array_equal(const, 1.0)


def test_scores_and_split() -> None:
    assert np.isnan(exp.scores(np.ones(4), np.zeros(4))["R2"])
    assert exp.scores(np.ones(4), np.zeros(4))["MAE"] == 1.0
    mask = exp.split(10, 0.7, np.random.default_rng(0))
    assert mask.sum() == 7


def _cfg(*extra: str) -> object:
    return load_config(
        [
            "N=4",
            "task.p_values=[0.0,0.2,1.0]",
            "task.samples_per_p=40",
            *extra,
        ],
        config_name="clifford_state_readout",
    )


def test_run_metrics() -> None:
    metrics, data = exp.run(_cfg())
    P = 3
    assert metrics["N"] == 4 and metrics["steps"] == 8
    assert metrics["n_features"] == exp.len_features(4)
    assert len(metrics["oracle"]) == P
    # p = 0 keeps the Bell pair, p = 1 purifies R.
    assert metrics["oracle_all"][0] == 1.0 and metrics["oracle_all"][2] == 0.0
    assert set(metrics["per_p"]) == set(exp.READOUTS)
    assert set(metrics["pooled"]) == set(exp.READOUTS) | {"p_mean"}
    for r in metrics["pooled"].values():
        assert len(r["R2"]) == len(r["mean_prediction"]) == P
    # p_mean knows p: exact at p = 0 and 1.
    pm = metrics["pooled"]["p_mean"]["mean_prediction"]
    assert pm[0] == 1.0 and pm[2] == 0.0
    assert data["features"].shape == (P, 40, exp.len_features(4))
    again, _ = exp.run(_cfg())
    assert json.dumps(again) == json.dumps(metrics)


def test_run_refuses_unknown_readout() -> None:
    with pytest.raises(ValueError, match="Unknown readouts"):
        exp.run(_cfg("task.readouts=[linear,knn]"))


def test_crossing() -> None:
    p = np.array([0.0, 0.1, 0.2])
    assert plot.crossing(p, np.array([1.0, 0.5, 0.0]), np.array([0.8, 0.5, 0.2])) == (
        pytest.approx(0.1)
    )
    assert plot.crossing(p, np.array([1.0, 0.6, 0.0]), np.array([0.8, 0.4, 0.2])) == (
        pytest.approx(0.15)
    )
    assert plot.crossing(p, np.ones(3), np.zeros(3)) is None


def test_launcher_and_plots(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "4 6",
        "SEEDS": "1 2",
        "PER_P": "30",
        "SCRAMBLE": "2N",
        "READOUTS": "[chance,n_meas,linear,abs]",
        "MAX_JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True)
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == ["N4_seed1", "N4_seed2", "N6_seed1", "N6_seed2"]
    m = json.loads((tmp_path / "N6_seed1" / "metrics.json").read_text())
    assert m["scramble_steps"] == 12
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["sizes"]) == {"4", "6"}
    assert "4-6" in summary["crossings"]["oracle"]
    for name in ("order_parameter", "r2_per_p", "r2_pooled"):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.json").is_file()
    # SKIP_DONE: a second launch reruns nothing.
    out = subprocess.run(
        ["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True, text=True
    )
    assert out.stdout.count("skip") == 4


def test_launcher_refuses_bad_max_jobs(tmp_path: Path) -> None:
    env = {**os.environ, "OUT_DIR": str(tmp_path), "MAX_JOBS": "0"}
    result = subprocess.run(["bash", str(BASH_SCRIPT)], env=env, capture_output=True)
    assert result.returncode != 0
