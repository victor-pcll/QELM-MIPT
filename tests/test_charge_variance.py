"""Per-trajectory charge variance read from the final U(1) state (issue #299)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config
from QELM.evaluation.charge_variance import (
    charge_moments,
    exact_features,
    pair_indices,
    sample_trajectories,
    sharpening_ratio,
    shot_estimates,
    superposition_state,
    variance_from_features,
)
from QELM.evaluation.learnability import UNMEASURED, charge_state, sample_records

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "charge_variance"
SCRIPT = FOLDER / "charge_variance.py"
BASH_SCRIPT = FOLDER / "charge_variance.sh"
PLOT_SCRIPT = FOLDER / "plot_charge_variance.py"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/charge_variance, not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


exp = _load(SCRIPT)
plot = _load(PLOT_SCRIPT)


# --- physics module -------------------------------------------------------


def test_superposition_state() -> None:
    L = 4
    psi = superposition_state(L, (2, 1))
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    np.testing.assert_allclose(
        psi, (charge_state(L, 2) + charge_state(L, 1)) / math.sqrt(2)
    )
    mean, var = charge_moments(np.abs(psi[None]) ** 2)
    assert mean[0] == pytest.approx(1.5) and var[0] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="distinct"):
        superposition_state(L, (1, 1))


def test_exact_features_of_known_states() -> None:
    L = 3
    # |001>: Z = (+1, +1, -1); site 0 is the most significant bit.
    probs = np.zeros((1, 2**L))
    probs[0, 0b001] = 1.0
    z, zz = exact_features(probs)
    np.testing.assert_array_equal(z[0], [1, 1, -1])
    i, j = pair_indices(L)
    np.testing.assert_array_equal(zz[0], z[0, i] * z[0, j])
    assert charge_moments(probs)[1][0] == 0.0


def test_variance_identity_on_random_states() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((20, 2**5))
    probs /= probs.sum(axis=1, keepdims=True)
    _, var = charge_moments(probs)
    z, zz = exact_features(probs)
    np.testing.assert_allclose(variance_from_features(z, zz), var, atol=1e-12)


def test_trajectories_p0_and_p1() -> None:
    rng = np.random.default_rng(1)
    tr = sample_trajectories(6, 0.0, 30, rng=rng)
    _, var = charge_moments(tr.probs)
    # No measurement: the weights of the two sectors stay 1/2.
    np.testing.assert_allclose(var, 0.25, atol=1e-12)
    assert np.all(tr.n_measured == 0)
    assert np.all(tr.outcomes == UNMEASURED)
    tr = sample_trajectories(6, 1.0, 30, rng=rng)
    _, var = charge_moments(tr.probs)
    # Every site measured: the charge is sharp.
    np.testing.assert_allclose(var, 0.0, atol=1e-12)
    assert np.all(tr.n_measured == 6 * 2 * 6)


def test_trajectories_conserve_charge_and_normalize() -> None:
    tr = sample_trajectories(6, 0.2, 50, rng=np.random.default_rng(2))
    assert tr.probs.shape == (50, 2**6)
    np.testing.assert_allclose(tr.probs.sum(axis=1), 1.0)
    assert tr.leakage.max() < 1e-12
    _, var = charge_moments(tr.probs)
    assert np.all((var >= 0) & (var <= 0.25 + 1e-12))
    assert tr.charges == (3, 2)


def test_trajectories_reproducible() -> None:
    a = sample_trajectories(4, 0.3, 10, rng=np.random.default_rng(5))
    b = sample_trajectories(4, 0.3, 10, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a.probs, b.probs)
    np.testing.assert_array_equal(a.outcomes, b.outcomes)


def test_records_match_learnability_in_distribution() -> None:
    """The measured fraction matches sample_records (same circuit, mixture)."""
    n = 2000
    tr = sample_trajectories(4, 0.3, n, rng=np.random.default_rng(3))
    rec = sample_records(4, 0.3, n, rng=np.random.default_rng(4))
    ones_tr = np.mean(tr.outcomes[tr.outcomes != UNMEASURED])
    ones_rec = np.mean(rec.outcomes[rec.outcomes != UNMEASURED])
    # Average filling of the two charges, (2 + 1) / 2 / 4 = 0.375.
    assert ones_tr == pytest.approx(0.375, abs=0.01)
    assert ones_rec == pytest.approx(0.375, abs=0.01)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"L": 1}, "L >= 2"),
        ({"p": 1.5}, "p must be"),
        ({"n": 0}, "n must be"),
        ({"charges": (2, 2)}, "two distinct"),
    ],
)
def test_trajectories_refuse_bad_arguments(kwargs: dict, match: str) -> None:
    args = {"L": 4, "p": 0.1, "n": 2, **kwargs}
    L, p, n = args.pop("L"), args.pop("p"), args.pop("n")
    with pytest.raises(ValueError, match=match):
        sample_trajectories(L, p, n, **args)


def test_shot_estimates_converge() -> None:
    tr = sample_trajectories(4, 0.15, 5, rng=np.random.default_rng(6))
    _, var = charge_moments(tr.probs)
    z, zz = exact_features(tr.probs)
    est = shot_estimates(tr.probs, 200_000, np.random.default_rng(7))
    np.testing.assert_allclose(est.z, z, atol=0.01)
    np.testing.assert_allclose(est.zz, zz, atol=0.01)
    np.testing.assert_allclose(est.var_direct, var, atol=0.01)
    with pytest.raises(ValueError, match="n_shots"):
        shot_estimates(tr.probs, 1, np.random.default_rng(0))


def test_sharpening_ratio() -> None:
    assert sharpening_ratio(np.full(10, 0.25)) == pytest.approx(2 / 3)
    # Half sharp, half fuzzy: <V^2> / <V>^2 = 2.
    assert sharpening_ratio(np.array([0.0, 0.25])) == pytest.approx(1 / 3)
    assert math.isnan(sharpening_ratio(np.zeros(4)))


# --- experiment script ----------------------------------------------------


def _cfg(*extra: str) -> object:
    return load_config(
        [
            "N=4",
            "task.p_values=[0.0,0.15,1.0]",
            "task.samples_per_p=80",
            "task.batch=50",
            "task.n_shots=[50]",
            "task.n_boot=20",
            *extra,
        ],
        config_name="charge_variance",
    )


def test_run_metrics() -> None:
    metrics, data = exp.run(_cfg())
    P = 3
    assert metrics["N"] == 4 and metrics["depth"] == 4
    assert metrics["feature_sets"] == ["exact", "shots50"]
    assert metrics["max_leakage"] < 1e-12
    var4 = metrics["oracle_all"]["var4"]
    assert var4[0] == pytest.approx(1.0) and var4[2] == pytest.approx(0.0)
    exact = metrics["results"]["exact"]
    readouts = set(exp.READOUTS) - {"direct"}
    assert set(exact["per_p"]) == readouts
    assert set(exact["pooled"]) == readouts | {"p_mean"}
    assert "direct" in metrics["results"]["shots50"]["per_p"]
    # R2 is undefined where Var(Q) is constant (p = 0 and p = 1).
    r2 = exact["per_p"]["quadratic_z"]["R2"]
    assert math.isnan(r2[0]) and math.isnan(r2[2])
    # Two sectors: Var(Q) is a quadratic of sum <Z_i>, so the quadratic
    # readout and the plug-in identity are exact on exact features.
    assert r2[1] > 0.99
    assert exact["per_p"]["plugin"]["R2"][1] == pytest.approx(1.0)
    assert exact["pooled"]["quadratic_z"]["R2_all"] > 0.99
    assert data["z_exact"].shape == (P, 80, 4)
    assert data["zz_exact"].shape == (P, 80, 6)
    assert data["direct_shots50"].shape == (P, 80)
    again, _ = exp.run(_cfg())
    assert json.dumps(again) == json.dumps(metrics)


def test_run_refuses_unknown_readout() -> None:
    with pytest.raises(ValueError, match="Unknown readouts"):
        exp.run(_cfg("task.readouts=[linear_z,knn]"))


def test_fit_predict_errors() -> None:
    with pytest.raises(ValueError, match="Unknown readout"):
        exp.fit_predict("knn", {}, {})
    with pytest.raises(ValueError, match="not a fitted"):
        exp.design("plugin", np.zeros((2, 2)), np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(ValueError, match="direct needs shots"):
        exp.predict_unfitted("direct", np.zeros((2, 2)), np.zeros((2, 1)), 0, None)


def test_quadratic_features_shape() -> None:
    z = np.arange(8.0).reshape(2, 4)
    q = exp.quadratic_features(z)
    assert q.shape == (2, 4 + 4 + 6)
    assert q[1, 8] == z[1, 0] * z[1, 1]


def test_scores_and_split() -> None:
    assert math.isnan(exp.scores(np.full(4, 0.25), np.zeros(4))["R2"])
    assert exp.scores(np.ones(4), np.zeros(4))["MAE"] == 1.0
    assert exp.split(10, 0.7, np.random.default_rng(0)).sum() == 7


# --- plots and launcher ---------------------------------------------------


def test_crossing() -> None:
    p = np.array([0.0, 0.1, 0.2, 0.3])
    # The tie at p = 0 is skipped.
    a = np.array([1.0, 0.6, 0.2, 0.0])
    b = np.array([1.0, 0.5, 0.3, 0.1])
    assert plot.crossing(p, a, b) == pytest.approx(0.15)
    assert plot.crossing(p, np.ones(4), np.zeros(4)) is None


def test_launcher_and_plots(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "4 6",
        "SEEDS": "1 2",
        "PER_P": "40",
        "N_SHOTS": "[30]",
        "MAX_JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True)
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == ["N4_seed1", "N4_seed2", "N6_seed1", "N6_seed2"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["sizes"]) == {"4", "6"}
    assert summary["feature_sets"] == ["exact", "shots30"]
    assert "4-6" in summary["crossings"]["oracle_var4"]
    assert "4-6" in summary["crossings"]["oracle_test_ratio"]
    assert "shots30_direct" in summary["crossings"]
    for name in ("order_parameter", "r2_per_p", "r2_pooled", "shots"):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.json").is_file()
    out = subprocess.run(
        ["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True, text=True
    )
    assert out.stdout.count("skip") == 4


def test_launcher_refuses_bad_max_jobs(tmp_path: Path) -> None:
    env = {**os.environ, "OUT_DIR": str(tmp_path), "MAX_JOBS": "0"}
    result = subprocess.run(["bash", str(BASH_SCRIPT)], env=env, capture_output=True)
    assert result.returncode != 0


def test_averaged_features_do_not_depend_on_p() -> None:
    """Averaged over gates and outcomes, <Z_i> and <Z_i Z_j> ignore p.

    The Haar U(1) average already dephases the state, so measurements leave
    the averaged diagonal unchanged: a readout linear in the state has the
    same mean prediction at every p.
    """
    means = []
    for p in (0.0, 0.5):
        tr = sample_trajectories(4, p, 4000, rng=np.random.default_rng(8))
        z, zz = exact_features(tr.probs)
        means.append(np.concatenate([z.mean(axis=0), zz.mean(axis=0)]))
    np.testing.assert_allclose(means[0], means[1], atol=0.04)
