"""Tests of the tomography and purity task (experiments/bloch_purity, #277)."""

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

from QELM import load_config
from QELM.physics.operators import PAULIS

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "bloch_purity"
SCRIPT = FOLDER / "bloch_purity.py"
BASH_SCRIPT = FOLDER / "bloch_purity.sh"
PLOT_SCRIPT = FOLDER / "plot_bloch_purity.py"
N = 4
P_VALUES = "[0.0,0.3,1.0]"
PER_P = 60


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/bloch_purity, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BP = _load(SCRIPT)


def _cfg(*extra: str):  # type: ignore[no-untyped-def]
    return load_config(
        [
            f"N={N}",
            f"bloch_task.p_values={P_VALUES}",
            f"bloch_task.samples_per_p={PER_P}",
            "bloch_task.depth_rule=N",
            *extra,
        ],
        config_name="bloch_purity",
    )


def _random_state(n: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return psi / np.linalg.norm(psi)


# ---- Targets ----------------------------------------------------------------


def test_reference_rho_against_partial_trace() -> None:
    rng = np.random.default_rng(0)
    psi = _random_state(4, rng)
    rho_full = np.outer(psi, psi.conj()).reshape(2, 8, 2, 8)
    rho_R = np.einsum("aibi->ab", rho_full)
    np.testing.assert_allclose(BP.reference_rho(psi), rho_R, atol=1e-12)


def test_bloch_vector_against_paulis() -> None:
    rng = np.random.default_rng(1)
    psi = _random_state(4, rng)
    rho = BP.reference_rho(psi)
    expected = [np.trace(rho @ PAULIS[c]).real for c in ("X", "Y", "Z")]
    np.testing.assert_allclose(BP.bloch_vector(rho), expected, atol=1e-12)
    # Purity (1 + |r|^2) / 2 = Tr(rho^2).
    np.testing.assert_allclose(
        BP.purity_from_bloch(BP.bloch_vector(rho)),
        np.trace(rho @ rho).real,
        atol=1e-12,
    )


def test_bloch_of_known_states() -> None:
    plus_i = np.array([1.0, 1j]) / np.sqrt(2.0)
    # R in |+i>, one system qubit in |0>: r = (0, 1, 0), purity 1.
    psi = np.kron(plus_i, [1.0, 0.0])
    np.testing.assert_allclose(
        BP.bloch_vector(BP.reference_rho(psi)), [0, 1, 0], atol=1e-12
    )
    bell = np.array([1.0, 0, 0, 1.0]) / np.sqrt(2.0)
    r = BP.bloch_vector(BP.reference_rho(bell))
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    assert BP.purity_from_bloch(r) == pytest.approx(0.5)


def test_purity_clip() -> None:
    r = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(BP.purity_from_bloch(r, clip=True), [1.0, 0.5])
    np.testing.assert_allclose(BP.purity_from_bloch(r), [2.5, 0.5])


def test_entropy_bits() -> None:
    np.testing.assert_allclose(
        BP._entropy_bits(np.array([[0, 0, 0], [0, 0, 1.0], [0, 0.6, 0]])),
        [1.0, 0.0, -(0.8 * np.log2(0.8) + 0.2 * np.log2(0.2))],
        atol=1e-12,
    )


# ---- Scores -----------------------------------------------------------------


def test_r2_vector() -> None:
    y = np.array([[1.0, 0.0], [-1.0, 2.0], [0.0, 1.0]])
    assert BP.r2_vector(y, y) == pytest.approx(1.0)
    assert BP.r2_vector(y, np.broadcast_to(y.mean(axis=0), y.shape)) == pytest.approx(
        0.0
    )
    assert np.isnan(BP.r2_vector(np.zeros((5, 3)), np.zeros((5, 3))))
    # One column: the usual R2.
    from sklearn.metrics import r2_score

    pred = y[:, 0] + 0.3
    assert BP.r2_vector(y[:, 0], pred) == pytest.approx(r2_score(y[:, 0], pred))


def test_linear_target_gives_r2_one() -> None:
    rng = np.random.default_rng(3)
    X = rng.choice([-1.0, 0.0, 1.0], size=(400, 12))
    W = rng.normal(size=(12, 3))
    Y = X @ W
    pred = BP.fit_ridge(X[:300], Y[:300], X[300:])
    assert pred.shape == (100, 3)
    assert BP.r2_vector(Y[300:], pred) > 0.999
    # Constant features: the mean target.
    flat = BP.fit_ridge(np.zeros((10, 3)), Y[:10], np.zeros((4, 3)))
    np.testing.assert_allclose(flat, np.broadcast_to(Y[:10].mean(axis=0), (4, 3)))


def test_score_level_bayes_and_chance() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 5))
    bloch = np.tanh(X[:, :3] * 0.3)
    purity = BP.purity_from_bloch(bloch)
    tr, te = np.arange(140), np.arange(140, 200)
    scores = BP.score_level({"M": X}, bloch, purity, tr, te)
    assert scores["r2_bloch_bayes"] == pytest.approx(1.0)
    assert scores["r2_purity_bayes"] == pytest.approx(1.0)
    assert abs(scores["r2_bloch_chance"]) < 0.1
    assert scores["r2_bloch_M"] > 0.9
    assert scores["mse_bloch_bayes"] == 0.0


# ---- Experiment -------------------------------------------------------------


def test_trajectory_at_p0_keeps_the_bell_pair() -> None:
    cfg = _cfg()
    from omegaconf import OmegaConf

    from QELM.models.circuits.haar import HaarBrickworkCircuit

    block = OmegaConf.to_container(cfg.haar_circuit, resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    params["depth"] = 3
    circuit = HaarBrickworkCircuit(params, np.random.default_rng(0))
    psi0 = BP.NL.initial_state(cfg, "bell")
    M, r, psi = BP.trajectory(psi0, circuit, 0.0, np.random.default_rng(1))
    np.testing.assert_array_equal(M, 0.0)
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    assert np.linalg.norm(psi) == pytest.approx(1.0)


def test_run_shapes_and_physics() -> None:
    out = BP.run(_cfg())
    L = 3 * PER_P
    assert out["M"].shape == (L, N, N)
    assert out["bloch"].shape == (L, 3)
    assert np.all(np.linalg.norm(out["bloch"], axis=1) <= 1.0 + 1e-9)
    m = out["metrics"]
    # p = 0: R maximally mixed, R2 undefined; p = 1: R purified by round 1.
    assert m["purity_mean_p0"] == pytest.approx(0.5)
    assert np.isnan(m["r2_bloch_M_p0"])
    assert m["purity_mean_p1"] == pytest.approx(1.0)
    assert m["S_R_mean_p1"] == pytest.approx(0.0, abs=1e-6)
    assert m["r2_bloch_bayes_p0.3"] == pytest.approx(1.0)
    assert m["n_features_M"] == N * N
    assert m["n_features_state"] == 3 * N + 3 * N * (N - 1) // 2
    for key in ("r2_x_state_p0.3", "r2_purity_M_state_p0.3", "mae_purity_M_p0.3"):
        assert key in m


def test_run_is_reproducible() -> None:
    a = BP.run(_cfg("seed=7"))
    b = BP.run(_cfg("seed=7"))
    np.testing.assert_array_equal(a["M"], b["M"])
    np.testing.assert_array_equal(a["bloch"], b["bloch"])
    assert json.dumps(a["metrics"]) == json.dumps(b["metrics"])
    c = BP.run(_cfg("seed=8"))
    assert not np.array_equal(a["bloch"], c["bloch"])


def test_run_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="Nr=1"):
        BP.run(_cfg("Nr=0"))
    with pytest.raises(ValueError, match="distinct"):
        BP.run(_cfg("bloch_task.p_values=[0.1,0.1]"))
    with pytest.raises(ValueError, match="m_features"):
        BP.run(_cfg("bloch_task.m_features=xor_full"))
    with pytest.raises(ValueError, match=">= 10"):
        BP.run(_cfg("bloch_task.samples_per_p=5"))


def test_sweep_and_figures(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path / "sweep"),
        "N_VALUES": "4",
        "SEEDS": "1 2",
        "PER_P": "40",
        "P_VALUES": "0 0.3 1",
        "DEPTH": "N",
        "SAVE_ARRAYS": "false",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=REPO_ROOT)
    sweep = tmp_path / "sweep"
    assert (sweep / "N4_seed1" / "metrics.json").exists()
    assert not (sweep / "N4_seed1" / "bloch_purity.npz").exists()
    stamp = (sweep / "N4_seed1" / "metrics.json").stat().st_mtime_ns
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "SKIP_DONE": "1"},
        cwd=REPO_ROOT,
    )
    assert (sweep / "N4_seed1" / "metrics.json").stat().st_mtime_ns == stamp
    fig = tmp_path / "fig"
    subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), str(sweep), "--out", str(fig)], check=True
    )
    for name in ("bloch_vector_N4.png", "purity_N4.png", "summary.json"):
        assert (fig / name).exists()
    summary = json.loads((fig / "summary.json").read_text())["N4"]
    assert summary["n_circuits"] == 2
    assert summary["p"] == [0.0, 0.3, 1.0]
    # NaN (p = 0) is written as null.
    assert summary["r2_bloch_M"][0] is None
    assert summary["r2_bloch_bayes"][1] == pytest.approx(1.0)


# ---- Single-shot labels -----------------------------------------------------


@pytest.mark.parametrize(
    ("state", "axis"),
    [
        (np.array([1.0, 1.0]) / np.sqrt(2.0), 0),
        (np.array([1.0, 1j]) / np.sqrt(2.0), 1),
        (np.array([1.0, 0.0]), 2),
    ],
)
def test_measure_reference_eigenstates(state: np.ndarray, axis: int) -> None:
    # R in the +1 eigenstate of P_axis, one system qubit in |0>: s = +1 always.
    rng = np.random.default_rng(5)
    psi = np.kron(state, [1.0, 0.0]).astype(complex)
    for _ in range(20):
        out, s = BP.measure_reference(psi, axis, rng)
        assert s == 1.0
        assert np.linalg.norm(out) == pytest.approx(1.0)
        # The chain is untouched.
        np.testing.assert_allclose(
            np.abs(out.reshape(2, 2)).sum(axis=0), [1.0, 0.0], atol=1e-12
        )
    _, s = BP.measure_reference(np.kron(state.conj(), [1.0, 0.0]), axis, rng)
    if axis == 1:
        # |-i>: the -1 eigenstate of Y.
        assert s == -1.0


def test_measure_reference_born_rule() -> None:
    rng = np.random.default_rng(6)
    psi = _random_state(3, rng)
    r = BP.bloch_vector(BP.reference_rho(psi))
    for axis in range(3):
        shots = [BP.measure_reference(psi, axis, rng)[1] for _ in range(4000)]
        # E[s] = r_a, standard error 1 / sqrt(4000) ~ 0.016.
        assert np.mean(shots) == pytest.approx(r[axis], abs=0.06)


def test_score_shots_bayes_ceiling() -> None:
    rng = np.random.default_rng(7)
    L = 3000
    bloch = rng.normal(size=(L, 3))
    bloch /= np.linalg.norm(bloch, axis=1, keepdims=True)
    axis = rng.integers(3, size=L)
    r_a = bloch[np.arange(L), axis]
    shot = np.where(rng.random(L) < (1.0 + r_a) / 2.0, 1.0, -1.0)
    X = bloch.copy()
    tr, te = np.arange(2000), np.arange(2000, L)
    scores = BP.score_shots({"M": X}, bloch, axis, shot, tr, te)
    # Pure states along random directions: E[r_a^2] = 1/3.
    assert scores["bayes_shots_ceiling"] == pytest.approx(1.0 / 3.0, abs=0.03)
    assert scores["r2_shots_bayes"] == pytest.approx(1.0 / 3.0, abs=0.06)
    # A ridge on features that contain r reaches the Bayes ceiling.
    assert scores["r2_shots_M"] == pytest.approx(scores["r2_shots_bayes"], abs=0.02)
    assert abs(scores["r2_shots_chance"]) < 0.02


def test_labels_modes() -> None:
    both = BP.run(_cfg("seed=3", "bloch_task.labels=both"))
    exact = BP.run(_cfg("seed=3", "bloch_task.labels=exact"))
    shots = BP.run(_cfg("seed=3", "bloch_task.labels=shots"))
    # The shots have their own generator: the exact scores do not move.
    for key, value in exact["metrics"].items():
        np.testing.assert_equal(both["metrics"][key], value)
    np.testing.assert_array_equal(both["M"], shots["M"])
    assert "r2_shots_M_p0.3" in both["metrics"]
    assert "r2_shots_M_p0.3" not in exact["metrics"]
    assert "r2_bloch_M_p0.3" not in shots["metrics"]
    assert np.all(exact["axis"] == -1)
    assert set(np.unique(shots["axis"])) == {0, 1, 2}
    assert set(np.unique(shots["shot"])) <= {-1.0, 1.0}
    with pytest.raises(ValueError, match="labels"):
        BP.run(_cfg("bloch_task.labels=bogus"))


def test_m_features_raw_abs() -> None:
    out = BP.run(_cfg("bloch_task.m_features=raw_abs", "bloch_task.labels=exact"))
    assert out["metrics"]["n_features_M"] == 2 * N * N


# ---- Nonlinear readouts (issue #282) ----------------------------------------


def test_fit_mlp_reads_a_nonlinear_target() -> None:
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 0.0, 1.0], size=(3000, 4))
    y = X[:, 0] * X[:, 1]  # invisible to a linear readout
    tr, te = slice(0, 2400), slice(2400, None)
    ridge = BP.fit_ridge(X[tr], y[tr], X[te])
    mlp = BP.fit_mlp(X[tr], y[tr], X[te])
    assert BP.r2_vector(y[te], ridge) < 0.05
    assert BP.r2_vector(y[te], mlp) > 0.9
    # Constant features: the mean target, as the ridge.
    flat = BP.fit_mlp(np.zeros((10, 3)), np.arange(10.0), np.zeros((2, 3)))
    np.testing.assert_allclose(flat, 4.5)


def test_fit_readout_dispatch() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(
        BP.fit_readout("M_abs", X, y, X[:5]), BP.fit_ridge(X, y, X[:5])
    )
    np.testing.assert_allclose(
        BP.fit_readout("M_mlp", X, y, X[:5]), BP.fit_mlp(X, y, X[:5])
    )


def test_nonlinear_readouts_option() -> None:
    base = BP.run(_cfg())["metrics"]
    out = BP.run(_cfg("bloch_task.nonlinear_readouts=true"))["metrics"]
    for name in BP.NONLINEAR_READOUTS:
        assert f"r2_bloch_{name}_p0.3" in out
        assert f"r2_shots_{name}_p0.3" in out
        assert f"mae_purity_{name}_p0.3" in out
        assert f"r2_bloch_{name}_p0.3" not in base
    # The other scores do not change.
    for key, value in base.items():
        if key.startswith(("r2_", "mae_", "mse_", "purity_")):
            np.testing.assert_allclose(out[key], value, equal_nan=True)
    # With raw M, the ridge on [M, |M|] is the raw_abs readout of #280.
    raw_abs = BP.run(_cfg("bloch_task.m_features=raw_abs"))["metrics"]
    np.testing.assert_allclose(out["r2_bloch_M_abs_p0.3"], raw_abs["r2_bloch_M_p0.3"])


# ---- Finite-size figure (issue #282) ----------------------------------------

FSS_SCRIPT = FOLDER / "plot_bloch_fss.py"
FSS = _load(FSS_SCRIPT)


def test_extremum_of_a_parabola() -> None:
    ps = np.linspace(0.0, 1.0, 21)
    x0, y0 = FSS.extremum(ps, (ps - 0.23) ** 2 + 0.1, "min")
    assert x0 == pytest.approx(0.23)
    assert y0 == pytest.approx(0.1)
    x1, y1 = FSS.extremum(ps, -((ps - 0.31) ** 2), "max")
    assert x1 == pytest.approx(0.31)
    assert y1 == pytest.approx(0.0, abs=1e-12)
    # Extremum at the edge of the range: the grid point itself.
    x2, _ = FSS.extremum(ps, ps, "min")
    assert x2 == pytest.approx(FSS.EXTREMUM_RANGE[0])


def test_crossings_of_two_lines() -> None:
    ps = np.linspace(0.0, 0.5, 11)
    assert FSS.crossings(ps, ps, 0.2 + 0 * ps) == [pytest.approx(0.2)]
    assert FSS.crossings(ps, ps, ps + 1.0) == []
    # A crossing outside CROSSING_RANGE is left out.
    assert FSS.crossings(ps, ps, 0.02 + 0 * ps) == []


def test_fss_sweep_and_figures(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    env = {
        **os.environ,
        "OUT_DIR": str(sweep),
        "N_VALUES": "3 4",
        "SEEDS": "1 2",
        "PER_P": "60",
        "P_VALUES": "0.1 0.3 1",
        "DEPTH": "N",
        "NONLINEAR": "true",
        "SAVE_ARRAYS": "false",
        "MAX_JOBS": "2",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=REPO_ROOT)
    metrics = json.loads((sweep / "N4_seed1" / "metrics.json").read_text())
    assert "r2_bloch_M_mlp_p0.3" in metrics
    fig = tmp_path / "fig"
    subprocess.run(
        [
            sys.executable,
            str(FSS_SCRIPT),
            str(sweep),
            "--out",
            str(fig),
            "--n-boot",
            "20",
        ],
        check=True,
    )
    for name in (
        "bloch_fss.png",
        "bloch_nonlinear.png",
        "purity_variance.png",
        "fss_summary.json",
    ):
        assert (fig / name).exists()
    summary = json.loads((fig / "fss_summary.json").read_text())
    assert summary["sizes"] == [3, 4]
    assert set(summary["extrema"]) == set(FSS.EXTREMA)
    assert "N3_N4" in summary["crossings"]["r2_bloch"]
    assert set(summary["extrema"]["var_oracle"]) == {"N3", "N4"}
    # The per-N figures also draw the nonlinear curves.
    subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), str(sweep), "--out", str(fig)], check=True
    )
    per_n = json.loads((fig / "summary.json").read_text())["N4"]
    assert any(v is not None for v in per_n["r2_bloch_M_mlp"])


def test_readout_learning_curve(tmp_path: Path) -> None:
    script = FOLDER / "readout_learning_curve.py"
    out = BP.run(_cfg("bloch_task.labels=exact", "bloch_task.samples_per_p=120"))
    npz = tmp_path / "bloch_purity.npz"
    keep = out["p"] > 0
    np.savez(npz, M=out["M"][keep], bloch=out["bloch"][keep], p=out["p"][keep])
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(npz),
            "--sizes",
            "40",
            "80",
            "500",
            "--n-test",
            "30",
        ],
        check=True,
    )
    curve = json.loads((tmp_path / f"learning_curve_N{N}.json").read_text())
    assert curve["N"] == N
    assert set(curve["p"]) == {"0.3", "1"}
    entry = curve["p"]["0.3"]
    assert set(entry) == {"ridge_M", "ridge_M_abs", "mlp"}
    assert entry["ridge_M"][2] is None  # 500 > 90 training trajectories
    assert all(v is not None for v in entry["mlp"][:2])
    assert (tmp_path / f"learning_curve_N{N}.png").exists()


def test_variance_probe_and_abs_option() -> None:
    out = BP.run(_cfg("bloch_task.nonlinear_readouts=abs"))
    m = out["metrics"]
    assert "r2_bloch_M_abs_p0.3" in m
    assert "r2_bloch_M_mlp_p0.3" not in m
    # Chance predicts a constant: no variance. The oracle variance is that
    # of the exact purity on the test trajectories, at most its total one.
    assert m["var_purity_chance_p0.3"] == pytest.approx(0.0, abs=1e-15)
    assert m["var_purity_bayes_p0.3"] == pytest.approx(
        m["var_purity_from_bloch_bayes_p0.3"]
    )
    assert 0.0 < m["var_purity_bayes_p0.3"] < 0.125
    with pytest.raises(ValueError, match="nonlinear_readouts"):
        BP.run(_cfg("bloch_task.nonlinear_readouts=mlp"))


def test_variance_fraction_curve() -> None:
    runs = [
        {
            "purity_mean_p0.1": 0.6,
            "var_purity_bayes_p0.1": 0.02,
            "var_purity_M_abs_p0.1": 0.005,
            "var_purity_from_bloch_M_p0.1": 0.01,
        },
    ]
    frac = FSS.curves(runs, "var_fraction", "M_abs")
    np.testing.assert_allclose(frac, [[0.25]])
    np.testing.assert_allclose(
        FSS.curves(runs, "var_fraction_from_bloch", "M"), [[0.5]]
    )
