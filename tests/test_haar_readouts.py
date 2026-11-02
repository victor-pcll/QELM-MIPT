"""Charge of a random Haar circuit read by several readouts (haar_readouts)."""

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
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.physics.operators import PAULIS, get_site_operator

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "haar_readouts"
SCRIPT = FOLDER / "haar_readouts.py"
BASH_SCRIPT = FOLDER / "haar_readouts.sh"
PLOT_SCRIPT = FOLDER / "plot_haar_readouts.py"
N = 4
P_VALUES = "[0.0,0.5,1.0]"
PER_P = 30
READOUTS = ("ridge", "ridge_pairs", "knn", "mlp", "gradient_boosting")


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/haar_readouts, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*extra: str) -> dict:
    cfg = load_config(
        [
            f"N={N}",
            f"readouts_task.p_values={P_VALUES}",
            f"readouts_task.samples_per_p={PER_P}",
            *extra,
        ],
        config_name="haar_readouts",
    )
    return _load(SCRIPT).run(cfg)


def _circuit(depth: int, seed: int) -> HaarBrickworkCircuit:
    return HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": depth}, np.random.default_rng(seed)
    )


def test_trajectory_record_and_purification() -> None:
    exp = _load(SCRIPT)
    psi0 = AncillaChargeSectorEncoding({"N": N}).state()
    rng = np.random.default_rng(0)
    # p = 0: nothing is measured, the gates never touch R, which stays
    # maximally entangled with the charge of the system.
    M, S_R, Y, psi = exp.trajectory(psi0, _circuit(3, 1), 0.0, rng)
    assert M.shape == (3, N) and not M.any()
    assert S_R == pytest.approx(1.0)
    assert Y in (1.0, -1.0)
    # The final state is the one after R is measured: R is in |0> for +1.
    weight = np.abs(psi.reshape(2, -1)) ** 2
    assert weight[0 if Y > 0 else 1].sum() == pytest.approx(1.0)
    # p = 1: every system qubit is measured at every round, the last round
    # fixes the state of the system, so R is purified.
    M, S_R, Y, _ = exp.trajectory(psi0, _circuit(3, 1), 1.0, rng)
    assert set(np.unique(M)) <= {-1.0, 1.0} and np.all(M != 0)
    assert S_R == pytest.approx(0.0, abs=1e-9)


def test_observables_match_dense_operators() -> None:
    exp = _load(SCRIPT)
    ops = exp.observables(N)
    assert [len(ops[name]) for name in exp.OBSERVABLE_SETS] == [N] * 3 + [
        N * (N - 1) // 2
    ] * 3
    rng = np.random.default_rng(5)
    psi = rng.normal(size=2 ** (N + 1)) + 1j * rng.normal(size=2 ** (N + 1))
    psi /= np.linalg.norm(psi)
    # <Y_1 Y_3> of system qubits 1 and 3, state qubits 2 and 4.
    YY = get_site_operator(PAULIS["Y"], 2, N + 1) @ get_site_operator(
        PAULIS["Y"], 4, N + 1
    )
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    values = exp.expectation_values(psi, ops["YY"])
    assert values[pairs.index((1, 3))] == pytest.approx(np.real(psi.conj() @ YY @ psi))
    Z0 = get_site_operator(PAULIS["Z"], 1, N + 1)
    assert exp.expectation_values(psi, ops["Z"])[0] == pytest.approx(
        np.real(psi.conj() @ Z0 @ psi)
    )


def test_observable_feature_sets() -> None:
    out = _run(
        "readouts_task.readouts=[ridge]",
        "readouts_task.features=[M,Z,XX,all]",
    )
    L = 3 * PER_P
    assert out["features_Z"].shape == (L, N)
    assert out["features_XX"].shape == (L, N * (N - 1) // 2)
    # "all" needs every observable set.
    for name in ("X", "Y", "YY", "ZZ"):
        assert f"features_{name}" in out
    assert np.all(np.abs(out["features_Z"]) <= 1.0 + 1e-9)
    metrics = out["metrics"]
    for name in ("ridge", "ridge_Z", "ridge_XX", "ridge_all"):
        assert 0.0 <= metrics[f"accuracy_{name}_p0.5"] <= 1.0
        assert f"r2_{name}_p1" in metrics
    assert "accuracy_ridge_X_p0.5" not in metrics


def test_haar_gates_do_not_conserve_the_charge() -> None:
    # One layer of Haar gates moves weight out of the charge sectors of the
    # initial state; U(1) gates would keep it there.
    psi0 = AncillaChargeSectorEncoding({"N": N}).state()
    psi = _circuit(1, 2).apply_layer(psi0, 0)
    system = np.arange(2**N)
    charge = np.array([bin(b).count("1") for b in system])
    weight = np.abs(psi.reshape(2, -1)) ** 2
    q1, q2 = N // 2, N // 2 - 1
    leak = weight[0, charge != q1].sum() + weight[1, charge != q2].sum()
    assert leak > 1e-3


def test_outputs_and_scores() -> None:
    out = _run()
    L = 3 * PER_P
    assert out["M"].shape == (L, 5, N)
    assert out["Y"].shape == out["S_R"].shape == out["p"].shape == (L,)
    assert set(np.unique(out["Y"])) <= {-1.0, 1.0}
    metrics = out["metrics"]
    assert metrics["n_rounds"] == 5
    np.testing.assert_allclose(out["S_R"][out["p"] == 0.0], 1.0, atol=1e-9)
    np.testing.assert_allclose(out["S_R"][out["p"] == 1.0], 0.0, atol=1e-9)
    for p in ("p0", "p0.5", "p1"):
        assert -1e-9 <= metrics[f"S_R_mean_{p}"] <= 1.0 + 1e-9
        for name in READOUTS:
            assert 0.0 <= metrics[f"accuracy_{name}_{p}"] <= 1.0
            assert metrics[f"r2_{name}_{p}"] <= 1.0
    # p = 0: the record is empty, every readout predicts the majority label,
    # and its constant output explains nothing (R2 <= 0).
    for name in READOUTS:
        assert metrics[f"accuracy_{name}_p0"] == metrics["accuracy_baseline_p0"]
        assert metrics[f"r2_{name}_p0"] <= 0.0


def test_readouts_learn_a_linear_and_a_parity_rule() -> None:
    exp = _load(SCRIPT)
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 1.0], size=(600, 6))
    linear = np.where(X.sum(1) + 0.5 * X[:, 0] >= 0, 1.0, -1.0)
    parity = X[:, 1] * X[:, 2]
    train, test = slice(0, 400), slice(400, None)
    for y, names in (
        (linear, (*READOUTS, "logistic")),
        (parity, ("ridge_pairs", "gradient_boosting")),
    ):
        for name in names:
            score = exp.READOUTS[name](X[train], y[train], X[test], rng)
            assert np.all(np.abs(score) <= 1.0 + 1e-9) or name.startswith("ridge")
            assert np.mean(np.sign(score) == y[test]) > 0.85, name
    # A parity of two outcomes is invisible to the linear readout.
    score = exp.READOUTS["ridge"](X[train], parity[train], X[test], rng)
    assert np.mean(np.sign(score) == parity[test]) < 0.65


def test_fixed_instance_and_reproducibility() -> None:
    a = _run("seed=3")
    b = _run("seed=3")
    np.testing.assert_array_equal(a["M"], b["M"])
    assert a["metrics"] == b["metrics"]
    c = _run("seed=4")
    assert not np.array_equal(a["M"], c["M"])
    fixed = _run("seed=3", "haar_circuit.fixed_instance=true")
    assert not np.array_equal(a["M"], fixed["M"])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("Nr=0", "Nr=1"),
        ("readouts_task.p_values=[0.2,1.5]", "in \\[0, 1\\]"),
        ("readouts_task.p_values=[0.2,0.2]", "distinct"),
        ("readouts_task.samples_per_p=5", "samples_per_p"),
        ("readouts_task.n_rounds=0", "n_rounds"),
        ("readouts_task.readouts=[ridge,svm]", "Unknown readouts"),
        ("readouts_task.features=[M,XY]", "features"),
        ("models=TFIM", "models"),
    ],
)
def test_rejects_bad_settings(override: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _run(override)


def test_script_writes_metrics(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"N={N}",
            f"readouts_task.p_values={P_VALUES}",
            f"readouts_task.samples_per_p={PER_P}",
            "readouts_task.readouts=[ridge,knn]",
            f"hydra.run.dir={tmp_path}",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for key in ("accuracy_ridge_p1", "r2_knn_p0.5", "S_R_mean_p0", "n_rounds"):
        assert key in metrics
    assert "accuracy_mlp_p1" not in metrics
    data = np.load(tmp_path / "haar_readouts.npz")
    assert data["M"].shape == (3 * PER_P, 5, N)


def test_bash_sweep_and_figures(tmp_path: Path) -> None:
    assert os.access(BASH_SCRIPT, os.X_OK)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "3",
        "SEEDS": "1 2",
        "PER_P": "20",
        "P_VALUES": "0 1",
        "MAX_JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
    )
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == [
        "haar_fixed_N3_seed1",
        "haar_fixed_N3_seed2",
        "haar_random_N3_seed1",
        "haar_random_N3_seed2",
    ]
    cfg = (tmp_path / "haar_fixed_N3_seed1" / ".hydra" / "config.yaml").read_text()
    assert "fixed_instance: true" in cfg
    _load(PLOT_SCRIPT).main([str(tmp_path)])
    for name in READOUTS:
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.json").is_file()


def test_bash_sweep_refuses_bad_settings(tmp_path: Path) -> None:
    for key, value in (("MAX_JOBS", "0"), ("GATES", "clifford"), ("MODELS", "tfim")):
        result = subprocess.run(
            ["bash", str(BASH_SCRIPT)],
            env={**os.environ, "OUT_DIR": str(tmp_path), key: value},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, key


def test_bash_sweep_with_observables(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "GATES": "fixed",
        "N_VALUES": "3",
        "SEEDS": "1",
        "PER_P": "20",
        "P_VALUES": "0 1",
        "READOUTS": "ridge",
        "FEATURES": "Z ZZ",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
    )
    metrics = json.loads(
        (tmp_path / "haar_fixed_N3_seed1" / "metrics.json").read_text()
    )
    assert "accuracy_ridge_ZZ_p1" in metrics and "accuracy_ridge_p1" not in metrics
    plot = _load(PLOT_SCRIPT)
    plot.main([str(tmp_path)])
    assert sorted(p.stem for p in tmp_path.glob("*.png")) == ["ridge_Z", "ridge_ZZ"]
    assert plot.title("ridge_ZZ") == "Ridge (linear) on <Z_i Z_j>"


def test_clifford_circuit_and_qelm_m_comparison(tmp_path: Path) -> None:
    out = _run(
        "models=clifford_circuit",
        "readouts_task.readouts=[ridge]",
        "readouts_task.features=[M,all,M_all]",
    )
    metrics = out["metrics"]
    for name in ("ridge", "ridge_all", "ridge_M_all"):
        assert 0.0 <= metrics[f"accuracy_{name}_p0.5"] <= 1.0
    np.testing.assert_allclose(out["S_R"][out["p"] == 0.0], 1.0, atol=1e-9)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "MODELS": "haar_circuit clifford_circuit",
        "GATES": "fixed",
        "N_VALUES": "3",
        "SEEDS": "1",
        "PER_P": "20",
        "P_VALUES": "0 1",
        "READOUTS": "ridge",
        "FEATURES": "M all M_all",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
    )
    cfg = (tmp_path / "clifford_fixed_N3_seed1" / ".hydra" / "config.yaml").read_text()
    assert "models: clifford_circuit" in cfg
    plot = _load(PLOT_SCRIPT)
    plot.main([str(tmp_path)])
    for circuit in ("haar", "clifford"):
        assert (tmp_path / f"compare_{circuit}.png").is_file()
        assert (tmp_path / f"{circuit}_ridge_M_all.png").is_file()
    assert (
        plot.title("ridge_M_all") == "Ridge (linear) on M and all six observable sets"
    )


def test_benchmark_feature_maps_match_dense_operators() -> None:
    exp = _load(SCRIPT)
    ops = exp.local2_operators(N)
    assert len(ops) == 3 * N + 9 * N * (N - 1) // 2
    rng = np.random.default_rng(5)
    phi = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    phi /= np.linalg.norm(phi)
    psi = np.kron([0.0, 1.0], phi)  # R measured in |1>
    np.testing.assert_allclose(exp.system_state(psi, N), phi)
    # <X_1 Z_3>, the first cross term of the pair (1, 3), on qubits 2 and 4.
    dense = get_site_operator(PAULIS["X"], 2, N + 1) @ get_site_operator(
        PAULIS["Z"], 4, N + 1
    )
    index = 3 * N + 9 * (N - 1) + 9 + 2
    values = exp.expectation_values(psi, ops)
    np.testing.assert_allclose(values[index], np.real(np.vdot(psi, dense @ psi)))
    probs = exp.born_probabilities(psi, N)
    np.testing.assert_allclose(probs, np.abs(phi) ** 2)
    # A basis state gives its own bits: |0110> -> (+1, -1, -1, +1).
    basis = np.zeros(2**N)
    basis[0b0110] = 1.0
    np.testing.assert_array_equal(exp.shot(basis, N, rng), [1.0, -1.0, -1.0, 1.0])


def test_qelm_benchmark_feature_sets() -> None:
    sets = "[local2,born,reservoir,shot,M_shot,M_x_all]"
    out = _run(
        "haar_circuit.fixed_instance=true",
        "readouts_task.readouts=[ridge,logistic]",
        f"readouts_task.features={sets}",
    )
    L = 3 * PER_P
    assert out["features_local2"].shape == (L, 3 * N + 9 * N * (N - 1) // 2)
    assert out["features_born"].shape == out["features_reservoir"].shape
    np.testing.assert_allclose(out["features_born"].sum(1), 1.0)
    np.testing.assert_allclose(out["features_reservoir"].sum(1), 1.0)
    assert set(np.unique(out["features_shot"])) <= {-1.0, 1.0}
    metrics = out["metrics"]
    for name in ("local2", "born", "reservoir", "shot", "M_shot", "M_x_all"):
        for readout in ("ridge", "logistic"):
            assert 0.0 <= metrics[f"accuracy_{readout}_{name}_p0.5"] <= 1.0
    # p = 0 with one fixed circuit: two fixed orthogonal final states, which
    # the linear readouts of their Z-basis probabilities tell apart.
    assert metrics["accuracy_ridge_born_p0"] == 1.0
    assert metrics["accuracy_ridge_reservoir_p0"] == 1.0
    # The benchmark sets draw from their own generators: the trajectories and
    # the observables are those of a run without them.
    base = _run(
        "haar_circuit.fixed_instance=true",
        "readouts_task.readouts=[ridge]",
        "readouts_task.features=[all]",
    )
    np.testing.assert_array_equal(base["M"], out["M"])
    np.testing.assert_array_equal(base["Y"], out["Y"])
    np.testing.assert_allclose(base["features_ZZ"], out["features_ZZ"])


def test_benchmark_figures_merge_two_sweeps(tmp_path: Path) -> None:
    base = {
        **os.environ,
        "GATES": "fixed",
        "N_VALUES": "3",
        "SEEDS": "1",
        "PER_P": "20",
        "P_VALUES": "0 1",
        "TQDM_DISABLE": "1",
    }
    for folder, readouts, features in (
        ("linear", "ridge logistic", "all born shot M_x_all"),
        ("pairs", "ridge_pairs", "all"),
    ):
        env = {
            **base,
            "OUT_DIR": str(tmp_path / folder),
            "READOUTS": readouts,
            "FEATURES": features,
        }
        subprocess.run(
            ["bash", str(BASH_SCRIPT)],
            check=True,
            env=env,
            cwd=tmp_path,
            capture_output=True,
        )
    plot = _load(PLOT_SCRIPT)
    out = tmp_path / "figures"
    plot.main([str(tmp_path / "linear"), str(tmp_path / "pairs"), "--out", str(out)])
    for stem in ("benchmark_haar_fixed_ridge", "benchmark_haar_fixed_logistic"):
        assert (out / f"{stem}.png").is_file()
    table = json.loads((out / "benchmark_haar_fixed_ridge.json").read_text())
    assert "QELM quadratic: products of <.>: accuracy_ridge_pairs_all" in json.dumps(
        table
    )
    assert (out / "ridge_pairs_all.png").is_file()
    assert plot.title("logistic_M_x_all") == (
        "Logistic regression (linear) on M, all six sets and their products"
    )
