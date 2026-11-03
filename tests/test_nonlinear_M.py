"""Non-linear features of the record M (issue #269)."""

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
from QELM.encoding import AncillaBellPairEncoding
from QELM.features import (
    MODES,
    all_pairs,
    extract_M_features,
    feature_label_correlation,
    light_cone_mask,
    spatial_pairs,
    temporal_pairs,
)
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.physics.operators import PAULIS, get_site_operator

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "nonlinear_M"
SCRIPT = FOLDER / "nonlinear_M.py"
BASH_SCRIPT = FOLDER / "nonlinear_M.sh"
PLOT_SCRIPT = FOLDER / "plot_nonlinear_M.py"
N = 4
P_VALUES = "[0.0,0.3,1.0]"
PER_P = 40


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/nonlinear_M, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*extra: str) -> dict:
    cfg = load_config(
        [
            f"N={N}",
            f"nonlinear_task.p_values={P_VALUES}",
            f"nonlinear_task.samples_per_p={PER_P}",
            *extra,
        ],
        config_name="nonlinear_M",
    )
    return _load(SCRIPT).run(cfg)


# ---- Feature maps -----------------------------------------------------------


def _record() -> np.ndarray:
    # One record, T = 2 rounds, N = 3 qubits.
    return np.array([[[1.0, -1.0, 0.0], [-1.0, -1.0, 1.0]]])


def test_pairs_by_hand() -> None:
    M = _record()
    np.testing.assert_array_equal(spatial_pairs(M), [[[-1.0, 0.0], [1.0, -1.0]]])
    np.testing.assert_array_equal(temporal_pairs(M), [[[-1.0, 1.0, 0.0]]])
    pairs = all_pairs(M)
    assert pairs.shape == (1, 15)
    flat = M.reshape(-1)
    a, b = np.triu_indices(6, k=1)
    np.testing.assert_array_equal(pairs[0], flat[a] * flat[b])


def test_modes_shapes_and_content() -> None:
    rng = np.random.default_rng(0)
    T, n = 3, 5
    M = rng.choice([-1.0, 0.0, 1.0], size=(7, T, n))
    K = T * n
    expected = {
        "raw": K,
        "abs": K,
        "raw_abs": 2 * K,
        "xor_local": K + T * (n - 1) + (T - 1) * n,
        "xor_full": 2 * K + K * (K - 1) // 2,
    }
    for mode, width in expected.items():
        assert extract_M_features(M, mode).shape == (7, width)
    np.testing.assert_array_equal(
        extract_M_features(M, "abs"), np.abs(M.reshape(7, -1))
    )
    full = extract_M_features(M, "xor_full")
    np.testing.assert_array_equal(full[:, :K], M.reshape(7, -1))
    cone = extract_M_features(M, "xor_cone", site=0)
    k = int(light_cone_mask(T, n, 0).sum())
    assert cone.shape == (7, 2 * K + k * (k - 1) // 2)
    assert set(MODES) == {*expected, "xor_cone"}


def test_unmeasured_entries_kill_products() -> None:
    M = np.zeros((1, 2, 2))
    M[0, 0, 0] = 1.0
    assert np.all(extract_M_features(M, "xor_full")[0, 8:] == 0.0)


def test_light_cone() -> None:
    mask = light_cone_mask(3, 9, 4)
    np.testing.assert_array_equal(mask[0], np.abs(np.arange(9) - 4) <= 2)
    assert mask[1].all() and mask[2].all()
    with pytest.raises(ValueError):
        light_cone_mask(3, 9, 9)


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        extract_M_features(np.zeros((2, 3)), "raw")
    with pytest.raises(ValueError):
        extract_M_features(np.full((1, 2, 2), 0.5), "raw")
    with pytest.raises(ValueError):
        extract_M_features(np.zeros((1, 2, 2)), "xor_cone")
    with pytest.raises(ValueError):
        extract_M_features(np.zeros((1, 2, 2)), "cubic")


def test_correlation() -> None:
    Y = np.array([1.0, -1.0, 1.0, -1.0])
    X = np.column_stack([Y, -Y, np.ones(4), [1.0, 1.0, -1.0, -1.0]])
    np.testing.assert_allclose(feature_label_correlation(X, Y), [1.0, -1.0, 0.0, 0.0])


def test_xor_is_invisible_to_a_line_and_seen_by_pairs() -> None:
    # The canonical XOR: Y = m_1 m_2 with independent uniform signs.
    rng = np.random.default_rng(1)
    M = rng.choice([-1.0, 1.0], size=(4000, 1, 2))
    Y = M[:, 0, 0] * M[:, 0, 1]
    raw = feature_label_correlation(extract_M_features(M, "raw"), Y)
    full = feature_label_correlation(extract_M_features(M, "xor_full"), Y)
    assert np.abs(raw).max() < 0.05
    assert full[-1] == pytest.approx(1.0)


# ---- Physics of one trajectory -------------------------------------------------


def test_reference_z_and_quantum_features_against_dense_operators() -> None:
    module = _load(SCRIPT)
    rng = np.random.default_rng(3)
    n = N + 1
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    Z0 = get_site_operator(PAULIS["Z"], 0, n)
    assert module.reference_z(psi) == pytest.approx(
        float(np.real(psi.conj() @ Z0 @ psi))
    )
    feats = module.quantum_features(psi, N)
    assert feats.shape == (3 * N + 3 * N * (N - 1) // 2,)
    X2 = get_site_operator(PAULIS["X"], 3, n)
    assert feats[2] == pytest.approx(float(np.real(psi.conj() @ X2 @ psi)))
    # First pair of the YY block: <Y_0 Y_1> of the system.
    Y1 = get_site_operator(PAULIS["Y"], 1, n)
    Y2 = get_site_operator(PAULIS["Y"], 2, n)
    first_yy = 3 * N + N * (N - 1) // 2
    assert feats[first_yy] == pytest.approx(float(np.real(psi.conj() @ Y1 @ Y2 @ psi)))


def test_trajectory_without_measurement_keeps_the_bell_pair() -> None:
    module = _load(SCRIPT)
    circuit = HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": 3}, np.random.default_rng(5)
    )
    psi0 = AncillaBellPairEncoding({"N": N}).state()
    M, r_z, S_R, Y, _ = module.trajectory(psi0, circuit, 0.0, np.random.default_rng(6))
    assert np.all(M == 0.0)
    assert r_z == pytest.approx(0.0, abs=1e-12)
    assert S_R == pytest.approx(1.0)
    assert Y in (-1.0, 1.0)


def test_depth_rule() -> None:
    module = _load(SCRIPT)
    assert module.depth_from_rule("N", 8) == 8
    assert module.depth_from_rule("2N", 8) == 16
    assert module.depth_from_rule(10, 8) == 10
    for bad in ("xN", "N2", "0", "-3"):
        with pytest.raises(ValueError):
            module.depth_from_rule(bad, 8)


# ---- The experiment -----------------------------------------------------------


def test_run_metrics_and_reproducibility() -> None:
    first = _run("seed=7")
    second = _run("seed=7")
    assert first["metrics"] == second["metrics"]
    np.testing.assert_array_equal(first["M"], second["M"])
    metrics = first["metrics"]
    assert first["M"].shape == (3 * PER_P, N, N)
    for p in ("p0", "p0.3", "p1"):
        for name in (
            "bayes",
            "baseline",
            "ridge_quantum",
            "mlp_raw",
            *(f"ridge_{m}" for m in MODES),
        ):
            assert 0.0 <= metrics[f"accuracy_{name}_{p}"] <= 1.0
        for group in _load(SCRIPT).GROUPS:
            assert 0.0 <= metrics[f"corr_rms_{group}_{p}"] <= 1.0
    # Empty record at p = 0: R is maximally mixed and nothing is learned,
    # but the final state (after R is measured) holds the label.
    assert metrics["S_R_mean_p0"] == pytest.approx(1.0)
    assert metrics["abs_rz_mean_p0"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["accuracy_bayes_expected_p0"] == pytest.approx(0.5)
    assert metrics["accuracy_ridge_quantum_p0"] == 1.0
    # A different seed draws another circuit.
    assert _run("seed=8")["metrics"] != metrics


def test_bayes_expected_matches_born() -> None:
    # sign(r_z) is right with probability (1 + |r_z|) / 2 under the Born rule.
    out = _run(
        "seed=9",
        "nonlinear_task.p_values=[1.0]",
        "nonlinear_task.samples_per_p=600",
        "nonlinear_task.modes=[raw]",
        "nonlinear_task.quantum_baseline=false",
        "nonlinear_task.mlp_reference=false",
    )
    r_z, Y = out["r_z"], out["Y"]
    hits = np.mean(np.where(r_z >= 0, 1.0, -1.0) == Y)
    expected = np.mean((1 + np.abs(r_z)) / 2)
    assert abs(hits - expected) < 4 * 0.5 / np.sqrt(r_z.size)
    assert np.all(np.abs(r_z) <= 1 + 1e-12)


def test_run_refuses_bad_settings() -> None:
    with pytest.raises(ValueError):
        _run("nonlinear_task.modes=[cubic]")
    with pytest.raises(ValueError):
        _run("nonlinear_task.p_values=[1.5]")
    with pytest.raises(ValueError):
        _run("nonlinear_task.samples_per_p=5")


def test_sweep_and_figures(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path / "sweep"),
        "N_VALUES": "4",
        "SEEDS": "1 2",
        "PER_P": "30",
        "P_VALUES": "0 0.5 1",
        "MODES": "raw abs xor_local",
        "MLP": "false",
        "SAVE_ARRAYS": "false",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=REPO_ROOT)
    sweep = tmp_path / "sweep"
    assert (sweep / "N4_seed1" / "metrics.json").exists()
    assert not (sweep / "N4_seed1" / "nonlinear_M.npz").exists()
    corr_map = json.loads((sweep / "N4_seed1" / "corr_map.json").read_text())
    assert np.asarray(corr_map["corr_m"]).shape == (3, 4, 4)
    # SKIP_DONE leaves the finished runs alone.
    stamp = (sweep / "N4_seed1" / "metrics.json").stat().st_mtime_ns
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "SKIP_DONE": "1"},
        cwd=REPO_ROOT,
    )
    assert (sweep / "N4_seed1" / "metrics.json").stat().st_mtime_ns == stamp
    subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), str(sweep), "--out", str(tmp_path / "fig")],
        check=True,
    )
    for name in (
        "accuracy_vs_p_N4.png",
        "gain_vs_p.png",
        "correlation_spectrum.png",
        "correlation_map_N4.png",
    ):
        assert (tmp_path / "fig" / name).stat().st_size > 0
    summary = json.loads((tmp_path / "fig" / "summary.json").read_text())
    assert summary["N4"]["n_circuits"] == 2
    assert len(summary["N4"]["accuracy_ridge_raw"]) == 3
