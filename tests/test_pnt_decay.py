"""Signal and noise of the QELM state readout against p N T (issue #305)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.special import ndtr

from QELM import load_config
from QELM.models.circuits.base import apply_two_qubit_gate
from QELM.models.circuits.haar import HaarBrickworkCircuit, haar_unitary
from QELM.physics.operators import PAULIS

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "pnt_decay"
SCRIPT = FOLDER / "pnt_decay.py"
BASH_SCRIPT = FOLDER / "pnt_decay.sh"
PLOT_SCRIPT = FOLDER / "plot_pnt_decay.py"


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pd() -> ModuleType:
    return _load(SCRIPT)


def _random_rho(N: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(2**N, 2**N)) + 1j * rng.normal(size=(2**N, 2**N))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def _pauli_string(labels: str) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for c in labels:
        out = np.kron(out, np.eye(2) if c == "I" else PAULIS[c])
    return out


def test_p_grid(pd: ModuleType) -> None:
    p = pd.p_grid(4, 2, [0, 1, 4, 8, 16])
    np.testing.assert_allclose(p, [0.0, 0.125, 0.5, 1.0])


def test_gate_on_density_matrix(pd: ModuleType) -> None:
    rng = np.random.default_rng(0)
    rho = _random_rho(4, rng)
    gate = haar_unitary(4, rng)
    for i, j in ((0, 1), (2, 1), (1, 3)):
        np.testing.assert_allclose(
            pd.apply_gate_rho(rho, gate, i, j),
            apply_two_qubit_gate(rho, gate, i, j),
            atol=1e-12,
        )
    # Against the full unitary on qubits 0 and 1.
    U = np.kron(gate, np.eye(4))
    np.testing.assert_allclose(
        pd.apply_gate_rho(rho, gate, 0, 1), U @ rho @ U.conj().T, atol=1e-12
    )


def test_dephasing_factor_is_the_measurement_channel(pd: ModuleType) -> None:
    """(1 - p)**hamming = the Z-dephasing of probability p on every site."""
    rng = np.random.default_rng(1)
    N, p = 3, 0.3
    rho = _random_rho(N, rng)
    expected = rho.copy()
    for k in range(N):
        P0 = _pauli_string("I" * k + "Z" + "I" * (N - 1 - k))
        P0 = (np.eye(2**N) + P0) / 2
        P1 = np.eye(2**N) - P0
        expected = (1 - p) * expected + p * (P0 @ expected @ P0 + P1 @ expected @ P1)
    np.testing.assert_allclose(
        rho * (1 - p) ** pd.hamming_matrix(N), expected, atol=1e-12
    )


def test_xy_weights(pd: ModuleType) -> None:
    masks = pd.dephasing_masks(3)
    np.testing.assert_allclose(
        pd.xy_weights(_pauli_string("XZI"), masks), [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        pd.xy_weights(_pauli_string("YIX") + _pauli_string("ZZZ"), masks),
        [0.5, 0.0, 0.5],
    )
    # A scrambled (Haar) operator has weight 1/2 on every site.
    U = haar_unitary(2**7, np.random.default_rng(2))
    op = U @ np.diag(np.linspace(-1, 1, 2**7)) @ U.conj().T
    np.testing.assert_allclose(pd.xy_weights(op, pd.dephasing_masks(7)), 0.5, atol=0.05)


def test_feature_expectations_match_the_state_features(pd: ModuleType) -> None:
    rng = np.random.default_rng(3)
    N = 4
    phi = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    phi /= np.linalg.norm(phi)
    psi = np.kron([1.0, 0.0], phi)  # R in |0>, as after its measurement
    np.testing.assert_allclose(
        pd.feature_expectations(np.outer(phi, phi.conj()), N),
        pd.NONLINEAR_M.quantum_features(psi, N),
        atol=1e-12,
    )


def test_averaged_channel_matches_the_trajectories(pd: ModuleType) -> None:
    """The class means of the features are the averaged-channel ones."""
    N, T, p = 3, 3, 0.25
    cfg = load_config([f"N={N}"], config_name="pnt_decay")
    circuit = HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": T}, np.random.default_rng(4)
    )
    psi0 = pd.NONLINEAR_M.initial_state(cfg, "charge")
    a, b = pd.sector_states(psi0, N)
    rp, rm, _ = pd.averaged_states(
        np.outer(a, a.conj()),
        np.outer(b, b.conj()),
        pd.system_layers(circuit),
        p,
        pd.hamming_matrix(N),
    )
    rng = np.random.default_rng(5)
    feats = {1.0: [], -1.0: []}
    for _ in range(6000):
        _, _, _, Y, psi = pd.NONLINEAR_M.trajectory(psi0, circuit, p, rng)
        feats[Y].append(pd.NONLINEAR_M.quantum_features(psi, N))
    for Y, rho in ((1.0, rp), (-1.0, rm)):
        mean = np.mean(feats[Y], axis=0)
        np.testing.assert_allclose(mean, pd.feature_expectations(rho, N), atol=0.05)
    assert np.trace(rp).real == pytest.approx(1.0)


def test_first_order_decay_is_the_effective_volume(pd: ModuleType) -> None:
    """d ln ||Delta||_HS^2 / dp at p = 0 is -V_eff."""
    N, T = 4, 3
    circuit = HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": T}, np.random.default_rng(6)
    )
    layers = pd.system_layers(circuit)
    rng = np.random.default_rng(7)
    rho_p, rho_m = _random_rho(N, rng), _random_rho(N, rng)
    hamming = pd.hamming_matrix(N)
    rp, rm, volume = pd.averaged_states(
        rho_p, rho_m, layers, 0.0, hamming, pd.dephasing_masks(N)
    )
    hs0 = np.sum(np.abs(rp - rm) ** 2)
    eps = 1e-5
    rp, rm, _ = pd.averaged_states(rho_p, rho_m, layers, eps, hamming)
    slope = (np.log(np.sum(np.abs(rp - rm) ** 2)) - np.log(hs0)) / eps
    assert slope == pytest.approx(-volume, rel=1e-3)
    assert 0.0 < volume <= 2 * N * T


def test_oracle_on_gaussian_data(pd: ModuleType) -> None:
    rng = np.random.default_rng(8)
    d, n = 5, 20000
    mu_p, mu_m = np.full(d, 0.3), np.full(d, -0.3)
    A = rng.normal(size=(d, d))
    cov = A @ A.T / d + 0.5 * np.eye(d)
    Y = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    X = np.where(Y[:, None] > 0, mu_p, mu_m) + rng.multivariate_normal(
        np.zeros(d), cov, size=n
    )
    score, snr2, noise = pd.oracle_readout(X[:10000], Y[:10000], X[10000:], mu_p, mu_m)
    s = mu_p - mu_m
    assert snr2 == pytest.approx(s @ np.linalg.solve(cov, s), rel=0.05)
    assert noise == pytest.approx(np.trace(cov) / d, rel=0.05)
    acc = np.mean(np.where(score >= 0, 1.0, -1.0) == Y[10000:])
    assert acc == pytest.approx(ndtr(np.sqrt(snr2) / 2), abs=0.015)


def _run(pd: ModuleType, *extra: str) -> dict:
    cfg = load_config(
        [
            "N=4",
            "pnt_task.T=3",
            "pnt_task.pnt_values=[0,2,6]",
            "pnt_task.samples_per_p=80",
            *extra,
        ],
        config_name="pnt_decay",
    )
    return pd.run(cfg)


def test_run_metrics(pd: ModuleType) -> None:
    out = _run(pd)
    m = out["metrics"]
    assert m["n_rounds"] == 3
    assert 0.0 < m["V_eff"] <= 2 * 4 * 3
    assert m["hs2_p0"] == pytest.approx(1.0)
    assert m["hs2_p0.5"] < m["hs2_p0.166667"] < 1.0
    for key in ("p0", "p0.166667", "p0.5"):
        for name in ("ridge", "oracle", "gauss"):
            assert 0.0 <= m[f"accuracy_{name}_{key}"] <= 1.0
        assert 0.0 <= m[f"trace_distance_{key}"] <= 1.0 + 1e-9
    # p = 0: a fixed circuit maps each sector to one state.
    assert m["accuracy_ridge_p0"] == 1.0
    assert m["accuracy_ridge_K0_p0"] == 1.0
    assert m["K_mean_p0"] == 0.0
    assert m["K_mean_p0.5"] == pytest.approx(0.5 * 4 * 3, rel=0.2)
    assert m["noise_p0.5"] > 0.0 and m["snr2_p0.5"] >= 0.0
    # Reproducible for a seed; the timings stay out of the metrics.
    assert _run(pd)["metrics"] == m
    assert set(out["timing"]) == {"time_exact_s", "time_sampling_s"}


def test_feature_volume_is_the_first_order_signal_rate(pd: ModuleType) -> None:
    """V_feat is -d ln ||s||^2 / dp at p = 0, inside the light cone: <= N T."""
    m = _run(pd, "pnt_task.sample=false", "pnt_task.pnt_values=[0,0.03]")["metrics"]
    N, T = 4, 3
    for name in ("V_feat", "V_Zfeat", "V_XYfeat"):
        assert 0.0 < m[name] <= 2 * N * T, name
    chord = -np.log(m["signal2_p0.0025"] / m["signal2_p0"]) / 0.0025
    assert chord == pytest.approx(m["V_feat"], rel=0.05)
    # sample=false: exact keys only, no trajectories.
    assert not any(k.startswith(("accuracy_", "noise_", "K_mean")) for k in m)


def test_run_refuses_bad_settings(pd: ModuleType) -> None:
    with pytest.raises(ValueError):
        _run(pd, "pnt_task.samples_per_p=10")
    with pytest.raises(ValueError):
        _run(pd, "pnt_task.T=0")
    with pytest.raises(ValueError):
        _run(pd, "Nr=2")


def test_bash_sweep_and_figures(tmp_path: Path) -> None:
    assert os.access(BASH_SCRIPT, os.X_OK)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "ENCODINGS": "charge bell",
        "SIZES": "3:2,4 4:2",
        "SEEDS": "1 2",
        "PER_P": "60",
        "PNT_VALUES": "0 1 3 6",
        "MAX_JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True)
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert len(runs) == 12
    assert "bell_N3_T4_seed2" in runs
    metrics = json.loads((tmp_path / "charge_N4_T2_seed1" / "metrics.json").read_text())
    assert metrics["n_rounds"] == 2 and "time_exact_s" in metrics
    # SKIP_DONE=1 resumes the sweep without rerunning.
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "SKIP_DONE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("already done, skipped") == 12
    plot = _load(PLOT_SCRIPT)
    plot.main([str(tmp_path)])
    for name in plot.FIGURES:
        assert (tmp_path / f"{name}.png").is_file(), name
        assert (tmp_path / f"{name}.json").is_file(), name
    summary = json.loads((tmp_path / "pnt_summary.json").read_text())
    assert set(summary["encodings"]) == {"charge", "bell"}


def test_volume_sweep_fills_old_runs(tmp_path: Path) -> None:
    """SAMPLE=0 writes the volumes only; --volumes adds them to runs lacking them."""
    env = {
        **os.environ,
        "ENCODINGS": "charge",
        "SIZES": "3:2,4",
        "SEEDS": "1 2",
        "PER_P": "60",
        "PNT_VALUES": "0 1 3 6",
        "TQDM_DISABLE": "1",
    }
    main, vol = tmp_path / "main", tmp_path / "vol"
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "OUT_DIR": str(main)},
        capture_output=True,
    )
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "OUT_DIR": str(vol), "SAMPLE": "0", "PNT_VALUES": "0"},
        capture_output=True,
    )
    v = json.loads((vol / "charge_N3_T4_seed1" / "metrics.json").read_text())
    assert "V_feat" in v and "accuracy_ridge_p0" not in v
    # Remove the volumes from the main runs, as in a sweep run before V_feat.
    for path in main.glob("*/metrics.json"):
        m = json.loads(path.read_text())
        path.write_text(json.dumps({k: x for k, x in m.items() if k[:2] != "V_"}))
    plot = _load(PLOT_SCRIPT)
    rows = plot.load_rows([main], [vol])
    assert len(rows) == 4
    assert all(r["V_feat"] == pytest.approx(r["V_feat"]) for r in rows)
    plot.main([str(main), "--volumes", str(vol), "--out", str(tmp_path / "fig")])
    summary = json.loads((tmp_path / "fig" / "pnt_summary.json").read_text())
    for e in summary["encodings"]["charge"]["sizes"].values():
        assert 0.0 < e["V_feat_over_NT"] <= 2.0


def test_bash_sweep_refuses_bad_settings(tmp_path: Path) -> None:
    for key, value in (("MAX_JOBS", "0"), ("SIZES", "4"), ("SIZES", "4:x")):
        result = subprocess.run(
            ["bash", str(BASH_SCRIPT)],
            env={**os.environ, "OUT_DIR": str(tmp_path), key: value},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (key, value)
