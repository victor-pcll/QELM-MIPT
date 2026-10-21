"""The QELM experiment on the charge Q, experiments/QELM_Q/exp_QELM_Q.py (#163)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.linalg import expm

from QELM import load_config
from QELM.physics import Z, get_site_operator
from QELM.training import ProjectiveMeasurement

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_Q" / "exp_QELM_Q.py"
BASH_SCRIPT = SCRIPT.with_suffix(".sh")
L, N = 40, 4
OVERRIDES = ["models=TFIM", f"N={N}", f"samples.L={L}"]


def _load_script() -> ModuleType:
    """Import experiments/QELM_Q/exp_QELM_Q.py, which is not a package."""
    spec = importlib.util.spec_from_file_location("exp_QELM_Q", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*extra: str) -> dict:
    return _load_script().run(load_config([*OVERRIDES, *extra]))


def test_features_are_final_z_and_target_is_the_charge() -> None:
    """X holds Ns values <Z_k> in [-1, 1] plus the bias; Y is Q1 or Q2."""
    out = _run("projective_measurement.p=0.5")
    assert out["X"].shape == (L, N - 1 + 1)
    np.testing.assert_array_equal(out["X"][:, -1], 1.0)
    assert np.all(np.abs(out["X"][:, :-1]) <= 1 + 1e-12)
    Ns = N - 1
    assert set(np.unique(out["Y"])) <= {Ns // 2, Ns // 2 - 1}
    assert 0.0 <= out["accuracy"] <= 1.0
    assert np.isfinite(out["r2"]) and np.isfinite(out["nmse"])


def test_p_zero_recovers_the_standard_qelm() -> None:
    """p = 0 with 5 steps is one unitary exp(-i H t), as with 1 step (#163)."""
    at_p0 = _run("projective_measurement.p=0.0")
    standard = _run("projective_measurement.nbr_time_evolution=1")
    assert at_p0["n_measured_mean"] == 0
    np.testing.assert_array_equal(at_p0["Y"], standard["Y"])
    np.testing.assert_allclose(at_p0["X"], standard["X"], atol=1e-10)
    assert at_p0["accuracy"] == standard["accuracy"]


def test_same_hamiltonians_and_labels_for_every_p() -> None:
    """The measurements draw from their own generator: Y does not move with p."""
    low, high = (
        _run("projective_measurement.p=0.2"),
        _run("projective_measurement.p=1.0"),
    )
    np.testing.assert_array_equal(low["Y"], high["Y"])
    # 4 intermediate rounds of Ns qubits at p = 1.
    assert high["n_measured_mean"] == 4 * (N - 1)
    assert 0 < low["n_measured_mean"] < high["n_measured_mean"]


def test_outcomes_are_not_features() -> None:
    """At p = 1 the last step evolves after the last collapse, so the final
    <Z_k> are not the +-1 outcomes of a measurement."""
    out = _run("projective_measurement.p=1.0")
    features = out["X"][:, :-1]
    assert not np.all(np.isin(features, [-1.0, 1.0]))


def test_final_round_measures_after_the_last_step() -> None:
    """final_measurement=true at p = 1: 5 rounds, the features are outcomes."""
    out = _run(
        "projective_measurement.p=1.0", "projective_measurement.final_measurement=true"
    )
    assert out["n_measured_mean"] == 5 * (N - 1)
    np.testing.assert_allclose(np.abs(out["X"][:, :-1]), 1.0, atol=1e-12)
    # Same disorder and labels as without the final round.
    np.testing.assert_array_equal(out["Y"], _run("projective_measurement.p=1.0")["Y"])


def test_final_round_changes_nothing_at_p_zero() -> None:
    a = _run("projective_measurement.p=0.0")
    b = _run(
        "projective_measurement.p=0.0", "projective_measurement.final_measurement=true"
    )
    np.testing.assert_array_equal(a["X"], b["X"])


def test_six_feature_sets() -> None:
    """One readout per set; <X_k> = <Y_k> = 0, since the dynamics and the Z
    measurements keep the parity of Q and X_k, Y_k flip it."""
    out = _run("projective_measurement.p=0.5")
    module = _load_script()
    Ns = N - 1
    pairs = Ns * (Ns - 1) // 2
    for name in module.FEATURE_SETS:
        phi = out[f"features_{name}"]
        width = {1: Ns, 2: pairs}[len(name)] * (2 if name == "XY" else 1)
        assert phi.shape == (L, width + 1)
        assert np.all(np.abs(phi) <= 1 + 1e-12)
        for score in ("accuracy", "balanced_accuracy", "r2"):
            assert np.isfinite(out[f"{score}_{name}"])
    np.testing.assert_array_equal(out["features_Z"], out["X"])
    assert out["accuracy_Z"] == out["accuracy"]
    for name in ("X", "Y"):
        np.testing.assert_allclose(out[f"features_{name}"][:, :-1], 0.0, atol=1e-12)
    assert np.any(np.abs(out["features_XX"][:, :-1]) > 1e-3)


def test_observable_features_match_the_full_operators() -> None:
    module = _load_script()
    n = 3
    rng = np.random.default_rng(1)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    system = [0, 1, 2]
    for name in module.FEATURE_SETS:
        if len(name) == 1:
            P = module.PAULIS[name]
            expected = [
                np.real(np.trace(get_site_operator(P, k, n) @ rho)) for k in system
            ]
        else:
            orders = [name] if name[0] == name[1] else [name, name[::-1]]
            expected = [
                np.real(
                    np.trace(
                        get_site_operator(module.PAULIS[a], i, n)
                        @ get_site_operator(module.PAULIS[b], j, n)
                        @ rho
                    )
                )
                for i, j in ((0, 1), (0, 2), (1, 2))
                for a, b in orders
            ]
        np.testing.assert_allclose(
            module.observable_features(rho, system, name), expected, atol=1e-12
        )
    with pytest.raises(ValueError, match="feature set"):
        module.observable_features(rho, system, "XZ")


def test_reproducible_with_the_seed() -> None:
    a, b = _run("projective_measurement.p=0.5"), _run("projective_measurement.p=0.5")
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["Y"], b["Y"])


def test_evolve_with_measurements_no_round_after_the_last_step() -> None:
    """One step and p = 1: no measurement at all, rho -> U rho U^dagger."""
    module = _load_script()
    n = 2
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    U = expm(-1j * (A + A.conj().T))
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1
    rho = np.outer(psi, psi.conj())
    P = ProjectiveMeasurement({"p": 1.0, "Nr": 0})
    out, n_measured = module.evolve_with_measurements(rho, U, 1, P, rng)
    np.testing.assert_allclose(out, U @ rho @ U.conj().T, atol=1e-12)
    assert n_measured == 0
    # Two steps: one round in between, every qubit collapsed once.
    out, n_measured = module.evolve_with_measurements(rho, U, 2, P, rng)
    assert n_measured == n
    # With the final round: one more round, after the last step.
    _, n_final = module.evolve_with_measurements(rho, U, 2, P, rng, final=True)
    assert n_final == 2 * n
    np.testing.assert_allclose(np.trace(out), 1.0, atol=1e-12)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    for k in range(n):
        z = np.real(np.trace(get_site_operator(Z, k, n) @ out))
        assert -1 <= z <= 1


def test_rejects_zero_steps() -> None:
    with pytest.raises(ValueError, match="nbr_time_evolution"):
        _run("projective_measurement.nbr_time_evolution=0")


def test_bash_sweep_writes_one_folder_per_run(tmp_path: Path) -> None:
    """The launcher runs N x h x final x seed x p and writes h into the config."""
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "3",
        "H_VALUES": "10 1",
        "SEEDS": "42",
        "P_VALUES": "0.0 1.0",
        "FINAL_VALUES": "false true",
        "L": "10",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=tmp_path)
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == sorted(
        f"N3_h{h}_t10_final{final}_seed42_p{p}"
        for h in (10, 1)
        for final in ("false", "true")
        for p in ("0.0", "1.0")
    )
    metrics = json.loads((tmp_path / runs[0] / "metrics.json").read_text())
    for key in (
        "accuracy",
        "accuracy_baseline",
        "balanced_accuracy",
        "S_half_mean",
        "accuracy_XX",
        "r2_ZZ",
    ):
        assert key in metrics
    assert (tmp_path / runs[0] / "exp_QELM_Q.npz").is_file()
