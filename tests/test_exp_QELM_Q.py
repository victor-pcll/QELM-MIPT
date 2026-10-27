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


def test_record_of_the_intermediate_measurements() -> None:
    """M holds one row per round, +-1 for a measured qubit, 0 otherwise (#179)."""
    Ns = N - 1
    for p, values in ((0.0, {0.0}), (0.5, {-1.0, 0.0, 1.0}), (1.0, {-1.0, 1.0})):
        out = _run(f"projective_measurement.p={p}")
        # 5 steps, a round between two: 4 rounds.
        assert out["M"].shape == (L, 4, Ns)
        assert set(np.unique(out["M"])) <= values
        np.testing.assert_array_equal(
            np.count_nonzero(out["M"], axis=(1, 2)), out["n_measured"]
        )
    # With the final round, its outcomes are the final <Z_k> at p = 1.
    final = _run(
        "projective_measurement.p=1.0", "projective_measurement.final_measurement=true"
    )
    assert final["M"].shape == (L, 5, Ns)
    np.testing.assert_allclose(final["M"][:, -1], final["X"][:, :-1], atol=1e-12)


def test_readouts_on_the_record_and_the_observables() -> None:
    """phi = [M, X]: the record first, then the observables, then the bias."""
    module = _load_script()
    out = _run("projective_measurement.p=0.5")
    record = out["M"].reshape(L, -1)
    for name, sets in module.RECORD_SETS.items():
        expected = np.hstack(
            [record, *(out[f"features_{s}"][:, :-1] for s in sets), np.ones((L, 1))]
        )
        np.testing.assert_array_equal(out[f"features_{name}"], expected)
        for score in ("accuracy", "balanced_accuracy", "r2"):
            assert np.isfinite(out[f"{score}_{name}"])
    # At p = 0 the record is all zeros: it adds nothing to <Z_k> or <X_i X_j>,
    # and alone it predicts a constant.
    at_p0 = _run("projective_measurement.p=0.0")
    for name, obs in (("M_Z", "Z"), ("M_XX", "XX")):
        assert at_p0[f"accuracy_{name}"] == at_p0[f"accuracy_{obs}"]
        np.testing.assert_allclose(at_p0[f"r2_{name}"], at_p0[f"r2_{obs}"], atol=1e-10)
    assert at_p0["balanced_accuracy_M"] == 0.5


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


@pytest.mark.parametrize(
    "extra",
    [
        ("projective_measurement.p=0.5",),
        (
            "projective_measurement.p=1.0",
            "projective_measurement.final_measurement=true",
        ),
        ("projective_measurement.p=0.3", "N=5", "TFIM.h=1"),
        ("projective_measurement.p=0.5", "TFIM.h=1", "TFIM.basis=Z"),
    ],
)
def test_state_vector_matches_the_density_matrix(extra: tuple[str, ...]) -> None:
    """state_representation=vector gives the results of rho for a seed (#183)."""
    rho = _run(*extra, "state_representation=density")
    psi = _run(*extra, "state_representation=vector")
    np.testing.assert_array_equal(psi["Y"], rho["Y"])
    np.testing.assert_array_equal(psi["M"], rho["M"])
    np.testing.assert_array_equal(psi["n_measured"], rho["n_measured"])
    for key in rho:
        if key.startswith("features_"):
            np.testing.assert_allclose(psi[key], rho[key], atol=1e-10)
    np.testing.assert_allclose(psi["S_half"], rho["S_half"], atol=1e-10)
    for key in ("accuracy", "accuracy_M_Z", "accuracy_XX"):
        assert psi[key] == rho[key]


def test_density_matrix_is_the_default() -> None:
    np.testing.assert_array_equal(
        _run("projective_measurement.p=0.5")["X"],
        _run("projective_measurement.p=0.5", "state_representation=density")["X"],
    )


def test_rejects_an_unknown_state_representation() -> None:
    with pytest.raises(ValueError, match="state_representation"):
        _run("state_representation=mps")


def test_evolve_with_measurements_takes_a_step_function() -> None:
    """A callable step evolves a state vector like U rho U^dagger evolves rho."""
    module = _load_script()
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    U = expm(-1j * (A + A.conj().T))
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    P = ProjectiveMeasurement({"p": 0.5, "Nr": 1})
    out_psi, n_psi = module.evolve_with_measurements(
        psi, lambda v: U @ v, 3, P, np.random.default_rng(1)
    )
    out_rho, n_rho = module.evolve_with_measurements(
        np.outer(psi, psi.conj()), U, 3, P, np.random.default_rng(1)
    )
    assert n_psi == n_rho
    np.testing.assert_allclose(np.outer(out_psi, out_psi.conj()), out_rho, atol=1e-12)


@pytest.mark.parametrize(("delta", "N_", "L_"), [(10, 4, 40), (7.5, 5, 37)])
def test_delta_sets_the_number_of_samples(delta: float, N_: int, L_: int) -> None:
    """samples.delta replaces L by int(delta * N); train_size follows (#201)."""
    module = _load_script()
    cfg = load_config(
        [*OVERRIDES, f"N={N_}", f"samples.delta={delta}", "samples.L=999"]
    )
    out = module.run(cfg)
    assert out["delta"] == delta
    assert out["Y"].shape == (L_,)
    assert cfg.samples.L == L_
    assert out["y_test"].shape == (L_ - int(0.7 * L_),)


def test_without_delta_l_is_kept() -> None:
    out = _run("projective_measurement.p=0.5")
    assert out["delta"] is None
    assert out["Y"].shape == (L,)


@pytest.mark.parametrize("delta", ["0", "-1", "0.1"])
def test_rejects_a_delta_without_samples(delta: str) -> None:
    with pytest.raises(ValueError, match="samples.delta"):
        _run(f"samples.delta={delta}")


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
        "accuracy_M_Z",
        "r2_M_XX",
    ):
        assert key in metrics
    assert (tmp_path / runs[0] / "exp_QELM_Q.npz").is_file()


def test_bash_sweep_in_parallel_with_the_state_vector(tmp_path: Path) -> None:
    """JOBS=2 runs two at a time, STATE=vector reaches the config (#183)."""
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "3 4",
        "H_VALUES": "1",
        "SEEDS": "42",
        "P_VALUES": "0.0 0.5 1.0",
        "L": "10",
        "STATE": "vector",
        "JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=tmp_path)
    runs = sorted(tmp_path.glob("*/metrics.json"))
    assert len(runs) == 6
    for run in runs:
        config = json.loads((run.parent / "resolved_config.json").read_text())
        assert config["state_representation"] == "vector"


def test_bash_sweep_with_delta_scales_l_with_n(tmp_path: Path) -> None:
    """DELTA=7.5 gives L = int(7.5 N): 22 at N = 3, 30 at N = 4 (#201)."""
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "3 4",
        "SEEDS": "42",
        "P_VALUES": "0.5",
        "L": "999",  # ignored with DELTA
        "DELTA": "7.5",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=tmp_path)
    for N_, L_ in ((3, 22), (4, 30)):
        (run,) = tmp_path.glob(f"N{N_}_*/metrics.json")
        metrics = json.loads(run.read_text())
        assert metrics["n_samples"] == L_
        assert metrics["delta"] == 7.5
        config = json.loads((run.parent / "resolved_config.json").read_text())
        assert config["samples"]["L"] == L_
        assert config["samples"]["delta"] == 7.5


def test_bash_sweep_in_parallel_fails_when_a_run_fails(tmp_path: Path) -> None:
    """A failed run fails the sweep but not the runs queued after it (#199)."""
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "3",
        "SEEDS": "42",
        "P_VALUES": "0.0 2.0 1.0",  # p = 2 is not a probability
        "L": "10",
        "MAX_JOBS": "2",
        "TQDM_DISABLE": "1",
    }
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 1
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == [
        "N3_h10_t10_finalfalse_seed42_p0.0",
        "N3_h10_t10_finalfalse_seed42_p1.0",
    ]
    failed = (tmp_path / "failed_runs.txt").read_text().splitlines()
    assert failed == ["exit 1: 3 10 10 false 42 2.0"]


# A stand-in for `uv` (first on PATH): each call records how many calls run at
# that moment, sleeps, writes metrics.json into hydra.run.dir, and kills
# itself with SIGKILL, as the OOM killer would, when p = 0.5.
FAKE_UV = r"""#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    hydra.run.dir=*) run_dir="${arg#hydra.run.dir=}" ;;
    projective_measurement.p=*) p="${arg#projective_measurement.p=}" ;;
  esac
done
mkdir -p "$ACTIVE_DIR" "$run_dir"
touch "$ACTIVE_DIR/$$"
ls "$ACTIVE_DIR" | wc -l >> "$COUNT_LOG"
sleep 0.3
rm "$ACTIVE_DIR/$$"
if [[ "$p" == "0.5" ]]; then kill -9 $$; fi
echo "{}" > "$run_dir/metrics.json"
"""


def _fake_uv_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text(FAKE_UV)
    fake.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "OUT_DIR": str(tmp_path / "sweep"),
        "ACTIVE_DIR": str(tmp_path / "active"),
        "COUNT_LOG": str(tmp_path / "counts.txt"),
        "N_VALUES": "3",
        "SEEDS": "42 43",
        "L": "10",
        **extra,
    }


def test_bash_sweep_keeps_at_most_max_jobs_runs_alive(tmp_path: Path) -> None:
    """MAX_JOBS=3 over 8 runs: never more than 3 at once, and 3 reached (#199)."""
    env = _fake_uv_env(tmp_path, P_VALUES="0.0 0.2 0.8 1.0", MAX_JOBS="3")
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=tmp_path)
    counts = [int(c) for c in (tmp_path / "counts.txt").read_text().split()]
    assert len(counts) == 8
    assert max(counts) == 3
    assert len(list((tmp_path / "sweep").glob("*/metrics.json"))) == 8
    # Nothing failed, so no list of failures is left behind.
    assert not (tmp_path / "sweep" / "failed_runs.txt").exists()


def test_bash_sweep_survives_a_run_killed_out_of_memory(tmp_path: Path) -> None:
    """A run killed by SIGKILL (exit 137) is listed; the queue goes on (#199)."""
    env = _fake_uv_env(tmp_path, P_VALUES="0.0 0.5 1.0", MAX_JOBS="2")
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 1
    sweep = tmp_path / "sweep"
    assert len(list(sweep.glob("*/metrics.json"))) == 4
    failed = sorted((sweep / "failed_runs.txt").read_text().splitlines())
    assert failed == [
        "exit 137: 3 10 10 false 42 0.5",
        "exit 137: 3 10 10 false 43 0.5",
    ]


def test_bash_sweep_refuses_a_grid_with_the_same_run_twice(tmp_path: Path) -> None:
    """P_VALUES="0.5 0.5" would make two runs write to one folder (#199)."""
    env = _fake_uv_env(tmp_path, P_VALUES="0.5 0.5")
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 2
    assert b"same run twice" in result.stderr
    assert not (tmp_path / "counts.txt").exists()


@pytest.mark.parametrize("max_jobs", ["0", "-1", "two"])
def test_bash_sweep_refuses_a_bad_max_jobs(tmp_path: Path, max_jobs: str) -> None:
    env = _fake_uv_env(tmp_path, P_VALUES="0.0", MAX_JOBS=max_jobs)
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 2
    assert b"MAX_JOBS" in result.stderr


def test_bash_sweep_writes_failures_to_failed_log(tmp_path: Path) -> None:
    """FAILED_LOG moves the list of failures, one per Slurm array task (#199)."""
    log = tmp_path / "logs" / "task_3.txt"
    env = _fake_uv_env(tmp_path, P_VALUES="0.5", SEEDS="42", FAILED_LOG=str(log))
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 1
    assert log.read_text() == "exit 137: 3 10 10 false 42 0.5\n"
    assert not (tmp_path / "sweep" / "failed_runs.txt").exists()
