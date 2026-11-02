"""The entropy-prediction experiment, experiments/QELM_S/exp_QELM_S.py (#230)."""

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
from QELM.physics import Z, get_site_operator

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "QELM_S"
SCRIPT = FOLDER / "exp_QELM_S.py"
BASH_SCRIPT = FOLDER / "exp_QELM_S.sh"
PLOT_SCRIPT = FOLDER / "plot_QELM_S.py"
FSS_SCRIPT = FOLDER / "plot_entropy_fss.py"
N = 4
P_VALUES = "[0.0,0.3,1.0]"
PER_P = 12


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/QELM_S, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*extra: str):
    return load_config(
        [
            f"N={N}",
            f"entropy_task.p_values={P_VALUES}",
            f"entropy_task.samples_per_p={PER_P}",
            *extra,
        ],
        config_name="QELM_S",
    )


def _run(*extra: str) -> dict:
    return _load(SCRIPT).run(_cfg(*extra))


def test_neel_state_and_its_observables() -> None:
    exp = _load(SCRIPT)
    psi = exp.neel_state(5)
    assert np.flatnonzero(psi).tolist() == [0b01010]
    np.testing.assert_allclose(exp.z_expectations(psi), [1, -1, 1, -1, 1])
    shot = exp.single_shot(psi, np.random.default_rng(0))
    np.testing.assert_array_equal(shot, [1, -1, 1, -1, 1])


def test_z_expectations_match_the_full_operators() -> None:
    exp = _load(SCRIPT)
    rng = np.random.default_rng(2)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    expected = [
        float(np.real(psi.conj() @ get_site_operator(Z, k, 4) @ psi)) for k in range(4)
    ]
    np.testing.assert_allclose(exp.z_expectations(psi), expected, atol=1e-12)


def test_single_shots_follow_the_born_rule() -> None:
    exp = _load(SCRIPT)
    # (|00> + |11>) / sqrt(2): the two qubits always agree, each +-1 half
    # of the time.
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rng = np.random.default_rng(0)
    shots = np.array([exp.single_shot(psi, rng) for _ in range(400)])
    np.testing.assert_array_equal(shots[:, 0], shots[:, 1])
    assert 0.4 < np.mean(shots[:, 0] == 1) < 0.6


def test_measure_round_at_p_zero_and_one() -> None:
    exp = _load(SCRIPT)
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rng = np.random.default_rng(0)
    same, outcomes = exp.measure_round(psi, 0.0, rng)
    np.testing.assert_array_equal(same, psi)
    np.testing.assert_array_equal(outcomes, [0, 0])
    collapsed, outcomes = exp.measure_round(psi, 1.0, rng)
    assert set(np.abs(outcomes)) == {1.0}
    assert outcomes[0] == outcomes[1]
    assert np.count_nonzero(np.abs(collapsed) > 1e-12) == 1


@pytest.mark.parametrize(
    "model",
    ["haar_circuit", "clifford_circuit", "U1_circuit", "floquet_circuit", "TFIM"],
)
def test_shapes_and_range_of_the_outputs(model: str) -> None:
    out = _run(f"models={model}")
    # The Floquet chain runs half-layers: 2 N periods are 4 N rounds, here
    # with N // 2 = 2 even, the last one already on the cut bond.
    L, n_rounds = 3 * PER_P, (4 if model == "floquet_circuit" else 2) * N
    assert out["M"].shape == (L, n_rounds, N)
    assert out["Z"].shape == out["Zshot"].shape == (L, N)
    assert out["S"].shape == out["p"].shape == (L,)
    assert np.all(out["S"] >= -1e-12) and np.all(out["S"] <= N // 2 + 1e-9)
    assert set(np.unique(out["Zshot"])) <= {-1.0, 1.0}
    assert np.all(np.abs(out["Z"]) <= 1 + 1e-12)
    assert set(np.unique(out["M"])) <= {-1.0, 0.0, 1.0}
    assert out["train"].size + out["test"].size == L
    assert out["X"].shape == (L, n_rounds * N + N + 1)
    assert out["S_max"] == N // 2 and out["n_rounds"] == n_rounds


def test_the_record_follows_p() -> None:
    out = _run("models=haar_circuit")
    p, M, S = out["p"], out["M"], out["S"]
    # p = 0: nothing measured; p = 1: everything measured, a product state.
    np.testing.assert_array_equal(M[p == 0.0], 0.0)
    assert np.all(np.abs(M[p == 1.0]) == 1.0)
    np.testing.assert_allclose(S[p == 1.0], 0.0, atol=1e-9)
    # Without measurements, the Haar brickwork entangles the two halves.
    assert S[p == 0.0].mean() > 1.0
    for level in (0.0, 0.3, 1.0):
        assert out[f"S_mean_p{level:g}"] == pytest.approx(S[p == level].mean())


def test_scores_of_every_feature_set() -> None:
    exp = _load(SCRIPT)
    out = _run("models=haar_circuit", "entropy_task.p_c=0.17")
    for name in exp.FEATURE_SETS:
        for key in (f"r2_{name}", f"nmse_{name}", f"r2_within_{name}"):
            assert np.isfinite(out[key]), key
        assert out[f"y_pred_{name}"].shape == out["y_test"].shape
        assert 0.0 <= out[f"accuracy_phase_{name}"] <= 1.0
        if name != "p":
            assert np.isfinite(out[f"r2_knn_{name}"])
    # The p baseline predicts the training mean at each p: it knows the phase
    # label, and nothing about the spread at fixed p.
    assert out["accuracy_phase_p"] == 1.0
    assert out["r2_within_p"] <= 1e-9
    # S has no spread at p = 1, so the R2 there is NaN.
    assert np.isnan(out["r2_within_p_p1"])
    # Per-p R2 only for the plotted sets.
    assert "r2_within_M_Z_p0.3" in out
    assert "r2_within_p_M_Z_even_p0.3" in out
    assert "r2_within_absM_p0.3" not in out
    np.testing.assert_array_equal(out["y_pred"], out["y_pred_M_Z"])
    assert out["r2"] == out["r2_M_Z"]


def test_sets_with_p_hold_the_indicators() -> None:
    exp = _load(SCRIPT)
    rng = np.random.default_rng(0)
    p = np.repeat([0.0, 0.5], 3)
    M = rng.choice([-1.0, 0.0, 1.0], size=(6, 2, 3))
    Z = rng.uniform(-1, 1, size=(6, 3))
    sets = exp.feature_sets(M, Z, np.sign(Z), p, np.array([0.0, 0.5]))
    assert set(sets) == set(exp.FEATURE_SETS)
    for name, base in exp.WITH_P.items():
        np.testing.assert_array_equal(sets[name][:, :2], sets["p"])
        np.testing.assert_array_equal(sets[name][:, 2:], sets[base])
    np.testing.assert_array_equal(sets["absM"], np.abs(M.reshape(6, -1)))
    np.testing.assert_array_equal(sets["n_meas"][:, 0], np.abs(M).sum(axis=(1, 2)))


def test_without_p_c_or_knn() -> None:
    out = _run("models=haar_circuit", "entropy_task.knn=false")
    assert not any(key.startswith(("accuracy_phase_", "r2_knn_")) for key in out)


def test_reproducible_with_the_seed() -> None:
    a = _run("models=haar_circuit", "seed=3")
    b = _run("models=haar_circuit", "seed=3")
    c = _run("models=haar_circuit", "seed=4")
    np.testing.assert_array_equal(a["M"], b["M"])
    np.testing.assert_array_equal(a["S"], b["S"])
    assert not np.array_equal(a["S"], c["S"])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("Nr=1", "without ancilla"),
        ("entropy_task.p_values=[0.2,1.5]", "in \\[0, 1\\]"),
        ("entropy_task.p_values=[0.2,0.2]", "distinct"),
        ("entropy_task.samples_per_p=1", "samples_per_p"),
        ("entropy_task.n_rounds=0", "n_rounds"),
        ("models=kitaev", "not supported"),
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
            "models=haar_circuit",
            f"N={N}",
            f"entropy_task.p_values={P_VALUES}",
            f"entropy_task.samples_per_p={PER_P}",
            f"hydra.run.dir={tmp_path}",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "TQDM_DISABLE": "1"},
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for key in ("r2", "r2_within_M_Z", "r2_p", "r2_knn_M_Z_even", "S_mean_p0.3"):
        assert key in metrics
    data = np.load(tmp_path / "exp_QELM_S.npz")
    assert data["S"].shape == (3 * PER_P,)


def test_bash_sweep_and_figures(tmp_path: Path) -> None:
    assert os.access(BASH_SCRIPT, os.X_OK)
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "MODELS": "haar_circuit TFIM",
        "N_VALUES": "4",
        "SEEDS": "1 2",
        "PER_P": "6",
        "P_VALUES": "0 0.5",
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
        "TFIM_N4_seed1",
        "TFIM_N4_seed2",
        "haar_circuit_N4_seed1",
        "haar_circuit_N4_seed2",
    ]
    cfg = (tmp_path / "haar_circuit_N4_seed1" / ".hydra" / "config.yaml").read_text()
    assert "p_c: 0.17" in cfg
    _load(PLOT_SCRIPT).main([str(tmp_path)])
    for name in ("entropy_vs_p", "r2_summary", "r2_within_vs_p", "predicted_vs_exact"):
        assert (tmp_path / f"{name}.png").is_file()
        assert (tmp_path / f"{name}.json").is_file()


def test_bash_sweep_refuses_a_bad_max_jobs(tmp_path: Path) -> None:
    env = {**os.environ, "OUT_DIR": str(tmp_path), "MAX_JOBS": "0"}
    result = subprocess.run(
        ["bash", str(BASH_SCRIPT)], env=env, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 2


def test_floquet_sweep_and_finite_size_figure(tmp_path: Path) -> None:
    """Issue #239: the sweep runs the Floquet chain and the U(1) circuit, with
    the p_c of Table I of Skinner et al. (2019) for the Floquet chain, and
    plot_entropy_fss.py draws the crossings and the collapse."""
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "MODELS": "floquet_circuit U1_circuit",
        "N_VALUES": "4 6",
        "SEEDS": "1 2",
        "PER_P": "4",
        "P_VALUES": "0 0.2 0.4 0.6",
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
    cfg = (tmp_path / "floquet_circuit_N4_seed1" / ".hydra" / "config.yaml").read_text()
    assert "p_c: 0.21" in cfg
    cfg = (tmp_path / "U1_circuit_N4_seed1" / ".hydra" / "config.yaml").read_text()
    assert "p_c: null" in cfg
    _load(FSS_SCRIPT).main([str(tmp_path)])
    data = json.loads((tmp_path / "entropy_fss.json").read_text())["extra"]
    assert (tmp_path / "entropy_fss.png").is_file()
    assert set(data) >= {"floquet_circuit", "U1_circuit"}
    floquet = data["floquet_circuit"]
    assert floquet["sizes"] == [4, 6]
    assert set(floquet["collapse"]) >= {"p_c", "nu", "cost", "p_c_err", "nu_err"}
    assert 0.05 <= floquet["collapse"]["p_c"] <= 0.45
    assert floquet["reference"]["p_c"] == 0.21
    # At p = 0 the entropy grows with N (volume law).
    S0 = [floquet["curves"][n]["S"][0] for n in ("4", "6")]
    assert S0[1] > S0[0]


def test_floquet_rounds_end_on_the_cut_bond() -> None:
    """The default number of rounds of the Floquet chain ends on the
    half-layer that holds the bond (N/2 - 1, N/2) across the cut."""
    from QELM.models.circuits import FloquetBrickworkCircuit

    exp = _load(SCRIPT)
    for N in (4, 6, 8, 10, 12):
        R = exp.floquet_rounds(N)
        assert R in (4 * N, 4 * N + 1)
        circuit = FloquetBrickworkCircuit({"N": N, "Nr": 0, "depth": R})
        last = [(i, j) for i, j, _ in circuit.layers[-1]]
        assert (N // 2 - 1, N // 2) in last


def test_average_start_takes_an_even_number_of_rounds() -> None:
    """Issue #251: the late-time average starts at floor(f * n_rounds), one
    later when needed so that both parities of the last half-layer weigh the
    same."""
    exp = _load(SCRIPT)
    assert exp.average_start(32, 0.5) == 16
    assert exp.average_start(33, 0.5) == 15
    assert exp.average_start(10, 0.0) == 0
    assert exp.average_start(11, 0.0) == 1
    for R in range(3, 40):
        for f in (0.0, 0.3, 0.5, 0.75):
            start = exp.average_start(R, f)
            assert (R - start) % 2 == 0 and R - start >= 2
            assert abs(start - int(f * R)) <= 1
    with pytest.raises(ValueError, match="average_from"):
        exp.average_start(10, 1.0)
    with pytest.raises(ValueError, match="fewer than two"):
        exp.average_start(1, 0.0)


def test_entanglement_diagnostics_match_the_entropy_functions() -> None:
    from QELM.evaluation import renyi_entropy, tripartite_mutual_information

    exp = _load(SCRIPT)
    rng = np.random.default_rng(3)
    for N in (6, 8):
        psi = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
        psi /= np.linalg.norm(psi)
        half = list(range(N // 2))
        values = exp.entanglement_diagnostics(psi, N)
        expected = [renyi_entropy(psi, n, half) for n in (1, 2, float("inf"))]
        np.testing.assert_allclose(values[:3], expected, atol=1e-10)
        if N % 4 == 0:
            assert values[3] == pytest.approx(tripartite_mutual_information(psi))
        else:
            assert np.isnan(values[3])


def test_late_time_averages() -> None:
    """With average_from, run returns the late-time averages of S_1, S_2,
    S_inf and I_3 (N a multiple of 4) per trajectory and per p."""
    out = _run("models=floquet_circuit", "entropy_task.average_from=0.5")
    R = int(out["n_rounds"])
    assert out["average_start"] == _load(SCRIPT).average_start(R, 0.5)
    assert (R - out["average_start"]) % 2 == 0
    for name in ("S1", "S2", "Sinf", "I3"):
        assert out[f"{name}_avg"].shape == (3 * PER_P,)
        assert f"{name}_avg_mean_p0.3" in out
    # S_1 >= S_2 >= S_inf, all within [0, N/2].
    assert np.all(out["S1_avg"] >= out["S2_avg"] - 1e-12)
    assert np.all(out["S2_avg"] >= out["Sinf_avg"] - 1e-12)
    assert np.all((out["S1_avg"] >= -1e-12) & (out["S1_avg"] <= N // 2 + 1e-12))
    # At p = 1 every qubit is measured each round: a product state.
    p1 = out["p"] == 1.0
    np.testing.assert_allclose(out["S1_avg"][p1], 0, atol=1e-10)
    np.testing.assert_allclose(out["I3_avg"][p1], 0, atol=1e-10)
    # Without the option nothing is added, and the final S is unchanged.
    plain = _run("models=floquet_circuit")
    assert "S1_avg" not in plain
    np.testing.assert_array_equal(plain["S"], out["S"])


def test_late_time_averages_skip_I3_when_4_does_not_divide_N() -> None:
    out = _run("models=floquet_circuit", "N=6", "entropy_task.average_from=0.5")
    assert "S1_avg" in out and "I3_avg" not in out


def test_fss_of_the_late_time_averages(tmp_path: Path) -> None:
    """Issue #251: with AVERAGE_FROM, plot_entropy_fss.py collapses a
    late-time Rényi entropy on the sizes N >= min_N, draws the cost
    landscape and the crossings of I_3."""
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "MODELS": "floquet_circuit",
        "N_VALUES": "4 6 8",
        "SEEDS": "1 2",
        "PER_P": "4",
        "P_VALUES": "0 0.1 0.2 0.3 0.4",
        "AVERAGE_FROM": "0.5",
        "MAX_JOBS": "3",
        "TQDM_DISABLE": "1",
    }
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
    )
    cfg = (tmp_path / "floquet_circuit_N8_seed1" / ".hydra" / "config.yaml").read_text()
    assert "average_from: 0.5" in cfg
    fss = _load(FSS_SCRIPT)
    fss.main([str(tmp_path), "--entropy", "S2_avg", "--min-N", "6", "--p-min", "0.1"])
    data = json.loads((tmp_path / "entropy_fss.json").read_text())["extra"]
    floquet = data["floquet_circuit"]
    assert floquet["entropy"] == "S2_avg"
    assert floquet["reference"]["p_c"] == 0.22
    assert floquet["sizes"] == [4, 6, 8]
    assert floquet["fit"] == {"min_N": 6, "p_min": 0.1}
    land = floquet["landscape"]
    cost = np.asarray(land["cost"])
    assert cost.shape == (len(land["nu"]), len(land["p_c"]))
    # The fitted minimum is at most the lowest cost of the grid.
    assert floquet["collapse"]["cost"] <= cost.min() + 1e-12
    # Statistical (bootstrap) and systematic (choices of the fit) errors.
    collapse = floquet["collapse"]
    for key in ("p_c", "nu"):
        assert collapse[f"{key}_stat"] >= 0 and collapse[f"{key}_sys"] >= 0
        assert collapse[f"{key}_total"] == pytest.approx(
            np.hypot(collapse[f"{key}_stat"], collapse[f"{key}_sys"])
        )
    choices = floquet["collapse_choices"]
    # Sizes 6 and 8 are kept: one smallest size, two p_min, p_max 0.4 only.
    assert [(c["min_N"], c["p_min"]) for c in choices] == [(6, 0.1), (6, 0.15)]
    assert collapse["p_c_sys"] == pytest.approx(
        (max(c["p_c"] for c in choices) - min(c["p_c"] for c in choices)) / 2
    )
    i3 = json.loads((tmp_path / "i3_crossings.json").read_text())["extra"]
    # I_3 needs 4 | N: only N = 4 and 8.
    assert set(i3["floquet_circuit"]["curves"]) == {"4", "8"}
    assert "N4-N8" in i3["floquet_circuit"]["crossings"]
    assert (tmp_path / "i3_crossings.png").is_file()
    # A sweep without AVERAGE_FROM has no late-time average to collapse.
    plain = tmp_path / "plain"
    env |= {"OUT_DIR": str(plain), "N_VALUES": "4", "SEEDS": "1", "AVERAGE_FROM": ""}
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="AVERAGE_FROM"):
        fss.main([str(plain), "--entropy", "S1_avg"])
