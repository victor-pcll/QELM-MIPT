"""The QELM experiment on the measurement record, experiments/QELM_M/exp_QELM_M.py."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM import TransverseFieldIsingModel, load_config
from QELM.encoding import AncillaChargeSectorEmbedding
from QELM.evaluation.metrics import accuracy, balanced_accuracy, majority_accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "QELM_M" / "exp_QELM_M.py"
BASH_SCRIPT = SCRIPT.with_suffix(".sh")
L, N = 50, 4
OVERRIDES = ["models=TFIM", f"N={N}", f"samples.L={L}"]


def _load_script() -> ModuleType:
    """Import experiments/QELM_M/exp_QELM_M.py, which is not a package."""
    spec = importlib.util.spec_from_file_location("exp_QELM_M", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_qubits_measured_at_p_1() -> None:
    """p = 1: every feature is an outcome +-1, plus the bias column."""
    out = _load_script().run(load_config(OVERRIDES))
    features = out["X"][:, :-1]
    # 5 time-evolution steps (the default) x (N - 1) system qubits.
    assert out["X"].shape == (L, 5 * (N - 1) + 1)
    np.testing.assert_array_equal(out["X"][:, -1], 1.0)
    assert set(np.unique(features)) <= {-1.0, 1.0}
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


@pytest.mark.parametrize("p", [0.5, 0.1])
def test_unmeasured_qubits_give_zero_features_not_nan(p: float) -> None:
    """p < 1: skipped qubits are 0 in X, so the fit and the metrics stay
    finite (issue #91), and the share of zeros follows 1 - p."""
    out = _load_script().run(load_config([*OVERRIDES, f"projective_measurement.p={p}"]))
    features = out["X"][:, :-1]
    assert np.all(np.isfinite(out["X"]))
    assert set(np.unique(features)) <= {-1.0, 0.0, 1.0}
    # One draw per sample, measurement step and system qubit.
    assert np.mean(features == 0.0) == pytest.approx(1 - p, abs=0.1)
    assert np.all(np.isfinite(out["W"]))
    assert np.all(np.isfinite(out["y_pred"]))
    assert np.isfinite(out["nmse"]) and np.isfinite(out["r2"])


def _spy_run(monkeypatch, cfg):
    """Run the script and record each propagator U(dt) and each state
    handed to the projective measurement, in call order."""
    module = _load_script()
    propagators: list[tuple[float, np.ndarray]] = []
    measured: list[np.ndarray] = []
    propagator = module.TransverseFieldIsingModel.propagator
    measure = module.ProjectiveMeasurement.measure

    def spy_propagator(self, *, dt):
        U = propagator(self, dt=dt)
        propagators.append((dt, U))
        return U

    def spy_measure(self, rho, *args, **kwargs):
        measured.append(np.array(rho, copy=True))
        return measure(self, rho, *args, **kwargs)

    monkeypatch.setattr(module.TransverseFieldIsingModel, "propagator", spy_propagator)
    monkeypatch.setattr(module.ProjectiveMeasurement, "measure", spy_measure)
    return module.run(cfg), propagators, measured


def test_every_sample_starts_from_the_embedded_state(monkeypatch) -> None:
    """Each sample evolves the embedded state of AncillaChargeSectorEmbedding,
    not the collapsed state left by the previous sample (issue #101)."""
    cfg = load_config(OVERRIDES)
    rho0 = AncillaChargeSectorEmbedding.from_config(cfg).rho()
    _, propagators, measured = _spy_run(monkeypatch, cfg)
    steps = 5  # default projective_measurement.nbr_time_evolution
    assert len(propagators) == L and len(measured) == L * steps
    # The first measurement of sample i sees U_i rho0 U_i^dagger, with U_i
    # acting on the system only (issue #104).
    for (_, U_S), rho in zip(propagators, measured[::steps], strict=True):
        U = np.kron(np.eye(2), U_S)
        np.testing.assert_allclose(rho, U @ rho0 @ U.conj().T, atol=1e-12)


def test_the_tfim_acts_on_the_system_only(monkeypatch) -> None:
    """Issue #104: R is a reference qubit. Each sample draws a TFIM on the
    Ns = N - 1 system qubits, so the propagator has dimension 2**Ns and R
    only sees the identity."""
    _, propagators, _ = _spy_run(monkeypatch, load_config(OVERRIDES))
    assert len(propagators) == L
    for _, U_S in propagators:
        assert U_S.shape == (2 ** (N - 1), 2 ** (N - 1))
        np.testing.assert_allclose(U_S @ U_S.conj().T, np.eye(2 ** (N - 1)), atol=1e-10)


def test_record_has_one_row_per_measurement_step(monkeypatch) -> None:
    """M[i] holds nbr_time_evolution separate measurements, each after a step
    of t / nbr_time_evolution, not one measurement copied N - 1 times
    (issue #102)."""
    steps = 2
    cfg = load_config(
        [*OVERRIDES, f"projective_measurement.nbr_time_evolution={steps}"]
    )
    out, propagators, measured = _spy_run(monkeypatch, cfg)
    assert out["M"].shape == (L, steps, N - 1)
    assert out["X"].shape == (L, steps * (N - 1) + 1)
    np.testing.assert_array_equal(out["X"][:, :-1], out["M"].reshape(L, -1))
    # One U(t / nbr_time_evolution) per sample, applied before each of the
    # nbr_time_evolution measurements.
    dt = float(cfg.evolution.t) / steps
    assert [step for step, _ in propagators] == [pytest.approx(dt)] * L
    assert len(measured) == L * steps
    # Evolution between the measurements changes the outcomes for some samples.
    assert np.any(out["M"][:, 0] != out["M"][:, 1])


def test_cached_propagator_matches_evolve() -> None:
    """U rho U^dagger with U = propagator(dt) is what model.evolve(rho, dt)
    computes, so caching U per sample changes no result."""
    cfg = load_config(OVERRIDES)
    model = TransverseFieldIsingModel.from_config(cfg)
    rho = AncillaChargeSectorEmbedding.from_config(cfg).rho()
    dt = float(cfg.evolution.t) / (N - 1)
    U = model.propagator(dt=dt)
    np.testing.assert_allclose(
        U @ rho @ U.conj().T, model.evolve(rho, dt=dt), atol=1e-10
    )


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_target_is_the_measured_reference_label(p: float) -> None:
    """Issue #104: Y[i] is the outcome +-1 of R measured in Z, and Z0[i] is
    <Z_0> just before, in [-1, 1]."""
    out = _load_script().run(load_config([*OVERRIDES, f"projective_measurement.p={p}"]))
    assert out["Y"].shape == out["Z0"].shape == (L,)
    assert set(np.unique(out["Y"])) <= {-1.0, 1.0}
    assert np.all(np.abs(out["Z0"]) <= 1 + 1e-12)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_label_is_balanced_whatever_p(p: float) -> None:
    """Issue #104: the label is the initial charge sector, +1 or -1 with
    probability 1/2, so its distribution does not depend on p. The old
    target, <Z_0> on the final state, was +-1 at p = 1 and continuous and
    mostly positive below."""
    out = _load_script().run(
        load_config([*OVERRIDES, "samples.L=400", f"projective_measurement.p={p}"])
    )
    # 400 fair coins: standard deviation of the mean 0.05.
    assert np.mean(out["Y"]) == pytest.approx(0.0, abs=0.2)


def test_no_measurement_leaves_the_reference_maximally_mixed() -> None:
    """p = 0: R does not evolve and the system is never measured, so
    rho_R = I / 2 at the end: Z0 = 0 and S(rho_R) = 1 bit in every sample.
    The record holds no information on the label."""
    out = _load_script().run(load_config([*OVERRIDES, "projective_measurement.p=0"]))
    np.testing.assert_allclose(out["Z0"], 0.0, atol=1e-10)
    np.testing.assert_allclose(out["S_vN"], 1.0, atol=1e-9)
    assert out["accuracy_optimal"] == pytest.approx(0.5)


def test_full_measurement_reveals_the_label_through_the_parity() -> None:
    """p = 1: the system TFIM conserves the parity of the system charge,
    Ns // 2 for Y = +1 and Ns // 2 - 1 for Y = -1, and the last round
    measures it: Y = (-1)**(Ns // 2) times the product of the last outcomes.
    Unlike the old target (issue #104), this is learnability, not an
    artefact: the label is fixed before the measurements, and this is the
    record revealing it."""
    out = _load_script().run(load_config(OVERRIDES))
    Ns = N - 1
    parity = (-1) ** (Ns // 2) * np.prod(out["M"][:, -1, :], axis=1)
    np.testing.assert_array_equal(out["Y"], parity)
    np.testing.assert_allclose(out["Z0"], out["Y"], atol=1e-10)
    assert out["accuracy_optimal"] == pytest.approx(1.0)


def test_p_override_sets_the_share_of_measured_qubits() -> None:
    """projective_measurement.p from the command line reaches the
    measurement: the share of non-zero features grows with p."""
    module = _load_script()
    shares = []
    for p in (0.2, 0.5, 0.9):
        cfg = load_config(
            [*OVERRIDES, "samples.L=200", f"projective_measurement.p={p}"]
        )
        assert cfg.projective_measurement.p == p
        features = module.run(cfg)["X"][:, :-1]
        shares.append(np.mean(features != 0.0))
        assert shares[-1] == pytest.approx(p, abs=0.06)
    assert shares == sorted(shares)


def test_reproducible_with_the_seed() -> None:
    module = _load_script()
    a = module.run(load_config([*OVERRIDES, "projective_measurement.p=0.5"]))
    b = module.run(load_config([*OVERRIDES, "projective_measurement.p=0.5"]))
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["Y"], b["Y"])


def test_bash_launcher_dry_run(tmp_path: Path) -> None:
    """Issue #94: the launcher runs end to end with L=10, N=3 and three
    probabilities, with another project's virtualenv active, and writes one
    run folder per p."""
    assert os.access(BASH_SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "10",
            "N_VALUES": "3",
            "P_VALUES": "0.2 0.5 1.0",
            "VIRTUAL_ENV": str(tmp_path / "other-project" / ".venv"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 1 x 1 x 1 x 3 runs" in result.stdout
    assert "does not match the project environment" not in result.stderr
    runs = sorted(tmp_path.glob("exp_QELM_M/*/*"))
    assert [run.name for run in runs] == [
        "N3_seed42_p0.2",
        "N3_seed42_p0.5",
        "N3_seed42_p1.0",
    ]
    for run in runs:
        assert (run / "metrics.json").is_file()
        metrics = json.loads((run / "metrics.json").read_text())
        expected_keys = {"nmse", "nmse_err", "r2", *_load_script().EXTRA_METRICS}
        assert expected_keys <= set(metrics)
        # Without ALPHA the run keeps the config default (issue #92).
        hydra_cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
        assert hydra_cfg.training.alpha == 1e-3
        assert (run / "exp_QELM_M.npz").is_file()
        with np.load(run / "exp_QELM_M.npz") as data:
            assert data["X"].shape == (10, 5 * 2 + 1)
            for name in _load_script().ENTROPIES:
                assert data[name].shape == (10,)
                assert metrics[f"{name}_mean"] == pytest.approx(np.mean(data[name]))


def test_bash_launcher_sweeps_the_field(tmp_path: Path) -> None:
    """Issue #174: H_VALUES sweeps TFIM.h, one run folder per field, named
    with it, and the config of each run holds its field."""
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "10",
            "N_VALUES": "3",
            "H_VALUES": "1 10",
            "P_VALUES": "0.5",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 1 x 2 x 1 x 1 runs" in result.stdout
    runs = sorted(tmp_path.glob("exp_QELM_M/*/*"))
    assert [run.name for run in runs] == [
        "N3_h10_seed42_p0.5",
        "N3_h1_seed42_p0.5",
    ]
    for run, h in zip(runs, (10.0, 1.0), strict=True):
        hydra_cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
        assert float(hydra_cfg.TFIM.h) == h


def test_bash_launcher_refuses_a_field_for_a_circuit(tmp_path: Path) -> None:
    """A circuit has no TFIM block: H_VALUES with MODEL=haar_circuit stops
    before any run instead of failing in Hydra."""
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "MODEL": "haar_circuit",
            "H_VALUES": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "needs MODEL=TFIM" in result.stderr
    assert not list(tmp_path.glob("exp_QELM_M/*/*"))


def test_nbr_time_evolution_must_be_positive() -> None:
    with pytest.raises(ValueError, match="nbr_time_evolution must be >= 1"):
        _load_script().run(
            load_config([*OVERRIDES, "projective_measurement.nbr_time_evolution=0"])
        )


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_nbr_time_evolution_defaults_to_5_whatever_n(n_qubits: int) -> None:
    """The number of time-evolution steps is 5 by default, for every N: it
    no longer follows Ns = N - Nr (issue #123)."""
    cfg = load_config(["models=TFIM", f"N={n_qubits}", "samples.L=4"])
    assert cfg.projective_measurement.nbr_time_evolution == 5
    out = _load_script().run(cfg)
    assert out["M"].shape == (4, 5, n_qubits - 1)


def test_nbr_time_evolution_lives_in_the_measurement_block() -> None:
    """The step count sits with the projective measurement it schedules, in
    conf/training/measurements/projective.yaml, and no longer in an
    experiment block of conf/config.yaml (issue #132)."""
    cfg = load_config(OVERRIDES)
    assert "experiment" not in cfg
    assert cfg.projective_measurement.nbr_time_evolution == 5


@pytest.mark.parametrize(
    "old_key", ["experiment.n_measurements", "experiment.nbr_time_evolution"]
)
def test_old_step_count_keys_are_rejected(old_key: str) -> None:
    """A config still using an old key fails loudly instead of being
    silently ignored (issues #123, #132)."""
    match = "moved to projective_measurement.nbr_time_evolution"
    with pytest.raises(ValueError, match=match):
        _load_script().run(load_config([*OVERRIDES, f"+{old_key}=2"]))


def _binary_entropy(q: np.ndarray) -> np.ndarray:
    """H(q) = -q log2 q - (1 - q) log2(1 - q), with 0 log 0 = 0."""
    q = np.clip(q, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return np.nan_to_num(h, nan=0.0)


@pytest.mark.parametrize("p", [1.0, 0.5, 0.1])
def test_ancilla_entropies_are_recorded(p: float) -> None:
    """One von Neumann and one Rényi-1 entropy of the ancilla per sample, in
    bits, equal to each other (Rényi order 1 is von Neumann), between 0 and
    1, with their means in the results."""
    out = _load_script().run(load_config([*OVERRIDES, f"projective_measurement.p={p}"]))
    S_vN, S_renyi_1 = out["S_vN"], out["S_renyi_1"]
    assert S_vN.shape == S_renyi_1.shape == (L,)
    np.testing.assert_array_equal(S_vN, S_renyi_1)
    # Rounding: an eigenvalue 1 + 1e-16 gives S = -1e-16.
    assert np.all(S_vN >= -1e-12) and np.all(S_vN <= 1.0 + 1e-12)
    assert out["S_vN_mean"] == pytest.approx(np.mean(S_vN))
    assert out["S_renyi_1_mean"] == pytest.approx(np.mean(S_renyi_1))
    # Dephasing rho_R in the Z basis can only raise its entropy, so S(rho_R)
    # is at most the binary entropy of its diagonal (1 +- <Z_0>) / 2.
    assert np.all(S_vN <= _binary_entropy((1 + out["Z0"]) / 2) + 1e-9)


def test_measuring_every_qubit_purifies_the_ancilla() -> None:
    """p = 1: the record fixes the charge parity, hence the reference
    (|<Z_0>| = 1, issue #104), so it is no longer entangled: S = 0."""
    out = _load_script().run(load_config(OVERRIDES))
    np.testing.assert_allclose(out["S_vN"], 0.0, atol=1e-9)


def test_rare_measurements_leave_the_ancilla_entangled() -> None:
    """p = 0.1: most samples keep the ancilla entangled with the system, so
    the mean entropy is well above the p = 1 value of 0."""
    out = _load_script().run(load_config([*OVERRIDES, "projective_measurement.p=0.1"]))
    assert out["S_vN_mean"] > 0.1


@pytest.mark.parametrize("p", [1.0, 0.5, 0.1])
def test_system_entropies_are_recorded(p: float) -> None:
    """Issue #124. The global state is pure, so the system sector has the
    entropy of the ancilla, S_2 <= S_vN, and each single-qubit mean lies in
    [0, 1] bit, with its sample mean in the results."""
    module = _load_script()
    out = module.run(load_config([*OVERRIDES, f"projective_measurement.p={p}"]))
    for name in module.ENTROPIES:
        assert out[name].shape == (L,)
        assert np.all(out[name] >= -1e-12) and np.all(out[name] <= 1 + 1e-12)
        assert out[f"{name}_mean"] == pytest.approx(np.mean(out[name]))
    np.testing.assert_allclose(out["S_vN_sys"], out["S_vN"], atol=1e-9)
    assert np.all(out["S_renyi_2_sys"] <= out["S_vN_sys"] + 1e-9)
    assert np.all(out["S_renyi_2_1q"] <= out["S_vN_1q"] + 1e-9)


def test_measuring_every_qubit_leaves_every_system_qubit_pure() -> None:
    """p = 1: the last step measures every system qubit, so each one is in a
    Z eigenstate and the single-qubit means vanish; the system sector, like
    the ancilla, is pure."""
    out = _load_script().run(load_config(OVERRIDES))
    for name in ("S_vN_sys", "S_renyi_2_sys", "S_vN_1q", "S_renyi_2_1q"):
        np.testing.assert_allclose(out[name], 0.0, atol=1e-9)


def test_half_system_entropy_is_that_of_the_first_half() -> None:
    """Issue #174: S_half is the von Neumann entropy of the first Ns // 2
    system qubits. With N = 6 (Ns = 5, two qubits in the half) it can pass
    1 bit, stays below 2, and the measurements lower it: at p = 1 every
    system qubit is pure, so S_half = 0; at p = 0.1 it stays well above."""
    overrides = ["models=TFIM", "N=6", "samples.L=20"]
    module = _load_script()
    out = module.run(load_config([*overrides, "projective_measurement.p=1.0"]))
    np.testing.assert_allclose(out["S_half"], 0.0, atol=1e-9)
    out = module.run(load_config([*overrides, "projective_measurement.p=0.1"]))
    assert out["S_half"].shape == (20,)
    assert np.all(out["S_half"] >= -1e-12) and np.all(out["S_half"] <= 2 + 1e-12)
    assert out["S_half_mean"] == pytest.approx(np.mean(out["S_half"]))
    assert out["S_half_mean"] > 0.3


def test_half_system_entropy_with_one_qubit_in_the_half() -> None:
    """N = 4 (Ns = 3): the half is the single system qubit 1, so S_half is
    the entropy of that qubit, between 0 and the mean single-qubit entropy
    times Ns (each single-qubit entropy is >= 0)."""
    out = _load_script().run(load_config([*OVERRIDES, "projective_measurement.p=0.3"]))
    assert np.all(out["S_half"] <= 1 + 1e-12)
    assert np.all(out["S_half"] <= 3 * out["S_vN_1q"] + 1e-9)


def test_rare_measurements_leave_the_system_qubits_entangled() -> None:
    """p = 0.1: most system qubits stay entangled with the rest."""
    out = _load_script().run(load_config([*OVERRIDES, "projective_measurement.p=0.1"]))
    assert out["S_vN_1q_mean"] > 0.1


def test_accuracy_is_recorded() -> None:
    """The sign accuracy of the test predictions (issue #124): 1 at p = 1,
    where the label is fixed by the last row of M."""
    out = _load_script().run(load_config(OVERRIDES))
    assert out["accuracy"] == accuracy(out["y_test"], out["y_pred"])
    assert out["accuracy"] == 1.0
    out = _load_script().run(load_config([*OVERRIDES, "projective_measurement.p=0.1"]))
    assert 0.0 <= out["accuracy"] <= 1.0


def test_no_measurement_scores_the_chance_level() -> None:
    """Issue #130: at p = 0 every feature is 0, the readout predicts one
    constant and learns nothing. Its sign accuracy is the share of the
    majority sign of the training labels, equal to the baseline, and its
    balanced accuracy is 0.5."""
    out = _load_script().run(load_config([*OVERRIDES, "projective_measurement.p=0"]))
    np.testing.assert_array_equal(out["X"][:, :-1], 0.0)
    assert np.ptp(out["y_pred"]) == 0.0
    train = len(out["Y"]) - len(out["y_test"])
    assert out["accuracy_baseline"] == majority_accuracy(
        out["Y"][:train], out["y_test"]
    )
    assert out["accuracy"] == out["accuracy_baseline"]
    assert out["balanced_accuracy"] == balanced_accuracy(out["y_test"], out["y_pred"])
    assert out["balanced_accuracy"] == 0.5
