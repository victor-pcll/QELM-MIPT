"""Convergence of the QELM ridge in PER_P (per_p_convergence, issue #267)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest

from QELM import load_config
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.evaluation.entropy import von_neumann_entropy
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.training.measurements.projective import measure_projective_psi

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "per_p_convergence"
N = 4


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package.

    The module is registered under its name, and its folder put on the path,
    so that the worker processes of ``run`` can import it by name.
    """
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


pc = _load(FOLDER / "per_p_convergence.py")
hr = _load(REPO_ROOT / "experiments" / "haar_readouts" / "haar_readouts.py")


def _states(n_qubits: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(B, 2**n_qubits)) + 1j * rng.normal(size=(B, 2**n_qubits))
    return np.asarray(psi / np.linalg.norm(psi, axis=1)[:, None])


def _cfg(*extra: str):  # type: ignore[no-untyped-def]
    return load_config(
        [
            f"N={N}",
            "convergence.n_circuits=2",
            "convergence.p_values=[0.1,0.4]",
            "convergence.regimes=[0.1,0.4]",
            "convergence.per_p_values=[20,40,80]",
            "convergence.n_rounds=2",
            "convergence.batch_size=30",
            "convergence.n_jobs=1",
            *extra,
        ],
        config_name="per_p_convergence",
    )


def test_features_match_dense_operators() -> None:
    phi = _states(N, 5, 0)
    features = pc.qelm_features(phi, N)
    ops = hr.observables(N)
    for b in range(5):
        # The operators of haar_readouts act on R (qubit 0) and the system.
        full = np.concatenate([phi[b], np.zeros(2**N)])
        expected = np.concatenate(
            [hr.expectation_values(full, ops[name]) for name in hr.OBSERVABLE_SETS]
        )
        np.testing.assert_allclose(features[b], expected, atol=1e-12)
    assert features.shape == (5, 3 * N + 3 * N * (N - 1) // 2)


def test_pauli_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown Pauli"):
        pc.apply_pauli_batch(_states(2, 1, 0), "W", 0)


def test_gates_match_the_circuit() -> None:
    psi = _states(N + 1, 4, 1)
    circuit = HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": 2}, np.random.default_rng(2)
    )
    out = psi
    for t in range(2):
        for i, j, gate in circuit.layers[t]:
            out = pc.apply_gate_batch(out, gate, i, j)
    for b in range(4):
        expected = circuit.apply_layer(circuit.apply_layer(psi[b], 0), 1)
        np.testing.assert_allclose(out[b], expected, atol=1e-12)


class _Fixed:
    """A generator whose random() returns one given number."""

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


def test_measurement_matches_the_single_state_rule() -> None:
    psi = _states(N + 1, 6, 3)
    u = np.random.default_rng(4).random(6)
    mask = np.array([True, True, False, True, True, True])
    out, outcomes = pc.measure_batch(psi, 2, mask, u)
    for b in range(6):
        if mask[b]:
            expected, m = measure_projective_psi(psi[b], 2, N + 1, _Fixed(u[b]))  # type: ignore[arg-type]
            np.testing.assert_allclose(out[b], expected, atol=1e-12)
            assert outcomes[b] == m
        else:
            np.testing.assert_array_equal(out[b], psi[b])
            assert outcomes[b] == 0.0
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)


def test_reference_entropy() -> None:
    psi = _states(N + 1, 3, 5)
    expected = [von_neumann_entropy(psi[b], [0]) for b in range(3)]
    np.testing.assert_allclose(pc.reference_entropy_batch(psi), expected, atol=1e-10)
    # A product state has S_R = 0.
    product = np.zeros((1, 2 ** (N + 1)), dtype=complex)
    product[0, 0] = 1.0
    assert pc.reference_entropy_batch(product)[0] == pytest.approx(0.0, abs=1e-12)


def test_simulate_at_p0_keeps_the_charge_and_does_not_depend_on_batches() -> None:
    cfg = _cfg()
    psi0 = AncillaChargeSectorEncoding.from_config(cfg).state()
    circuit = HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": 3}, np.random.default_rng(6)
    )
    a = pc.simulate(psi0, circuit, 0.0, 50, np.random.default_rng(7), batch_size=50)
    b = pc.simulate(psi0, circuit, 0.0, 50, np.random.default_rng(7), batch_size=50)
    np.testing.assert_array_equal(a["features"], b["features"])
    # Unitary only: R stays maximally entangled, nothing is measured.
    np.testing.assert_allclose(a["S_R"], 1.0, atol=1e-10)
    assert np.all(a["n_measured"] == 0)
    assert set(np.unique(a["Y"])) <= {-1.0, 1.0}
    # On a fixed circuit at p = 0, the state depends only on Y: two rows.
    assert np.unique(np.round(a["features"], 10), axis=0).shape[0] == 2


def test_simulate_at_p1_measures_everything() -> None:
    cfg = _cfg()
    psi0 = AncillaChargeSectorEncoding.from_config(cfg).state()
    circuit = HaarBrickworkCircuit(
        {"N": N, "Nr": 1, "depth": 2}, np.random.default_rng(8)
    )
    out = pc.simulate(psi0, circuit, 1.0, 40, np.random.default_rng(9), batch_size=15)
    assert np.all(out["n_measured"] == 2 * N)
    # The system ends in a Z product state: every <Z_k> is +-1.
    z = out["features"][:, 2 * N : 3 * N]
    np.testing.assert_allclose(np.abs(z), 1.0, atol=1e-10)


def test_effective_rank() -> None:
    assert pc.effective_rank(np.ones(5)) == pytest.approx(5.0)
    assert pc.effective_rank(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert pc.effective_rank(np.zeros(3)) == 0.0


def test_saturation_threshold() -> None:
    grid = [500, 1000, 2000, 4000, 8000]
    # Steps 0.1, 0.02, 0.003, 0.001: converged from 2000.
    mu = [0.6, 0.7, 0.72, 0.723, 0.724]
    assert pc.saturation_threshold(grid, mu, 0.005) == 2000
    # A small step followed by a large one does not count.
    mu = [0.6, 0.601, 0.7, 0.701, 0.702]
    assert pc.saturation_threshold(grid, mu, 0.005) == 2000
    # The last step still moves: no threshold inside the grid.
    assert pc.saturation_threshold(grid, [0.5, 0.5, 0.5, 0.5, 0.6], 0.005) is None


def test_ridge_scores_learns_a_linear_rule() -> None:
    rng = np.random.default_rng(10)
    X = rng.normal(size=(400, 3))
    Y = np.where(X[:, 0] > 0, 1.0, -1.0)
    scores = pc.ridge_scores(X, Y, 0.8, np.random.default_rng(11))
    assert scores["accuracy"] > 0.9
    assert scores["eigenvalues"].shape == (3,)
    assert np.all(np.diff(scores["eigenvalues"]) <= 0)
    assert 2.5 < scores["effective_rank"] <= 3.0
    # Constant features: nothing is fitted.
    flat = pc.ridge_scores(np.ones((40, 2)), Y[:40], 0.8, np.random.default_rng(1))
    assert np.isnan(flat["alpha"])


def test_run_shapes_reproducibility_and_parallel() -> None:
    serial = pc.run(_cfg())
    assert serial["accuracy"].shape == (2, 2, 3)
    assert serial["eigenvalues"].shape == (2, 2, 3, 3 * N + 3 * N * (N - 1) // 2)
    assert serial["S_R"].shape == (2, 2)
    assert np.all((serial["accuracy"] >= 0) & (serial["accuracy"] <= 1))
    # Each circuit has its own generators: two processes give the same numbers.
    parallel = pc.run(_cfg("convergence.n_jobs=2"))
    for key in ("accuracy", "accuracy_train", "effective_rank", "S_R"):
        np.testing.assert_array_equal(serial[key], parallel[key])
    # Circuit 0 does not depend on how many circuits are run.
    alone = pc.run(_cfg("convergence.n_circuits=1"))
    np.testing.assert_array_equal(alone["accuracy"][0], serial["accuracy"][0])
    # More measurements purify R more.
    assert serial["S_R"][:, 1].mean() < serial["S_R"][:, 0].mean()


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("Nr=2", "Nr=1"),
        ("convergence.p_values=[1.5]", "p_values"),
        ("convergence.p_values=[0.1,0.1]", "distinct"),
        ("convergence.per_p_values=[40,20]", "per_p_values"),
        ("convergence.n_rounds=0", "n_rounds"),
    ],
)
def test_run_rejects_bad_settings(override: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pc.run(_cfg(override))


def test_summary_save_and_plots(tmp_path: Path) -> None:
    matplotlib.use("Agg")
    plot = _load(FOLDER / "plot_per_p_convergence.py")
    folders = []
    for n in (3, N):
        cfg = _cfg(f"N={n}")
        folder = tmp_path / f"N{n}"
        summary = pc.save(pc.run(cfg), cfg, folder)
        folders.append(folder)
        assert set(summary["by_p"]) == {"0.1", "0.4"}
        row = summary["by_p"]["0.1"]
        assert len(row["mu"]) == 3 and len(row["delta"]) == 2
        assert json.loads((folder / "summary.json").read_text()) == summary
    paths = plot.plot_all(folders, tmp_path / "figures")
    names = [path.stem for path in paths]
    assert names == [
        "convergence_profile",
        "convergence_steps",
        "convergence_heatmap",
        "convergence_spectrum",
        "per_p_scaling",
    ]
    for path in paths:
        assert path.is_file() and path.with_suffix(".json").is_file()
    table = plot.scaling_table([plot.load_run(f) for f in folders])
    assert table["N"] == [3, N]


def test_projection_law() -> None:
    plot = _load(FOLDER / "plot_per_p_convergence.py")
    table = {
        "N": [6, 8, 10],
        "by_p": {
            "0.05": {"per_p_star": [1000, 4000, 16000]},
            "0.4": {"per_p_star": [500, None, None]},
        },
    }
    law = plot.projection_law(table, targets=(12,))
    assert law["0.05"]["slope"] == pytest.approx(1.0)
    assert law["0.05"]["projection"][12] == pytest.approx(64000)
    assert law["0.4"] is None


def test_script_runs_with_hydra(tmp_path: Path) -> None:
    out = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            str(FOLDER / "per_p_convergence.py"),
            f"N={N}",
            "convergence.n_circuits=1",
            "convergence.p_values=[0.2]",
            "convergence.per_p_values=[20,40]",
            "convergence.n_rounds=1",
            "convergence.n_jobs=1",
            f"hydra.run.dir={out}",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    assert (out / "per_p_convergence.npz").is_file()
    assert (out / "summary.json").is_file()


def test_circuit_subsets_merge_into_the_full_run(tmp_path: Path) -> None:
    full = pc.run(_cfg())
    folders = []
    for ids in ("[1]", "[0]"):
        cfg = _cfg(f"convergence.circuit_ids={ids}")
        folder = tmp_path / f"c{ids[1]}"
        pc.save(pc.run(cfg), cfg, folder)
        folders.append(folder)
    summary = pc.merge(folders, tmp_path / "merged")
    assert summary["n_circuits"] == 2
    with np.load(tmp_path / "merged" / "per_p_convergence.npz") as merged:
        np.testing.assert_array_equal(merged["circuit_ids"], [0, 1])
        for key in ("accuracy", "eigenvalues", "S_R"):
            np.testing.assert_array_equal(merged[key], full[key])
    with pytest.raises(ValueError, match="share circuits"):
        pc.merge([folders[0], folders[0]], tmp_path / "bad")
    other = _cfg("convergence.circuit_ids=[2]", "seed=7")
    pc.save(pc.run(other), other, tmp_path / "other")
    with pytest.raises(ValueError, match="must share"):
        pc.merge([folders[0], tmp_path / "other"], tmp_path / "bad")


def test_run_rejects_repeated_circuit_ids() -> None:
    with pytest.raises(ValueError, match="circuit_ids"):
        pc.run(_cfg("convergence.circuit_ids=[0,0]"))


def test_launcher_runs_two_sizes_and_plots(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    env = {
        **os.environ,
        "OUT_DIR": str(out),
        "N_MAIN": "3",
        "N_VALUES": "3 4",
        "P_VALUES": "0.1 0.3",
        "REGIMES": "0.1 0.3",
        "PER_P_VALUES": "20 40",
        "N_CIRCUITS": "2",
        "N_JOBS": "1",
    }
    subprocess.run(
        ["bash", str(FOLDER / "per_p_convergence.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    for N in (3, 4):
        assert (out / f"N{N}" / "summary.json").is_file()
    assert (out / "per_p_scaling.png").is_file()
