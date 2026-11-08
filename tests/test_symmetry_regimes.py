"""Gate families and targets of nonlinear_M, and the regime figures (issue #311)."""

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
from QELM.models.circuits.haar import U1BrickworkCircuit
from QELM.physics.operators import PAULIS, get_site_operator

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "nonlinear_M" / "nonlinear_M.py"
FOLDER = REPO_ROOT / "experiments" / "symmetry_regimes"
BASH_SCRIPT = FOLDER / "symmetry_regimes.sh"
PLOT_SCRIPT = FOLDER / "plot_symmetry_regimes.py"
N = 4


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NL = _load(SCRIPT)


def _run(*extra: str) -> dict:
    cfg = load_config(
        [
            f"N={N}",
            "nonlinear_task.p_values=[0.0,0.3,1.0]",
            "nonlinear_task.samples_per_p=60",
            "nonlinear_task.modes=[raw]",
            "nonlinear_task.mlp_reference=false",
            *extra,
        ],
        config_name="nonlinear_M",
    )
    return NL.run(cfg)


def test_reference_bloch_x_against_dense_operator() -> None:
    rng = np.random.default_rng(11)
    n = N + 1
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    X0 = get_site_operator(PAULIS["X"], 0, n)
    assert NL.reference_bloch(psi, "x") == pytest.approx(
        float(np.real(psi.conj() @ X0 @ psi))
    )
    assert NL.reference_bloch(psi, "z") == NL.reference_z(psi)
    with pytest.raises(ValueError):
        NL.reference_bloch(psi, "y")


def test_every_family_keeps_the_norm_and_its_charge() -> None:
    rng = np.random.default_rng(12)
    params = {"N": N, "Nr": 1, "depth": 2}
    psi0 = AncillaBellPairEncoding({"N": N}).state()
    # Total Z of the system, diagonal in the computational basis.
    bits = (np.arange(2 ** (N + 1))[:, None] >> np.arange(N)[::-1]) & 1
    charge = bits.sum(axis=1)
    for name in NL.CIRCUITS:
        circuit = NL.build_circuit(name, params, rng)
        psi = psi0
        for t in range(circuit.depth):
            psi = circuit.apply_layer(psi, t)
        assert np.linalg.norm(psi) == pytest.approx(1.0)
        if name in ("u1", "dipole", "diagonal"):
            # The charge distribution of the initial state is kept.
            for q in range(N + 1):
                assert np.sum(np.abs(psi[charge == q]) ** 2) == pytest.approx(
                    np.sum(np.abs(psi0[charge == q]) ** 2)
                )
    with pytest.raises(ValueError):
        NL.build_circuit("ising", params, rng)


def test_default_family_reproduces_the_haar_run() -> None:
    assert (
        _run("seed=3")["metrics"]
        == _run("seed=3", "nonlinear_task.circuit=haar")["metrics"]
    )


def test_diagonal_gates_keep_the_label_in_the_state_and_the_record() -> None:
    # Every Z_i is conserved: the middle site keeps the Z label of R. The
    # state <Z_k> reads it at every p, and Bayes too once the site is measured.
    out = _run("seed=4", "nonlinear_task.circuit=diagonal")
    m = out["metrics"]
    for key in ("p0", "p0.3", "p1"):
        assert m[f"accuracy_ridge_quantum_{key}"] == 1.0
    assert m["accuracy_bayes_p1"] == 1.0
    assert m["accuracy_ridge_raw_p1"] == 1.0


def test_x_label_is_out_of_reach_with_a_conserved_charge() -> None:
    # Bell encoding: the two branches of R hold charges 0 and 1, which
    # U(1) gates and Z measurements never mix, so <X_R> = 0 given any record.
    out = _run("seed=5", "nonlinear_task.circuit=u1", "nonlinear_task.target=x")
    assert np.allclose(out["r_z"], 0.0, atol=1e-12)
    assert out["metrics"]["target_x"] == 1.0
    for key in ("p0", "p0.3", "p1"):
        assert out["metrics"][f"accuracy_bayes_expected_{key}"] == pytest.approx(0.5)


def test_x_trajectory_without_gates_reads_the_x_state() -> None:
    # R in |+> and no entanglement: <X_R> = 1 and the X outcome is +1.
    circuit = U1BrickworkCircuit({"N": N, "Nr": 1, "depth": 1})
    circuit.layers = [[]]
    psi0 = np.zeros(2 ** (N + 1), dtype=complex)
    psi0[0] = psi0[2**N] = 1 / np.sqrt(2)
    rng = np.random.default_rng(6)
    _, r, S_R, Y, psi = NL.trajectory(psi0, circuit, 1.0, rng, axis="x")
    assert r == pytest.approx(1.0)
    assert S_R == pytest.approx(0.0, abs=1e-9)
    assert Y == 1.0
    assert np.linalg.norm(psi) == pytest.approx(1.0)


def test_run_refuses_an_unknown_target_or_family() -> None:
    with pytest.raises(ValueError):
        _run("nonlinear_task.target=y")
    with pytest.raises(ValueError):
        _run("nonlinear_task.circuit=ising")


def test_sweep_and_figures(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path / "sweep"),
        "N": "4",
        "DEPTH": "N",
        "CIRCUITS": "haar u1",
        "ENCODINGS": "bell ghz",
        "TARGETS": "z",
        "SEEDS": "1 2",
        "PER_P": "30",
        "P_VALUES": "0 0.05 0.2 0.5 1",
        "MAX_JOBS": "2",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=REPO_ROOT)
    sweep = tmp_path / "sweep"
    assert (sweep / "u1_ghz_z" / "N4_seed2" / "metrics.json").exists()
    subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), str(sweep), "--out", str(tmp_path / "fig")],
        check=True,
    )
    fig = tmp_path / "fig"
    for name in (
        "qelm_by_symmetry_bell_z.png",
        "full_overlay_ghz_z.png",
        "regime_map_z.png",
    ):
        assert (fig / name).stat().st_size > 0
    summary = json.loads((fig / "summary.json").read_text())
    assert summary["tasks"]["haar_bell_z"]["n_circuits"] == 2
    table = summary["regimes"]["z"]
    assert table["circuits"] == ["haar", "u1"]
    assert table["encodings"] == ["bell", "ghz"]
    assert np.asarray(table["0 < p <= 0.05"]["ridge_raw"]).shape == (2, 2)


def test_two_regimes_crossing_and_figure(tmp_path: Path) -> None:
    module = _load(FOLDER / "plot_two_regimes.py")
    entry = {
        "p": [0.0, 0.1, 0.2, 0.3],
        "accuracy_ridge_quantum": [1.0, 0.8, 0.5, 0.5],
        "accuracy_ridge_raw": [0.5, 0.6, 0.7, 0.8],
    }
    # State - M: 0.2 at p = 0.1, -0.2 at p = 0.2, so the crossing is 0.15.
    assert module.crossing(entry) == pytest.approx(0.15)
    with pytest.raises(ValueError):
        module.crossing({**entry, "accuracy_ridge_raw": [0.4] * 4})
    paths = module.main(["--out", str(tmp_path)])
    assert [p.name for p in paths] == [
        "two_regimes_haar_bell_z.png",
        "two_regimes_haar_bell_z_r2.png",
    ]
    for path in paths:
        assert path.stat().st_size > 0
        data = json.loads(path.with_suffix(".json").read_text())
        assert 0.0 < data["p_x"] < 0.1
