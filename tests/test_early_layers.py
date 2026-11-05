"""Early-layers hypothesis of the linear QELM on M (issue #288)."""

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
from QELM.evaluation.early_layers import (
    averaged_states,
    dephase,
    haar_mass_rate,
    haar_pauli_chain,
    pauli_type_mass,
    reference_helstrom,
    split_reference_state,
    suffix_bounds,
    trace_distance,
    z_expectations,
)
from QELM.models.circuits.haar import HaarBrickworkCircuit, U1BrickworkCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "early_layers"
SCRIPT = FOLDER / "early_layers.py"
BASH_SCRIPT = FOLDER / "early_layers.sh"
PLOT_SCRIPT = FOLDER / "plot_early_layers.py"
STORY_SCRIPT = FOLDER / "plot_story.py"
N = 4
P_VALUES = "[0.3,1.0]"
PER_P = 60


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*extra: str) -> dict:
    cfg = load_config(
        [
            f"N={N}",
            f"early_task.p_values={P_VALUES}",
            f"early_task.samples_per_p={PER_P}",
            *extra,
        ],
        config_name="early_layers",
    )
    return _load(SCRIPT).run(cfg)


def _basis(n: int, index: int) -> np.ndarray:
    v = np.zeros(2**n, dtype=complex)
    v[index] = 1.0
    return v


def _circuit(cls: type, n: int, depth: int, seed: int) -> HaarBrickworkCircuit:
    return cls({"N": n, "Nr": 1, "depth": depth}, np.random.default_rng(seed))


# --- split, dephasing, trace distance -------------------------------------


def test_split_reference_state() -> None:
    a, b = _basis(2, 0), _basis(2, 3)
    psi = np.concatenate([a, b]) / np.sqrt(2.0)
    a2, b2 = split_reference_state(psi)
    np.testing.assert_allclose(a2, a)
    np.testing.assert_allclose(b2, b)
    with pytest.raises(ValueError, match="1/2"):
        split_reference_state(np.concatenate([a, 0 * b]))


def test_dephase_is_the_average_of_a_measurement() -> None:
    rng = np.random.default_rng(0)
    G = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    p = 0.3
    for site in range(3):
        bit = (np.arange(8) >> (2 - site)) & 1
        P0, P1 = np.diag(bit == 0).astype(float), np.diag(bit == 1).astype(float)
        expected = (1 - p) * rho + p * (P0 @ rho @ P0 + P1 @ rho @ P1)
        np.testing.assert_allclose(dephase(rho, site, p), expected, atol=1e-12)
    np.testing.assert_allclose(dephase(rho, 1, 0.0), rho)


def test_trace_distance_and_z() -> None:
    a, b = _basis(2, 0), _basis(2, 1)
    rho, sigma = np.outer(a, a), np.outer(b, b)
    assert trace_distance(rho, sigma) == pytest.approx(1.0)
    assert trace_distance(rho, rho) == pytest.approx(0.0)
    # |01>: Z_0 = +1, Z_1 = -1.
    np.testing.assert_allclose(z_expectations(sigma), [1.0, -1.0])


# --- the exact bounds ------------------------------------------------------


def test_averaged_states_at_p0_are_the_unitary_evolution() -> None:
    circuit = _circuit(HaarBrickworkCircuit, 3, 2, 1)
    a, b = _basis(3, 0), _basis(3, 2)
    rho_p, _ = averaged_states(circuit.layers, a, b, 0.0)
    psi = np.concatenate([a, np.zeros(8)])  # R in |0>
    for t in range(2):
        psi = circuit.apply_layer(psi, t)
        chain = psi[:8]
        np.testing.assert_allclose(rho_p[t], np.outer(chain, chain.conj()), atol=1e-12)


def test_averaged_states_match_sampled_trajectories() -> None:
    """``E[m_{t,i} Y]`` of the exact channel against Monte Carlo."""
    early = _load(SCRIPT)
    circuit = _circuit(HaarBrickworkCircuit, 3, 3, 4)
    a, b = _basis(3, 0), _basis(3, 2)
    psi0 = np.concatenate([a, b]) / np.sqrt(2.0)
    p = 0.6
    exact = suffix_bounds(circuit.layers, a, b, p)["c"]
    rng = np.random.default_rng(5)
    n = 6000
    acc = np.zeros_like(exact)
    for _ in range(n):
        M, _, _, Y = early.trajectory(psi0, circuit, p, rng)
        acc += M * Y
    sampled = acc / n
    se = np.sqrt(p) / np.sqrt(n)
    assert np.max(np.abs(sampled - exact)) < 5 * se


def test_trace_distance_never_grows_and_bounds_the_signal() -> None:
    circuit = _circuit(HaarBrickworkCircuit, 4, 6, 2)
    a, b = _basis(4, 0), _basis(4, 0b0010)
    for p in (0.0, 0.2, 1.0):
        out = suffix_bounds(circuit.layers, a, b, p)
        D = out["D"]
        assert D[0] == pytest.approx(1.0)
        assert np.all(np.diff(D) <= 1e-10)
        assert np.all(np.abs(out["c"]) <= p * D[:, None] + 1e-12)
    assert np.allclose(suffix_bounds(circuit.layers, a, b, 0.0)["D"], 1.0)


def test_u1_gates_conserve_the_charge_difference() -> None:
    circuit = _circuit(U1BrickworkCircuit, 4, 5, 3)
    a, b = _basis(4, 0), _basis(4, 0b0010)
    out = suffix_bounds(circuit.layers, a, b, 0.5)
    np.testing.assert_allclose(out["charge"], -1.0, atol=1e-12)
    # Sum of the linear signal of a round: p / 2 * Tr(sum_i Z_i Delta) = p.
    np.testing.assert_allclose(out["c"].sum(axis=1), 0.5, atol=1e-12)
    assert np.all(out["D"] >= 1.0 / (2 * 4) - 1e-12)


def test_reference_helstrom_known_states() -> None:
    a, b = _basis(2, 0), _basis(2, 3)
    r_z, H = reference_helstrom(np.concatenate([a, b]) / np.sqrt(2))
    assert (r_z, H) == pytest.approx((0.0, 1.0))
    r_z, H = reference_helstrom(np.concatenate([a, a]) / np.sqrt(2))
    # A square root of a rounding error: H is only zero to about 1e-8.
    assert (r_z, H) == pytest.approx((0.0, 0.0), abs=1e-6)
    r_z, H = reference_helstrom(np.concatenate([a, 0 * a]))
    assert (r_z, H) == pytest.approx((1.0, 1.0))
    rng = np.random.default_rng(1)
    for _ in range(20):
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        r_z, H = reference_helstrom(psi)
        # H is the trace norm of |A><A| - |B><B|.
        A, B = psi[:4], psi[4:]
        op = np.outer(A, A.conj()) - np.outer(B, B.conj())
        assert H == pytest.approx(np.abs(np.linalg.eigvalsh(op)).sum())
        assert H >= abs(r_z) - 1e-12


# --- the Haar average ------------------------------------------------------


def test_pauli_type_mass() -> None:
    # Z on one qubit: a_Z = Tr(Z Z) = 2.
    mass = pauli_type_mass(np.diag([1.0, -1.0]).astype(complex))
    np.testing.assert_allclose(mass, [0.0, 4.0, 0.0])
    # X has the type XY.
    mass = pauli_type_mass(np.array([[0, 1], [1, 0]], dtype=complex))
    np.testing.assert_allclose(mass, [0.0, 0.0, 4.0])
    # Total mass = 2**N Tr(Delta^2) = 2**(N + 1) for two orthogonal states.
    a, b = _basis(3, 0), _basis(3, 5)
    delta = np.outer(a, a) - np.outer(b, b)
    assert pauli_type_mass(delta).sum() == pytest.approx(2**4)


def test_haar_chain_matches_sampled_circuits() -> None:
    n, T, p = 4, 3, 0.3
    a, b = _basis(n, 0), _basis(n, 0b0010)
    rng = np.random.default_rng(11)
    K = 600
    mu = np.zeros(T)
    z2 = np.zeros((T, n))
    for _ in range(K):
        circuit = HaarBrickworkCircuit({"N": n, "Nr": 1, "depth": T}, rng)
        rho_p, rho_m = averaged_states(circuit.layers, a, b, p)
        d = rho_p - rho_m
        mu += [2**n * np.real(np.trace(x @ x)) for x in d]
        z2 += (z_expectations(rho_p) - z_expectations(rho_m)) ** 2
    rounds = [[(i - 1, j - 1) for i, j, _ in layer] for layer in circuit.layers]
    chain = haar_pauli_chain(
        pauli_type_mass(np.outer(a, a) - np.outer(b, b)), rounds, p
    )
    np.testing.assert_allclose(mu / K, chain["mu"], rtol=0.1)
    np.testing.assert_allclose(z2 / K, chain["single_z"], rtol=0.25, atol=3e-3)


def test_haar_mass_contracts_at_least_by_g() -> None:
    n, T = 6, 6
    a, b = _basis(n, 0), _basis(n, 0b000100)
    circuit = _circuit(HaarBrickworkCircuit, n, T, 0)
    rounds = [[(i - 1, j - 1) for i, j, _ in layer] for layer in circuit.layers]
    mass = pauli_type_mass(np.outer(a, a) - np.outer(b, b))
    unitary = haar_pauli_chain(mass, rounds, 0.0)["mu"]
    np.testing.assert_allclose(unitary, 2 ** (n + 1))
    for p in (0.1, 0.5, 1.0):
        mu = haar_pauli_chain(mass, rounds, p)["mu"]
        assert np.all(mu[1:] <= haar_mass_rate(p) * mu[:-1] + 1e-9)
    assert haar_mass_rate(0.0) == 1.0
    assert haar_mass_rate(1.0) == pytest.approx(1.0 / 3.0)


# --- the experiment --------------------------------------------------------


def test_run_shapes_and_shares() -> None:
    results = _run()
    m = results["metrics"]
    T = 2 * N
    assert m["T"] == T and len(m["per_p"]) == 2
    assert results["M"].shape == (2 * PER_P, T, N)
    assert results["r_z"].shape == (2 * PER_P, T + 1)
    for entry in m["per_p"]:
        for name in ("weight_l2", "share", "prefix", "suffix", "ablate_late", "D"):
            assert len(entry[name]) == T
        assert len(entry["bayes_prefix"]) == T + 1
        assert sum(entry["share"]) == pytest.approx(1.0)
        assert np.all(np.abs(entry["share"]) <= np.asarray(entry["share_bound"]) + 1e-9)
        # The full model with nothing blanked is the full model.
        assert entry["ablate_late"][-1] == pytest.approx(entry["accuracy"])
        assert entry["ablate_early"][0] == pytest.approx(entry["accuracy"])
        # Bayes on more rounds is never worse, the Helstrom bound never better.
        assert np.all(np.diff(entry["bayes_prefix"]) >= -0.02)
        assert np.all(
            np.asarray(entry["helstrom"]) >= np.asarray(entry["bayes_prefix"]) - 1e-12
        )
        assert "haar_mu" in entry


def test_run_is_reproducible_and_u1_control() -> None:
    first = _run("seed=3")["metrics"]
    again = _run("seed=3")["metrics"]
    assert first == again
    u1 = _run("early_task.gates=u1", "early_task.encoding=charge")["metrics"]
    assert u1["gates"] == "u1"
    for entry in u1["per_p"]:
        assert "haar_mu" not in entry
        np.testing.assert_allclose(np.abs(entry["charge"]), 1.0, atol=1e-10)


def test_run_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError, match="gates"):
        _run("early_task.gates=clifford")
    with pytest.raises(ValueError, match="Nr=1"):
        _run("Nr=0")
    with pytest.raises(ValueError, match="p_values"):
        _run("early_task.p_values=[1.5]")


def test_launcher_and_figures(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path / "sweep"),
        "N_VALUES": "4",
        "SEEDS": "1 2",
        "CONFIGS": "haar:bell u1:bell",
        "PER_P": "30",
        "P_VALUES": "0.4 1",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, cwd=REPO_ROOT)
    sweep = tmp_path / "sweep"
    done = sweep / "haar_bell_N4_seed1" / "metrics.json"
    assert done.exists()
    stamp = done.stat().st_mtime_ns
    subprocess.run(
        ["bash", str(BASH_SCRIPT)],
        check=True,
        env={**env, "SKIP_DONE": "1"},
        cwd=REPO_ROOT,
    )
    assert done.stat().st_mtime_ns == stamp
    fig = tmp_path / "fig"
    subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), str(sweep), "--out", str(fig)],
        check=True,
    )
    for key in ("haar_bell_N4", "u1_bell_N4"):
        for kind in ("weights", "windows", "bounds"):
            assert (fig / f"{kind}_{key}.png").exists()
    assert (fig / "early_share_vs_p.png").exists()
    summary = json.loads((fig / "summary.json").read_text())
    assert summary["haar_bell_N4"]["n_circuits"] == 2
    # The two summary figures, Haar against U(1).
    story = _load(STORY_SCRIPT)
    renamed = {k.replace("N4", "N8"): v for k, v in summary.items()}
    story.plot_information(renamed, [0.4, 1.0], tmp_path / "story")
    story.plot_depth(renamed, [0.4], tmp_path / "story")
    assert (tmp_path / "story" / "story_information.png").exists()
    assert (tmp_path / "story" / "story_depth.png").exists()
