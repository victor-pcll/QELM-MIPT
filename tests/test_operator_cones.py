"""Operator light cones of monitored Haar circuits, OTOC and Krylov (issue #315)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest

from QELM.evaluation.chaos_locality import (
    dense_operator,
    gate_layer_operator,
    ising_chain_hamiltonian,
)
from QELM.evaluation.operator_cones import (
    arnoldi,
    binary_entropy,
    bonds_of_step,
    cone_width,
    conjugate_gate,
    dephasing_factors,
    floquet_heisenberg_channel,
    free_fermion_lanczos,
    front_positions,
    front_velocity,
    krylov_wavefunction,
    liouvillian_lanczos,
    partial_trace_site,
    pauli_weight_chain,
    site_operator,
    trajectory_cones,
)
from QELM.models.circuits.haar import haar_gate

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


# --- A1: Haar-averaged Pauli chain ------------------------------------------


def test_chain_unitary_butterfly_velocity() -> None:
    """At p = 0 the front of rho moves at v_B = 3/5 and the cone saturates at 3/4."""
    rng = np.random.default_rng(0)
    r = pauli_weight_chain(41, 20, 0.0, rng, population=20000)
    np.testing.assert_allclose(r.survival, 1.0)
    right, left = front_positions(r.density, 20, 0.375)
    assert front_velocity(right, 5, 20) == pytest.approx(0.6, abs=0.05)
    assert front_velocity(left, 5, 20) == pytest.approx(0.6, abs=0.05)
    assert r.density[20, 18:23].mean() == pytest.approx(0.75, abs=0.02)
    # nothing outside the Lieb-Robinson cone |x - x0| <= t
    for t in range(21):
        outside = np.abs(np.arange(41) - 20) > t + 1
        assert np.all(r.density[t, outside] == 0.0)


def test_chain_first_step_survival() -> None:
    """After one gate, E sum |c|^2 = ((2 + 2q)^2 - 1) / 15 with q = 1 - p."""
    p = 0.3
    q = 1 - p
    r = pauli_weight_chain(6, 1, p, np.random.default_rng(1), population=200000)
    assert r.survival[1] == pytest.approx(((2 + 2 * q) ** 2 - 1) / 15, rel=0.01)


def test_chain_reproducible_and_validates_p() -> None:
    a = pauli_weight_chain(11, 6, 0.2, np.random.default_rng(5), population=500)
    b = pauli_weight_chain(11, 6, 0.2, np.random.default_rng(5), population=500)
    np.testing.assert_array_equal(a.density, b.density)
    with pytest.raises(ValueError):
        pauli_weight_chain(5, 2, 1.5, np.random.default_rng(0))


def test_cone_width_counts_sites_above_threshold() -> None:
    assert cone_width(np.array([0.1, 0.5, 0.8, np.nan]), 0.4) == 2


# --- dense helpers -------------------------------------------------------


@pytest.mark.parametrize("i,j", [(0, 1), (2, 3), (1, 3)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_conjugate_gate_matches_dense(i: int, j: int, adjoint: bool) -> None:
    N = 4
    rng = np.random.default_rng(2)
    M = rng.normal(size=(2, 16, 16)) + 1j * rng.normal(size=(2, 16, 16))
    g = haar_gate(rng)
    G = dense_operator(gate_layer_operator(N, [(i, j, g)]))
    if adjoint:
        ref = np.stack([G.conj().T @ m @ G for m in M])
    else:
        ref = np.stack([G @ m @ G.conj().T for m in M])
    np.testing.assert_allclose(conjugate_gate(M, g, i, j, N, adjoint), ref, atol=1e-12)


def test_partial_trace_site_of_product() -> None:
    rng = np.random.default_rng(3)
    A, B, C = (rng.normal(size=(2, 2)) for _ in range(3))
    M = np.kron(np.kron(A, B), C)[None]
    np.testing.assert_allclose(
        partial_trace_site(M, 1, 3)[0], np.trace(B) * np.kron(A, C)
    )


def test_binary_entropy() -> None:
    assert binary_entropy(0.0) == pytest.approx(1.0)
    assert binary_entropy(1.0) == pytest.approx(0.0)


# --- A2: trajectories ---------------------------------------------------------


def test_trajectories_unitary_case() -> None:
    """At p = 0: w = 1, S_R = 1, rho is 1 on x0 at t = 0 and stays in the cone."""
    N = 6
    r = trajectory_cones(N, 4, 0.0, 3, np.random.default_rng(4))
    np.testing.assert_allclose(r.weight, 1.0, atol=1e-10)
    np.testing.assert_allclose(r.entropy_R, 1.0, atol=1e-10)
    expected = np.zeros(N)
    expected[3] = 1.0
    np.testing.assert_allclose(r.weighted_density[0], expected, atol=1e-12)
    support = {x for bond in bonds_of_step(N, 1) if 3 in bond for x in bond}
    for x in range(N):
        if x not in support:
            assert abs(r.weighted_density[1, x]) < 1e-12


def test_trajectories_all_measured_kill_quantum_image() -> None:
    """At p = 1 every site is measured after each half-layer.

    The system collapses on a basis state, so every ``M_a`` is proportional
    to ``M_0``: no connected image (``w = 0``) and ``R`` is decoupled, its
    entropy frozen after ``t = 1``. It is not zero: the first gate entangled
    ``R`` with a neighbor of ``x0``, itself purified by the environment of
    the infinite-temperature state, which no measurement reaches.
    """
    r = trajectory_cones(4, 3, 1.0, 2, np.random.default_rng(6))
    np.testing.assert_allclose(r.weight[1:], 0.0, atol=1e-8)
    s = r.per_traj_entropy
    np.testing.assert_allclose(s[:, 1:], s[:, 1:2] * np.ones((1, 3)), atol=1e-10)


def test_trajectories_reproduce_haar_chain_at_p0() -> None:
    """The circuit average of rho at p = 0 is the Haar Markov chain."""
    N, T = 6, 4
    traj = trajectory_cones(N, T, 0.0, 300, np.random.default_rng(7))
    chain = pauli_weight_chain(
        N, T, 0.0, np.random.default_rng(8), x0=3, population=100000
    )
    np.testing.assert_allclose(traj.weighted_density, chain.density, atol=0.04)


def test_trajectories_reproducible() -> None:
    a = trajectory_cones(4, 3, 0.3, 2, np.random.default_rng(9))
    b = trajectory_cones(4, 3, 0.3, 2, np.random.default_rng(9))
    np.testing.assert_array_equal(a.weighted_density, b.weighted_density)


# --- B: Krylov -----------------------------------------------------------------


def test_dephasing_factors_match_channel() -> None:
    p, N = 0.3, 2
    rng = np.random.default_rng(10)
    op = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    out = op.copy()
    for x in range(N):
        P0 = site_operator(N, x, 0) / 2 + site_operator(N, x, 3) / 2
        P1 = np.eye(4) - P0
        out = (1 - p) * out + p * (P0 @ out @ P0 + P1 @ out @ P1)
    np.testing.assert_allclose(dephasing_factors(N, p) * op, out, atol=1e-12)


def test_floquet_channel_unital_and_unitary_at_p0() -> None:
    N = 4
    op = site_operator(N, 2, 3)
    phi0 = floquet_heisenberg_channel(N, 0.0, np.random.default_rng(11))
    out = phi0(op)
    assert np.vdot(out, out).real == pytest.approx(np.vdot(op, op).real)
    assert abs(np.trace(out)) < 1e-10
    phi = floquet_heisenberg_channel(N, 0.4, np.random.default_rng(11))
    assert np.vdot(phi(op), phi(op)).real < np.vdot(op, op).real
    np.testing.assert_allclose(phi(np.eye(16)), 0.0, atol=1e-12)  # traceless part


def test_arnoldi_basis_and_complexity_bound() -> None:
    N = 4
    op = site_operator(N, 2, 3)
    for p in (0.0, 0.3):
        phi = floquet_heisenberg_channel(N, p, np.random.default_rng(12))
        a = arnoldi(phi, op, 20)
        n = a.basis.shape[0]
        gram = a.basis.conj() @ a.basis.T / 16
        np.testing.assert_allclose(gram, np.eye(n), atol=1e-10)
        wf = krylov_wavefunction(phi, op, a.basis, n - 1)
        assert np.all(wf.complexity <= np.arange(n) + 1e-9)
        assert np.max(np.abs(wf.leak)) < 1e-9
        if p == 0.0:
            assert np.all(a.subdiagonal <= 1 + 1e-10)
            np.testing.assert_allclose(wf.norm, 1.0)


def test_lanczos_single_spin() -> None:
    """H = -X on one spin, O = X ... O = Z: b_1 = 2, then the chain closes."""
    H = ising_chain_hamiltonian([], h_x=1.0, h_z=0.0)
    b = liouvillian_lanczos(H, site_operator(1, 0, 3), 5)
    np.testing.assert_allclose(b, [2.0])


def test_free_fermion_lanczos_matches_dense() -> None:
    N = 5
    H = ising_chain_hamiltonian(np.ones(N - 1), h_x=1.05, h_z=0.0)
    dense = liouvillian_lanczos(H, site_operator(N, 2, 1), 12)
    exact = free_fermion_lanczos(N, 1.0, 1.05, 2, 12)
    np.testing.assert_allclose(dense[:12], exact[:12], rtol=1e-8)


# --- scripts ---------------------------------------------------------------------


def _load(name: str) -> ModuleType:
    path = ROOT / "experiments" / "operator_cones" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scripts_run_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _load("operator_cones")
    plot = _load("plot_operator_cones")
    p = "0,0.05,0.17,0.35,0.65"
    commands = [
        ["chain", "--N", "11", "--T", "6", "--population", "200", "--p", p],
        [
            "traj",
            "--N",
            "4",
            "--T",
            "3",
            "--traj",
            "2",
            "--chunk",
            "1",
            "--jobs",
            "1",
            "--p",
            p,
        ],
        [
            "traj",
            "--N",
            "6",
            "--T",
            "3",
            "--traj",
            "2",
            "--chunk",
            "2",
            "--jobs",
            "1",
            "--p",
            p,
        ],
        ["krylov", "--N", "4", "--n-max", "6", "--seeds", "2", "--p", p],
        ["parker", "--N", "5", "--n-max", "6"],
        ["single", "--N", "3", "--seeds", "2", "--rank", "--p", p],
        ["single", "--N", "4", "--seeds", "2", "--p", p],
    ]
    for cmd in commands:
        monkeypatch.setattr(sys, "argv", ["x", *cmd, "--out", str(tmp_path)])
        run.main()
    data = json.loads((tmp_path / "traj_N6.json").read_text())
    assert data["runs"]["0"]["n_traj"] == 2
    monkeypatch.setattr(sys, "argv", ["x", "--data", str(tmp_path)])
    plot.main()
    for name in (
        "otoc_chain_cones",
        "otoc_trajectory_cones",
        "otoc_trajectory_purification",
        "krylov_cones",
        "krylov_coefficients",
        "single_operator_cones",
        "single_operator_growth",
    ):
        assert (tmp_path / f"{name}.png").exists()


def test_trajectory_system_entropy() -> None:
    """S(M_0) is N bits at p = 0 and decreases (purification) with p > 0."""
    N = 4
    r = trajectory_cones(N, 3, 0.0, 2, np.random.default_rng(13))
    np.testing.assert_allclose(r.per_traj_system_entropy, N, atol=1e-8)
    r = trajectory_cones(N, 6, 0.5, 4, np.random.default_rng(14))
    s = r.per_traj_system_entropy
    assert s[:, -1].mean() < N


# --- C: one operator in the circuits of the effect rank (PR #314) ----------


def test_single_operator_span_unitary_and_bounded_by_effect_rank() -> None:
    from QELM.evaluation.effect_rank import draw_circuit, effect_rank
    from QELM.evaluation.operator_cones import single_operator_span

    n = 4
    circuit = draw_circuit("haar", n, n, np.random.default_rng(20))
    u = np.random.default_rng(21).random((n, n))
    r0 = single_operator_span(circuit, u < 0.0, 2)
    np.testing.assert_array_equal(r0.dims, 1)
    np.testing.assert_allclose(r0.averaged_norm, 1.0)
    np.testing.assert_allclose(r0.density, r0.averaged_density, atol=1e-12)
    for p in (0.2, 0.5, 1.0):
        r = single_operator_span(circuit, u < p, 2)
        assert r.dims[-1] <= effect_rank(circuit, u < p).dense
        assert np.all(np.diff(r.averaged_norm) <= 1e-12)
        assert np.all(np.nan_to_num(r.density) <= 1 + 1e-12)
