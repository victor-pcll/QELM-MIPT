"""Clifford QELM: purification probes and Pauli-input ridge (issue #337)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM.evaluation.clifford_ridge import (
    CliffordQELM,
    GroupBasis,
    bloch_components,
    degree_one_values,
    draw_circuit,
    draw_inputs,
    effect_groups,
    effective_dimension,
    empirical_spectrum,
    enumerate_group,
    interval,
    interval_entropies,
    local_visible_weights,
    monomial_values,
    population_r2,
    purification,
    range_targets,
    sample_inputs,
    targets_for,
    visible_weights,
    with_mask,
)
from QELM.models.circuits.base import apply_two_qubit_gate
from QELM.models.circuits.clifford import clifford_group_2q
from QELM.models.circuits.stabilizer import periodic_bonds

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
P = [np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)]


def _site(op: np.ndarray, site: int, n: int) -> np.ndarray:
    out = np.ones((1, 1), dtype=complex)
    for q in range(n):
        out = np.kron(out, op if q == site else I2)
    return out


def _pauli(n: int, ops: dict[int, str]) -> np.ndarray:
    out = np.eye(2**n, dtype=complex)
    for site, op in ops.items():
        out = out @ _site(X if op == "X" else Z, site, n)
    return out


def _step_unitary(circuit: CliffordQELM, t: int) -> np.ndarray:
    group = clifford_group_2q()
    n = circuit.N
    u = np.eye(2**n, dtype=complex)
    for g, (a, b) in zip(circuit.gates[t], periodic_bonds(n, t), strict=True):
        # Apply the gate to the columns of u: G @ u.
        cols = [apply_two_qubit_gate(u[:, j], group[int(g)], a, b) for j in range(2**n)]
        u = np.stack(cols, axis=1)
    return u


def _kraus(circuit: CliffordQELM, outcomes: list[int], final: bool) -> np.ndarray:
    """Kraus operator of one record, brute force on the state vector."""
    n = circuit.N
    k = np.eye(2**n, dtype=complex)
    it = iter(outcomes)
    for t in range(circuit.T):
        k = _step_unitary(circuit, t) @ k
        for site in np.flatnonzero(circuit.measured[t]):
            k = _site(P[next(it)], int(site), n) @ k
    if final:
        for site in range(n):
            k = _site(P[next(it)], site, n) @ k
    return k


def _nonzero_record(circuit: CliffordQELM, final: bool, rng) -> np.ndarray:
    n_out = circuit.n_mid + (circuit.N if final else 0)
    while True:
        k = _kraus(circuit, list(rng.integers(2, size=n_out)), final)
        e = k.conj().T @ k
        if np.trace(e).real > 1e-9:
            return e


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_purification_matches_the_effect(seed: int) -> None:
    """Delta = log2 rank(E_m); membership iff Tr[E sigma_a] = +-Tr E."""
    n, t = 4, 3
    rng = np.random.default_rng(seed)
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), 0.35)
    targets = {**targets_for(n), "x0": {0: "X"}, "z1z3": {1: "Z", 3: "Z"}}
    probe = purification(circuit, targets)
    e_mid = _nonzero_record(circuit, False, rng)
    rank = np.linalg.matrix_rank(e_mid, tol=1e-8)
    assert probe.delta == pytest.approx(np.log2(rank))
    for final, member in ((False, probe.member_mid), (True, probe.member_full)):
        e = e_mid if not final else _nonzero_record(circuit, True, rng)
        for name, ops in targets.items():
            ratio = np.trace(e @ _pauli(n, ops)).real / np.trace(e).real
            assert abs(ratio) == pytest.approx(1.0 if member[name] else 0.0, abs=1e-8)


def test_inputs_have_the_moments_of_the_circle() -> None:
    codes = draw_inputs(200_000, 2, np.random.default_rng(3))
    x, z = bloch_components(codes)
    assert x.mean() == pytest.approx(0, abs=1e-2)
    assert (x**2).mean() == pytest.approx(0.5, abs=1e-2)
    assert (z**2).mean() == pytest.approx(0.5, abs=1e-2)
    assert (x * z).mean() == 0
    y = monomial_values(codes, {0: "X", 1: "Z"})
    assert y.var() == pytest.approx(1, abs=2e-2)
    w = degree_one_values(codes, np.random.default_rng(4).normal(size=(2, 2)))
    assert w.var() == pytest.approx(1, abs=2e-2)


def test_sampled_records_have_the_exact_means() -> None:
    """stim shots average to the means of the density-matrix evolution."""
    n, t = 4, 3
    rng = np.random.default_rng(5)
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), 0.4)
    codes = np.array([[0, 3, 2, 1]])
    shots = sample_inputs(circuit, codes, 40_000, seed=1)
    assert shots.shape == (1, 40_000, circuit.n_mid + n)
    states = {0: [1, 0], 1: [0, 1], 2: [1, 1], 3: [1, -1]}
    psi = np.ones(1, dtype=complex)
    for c in codes[0]:
        v = np.array(states[int(c)], dtype=complex)
        psi = np.kron(psi, v / np.linalg.norm(v))
    rho = np.outer(psi, psi.conj())
    means = []
    for step in range(t):
        u = _step_unitary(circuit, step)
        rho = u @ rho @ u.conj().T
        for site in np.flatnonzero(circuit.measured[step]):
            zs = _site(Z, int(site), n)
            means.append(np.trace(zs @ rho).real)
            p0, p1 = _site(P[0], int(site), n), _site(P[1], int(site), n)
            rho = p0 @ rho @ p0 + p1 @ rho @ p1
    means += [np.trace(_site(Z, q, n) @ rho).real for q in range(n)]
    np.testing.assert_allclose(shots[0].mean(axis=0), means, atol=0.02)


def test_empirical_spectrum_is_unbiased_for_noise_only() -> None:
    """Pure noise: the signal estimate is close to zero."""
    rng = np.random.default_rng(6)
    shots = rng.choice(np.array([-1, 1], dtype=np.int8), size=(300, 64, 5))
    signal, noise, means = empirical_spectrum(shots)
    np.testing.assert_allclose(noise, np.eye(5), atol=0.03)
    assert np.abs(signal).max() < 0.01
    assert means.shape == (300, 5)
    cov = np.zeros(5)
    assert population_r2(signal, noise, cov, 16) == 0.0


def _all_effects(circuit: CliffordQELM, final: bool) -> list[np.ndarray]:
    n_out = circuit.n_mid + (circuit.N if final else 0)
    effects = []
    for m in range(2**n_out):
        bits = [(m >> k) & 1 for k in range(n_out)]
        k = _kraus(circuit, bits, final)
        e = k.conj().T @ k
        if np.trace(e).real > 1e-9:
            effects.append(e)
    return effects


_PAULIS = {
    (0, 0): I2,
    (1, 0): X,
    (0, 1): Z,
    (1, 1): np.array([[0, -1j], [1j, 0]]),
}


def _string(bits: np.ndarray, n: int) -> np.ndarray:
    out = np.ones((1, 1), dtype=complex)
    for q in range(n):
        out = np.kron(out, _PAULIS[(int(bits[q]), int(bits[n + q]))])
    return out


def _all_strings(n: int) -> np.ndarray:
    codes = np.arange(4**n)
    digits = (codes[:, None] // 4 ** np.arange(n)[None, :]) % 4
    return np.concatenate([digits % 2, digits // 2], axis=1).astype(bool)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_effect_group_is_the_pauli_support_of_the_records(seed: int) -> None:
    """G = union of the Pauli supports of the E_m, of rank N - Delta or N."""
    n, t = 4, 3
    rng = np.random.default_rng(10 + seed)
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), 0.35)
    groups = effect_groups(circuit)
    strings = _all_strings(n)
    for gens, final in ((groups.mid, False), (groups.full, True)):
        basis = GroupBasis(gens)
        support = np.zeros(len(strings), dtype=bool)
        for e in _all_effects(circuit, final):
            coeff = [abs(np.trace(e @ _string(s, n))) for s in strings]
            support |= np.array(coeff) > 1e-9
        np.testing.assert_array_equal(support, basis.contains(strings))
        assert support.sum() == 2**basis.rank
        elements = enumerate_group(gens)
        assert len({e.tobytes() for e in elements}) == 2**basis.rank
        for start, length in ((0, 1), (1, 2), (3, 2), (0, 3)):
            region = interval(start, length, n)
            outside = [q for q in range(n) if q not in region]
            cols = outside + [n + q for q in outside]
            count = int((~elements[:, cols].any(axis=1)).sum())
            assert count == 2 ** basis.local_rank(region)
    assert GroupBasis(groups.full).rank == n
    delta = purification(circuit, {}).delta
    assert GroupBasis(groups.mid).rank == n - delta


@pytest.mark.parametrize("seed", [0, 1])
def test_infinite_shot_ridge_is_the_projector_on_the_group(seed: int) -> None:
    """Least squares on the exact Born features of all 4^N inputs."""
    n, t = 4, 3
    rng = np.random.default_rng(20 + seed)
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), 0.3)
    digits = (np.arange(4**n)[:, None] // 4 ** np.arange(n)[None, :]) % 4
    codes = digits
    states = {0: [1, 0], 1: [0, 1], 2: [1, 1], 3: [1, -1]}
    rhos = []
    for row in codes:
        psi = np.ones(1, dtype=complex)
        for c in row:
            v = np.array(states[int(c)], dtype=complex)
            psi = np.kron(psi, v / np.linalg.norm(v))
        rhos.append(np.outer(psi, psi.conj()))
    basis = GroupBasis(effect_groups(circuit).full)
    effects = _all_effects(circuit, True)
    features = np.array([[np.trace(e @ r).real for e in effects] for r in rhos])
    design = np.concatenate([np.ones((len(rhos), 1)), features], axis=1)

    def r2(y: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(1 - np.mean((y - design @ coef) ** 2) / y.var())

    targets = range_targets(n, [1, 2])
    for rows in targets.values():
        for row in rows:
            ops = {q: "X" for q in np.flatnonzero(row[:n])}
            ops |= {int(q): "Z" for q in np.flatnonzero(row[n:])}
            y = monomial_values(codes, ops)
            member = bool(basis.contains(row)[0])
            assert r2(y) == pytest.approx(1.0 if member else 0.0, abs=1e-8)
    weights = rng.normal(size=(2, n))
    single = np.concatenate([targets["X"], targets["Z"]])
    inside = basis.contains(single)
    expected = float((weights.ravel()[inside] ** 2).sum() / (weights**2).sum())
    y = degree_one_values(codes, weights)
    assert r2(y) == pytest.approx(expected, abs=1e-8)
    # The covariance spectrum: eigenvalues proportional to 2^{-|g|}.
    centred = features - features.mean(axis=0)
    mu = np.linalg.eigvalsh(centred.T @ centred / len(rhos))
    mu = np.sort(mu[mu > 1e-12 * mu.max()])
    w = np.sort(0.5 ** visible_weights(enumerate_group(basis.rows)))
    np.testing.assert_allclose(mu / mu.max(), w / w.max(), rtol=1e-6)


def test_local_visible_weights_count_each_string_once() -> None:
    n, t = 8, 6
    rng = np.random.default_rng(30)
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), 0.3)
    basis = GroupBasis(effect_groups(circuit).full)
    elements = enumerate_group(basis.rows)
    span = 3
    x, z = elements[:, :n], elements[:, n:]
    support = x | z
    fits = np.zeros(len(elements), dtype=bool)
    for start in range(n):
        outside = [q for q in range(n) if q not in interval(start, span, n)]
        fits |= ~support[:, outside].any(axis=1)
    brute = visible_weights(elements[fits])
    local = local_visible_weights(basis, span)
    np.testing.assert_array_equal(np.sort(local), np.sort(brute))
    d = effective_dimension(np.array([1, 2]), [0.0, 0.25])
    np.testing.assert_allclose(d, [2.0, 0.5 / 0.75 + 0.25 / 0.5])


def test_scripts_run_and_plot(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "experiments" / "clifford_ridge"

    def load(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    script = load("clifford_ridge")
    for n in (8, 12):
        tr = script.run_transition(n, 2 * n, [0.1, 0.3], 6, 0)
        (tmp_path / f"transition_N{n}_T{2 * n}_s0.json").write_text(json.dumps(tr))
        for seed in (0, 1):
            ri = script.run_ridge(n, seed, [0.1, 0.3], 2 * n, 8, (60, 30, 30), 2, True)
            (tmp_path / f"ridge_N{n}_T{2 * n}_seed{seed}.json").write_text(
                json.dumps(ri)
            )
    assert 0 <= tr["per_p"][0]["delta_per_site"][0] <= 1
    assert set(ri["per_p"][0]["targets"]) == {
        "z_mid",
        "x_mid",
        "zz_mid",
        "xx_mid",
        "deg1",
    }
    plots = load("plot_clifford_ridge")
    plots.main([str(tmp_path)])
    assert (tmp_path / "figures" / "transition.png").is_file()
    for n in (8, 12):
        for tag, ps in (("c0", [0.1, 0.2]), ("c1", [0.3])):
            po = script.run_population(n, 2 * n, ps, 3, 0)
            (tmp_path / f"population_N{n}_T{2 * n}_s0_{tag}.json").write_text(
                json.dumps(po)
            )
    row = po["per_p"][0]
    assert row["full"]["rank"]["mean"] == 12
    assert set(row["full"]["member"]) >= {"Z", "X", "ZZ_1", "XX_6", "ZX_4"}
    load("plot_population").main([str(tmp_path)])
    for name in ("order", "eperp", "range", "spectrum", "collapse"):
        assert (tmp_path / "figures" / f"population_{name}.png").is_file()


def test_interval_entropies_match_the_local_ranks() -> None:
    n, t = 10, 8
    rng = np.random.default_rng(40)
    for p in (0.1, 0.3):
        circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
        full = effect_groups(circuit).full
        basis = GroupBasis(full)
        s = interval_entropies(full)
        assert s.shape == (n, n + 1)
        for start in (0, 3, 9):
            for ell in range(n + 1):
                k = basis.local_rank(interval(start, ell, n))
                assert s[start, ell] == ell - k
        np.testing.assert_array_equal(s[:, n], 0)
        # Pure state: S of [0, l) equals S of its complement [l, n).
        for ell in range(1, n):
            assert s[0, ell] == s[ell, n - ell]
