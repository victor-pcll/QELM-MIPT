"""Delta, S_2(N/2), I3 without final readout from the Fourier weights (#378)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
import stim

from QELM.evaluation.clifford_ridge import (
    CliffordQELM,
    _reference_group,
    body_program,
    draw_circuit,
    enumerate_group,
)
from QELM.evaluation.haar_mipt import evolve_kraus, haar_record
from QELM.evaluation.no_readout import (
    clifford_entropies,
    clifford_sample,
    entropies_from_weights,
    haar_sample,
    interval,
    observables,
    region_weights,
)
from QELM.models.circuits.stabilizer import stabilizer_entropy, stabilizer_matrix

PAULI = {
    (0, 0): np.eye(2),
    (1, 0): np.array([[0, 1], [1, 0]]),
    (0, 1): np.diag([1, -1]),
    (1, 1): np.array([[0, 1], [-1, 0]]),  # X Z, Y up to a phase
}


def _pauli(x: Sequence[int], z: Sequence[int]) -> np.ndarray:
    out = np.eye(1)
    for a, b in zip(x, z, strict=True):
        out = np.kron(out, PAULI[(int(a), int(b))])
    return out


def _purity_dense(rho: np.ndarray, n: int, region: tuple[int, ...]) -> float:
    """Tr rho_A^2 by an explicit partial trace (qubit 0 most significant)."""
    t = rho.reshape([2] * (2 * n))
    rest = [s for s in range(n) if s not in region]
    order = list(region) + rest + [n + s for s in region] + [n + s for s in rest]
    t = t.transpose(order)
    da, db = 2 ** len(region), 2 ** len(rest)
    t = t.reshape(da, db, da, db)
    red = np.einsum("ibjb->ij", t)
    return float(np.vdot(red, red).real)


def _random_rho(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(2**n, 3)) + 1j * rng.normal(size=(2**n, 3))
    rho = a @ a.conj().T
    return np.asarray(rho / np.trace(rho).real)


def test_region_weights_match_a_dense_partial_trace() -> None:
    n = 4
    rho = _random_rho(n, np.random.default_rng(0))
    w = region_weights(rho, n)
    assert w[0] == pytest.approx(1.0)
    for size in range(1, n + 1):
        for region in itertools.combinations(range(n), size):
            mask = sum(1 << (n - 1 - s) for s in region)
            # 1 + W_A = 2^|A| Tr rho_A^2
            target = 2**size * _purity_dense(rho, n, region)
            assert w[mask] == pytest.approx(target, rel=1e-10)


def test_region_weights_are_the_sum_of_the_pauli_coefficients() -> None:
    n = 3
    rho = _random_rho(n, np.random.default_rng(1))
    w = region_weights(rho, n)
    for region in [(0,), (1, 2), (0, 2), (0, 1, 2)]:
        total = 0.0
        for x in itertools.product([0, 1], repeat=n):
            for z in itertools.product([0, 1], repeat=n):
                if any((x[s] or z[s]) and s not in region for s in range(n)):
                    continue
                total += abs(np.trace(_pauli(list(x), list(z)) @ rho)) ** 2
        mask = sum(1 << (n - 1 - s) for s in region)
        assert w[mask] == pytest.approx(total, rel=1e-10)


def _clifford_state(n: int, p: float, t: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    circuit = CliffordQELM(
        gates=draw_circuit(n, t, rng), measured=rng.random((t, n)) < p
    )
    sim = stim.TableauSimulator(seed=seed)
    sim.do_circuit(
        stim.Circuit("\n".join(f"H {n + i}\nCX {n + i} {i}" for i in range(n)))
    )
    sim.do_circuit(stim.Circuit(body_program(circuit, final=False)))
    stab = stabilizer_matrix(sim)
    return stab, _reference_group(stab, n)


@pytest.mark.parametrize("n", [4, 8])
def test_clifford_ranks_match_the_pure_state_entropies(n: int) -> None:
    for seed in range(5):
        stab, group = _clifford_state(n, 0.15, 2 * n, seed)
        table = clifford_entropies(group, n)
        for region, s in table.items():
            refs = [n + a for a in region]
            assert s == stabilizer_entropy(stab, refs)


def test_clifford_ranks_match_the_dense_fourier_weights() -> None:
    n = 4
    for seed in range(4):
        _, group = _clifford_state(n, 0.2, 2 * n, seed)
        rho = sum(_pauli(g[:n], g[n:]) for g in enumerate_group(group).astype(int))
        rho = np.asarray(rho, dtype=complex) / 2**n
        weights = region_weights(rho, n)
        dense = entropies_from_weights(weights, n)
        table = clifford_entropies(group, n)
        for region, s in table.items():
            assert dense(region) == pytest.approx(s, abs=1e-9)


def test_observables_of_the_maximally_mixed_state() -> None:
    n = 8
    obs = observables(n, lambda region: float(len(region)))
    assert obs.tolist() == [n, n / 2, 0.0, 1.0]
    assert np.isnan(observables(6, lambda region: float(len(region)))[2])


def test_interval_wraps_around() -> None:
    assert interval(6, 4, 8) == (0, 1, 6, 7)


def test_haar_delta_matches_haar_record() -> None:
    n, p, cps = 6, 0.15, (6, 12)
    obs = haar_sample(n, p, cps, np.random.default_rng(3))
    rec = haar_record(n, p, cps, np.random.default_rng(3))
    assert obs[:, 0] == pytest.approx(rec.delta, abs=1e-9)


def test_haar_regions_match_a_dense_partial_trace() -> None:
    n, p = 4, 0.2
    obs = haar_sample(n, p, (8,), np.random.default_rng(5))
    _, k, _ = next(evolve_kraus(n, p, (8,), np.random.default_rng(5)))
    rho = k.conj().T @ k
    rho = rho / np.trace(rho).real
    s2 = {
        r: len(r) - np.log2(2 ** len(r) * _purity_dense(rho, n, r))
        for r in [interval(s, 2, n) for s in range(n)] + [(s,) for s in range(n)]
    }
    assert obs[0, 1] == pytest.approx(
        np.mean([s2[interval(s, 2, n)] for s in range(n)])
    )
    assert obs[0, 3] == pytest.approx(np.mean([s2[(s,)] for s in range(n)]))


@pytest.mark.parametrize("model", ["haar", "clifford"])
def test_no_measurement_leaves_the_maximally_mixed_state(model: str) -> None:
    n = 8 if model == "clifford" else 4
    rng = np.random.default_rng(0)
    sample = haar_sample if model == "haar" else clifford_sample
    obs = sample(n, 0.0, (n, 2 * n), rng)
    assert obs == pytest.approx(np.array([[n, n / 2, 0.0, 1.0]] * 2), abs=1e-9)


@pytest.mark.parametrize("model", ["haar", "clifford"])
def test_measuring_every_site_purifies(model: str) -> None:
    n = 8 if model == "clifford" else 4
    sample = haar_sample if model == "haar" else clifford_sample
    obs = sample(n, 1.0, (n,), np.random.default_rng(0))
    assert obs[0, 0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("model", ["haar", "clifford"])
def test_samples_are_reproducible_with_the_seed(model: str) -> None:
    n = 8 if model == "clifford" else 4
    sample = haar_sample if model == "haar" else clifford_sample
    a = sample(n, 0.15, (n, 2 * n), np.random.default_rng(7))
    b = sample(n, 0.15, (n, 2 * n), np.random.default_rng(7))
    assert np.array_equal(a, b, equal_nan=True)


def _load_script() -> object:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments/pc_nu_no_readout/run.py"
    spec = importlib.util.spec_from_file_location("pc_nu_no_readout_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the worker processes pickle _chunk by name
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model,n", [("haar", 4), ("clifford", 8)])
def test_run_script_writes_samples_and_resumes(
    model: str, n: int, tmp_path: object, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    from pathlib import Path

    out = Path(str(tmp_path))
    script = _load_script()
    argv = ["--model", model, "--n", str(n), "--samples", "6", "--p", "0.1", "0.2",
            "--workers", "1", "--out", str(out)]  # fmt: skip
    script.main(argv)  # type: ignore[attr-defined]
    data = json.loads((out / f"{model}_N{n}.json").read_text())
    assert [r["p"] for r in data["per_p"]] == [0.1, 0.2]
    row = data["per_p"][0]
    assert len(row["samples_delta_T4N"]) == 6
    assert row["delta_T4N"][0] == pytest.approx(np.mean(row["samples_delta_T4N"]))
    assert "i3_T4N" in row
    capsys.readouterr()
    script.main(argv)  # type: ignore[attr-defined]
    assert capsys.readouterr().out.count("already written") == 2
