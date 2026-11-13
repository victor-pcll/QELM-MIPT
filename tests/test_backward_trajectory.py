"""Tests of the backward trajectories of a monitored brickwork (issue #352)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from QELM.evaluation.backward_trajectory import (
    backward_trajectory,
    cut_profile,
    entropy,
    tripartite_information,
)
from QELM.evaluation.effect_rank import brick_pairs
from QELM.models.circuits.haar import haar_gate


def _embed(gate: np.ndarray, site: int, n: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2**site), gate), np.eye(2 ** (n - site - 2)))


def _projector(site: int, bit: int, n: int) -> np.ndarray:
    p = np.zeros((2, 2))
    p[bit, bit] = 1.0
    return np.kron(np.kron(np.eye(2**site), p), np.eye(2 ** (n - site - 1)))


def _kraus(gates, masks, mid: dict, final) -> np.ndarray:
    """Forward ``K_m = |m_f><m_f| prod_t P_t U_t`` as a full matrix."""
    depth, n = masks.shape
    k = np.eye(2**n, dtype=complex)
    for t in range(depth):
        for left, g in zip(brick_pairs(n), gates[t], strict=True):
            k = _embed(g, left, n) @ k
        for site in np.flatnonzero(masks[t]):
            k = _projector(int(site), mid[(t, int(site))], n) @ k
    for site, bit in enumerate(final):
        k = _projector(site, int(bit), n) @ k
    return k


def _circuit(n: int, depth: int, p: float, seed: int):
    rng = np.random.default_rng(seed)
    gates = [[haar_gate(rng) for _ in brick_pairs(n)] for _ in range(depth)]
    masks = rng.random((depth, n)) < p
    return gates, masks


def test_entropies_of_known_states() -> None:
    n = 4
    prod = np.zeros(2**n, dtype=complex)
    prod[5] = 1
    assert np.allclose(cut_profile(prod, n), 0)
    assert tripartite_information(prod, n) == pytest.approx(0, abs=1e-12)
    ghz = np.zeros(2**n, dtype=complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    assert np.allclose(cut_profile(ghz, n)[1:-1], 1)
    assert entropy(ghz, n, [0, 2]) == pytest.approx(1)
    # Bell pairs (0,2) and (1,3): the cut {0, 1} | {2, 3} holds 2 bits.
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    psi = np.kron(bell, bell).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(-1)
    assert entropy(psi, n, [0, 1]) == pytest.approx(2)
    assert entropy(psi, n, [0, 2]) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_state_is_the_effect(seed: int) -> None:
    """``psi_m`` is ``K_m^dagger |m_f>`` normalised, so ``E_m ∝ |psi><psi|``."""
    n, depth = 3, 3
    gates, masks = _circuit(n, depth, 0.5, seed)
    psi, final, mid = backward_trajectory(
        gates, masks, np.random.default_rng(100 + seed)
    )
    assert np.linalg.norm(psi) == pytest.approx(1)
    k = _kraus(gates, masks, {(t, s): b for t, s, b in mid}, final)
    effect = k.conj().T @ k
    assert np.trace(effect).real > 1e-12
    rho = np.outer(psi, psi.conj())
    np.testing.assert_allclose(effect / np.trace(effect), rho, atol=1e-10)


def test_records_follow_the_maximally_mixed_input() -> None:
    """Records are drawn with probability ``Tr(E_m) / 2^N``."""
    n, depth = 2, 2
    gates, masks = _circuit(n, depth, 0.6, 7)
    rng = np.random.default_rng(3)
    counts: dict = {}
    draws = 20000
    for _ in range(draws):
        _, final, mid = backward_trajectory(gates, masks, rng)
        key = (tuple(int(b) for b in final), tuple(sorted(mid)))
        counts[key] = counts.get(key, 0) + 1
    sites = [(t, int(s)) for t in range(depth) for s in np.flatnonzero(masks[t])]
    total = 0.0
    for final in itertools.product((0, 1), repeat=n):
        for bits in itertools.product((0, 1), repeat=len(sites)):
            mid = dict(zip(sites, bits, strict=True))
            k = _kraus(gates, masks, mid, final)
            prob = float(np.trace(k.conj().T @ k).real) / 2**n
            total += prob
            key = (final, tuple(sorted((t, s, b) for (t, s), b in mid.items())))
            freq = counts.get(key, 0) / draws
            assert freq == pytest.approx(prob, abs=4 * np.sqrt(prob / draws) + 1e-3)
    assert total == pytest.approx(1)


def test_haar_order_script_and_plot(tmp_path) -> None:
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "experiments" / "clifford_ridge"

    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    script = load("haar_order")
    for n in (8, 12):
        for gates in ("haar", "clifford"):
            res = script.run(n, [0.1, 0.5], 3, gates, 0)
            row = res["per_p"][0]
            assert len(row["S_profile"]) == n + 1
            assert row["S_profile"][0] == 0 and row["S_profile"][-1] < 1e-9
            assert "I3" in row
            (tmp_path / f"traj_{gates}_N{n}_s0.json").write_text(json.dumps(res))
    span = load("haar_span").run(3, 0, [0.0, 0.3])
    (tmp_path / "haar_span_N3_seed0.json").write_text(json.dumps(span))
    out = tmp_path / "fig"
    load("plot_haar_order").main(
        ["--traj", str(tmp_path), "--span", str(tmp_path), "--out", str(out)]
    )
    assert (out / "haar_order.png").is_file()
