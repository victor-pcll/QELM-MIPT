"""Exact W(m) and Delta of the periodic Haar brickwork (issue #348)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM.evaluation.effect_mipt import apply_gate, fourier_weight
from QELM.evaluation.haar_mipt import apply_pair, haar_record
from QELM.models.circuits.haar import haar_gate
from QELM.models.circuits.stabilizer import periodic_bonds

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "haar_collapse_map" / "analyse.py"


def _reference_kraus(
    n: int, p: float, half_layers: int, rng: np.random.Generator
) -> np.ndarray:
    """Slow K_m (columns: inputs), same random stream as ``haar_record``."""
    dim = 2**n
    kt = np.eye(dim, dtype=complex) / np.sqrt(dim)  # rows: K|j>
    for t in range(half_layers):
        for a, b in periodic_bonds(n, t):
            kt = apply_pair(kt, n, a, b, haar_gate(rng))
        for site in np.flatnonzero(rng.random(n) < p).tolist():
            v = kt.reshape(dim, 2**site, 2, 2 ** (n - site - 1))
            p0 = float(np.sum(np.abs(v[:, :, 0, :]) ** 2))
            total = float(np.sum(np.abs(kt) ** 2))
            o = 0 if rng.random() * total < p0 else 1
            v[:, :, 1 - o, :] = 0.0
            kt = kt / np.sqrt(p0 if o == 0 else total - p0)
    return np.asarray(kt.T)


def test_apply_pair_matches_adjacent_gate_and_wraps() -> None:
    rng = np.random.default_rng(1)
    n = 4
    psi = rng.normal(size=(3, 2**n)) + 1j * rng.normal(size=(3, 2**n))
    g = haar_gate(rng)
    assert np.allclose(apply_pair(psi, n, 1, 2, g), apply_gate(psi, n, 1, g))
    # Bond (3, 0): the gate acts on |q_3 q_0>; build it as a full matrix.
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    # On (q_0, q_3) the gate is SWAP g SWAP; embed with q_1 q_2 in the middle.
    full = np.zeros((16, 16), dtype=complex)
    g03 = swap @ g @ swap
    for i in range(16):
        q = [(i >> (3 - s)) & 1 for s in range(4)]
        for a in range(4):
            q0, q3 = a >> 1, a & 1
            j = (q0 << 3) | (q[1] << 2) | (q[2] << 1) | q3
            full[j, i] += g03[a, (q[0] << 1) | q[3]]
    assert np.allclose(apply_pair(psi, n, 3, 0, g), psi @ full.T)


@pytest.mark.parametrize(("n", "p"), [(4, 0.2), (6, 0.15), (6, 0.4)])
def test_haar_record_equals_the_slow_reference(n: int, p: float) -> None:
    cps = (n, 2 * n)
    rec = haar_record(n, p, cps, np.random.default_rng(7))
    rng = np.random.default_rng(7)
    k_ref = _reference_kraus(n, p, 2 * n, rng)
    e = k_ref.conj().T @ k_ref
    purity = float(np.trace(e @ e).real) / float(np.trace(e).real) ** 2
    assert rec.delta[-1] == pytest.approx(n - np.log2(2**n * purity), abs=1e-9)
    # 1 + W equals the dense Fourier weight of #314 on the same effect.
    assert rec.weight[-1] == pytest.approx(fourier_weight(k_ref).dense, rel=1e-8)


def test_limits_and_identities() -> None:
    n = 6
    rec = haar_record(n, 0.0, (1, 6, 12), np.random.default_rng(0))
    assert np.allclose(rec.delta, n) and np.allclose(rec.weight, 0.0)
    assert np.all(rec.measured == 0)
    rec = haar_record(n, 1.0, (1, 3), np.random.default_rng(0))
    assert np.allclose(rec.delta, 0.0, atol=1e-9)
    assert list(rec.measured) == [n, 3 * n]
    rec = haar_record(n, 0.2, (6, 12), np.random.default_rng(3))
    assert np.allclose(rec.delta, n - np.log2(1 + rec.weight))
    assert np.all((rec.delta >= -1e-9) & (rec.delta <= n + 1e-9))


def test_seed_and_single_precision() -> None:
    a = haar_record(8, 0.15, (8, 16), np.random.default_rng(5))
    b = haar_record(8, 0.15, (8, 16), np.random.default_rng(5))
    c = haar_record(8, 0.15, (8, 16), np.random.default_rng(5), np.complex64)
    assert np.array_equal(a.delta, b.delta)
    assert np.allclose(a.delta, c.delta, atol=1e-4)


def test_invalid_arguments() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="even"):
        haar_record(5, 0.1, (5,), rng)
    with pytest.raises(ValueError, match="increasing"):
        haar_record(4, 0.1, (4, 4), rng)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("haar_analyse", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["haar_analyse"] = module
    spec.loader.exec_module(module)
    return module


def _write_run(folder: Path, n: int, p_c: float, nu: float) -> None:
    """A run file whose Delta at T = 2N scales exactly with (p_c, nu)."""
    p = np.round(np.arange(0.04, 0.361, 0.02), 2)
    x = (p - p_c) * n ** (1 / nu)
    delta = 1.5 * np.exp(-1.2 * x)
    rows = [
        {"p": float(pi), "delta_T2N": [float(d), 0.01 * float(d)]}
        for pi, d in zip(p, delta, strict=True)
    ]
    run = {"n": n, "ratios": [2], "per_p": rows}
    (folder / f"haar_N{n}.json").write_text(json.dumps(run))


def test_analysis_recovers_synthetic_p_c_and_nu(tmp_path: Path) -> None:
    mod = _load()
    for n in (4, 6, 8, 10):
        _write_run(tmp_path, n, 0.17, 1.2)
    curves = mod.load_curves(tmp_path, mod.MAIN)
    assert sorted(curves) == [4, 6, 8, 10]
    assert not mod.load_curves(tmp_path, mod.Case("x", "x", 4))
    assert sorted(mod.load_curves(tmp_path, mod.VARIANTS[0])) == [6, 8, 10]
    result = mod.analyse(curves, wide=False, n_boot=0)
    assert result["fit"]["p_c"] == pytest.approx(0.17, abs=2e-3)
    assert result["fit"]["nu"] == pytest.approx(1.2, abs=0.05)
    for row in result["crossings"]:
        assert row["p"] == pytest.approx(0.17, abs=1e-3)


def test_committed_runs_and_summary_agree() -> None:
    mod = _load()
    summary = json.loads((SCRIPT.parent / "figures" / "summary.json").read_text())
    curves = mod.load_curves(SCRIPT.parent / "runs", mod.MAIN)
    assert sorted(curves) == summary["main"]["sizes"]
    fit = summary["main"]["fit"]
    assert fit["quality"] == pytest.approx(
        mod.collapse_quality(curves, fit["p_c"], fit["nu"])
    )
