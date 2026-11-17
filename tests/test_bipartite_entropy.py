"""Backward trajectories and entropy profiles of the periodic brickwork (#358)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM.evaluation.bipartite_entropy import (
    HaarCircuit,
    backward_trajectory,
    draw_haar_circuit,
    growth_exponent,
    stabilizer_profile,
    state_profile,
)
from QELM.evaluation.clifford_ridge import draw_circuit, effect_groups, with_mask
from QELM.evaluation.haar_mipt import apply_pair
from QELM.models.circuits.stabilizer import periodic_bonds

REPO_ROOT = Path(__file__).resolve().parents[1]


def _forward_kraus(
    circuit: HaarCircuit, outcomes: dict[tuple[int, int], int], final: np.ndarray
) -> np.ndarray:
    """``K_m`` as a dense matrix, forward in time (columns: inputs)."""
    n = circuit.n
    dim = 2**n
    rows = np.eye(dim, dtype=complex)  # row j = K |j>
    for t in range(circuit.depth):
        for (a, b), g in zip(periodic_bonds(n, t), circuit.gates[t], strict=True):
            rows = apply_pair(rows, n, a, b, g)
        for site in np.flatnonzero(circuit.measured[t]).tolist():
            v = rows.reshape(dim, 2**site, 2, -1)
            v[:, :, 1 - outcomes[(t, site)], :] = 0.0
    index = int("".join(str(int(b)) for b in final), 2)
    keep = np.zeros(dim)
    keep[index] = 1.0
    return np.asarray((rows * keep).T)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_state_and_probability_match_forward_kraus(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 4
    circuit = draw_haar_circuit(n, 6, 0.3, rng)
    psi, log2_prob, outcomes, final = backward_trajectory(circuit, rng)
    k = _forward_kraus(circuit, outcomes, final)
    e = k.conj().T @ k
    # Rank one, E = |phi><phi| with phi proportional to psi.
    phi = k.conj().T[:, int("".join(str(int(b)) for b in final), 2)]
    assert np.isclose(np.trace(e).real / 2**n, 2.0**log2_prob)
    assert np.isclose(abs(np.vdot(phi / np.linalg.norm(phi), psi)), 1.0)
    assert np.allclose(e, np.outer(phi, phi.conj()))
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_record_distribution_sums_to_one() -> None:
    rng = np.random.default_rng(5)
    n = 4
    circuit = HaarCircuit(
        gates=draw_haar_circuit(n, 3, 0.0, rng).gates,
        measured=np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool),
    )
    sites = [(t, int(s)) for t in range(3) for s in np.flatnonzero(circuit.measured[t])]
    total = 0.0
    for bits in itertools.product((0, 1), repeat=len(sites)):
        forced = dict(zip(sites, bits, strict=True))
        for final in itertools.product((0, 1), repeat=n):
            k = _forward_kraus(circuit, forced, np.array(final))
            total += float(np.sum(np.abs(k) ** 2)) / 2**n
            if np.sum(np.abs(k) ** 2) > 1e-12:
                _, lp, _, _ = backward_trajectory(
                    circuit, rng, outcomes=forced, final=final
                )
                assert np.isclose(2.0**lp, np.sum(np.abs(k) ** 2) / 2**n)
    assert np.isclose(total, 1.0)


def test_unmeasured_circuit_is_reproducible_with_seed() -> None:
    a = draw_haar_circuit(6, 4, 0.2, np.random.default_rng(3))
    b = draw_haar_circuit(6, 4, 0.2, np.random.default_rng(3))
    assert np.array_equal(a.measured, b.measured)
    assert np.allclose(a.gates, b.gates)
    psi_a = backward_trajectory(a, np.random.default_rng(4))[0]
    psi_b = backward_trajectory(b, np.random.default_rng(4))[0]
    assert np.allclose(psi_a, psi_b)


def _bell_pairs_ring(n: int, shift: int) -> np.ndarray:
    """Bell pairs on the bonds ``(shift + 2k, shift + 2k + 1) mod n``."""
    psi = np.zeros(2**n, dtype=complex)
    pairs = [((shift + 2 * k) % n, (shift + 2 * k + 1) % n) for k in range(n // 2)]
    for bits in itertools.product((0, 1), repeat=n // 2):
        q = [0] * n
        for (a, b), x in zip(pairs, bits, strict=True):
            q[a] = q[b] = x
        psi[int("".join(map(str, q)), 2)] = 1.0
    return psi / np.linalg.norm(psi)


def test_profile_of_known_states() -> None:
    n = 8
    product = np.zeros(2**n)
    product[0] = 1.0
    prof = state_profile(product, n)
    assert np.allclose(prof.s1, 0.0) and np.allclose(prof.s2, 0.0)
    assert np.isclose(prof.i3, 0.0)
    # Dimers on the even bonds: an interval of length l starting at s cuts
    # one bond at each odd end. S(l) averaged over starts: 1 for odd l,
    # 0 or 2 (mean 1) for even l.
    prof = state_profile(_bell_pairs_ring(n, 0), n)
    assert np.allclose(prof.s1[1:], 1.0)
    assert np.allclose(prof.s2, prof.s1)
    # Bell pairs never join three quarters: I3 = 0.
    assert np.isclose(prof.i3, 0.0)
    # GHZ: S = 1 for every cut, I3 = 1 + 1 + 1 - 1 - 1 - 1 + 1 = 1.
    ghz = np.zeros(2**n)
    ghz[0] = ghz[-1] = 2**-0.5
    prof = state_profile(ghz, n)
    assert np.allclose(prof.s1[1:], 1.0) and np.isclose(prof.i3, 1.0)


def test_haar_random_state_is_volume_law() -> None:
    rng = np.random.default_rng(0)
    n = 12
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    prof = state_profile(psi, n, starts=[0])
    # Page: S(l) ~ l - 2^(2l - n) / (2 ln 2) for l <= n/2.
    page = np.array([ell - 2.0 ** (2 * ell - n) / (2 * np.log(2)) for ell in range(7)])
    assert np.allclose(prof.s1[1:], page[1:], atol=0.05)
    assert np.all(prof.s2 <= prof.s1 + 1e-12)
    assert prof.i3 < -3.5  # scrambled: 3 S(q) + S(3q) - 3 S(2q) = -3.8


@pytest.mark.parametrize("p", [0.05, 0.3])
def test_stabilizer_profile_matches_state_vector(p: float) -> None:
    """Clifford: GF(2) entropies equal those of the reference state vector."""
    import stim

    rng = np.random.default_rng(7)
    n = 8
    circuit = with_mask(draw_circuit(n, 2 * n, rng), rng.random((2 * n, n)), p)
    gens = effect_groups(circuit).full
    prof = stabilizer_profile(gens)
    # State vector of the stabilizer state of the references.
    paulis = []
    for row in gens:
        x, z = row[:n], row[n:]
        paulis.append(
            stim.PauliString(
                "".join("_XZY"[int(a) + 2 * int(b)] for a, b in zip(x, z, strict=True))
            )
        )
    # Signs do not change entropies: fix them all to +.
    tab = stim.Tableau.from_stabilizers(paulis, allow_redundant=False)
    psi = tab.to_state_vector(endian="big")
    ref = state_profile(psi, n)
    assert np.allclose(prof.s1, ref.s1)
    assert np.isclose(prof.i3, ref.i3)


def test_growth_exponent() -> None:
    mid, alpha = growth_exponent([8, 16, 32], [2.0, 4.0, 4.0])
    assert np.allclose(mid, [np.sqrt(128), np.sqrt(512)])
    assert np.allclose(alpha, [1.0, 0.0])


def _load(name: str) -> ModuleType:
    path = REPO_ROOT / "experiments" / "bipartite_entropy" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bipartite_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


def test_run_and_analyse_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, analyse = _load("run"), _load("analyse")
    runs = tmp_path / "runs"
    # Three p inside the S(N/2) collapse windows (0.10-0.24 and 0.10-0.30).
    grid = ["0.05", "0.12", "0.17", "0.22", "0.4"]
    for model, sizes in (("haar", (4, 8)), ("clifford", (8, 12))):
        for n in sizes:
            argv = [
                "run.py",
                model,
                "--n",
                str(n),
                "--samples",
                "6",
                "--out",
                str(runs),
            ]
            monkeypatch.setattr(sys, "argv", [*argv, "--p", *grid])
            run.main()
    data = json.loads((runs / "haar_N8_T16_s0.json").read_text())
    assert len(data["per_p"]) == 5 and len(data["per_p"][0]["s1"]) == 5
    out = tmp_path / "figures"
    analyse.main(["--runs", str(runs), "--out", str(out), "--n-boot", "2"])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"clifford", "haar"}
    assert "S1" in summary["haar"] and "alpha" in summary["clifford"]
    for name in ("order", "collapse", "collapse_map", "profiles"):
        assert (out / f"{name}.png").exists()


def test_summary_table_from_committed_summaries(tmp_path: Path) -> None:
    table = _load("summary_table")
    table.main(["--out", str(tmp_path)])
    data = json.loads((tmp_path / "summary_table.json").read_text())["extra"]
    assert set(data["cells"]) == {"clifford", "haar"}
    assert data["cells"]["clifford"]["I3"]["p_c"].startswith("0.15")
    assert (tmp_path / "summary_table.png").exists()
