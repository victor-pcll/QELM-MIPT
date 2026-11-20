"""Minimal parity weight of the Clifford effects (issue #386)."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from QELM.evaluation import r2_budget as rb
from QELM.evaluation import readout_locality as rl
from QELM.evaluation.clifford_ridge import draw_circuit, parity_strings, with_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "readout_locality"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, FOLDER / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def circuit(n: int, p: float, seed: int):
    rng = np.random.default_rng(seed)
    return with_mask(draw_circuit(n, 2 * n, rng), rng.random((2 * n, n)), p)


def code_index(n: int) -> np.ndarray:
    """``visible_index`` of :mod:`r2_budget` on the ``(4,) * n`` grid."""
    idx = np.zeros((4,) * n, dtype=np.int64)
    for i in range(n):
        shape = [1] * n
        shape[i] = 4
        idx = idx + (np.arange(4) * 3**i).reshape(shape)
    return idx


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
def test_linear_maps_match_parity_strings(p: float) -> None:
    c = circuit(6, p, 1)
    a, b = rl.parity_linear_maps(c)
    nb = c.n_mid + c.N
    assert a.shape == (c.n_mid, nb) and b.shape == (2 * c.N, nb)
    rng = np.random.default_rng(2)
    sets = [
        sorted(rng.choice(nb, rng.integers(1, min(6, nb) + 1), replace=False))
        for _ in range(2000)
    ]
    strings, alive = parity_strings(c, sets)
    s = np.zeros((len(sets), nb), dtype=np.int64)
    for i, x in enumerate(sets):
        s[i, x] = 1
    lin_alive = ~((s @ a.T.astype(np.int64)) % 2).any(axis=1)
    image = ((s @ b.T.astype(np.int64)) % 2).astype(bool)
    np.testing.assert_array_equal(lin_alive, alive)
    np.testing.assert_array_equal(image[alive], strings[alive])


def test_nullspace_gf2() -> None:
    rng = np.random.default_rng(3)
    m = rng.random((7, 12)) < 0.4
    k = rl.nullspace_gf2(m)
    assert k.shape[0] >= 12 - 7
    np.testing.assert_array_equal((k.astype(int) @ m.T.astype(int)) % 2, 0)
    # Independent rows: the free columns carry an identity.
    assert np.linalg.matrix_rank(k.astype(float)) == k.shape[0]


@pytest.mark.parametrize(("n", "p"), [(4, 0.0), (4, 0.3), (6, 0.1), (6, 0.5)])
def test_dp_matches_brute_force_over_alive_sets(n: int, p: float) -> None:
    c = circuit(n, p, 4)
    w = rl.min_parity_weights(c)
    a, b = rl.parity_linear_maps(c)
    basis = rl.nullspace_gf2(a).astype(np.int64)
    assert basis.shape[0] <= 18
    best = np.full((4,) * n, rl.INF, dtype=np.int64)
    for bits in itertools.product([0, 1], repeat=basis.shape[0]):
        s = (np.array(bits) @ basis) % 2
        img = (b.astype(np.int64) @ s) % 2
        code = tuple((img[:n] + 2 * img[n:]).tolist())
        best[code] = min(best[code], int(s.sum()))
    np.testing.assert_array_equal(w, best)
    assert w[(0,) * n] == 0


@pytest.mark.parametrize("p", [0.0, 0.15, 0.4])
def test_dp_matches_clifford_masks(p: float) -> None:
    n = 6
    c = circuit(n, p, 5)
    w = rl.min_parity_weights(c)
    vis = rl.visible_codes(n)
    in_g, in_d = rb.clifford_masks(c, (1, 2, 3))
    idx = code_index(n)[vis]
    np.testing.assert_array_equal(in_g[idx], w[vis] < rl.INF)
    for j, k in enumerate((1, 2, 3)):
        np.testing.assert_array_equal(in_d[j][idx], w[vis] <= k)


@pytest.mark.parametrize("prior", ["flat", "iid"])
def test_relative_readout_is_the_budget_ratio(prior: str) -> None:
    n = 6
    c = circuit(n, 0.25, 6)
    ks = (1, 2, 3)
    w = rl.min_parity_weights(c)
    vis = rl.visible_codes(n)
    size = rl.string_weights(n)[vis]
    in_g, in_d = rb.clifford_masks(c, ks)
    # Mean power per string of each prior on the visible strings.
    q_vis = np.ones(vis.sum()) if prior == "flat" else 0.5**size
    q = np.zeros((1, 3**n))
    q[0, code_index(n)[vis]] = q_vis
    budget = rb.counting_budget(in_g, in_d, q)
    keep = w[vis] < rl.INF
    got = rl.relative_readout(w[vis][keep], np.array(ks), q_vis[keep])
    np.testing.assert_allclose(got, budget.readout[:, 0] / (1 - budget.circuit[0]))


def test_p_zero_weights_are_z_string_weights() -> None:
    # Without mid-circuit measurement, G is the image of the Z strings and
    # w(g) is the number of Z factors of its preimage: N/2 on average.
    n = 6
    c = circuit(n, 0.0, 7)
    w = rl.min_parity_weights(c)
    finite = w[w < rl.INF]
    assert finite.size == 2**n
    assert np.bincount(finite).tolist() == [1, 6, 15, 20, 15, 6, 1]


def test_visible_codes_and_weights() -> None:
    vis = rl.visible_codes(3)
    assert vis.sum() == 3**3 - 1
    assert rl.string_weights(3)[1, 0, 2] == 2
    assert rl.relative_readout(np.array([]), np.array([1]))[0] != 0


def test_run_script_is_reproducible(tmp_path: Path) -> None:
    run = load("run")
    args = ["--n", "4", "--circuits", "3", "--p", "0", "0.3", "--workers", "1"]
    run.main([*args, "--out", str(tmp_path / "a")])
    run.main([*args, "--out", str(tmp_path / "b")])
    a = np.load(tmp_path / "a" / "locality_N4.npz")
    b = np.load(tmp_path / "b" / "locality_N4.npz")
    np.testing.assert_array_equal(a["hist"], b["hist"])
    assert a["hist"].shape == (3, 2, run.W_MAX + 1, 5)
    # p = 0: 2^N - 1 non-identity strings in G, the visible ones are counted.
    assert np.all(a["hist"][:, 0].sum(axis=(1, 2)) <= 2**4 - 1)
    assert json.loads(str(a["config"]))["depth"] == 8


def test_haar_script_smoke(tmp_path: Path) -> None:
    haar = load("haar")
    haar.main(
        [
            "--n",
            "3",
            "--circuits",
            "2",
            "--k-max",
            "2",
            "--p",
            "0",
            "0.3",
            "--workers",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    d = np.load(tmp_path / "haar_N3.npz")
    rank = d["rank"]
    assert rank.shape == (2, 2, 2, 3)
    # Nested spans: rank grows with k and is bounded by the full span.
    assert np.all(np.diff(rank, axis=-1) >= 0)
