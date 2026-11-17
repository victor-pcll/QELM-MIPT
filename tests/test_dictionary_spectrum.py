"""Spectrum of the effect dictionary (issue #356)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import comb

from QELM.evaluation.clifford_ridge import (
    GroupBasis,
    draw_circuit,
    effect_groups,
    enumerate_group,
    pauli_bits,
    purification,
    visible_weights,
    with_mask,
)
from QELM.evaluation.dictionary_spectrum import (
    born_profile,
    degree_profile,
    effective_ranks,
    mean_weight,
    product_z_reference,
    random_strings_null,
    span_profile,
    weight_enumerator,
)
from QELM.evaluation.effect_rank import GATES, brick_pairs
from QELM.evaluation.ridge_spectrum import (
    conditional_means,
    effect_span,
    sample_effect,
)


def test_z_group_is_binomial() -> None:
    n = 5
    gens = np.stack([pauli_bits(n, {i: "Z"}) for i in range(n)])
    all_s, vis = weight_enumerator(gens)
    np.testing.assert_array_equal(all_s, product_z_reference(n))
    np.testing.assert_array_equal(vis, all_s)


def test_dependent_generators_are_reduced() -> None:
    n = 4
    a = pauli_bits(n, {0: "X", 1: "Z"})
    b = pauli_bits(n, {1: "Z", 2: "X"})
    all_s, _ = weight_enumerator(np.stack([a, b, a ^ b, a]))
    assert all_s.sum() == 4  # rank 2


def test_y_strings_are_not_visible() -> None:
    n = 2
    x = pauli_bits(n, {0: "X"})
    z = pauli_bits(n, {0: "Z"})
    all_s, vis = weight_enumerator(np.stack([x, z]))  # {I, X, Z, Y} on site 0
    np.testing.assert_array_equal(all_s, [1, 3, 0])
    np.testing.assert_array_equal(vis, [1, 2, 0])


@pytest.mark.parametrize("p", [0.0, 0.1, 0.3, 0.7])
def test_sum_is_two_to_the_rank(p: float) -> None:
    """Exact: sum_l N(l) = 2^N with the readout, 2^(N - Delta) without."""
    n, t = 8, 16
    rng = np.random.default_rng(3)
    gates = draw_circuit(n, t, rng)
    circ = with_mask(gates, rng.random((t, n)), p)
    groups = effect_groups(circ)
    delta = purification(circ, {}).delta
    full, full_vis = weight_enumerator(groups.full)
    mid, mid_vis = weight_enumerator(groups.mid)
    assert full.sum() == 2**n
    assert mid.sum() == 2 ** round(n - delta)
    # Same counts as the enumeration of clifford_ridge.
    for gens, vis in ((groups.full, full_vis), (groups.mid, mid_vis)):
        w = visible_weights(enumerate_group(GroupBasis(gens).rows))
        np.testing.assert_array_equal(vis[1:], np.bincount(w, minlength=n + 1)[1:])
        assert vis[0] == 1


def test_null_model_sums() -> None:
    n, r = 10, 7
    all_s, vis = random_strings_null(n, r)
    assert all_s.sum() == pytest.approx(2.0**r)
    assert vis.sum() == pytest.approx(2.0**r * 0.75**n)
    assert int(np.argmax(all_s)) in (7, 8)  # 3N/4 = 7.5


def test_mean_weight() -> None:
    assert mean_weight(np.array([1, 0, 2, 0])) == 2.0
    assert mean_weight(product_z_reference(6)) == pytest.approx(6 * 32 / 63)


def test_degree_profile_counts_monomials() -> None:
    n = 3
    prof = degree_profile(np.ones(3**n), n)
    np.testing.assert_allclose(prof, [comb(n, k) * 2**k for k in range(n + 1)])


def _open_circuit(kind: str, n: int, depth: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [[GATES[kind](rng) for _ in brick_pairs(n)] for _ in range(depth)]


@pytest.mark.parametrize("final", [False, True])
def test_clifford_span_and_born_profiles_agree(final: bool) -> None:
    """For Clifford gates both profiles are the integer histogram N_vis(l)."""
    n, depth = 4, 8
    gates = _open_circuit("clifford", n, depth, 5)
    masks = np.random.default_rng(6).random((depth, n)) < 0.3
    span = effect_span(gates, masks, final)
    sp = span_profile(span, n)
    assert sp.sum() == pytest.approx(span.shape[0])
    np.testing.assert_allclose(sp, np.round(sp), atol=1e-6)
    rng = np.random.default_rng(7)
    v = np.array(
        [conditional_means(sample_effect(gates, masks, final, rng)) for _ in range(30)]
    )
    np.testing.assert_allclose(born_profile(v, n), sp, atol=1e-6)


def test_haar_profiles_are_bounded() -> None:
    """Haar: both profiles lie between 0 and the number of degree-l modes."""
    n, depth = 3, 6
    gates = _open_circuit("haar", n, depth, 1)
    masks = np.random.default_rng(2).random((depth, n)) < 0.2
    span = effect_span(gates, masks, True)
    sp = span_profile(span, n)
    rng = np.random.default_rng(3)
    v = np.array(
        [conditional_means(sample_effect(gates, masks, True, rng)) for _ in range(200)]
    )
    bp = born_profile(v, n)
    cap = np.array([comb(n, k) * 2**k for k in range(n + 1)])
    assert bp[0] == pytest.approx(1.0)
    assert np.all(bp >= -1e-12) and np.all(bp <= cap + 1e-9)
    assert np.all(sp <= cap + 1e-9)
    assert sp.sum() == pytest.approx(span.shape[0])


def test_effective_ranks() -> None:
    mu = np.array([0.5, 0.25, 0.25, 0.0])
    out = effective_ranks(mu, [0.3, 0.1, 0.0])
    assert out["above"] == [1, 3, 3]
    assert out["participation"] == pytest.approx(1.0 / 0.375)
    assert out["entropy"] == pytest.approx(np.exp(1.5 * np.log(2)))
    assert effective_ranks(np.zeros(3), [0.1])["participation"] == 0.0


def _script(name: str):  # type: ignore[no-untyped-def]
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "experiments" / "dictionary_spectrum" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scripts_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    cw = _script("clifford_weights.py")
    for n in (6, 8):
        res = cw.run(n, 3, [0.05, 0.3], seed=0)
        assert np.allclose(np.sum(res["full_all"]["mean"], axis=1), 2**n)
        (tmp_path / f"clifford_weights_N{n}.json").write_text(json.dumps(res))
    _script("plot_clifford_weights.py").main([str(tmp_path), "--rows", "6"])
    assert (tmp_path / "clifford_weights.png").exists()
    hs = _script("haar_spectrum.py")
    res = hs.run(3, 0, [0.04, 0.16, 0.3, 0.5], records=60)
    rows = res["per_kind"]["clifford"]
    for row in rows:
        for rec in ("mid", "full"):
            assert sum(row[rec]["span_profile"]) == pytest.approx(row[rec]["span_rank"])
    (tmp_path / "haar_spectrum_N3_seed0.json").write_text(json.dumps(res))
    _script("plot_haar_spectrum.py").main([str(tmp_path), "--n", "3"])
    assert (tmp_path / "haar_spectrum.png").exists()
