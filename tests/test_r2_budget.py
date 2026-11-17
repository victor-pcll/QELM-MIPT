"""Population R^2 budget of random dense targets (issue #360)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb
from QELM.evaluation.clifford_ridge import draw_circuit, with_mask
from QELM.evaluation.two_filters import all_codes, ridge_fit, string_values


@pytest.mark.parametrize("kind", ["haar", "clifford"])
@pytest.mark.parametrize("p", [0.0, 0.3])
def test_dense_budget_sums_to_one_without_cross_term(kind: str, p: float) -> None:
    rng = np.random.default_rng(1)
    circuit = ff.draw_circuit(3, 3, p, kind, rng)
    g = rb.draw_whitened_targets(3, 20, "iid", rng)
    b = rb.dense_budget(circuit, (1, 2), g)
    total = b.circuit[None] + b.readout + b.r2
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert np.abs(b.cross).max() < 1e-12
    for arr in (b.circuit, b.readout, b.r2):
        assert np.all(arr > -1e-12) and np.all(arr < 1 + 1e-12)
    # Larger k reads more: E_readout decreases, R^2 increases.
    assert np.all(b.readout[1] <= b.readout[0] + 1e-12)


@pytest.mark.parametrize("kind", ["haar", "clifford"])
def test_dense_budget_matches_population_error(kind: str) -> None:
    rng = np.random.default_rng(2)
    n = 3
    circuit = ff.draw_circuit(n, 3, 0.25, kind, rng)
    records = ff.enumerate_records(circuit)
    c = ff.feature_matrix(records.effects)
    w = ff.input_weights(n, "xz")
    g = rb.draw_whitened_targets(n, 5, "flat", rng)
    b = rb.dense_budget(circuit, (1, n_bits := ff.n_bits(circuit)), g)
    for t in range(5):
        beta = np.zeros(4**n)
        sw = np.sqrt(w[1:])
        np.divide(g[t], sw, out=beta[1:], where=sw > 0)
        assert b.circuit[t] == pytest.approx(ff.population_error(c, w, beta), abs=1e-10)
        c1 = ff.parity_features(c, records.bits, 1)
        split = ff.two_filter_split(c1, c, w, beta)
        assert b.readout[0, t] == pytest.approx(split.readout, abs=1e-10)
    # Every parity (k = number of bits) reads the whole span: no readout error.
    assert n_bits > 1
    np.testing.assert_allclose(b.readout[1], 0.0, atol=1e-12)


def test_whitened_targets_prior() -> None:
    rng = np.random.default_rng(3)
    g = rb.draw_whitened_targets(3, 20_000, "iid", rng)
    w = ff.input_weights(3, "xz")[1:]
    np.testing.assert_allclose(np.mean(g**2, axis=0), w, atol=0.05)
    assert np.all(g[:, w == 0] == 0)
    flat = rb.draw_whitened_targets(3, 20_000, "flat", rng)
    np.testing.assert_allclose(np.mean(flat[:, w > 0] ** 2, axis=0), 1.0, atol=0.06)
    with pytest.raises(ValueError):
        rb.draw_whitened_targets(3, 1, "other", rng)


def test_visible_index_is_a_bijection() -> None:
    s = rb.visible_strings(3)
    np.testing.assert_array_equal(rb.visible_index(s), np.arange(27))


@pytest.mark.parametrize("p", [0.1, 0.35])
def test_counting_route_equals_ridge_on_the_whole_population(p: float) -> None:
    """Clifford counting = lambda -> 0 ridge on all 4^N product inputs (N = 4)."""
    rng = np.random.default_rng(4)
    n, t = 4, 8
    circuit = with_mask(draw_circuit(n, t, rng), rng.random((t, n)), p)
    ks = (2, 1)
    in_g, in_d = rb.clifford_masks(circuit, ks)
    assert not np.any(in_d & ~in_g)
    strings = rb.visible_strings(n)
    codes = all_codes(n)
    mono = string_values(codes, strings[1:]).astype(float)
    weight = strings[1:].sum(axis=1)
    beta = rng.standard_normal((3, strings.shape[0] - 1))
    q = np.zeros((3, 3**n))
    q[:, 1:] = 0.5**weight * beta**2
    b = rb.counting_budget(in_g, in_d, q)
    phi_s = string_values(codes, strings[in_g])
    for j in range(3):
        y = mono @ beta[j]
        var = np.mean(y**2)
        pred_s = ridge_fit(phi_s, y, 0.0)
        assert b.circuit[j] == pytest.approx(np.mean((y - pred_s) ** 2) / var, abs=1e-9)
        for i, _ in enumerate(ks):
            phi_k = string_values(codes, strings[in_d[i]])
            pred_k = ridge_fit(phi_k, y, 0.0)
            r2 = 1 - np.mean((y - pred_k) ** 2) / var
            assert b.r2[i, j] == pytest.approx(r2, abs=1e-9)
            read = np.mean((pred_s - pred_k) ** 2) / var
            assert b.readout[i, j] == pytest.approx(read, abs=1e-9)
    # ks given out of order: rows follow ks.
    assert in_d[1].sum() <= in_d[0].sum()


def test_seed_reproducibility() -> None:
    def one(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        circuit = with_mask(draw_circuit(6, 12, rng), rng.random((12, 6)), 0.2)
        q = rb.draw_power(6, 4, "iid", rng)
        return rb.clifford_budget(circuit, (1, 2), q).r2

    np.testing.assert_array_equal(one(7), one(7))


def test_run_and_plot_smoke(tmp_path: Path) -> None:
    import importlib.util

    root = Path(__file__).parents[1] / "experiments" / "r2_budget"

    def load(name: str):  # type: ignore[no-untyped-def]
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    run = load("run")
    common = ["--circuits", "2", "--targets", "3", "--p", "0.0", "0.2"]
    run.main(["dense", "--n", "3", *common, "--out", str(tmp_path)])
    run.main(["clifford", "--n", "4", "--k", "1", "2", *common, "--out", str(tmp_path)])
    data = json.loads((tmp_path / "dense_N3_iid.json").read_text())
    assert data["per_p"]["haar"][0]["max_abs_sum_minus_one"] < 1e-10
    load("plot").main(["--data", str(tmp_path), "--n", "3", "--clifford-n", "4"])
    assert (tmp_path / "figures" / "budget_hist_k2_iid.png").exists()
