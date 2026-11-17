"""Exact small-N checks of the general Fourier framework (issue #353).

Each test checks one statement of ``documentation/notes/cadre-fourier-general.md``.
Statements that the note calls *general* are checked on Haar circuits (and on
Clifford circuits when cheap); statements that the note calls *Clifford only*
are checked on Clifford circuits and shown to fail on a Haar circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation import fourier_framework as ff

N = 3
DEPTH = 3


def _circuit(
    kind: str, p: float, seed: int, n: int = N, depth: int = DEPTH
) -> ff.Circuit:
    return ff.draw_circuit(n, depth, p, kind, np.random.default_rng(seed))


def _index(codes: str) -> int:
    lookup = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    out = 0
    for ch in codes:
        out = 4 * out + lookup[ch]
    return out


def _random_target(n: int, ensemble: str, rng: np.random.Generator) -> np.ndarray:
    beta = rng.standard_normal(4**n) * ff.visible_mask(n, ensemble)
    beta[0] = 0.0
    return beta


# --- Section 1: inputs and Fourier basis (general) --------------------------


@pytest.mark.parametrize("ensemble", ["xz", "sphere"])
def test_monomials_are_orthogonal_with_weights(ensemble: str) -> None:
    inputs = ff.eigenstate_inputs(N, ensemble)
    gram = inputs.monomials.T @ inputs.monomials / len(inputs.monomials)
    np.testing.assert_allclose(gram, np.diag(ff.input_weights(N, ensemble)), atol=1e-12)


@pytest.mark.parametrize("kind", ["haar", "clifford"])
def test_features_are_the_fourier_transform_of_the_effects(kind: str) -> None:
    circuit = _circuit(kind, 0.3, seed=1)
    records = ff.enumerate_records(circuit)
    c = ff.feature_matrix(records.effects)
    inputs = ff.eigenstate_inputs(N, "sphere")
    rhos = ff.density_matrices(inputs)
    born = np.einsum("mij,xji->xm", records.effects, rhos).real
    np.testing.assert_allclose(born, inputs.monomials @ c.T, atol=1e-12)
    # Completeness: sum_m E_m = I, so sum_m C[m, a] = delta_{a, I}.
    np.testing.assert_allclose(c.sum(axis=0), np.eye(4**N)[0], atol=1e-12)


@pytest.mark.parametrize("kind", ["haar", "clifford"])
def test_pauli_transfer_matrix(kind: str) -> None:
    gate = ff.draw_circuit(2, 1, 0.0, kind, np.random.default_rng(2)).gates[0][0][1]
    r = ff.pauli_transfer_matrix(gate)
    np.testing.assert_allclose(r @ r.T, np.eye(16), atol=1e-12)
    nonzero_per_row = np.count_nonzero(np.abs(r) > 1e-9, axis=1)
    if kind == "clifford":
        assert np.all(nonzero_per_row == 1)  # signed permutation
    else:
        assert nonzero_per_row[1:].min() > 1  # a Haar gate spreads every string


# --- Section 2: population ridge = weighted projection (general) ------------


@pytest.mark.parametrize("kind", ["haar", "clifford"])
@pytest.mark.parametrize("lam", [0.0, 0.05])
def test_population_ridge_matches_brute_force(kind: str, lam: float) -> None:
    circuit = _circuit(kind, 0.3, seed=4)
    records = ff.enumerate_records(circuit)
    c = ff.feature_matrix(records.effects)
    inputs = ff.eigenstate_inputs(N, "xz")
    w = ff.input_weights(N, "xz")
    rng = np.random.default_rng(5)
    beta = _random_target(N, "xz", rng)
    y = inputs.monomials @ beta
    phi = inputs.monomials @ c.T
    phi_c = phi - phi.mean(axis=0)
    y_c = y - y.mean()
    n_in = len(y)
    gram = phi_c.T @ phi_c / n_in
    rhs = phi_c.T @ y_c / n_in
    if lam == 0.0:
        coef = np.linalg.lstsq(phi_c, y_c, rcond=None)[0]
    else:
        coef = np.linalg.solve(gram + lam * np.eye(len(gram)), rhs)
    brute = np.mean((y_c - phi_c @ coef) ** 2) / np.var(y)
    assert ff.population_error(c, w, beta, lam) == pytest.approx(brute, abs=1e-9)


def test_ridge_spectrum_is_that_of_c_w_half_not_of_c() -> None:
    circuit = _circuit("haar", 0.3, seed=6)
    c = ff.feature_matrix(ff.enumerate_records(circuit).effects)
    w = ff.input_weights(N, "xz")
    eig = np.sort(np.linalg.eigvalsh(ff.population_covariance(c, w)))[::-1]
    sv = np.linalg.svd(c[:, 1:] * np.sqrt(w[1:]), compute_uv=False) ** 2
    np.testing.assert_allclose(eig[: len(sv)], sv, atol=1e-12)
    sv_raw = np.linalg.svd(c[:, 1:], compute_uv=False) ** 2
    assert not np.allclose(sv_raw[:5], sv[:5])


# --- Section 3: Walsh-Pauli duality and nested readouts (general) -----------


@pytest.mark.parametrize("kind", ["haar", "clifford"])
def test_all_parities_span_the_effects(kind: str) -> None:
    circuit = _circuit(kind, 0.3, seed=7)
    records = ff.enumerate_records(circuit)
    c = ff.feature_matrix(records.effects)
    par = ff.parity_features(c, records.bits, None)
    stacked = np.vstack([c, par])
    rank_effects = ff.span_rank(c)
    # The constant (empty parity) is the identity row; add it back.
    rank_par = ff.span_rank(np.vstack([np.eye(4**N)[0], par]))
    assert rank_par == rank_effects == ff.span_rank(stacked)


def test_parity_operators_single_string_only_for_clifford() -> None:
    ipr = {}
    for kind in ("haar", "clifford"):
        records = ff.enumerate_records(_circuit(kind, 0.3, seed=8))
        subsets = ff.parity_subsets(records.bits.shape[1])
        ops = ff.walsh_operators(records, subsets)
        rows = ff.pauli_vectors(ops).real
        ipr[kind] = [ff.participation(r) for r in rows if np.linalg.norm(r) > 1e-9]
    np.testing.assert_allclose(ipr["clifford"], 1.0, atol=1e-9)
    assert max(ipr["haar"]) > 2.0


def test_pythagoras_of_nested_readouts_in_haar() -> None:
    circuit = _circuit("haar", 0.3, seed=9)
    records = ff.enumerate_records(circuit)
    c = ff.feature_matrix(records.effects)
    w = ff.input_weights(N, "xz")
    beta = _random_target(N, "xz", np.random.default_rng(10))
    c1 = ff.parity_features(c, records.bits, 1)
    split = ff.two_filter_split(c1, c, w, beta, lam=0.0)
    assert split.cross == pytest.approx(0.0, abs=1e-9)
    assert split.total == pytest.approx(split.circuit + split.readout, abs=1e-9)
    assert split.circuit == pytest.approx(ff.population_error(c, w, beta), abs=1e-9)
    assert split.total == pytest.approx(ff.population_error(c1, w, beta), abs=1e-9)
    assert split.readout > 1e-3
    # At lambda > 0 the two projections are no longer orthogonal.
    shrunk = ff.two_filter_split(c1, c, w, beta, lam=0.05)
    assert abs(shrunk.cross) > 1e-4
    assert shrunk.total == pytest.approx(
        shrunk.circuit + shrunk.readout + shrunk.cross, abs=1e-12
    )


def test_clifford_split_is_a_sum_of_indicators_haar_is_not() -> None:
    w = ff.input_weights(N, "xz")
    target = _index("ZII")
    beta = np.zeros(4**N)
    beta[target] = 1.0
    errors = {}
    for kind in ("haar", "clifford"):
        values = []
        for seed in range(12):
            c = ff.feature_matrix(
                ff.enumerate_records(_circuit(kind, 0.3, seed)).effects
            )
            values.append(ff.population_error(c, w, beta))
        errors[kind] = np.array(values)
    clif = errors["clifford"]
    assert np.all(np.isclose(clif, 0.0, atol=1e-9) | np.isclose(clif, 1.0, atol=1e-9))
    haar = errors["haar"]
    assert np.any((haar > 1e-3) & (haar < 1 - 1e-3))


# --- Section 4: finite shots, canonical-correlation filter (general) --------


@pytest.mark.parametrize("kind", ["haar", "clifford"])
def test_shot_formula(kind: str) -> None:
    circuit = _circuit(kind, 0.3, seed=11)
    c = ff.feature_matrix(ff.enumerate_records(circuit).effects)
    w = ff.input_weights(N, "xz")
    beta = _random_target(N, "xz", np.random.default_rng(12))
    spec = ff.canonical_spectrum(c, w)
    assert spec.mu.max() <= 1 + 1e-9 and spec.mu.min() >= 0
    for shots in (1, 4, 64, 4096):
        assert ff.spectral_r2(c, w, beta, shots) == pytest.approx(
            ff.histogram_r2(c, w, beta, shots), abs=1e-8
        )
    big = ff.spectral_r2(c, w, beta, 1e12)
    assert big == pytest.approx(1 - ff.population_error(c, w, beta), abs=1e-6)


# --- Section 5: span, commutation, purification (Clifford vs Haar) ----------


def test_p_zero_is_commutative_for_every_circuit() -> None:
    for kind in ("haar", "clifford"):
        c = ff.feature_matrix(
            ff.enumerate_records(_circuit(kind, 0.0, seed=13)).effects
        )
        assert ff.span_rank(c) == 2**N
        assert ff.max_commutator(c) < 1e-9


def test_clifford_effects_commute_and_span_is_2_to_n_minus_delta() -> None:
    for seed in range(6):
        circuit = _circuit("clifford", 0.4, seed)
        full = ff.feature_matrix(ff.enumerate_records(circuit).effects)
        assert ff.max_commutator(full) < 1e-9
        assert ff.span_rank(full) == 2**N
        mid_records = ff.enumerate_records(circuit, final=False)
        mid = ff.feature_matrix(mid_records.effects)
        spec = ff.purification_spectrum(mid_records.effects)
        assert spec.renyi0 == pytest.approx(spec.renyi1, abs=1e-9)
        assert spec.renyi0 == pytest.approx(spec.renyi2, abs=1e-9)
        delta = round(spec.renyi0)
        assert spec.renyi0 == pytest.approx(delta, abs=1e-9)
        assert ff.span_rank(mid) == 2 ** (N - delta)


def test_haar_breaks_the_clifford_structure() -> None:
    circuit = _circuit("haar", 0.3, seed=14, n=4, depth=4)
    full = ff.feature_matrix(ff.enumerate_records(circuit).effects)
    assert ff.max_commutator(full) > 0.1
    assert ff.span_rank(full) > 2**4  # impossible with commuting effects
    mid_records = ff.enumerate_records(circuit, final=False)
    spec = ff.purification_spectrum(mid_records.effects)
    assert spec.renyi0 > spec.renyi1 + 0.05 > spec.renyi2 + 0.05


# --- Section 6: invisibility theorem (Clifford) and its failure (Haar) ------


def test_invisibility_theorem_in_clifford() -> None:
    inputs = ff.eigenstate_inputs(N, "xz")
    w = ff.input_weights(N, "xz")
    vis = np.flatnonzero(ff.visible_mask(N, "xz"))[1:]
    checked = 0
    for seed in range(4):
        circuit = _circuit("clifford", 0.3, seed)
        records = ff.enumerate_records(circuit)
        c = ff.feature_matrix(records.effects)
        phi = inputs.monomials @ c.T
        for a in vis:
            beta = np.zeros(4**N)
            beta[a] = 1.0
            if ff.population_error(c, w, beta) < 0.5:
                continue
            q = ff.anticommuting_centralizer_element(c, int(a), N)
            assert q is not None
            perm = ff.conjugate_inputs(inputs, q)
            # Same law of the records for x and Q x Q, opposite target: true at
            # every number of shots, not only K = inf.
            np.testing.assert_allclose(phi[perm], phi, atol=1e-12)
            np.testing.assert_allclose(
                inputs.monomials[perm, a], -inputs.monomials[:, a], atol=1e-12
            )
            assert ff.fibre_error(phi, inputs.monomials[:, a]) == pytest.approx(1.0)
            checked += 1
    assert checked > 10


def test_haar_features_separate_inputs_beyond_the_linear_span() -> None:
    inputs = ff.eigenstate_inputs(N, "xz")
    w = ff.input_weights(N, "xz")
    circuit = _circuit("haar", 0.3, seed=15)
    c = ff.feature_matrix(ff.enumerate_records(circuit).effects)
    phi = inputs.monomials @ c.T
    assert ff.n_distinct_features(phi) == len(phi)  # injective on the ensemble
    vis = np.flatnonzero(ff.visible_mask(N, "xz"))[1:]
    eperp = []
    for a in vis:
        beta = np.zeros(4**N)
        beta[a] = 1.0
        eperp.append(ff.population_error(c, w, beta))
        assert ff.fibre_error(phi, inputs.monomials[:, a]) == pytest.approx(
            0.0, abs=1e-12
        )
    assert max(eperp) > 0.05  # the linear readout misses part of it


def test_ridge_on_histogram_equals_ridge_on_all_parities() -> None:
    circuit = _circuit("haar", 0.3, seed=16)
    records = ff.enumerate_records(circuit)
    c = ff.feature_matrix(records.effects)
    w = ff.input_weights(N, "xz")
    beta = _random_target(N, "xz", np.random.default_rng(17))
    par = ff.parity_features(c, records.bits, None)
    lam = 1e-3
    # The Walsh matrix H has H H^T = 2^{n_bits}, so lambda scales by 2^{n_bits}.
    scale = 2.0 ** records.bits.shape[1]
    assert ff.population_error(par, w, beta, lam * scale) == pytest.approx(
        ff.population_error(c, w, beta, lam), abs=1e-9
    )


def test_p_one_is_commutative_for_every_circuit() -> None:
    for kind in ("haar", "clifford"):
        circuit = ff.with_mask(_circuit(kind, 0.0, seed=18), np.ones((DEPTH, N), bool))
        c = ff.feature_matrix(ff.enumerate_records(circuit).effects)
        assert ff.span_rank(c) == 2**N
        assert ff.max_commutator(c) < 1e-9


def test_pythagoras_when_the_local_readout_sees_only_invisible_strings() -> None:
    # Regression: N = 4, p = 0, a Clifford circuit whose single-bit features
    # carry only Y strings (all invisible). Their span is then {0} and the
    # round-off directions must not be kept.
    n = 4
    rng = np.random.default_rng(1004)
    haar = ff.draw_circuit(n, n, 0.0, "haar", rng)
    clif = ff.with_mask(ff.draw_circuit(n, n, 0.0, "clifford", rng), haar.mask)
    records = ff.enumerate_records(clif)
    c = ff.feature_matrix(records.effects)
    w = ff.input_weights(n, "xz")
    beta = np.zeros(4**n)
    beta[3 * 4 ** (n - 3)] = 1.0  # Z on site 2
    c1 = ff.parity_features(c, records.bits, 1)
    assert ff.span_rank(c1, ff.visible_mask(n, "xz")) == 0
    split = ff.two_filter_split(c1, c, w, beta)
    assert split.cross == pytest.approx(0.0, abs=1e-12)
    assert split.total == pytest.approx(1.0, abs=1e-12)
