"""KL / Jensen-Shannon divergences between P(M | Q+) and P(M | Q-) (#270).

The Monte Carlo estimators are checked against the exact sums over every
record of a short chain, where the ``3**(rounds L)`` records can be listed.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import norm

from QELM.evaluation.kl_divergence import (
    BOUNDS,
    REFERENCE_STATISTICS,
    STATISTICS,
    averaged_log_likelihood,
    bloch_entropy,
    circuit_ensemble_statistics,
    circuit_log_likelihood,
    circuit_log_likelihood_ratio,
    divergence_statistics,
    draw_circuit,
    hypothesis_states,
    log_ratio,
    random_gates,
    reference_bloch,
    reference_statistics,
    sample_circuit_records,
    sample_state_records,
    sep_log_likelihood_ratio,
    split_by_charge,
    state_log_likelihood,
)
from QELM.evaluation.learnability import (
    UNMEASURED,
    charge_state,
    default_charges,
    sample_records,
    sep_log_likelihood,
)

L, DEPTH, P = 4, 1, 0.45
CHARGES = default_charges(L)  # (2, 1)
ROUNDS = 2 * DEPTH  # measured after each layer


def all_records(L: int = L, rounds: int = ROUNDS) -> np.ndarray:
    """Every record of ``rounds`` rounds on ``L`` sites, shape (3**(rL), r, L)."""
    values = (UNMEASURED, 0, 1)
    grid = np.array(list(itertools.product(values, repeat=rounds * L)), np.int8)
    return grid.reshape(-1, rounds, L)


def location_log_prob(records: np.ndarray, p: float) -> np.ndarray:
    """``ln p**k (1 - p)**(n - k)`` of the measured positions of each record."""
    k = (records != UNMEASURED).reshape(len(records), -1).sum(axis=1)
    n = records[0].size
    return k * math.log(p) + (n - k) * math.log(1 - p)


def exact_divergences(log_plus: np.ndarray, log_minus: np.ndarray) -> dict:
    """Exact divergences from the log-probabilities of every record."""
    Pp, Pm = np.exp(log_plus), np.exp(log_minus)
    both = (Pp > 0) & (Pm > 0)
    inf_p = Pp[(Pp > 0) & (Pm == 0)].sum()
    inf_m = Pm[(Pm > 0) & (Pp == 0)].sum()
    Sp, Sm = Pp[both].sum(), Pm[both].sum()
    kl_pm = np.sum(Pp[both] / Sp * np.log((Pp[both] / Sp) / (Pm[both] / Sm)))
    kl_mp = np.sum(Pm[both] / Sm * np.log((Pm[both] / Sm) / (Pp[both] / Sp)))
    mix = 0.5 * (Pp + Pm)
    with np.errstate(divide="ignore", invalid="ignore"):
        js = 0.5 * np.sum(np.where(Pp > 0, Pp * np.log(Pp / mix), 0.0)) + 0.5 * np.sum(
            np.where(Pm > 0, Pm * np.log(Pm / mix), 0.0)
        )
    return {
        "infinite_mass_plus": inf_p,
        "infinite_mass_minus": inf_m,
        "kl_plus_minus_finite": kl_pm,
        "kl_minus_plus_finite": kl_mp,
        "js": js,
        "bhattacharyya": np.sum(np.sqrt(Pp * Pm)),
        "bayes_error": 0.5 * np.sum(np.minimum(Pp, Pm)),
    }


@pytest.fixture(scope="module")
def sep_exact() -> dict:
    records = all_records()
    loc = location_log_prob(records, P)
    logs = [sep_log_likelihood(records, q) + loc for q in CHARGES]
    return {"logs": logs, **exact_divergences(*logs)}


def assert_close(stats: dict, exact: dict, n_sigma: float = 5.0) -> None:
    for key, value in exact.items():
        if key == "logs":
            continue
        err = stats[f"{key}_err"]
        assert abs(stats[key] - value) <= n_sigma * err + 1e-9, (
            key,
            stats[key],
            value,
            err,
        )


def test_exact_laws_are_normalized(sep_exact: dict) -> None:
    for log_p in sep_exact["logs"]:
        assert np.exp(log_p).sum() == pytest.approx(1.0, abs=1e-12)
    # Records impossible under one charge only exist, so the KL is infinite.
    assert sep_exact["infinite_mass_plus"] > 0
    assert sep_exact["infinite_mass_minus"] > 0


def test_sep_estimates_match_the_exact_sums(sep_exact: dict) -> None:
    rng = np.random.default_rng(0)
    records = sample_records(L, P, 20000, depth=DEPTH, rng=rng)
    llr = sep_log_likelihood_ratio(records.outcomes, records.charges)
    stats = divergence_statistics(*split_by_charge(llr, records.charge, CHARGES))
    assert_close(stats, sep_exact)
    assert stats["kl_plus_minus"] == math.inf
    assert stats["jeffreys"] == math.inf
    assert stats["bound_le_cam"] == 0.0
    # The empirical accuracy estimates the same Bayes accuracy.
    assert stats["accuracy"] == pytest.approx(
        stats["bayes_accuracy"], abs=5 * stats["accuracy_err"]
    )
    # The bounds bracket the exact Bayes error.
    pe = sep_exact["bayes_error"]
    assert stats["bound_js_lower"] <= pe <= stats["bound_js_upper"]
    assert stats["bound_hellinger_lower"] <= pe <= stats["bound_hellinger_upper"]


def test_entropy_identity(sep_exact: dict) -> None:
    """E[H(P(Q | M))] = ln 2 - JS: the mean entropy of Fig. 2c is ln 2 - I."""
    Pp, Pm = (np.exp(x) for x in sep_exact["logs"])
    mix = 0.5 * (Pp + Pm)
    keep = mix > 0
    post = 0.5 * Pp[keep] / mix[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(post > 0, post * np.log(post), 0) - np.where(
            post < 1, (1 - post) * np.log(1 - post), 0
        )
    assert np.sum(mix[keep] * h) == pytest.approx(math.log(2) - sep_exact["js"])


def test_gaussian_closed_forms() -> None:
    """P+- = N(+-mu, 1): l = 2 mu x, KL = 2 mu**2, BC = exp(-mu**2 / 2)."""
    mu, n = 0.7, 200_000
    rng = np.random.default_rng(1)
    lp = 2 * mu * rng.normal(mu, 1, n)
    lm = 2 * mu * rng.normal(-mu, 1, n)
    s = divergence_statistics(lp, lm)
    for key, exact in (
        ("kl_plus_minus", 2 * mu**2),
        ("kl_minus_plus", 2 * mu**2),
        ("jeffreys", 2 * mu**2),
        ("kl_plus_minus_finite", 2 * mu**2),
        ("bhattacharyya", math.exp(-(mu**2) / 2)),
        ("bayes_error", norm.cdf(-mu)),
        ("accuracy", norm.cdf(mu)),
    ):
        assert s[key] == pytest.approx(exact, abs=5 * s[f"{key}_err"]), key
    assert s["infinite_mass"] == 0.0
    assert s["bound_le_cam"] == pytest.approx(0.25 * math.exp(-s["kl_plus_minus"]))
    assert s["bound_le_cam"] <= s["bayes_error"]
    assert set(s) == {*STATISTICS, *(f"{k}_err" for k in STATISTICS), *BOUNDS}


def test_identical_and_disjoint_laws() -> None:
    zeros = np.zeros(100)
    s = divergence_statistics(zeros, zeros)
    assert s["kl_plus_minus"] == s["js"] == 0.0
    assert s["bayes_error"] == 0.5
    assert s["bhattacharyya"] == 1.0
    assert s["accuracy"] == 0.5
    s = divergence_statistics(np.full(10, np.inf), np.full(10, -np.inf))
    assert s["kl_plus_minus"] == math.inf
    assert s["infinite_mass"] == 1.0
    assert s["js"] == pytest.approx(math.log(2))
    assert s["js_bits"] == pytest.approx(1.0)
    assert s["bayes_error"] == 0.0
    assert s["accuracy"] == 1.0
    assert math.isnan(s["kl_plus_minus_finite"])


def test_finite_part_is_never_negative() -> None:
    """Restricted to the common support, the KL is a divergence."""
    rng = np.random.default_rng(2)
    lp = np.concatenate([rng.normal(-0.3, 1, 300), np.full(200, np.inf)])
    lm = rng.normal(0.3, 1, 500)
    s = divergence_statistics(lp, lm)
    assert np.mean(lp[np.isfinite(lp)]) < 0  # the naive conditional mean
    assert s["kl_plus_minus_finite"] > 0


def test_divergence_statistics_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="both charges"):
        divergence_statistics(np.zeros(0), np.zeros(3))
    with pytest.raises(ValueError, match="nan"):
        divergence_statistics(np.array([np.nan]), np.zeros(3))


@pytest.fixture(scope="module")
def circuit() -> list[np.ndarray]:
    return draw_circuit(L, DEPTH, rng=np.random.default_rng(3))


def test_circuit_laws_are_normalized(circuit: list[np.ndarray]) -> None:
    records = all_records()
    loc = location_log_prob(records, P)
    for q in CHARGES:
        log_p = circuit_log_likelihood(records, circuit, q) + loc
        assert np.exp(log_p).sum() == pytest.approx(1.0, abs=1e-10)


def test_circuit_estimates_match_the_exact_sums(circuit: list[np.ndarray]) -> None:
    records = all_records()
    loc = location_log_prob(records, P)
    exact = exact_divergences(
        *(circuit_log_likelihood(records, circuit, q) + loc for q in CHARGES)
    )
    rng = np.random.default_rng(4)
    outcomes, charge = sample_circuit_records(
        L, P, 20000, circuit, charges=CHARGES, rng=rng
    )
    assert outcomes.shape == (20000, ROUNDS, L)
    assert np.array_equal(np.unique(charge), sorted(CHARGES))
    llr = circuit_log_likelihood_ratio(outcomes, circuit, CHARGES)
    plus, minus = split_by_charge(llr, charge, CHARGES)
    # Records drawn from a charge are possible under it.
    assert not np.isneginf(plus).any() and not np.isposinf(minus).any()
    assert_close(divergence_statistics(plus, minus), exact)


def test_haar_average_of_the_circuit_law_is_the_sep() -> None:
    """Averaged over the gates, P(M | Q, U) is the SEP law P(M | Q)."""
    record = np.array([[[1, UNMEASURED, 0, UNMEASURED], [UNMEASURED, 1, 1, 0]]])
    rng = np.random.default_rng(5)
    for q in CHARGES:
        probs = [
            math.exp(circuit_log_likelihood(record, draw_circuit(L, 1, rng=rng), q)[0])
            for _ in range(3000)
        ]
        sem = np.std(probs) / math.sqrt(len(probs))
        sep = math.exp(sep_log_likelihood(record, q)[0])
        assert np.mean(probs) == pytest.approx(sep, abs=5 * sem)


def test_knowing_the_gates_helps(sep_exact: dict) -> None:
    """I(Q; M | U), averaged over U, is at least I(Q; M) (U independent of Q)."""
    records = all_records()
    loc = location_log_prob(records, P)
    rng = np.random.default_rng(6)
    js = []
    for _ in range(20):
        c = draw_circuit(L, DEPTH, rng=rng)
        logs = [circuit_log_likelihood(records, c, q) + loc for q in CHARGES]
        js.append(exact_divergences(*logs)["js"])
    assert np.mean(js) > sep_exact["js"]


def test_sampling_is_reproducible(circuit: list[np.ndarray]) -> None:
    a = sample_circuit_records(
        L, P, 50, circuit, charges=CHARGES, rng=np.random.default_rng(7)
    )
    b = sample_circuit_records(
        L, P, 50, circuit, charges=CHARGES, rng=np.random.default_rng(7)
    )
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c1 = draw_circuit(L, 2, rng=np.random.default_rng(8))
    c2 = draw_circuit(L, 2, rng=np.random.default_rng(8))
    assert all(np.array_equal(x, y) for x, y in zip(c1, c2, strict=True))


def test_circuit_gates_are_unitary(circuit: list[np.ndarray]) -> None:
    assert len(circuit) == ROUNDS
    for gates in circuit:
        eye = np.einsum("nij,nkj->nik", gates, gates.conj())
        assert np.allclose(eye, np.eye(4))


def test_argument_checks(circuit: list[np.ndarray]) -> None:
    with pytest.raises(ValueError, match="two distinct"):
        sep_log_likelihood_ratio(np.zeros((1, 2, L), np.int8), (1, 1))
    with pytest.raises(ValueError, match="shape"):
        circuit_log_likelihood(
            np.zeros((1, 2, L), np.int8), [circuit[0], circuit[0]], 2
        )
    with pytest.raises(ValueError, match="rounds"):
        circuit_log_likelihood(np.zeros((1, 4, L), np.int8), circuit, 2)
    with pytest.raises(ValueError, match="p must"):
        sample_circuit_records(L, 1.5, 4, circuit, charges=CHARGES)


def test_zero_measurements_carry_no_information() -> None:
    rng = np.random.default_rng(9)
    records = sample_records(L, 0.0, 200, depth=DEPTH, rng=rng)
    llr = sep_log_likelihood_ratio(records.outcomes, records.charges)
    s = divergence_statistics(*split_by_charge(llr, records.charge, CHARGES))
    assert s["js"] == 0.0 and s["bayes_accuracy"] == 0.5


def test_circuit_ensemble_statistics() -> None:
    rng = np.random.default_rng(10)
    per = [
        divergence_statistics(rng.normal(1, 1, 50) + 0.5 * k, rng.normal(-1, 1, 50))
        for k in range(4)
    ]
    out = circuit_ensemble_statistics(per)
    js = np.array([s["js"] for s in per])
    assert out["js"] == pytest.approx(js.mean())
    assert out["js_std"] == pytest.approx(js.std(ddof=1))
    assert out["js_err"] == pytest.approx(js.std(ddof=1) / 2)
    per[0]["kl_plus_minus"] = math.inf
    out = circuit_ensemble_statistics(per)
    assert out["kl_plus_minus"] == math.inf
    assert out["kl_plus_minus_n_infinite"] == 1
    with pytest.raises(ValueError):
        circuit_ensemble_statistics([])


# ---------------------------------------------------------------------------
# Local encoding and Haar U(4) gates (reference qubit, issue #270 follow-up)


@pytest.mark.parametrize("gates", ["u1", "u4"])
def test_averaged_law_is_normalized_and_is_the_haar_average(gates: str) -> None:
    records = all_records()
    loc = location_log_prob(records, P)
    _, states = hypothesis_states(L, "local")
    for psi in states:
        log_p = averaged_log_likelihood(records, psi, gates) + loc
        assert np.exp(log_p).sum() == pytest.approx(1.0, abs=1e-12)
    record = np.array([[[1, UNMEASURED, 0, UNMEASURED], [UNMEASURED, 1, 1, 0]]])
    rng = np.random.default_rng(11)
    for psi in states:
        probs = [
            math.exp(
                state_log_likelihood(
                    record, draw_circuit(L, 1, rng=rng, gates=gates), psi
                )[0]
            )
            for _ in range(3000)
        ]
        sem = np.std(probs) / math.sqrt(len(probs))
        exact = math.exp(averaged_log_likelihood(record, psi, gates)[0])
        assert np.mean(probs) == pytest.approx(exact, abs=5 * sem + 1e-12)


def test_averaged_u1_law_from_charge_states_is_the_sep() -> None:
    records = all_records()[:500]
    for q in CHARGES:
        assert np.allclose(
            averaged_log_likelihood(records, charge_state(L, q), "u1"),
            sep_log_likelihood(records, q),
        )


def test_hypothesis_states() -> None:
    labels, states = hypothesis_states(4, "local")
    assert labels == (0, 1)
    # Neel |0101> with site 2 set to 0 or 1: |0101> -> |0101>, |0111>.
    assert [int(np.argmax(np.abs(s))) for s in states] == [0b0101, 0b0111]
    labels, states = hypothesis_states(4, "local", site=1)
    assert [int(np.argmax(np.abs(s))) for s in states] == [0b0001, 0b0101]
    labels, states = hypothesis_states(4, "charge")
    assert labels == (2, 1)
    assert np.allclose(states[0], charge_state(4, 2))
    with pytest.raises(ValueError, match="encoding"):
        hypothesis_states(4, "global")
    with pytest.raises(ValueError, match="site"):
        hypothesis_states(4, "local", site=4)


def test_u4_gates_are_unitary_and_rejected_when_unknown() -> None:
    gates = random_gates(50, "u4", np.random.default_rng(12))
    eye = np.einsum("nij,nkj->nik", gates, gates.conj())
    assert np.allclose(eye, np.eye(4))
    with pytest.raises(ValueError, match="gates"):
        random_gates(2, "so4", np.random.default_rng(0))
    with pytest.raises(ValueError, match="gates"):
        averaged_log_likelihood(np.zeros((1, 2, L), np.int8), charge_state(L, 2), "x")


def test_local_u4_has_no_infinite_mass_and_matches_exact_sums() -> None:
    """Without a conserved charge, no record is impossible: the KL is finite."""
    records = all_records()
    loc = location_log_prob(records, P)
    _, states = hypothesis_states(L, "local")
    circuit = draw_circuit(L, DEPTH, rng=np.random.default_rng(13), gates="u4")
    logs = [state_log_likelihood(records, circuit, s) + loc for s in states]
    exact = exact_divergences(*logs)
    assert exact["infinite_mass_plus"] == exact["infinite_mass_minus"] == 0.0
    outcomes, label = sample_state_records(
        L, P, 20000, states, circuit=circuit, rng=np.random.default_rng(14)
    )
    llr = log_ratio(*(state_log_likelihood(outcomes, circuit, s) for s in states))
    stats = divergence_statistics(llr[label == 0], llr[label == 1])
    assert math.isfinite(stats["kl_plus_minus"])
    assert stats["bound_le_cam"] > 0
    assert_close(stats, exact)
    # Averaged over the Haar U(4) gates, the two laws are the same.
    averaged = [averaged_log_likelihood(records, s, "u4") for s in states]
    assert np.allclose(averaged[0], averaged[1])


def test_local_u1_keeps_the_infinite_mass() -> None:
    """With U(1) gates the two branches differ by one particle, as for Q."""
    records = all_records()
    loc = location_log_prob(records, P)
    _, states = hypothesis_states(L, "local")
    logs = [averaged_log_likelihood(records, s, "u1") + loc for s in states]
    assert exact_divergences(*logs)["infinite_mass_plus"] > 0


def test_sample_state_records_needs_a_circuit_or_a_depth() -> None:
    _, states = hypothesis_states(L, "local")
    with pytest.raises(ValueError, match="circuit or a depth"):
        sample_state_records(L, P, 4, states)


# ---------------------------------------------------------------------------
# Bloch vector and entropy of the reference qubit (local encoding)


@pytest.mark.parametrize("gates", ["u1", "u4"])
def test_reference_bloch_is_the_bayes_posterior_and_a_martingale(gates: str) -> None:
    records = all_records()
    loc = location_log_prob(records, P)
    _, states = hypothesis_states(L, "local")
    circuit = draw_circuit(L, DEPTH, rng=np.random.default_rng(15), gates=gates)
    logs = [state_log_likelihood(records, circuit, s) + loc for s in states]
    weight = 0.5 * (np.exp(logs[0]) + np.exp(logs[1]))
    keep = weight > 0
    bloch = reference_bloch(records[keep], circuit, states)
    # z = 2 P(r = 0 | M, U) - 1.
    posterior = 0.5 * np.exp(logs[0][keep]) / weight[keep]
    assert np.allclose(bloch[:, 2], 2 * posterior - 1)
    assert np.all(np.linalg.norm(bloch, axis=1) <= 1 + 1e-12)
    # Averaged over the records, R is still maximally mixed.
    assert np.allclose(weight[keep] @ bloch, 0.0, atol=1e-12)


def test_reference_bloch_without_measurements_is_mixed() -> None:
    _, states = hypothesis_states(L, "local")
    circuit = draw_circuit(L, 2, rng=np.random.default_rng(16), gates="u4")
    outcomes, _ = sample_state_records(
        L, 0.0, 10, states, circuit=circuit, rng=np.random.default_rng(17)
    )
    bloch = reference_bloch(outcomes, circuit, states)
    assert np.allclose(bloch, 0.0)
    stats = reference_statistics(bloch)
    assert stats["S_R"] == pytest.approx(1.0)
    assert stats["information_R"] == pytest.approx(0.0, abs=1e-12)
    assert stats["z_accuracy"] == pytest.approx(0.5)


def test_bloch_entropy_and_reference_statistics() -> None:
    bloch = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0], [0.6, 0, 0]])
    S = bloch_entropy(bloch)
    h = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
    assert np.allclose(S, [0, 0, 1, h])
    stats = reference_statistics(bloch)
    assert stats["S_R"] == pytest.approx(np.mean(S))
    assert stats["purity_R"] == pytest.approx(np.mean([1, 1, 0.5, 0.68]))
    assert stats["abs_z"] == pytest.approx(0.25)
    assert stats["z_accuracy"] == pytest.approx(0.625)
    assert set(stats) == {
        *REFERENCE_STATISTICS,
        *(f"{k}_err" for k in REFERENCE_STATISTICS),
    }


def test_reference_bloch_checks_its_arguments() -> None:
    _, states = hypothesis_states(L, "local")
    circuit = draw_circuit(L, 1, rng=np.random.default_rng(18))
    with pytest.raises(ValueError, match="two branches"):
        reference_bloch(np.zeros((1, 2, L), np.int8), circuit, states[:1])
    with pytest.raises(ValueError, match="rounds"):
        reference_bloch(np.zeros((1, 4, L), np.int8), circuit, states)
