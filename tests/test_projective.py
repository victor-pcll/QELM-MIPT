"""Tests des mesures projectives et de la classe Readout commune."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.training import (
    DoubleObservable,
    ProjectiveMeasurement,
    Readout,
    SingleObservable,
    get_readout,
    measure_projective_rho,
)


def _pure(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def _random_rho(N: int, seed: int) -> np.ndarray:
    """État mixte aléatoire de rang 3 sur N qubits."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2**N, 3)) + 1j * rng.normal(size=(2**N, 3))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


ZERO = _pure([1, 0])
ONE = _pure([0, 1])
PLUS = _pure([1, 1])
BELL = _pure([1, 0, 0, 1])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_collapsed_state_is_a_density_matrix(seed: int, qubit: int) -> None:
    """Après effondrement : trace 1, hermitienne, positive."""
    rho = _random_rho(3, seed)
    out, mi = measure_projective_rho(rho, qubit, 3, np.random.default_rng(seed))
    assert mi in (-1, 1)
    np.testing.assert_allclose(np.trace(out), 1, atol=1e-12)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() > -1e-12


@pytest.mark.parametrize("seed", range(3))
def test_collapse_projects_the_measured_qubit(seed: int) -> None:
    """<Z_q> vaut mi après la mesure : mi est la valeur propre de Z obtenue."""
    rho = _random_rho(3, seed)
    out, mi = measure_projective_rho(rho, 1, rng=np.random.default_rng(seed))
    z1 = SingleObservable({"observable": "Z"}).expectation(out)[1]
    np.testing.assert_allclose(z1, mi, atol=1e-12)


def test_remeasuring_gives_the_same_outcome() -> None:
    """Mesure projective idempotente : re-mesurer redonne le même résultat."""
    rng = np.random.default_rng(1)
    rho = _random_rho(2, 1)
    out, mi = measure_projective_rho(rho, 0, rng=rng)
    for _ in range(10):
        out2, mi2 = measure_projective_rho(out, 0, rng=rng)
        assert mi2 == mi
        np.testing.assert_allclose(out2, out, atol=1e-12)


def test_basis_states_are_deterministic() -> None:
    """|0> donne toujours mi=+1 et |1> toujours mi=-1, sans changer l'état."""
    rng = np.random.default_rng(0)
    for rho, expected in ((ZERO, 1), (ONE, -1)):
        for _ in range(50):
            out, mi = measure_projective_rho(rho, 0, rng=rng)
            assert mi == expected
            np.testing.assert_allclose(out, rho)
    psi = np.zeros(8)
    psi[0b010] = 1  # |010>
    rho = _pure(psi)
    readout = ProjectiveMeasurement({})
    for _ in range(20):
        result = readout.measure(rho, rng=rng)
        np.testing.assert_array_equal(result.outcomes, [1, -1, 1])
        np.testing.assert_allclose(result.rho, rho)


def _frequency_of_one(rho: np.ndarray, qubit: int, n: int, seed: int) -> float:
    """Fréquence du résultat |1>, soit mi = -1."""
    rng = np.random.default_rng(seed)
    hits = sum(measure_projective_rho(rho, qubit, rng=rng)[1] == -1 for _ in range(n))
    return hits / n


def test_plus_state_follows_born_rule() -> None:
    """|+> : p1 = 1/2. Avec n = 20000 tirages, écart-type 0.0035 ; tolérance 5 sigma."""
    n = 20000
    sigma = np.sqrt(0.25 / n)
    assert abs(_frequency_of_one(PLUS, 0, n, seed=3) - 0.5) < 5 * sigma


def test_tilted_state_follows_born_rule() -> None:
    """cos(t)|0> + sin(t)|1> : p1 = sin(t)^2."""
    t = 0.4
    rho = _pure([np.cos(t), np.sin(t)])
    p1 = np.sin(t) ** 2
    n = 20000
    sigma = np.sqrt(p1 * (1 - p1) / n)
    assert abs(_frequency_of_one(rho, 0, n, seed=4) - p1) < 5 * sigma


@pytest.mark.parametrize("seed", range(2))
def test_mean_outcome_matches_z_expectation(seed: int) -> None:
    """La moyenne des mi sur n tirages tend vers <Z_i> = Tr(Z_i rho)."""
    rho = _random_rho(2, seed)
    z = SingleObservable({"observable": "Z"}).expectation(rho)
    n = 20000
    rng = np.random.default_rng(seed)
    readout = ProjectiveMeasurement({})
    mean = np.mean([readout.measure(rho, rng=rng).outcomes for _ in range(n)], axis=0)
    # Var(mi) = 1 - <Z>^2 <= 1, donc sigma <= 1 / sqrt(n) ; tolérance 5 sigma.
    np.testing.assert_allclose(mean, z, atol=5 / np.sqrt(n))


def test_bell_state_outcomes_are_correlated() -> None:
    """Bell : chaque qubit 50/50, et les deux résultats toujours égaux."""
    n = 5000
    rng = np.random.default_rng(5)
    readout = ProjectiveMeasurement({})
    outcomes = np.array([readout.measure(BELL, rng=rng).outcomes for _ in range(n)])
    np.testing.assert_array_equal(outcomes[:, 0], outcomes[:, 1])
    frac = np.mean(outcomes[:, 0] == 1)
    assert abs(frac - 0.5) < 5 * np.sqrt(0.25 / n)


def test_bell_collapse_gives_product_state() -> None:
    """Mesurer le qubit 0 d'un Bell laisse |00> ou |11>."""
    out, mi = measure_projective_rho(BELL, 0, rng=np.random.default_rng(0))
    expected = _pure([1, 0, 0, 0]) if mi == 1 else _pure([0, 0, 0, 1])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_same_seed_gives_same_outcomes() -> None:
    """Deux Generator de même seed donnent la même suite de résultats."""
    rho = _random_rho(4, 7)
    readout = ProjectiveMeasurement({})
    a = readout.measure(rho, rng=np.random.default_rng(42))
    b = readout.measure(rho, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a.outcomes, b.outcomes)
    np.testing.assert_array_equal(a.rho, b.rho)


def test_global_rng_is_used_without_generator() -> None:
    """Sans rng, np.random.seed rend les tirages reproductibles."""
    rho = _random_rho(4, 8)
    readout = ProjectiveMeasurement({})
    np.random.seed(123)
    a = readout.measure(rho).outcomes
    np.random.seed(123)
    b = readout.measure(rho).outcomes
    np.testing.assert_array_equal(a, b)


def test_selected_qubits_are_measured_in_order() -> None:
    """``qubits`` choisit les qubits mesurés et l'ordre des valeurs."""
    psi = np.zeros(8)
    psi[0b011] = 1  # |011>
    result = ProjectiveMeasurement({"qubits": [2, 0]}).measure(
        _pure(psi), rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(result.outcomes, [-1, 1])


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="qubit_idx"):
        measure_projective_rho(BELL, 2)
    with pytest.raises(ValueError, match="does not match"):
        measure_projective_rho(BELL, 0, N=3)


def test_observables_are_not_readouts() -> None:
    """Observables (ce qu'on mesure) et readouts (comment on mesure) sont séparés."""
    assert not issubclass(SingleObservable, Readout)
    assert not issubclass(DoubleObservable, Readout)
    assert issubclass(ProjectiveMeasurement, Readout)


def test_registry_and_config() -> None:
    """get_readout et from_config retrouvent chaque type de readout."""
    assert get_readout("projective_measurement") is ProjectiveMeasurement
    for name in ("weak", "single_observable"):
        with pytest.raises(KeyError, match="Unknown readout"):
            get_readout(name)
    cfg = OmegaConf.create({"projective_measurement": {"qubits": [1]}})
    assert ProjectiveMeasurement.from_config(cfg).qubits == [1]


# --- Probabilité de mesure p (issue #16) ---


def test_p_one_is_the_default_and_draws_nothing_extra() -> None:
    """p=1 mesure tout et donne exactement les mêmes tirages qu'avant p."""
    rho = _random_rho(4, 11)
    readout = ProjectiveMeasurement({})
    default = readout.measure(rho, rng=np.random.default_rng(5))
    explicit = readout.measure(rho, p=1.0, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(default.outcomes, explicit.outcomes)
    np.testing.assert_array_equal(default.rho, explicit.rho)
    assert not np.any(np.isnan(default.outcomes))

    # Même suite que des appels directs à measure_projective_rho.
    rng = np.random.default_rng(5)
    out = rho
    expected = []
    for q in range(4):
        out, mi = measure_projective_rho(out, q, rng=rng)
        expected.append(mi)
    np.testing.assert_array_equal(default.outcomes, expected)
    np.testing.assert_array_equal(default.rho, out)


def test_p_zero_measures_nothing() -> None:
    """p=0 : aucun qubit mesuré, rho inchangé, que des nan."""
    rho = _random_rho(3, 12)
    result = ProjectiveMeasurement({}).measure(rho, p=0.0, rng=np.random.default_rng(0))
    assert result.outcomes.shape == (3,)
    assert np.all(np.isnan(result.outcomes))
    np.testing.assert_array_equal(result.rho, rho)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_fraction_of_measured_qubits_matches_p(p: float) -> None:
    """La fraction de qubits mesurés suit p (tolérance 5 sigma)."""
    rho = _random_rho(3, 13)
    readout = ProjectiveMeasurement({})
    rng = np.random.default_rng(14)
    n = 3000
    outcomes = np.array([readout.measure(rho, p=p, rng=rng).outcomes for _ in range(n)])
    measured = ~np.isnan(outcomes)
    sigma = np.sqrt(p * (1 - p) / n)
    for q in range(3):
        assert abs(measured[:, q].mean() - p) < 5 * sigma
    assert set(np.unique(outcomes[measured])) <= {-1.0, 1.0}


def test_unmeasured_qubits_keep_their_state() -> None:
    """Un qubit non mesuré n'est pas effondré ; un qubit mesuré l'est."""
    plus = np.array([1, 1]) / np.sqrt(2)
    rho = _pure(np.kron(np.kron(plus, plus), plus))
    readout = ProjectiveMeasurement({})
    rng = np.random.default_rng(15)
    for _ in range(50):
        result = readout.measure(rho, p=0.5, rng=rng)
        for q, mi in enumerate(result.outcomes):
            X_q = np.eye(1)
            for k in range(3):
                X_q = np.kron(X_q, np.array([[0, 1], [1, 0]]) if k == q else np.eye(2))
            Z_q = np.eye(1)
            for k in range(3):
                Z_q = np.kron(Z_q, np.diag([1, -1]) if k == q else np.eye(2))
            if np.isnan(mi):
                # |+> intact : <X_q> = 1.
                assert np.trace(result.rho @ X_q).real == pytest.approx(1.0)
            else:
                # Effondré sur le résultat : <Z_q> = mi et <X_q> = 0.
                assert np.trace(result.rho @ Z_q).real == pytest.approx(mi)
                assert np.trace(result.rho @ X_q).real == pytest.approx(0.0, abs=1e-12)
        assert np.trace(result.rho).real == pytest.approx(1.0)


def test_p_is_reproducible_with_seed() -> None:
    """Même seed, même p : mêmes qubits mesurés et mêmes résultats."""
    rho = _random_rho(4, 16)
    readout = ProjectiveMeasurement({})
    a = readout.measure(rho, p=0.4, rng=np.random.default_rng(3))
    b = readout.measure(rho, p=0.4, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a.outcomes, b.outcomes)
    np.testing.assert_array_equal(a.rho, b.rho)


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan"), True, "0.5"])
def test_invalid_p_is_rejected(p: object) -> None:
    with pytest.raises(ValueError, match="probability"):
        ProjectiveMeasurement({}).measure(BELL, p=p)  # type: ignore[arg-type]


def test_numpy_scalar_p_is_accepted() -> None:
    result = ProjectiveMeasurement({}).measure(
        BELL, p=np.float64(1.0), rng=np.random.default_rng(0)
    )
    assert not np.any(np.isnan(result.outcomes))


# --- Bloc de config absent (issue #25) ---


def test_from_full_config() -> None:
    """La config composée contient le bloc : tous les qubits sont mesurés."""
    from QELM import load_config

    readout = ProjectiveMeasurement.from_config(load_config())
    assert readout.qubits is None
    result = readout.measure(_random_rho(3, 21), rng=np.random.default_rng(0))
    assert result.outcomes.shape == (3,)


@pytest.mark.parametrize("cfg", [{"seed": 0}, {"projective_measurement": None}])
def test_missing_block_uses_defaults(cfg: dict) -> None:
    """Bloc absent ou null : valeurs par défaut, pas de ConfigKeyError."""
    conf = OmegaConf.create(cfg)
    OmegaConf.set_struct(conf, True)
    readout = ProjectiveMeasurement.from_config(conf)
    assert readout.qubits is None
    assert readout.measure(BELL, rng=np.random.default_rng(1)).outcomes.shape == (2,)


# --- p lu dans la config (issue #47) ---


def test_p_defaults_to_one_without_config() -> None:
    """Sans p ni en argument ni en config : p = 1, tout est mesuré."""
    readout = ProjectiveMeasurement({})
    assert readout.p is None
    result = readout.measure(_random_rho(3, 4), rng=np.random.default_rng(0))
    assert not np.any(np.isnan(result.outcomes))


def test_p_from_config_is_used() -> None:
    """p de la config est utilisé quand measure ne reçoit pas de p."""
    rho = _random_rho(3, 5)
    assert np.all(np.isnan(ProjectiveMeasurement({"p": 0.0}).measure(rho).outcomes))
    from_config = ProjectiveMeasurement({"p": 0.5}).measure(
        rho, rng=np.random.default_rng(8)
    )
    explicit = ProjectiveMeasurement({}).measure(
        rho, p=0.5, rng=np.random.default_rng(8)
    )
    np.testing.assert_array_equal(from_config.outcomes, explicit.outcomes)


def test_explicit_p_overrides_config() -> None:
    """measure(rho, p=...) l'emporte sur p de la config."""
    readout = ProjectiveMeasurement({"p": 0.0})
    result = readout.measure(_random_rho(3, 6), p=1.0, rng=np.random.default_rng(0))
    assert not np.any(np.isnan(result.outcomes))


@pytest.mark.parametrize("p", [-0.1, 1.5, True, "0.5"])
def test_invalid_config_p_is_rejected(p: object) -> None:
    """Un p invalide dans la config est refusé dès la construction."""
    with pytest.raises(ValueError, match="probability"):
        ProjectiveMeasurement({"p": p})


def test_full_config_sets_p() -> None:
    """La config composée lit projective_measurement.p (1.0 par défaut)."""
    from QELM import load_config

    assert ProjectiveMeasurement.from_config(load_config()).p == 1.0
    cfg = load_config(["projective_measurement.p=0.25"])
    assert ProjectiveMeasurement.from_config(cfg).p == 0.25
