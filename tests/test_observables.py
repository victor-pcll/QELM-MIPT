"""Tests des observables simples <O_i> et doubles <A_i B_j>."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.training.readout import DoubleObservable, SingleObservable


def _product_state(single: np.ndarray, N: int) -> np.ndarray:
    """Densité de |s>^{(x) N} pour un état à un qubit ``single``."""
    psi = single
    for _ in range(N - 1):
        psi = np.kron(psi, single)
    return np.outer(psi, psi.conj())


ZERO = np.array([1, 0], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_feature_counts(N: int) -> None:
    """N features simples et N (N - 1) / 2 corrélateurs."""
    rho = _product_state(ZERO, N)
    single = SingleObservable({"observable": "Z"}).expectation(rho)
    double = DoubleObservable({"observable_1": "Z", "observable_2": "Z"}).expectation(
        rho
    )
    assert single.shape == (N,)
    assert double.shape == (N * (N - 1) // 2,)


def test_known_values_on_all_zero_state() -> None:
    """|000> : <Z_i> = 1, <X_i> = 0, <Z_i Z_j> = 1."""
    rho = _product_state(ZERO, 3)
    np.testing.assert_allclose(
        SingleObservable({"observable": "Z"}).expectation(rho), 1
    )
    np.testing.assert_allclose(
        SingleObservable({"observable": "X"}).expectation(rho), 0, atol=1e-12
    )
    zz = DoubleObservable({"observable_1": "Z", "observable_2": "Z"})
    np.testing.assert_allclose(zz.expectation(rho), 1)


def test_known_values_on_plus_state() -> None:
    """|+++> : <X_i> = 1, <Z_i> = 0, <X_i X_j> = 1."""
    rho = _product_state(PLUS, 3)
    np.testing.assert_allclose(
        SingleObservable({"observable": "X"}).expectation(rho), 1
    )
    np.testing.assert_allclose(
        SingleObservable({"observable": "Z"}).expectation(rho), 0, atol=1e-12
    )
    xx = DoubleObservable({"observable_1": "X", "observable_2": "X"})
    np.testing.assert_allclose(xx.expectation(rho), 1)


def test_bell_state_has_correlations_without_magnetisation() -> None:
    """(|00> + |11>)/sqrt(2) : <Z_i> = 0 mais <Z_0 Z_1> = 1, <X_0 X_1> = 1."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    np.testing.assert_allclose(
        SingleObservable({"observable": "Z"}).expectation(rho), 0, atol=1e-12
    )
    for label in ("Z", "X"):
        obs = DoubleObservable({"observable_1": label, "observable_2": label})
        np.testing.assert_allclose(obs.expectation(rho), [1])


def test_pair_order_is_lexicographic() -> None:
    """Pour N = 3 l'ordre est (0,1), (0,2), (1,2) : seul le qubit 2 est flippé."""
    psi = np.zeros(8, dtype=complex)
    psi[0b001] = 1  # |001> : qubit 2 dans |1>
    rho = np.outer(psi, psi.conj())
    zz = DoubleObservable({"observable_1": "Z", "observable_2": "Z"})
    np.testing.assert_allclose(zz.expectation(rho), [1, -1, -1])


def test_features_are_bounded_for_random_state() -> None:
    """Les valeurs moyennes de Paulis sont réelles et dans [-1, 1]."""
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    for obs in (
        SingleObservable({"observable": "Y"}),
        DoubleObservable({"observable_1": "X", "observable_2": "Z"}),
    ):
        values = obs.expectation(rho)
        assert values.dtype == np.float64
        assert np.all(np.abs(values) <= 1 + 1e-12)


def test_unknown_label_is_rejected() -> None:
    """Une étiquette qui n'est pas X, Y ou Z lève ValueError."""
    with pytest.raises(ValueError, match="Unknown observable"):
        SingleObservable({"observable": "W"})


def test_rho_must_be_a_qubit_density_matrix() -> None:
    """Une matrice 3x3 n'est pas un état de qubits."""
    with pytest.raises(ValueError, match="2\\*\\*N"):
        SingleObservable({"observable": "Z"}).expectation(np.eye(3))


def test_from_config_reads_its_block() -> None:
    """from_config lit cfg[name], comme les modèles."""
    cfg = OmegaConf.create(
        {
            "single_observable": {"observable": "X"},
            "double_observable": {"observable_1": "Z", "observable_2": "X"},
        }
    )
    assert SingleObservable.from_config(cfg).observables == ["X"]
    assert DoubleObservable.from_config(cfg).pairs == [("Z", "X")]


# --- Blocs absents ou multiples (issue #24) ---


def test_full_config_has_both_blocks() -> None:
    """La config composée contient les deux blocs : <Z_i> et <Z_i Z_j>."""
    from QELM import load_config

    cfg = load_config()
    single = SingleObservable.from_config(cfg)
    double = DoubleObservable.from_config(cfg)
    assert single.observables == ["Z"]
    assert double.pairs == [("Z", "Z")]
    rho = _product_state(ZERO, 3)
    np.testing.assert_allclose(single.expectation(rho), [1, 1, 1])
    np.testing.assert_allclose(double.expectation(rho), [1, 1, 1])


@pytest.mark.parametrize(
    "cfg",
    [
        {"seed": 0},
        {"single_observable": None, "double_observable": None},
        {"single_observable": {}, "double_observable": {}},
        {"single_observable": {"observables": []}, "double_observable": {"pairs": []}},
    ],
)
def test_missing_or_empty_block_gives_no_features(cfg: dict) -> None:
    """Bloc absent ou vide : aucune observable, aucun ConfigKeyError."""
    conf = OmegaConf.create(cfg)
    OmegaConf.set_struct(conf, True)
    rho = _product_state(ZERO, 3)
    single = SingleObservable.from_config(conf)
    double = DoubleObservable.from_config(conf)
    assert single.observables == [] and double.pairs == []
    assert single.expectation(rho).shape == (0,)
    assert double.expectation(rho).shape == (0,)


def test_several_single_observables() -> None:
    """[Z, X] donne N features <Z_i> puis N features <X_i>."""
    N = 3
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = _product_state(plus, N)
    features = SingleObservable({"observables": ["Z", "X"]}).expectation(rho)
    assert features.shape == (2 * N,)
    np.testing.assert_allclose(features[:N], 0, atol=1e-12)
    np.testing.assert_allclose(features[N:], 1)


def test_several_double_pairs() -> None:
    """[[Z, Z], [X, X]] donne N(N-1)/2 features par paire, paire par paire."""
    N = 4
    n_pairs = N * (N - 1) // 2
    bell = np.zeros(4, dtype=complex)
    bell[[0, 3]] = 1 / np.sqrt(2)
    rho = np.kron(np.outer(bell, bell.conj()), _product_state(ZERO, 2))
    double = DoubleObservable({"pairs": [["Z", "Z"], ["X", "X"]]})
    features = double.expectation(rho)
    assert features.shape == (2 * n_pairs,)
    zz = DoubleObservable({"pairs": [["Z", "Z"]]}).expectation(rho)
    xx = DoubleObservable({"pairs": [["X", "X"]]}).expectation(rho)
    np.testing.assert_allclose(features, np.concatenate([zz, xx]))
    assert xx[0] == pytest.approx(1.0)  # <X_0 X_1> = 1 sur |00> + |11>


def test_legacy_and_list_forms_add_up() -> None:
    single = SingleObservable({"observable": "Z", "observables": ["X"]})
    assert single.observables == ["Z", "X"]
    double = DoubleObservable(
        {"observable_1": "Z", "observable_2": "Z", "pairs": [["X", "Y"]]}
    )
    assert double.pairs == [("Z", "Z"), ("X", "Y")]


def test_bad_pair_or_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="two labels"):
        DoubleObservable({"pairs": [["Z"]]})
    with pytest.raises(ValueError, match="together"):
        DoubleObservable({"observable_1": "Z"})
    with pytest.raises(ValueError, match="Unknown observable"):
        DoubleObservable({"pairs": [["Z", "W"]]})
    with pytest.raises(ValueError, match="Unknown observable"):
        SingleObservable({"observables": ["Z", "W"]})


# --- n_qubits optionnel dans operators() (issues #46 et #235) ---


@pytest.mark.parametrize("cls", [SingleObservable, DoubleObservable])
def test_operators_uses_config_n_qubits(cls: type) -> None:
    """Sans argument, operators() prend n_qubits dans les paramètres."""
    params = {"n_qubits": 3, "observables": ["Z"], "pairs": [["Z", "Z"]]}
    obs = cls(params)
    assert obs.n_qubits == 3
    ops = obs.operators()
    assert all(op.shape == (8, 8) for op in ops)
    for got, want in zip(ops, obs.operators(3), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("cls", [SingleObservable, DoubleObservable])
def test_operators_explicit_n_qubits_overrides_config(cls: type) -> None:
    """operators(n_qubits=2) l'emporte sur n_qubits=4 de la config."""
    obs = cls({"n_qubits": 4, "observables": ["Z"], "pairs": [["Z", "Z"]]})
    ops = obs.operators(n_qubits=2)
    assert ops and all(op.shape == (4, 4) for op in ops)


@pytest.mark.parametrize("cls", [SingleObservable, DoubleObservable])
def test_operators_without_n_qubits_raises(cls: type) -> None:
    """Ni argument ni n_qubits dans la config : ValueError explicite."""
    obs = cls({"observables": ["Z"], "pairs": [["Z", "Z"]]})
    assert obs.n_qubits is None
    with pytest.raises(ValueError, match="n_qubits was not passed"):
        obs.operators()


def test_full_config_gives_N_total_to_observables() -> None:
    """Les readouts lisent tout l'état : N_total = N + Nr qubits (issue #235)."""
    from QELM import load_config

    cfg = load_config(["N=3"])
    assert SingleObservable.from_config(cfg).n_qubits == 4
    assert DoubleObservable.from_config(cfg).n_qubits == 4
    assert len(SingleObservable.from_config(cfg).operators()) == 4
    assert len(DoubleObservable.from_config(cfg).operators()) == 6
