"""Tests du jeu de données QELM (Phi, y) (issue #19)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.training import (
    ProjectiveMeasurement,
    QELMDataset,
    create_qelm_dataset,
    split_dataset,
)


def _samples(K: int, n_dyn: int, Ns: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.choice([-1.0, 1.0], size=(K, n_dyn))
    X = rng.choice([-1.0, 1.0], size=(K, Ns))
    y = rng.choice([-1.0, 1.0], size=K)
    return M, X, y


@pytest.mark.parametrize(("K", "n_dyn", "Ns"), [(1, 3, 2), (10, 12, 3), (7, 0, 4)])
def test_phi_shape_and_concatenation(K: int, n_dyn: int, Ns: int) -> None:
    """Phi a la forme (K, n_dynamic + Ns) et vaut [M || X]."""
    M, X, y = _samples(K, n_dyn, Ns)
    ds = create_qelm_dataset(M, X, y)
    assert isinstance(ds, QELMDataset)
    assert ds.phi.shape == (K, n_dyn + Ns)
    assert ds.y.shape == (K,)
    assert ds.phi.dtype == np.float64
    np.testing.assert_array_equal(ds.phi[:, :n_dyn], M)
    np.testing.assert_array_equal(ds.phi[:, n_dyn:], X)
    np.testing.assert_array_equal(ds.y, y)


def test_spatio_temporal_trace_is_flattened_step_by_step() -> None:
    """M de forme (K, T, Ns) devient (K, T*Ns), pas 1 puis pas 2..."""
    K, T, Ns = 4, 3, 2
    M = np.arange(K * T * Ns, dtype=float).reshape(K, T, Ns)
    X = np.zeros((K, Ns))
    ds = create_qelm_dataset(M, X, np.ones(K))
    assert ds.phi.shape == (K, T * Ns + Ns)
    np.testing.assert_array_equal(ds.phi[1, :Ns], M[1, 0])
    np.testing.assert_array_equal(ds.phi[1, Ns : 2 * Ns], M[1, 1])


def test_unmeasured_sites_keep_nan_or_are_filled() -> None:
    M = np.array([[1.0, np.nan], [np.nan, -1.0]])
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, -1.0])
    assert np.isnan(create_qelm_dataset(M, X, y).phi).sum() == 2
    filled = create_qelm_dataset(M, X, y, fill_value=0.0).phi
    np.testing.assert_array_equal(filled, [[1.0, 0.0, 1.0], [0.0, -1.0, -1.0]])


def test_inputs_are_not_modified() -> None:
    M = np.array([[np.nan, 1.0]])
    create_qelm_dataset(M, np.ones((1, 1)), np.ones(1), fill_value=0.0)
    assert np.isnan(M[0, 0])


@pytest.mark.parametrize(
    ("M", "X", "y", "match"),
    [
        (np.ones((3, 2)), np.ones((4, 2)), np.ones(3), "mismatch"),
        (np.ones((3, 2)), np.ones((3, 2)), np.ones(2), "mismatch"),
        (np.ones(3), np.ones((3, 2)), np.ones(3), "dynamic_measurements"),
        (np.ones((3, 2)), np.ones(3), np.ones(3), "final_measurements"),
        (np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 1)), "ancilla_outcomes"),
        (np.ones((2, 2)), np.ones((2, 2)), np.array([1.0, np.nan]), "nan"),
    ],
)
def test_bad_shapes_are_rejected(M, X, y, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        create_qelm_dataset(M, X, y)


def test_split_sizes_follow_config_rounding() -> None:
    """Le train a int(train_fraction * K) échantillons, comme samples.train_size."""
    cfg = OmegaConf.create({"samples": {"L": 10, "train_fraction": 0.7}})
    M, X, y = _samples(cfg.samples.L, 2, 2)
    train, test = split_dataset(
        create_qelm_dataset(M, X, y), cfg.samples.train_fraction
    )
    assert train.phi.shape == (int(0.7 * 10), 4)
    assert test.phi.shape == (10 - int(0.7 * 10), 4)
    assert train.y.shape == (7,) and test.y.shape == (3,)


@pytest.mark.parametrize("rng", [None, np.random.default_rng(1)])
def test_split_has_no_leakage_and_keeps_labels(rng) -> None:
    """Chaque échantillon est dans un seul ensemble, avec son propre label."""
    K = 50
    M = np.arange(K, dtype=float)[:, None]  # identifiant unique par ligne
    X = np.zeros((K, 1))
    y = np.where(np.arange(K) % 3 == 0, 1.0, -1.0)
    train, test = split_dataset(create_qelm_dataset(M, X, y), 0.6, rng)
    ids_train = train.phi[:, 0].astype(int)
    ids_test = test.phi[:, 0].astype(int)
    assert set(ids_train).isdisjoint(ids_test)
    assert sorted([*ids_train, *ids_test]) == list(range(K))
    np.testing.assert_array_equal(train.y, y[ids_train])
    np.testing.assert_array_equal(test.y, y[ids_test])


def test_split_is_reproducible_with_seed() -> None:
    ds = create_qelm_dataset(*_samples(20, 3, 2))
    a, _ = split_dataset(ds, 0.5, np.random.default_rng(4))
    b, _ = split_dataset(ds, 0.5, np.random.default_rng(4))
    np.testing.assert_array_equal(a.phi, b.phi)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_bad_train_fraction(fraction: float) -> None:
    ds = create_qelm_dataset(*_samples(4, 1, 1))
    with pytest.raises(ValueError, match="train_fraction"):
        split_dataset(ds, fraction)


def test_dataset_from_projective_measurements() -> None:
    """Bout en bout : Bell entre R (site 0) et S (site 1), p<1 en cours de route.

    Avec |00> + |11>, la mesure finale de S donne toujours le même résultat
    que celle de R, donc y == X sur chaque trajectoire.
    """
    bell = np.zeros(4, dtype=complex)
    bell[[0, 3]] = 1 / np.sqrt(2)
    rho0 = np.outer(bell, bell.conj())
    on_S = ProjectiveMeasurement({"qubits": [1]})
    on_R = ProjectiveMeasurement({"qubits": [0]})
    rng = np.random.default_rng(0)
    M_rows, X_rows, y_rows = [], [], []
    for _ in range(40):
        rho = rho0
        trace = []
        for _step in range(3):
            res = on_S.measure(rho, p=0.3, rng=rng)
            trace.append(res.outcomes)
            rho = res.rho
        final = on_S.measure(rho, rng=rng)
        anc = on_R.measure(final.rho, rng=rng)
        M_rows.append(np.stack(trace))
        X_rows.append(final.outcomes)
        y_rows.append(anc.outcomes[0])
    ds = create_qelm_dataset(np.array(M_rows), np.array(X_rows), np.array(y_rows))
    assert ds.phi.shape == (40, 3 + 1)
    np.testing.assert_array_equal(ds.phi[:, -1], ds.y)
