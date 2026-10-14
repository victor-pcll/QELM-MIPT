"""Désordre : différent entre instances, reproductible par la seed."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.seed import make_rng, spawn_rngs

PARAMS = {"N": 4, "J": 1.0, "h": 0.5}


def _cfg(seed: int = 42):
    return OmegaConf.create({"seed": seed, "TFIM": PARAMS})


def _couplings(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    return [TransverseFieldIsingModel(PARAMS, rng=rng).J_matrix for _ in range(n)]


def test_successive_instances_draw_different_couplings() -> None:
    """Deux hamiltoniens construits avec le même générateur ont des J_ij différents."""
    rng = make_rng(_cfg())
    first, second = _couplings(rng, 2)
    assert not np.array_equal(first, second)


def test_same_seed_reproduces_the_same_sequence() -> None:
    """Même seed, même suite de réalisations J_ij, dans le même ordre."""
    run_a = _couplings(make_rng(_cfg()), 5)
    run_b = _couplings(make_rng(_cfg()), 5)
    for a, b in zip(run_a, run_b, strict=True):
        np.testing.assert_array_equal(a, b)


def test_different_seed_gives_a_different_sequence() -> None:
    first = _couplings(make_rng(_cfg(42)), 1)[0]
    other = _couplings(make_rng(_cfg(43)), 1)[0]
    assert not np.array_equal(first, other)


def test_from_config_draws_from_the_given_generator() -> None:
    cfg = _cfg()
    rng = make_rng(cfg)
    first = TransverseFieldIsingModel.from_config(cfg, rng=rng)
    second = TransverseFieldIsingModel.from_config(cfg, rng=rng)
    assert not np.array_equal(first.J_matrix, second.J_matrix)

    replay = TransverseFieldIsingModel.from_config(cfg, rng=make_rng(cfg))
    np.testing.assert_array_equal(first.J_matrix, replay.J_matrix)


def test_generator_leaves_global_state_untouched() -> None:
    """Le générateur explicite ne consomme pas l'état global de NumPy."""
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    _couplings(make_rng(_cfg()), 3)
    assert np.random.random() == expected


def test_couplings_stay_in_range() -> None:
    for J in _couplings(make_rng(_cfg()), 10):
        assert np.all(np.abs(J) <= PARAMS["J"] / 2)


def test_spawned_realisations_are_distinct_and_reproducible() -> None:
    """Chaque réalisation a son générateur, fixé par (seed, k)."""
    first = [_couplings(rng, 1)[0] for rng in spawn_rngs(_cfg(), 4)]
    again = [_couplings(rng, 1)[0] for rng in spawn_rngs(_cfg(), 4)]
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(first[i], first[j])


def test_spawned_realisation_does_not_depend_on_the_count() -> None:
    """La réalisation k est la même qu'on en tire 3 ou 10."""
    few = spawn_rngs(_cfg(), 3)
    many = spawn_rngs(_cfg(), 10)
    for k in range(3):
        np.testing.assert_array_equal(
            _couplings(few[k], 1)[0], _couplings(many[k], 1)[0]
        )


def test_disorder_average_is_reproducible() -> None:
    """Une moyenne sur le désordre donne le même nombre à seed égale."""

    def mean_ground_energy(seed: int) -> float:
        cfg = _cfg(seed)
        rng = make_rng(cfg)
        energies = [
            np.linalg.eigvalsh(
                TransverseFieldIsingModel.from_config(cfg, rng=rng).hamiltonian()
            )[0]
            for _ in range(8)
        ]
        return float(np.mean(energies))

    assert mean_ground_energy(42) == mean_ground_energy(42)
