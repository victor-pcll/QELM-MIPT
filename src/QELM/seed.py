"""Seed NumPy from the Hydra config.

Two kinds of randomness come from ``cfg.seed``:

- ``set_seed`` seeds NumPy's global generator, for code that still calls
  ``np.random.*`` directly.
- ``make_rng`` and ``spawn_rngs`` build explicit ``numpy.random.Generator``
  objects. Disorder realisations draw from these, so each new Hamiltonian
  gets new couplings while the whole run stays reproducible from the seed.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def set_seed(cfg: DictConfig) -> int:
    """Seed NumPy's global generator from ``cfg.seed``.

    Call this once at startup, before any stochastic work. The default is
    ``seed: 42`` in ``conf/config.yaml``. PyTorch and JAX are not
    dependencies, so only NumPy is seeded.

    Args:
        cfg: Composed Hydra config. ``cfg.seed`` is a scalar integer.

    Returns:
        The seed applied to ``numpy.random``.
    """
    seed = int(cfg.seed)
    np.random.seed(seed)
    return seed


def make_rng(cfg: DictConfig) -> np.random.Generator:
    """Return one generator for the whole experiment, seeded by ``cfg.seed``.

    Pass the same generator to every model built in the run. Each model
    draws its disorder from it and moves it forward, so successive models
    get different couplings, and the k-th model of a run with the same seed
    always gets the same couplings. The global ``np.random`` state is not
    touched.

    Args:
        cfg: Composed Hydra config. ``cfg.seed`` is a scalar integer.

    Returns:
        A fresh ``numpy.random.Generator``.
    """
    return np.random.default_rng(int(cfg.seed))


def spawn_rngs(cfg: DictConfig, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators, one per disorder realisation.

    The generators come from ``numpy.random.SeedSequence(cfg.seed).spawn``.
    Realisation ``k`` depends only on ``cfg.seed`` and ``k``: it does not
    change if other realisations draw more numbers, run in another order,
    or run in parallel, and ``spawn_rngs(cfg, 10)[k]`` matches
    ``spawn_rngs(cfg, 100)[k]``.

    Args:
        cfg: Composed Hydra config. ``cfg.seed`` is a scalar integer.
        n: Scalar number of realisations.

    Returns:
        A list of ``n`` generators.
    """
    children = np.random.SeedSequence(int(cfg.seed)).spawn(n)
    return [np.random.default_rng(child) for child in children]
