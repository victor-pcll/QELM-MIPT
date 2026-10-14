"""QELM: quantum extreme learning machine and the MIPT."""

from QELM.models import TransverseFieldIsingModel
from QELM.pipeline import resolve_pipeline, run_pipeline
from QELM.seed import make_rng, set_seed, spawn_rngs


def hello() -> str:
    """Return a short package greeting.

    Returns:
        The fixed greeting string.
    """
    return "Hello from QELM!"


__all__ = [
    "TransverseFieldIsingModel",
    "hello",
    "make_rng",
    "resolve_pipeline",
    "run_pipeline",
    "set_seed",
    "spawn_rngs",
]
