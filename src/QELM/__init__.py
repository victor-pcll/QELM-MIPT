"""QELM: quantum extreme learning machine and the MIPT."""

from QELM.config import load_config
from QELM.models import TransverseFieldIsingModel
from QELM.pipeline import resolve_pipeline, run_pipeline
from QELM.seed import set_seed


def hello() -> str:
    """Return a short package greeting.

    Returns:
        The fixed greeting string.
    """
    return "Hello from QELM!"


__all__ = [
    "TransverseFieldIsingModel",
    "hello",
    "load_config",
    "resolve_pipeline",
    "run_pipeline",
    "set_seed",
]
