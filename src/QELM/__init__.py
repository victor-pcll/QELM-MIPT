"""QELM: quantum extreme learning machine and the MIPT."""

from QELM.config import check_qubit_counts, load_config
from QELM.models import TransverseFieldIsingModel
from QELM.pipeline import resolve_pipeline, run_pipeline
from QELM.resolvers import register_resolvers
from QELM.seed import set_seed

register_resolvers()


def hello() -> str:
    """Return a short package greeting.

    Returns:
        The fixed greeting string.
    """
    return "Hello from QELM!"


__all__ = [
    "TransverseFieldIsingModel",
    "hello",
    "check_qubit_counts",
    "load_config",
    "register_resolvers",
    "resolve_pipeline",
    "run_pipeline",
    "set_seed",
]
