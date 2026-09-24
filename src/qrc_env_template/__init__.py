"""QRC environment template."""

from qrc_env_template.pipeline import resolve_pipeline, run_pipeline
from qrc_env_template.seed import set_seed


def hello() -> str:
    """Return a short package greeting.

    Returns:
        The fixed greeting string.
    """
    return "Hello from qrc-env-template!"


__all__ = ["hello", "resolve_pipeline", "run_pipeline", "set_seed"]
