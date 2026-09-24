"""QRC environment template."""

from qrc_env_template.pipeline import resolve_pipeline, run_pipeline


def hello() -> str:
    return "Hello from qrc-env-template!"


__all__ = ["hello", "resolve_pipeline", "run_pipeline"]
