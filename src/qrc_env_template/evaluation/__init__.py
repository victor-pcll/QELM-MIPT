"""Metrics and plot generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from omegaconf import DictConfig

from qrc_env_template.evaluation.metrics import metric_status
from qrc_env_template.evaluation.plots import render_plot

Evaluator = Callable[..., str]


def evaluate(
    *,
    evolution: Mapping[str, Any],
    training_status: str,
    cfg: DictConfig,
) -> str:
    """Summarize a scaffold evaluation. Does not score predictions."""
    names = [str(name) for name in cfg.evaluation.metrics]
    metrics = metric_status(names)
    plot = render_plot(str(cfg.evaluation.plot))
    return (
        f"evaluation={cfg.evaluation.name} {metrics} {plot} "
        f"fit={training_status} evolved={evolution['status']}"
    )


_EVALUATORS: dict[str, Evaluator] = {"default": evaluate}


def get_evaluator(name: str) -> Evaluator:
    """Return the evaluator for an ``evaluation.name`` config value."""
    try:
        return _EVALUATORS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_EVALUATORS))
        raise KeyError(f"Unknown evaluation {name!r}. Known: {known}.") from exc


__all__ = ["evaluate", "get_evaluator", "metric_status", "render_plot"]
