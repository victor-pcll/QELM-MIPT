"""Metric names. No scores are computed."""

from __future__ import annotations


def metric_status(names: list[str]) -> str:
    """Report which metrics a run would compute."""
    return "metrics=" + ",".join(names) + " status=scaffold"
