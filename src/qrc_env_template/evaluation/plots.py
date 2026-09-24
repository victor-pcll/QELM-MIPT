"""Plot generation scaffold. Does not write image files."""

from __future__ import annotations


def render_plot(kind: str) -> str:
    """Record a plot request without drawing it."""
    return f"plot={kind} status=scaffold"
