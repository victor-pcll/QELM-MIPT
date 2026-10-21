"""Save a figure with the data it shows, in the output folder of a run.

:func:`save_figure` writes ``<name>.png`` and ``<name>.json`` side by side,
typically in the sweep folder ``outputs/<experiment>/<date_time>/`` that the
launcher created (issue #147). The JSON holds every curve, error bar,
histogram and color map drawn, read back from the figure by :func:`figure_data`, so a
figure can be redrawn or checked without rerunning the sweep.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def _listed(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _errorbar(container: ErrorbarContainer) -> dict[str, Any]:
    line, _, bar_cols = container.lines
    curve: dict[str, Any] = {
        "label": container.get_label(),
        "x": _listed(line.get_xdata()),
        "y": _listed(line.get_ydata()),
    }
    if container.has_yerr and bar_cols:
        # The y bars are the last collection: one segment per point, from
        # y - lower to y + upper.
        column: LineCollection = bar_cols[-1]
        segments = column.get_segments()
        y = np.asarray(curve["y"])
        if len(segments) == y.size:
            # A point whose bar is not drawn (nan) has an empty segment.
            ends = np.array(
                [
                    [s[0][1], s[-1][1]] if len(s) >= 2 else [np.nan, np.nan]
                    for s in segments
                ],
                dtype=float,
            ).reshape(-1, 2)
            curve["yerr_lower"] = _listed(y - ends.min(axis=1))
            curve["yerr_upper"] = _listed(ends.max(axis=1) - y)
    return curve


def figure_data(fig: Figure) -> dict[str, Any]:
    """Return the data drawn on every axis of a figure.

    Args:
        fig: The figure.

    Returns:
        ``{"title": suptitle, "panels": [...]}``, one panel per axis with its
        ``title``, ``xlabel``, ``ylabel``, ``xscale``, ``yscale`` and:

        - ``curves``: every labelled line, ``{label, x, y}``, with
          ``yerr_lower`` and ``yerr_upper`` for error-bar curves;
        - ``histograms``: every bar plot, ``{label, edges, heights}``;
        - ``heatmaps``: every labelled color map (``pcolormesh``),
          ``{label, x_edges, y_edges, values}``, ``values`` of shape
          ``(len(y_edges) - 1, len(x_edges) - 1)``, ``nan`` for a blank
          cell (issue #164).
    """
    suptitle = fig.get_suptitle()
    panels = []
    for ax in fig.axes:
        curves = []
        # Lines of the error-bar curves (data line and caps), read above.
        in_errorbar: set[int] = set()
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                curves.append(_errorbar(container))
                data_line, caplines, _ = container.lines
                in_errorbar.update(id(line) for line in (data_line, *caplines))
        for line in ax.get_lines():
            unlabelled = str(line.get_label()).startswith("_")
            if id(line) in in_errorbar or unlabelled or not isinstance(line, Line2D):
                continue
            curves.append(
                {
                    "label": str(line.get_label()),
                    "x": _listed(line.get_xdata()),
                    "y": _listed(line.get_ydata()),
                }
            )
        histograms = []
        for container in ax.containers:
            if isinstance(container, BarContainer) and container.patches:
                left = [p.get_x() for p in container.patches]
                right = (
                    container.patches[-1].get_x() + container.patches[-1].get_width()
                )
                histograms.append(
                    {
                        "label": str(container.get_label()),
                        "edges": _listed([*left, right]),
                        "heights": _listed([p.get_height() for p in container.patches]),
                    }
                )
        heatmaps = []
        for mesh in ax.collections:
            if isinstance(mesh, QuadMesh) and not str(mesh.get_label()).startswith("_"):
                coords = np.asarray(mesh.get_coordinates(), dtype=float)
                shape = (coords.shape[0] - 1, coords.shape[1] - 1)
                values = np.ma.filled(
                    np.ma.asarray(mesh.get_array(), dtype=float), np.nan
                ).reshape(shape)
                heatmaps.append(
                    {
                        "label": str(mesh.get_label()),
                        "x_edges": _listed(coords[0, :, 0]),
                        "y_edges": _listed(coords[:, 0, 1]),
                        "values": [_listed(row) for row in values],
                    }
                )
        panels.append(
            {
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xscale": ax.get_xscale(),
                "yscale": ax.get_yscale(),
                "curves": curves,
                "histograms": histograms,
                "heatmaps": heatmaps,
            }
        )
    return {"title": suptitle, "panels": panels}


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    name: str,
    data: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``<name>.png`` and its data ``<name>.json`` in ``out_dir``.

    Args:
        fig: The figure, left open (the notebook can still show it).
        out_dir: Destination folder, created if needed: the output folder of
            the sweep the figure comes from.
        name: File name without suffix, a plain name without directory.
        data: Extra entries for the JSON (e.g. the summary table), stored
            under ``"extra"`` next to :func:`figure_data`.

    Returns:
        Path of the PNG file.

    Raises:
        ValueError: If ``name`` is empty, contains a path separator or has a
            suffix.
    """
    if not name or Path(name).name != name or Path(name).suffix:
        raise ValueError(
            f"name must be a plain file name without suffix, got {name!r}."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png)
    content = figure_data(fig)
    if data is not None:
        content["extra"] = dict(data)
    (out_dir / f"{name}.json").write_text(
        json.dumps(content, indent=2, default=_json_default) + "\n"
    )
    return png


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
