"""Read curves digitized from a figure with WebPlotDigitizer.

WebPlotDigitizer (https://automeris.io) exports the points of its datasets
as CSV in two layouts:

- one dataset: one ``x,y`` row per point, without a header;
- several datasets ("all datasets" export): a first row with the dataset
  names, each followed by an empty cell (``L=6,,L=8,,``), a second row
  ``X,Y,X,Y,...``, then the points, two columns per dataset. A shorter
  dataset leaves its cells empty at the bottom.

:func:`load_wpd_csv` reads both into ``{name: (x, y)}``, each curve sorted
by ``x``. Used to overlay the reference data of issue #137 on our results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _is_number(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return True


def load_wpd_csv(
    path: str | Path, default_name: str = "dataset"
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the curves of a WebPlotDigitizer CSV export.

    Args:
        path: CSV file.
        default_name: Name of the curves when the file has no header row;
            with several column pairs and no header they are numbered
            ``<default_name>_0``, ``<default_name>_1``, ...

    Returns:
        ``{name: (x, y)}``, with ``x`` sorted in increasing order and ``y``
        in the same order, both ``float64``; empty cells are skipped.

    Raises:
        ValueError: If the file holds no point, or a row has an odd number of
            columns where numbers are expected.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(handle)
            if any(cell.strip() for cell in row)
        ]
    names: list[str] = []
    # Header rows: the dataset names, then X,Y labels (or either alone).
    while rows and not any(_is_number(cell) for cell in rows[0] if cell):
        header = rows.pop(0)
        if not names and not all(c.upper() in ("X", "Y", "") for c in header):
            names = [header[k] for k in range(0, len(header), 2)]
    if not rows:
        raise ValueError(f"No data point in {path}.")
    width = max(len(row) for row in rows)
    width += width % 2
    n_pairs = width // 2
    if not names:
        names = (
            [default_name]
            if n_pairs == 1
            else [f"{default_name}_{k}" for k in range(n_pairs)]
        )
    names = [name or f"{default_name}_{k}" for k, name in enumerate(names)]
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(n_pairs):
        points = []
        for row in rows:
            cells = row[2 * k : 2 * k + 2]
            if len(cells) == 2 and cells[0] and cells[1]:
                points.append((float(cells[0]), float(cells[1])))
        if not points:
            continue
        xy = np.array(sorted(points), dtype=float)
        name = names[k] if k < len(names) else f"{default_name}_{k}"
        curves[name] = (xy[:, 0], xy[:, 1])
    if not curves:
        raise ValueError(f"No data point in {path}.")
    return curves


def compare_to_reference(
    x: np.ndarray, y: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray
) -> dict[str, float]:
    """Compare a curve ``(x, y)`` to reference points ``(x_ref, y_ref)``.

    Our curve is interpolated linearly at the reference abscissae inside its
    range (no extrapolation), and the residuals ``y(x_ref) - y_ref`` are
    summarized.

    Args:
        x: Our abscissae, shape ``(n,)``, any order.
        y: Our values, shape ``(n,)``.
        x_ref: Reference abscissae, shape ``(m,)``.
        y_ref: Reference values, shape ``(m,)``.

    Returns:
        ``n_points`` (references inside our range), ``max_abs_diff``,
        ``rms_diff`` and ``mean_diff`` (signed bias); ``nan`` when no
        reference point falls inside our range.
    """
    order = np.argsort(x)
    xs, ys = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    x_ref = np.asarray(x_ref, dtype=float)
    inside = (x_ref >= xs[0]) & (x_ref <= xs[-1])
    if not np.any(inside):
        nan = float("nan")
        return {"n_points": 0, "max_abs_diff": nan, "rms_diff": nan, "mean_diff": nan}
    diff = np.interp(x_ref[inside], xs, ys) - np.asarray(y_ref, dtype=float)[inside]
    return {
        "n_points": int(np.sum(inside)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "mean_diff": float(np.mean(diff)),
    }
