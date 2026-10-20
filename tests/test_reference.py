"""WebPlotDigitizer reader and curve comparison of QELM.utils.reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from QELM.utils import compare_to_reference, load_wpd_csv


def test_single_dataset_without_header(tmp_path: Path) -> None:
    path = tmp_path / "one.csv"
    path.write_text("0.5,0.2\n0.1,0.9\n0.3, 0.6\n")
    curves = load_wpd_csv(path, default_name="ref")
    x, y = curves["ref"]
    np.testing.assert_allclose(x, [0.1, 0.3, 0.5])
    np.testing.assert_allclose(y, [0.9, 0.6, 0.2])


def test_several_datasets_with_header_and_ragged_columns(tmp_path: Path) -> None:
    path = tmp_path / "all.csv"
    path.write_text(
        "L=6,,L=10,\nX,Y,X,Y\n0.0,1.0,0.0,0.9\n0.5,0.4,0.5,0.3\n,,1.0,0.0\n"
    )
    curves = load_wpd_csv(path)
    assert list(curves) == ["L=6", "L=10"]
    np.testing.assert_allclose(curves["L=6"][0], [0.0, 0.5])
    np.testing.assert_allclose(curves["L=10"][1], [0.9, 0.3, 0.0])


def test_swap_xy(tmp_path: Path) -> None:
    path = tmp_path / "swapped.csv"
    path.write_text("A,\nX,Y\n0.6,0.1\n0.2,0.5\n")
    x, y = load_wpd_csv(path, swap_xy=True)["A"]
    np.testing.assert_allclose(x, [0.1, 0.5])
    np.testing.assert_allclose(y, [0.6, 0.2])


def test_xy_header_only_and_bom(tmp_path: Path) -> None:
    path = tmp_path / "xy.csv"
    path.write_text("﻿X,Y\n1e-1,2E-1\n")
    curves = load_wpd_csv(path)
    np.testing.assert_allclose(curves["dataset"][0], [0.1])


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("X,Y\n")
    with pytest.raises(ValueError):
        load_wpd_csv(path)


def test_compare_to_reference() -> None:
    x = np.array([1.0, 0.0, 0.5])
    y = 2 * x
    stats = compare_to_reference(
        x, y, np.array([0.25, 0.75, 2.0]), np.array([0.5, 1.6, 0.0])
    )
    assert stats["n_points"] == 2  # 2.0 lies outside our range
    assert stats["max_abs_diff"] == pytest.approx(0.1)
    assert stats["mean_diff"] == pytest.approx(-0.05)
    empty = compare_to_reference(x, y, np.array([3.0]), np.array([0.0]))
    assert empty["n_points"] == 0 and np.isnan(empty["rms_diff"])
    none = compare_to_reference(np.array([]), np.array([]), x, y)
    assert none["n_points"] == 0 and np.isnan(none["max_abs_diff"])
