"""p_linear*, the critical p of the ridge readout (issue #227)."""

from __future__ import annotations

import numpy as np
import pytest

from QELM.evaluation.linear_critical import (
    locate_extremum,
    merge_summaries,
    p_linear_table,
    parabola_vertex,
)

P = np.linspace(0.0, 1.0, 21)


def test_parabola_vertex_is_exact_for_a_parabola() -> None:
    # A parabola with its vertex between two grid points, on a non-uniform grid.
    p = np.array([0.0, 0.1, 0.25, 0.5, 0.6, 1.0])
    y = (p - 0.43) ** 2
    i = int(np.argmin(y))
    assert parabola_vertex(p, y, i) == pytest.approx(0.43)


def test_parabola_vertex_on_the_edge_or_flat() -> None:
    y = np.array([0.0, 1.0, 2.0])
    assert parabola_vertex(P[:3], y, 0) == P[0]
    assert parabola_vertex(P[:3], y, 2) == P[2]
    assert parabola_vertex(P[:3], np.ones(3), 1) == P[1]


@pytest.mark.parametrize("kind", ["min", "max"])
def test_locate_extremum_of_a_clean_valley(kind: str) -> None:
    sign = 1.0 if kind == "min" else -1.0
    y = sign * (P - 0.62) ** 2
    found = locate_extremum(P, y, kind=kind)  # type: ignore[arg-type]
    assert found.p_grid == pytest.approx(0.6)
    assert found.p_refined == pytest.approx(0.62)
    assert found.value == pytest.approx(y[12])
    assert not found.at_edge
    # No error bars: no interval, and an infinitely prominent extremum.
    assert found.p_low == found.p_high == found.p_refined
    assert found.prominence == np.inf


def test_locate_extremum_flags_a_monotone_curve() -> None:
    found = locate_extremum(P, -P, err=np.full_like(P, 0.01), kind="min")
    assert found.p_grid == 1.0
    assert found.at_edge
    assert found.prominence == 0.0


def test_prominence_uses_the_closest_end() -> None:
    # Minimum 0 at p = 0.5, ends 1 and 0.3, error bars 0.1: the depth below the
    # lower end, 0.3, is 0.3 / hypot(0.1, 0.1) bars.
    y = np.array([1.0, 0.5, 0.0, 0.2, 0.3])
    p = np.linspace(0.0, 1.0, 5)
    found = locate_extremum(p, y, err=np.full(5, 0.1), kind="min")
    assert found.prominence == pytest.approx(0.3 / np.hypot(0.1, 0.1))


def test_interval_widens_with_the_error_bars_and_is_reproducible() -> None:
    y = (P - 0.5) ** 2
    narrow = locate_extremum(P, y, err=np.full_like(P, 1e-3))
    wide = locate_extremum(P, y, err=np.full_like(P, 3e-2))
    assert narrow.p_low <= 0.5 <= narrow.p_high
    assert wide.p_high - wide.p_low > narrow.p_high - narrow.p_low
    # rng=None is a fixed seed: the same interval twice.
    assert locate_extremum(P, y, err=np.full_like(P, 3e-2)) == wide


def test_locate_extremum_skips_nan_points() -> None:
    y = (P - 0.3) ** 2
    y[6] = np.nan  # p = 0.3
    found = locate_extremum(P, y)
    assert found.p_grid in (0.25, 0.35)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kind": "mid"}, "kind"),
        ({"level": 1.0}, "level"),
        ({"y": np.full(21, np.nan)}, "nan"),
        ({"p": P[::-1]}, "increasing"),
        ({"y": np.zeros(3)}, "shape"),
    ],
)
def test_locate_extremum_rejects_bad_input(kwargs: dict, message: str) -> None:
    args = {"p": P, "y": (P - 0.5) ** 2} | kwargs
    with pytest.raises(ValueError, match=message):
        locate_extremum(**args)


def _summary() -> list[dict]:
    """Two sizes, one field: valleys at 0.6 (accuracy, 1 - R2) and B falling."""
    rows = []
    for N, shift in ((8, 0.0), (10, -0.1)):
        for p in P:
            rows.append(
                {
                    "N": N,
                    "h": 1.0,
                    "p": float(p),
                    "n_runs": 5,
                    "accuracy_M_Z_mean": 0.6 + (p - 0.6 - shift) ** 2,
                    "accuracy_M_Z_sem": 0.0,
                    "r2_M_Z_mean": 0.2 + (p - 0.6 - shift) ** 2,
                    "r2_M_Z_sem": 0.0,
                }
            )
    return rows


def test_p_linear_table_runs_the_three_estimators() -> None:
    summary = _summary()
    binder = [
        {"N": r["N"], "h": r["h"], "p": r["p"], "binder_M_Z_mean": -r["p"]}
        for r in summary
    ]
    table = p_linear_table(merge_summaries(summary, binder))
    assert [(e["N"], e["estimator"]) for e in table] == [
        (8, "accuracy"),
        (8, "binder"),
        (8, "residual"),
        (10, "accuracy"),
        (10, "binder"),
        (10, "residual"),
    ]
    by = {(e["N"], e["estimator"]): e for e in table}
    assert by[8, "accuracy"]["p_refined"] == pytest.approx(0.6)
    assert by[10, "accuracy"]["p_refined"] == pytest.approx(0.5)
    # 1 - R2 = 0.8 - (p - p0)^2 peaks where R2 dips.
    assert by[10, "residual"]["p_refined"] == pytest.approx(0.5)
    assert by[10, "residual"]["value"] == pytest.approx(0.8)
    assert by[8, "binder"]["at_edge"] and by[8, "binder"]["p_grid"] == 1.0
    assert by[8, "binder"]["metric"] == "binder_M_Z"


def test_p_linear_table_leaves_out_missing_or_nan_metrics() -> None:
    summary = _summary()
    table = p_linear_table(summary)
    assert {e["estimator"] for e in table} == {"accuracy", "residual"}
    for r in summary:
        r["binder_M_Z_mean"] = float("nan")
    assert {e["estimator"] for e in p_linear_table(summary)} == {
        "accuracy",
        "residual",
    }


def test_merge_summaries_requires_the_same_points() -> None:
    a = [{"N": 8, "h": 1.0, "p": 0.0, "x": 1}]
    b = [{"N": 8, "h": 1, "p": 0, "y": 2}]
    assert merge_summaries(a, b) == [{"N": 8, "h": 1, "p": 0, "x": 1, "y": 2}]
    with pytest.raises(ValueError, match="same"):
        merge_summaries(a, [{"N": 10, "h": 1.0, "p": 0.0}])
