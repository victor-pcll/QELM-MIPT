"""Fake sweeps that sit exactly on the digitized reference curves (#149).

Each run's metrics are the reference curve of the paper's size, read here
from the CSV with the transforms written out by hand (columns stored as
(value, x), Fig. 2b accuracy digitized on 0..1 for 0.564..1), plus a known
offset ``DELTA``. The swept values are the reference abscissae themselves,
so a plot script that reads the references correctly finds a deviation of
exactly ``DELTA`` at every reference point; a wrong swap, scale or size
mapping cannot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARN_REF = REPO_ROOT / "experiments" / "learnability" / "reference"
CS_REF = REPO_ROOT / "experiments" / "charge_sharpening" / "reference"
DELTA = 0.01


def read_curve(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the curve ``name`` of an "all datasets" export as (x, value).

    The files hold (value, x) pairs: the columns are swapped here. Points
    sharing an abscissa are averaged, so that the curve is a function.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    names = [cell.strip() for cell in rows[0]]
    k = names.index(name)
    points: dict[float, list[float]] = {}
    for row in rows[2:]:
        cells = row[k : k + 2]
        if len(cells) == 2 and cells[0].strip() and cells[1].strip():
            value, x = float(cells[0]), float(cells[1])
            points.setdefault(x, []).append(value)
    x = np.array(sorted(points))
    return x, np.array([np.mean(points[v]) for v in x])


def learnability_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The paper's 6-site curves of Fig. 2b (accuracy) and 2c (entropy)."""
    p_acc, v_acc = read_curve(LEARN_REF / "barratt2022_fig2b_accuracy.csv", "L = 6")
    p_ent, v_ent = read_curve(LEARN_REF / "barratt2022_fig2c_entropy.csv", "L = 6")
    # The accuracy axis was digitized on 0..1 for 0.564..1.
    return {
        "accuracy": (p_acc, 0.564 + (1.0 - 0.564) * v_acc),
        "entropy_mean": (p_ent, v_ent),
    }


def charge_sharpening_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The paper's 6-site curves of Fig. 7, our N = 7 with the ancilla."""
    return {
        metric: read_curve(CS_REF / f"agrawal2024_fig7_{stem}.csv", "L = 6")
        for metric, stem in (
            ("S_R_mean", "S_R"),
            ("S_R_var", "S_R_var"),
            ("binder", "binder"),
        )
    }


def _write_run(run_dir: Path, config: str, metrics: dict[str, float]) -> None:
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text(config)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def _on_union_grid(
    refs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Every reference abscissa, and each curve interpolated there + DELTA.

    At its own abscissae a curve is exact, which is where it is compared.
    """
    grid = np.unique(np.concatenate([x for x, _ in refs.values()]))
    return grid, {m: np.interp(grid, x, y) + DELTA for m, (x, y) in refs.items()}


def fake_learnability_sweep(sweep_dir: Path) -> Path:
    """One N = 6 run per reference abscissa of Fig. 2b and 2c."""
    grid, values = _on_union_grid(learnability_reference())
    for k, p in enumerate(grid):
        metrics = {
            "accuracy": float(values["accuracy"][k]),
            "entropy_mean": float(values["entropy_mean"][k]),
            "P_corr_binder": 0.1,
            "entropy_binder": 0.1,
        }
        metrics.update({f"{m}_err": 0.001 for m in list(metrics)})
        _write_run(
            sweep_dir / f"N6_seed0_p{k}",
            f"N: 6\nseed: 0\nsamples:\n  L: 100\nlearnability:\n  p: {float(p)!r}\n",
            metrics,
        )
    return sweep_dir


def fake_charge_sharpening_sweep(sweep_dir: Path) -> Path:
    """One N = 6 run (6 sites + the ancilla) per reference abscissa of Fig. 7."""
    grid, values = _on_union_grid(charge_sharpening_reference())
    for k, gamma in enumerate(grid):
        metrics = {m: float(v[k]) for m, v in values.items()}
        metrics.update({f"{m}_err": 0.001 for m in list(metrics)})
        _write_run(
            sweep_dir / f"N6_seed0_g{k}",
            f"N: 6\nNr: 1\nN_total: 7\nseed: 0\nsamples:\n  L: 100\n"
            f"weak_measurement:\n  gamma: {float(gamma)!r}\n",
            metrics,
        )
    return sweep_dir
