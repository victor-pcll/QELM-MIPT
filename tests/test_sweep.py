"""QELM.utils.sweep: read back the runs of a sweep (issue #94)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from QELM.utils import latest_sweep, load_sweep, new_sweep_dir, summarize


def _write_run(sweep: Path, name: str, N: int, seed: int, p: float, r2: float) -> None:
    run = sweep / name
    (run / ".hydra").mkdir(parents=True)
    cfg = {
        "N": N,
        "seed": seed,
        "samples": {"L": 10},
        "projective_measurement": {"p": p},
    }
    OmegaConf.save(OmegaConf.create(cfg), run / ".hydra" / "config.yaml")
    metrics = {"nmse": 1 - r2, "r2": r2, "n_samples": 10}
    (run / "metrics.json").write_text(json.dumps(metrics))


def test_rows_come_from_the_config_not_the_folder_name(tmp_path: Path) -> None:
    _write_run(tmp_path, "renamed", N=4, seed=1, p=0.5, r2=0.3)
    (row,) = load_sweep(tmp_path)
    assert row == {
        "run": "renamed",
        "N": 4,
        "seed": 1,
        "p": 0.5,
        "L": 10,
        "nmse": pytest.approx(0.7),
        "r2": 0.3,
        "n_samples": 10,
    }


def test_a_run_written_before_issue_235_is_read_with_its_system_size(
    tmp_path: Path,
) -> None:
    """Old runs held N = Ns + Nr and a top-level Ns; the row N is Ns now."""
    for name, cfg in (
        ("old", {"N": 7, "Nr": 1, "Ns": "${int:${sub:${N},${Nr}}}"}),
        ("new", {"N": 6, "Nr": 1, "N_total": "${int:${add:${N},${Nr}}}"}),
    ):
        run = tmp_path / name
        (run / ".hydra").mkdir(parents=True)
        cfg = {**cfg, "seed": 0, "projective_measurement": {"p": 0.5}}
        OmegaConf.save(OmegaConf.create(cfg), run / ".hydra" / "config.yaml")
        (run / "metrics.json").write_text(json.dumps({"r2": 0.5}))
    rows = load_sweep(tmp_path, keys=(("N", "N"), ("N_total", "N_total")))
    assert [(r["run"], r["N"], r["N_total"]) for r in rows] == [
        ("new", 6, 7),
        ("old", 6, 7),
    ]


def test_folders_without_metrics_are_skipped(tmp_path: Path) -> None:
    _write_run(tmp_path, "a", N=4, seed=1, p=0.5, r2=0.3)
    (tmp_path / "unfinished").mkdir()
    assert [r["run"] for r in load_sweep(tmp_path)] == ["a"]


def test_empty_sweep_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        load_sweep(tmp_path)


def test_summarize_averages_the_seeds(tmp_path: Path) -> None:
    for seed, r2 in [(1, 0.2), (2, 0.4), (3, 0.6)]:
        _write_run(tmp_path, f"N4_s{seed}_p0.5", N=4, seed=seed, p=0.5, r2=r2)
    _write_run(tmp_path, "N4_s1_p1.0", N=4, seed=1, p=1.0, r2=0.9)
    first, second = summarize(load_sweep(tmp_path))
    assert (first["N"], first["p"], first["n_runs"]) == (4, 0.5, 3)
    assert first["r2_mean"] == pytest.approx(0.4)
    assert first["r2_std"] == pytest.approx(0.2)
    assert first["r2_sem"] == pytest.approx(0.2 / math.sqrt(3))
    assert (second["p"], second["n_runs"]) == (1.0, 1)
    assert math.isnan(second["r2_std"])  # one run: no spread


def test_latest_sweep(tmp_path: Path) -> None:
    for stamp in ["2026-10-19_23-00-00", "2026-10-20_01-00-00", "2026-10-19_09-00-00"]:
        (tmp_path / stamp).mkdir()
    assert latest_sweep(tmp_path).name == "2026-10-20_01-00-00"
    with pytest.raises(FileNotFoundError):
        latest_sweep(tmp_path / "2026-10-19_09-00-00")


def test_new_sweep_dir_is_new_and_found_by_latest_sweep(tmp_path: Path) -> None:
    """Two sweeps started in the same second never share a folder (#151)."""
    first = new_sweep_dir(tmp_path, "learnability_fig2b")
    second = new_sweep_dir(tmp_path, "learnability_fig2b")
    other = new_sweep_dir(tmp_path, "learnability_fig2c")
    assert first != second and first.is_dir() and second.is_dir()
    assert first.parent == second.parent == tmp_path / "learnability_fig2b"
    assert other.parent == tmp_path / "learnability_fig2c"
    assert not any(other.iterdir())
    assert latest_sweep(tmp_path / "learnability_fig2b") in (first, second)
    assert latest_sweep(tmp_path / "learnability_fig2c") == other


@pytest.mark.parametrize("name", ["", "a/b"])
def test_new_sweep_dir_rejects_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="plain name"):
        new_sweep_dir(tmp_path, name)
