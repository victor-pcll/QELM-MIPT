"""The reference-qubit purification benchmark in experiments/purification/ (#185)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "purification"
SCRIPT = FOLDER / "purification.py"
PLOT = FOLDER / "plot_purification.py"
BASH_SCRIPT = FOLDER / "purification.sh"
FOREIGN_VENV = REPO_ROOT.parent / "other-project" / ".venv"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*overrides: str):  # noqa: ANN202
    return load_config(
        ["samples.L=30", "N=6", *overrides], config_name="purification_gullans_huse"
    )


def test_config_has_the_paper_protocol() -> None:
    cfg = load_config(["N=32"], config_name="purification_gullans_huse")
    assert (cfg.N, cfg.Nr, cfg.N_total) == (32, 1, 33)
    # t0 = 2L without measurement, then 2L measured steps.
    assert cfg.probe.scramble_steps == 64 and cfg.probe.steps == 64
    assert cfg.ancilla_bell_pair.site is None


def test_run_shapes_and_limits() -> None:
    run = _load(SCRIPT).run
    out = run(_cfg("probe.p=0.2"))
    assert out["S_Q"].shape == (30, 24)
    assert set(np.unique(out["S_Q"])) <= {0, 1}
    # Unitary scrambling: R stays maximally mixed up to t0 = 2N.
    assert np.all(out["S_Q"][:, :12] == 1)
    assert len(out["S_Q_t"]) == 24
    assert 0.1 < out["measurement_rate"] < 0.3
    assert out["S_Q_mean"] == pytest.approx(out["S_Q"][:, -1].mean())
    zero = run(_cfg("probe.p=0.0"))
    assert zero["S_Q_mean"] == 1.0 and zero["measurement_rate"] == 0.0
    one = run(_cfg("probe.p=1.0"))
    assert one["S_Q_mean"] == 0.0 and one["measurement_rate"] == 1.0


def test_entropy_decreases_with_p() -> None:
    run = _load(SCRIPT).run
    means = [
        run(_cfg(f"probe.p={p}", "samples.L=200", "N=8"))["S_Q_mean"]
        for p in (0.05, 0.2, 0.5)
    ]
    assert means[0] > means[1] > means[2]


def test_same_seed_same_run() -> None:
    run = _load(SCRIPT).run
    a = run(_cfg("probe.p=0.2", "seed=3"))
    b = run(_cfg("probe.p=0.2", "seed=3"))
    np.testing.assert_array_equal(a["S_Q"], b["S_Q"])
    c = run(_cfg("probe.p=0.2", "seed=4", "samples.L=200"))
    d = run(_cfg("probe.p=0.2", "seed=3", "samples.L=200"))
    assert not np.array_equal(c["S_Q"], d["S_Q"])


def test_odd_N_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        _load(SCRIPT).run(_cfg("N=5"))


def test_bash_script_and_plot(tmp_path: Path) -> None:
    """The launcher runs 2 N x 3 p x 2 seeds, then the figure and fss.json."""
    assert os.access(BASH_SCRIPT, os.X_OK) and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(BASH_SCRIPT)], check=True)
    result = subprocess.run(
        [str(BASH_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OUT_ROOT": str(tmp_path),
            "L": "40",
            "N_VALUES": "4 8",
            "P_VALUES": "0.0 0.2 1.0",
            "SEEDS": "0 1",
            "JOBS": "2",
            "VIRTUAL_ENV": str(FOREIGN_VENV),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Done: 2 x 3 x 2 runs" in result.stdout
    sweep = next((tmp_path / "purification_gullans_huse").iterdir())
    runs = sorted(p.name for p in sweep.iterdir() if p.is_dir())
    assert len(runs) == 12 and "N8_p0.2_seed1" in runs
    metrics = json.loads((sweep / "N8_p1.0_seed0" / "metrics.json").read_text())
    assert metrics["S_Q_mean"] == 0.0 and metrics["steps"] == 16
    resolved = json.loads(
        (sweep / "N8_p1.0_seed0" / "resolved_config.json").read_text()
    )
    assert resolved["probe"]["scramble_steps"] == 16
    assert (sweep / "purification_gullans_huse.png").is_file()
    assert (sweep / "purification_gullans_huse.json").is_file()
    fss = json.loads((sweep / "fss.json").read_text())
    assert fss["paper"] == {"p_c": 0.1598, "nu": 1.3}
    assert set(fss["crossings"]) == {"N4-N8"}

    # The two seeds of a point are pooled: 80 trajectories.
    curves, traces = _load(PLOT).pooled_curves(sweep)
    p, y, e = curves[8]
    np.testing.assert_allclose(p, [0.0, 0.2, 1.0])
    assert y[0] == 1.0 and y[-1] == 0.0
    assert traces[8][0] == 0.2 and traces[8][1][-1] == 4.0
