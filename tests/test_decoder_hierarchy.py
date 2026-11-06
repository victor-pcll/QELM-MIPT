"""Three decoders of the charge on the same U(1) trajectories (issue #307)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config
from QELM.evaluation.charge_variance import sample_trajectories
from QELM.evaluation.decoder_hierarchy import (
    bayes_share,
    draw_labels,
    expected_accuracy,
    guess_from_posterior,
    record_features,
    sector_weight,
)
from QELM.evaluation.learnability import UNMEASURED, charge_state, sep_posterior

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "decoder_hierarchy"
SCRIPT = FOLDER / "decoder_hierarchy.py"
BASH_SCRIPT = FOLDER / "decoder_hierarchy.sh"
PLOT_SCRIPT = FOLDER / "plot_decoder_hierarchy.py"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/decoder_hierarchy, not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


exp = _load(SCRIPT)
plot = _load(PLOT_SCRIPT)


# --- helpers ----------------------------------------------------------------


def test_sector_weight_of_known_states() -> None:
    L = 4
    psi = charge_state(L, 2) * np.sqrt(0.3) + charge_state(L, 1) * np.sqrt(0.7)
    probs = np.abs(psi[None, :]) ** 2
    assert sector_weight(probs, 2)[0] == pytest.approx(0.3)
    assert sector_weight(probs, 1)[0] == pytest.approx(0.7)
    assert sector_weight(probs, 0)[0] == pytest.approx(0.0)


def test_record_features_encoding() -> None:
    m = np.array([[[0, 1], [UNMEASURED, 0]]], dtype=np.int8)
    assert record_features(m).tolist() == [[1.0, -1.0, 0.0, 1.0]]


def test_expected_accuracy_and_guess() -> None:
    w = np.array([0.9, 0.2, 0.5])
    g = guess_from_posterior(w)
    assert g.tolist() == [1.0, 0.0, 0.5]
    assert expected_accuracy(w, g).tolist() == pytest.approx([0.9, 0.8, 0.5])
    # The Bayes guess is the best guess of every trajectory.
    for other in (np.zeros(3), np.ones(3), np.full(3, 0.5)):
        assert np.all(expected_accuracy(w, g) >= expected_accuracy(w, other) - 1e-12)


def test_draw_labels_follow_w() -> None:
    rng = np.random.default_rng(0)
    w = np.full(20000, 0.3)
    assert draw_labels(w, rng).mean() == pytest.approx(0.3, abs=0.01)
    assert draw_labels(np.array([0.0, 1.0]), rng).tolist() == [0, 1]


def test_bayes_share() -> None:
    g = bayes_share(np.array([0.75, 0.5, 0.6]), np.array([1.0, 0.5, 0.7]))
    assert g[0] == pytest.approx(0.5)
    assert np.isnan(g[1])
    assert g[2] == pytest.approx(0.5)


def test_bayes_beats_sep_on_the_same_trajectories() -> None:
    """Knowing the gates can only help: Bayes >= SEP in expected accuracy."""
    tr = sample_trajectories(6, 0.15, 400, rng=np.random.default_rng(3))
    w = sector_weight(tr.probs, tr.charges[0])
    sep = sep_posterior(tr.outcomes, tr.charges)[:, 0]
    bayes = expected_accuracy(w, guess_from_posterior(w)).mean()
    gate_blind = expected_accuracy(w, guess_from_posterior(sep)).mean()
    assert bayes >= gate_blind
    assert bayes > 0.6


# --- script -------------------------------------------------------------------


def _cfg(*extra: str) -> object:
    return load_config(
        [
            "N=4",
            "task.p_values=[0.0,0.2,1.0]",
            "task.samples_per_p=120",
            "task.batch=50",
            *extra,
        ],
        config_name="decoder_hierarchy",
    )


def test_run_metrics() -> None:
    metrics = exp.run(_cfg())
    assert metrics["N"] == 4 and metrics["depth"] == 4
    assert metrics["max_leakage"] < 1e-12
    acc = metrics["accuracy"]
    assert set(acc) == set(exp.DECODERS)
    for d in exp.DECODERS:
        # p = 0: the record is empty, every decoder is at chance.
        assert acc[d][0] == pytest.approx(0.5)
    # p = 1: the first round reveals the charge.
    assert acc["bayes"][2] == pytest.approx(1.0)
    assert acc["sep"][2] == pytest.approx(1.0)
    for d in exp.DECODERS:
        assert acc[d][1] <= acc["bayes"][1] + 1e-12
    assert np.isnan(metrics["share"]["sep"][0])
    again = exp.run(_cfg())
    assert json.dumps(again) == json.dumps(metrics)


def test_run_refuses_unknown_decoder() -> None:
    with pytest.raises(ValueError, match="Unknown decoders"):
        exp.run(_cfg("task.decoders=[bayes,knn]"))


def test_ridge_guess_degenerate() -> None:
    X = np.zeros((4, 2))
    assert exp.ridge_guess(X, np.array([1, 0, 1, 0]), X).tolist() == [0.5] * 4
    X = np.arange(8.0).reshape(4, 2)
    assert exp.ridge_guess(X, np.ones(4), X).tolist() == [0.5] * 4


# --- plots and launcher -------------------------------------------------------


def test_crossing() -> None:
    p = np.array([0.0, 0.1, 0.2, 0.3])
    d = np.array([0.0, -0.1, 0.1, 0.2])
    c = plot.crossing(p, d, np.full(4, 0.01))
    assert c is not None and c["p"] == pytest.approx(0.15) and c["resolved"]
    c = plot.crossing(p, d, np.full(4, 0.2))
    assert c is not None and not c["resolved"]
    assert plot.crossing(p, np.ones(4), np.full(4, 0.1)) is None


def test_launcher_and_plots(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "OUT_DIR": str(tmp_path),
        "N_VALUES": "4 6",
        "SEEDS": "1 2",
        "PER_P": "60",
        "P_VALUES": "[0.0,0.1,0.3]",
        "MAX_JOBS": "2",
    }
    subprocess.run(["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True)
    runs = sorted(p.parent.name for p in tmp_path.glob("*/metrics.json"))
    assert runs == ["N4_seed1", "N4_seed2", "N6_seed1", "N6_seed2"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["sizes"]) == {"4", "6"}
    assert set(summary["crossings"]) == set(exp.DECODERS)
    for name in ("decoder_hierarchy", "crossings"):
        assert (tmp_path / f"{name}.png").is_file()
    out = subprocess.run(
        ["bash", str(BASH_SCRIPT)], check=True, env=env, capture_output=True, text=True
    )
    assert out.stdout.count("Skip") == 4


def test_launcher_refuses_bad_max_jobs(tmp_path: Path) -> None:
    env = {**os.environ, "OUT_DIR": str(tmp_path), "MAX_JOBS": "0"}
    result = subprocess.run(["bash", str(BASH_SCRIPT)], env=env, capture_output=True)
    assert result.returncode != 0
