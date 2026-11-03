"""The experiment of experiments/kl_divergence/ (issue #270)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from QELM import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDER = REPO_ROOT / "experiments" / "kl_divergence"
SCRIPT = FOLDER / "kl_divergence.py"
PLOT = FOLDER / "plot_kl_divergence.py"


def _load(path: Path) -> ModuleType:
    """Import a script of experiments/, which is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(*overrides: str):  # noqa: ANN202
    return load_config(
        [
            "N=4",
            "samples.L=100",
            "kl.batch=40",
            "kl.n_circuits=3",
            "kl.records_per_circuit=20",
            *overrides,
        ],
        config_name="kl_divergence",
    )


def test_config_composes_with_the_issue_parameters() -> None:
    cfg = load_config(config_name="kl_divergence")
    assert (cfg.N, cfg.Nr, cfg.N_total) == (8, 0, 8)
    assert cfg.samples.L == 8000
    assert (cfg.kl.L, cfg.kl.depth) == (8, 8)
    assert (cfg.kl.n_circuits, cfg.kl.records_per_circuit) == (20, 1000)
    cfg = load_config(["N=10"], config_name="kl_divergence")
    assert (cfg.kl.L, cfg.kl.depth) == (10, 10)


def test_run_shapes_and_metrics(tmp_path: Path) -> None:
    script = _load(SCRIPT)
    out = script.run(_cfg("kl.p=0.3"))
    assert out["llr"].shape == out["charge"].shape == (100,)
    assert out["circuit_llr"].shape == out["circuit_charge"].shape == (3, 20)
    assert out["charges"] == (2, 1)
    assert 0.5 < out["sep_bayes_accuracy"] <= 1.0
    assert 0.0 < out["sep_js_bits"] <= 1.0
    assert out["circuit_js_bits_std"] >= 0.0
    script.save(out, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_records"] == 100 and metrics["n_circuits"] == 3
    assert math.isinf(metrics["sep_kl_plus_minus"]) or metrics["sep_kl_plus_minus"] > 0
    assert "circuit_kl_plus_minus_n_infinite" in metrics
    with np.load(tmp_path / "kl_divergence.npz") as data:
        assert data["circuit_llr"].shape == (3, 20)


def test_run_is_reproducible() -> None:
    script = _load(SCRIPT)
    a = script.run(_cfg("kl.p=0.2", "seed=3"))
    b = script.run(_cfg("kl.p=0.2", "seed=3"))
    assert np.array_equal(a["llr"], b["llr"])
    assert np.array_equal(a["circuit_llr"], b["circuit_llr"])


def test_run_without_circuits() -> None:
    out = _load(SCRIPT).run(_cfg("kl.p=0.2", "kl.n_circuits=0"))
    assert out["circuit_llr"].shape == (0, 20)
    assert not any(key.startswith("circuit_js") for key in out)


@pytest.mark.parametrize("key", ["kl.batch=3", "kl.records_per_circuit=5"])
def test_odd_batches_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="even"):
        _load(SCRIPT).run(_cfg(key))


def _fake_sweep(root: Path) -> Path:
    """Run folders of a small sweep, as kl_divergence.sh writes them."""
    script = _load(SCRIPT)
    for N in (4, 6):
        for p in (0.1, 0.2, 0.3):
            cfg = _cfg(f"N={N}", f"kl.p={p}")
            run_dir = root / f"N{N}_seed0_p{p}"
            script.save(script.run(cfg), run_dir)
            (run_dir / ".hydra").mkdir()
            from omegaconf import OmegaConf

            OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    return root


def test_plot_writes_the_figures_and_fits(tmp_path: Path) -> None:
    sweep = _fake_sweep(tmp_path)
    plot = _load(PLOT)
    fits = plot.plot_sweep(sweep, p_c=0.15)
    for name in ("kl_vs_accuracy", "kl_finite_size", "kl_bounds"):
        assert (sweep / f"{name}.png").is_file()
    assert {r["N"] for r in fits["beta"]} == {4, 6}
    assert [r["p"] for r in fits["alpha"]["sep"]] == [0.1, 0.2, 0.3]
    scaling = json.loads((sweep / "scaling.json").read_text())
    assert scaling["p_c"] == 0.15


def test_fit_alpha_recovers_an_exponential() -> None:
    plot = _load(PLOT)
    summary = [
        {"N": N, "p": 0.1, "sep_js_bits_mean": 2.0 ** (-0.3 * N)} for N in (6, 8, 10)
    ]
    (fit,) = plot.fit_alpha(summary)
    assert fit["alpha"] == pytest.approx(0.3)


def test_fit_beta_recovers_a_power_law() -> None:
    plot = _load(PLOT)
    summary = [
        {"N": 8, "p": p, "sep_jeffreys_finite_mean": 2.0 * (p - 0.2) ** 0.7}
        for p in (0.1, 0.25, 0.3, 0.4)
    ]
    (fit,) = plot.fit_beta(summary, 0.2)
    assert fit["beta"] == pytest.approx(0.7)
    assert fit["n_points"] == 3
