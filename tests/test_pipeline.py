from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from QELM.evaluation import evaluate
from QELM.models import KitaevModel, TransverseFieldIsingModel
from QELM.pipeline import resolve_pipeline, run_pipeline
from QELM.training import Readout, RidgeRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"


def setup_function() -> None:
    GlobalHydra.instance().clear()


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_dispatcher_resolves_config_to_modules() -> None:
    ising = resolve_pipeline(
        _compose(["sim_type=ising", "training=ridge", "evaluation=default"])
    )
    assert ising.sim_type == "ising"
    assert ising.model is TransverseFieldIsingModel
    assert ising.trainer is RidgeRegression
    assert ising.evaluator is evaluate

    kitaev = resolve_pipeline(_compose(["sim_type=kitaev", "training=readout"]))
    assert kitaev.model is KitaevModel
    assert kitaev.trainer is Readout
    assert kitaev.evaluator is evaluate


def test_run_pipeline_uses_cli_overrides(tmp_path: Path) -> None:
    report = run_pipeline(
        _compose(
            [
                "sim_type=ising",
                "ising.N=12",
                "test.dt=0.05",
                "training=readout",
                f"checkpoint.dir={tmp_path}",
                "checkpoint.use_uuid=false",
                "checkpoint.filename=checkpoint.npz",
            ]
        )
    )
    assert "model=ising" in report
    assert "training=readout" in report
    assert "dt=0.05" in report
    assert "status=scaffold" in report
    checkpoint = tmp_path / "checkpoint.npz"
    assert checkpoint.is_file()
    assert f"checkpoint={checkpoint}" in report


def test_run_pipeline_kitaev_scaffold(tmp_path: Path) -> None:
    report = run_pipeline(_compose(["sim_type=kitaev", f"checkpoint.dir={tmp_path}"]))
    assert "model=kitaev" in report
    assert (tmp_path / "checkpoint.npz").is_file()
