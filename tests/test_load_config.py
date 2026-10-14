import os
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra

from QELM import TransverseFieldIsingModel, load_config, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def test_overrides_apply_and_defaults_kept() -> None:
    cfg = load_config(["sim_type=TFIM", "TFIM.N=6"])
    assert cfg.sim_type == "TFIM"
    assert cfg.TFIM.N == 6
    assert cfg.TFIM.J == load_config().TFIM.J
    assert int(cfg.seed) == 42


def test_works_from_notebooks_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jupyter starts in notebooks/, where conf/ is not a relative path."""
    monkeypatch.chdir(REPO_ROOT / "notebooks")
    assert not Path("conf").exists()
    cfg = load_config(["TFIM.N=4"])
    assert cfg.TFIM.N == 4
    assert Path.cwd() == REPO_ROOT / "notebooks"


def test_can_be_called_twice_without_clearing() -> None:
    """A re-run notebook cell must not hit 'GlobalHydra is already initialized'."""
    load_config()
    assert load_config(["TFIM.N=3"]).TFIM.N == 3


def test_notebook_two_liner_builds_model() -> None:
    cfg = load_config(["sim_type=TFIM", "TFIM.N=3"])
    set_seed(cfg)
    model = TransverseFieldIsingModel.from_config(cfg)
    assert (model.N, model.dim) == (3, 8)


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(config_dir=tmp_path / "nope")


def test_explicit_config_dir() -> None:
    cfg = load_config(["TFIM.N=5"], config_dir=os.path.join(REPO_ROOT, "conf"))
    assert cfg.TFIM.N == 5
