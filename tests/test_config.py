import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from QELM.seed import set_seed
from QELM.selection import format_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"
OVERRIDES = ["models=TFIM", "N=12", "evolution.dt=0.05"]


def setup_function() -> None:
    GlobalHydra.instance().clear()


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_default_seed_is_42() -> None:
    assert int(_compose().seed) == 42


def test_set_seed_uses_config_value() -> None:
    assert set_seed(_compose(["seed=7"])) == 7
    sample = np.random.random()
    np.random.seed(7)
    assert sample == np.random.random()


def test_config_loads_and_cli_overrides_apply() -> None:
    default = _compose()
    chosen = _compose(OVERRIDES)

    assert set(chosen.TFIM.keys()) == {"N", "J", "h", "basis", "RUNS"}
    assert set(chosen.evolution.keys()) == {"t", "n_steps", "dt"}
    assert chosen.models == "TFIM"
    assert chosen.TFIM.N == 12
    assert chosen.evolution.dt == 0.05
    assert chosen.TFIM.N != default.TFIM.N
    assert chosen.evolution.dt != default.evolution.dt
    assert chosen.TFIM.J == default.TFIM.J
    # n_steps = 4 N follows the TFIM.N override.
    assert chosen.evolution.n_steps == 4 * chosen.TFIM.N

    text = format_selection(chosen)
    assert "models=TFIM" in text
    assert "N=12" in text
    assert "evolution.dt=0.05" in text


def test_main_cli_override(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            *OVERRIDES,
            f"hydra.run.dir={tmp_path}",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "N=12" in result.stdout
    assert "evolution.dt=0.05" in result.stdout
    assert (tmp_path / "checkpoint.npz").is_file()
