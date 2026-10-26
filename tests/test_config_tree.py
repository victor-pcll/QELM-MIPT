"""The conf/ tree mirrors src/QELM/ and keeps the composed keys stable."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from QELM.config import load_config

CONF = Path(__file__).resolve().parents[1] / "conf"


def test_conf_tree_mirrors_the_package() -> None:
    """Chaque groupe de conf/ correspond à un module de src/QELM/."""
    expected = {
        "models/TFIM.yaml",
        "models/kitaev.yaml",
        "models/haar_circuit.yaml",
        "models/clifford_circuit.yaml",
        "training/measurements/projective.yaml",
        "training/measurements/weak.yaml",
        "training/readout/single.yaml",
        "training/readout/double.yaml",
        "training/regression/linear.yaml",
        "training/regression/ridge.yaml",
        "evaluation/default.yaml",
    }
    found = {
        str(path.relative_to(CONF))
        for path in CONF.rglob("*.yaml")
        # Primary configs: config.yaml, and one per experiment with its own
        # model (charge_sharpening.yaml, issue #137; learnability.yaml, #113;
        # QELM_M_ratio.yaml, #96; double_descent.yaml, #181;
        # QELM_bayes_h.yaml, #194).
        if path.name
        not in (
            "config.yaml",
            "charge_sharpening.yaml",
            "learnability.yaml",
            "QELM_M_ratio.yaml",
            "double_descent.yaml",
            "QELM_bayes_h.yaml",
        )
    }
    assert found == expected
    assert not (CONF / "sim_type").exists()


def test_default_composition_keeps_the_runtime_keys() -> None:
    """Les blocs lus par le code gardent leur nom à la racine de la config."""
    cfg = load_config()
    assert cfg.models == "TFIM"
    assert cfg.training.name == "ridge"
    assert cfg.projective_measurement.p == 1.0
    assert cfg.projective_measurement.qubits is None
    assert list(cfg.single_observable.observables) == ["Z"]
    assert [list(pair) for pair in cfg.double_observable.pairs] == [["Z", "Z"]]
    assert cfg.single_observable.N == cfg.TFIM.N
    assert cfg.double_observable.N == cfg.TFIM.N


@pytest.mark.parametrize(
    ("overrides", "model", "trainer"),
    [
        (["models=kitaev"], "kitaev", "ridge"),
        (["models=haar_circuit"], "haar_circuit", "ridge"),
        (["models=clifford_circuit"], "clifford_circuit", "ridge"),
        (["training/regression=linear"], "TFIM", "linear"),
        (["models=kitaev", "training/regression=linear"], "kitaev", "linear"),
    ],
)
def test_group_overrides_select_the_files(
    overrides: list[str], model: str, trainer: str
) -> None:
    """models=... et training/regression=... choisissent les fichiers."""
    cfg = load_config(overrides)
    assert cfg.models == model
    assert cfg.training.name == trainer
    assert cfg.single_observable.N == cfg[model].N


@pytest.mark.parametrize("override", ["sim_type=TFIM", "training=linear"])
def test_old_group_names_are_rejected(override: str) -> None:
    """Les anciens noms de groupe ne composent plus."""
    with pytest.raises(ConfigCompositionException):
        load_config([override])
