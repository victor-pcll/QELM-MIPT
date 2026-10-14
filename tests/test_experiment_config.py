"""Experiment parameters in conf/ and the arithmetic resolvers behind them."""

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from QELM import load_config, register_resolvers


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def test_default_experiment_values() -> None:
    cfg = load_config()
    assert cfg.TFIM.J == 1.0
    assert cfg.TFIM.h == pytest.approx(10.0)
    assert cfg.TFIM.R == 10
    assert cfg.test.t == pytest.approx(10.0)
    assert cfg.test.n_steps == 40
    assert cfg.test.dt == pytest.approx(0.25)
    assert cfg.samples.L == 10000
    assert cfg.samples.train_size == 7000
    assert cfg.dynamic_readout.p == 0.01


def test_whole_config_resolves() -> None:
    for sim_type in ("TFIM", "kitaev"):
        resolved = OmegaConf.to_container(
            load_config([f"sim_type={sim_type}"]), resolve=True
        )
        assert isinstance(resolved, dict)


@pytest.mark.parametrize(("J", "N"), [(2.0, 4), (0.5, 6), (1.0, 3)])
def test_derived_values_follow_sources(J: float, N: int) -> None:
    cfg = load_config([f"TFIM.J={J}", f"TFIM.N={N}"])
    assert cfg.TFIM.h == pytest.approx(10.0 * J)
    assert cfg.test.t == pytest.approx(10.0 / J)
    assert cfg.test.n_steps == 4 * N
    assert isinstance(cfg.test.n_steps, int)
    assert cfg.test.dt == pytest.approx((10.0 / J) / (4 * N))
    assert cfg.test.n_steps * cfg.test.dt == pytest.approx(cfg.test.t)


def test_train_size_follows_L() -> None:
    cfg = load_config(["samples.L=1001"])
    assert cfg.samples.train_size == int(0.7 * 1001)
    assert isinstance(cfg.samples.train_size, int)


def test_derived_values_can_be_overridden() -> None:
    cfg = load_config(["TFIM.h=3.0", "test.dt=0.05", "samples.train_size=10"])
    assert cfg.TFIM.h == 3.0
    assert cfg.test.dt == 0.05
    assert cfg.samples.train_size == 10
    assert cfg.test.t == pytest.approx(10.0)


def test_model_receives_resolved_h() -> None:
    from QELM import TransverseFieldIsingModel

    model = TransverseFieldIsingModel.from_config(
        load_config(["TFIM.N=3", "TFIM.J=2.0"])
    )
    assert model.h == pytest.approx(20.0)


def test_kitaev_keeps_fixed_evolution_step() -> None:
    cfg = load_config(["sim_type=kitaev"])
    assert cfg.test.dt == 0.01
    assert cfg.test.n_steps == 100


def test_register_resolvers_is_idempotent() -> None:
    register_resolvers()
    register_resolvers()
    cfg = OmegaConf.create({"a": 3, "b": "${mul:${a},2}", "c": "${div:${a},2}"})
    assert cfg.b == 6
    assert cfg.c == 1.5
    assert OmegaConf.create({"x": "${int:2.9}"}).x == 2
