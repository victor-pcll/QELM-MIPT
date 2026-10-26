"""Qubit-count convention N / Ns / Nr in the config (issue #88).

N is the total number of qubits, Nr the number of ancillas and Ns = N - Nr
the number of system qubits, all at the top level of the config.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from QELM import check_qubit_counts, load_config
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.training import ProjectiveMeasurement


def test_default_counts() -> None:
    cfg = load_config()
    N, Ns, Nr = check_qubit_counts(cfg)
    assert (N, Ns, Nr) == (int(cfg.N), N - 1, 1)
    assert (cfg.Ns, cfg.Nr) == (Ns, Nr)


@pytest.mark.parametrize("model", ["TFIM", "kitaev"])
def test_every_block_follows_N(model: str) -> None:
    """N=6 reaches the model, the encoding, the readouts and Ns."""
    cfg = load_config([f"models={model}", "N=6"])
    assert check_qubit_counts(cfg) == (6, 5, 1)
    assert cfg[model].N == 6
    assert cfg.ancilla_charge_sector.N == 6
    assert cfg.single_observable.N == cfg.double_observable.N == 6
    assert AncillaChargeSectorEncoding.from_config(cfg).Ns == 5


def test_n_steps_follows_N() -> None:
    assert load_config(["N=6"]).evolution.n_steps == 24


def test_Nr_reaches_the_projective_measurement() -> None:
    cfg = load_config(["N=6", "Nr=2"])
    assert check_qubit_counts(cfg) == (6, 4, 2)
    assert ProjectiveMeasurement.from_config(cfg).Nr == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("TFIM.N=6", "Override N, not TFIM.N"),
        ("ancilla_charge_sector.N=6", "Override N, not ancilla_charge_sector.N"),
        ("projective_measurement.Nr=2", "Override Nr, not projective_measurement.Nr"),
        ("Ns=2", "Set N and Nr, never Ns"),
        ("Nr=-1", "Nr must be >= 0"),
    ],
)
def test_inconsistent_counts_are_rejected(override: str, message: str) -> None:
    """Overriding a copy instead of its source fails at load time."""
    with pytest.raises(ValueError, match=message):
        load_config(["N=4", override])


def test_check_accepts_a_plain_config() -> None:
    """Works on any DictConfig, e.g. one built in a notebook."""
    cfg = OmegaConf.create({"models": "TFIM", "N": 3, "Nr": 1, "Ns": 2})
    assert check_qubit_counts(cfg) == (3, 2, 1)


def test_sub_resolver() -> None:
    cfg = OmegaConf.create({"a": 7, "b": 2, "c": "${sub:${a},${b}}"})
    assert cfg.c == 5
