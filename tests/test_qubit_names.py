"""Qubit-count convention N / Nr / N_total in the config (issue #235).

N is the number of system qubits, ancilla excluded, Nr the number of
ancillas and N_total = N + Nr the number of qubits of the simulated state,
all at the top level of the config. Issue #235 replaces the convention of
issue #88, where N counted the ancilla too.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM import check_qubit_counts, load_config
from QELM.config import upgrade_legacy_counts
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.training import ProjectiveMeasurement, SingleObservable


def test_default_counts() -> None:
    cfg = load_config()
    N, Nr, N_total = check_qubit_counts(cfg)
    assert (N, Nr, N_total) == (int(cfg.N), 1, N + 1)
    assert (cfg.Nr, cfg.N_total) == (Nr, N_total)
    assert "Ns" not in cfg


@pytest.mark.parametrize("model", ["TFIM", "kitaev"])
def test_every_block_follows_N(model: str) -> None:
    """N=6 reaches the model and the encoding, N_total = 7 the readouts."""
    cfg = load_config([f"models={model}", "N=6"])
    assert check_qubit_counts(cfg) == (6, 1, 7)
    assert cfg[model].N == 6
    assert cfg.ancilla_charge_sector.N == 6
    assert cfg.single_observable.n_qubits == cfg.double_observable.n_qubits == 7
    emb = AncillaChargeSectorEncoding.from_config(cfg)
    assert (emb.N, emb.N_total, emb.dim) == (6, 7, 2**7)


def test_the_circuit_allocates_N_plus_Nr_qubits() -> None:
    """The state of an N=6 run has 7 qubits; the readout reads all of them."""
    cfg = load_config(["N=6"])
    rho = AncillaChargeSectorEncoding.from_config(cfg).density_matrix()
    assert rho.shape == (2**7, 2**7)
    features = SingleObservable.from_config(cfg).expectation(rho)
    assert features.shape == (7,)
    # The ancilla is maximally entangled with the system: <Z_R> = 0.
    assert features[0] == pytest.approx(0.0, abs=1e-12)
    # The projective measurement collapses the N system qubits only.
    result = ProjectiveMeasurement.from_config(cfg).measure(
        rho, rng=np.random.default_rng(0)
    )
    assert result.outcomes.shape == (6,)


def test_n_steps_follows_N() -> None:
    assert load_config(["N=6"]).evolution.n_steps == 24


def test_Nr_reaches_the_projective_measurement() -> None:
    cfg = load_config(["N=6", "Nr=2"])
    assert check_qubit_counts(cfg) == (6, 2, 8)
    assert ProjectiveMeasurement.from_config(cfg).Nr == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("TFIM.N=6", "Override N, not TFIM.N"),
        ("ancilla_charge_sector.N=6", "Override N, not ancilla_charge_sector.N"),
        ("projective_measurement.Nr=2", "Override Nr, not projective_measurement.Nr"),
        (
            "single_observable.n_qubits=3",
            "Override N_total, not single_observable.n_qubits",
        ),
        ("N_total=2", "Set N and Nr, never N_total"),
        ("Nr=-1", "Nr must be >= 0"),
        ("N=0", "must be >= 1"),
    ],
)
def test_inconsistent_counts_are_rejected(override: str, message: str) -> None:
    """Overriding a copy instead of its source fails at load time."""
    with pytest.raises(ValueError, match=message):
        load_config(["N=4", override])


def test_the_old_Ns_key_is_gone() -> None:
    """Ns no longer exists: an old override fails instead of being ignored."""
    with pytest.raises(Exception, match="Ns"):
        load_config(["Ns=3"])


def test_check_accepts_a_plain_config() -> None:
    """Works on any DictConfig, e.g. one built in a notebook."""
    cfg = OmegaConf.create({"models": "TFIM", "N": 3, "Nr": 1, "N_total": 4})
    counts = check_qubit_counts(cfg)
    assert counts == (3, 1, 4)
    assert (counts.N, counts.Nr, counts.N_total) == (3, 1, 4)


def test_add_and_sub_resolvers() -> None:
    cfg = OmegaConf.create(
        {"a": 7, "b": 2, "c": "${sub:${a},${b}}", "d": "${add:${a},${b}}"}
    )
    assert (cfg.c, cfg.d) == (5, 9)


def _legacy_config() -> OmegaConf:
    """The qubit counts of a run written before issue #235 (N = Ns + Nr)."""
    return OmegaConf.create(
        {
            "models": "TFIM",
            "N": 10,
            "Nr": 1,
            "Ns": "${int:${sub:${N},${Nr}}}",
            "TFIM": {"N": "${N}", "J": 1.0},
            "ancilla_charge_sector": {"N": "${N}"},
            "single_observable": {"N": "${N}", "observables": ["Z"]},
            "evolution": {"n_steps": "${mul:4,${N}}"},
            "samples": {"L": 100},
        }
    )


def test_a_legacy_config_is_read_in_the_new_convention() -> None:
    """N becomes the system size; every value keeps the one of the run."""
    cfg = upgrade_legacy_counts(_legacy_config())
    assert check_qubit_counts(cfg) == (9, 1, 10)
    assert "Ns" not in cfg
    assert cfg.TFIM.N == cfg.ancilla_charge_sector.N == 9
    assert cfg.single_observable.n_qubits == 10
    assert "N" not in cfg.single_observable
    # n_steps was 4 * 10 in the run, and stays so.
    assert cfg.evolution.n_steps == 40
    assert cfg.samples.L == 100


def test_a_new_config_is_left_unchanged() -> None:
    cfg = load_config(["N=6"])
    assert upgrade_legacy_counts(cfg) is cfg


def test_a_legacy_config_without_ancilla() -> None:
    """With Nr = 0, Ns = N: only the keys change."""
    old = OmegaConf.create(
        {"models": "TFIM", "N": 6, "Nr": 0, "Ns": 6, "TFIM": {"N": 6}}
    )
    assert check_qubit_counts(upgrade_legacy_counts(old)) == (6, 0, 6)
