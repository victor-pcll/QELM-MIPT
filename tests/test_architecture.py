"""Module boundaries: physics holds primitives, training holds the QELM pipeline."""

from __future__ import annotations

import QELM.physics as physics
import QELM.training as training


def test_physics_only_exposes_primitives() -> None:
    """Measurements and observables are no longer in physics."""
    assert set(physics.__all__) == {
        "PAULIS",
        "I",
        "X",
        "Y",
        "Z",
        "get_site_operator",
        "get_site_operator_sparse",
        "n_qubits",
        "pauli",
    }
    for moved in ("Readout", "ProjectiveMeasurement", "SingleObservable"):
        assert not hasattr(physics, moved)


def test_training_modules_split_by_role() -> None:
    """Each class lives in the module named after its role."""
    assert training.ProjectiveMeasurement.__module__ == "QELM.training.measurements"
    assert training.ReadoutResult.__module__ == "QELM.training.measurements"
    assert training.SingleObservable.__module__ == "QELM.training.readout"
    assert training.DoubleObservable.__module__ == "QELM.training.readout"
    assert training.LinearReadout.__module__ == "QELM.training.regression"
    assert training.RidgeRegression.__module__ == "QELM.training.regression"


def test_trainer_and_readout_registries() -> None:
    """Config keys still select the same classes."""
    assert training.get_trainer("readout") is training.LinearReadout
    assert training.get_trainer("ridge") is training.RidgeRegression
    assert training.get_readout("projective_measurement") is (
        training.ProjectiveMeasurement
    )
