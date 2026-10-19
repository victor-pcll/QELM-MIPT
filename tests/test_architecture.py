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
        "local_expectation",
        "qubit_counts",
        "pauli",
    }
    for moved in ("Readout", "ProjectiveMeasurement", "SingleObservable"):
        assert not hasattr(physics, moved)


def test_training_modules_split_by_role() -> None:
    """Each class lives in its domain subpackage: parent in base.py (#29)."""
    expected = {
        training.Readout: "QELM.training.measurements.base",
        training.ReadoutResult: "QELM.training.measurements.base",
        training.ProjectiveMeasurement: "QELM.training.measurements.projective",
        training.Observables: "QELM.training.readout.base",
        training.SingleObservable: "QELM.training.readout.single",
        training.DoubleObservable: "QELM.training.readout.double",
        training.Trainer: "QELM.training.regression.base",
        training.LinearReadout: "QELM.training.regression.linear",
        training.RidgeReadout: "QELM.training.regression.ridge",
    }
    for cls, module in expected.items():
        assert cls.__module__ == module, cls


def test_children_inherit_from_their_base() -> None:
    assert issubclass(training.ProjectiveMeasurement, training.Readout)
    assert issubclass(training.SingleObservable, training.Observables)
    assert issubclass(training.DoubleObservable, training.Observables)
    assert issubclass(training.LinearReadout, training.Trainer)
    assert issubclass(training.RidgeReadout, training.Trainer)


def test_subpackage_imports_still_work() -> None:
    """Imports by domain keep working, as before the split."""
    from QELM.training.measurements import ProjectiveMeasurement, Readout
    from QELM.training.readout import DoubleObservable, SingleObservable
    from QELM.training.regression import RidgeReadout, Trainer

    assert ProjectiveMeasurement is training.ProjectiveMeasurement
    assert Readout is training.Readout
    assert SingleObservable is training.SingleObservable
    assert DoubleObservable is training.DoubleObservable
    assert RidgeReadout is training.RidgeReadout
    assert Trainer is training.Trainer


def test_trainer_and_readout_registries() -> None:
    """Config keys still select the same classes."""
    assert training.get_trainer("linear") is training.LinearReadout
    assert training.get_trainer("ridge") is training.RidgeReadout
    assert training.get_readout("projective_measurement") is (
        training.ProjectiveMeasurement
    )


def test_old_trainer_name_readout_is_gone() -> None:
    """``readout`` a été renommé ``linear`` (issue #49)."""
    import pytest

    with pytest.raises(KeyError, match="Known: linear, ridge"):
        training.get_trainer("readout")
