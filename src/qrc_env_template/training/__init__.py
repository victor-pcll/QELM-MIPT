"""Training algorithms."""

from __future__ import annotations

from qrc_env_template.training.base import Trainer
from qrc_env_template.training.readout import Readout
from qrc_env_template.training.ridge import RidgeRegression

_TRAINERS: dict[str, type[Trainer]] = {
    Readout.name: Readout,
    RidgeRegression.name: RidgeRegression,
}


def get_trainer(name: str) -> type[Trainer]:
    """Return the trainer class for a ``training.name`` config value."""
    try:
        return _TRAINERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TRAINERS))
        raise KeyError(f"Unknown training {name!r}. Known: {known}.") from exc


__all__ = ["Readout", "RidgeRegression", "Trainer", "get_trainer"]
