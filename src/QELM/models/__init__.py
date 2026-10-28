"""Hamiltonians and time evolution."""

from __future__ import annotations

from QELM.models.base import HamiltonianModel
from QELM.models.kitaev import KitaevModel
from QELM.models.TFIM import TransverseFieldIsingModel
from QELM.models.XXZ import XXZChainModel

_MODELS: dict[str, type[HamiltonianModel]] = {
    TransverseFieldIsingModel.name: TransverseFieldIsingModel,
    KitaevModel.name: KitaevModel,
    XXZChainModel.name: XXZChainModel,
}


def get_model(name: str) -> type[HamiltonianModel]:
    """Return the model class for a ``models`` config value.

    Args:
        name: Scalar name such as ``"TFIM"``, ``"XXZ"`` or ``"kitaev"``.

    Returns:
        The matching :class:`HamiltonianModel` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _MODELS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_MODELS))
        raise KeyError(f"Unknown model {name!r}. Known: {known}.") from exc


__all__ = [
    "HamiltonianModel",
    "KitaevModel",
    "TransverseFieldIsingModel",
    "XXZChainModel",
    "get_model",
]
