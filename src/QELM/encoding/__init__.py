"""State encodings: how the input is encoded into the reservoir state (issue #197)."""

from __future__ import annotations

from QELM.encoding.base import Encoding
from QELM.encoding.charge_sector import (
    AncillaChargeSectorEncoding,
    charge_sector_indices,
    charge_sector_state,
    product_charge_index,
)
from QELM.encoding.rotation import RotationEncoding

_ENCODINGS: dict[str, type[Encoding]] = {
    AncillaChargeSectorEncoding.name: AncillaChargeSectorEncoding,
    RotationEncoding.name: RotationEncoding,
}


def get_encoding(name: str) -> type[Encoding]:
    """Return the encoding class for a config key.

    Args:
        name: Encoding name such as ``"ancilla_charge_sector"``.

    Returns:
        The matching :class:`Encoding` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _ENCODINGS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ENCODINGS))
        raise KeyError(f"Unknown encoding {name!r}. Known: {known}.") from exc


__all__ = [
    "AncillaChargeSectorEncoding",
    "Encoding",
    "RotationEncoding",
    "charge_sector_indices",
    "charge_sector_state",
    "get_encoding",
    "product_charge_index",
]
