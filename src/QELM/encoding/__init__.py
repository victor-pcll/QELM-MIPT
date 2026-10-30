"""State encodings: how the input is encoded into the reservoir state (issue #197).

The tensor-product encodings of issue #196 (:class:`ProductEncoding`):
:class:`AngleEncoding`, :class:`DenseAngleEncoding`,
:class:`UniformBlochEncoding` and :class:`GeneralEncoding`.
"""

from __future__ import annotations

from QELM.encoding.angle import AngleEncoding
from QELM.encoding.base import Encoding, OutOfBoundsError, ProductEncoding
from QELM.encoding.bell_pair import AncillaBellPairEncoding
from QELM.encoding.charge_sector import (
    AncillaChargeSectorEncoding,
    charge_sector_indices,
    charge_sector_state,
    product_charge_index,
)
from QELM.encoding.dense_angle import DenseAngleEncoding
from QELM.encoding.general import GeneralEncoding
from QELM.encoding.rotation import RotationEncoding
from QELM.encoding.uniform_bloch import UniformBlochEncoding

_ENCODINGS: dict[str, type[Encoding]] = {
    AncillaBellPairEncoding.name: AncillaBellPairEncoding,
    AncillaChargeSectorEncoding.name: AncillaChargeSectorEncoding,
    RotationEncoding.name: RotationEncoding,
    AngleEncoding.name: AngleEncoding,
    DenseAngleEncoding.name: DenseAngleEncoding,
    UniformBlochEncoding.name: UniformBlochEncoding,
    GeneralEncoding.name: GeneralEncoding,
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
    "AncillaBellPairEncoding",
    "AncillaChargeSectorEncoding",
    "AngleEncoding",
    "DenseAngleEncoding",
    "Encoding",
    "GeneralEncoding",
    "OutOfBoundsError",
    "ProductEncoding",
    "RotationEncoding",
    "UniformBlochEncoding",
    "charge_sector_indices",
    "charge_sector_state",
    "get_encoding",
    "product_charge_index",
]
