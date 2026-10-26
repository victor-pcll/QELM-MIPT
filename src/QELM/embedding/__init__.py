"""State embeddings: how the input is encoded into the reservoir state."""

from __future__ import annotations

from QELM.embedding.base import Embedding
from QELM.embedding.charge_sector import (
    AncillaChargeSectorEmbedding,
    charge_sector_indices,
    charge_sector_state,
    product_charge_index,
)
from QELM.embedding.rotation import RotationEmbedding

_EMBEDDINGS: dict[str, type[Embedding]] = {
    AncillaChargeSectorEmbedding.name: AncillaChargeSectorEmbedding,
    RotationEmbedding.name: RotationEmbedding,
}


def get_embedding(name: str) -> type[Embedding]:
    """Return the embedding class for a config key.

    Args:
        name: Embedding name such as ``"ancilla_charge_sector"``.

    Returns:
        The matching :class:`Embedding` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _EMBEDDINGS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_EMBEDDINGS))
        raise KeyError(f"Unknown embedding {name!r}. Known: {known}.") from exc


__all__ = [
    "AncillaChargeSectorEmbedding",
    "Embedding",
    "RotationEmbedding",
    "charge_sector_indices",
    "charge_sector_state",
    "get_embedding",
    "product_charge_index",
]
