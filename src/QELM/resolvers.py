"""Arithmetic resolvers for the Hydra YAML in ``conf/``.

OmegaConf interpolation copies values but does not compute them. These
resolvers let a derived value stay linked to its source, for example
``h: ${mul:10.0,${TFIM.J}}``: overriding ``TFIM.J`` on the command line
then rescales ``h``. A derived value can still be overridden directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from omegaconf import OmegaConf


def _mul(*factors: float) -> float:
    """Return the product of the factors, as ``${mul:a,b,...}``."""
    return math.prod(factors)


def _sub(a: float, b: float) -> float:
    """Return ``a - b``, as ``${sub:a,b}``."""
    return a - b


def _div(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, as ``${div:a,b}``."""
    return numerator / denominator


def _int(value: float) -> int:
    """Truncate toward zero like Python's ``int``, as ``${int:x}``."""
    return int(value)


RESOLVERS: dict[str, Callable[..., Any]] = {
    "mul": _mul,
    "sub": _sub,
    "div": _div,
    "int": _int,
}


def register_resolvers() -> None:
    """Register ``mul``, ``sub``, ``div`` and ``int`` with OmegaConf.

    Safe to call more than once. ``import QELM`` calls it, so ``main.py``,
    ``load_config`` and the tests see the resolvers before Hydra resolves
    the config.
    """
    for name, resolver in RESOLVERS.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)
