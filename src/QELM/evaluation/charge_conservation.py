"""Keep or drop runs according to their charge conservation (issue #134).

Each run of a U(1)-symmetric simulation (a trajectory of the
charge-sharpening benchmark, a record of the learnability benchmark) is
checked with :func:`QELM.physics.charge.charge_leakage` after every step;
its largest leakage over the evolution says whether the charge was
conserved. With ``charge_conservation.filter_non_conserved_charge: true``
the runs whose leakage exceeds ``charge_conservation.atol`` are dropped
before the statistics; by default (``false``) every run is kept and the
check is only reported, so the baseline results do not change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def conserved_mask(
    leakage: np.ndarray, atol: float, filter_non_conserved: bool
) -> np.ndarray:
    """Return which runs to keep, given their charge leakage.

    Args:
        leakage: Largest leakage of each run over its evolution, shape
            ``(n,)``.
        atol: A run conserves the charge if its leakage is at most ``atol``.
        filter_non_conserved: ``True`` to drop the runs that do not conserve
            the charge; ``False`` keeps every run (the baseline behaviour).

    Returns:
        Boolean array of shape ``(n,)``, ``True`` for the runs to keep.

    Raises:
        ValueError: If ``atol`` is negative.
    """
    if atol < 0:
        raise ValueError(f"atol must be >= 0, got {atol}.")
    leakage = np.asarray(leakage, dtype=float)
    if not filter_non_conserved:
        return np.ones(leakage.shape, dtype=bool)
    return np.asarray(leakage <= atol)


def conservation_metrics(
    leakage: np.ndarray, atol: float, kept: np.ndarray
) -> dict[str, float]:
    """Summarize the charge check of a set of runs, for ``metrics.json``.

    Args:
        leakage: Largest leakage of each run, shape ``(n,)``.
        atol: Tolerance of :func:`conserved_mask`.
        kept: Output of :func:`conserved_mask`.

    Returns:
        ``max_charge_leakage``, ``n_non_conserved`` (runs with a leakage
        above ``atol``), ``n_discarded`` and ``discarded_fraction``.
    """
    leakage = np.asarray(leakage, dtype=float)
    n = leakage.size
    n_discarded = int(n - np.count_nonzero(kept))
    return {
        "max_charge_leakage": float(leakage.max()) if n else 0.0,
        "n_non_conserved": int(np.count_nonzero(leakage > atol)),
        "n_discarded": n_discarded,
        "discarded_fraction": n_discarded / n if n else 0.0,
    }


def charge_filter(
    leakage: np.ndarray, params: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the ``charge_conservation`` block of a config to the runs.

    Args:
        leakage: Largest leakage of each run, shape ``(n,)``.
        params: The ``charge_conservation`` block: ``atol`` and
            ``filter_non_conserved_charge`` (``False`` by default).

    Returns:
        ``(kept, metrics)``: the output of :func:`conserved_mask` and of
        :func:`conservation_metrics`.

    Raises:
        ValueError: If the filter drops every run.
    """
    atol = float(params["atol"])
    enabled = bool(params.get("filter_non_conserved_charge", False))
    kept = conserved_mask(leakage, atol, enabled)
    if np.size(leakage) and not np.any(kept):
        raise ValueError(
            "Every run broke charge conservation (largest leakage "
            f"{float(np.max(leakage)):.3g} > atol = {atol:.3g}); nothing is left "
            "to score."
        )
    return kept, conservation_metrics(leakage, atol, kept)
