#!/usr/bin/env python3
"""KL and Jensen-Shannon divergences between ``P(M | Q+)`` and ``P(M | Q-)``.

Issue #270. At one measurement probability ``p``, in the learnability setup
of Barratt et al., PRL 129, 200602 (2022) (``experiments/learnability/``):

1. **Observer who does not know the gates** (Eve, the "blue curve"): draw
   ``samples.L`` records, each from a fresh random U(1) circuit, half
   starting at ``Q+ = N // 2``, half at ``Q- = N // 2 - 1``
   (:func:`QELM.evaluation.learnability.sample_records`). For each record,
   the log-likelihood ratio ``l(M) = ln P(M | Q+) / P(M | Q-)`` with the
   Haar-averaged law, a symmetric exclusion process
   (:func:`QELM.evaluation.kl_divergence.sep_log_likelihood_ratio`).
2. **Observer who knows the circuit**: draw ``kl.n_circuits`` circuits
   (``N_c``); from each, ``kl.records_per_circuit`` records by the Born rule,
   and ``l(M) = ln P(M | Q+, U) / P(M | Q-, U)`` from the state vector
   (:func:`QELM.evaluation.kl_divergence.circuit_log_likelihood_ratio`).
3. From the ratios, the divergences of
   :func:`QELM.evaluation.kl_divergence.divergence_statistics`: KL both ways
   (infinite at every ``p > 0``, see the module), their finite part and the
   infinite mass, Jensen-Shannon (the mutual information ``I(Q; M)``),
   Bhattacharyya, the Bayes error and accuracy, and the bounds on the Bayes
   error. For the circuits, each statistic is averaged over the ``N_c``
   circuits, with its spread between circuits.

Run from the repository root:

    uv run python experiments/kl_divergence/kl_divergence.py N=8 kl.p=0.2

Everything goes to the Hydra run directory (``outputs/<date>/<time>/`` by
default, which git ignores):

- ``kl_divergence.npz``: ``llr`` and ``charge`` of the records of step 1,
  and ``circuit_llr``, ``circuit_charge`` (shape ``(N_c, records)``) of
  step 2;
- ``metrics.json``: ``p``, ``L`` (chain length), ``n_records``,
  ``n_circuits``, every statistic of step 1 as ``sep_<key>`` with its Monte
  Carlo error ``sep_<key>_err``, and of step 2 as ``circuit_<key>`` (mean
  over the circuits), ``circuit_<key>_std`` (spread) and
  ``circuit_<key>_err`` (``std / sqrt(N_c)``). An infinite KL is written
  ``Infinity``, which Python's ``json`` reads back as ``inf``;
- ``.hydra/config.yaml`` and ``resolved_config.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from QELM import check_qubit_counts, set_seed
from QELM.evaluation.kl_divergence import (
    BOUNDS,
    STATISTICS,
    circuit_ensemble_statistics,
    circuit_log_likelihood_ratio,
    divergence_statistics,
    draw_circuit,
    sample_circuit_records,
    sep_log_likelihood_ratio,
    split_by_charge,
)
from QELM.evaluation.learnability import default_charges, sample_records
from QELM.utils import save_config


def run(cfg: DictConfig) -> dict[str, Any]:
    """Draw the records at one ``p`` and estimate the divergences.

    Args:
        cfg: Composed config (``load_config(config_name="kl_divergence")`` or
            Hydra). Reads ``seed``, ``samples.L`` and ``kl``.

    Returns:
        ``llr``, ``charge`` (step 1), ``circuit_llr``, ``circuit_charge``
        (step 2), ``p``, ``L``, ``charges``, ``n_records``, ``n_circuits``,
        and the statistics ``sep_*`` and ``circuit_*``.

    Raises:
        ValueError: If ``kl.batch`` or ``kl.records_per_circuit`` is odd or
            below 2.
    """
    seed = set_seed(cfg)
    check_qubit_counts(cfg)
    params = cfg.kl
    L, p, depth = int(params.L), float(params.p), int(params.depth)
    boundary, every = str(params.boundary), str(params.measure_every)
    charges = (
        default_charges(L)
        if params.charges is None
        else tuple(int(q) for q in params.charges)
    )
    n_records, batch = int(cfg.samples.L), int(params.batch)
    n_circuits = int(params.n_circuits)
    per_circuit = int(params.records_per_circuit)
    for name, value in (("batch", batch), ("records_per_circuit", per_circuit)):
        if value < 2 or value % 2:
            raise ValueError(f"kl.{name} must be even and >= 2, got {value}.")
    rng = np.random.default_rng(seed)

    llr, charge = [], []
    for start in range(0, n_records, batch):
        records = sample_records(
            L,
            p,
            min(batch, n_records - start),
            depth=depth,
            charges=charges,
            boundary=boundary,
            measure_every=every,
            rng=rng,
        )
        llr.append(sep_log_likelihood_ratio(records.outcomes, charges, boundary, every))
        charge.append(records.charge)
    sep_llr, sep_charge = np.concatenate(llr), np.concatenate(charge)
    sep = divergence_statistics(*split_by_charge(sep_llr, sep_charge, charges))

    circuit_llr = np.zeros((n_circuits, per_circuit))
    circuit_charge = np.zeros((n_circuits, per_circuit), dtype=int)
    per_stats = []
    for c in range(n_circuits):
        circuit = draw_circuit(L, depth, boundary, every, rng)
        outcomes, ch = sample_circuit_records(
            L,
            p,
            per_circuit,
            circuit,
            charges=charges,
            boundary=boundary,
            measure_every=every,
            rng=rng,
        )
        circuit_llr[c] = circuit_log_likelihood_ratio(
            outcomes, circuit, charges, boundary, every
        )
        circuit_charge[c] = ch
        per_stats.append(
            divergence_statistics(*split_by_charge(circuit_llr[c], ch, charges))
        )
    ensemble = circuit_ensemble_statistics(per_stats) if per_stats else {}

    return {
        "llr": sep_llr,
        "charge": sep_charge,
        "circuit_llr": circuit_llr,
        "circuit_charge": circuit_charge,
        "p": p,
        "L": L,
        "charges": charges,
        "n_records": int(sep_llr.size),
        "n_circuits": n_circuits,
        **{f"sep_{k}": v for k, v in sep.items()},
        **{f"circuit_{k}": v for k, v in ensemble.items()},
    }


def metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Return the entries of ``metrics.json``: the scalars of :func:`run`."""
    out: dict[str, Any] = {
        "p": float(results["p"]),
        "L": int(results["L"]),
        "charges": [int(q) for q in results["charges"]],
        "n_records": int(results["n_records"]),
        "n_circuits": int(results["n_circuits"]),
    }
    for key, value in results.items():
        if key.startswith(("sep_", "circuit_")) and np.ndim(value) == 0:
            out[key] = float(value)
    return out


def save(results: dict[str, Any], out_dir: Path) -> Path:
    """Write ``kl_divergence.npz`` and ``metrics.json`` to ``out_dir``.

    Args:
        results: Output of :func:`run`.
        out_dir: Destination folder, created if needed.

    Returns:
        Path of the ``.npz`` file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{Path(__file__).stem}.npz"
    np.savez_compressed(
        npz_path,
        llr=results["llr"],
        charge=results["charge"],
        circuit_llr=results["circuit_llr"],
        circuit_charge=results["circuit_charge"],
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics(results), indent=2) + "\n")
    return npz_path


def _fmt(value: float, err: float) -> str:
    return f"{value:.4g}" if not math.isfinite(err) else f"{value:.4g} +- {err:.2g}"


@hydra.main(version_base="1.3", config_path="../../conf", config_name="kl_divergence")
def main(cfg: DictConfig) -> None:
    """Run one ``p`` and save it to the Hydra run directory."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    save_config(cfg, out_dir)
    print(f"N: {results['L']}  p: {results['p']:.4g}  records: {results['n_records']}")
    for key in STATISTICS:
        line = f"{key}: {_fmt(results[f'sep_{key}'], results[f'sep_{key}_err'])}"
        if results["n_circuits"]:
            line += (
                f"   | circuits: {results[f'circuit_{key}']:.4g}"
                f" (std {results[f'circuit_{key}_std']:.2g})"
            )
        print(line)
    for key in BOUNDS:
        print(f"{key}: {results[f'sep_{key}']:.4g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
