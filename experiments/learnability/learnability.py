#!/usr/bin/env python3
"""Charge learnability: accuracy of Eve's optimal classifier at one ``p``.

Reproduces the simulation of Fig. 2 of Barratt, Agrawal, Potter,
Gopalakrishnan and Vasseur, "Transitions in the learnability of global
charges from local measurements", PRL 129, 200602 (2022), arXiv:2206.12429,
at one measurement probability ``p``: Fig. 2b (accuracy, issue #113) and
Fig. 2c (mean entropy, issue #115) come from the same records.

1. Draw ``samples.L`` records (:func:`QELM.evaluation.learnability.sample_records`):
   a chain of ``L`` qubits starts in the uniform superposition of the
   bitstrings of charge ``Q0 = L // 2`` (half of the records) or
   ``Q1 = L // 2 - 1`` (the other half), evolves for ``t_f = L`` steps of a
   random U(1) brickwork (a fresh circuit per record), and after every step
   each site is measured in Z with probability ``p``, by the Born rule.
2. Eve, who knows the record but not the gates, computes the Bayes posterior
   ``P(Q | m)`` with the Haar-averaged dynamics, a symmetric exclusion
   process (:func:`~QELM.evaluation.learnability.correct_probability`). The
   statistic of a record is ``P_corr = P(Q* | m)``, the posterior of its
   true charge.
3. Score the records (:func:`~QELM.evaluation.learnability.learnability_statistics`):
   the accuracy ``P(P_corr > 1/2)`` of Fig. 2b (a tie ``P_corr = 1/2``
   counts 1/2, so a blind guess scores 1/2; ``accuracy_strict`` counts it
   0), the Binder ratio of ``P_corr`` (inset of Fig. 2b), the mean binary
   entropy ``H(P_corr)`` of Fig. 2c and its Binder ratio (inset), with
   bootstrap errors.

The records are simulated ``learnability.batch`` at a time, so the memory
stays ``batch * 2**L`` amplitudes whatever ``samples.L``.

Run from the repository root:

    uv run python experiments/learnability/learnability.py \
        N=6 learnability.p=0.2 samples.L=40000

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``learnability.npz``: ``P_corr`` and ``charge`` (one per record, shape
  ``(n,)``) and the records ``outcomes`` (shape ``(n, rounds, L)``, with
  ``rounds = 2 t_f`` for ``learnability.measure_every=layer`` and ``t_f``
  for ``period``; 0 or 1 for a measured site, -1 for an unmeasured one);
- ``metrics.json``: ``p``, the statistics, their bootstrap errors
  ``<name>_err`` and the measured fraction of the sites;
- ``.hydra/config.yaml``: the full composed config, to reproduce the run.
- ``resolved_config.json``: the same config with every ``${...}`` evaluated,
  the numbers the run used (issue #85).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from QELM import check_qubit_counts, set_seed
from QELM.evaluation.learnability import (
    UNMEASURED,
    correct_probability,
    learnability_bootstrap_errors,
    learnability_statistics,
    sample_records,
)
from QELM.utils import save_config

# Statistics written to metrics.json, each with its bootstrap error.
STATISTICS = (
    "accuracy",
    "accuracy_strict",
    "P_corr_mean",
    "P_corr_binder",
    "entropy_mean",
    "entropy_var",
    "entropy_binder",
)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Draw the records at one ``p``, classify them and score the classifier.

    Args:
        cfg: Composed config (``load_config(config_name="learnability")`` or
            Hydra). Reads ``seed``, ``samples.L`` and ``learnability``.

    Returns:
        ``P_corr`` and ``charge`` (shape ``(n,)``), ``outcomes`` (shape
        ``(n, rounds, L)``), ``p``, ``L``, ``measured_fraction`` and the
        :data:`STATISTICS` with their ``_err``.
    """
    seed = set_seed(cfg)
    check_qubit_counts(cfg)
    params = cfg.learnability
    L = int(params.L)
    p = float(params.p)
    n_records = int(cfg.samples.L)
    batch = int(params.batch)
    base = float(params.entropy_base)
    charges = None if params.charges is None else [int(q) for q in params.charges]
    rng = np.random.default_rng(seed)
    if batch < 2 or batch % 2:
        # Each batch starts at the first charge: with an odd batch the two
        # charges are not drawn equally often.
        raise ValueError(f"learnability.batch must be even and >= 2, got {batch}.")

    P_corr, charge, outcomes = [], [], []
    for start in range(0, n_records, batch):
        records = sample_records(
            L,
            p,
            min(batch, n_records - start),
            depth=int(params.depth),
            charges=charges,
            boundary=str(params.boundary),
            measure_every=str(params.measure_every),
            rng=rng,
        )
        P_corr.append(correct_probability(records))
        charge.append(records.charge)
        outcomes.append(records.outcomes)
    P = np.concatenate(P_corr)
    out = np.concatenate(outcomes)
    stats = learnability_statistics(P, base=base)
    errors = learnability_bootstrap_errors(
        P, n_boot=int(params.n_boot), base=base, rng=rng
    )
    return {
        "P_corr": P,
        "charge": np.concatenate(charge),
        "outcomes": out,
        "p": p,
        "L": L,
        "measured_fraction": float(np.mean(out != UNMEASURED)) if out.size else 0.0,
        **stats,
        **errors,
    }


def save(results: dict[str, Any], out_dir: Path) -> Path:
    """Write ``learnability.npz`` and ``metrics.json`` to ``out_dir``.

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
        P_corr=results["P_corr"],
        charge=results["charge"],
        outcomes=results["outcomes"],
    )
    keys = ("p", "measured_fraction", *STATISTICS, *(f"{k}_err" for k in STATISTICS))
    metrics: dict[str, Any] = {key: float(results[key]) for key in keys}
    metrics["L"] = int(results["L"])
    metrics["n_records"] = int(results["P_corr"].shape[0])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return npz_path


@hydra.main(version_base="1.3", config_path="../../conf", config_name="learnability")
def main(cfg: DictConfig) -> None:
    """Run one ``p`` and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/learnability.yaml`` and CLI
            overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    save_config(cfg, out_dir)
    print(
        f"L: {results['L']}  p: {results['p']:.4g}  records: {len(results['P_corr'])}"
    )
    for key in STATISTICS:
        print(f"{key}: {results[key]:.4f} +- {results[f'{key}_err']:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
