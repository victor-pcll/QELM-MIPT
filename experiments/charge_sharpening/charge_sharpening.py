#!/usr/bin/env python3
"""Charge-sharpening benchmark: ancilla entropy under weak U(1) monitoring.

Reproduces the simulation of Fig. 7 of Agrawal et al., "Observing quantum
measurement collapse as a learnability phase transition", PRX 14, 041012
(2024) (appendix on charge sharpening, issue #137), at one measurement
strength ``gamma``. It is a test of the library on a published result: the
embedding, the partial trace and the entropies are the ones of the QELM
experiments; only the circuit and the measurement are new.

1. Prepare Eq. (B1), ``(|0>_R |Q0>_S + |1>_R |Q1>_S) / sqrt(2)``, with
   :class:`AncillaChargeSectorEmbedding` (``initial: product``): the ancilla
   ``R`` (qubit 0) entangled with the charge ``Q0 = Ns // 2`` or
   ``Q1 = Ns // 2 - 1`` of the ``Ns`` system qubits.
2. Draw a :class:`U1BrickworkCircuit` and scramble the state with its
   all-to-all layers (``U1_circuit.scrambling_layers``). With
   ``U1_circuit.fixed_instance: true`` (the paper's choice) this circuit is
   drawn once and shared by every trajectory, and the seed selects it;
   with ``false`` every trajectory draws its own.
3. For each of the ``samples.L`` trajectories, apply ``U1_circuit.depth``
   brickwork layers (``t = Ns / 2`` by default, each a full period of the
   brickwork) and weakly measure every system qubit with strength
   ``weak_measurement.gamma`` (:class:`WeakMeasurement`) between two
   layers: ``t - 1`` measurement rounds. A last round after the final layer
   is added with ``charge_sharpening.measure_after_last_layer=true``. The
   outcomes are drawn with the Born rule, so trajectories come with their
   weight ``p_m``.

   This timing is not spelled out in the paper. It was chosen against the
   digitized Fig. 7 (``reference/``): at ``L = 6`` it matches ``S_R`` within
   about 0.01 nats, while measuring after every layer (one more round)
   sharpens the charge about 0.1 too early in ``gamma``.
4. Record the entanglement entropy ``S_R^(m)`` of the ancilla with the system
   (von Neumann, in nats as in the paper: ``charge_sharpening.entropy_base``).

The scores are the observables of Fig. 7, from
:mod:`QELM.evaluation.charge_sharpening`: the mean ``S_R``, the variance
``mu_2`` and the Binder ratio ``B = 1 - mu_4 / (3 mu_2**2)``, with bootstrap
error bars. The Binder ratio is the standard ``1 - mu_4 / (3 mu_2**2)``:
the digitized curve has the same large negative values away from the peak
(about ``-10`` at ``gamma = 0.2``), which the printed formula of the paper
cannot give. At ``gamma = 0`` the record holds no information and ``S_R``
stays at its scrambled value; as ``gamma`` grows the measurements learn the
charge and ``S_R`` drops to 0: the charge sharpens.

The state stays pure (unitary gates and Kraus operators on a pure state),
so trajectories are simulated with state vectors of size ``2**N``.

Run from the repository root:

    uv run python experiments/charge_sharpening/charge_sharpening.py \
        weak_measurement.gamma=0.4 samples.L=10000

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``charge_sharpening.npz``: ``S_R`` (one entropy per trajectory, shape
  ``(L,)``) and ``M`` (measurement record, shape ``(L, rounds, Ns)``, 1 for
  a click);
- ``metrics.json``: ``gamma``, ``S_R_mean``, ``S_R_var``, ``binder``, their
  bootstrap errors ``<name>_err`` and the click rate;
- ``.hydra/config.yaml``: the full composed config, to reproduce the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from QELM import check_qubit_counts, set_seed
from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.evaluation import (
    bootstrap_errors,
    sharpening_statistics,
    von_neumann_entropy,
)
from QELM.models.circuits import U1BrickworkCircuit
from QELM.training import WeakMeasurement

# The ancilla R, qubit 0, is the subsystem of S_R.
SECTOR_R = [0]
# Scalars written to metrics.json.
METRICS = (
    "gamma",
    "S_R_mean",
    "S_R_var",
    "binder",
    "S_R_mean_err",
    "S_R_var_err",
    "binder_err",
    "click_rate",
)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories at one ``gamma`` and compute the statistics.

    Args:
        cfg: Composed config (``load_config(config_name="charge_sharpening")``
            or Hydra). Reads ``samples.L``, ``ancilla_charge_sector``,
            ``U1_circuit``, ``weak_measurement`` and ``charge_sharpening``.

    Returns:
        ``S_R`` (shape ``(L,)``), ``M`` (shape ``(L, rounds, Ns)``, with
        ``rounds = depth - 1``, or ``depth`` when measuring after the last
        layer) and the scalars of :data:`METRICS`.
    """
    set_seed(cfg)
    N, Ns, Nr = check_qubit_counts(cfg)
    base = float(cfg.charge_sharpening.entropy_base)

    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    measurement = WeakMeasurement.from_config(cfg)
    gamma = measurement.gamma
    # Drawn first, from the seeded state: with a fixed instance, every gamma
    # of a sweep sees the same circuit for a given seed.
    circuit = U1BrickworkCircuit.from_config(cfg)
    psi0 = sector.state()
    psi_scrambled = circuit.scramble(psi0)

    L = int(cfg.samples.L)
    depth = circuit.depth
    measure_last = bool(cfg.charge_sharpening.measure_after_last_layer)
    # Layers followed by a measurement round.
    measured = [t for t in range(depth) if t < depth - 1 or measure_last]
    S_R = np.zeros(L)
    M = np.zeros((L, len(measured), Ns))
    for i in tqdm(range(L)):
        if circuit.fixed_instance:
            psi = psi_scrambled
        else:
            circuit.redraw()
            psi = circuit.scramble(psi0)
        for t in range(depth):
            psi = circuit.apply_layer(psi, t)
            if t in measured:
                result = measurement.measure(psi)
                psi = result.rho
                M[i, t] = result.outcomes
        S_R[i] = von_neumann_entropy(psi, SECTOR_R, base=base)

    stats = sharpening_statistics(S_R)
    errors = bootstrap_errors(S_R, n_boot=int(cfg.charge_sharpening.n_boot))
    return {
        "S_R": S_R,
        "M": M,
        "gamma": gamma,
        **stats,
        **errors,
        "click_rate": float(np.mean(M)) if M.size else 0.0,
    }


def save(results: dict[str, Any], out_dir: Path) -> Path:
    """Write ``charge_sharpening.npz`` and ``metrics.json`` to ``out_dir``.

    Args:
        results: Output of :func:`run`.
        out_dir: Destination folder, created if needed.

    Returns:
        Path of the ``.npz`` file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{Path(__file__).stem}.npz"
    np.savez(npz_path, S_R=results["S_R"], M=results["M"])
    metrics = {key: float(results[key]) for key in METRICS}
    metrics["n_samples"] = int(results["S_R"].shape[0])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return npz_path


@hydra.main(
    version_base="1.3", config_path="../../conf", config_name="charge_sharpening"
)
def main(cfg: DictConfig) -> None:
    """Run one gamma and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/charge_sharpening.yaml`` and
            CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    print(f"gamma: {results['gamma']:.4g}  samples: {results['S_R'].shape[0]}")
    print(f"S_R: {results['S_R_mean']:.4f} +- {results['S_R_mean_err']:.4f}")
    print(f"var S_R: {results['S_R_var']:.4f} +- {results['S_R_var_err']:.4f}")
    print(f"Binder: {results['binder']:.4f} +- {results['binder_err']:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
