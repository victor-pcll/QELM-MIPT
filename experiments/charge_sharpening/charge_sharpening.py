#!/usr/bin/env python3
"""Charge-sharpening benchmark: ancilla entropy under weak U(1) monitoring.

Reproduces the simulation of Fig. 7 of Agrawal et al., "Observing quantum
measurement collapse as a learnability phase transition", PRX 14, 041012
(2024) (appendix on charge sharpening, issue #137), at one measurement
strength ``gamma``. It is a test of the library on a published result: the
encoding, the partial trace and the entropies are the ones of the QELM
experiments; only the circuit and the measurement are new.

1. Prepare Eq. (B1), ``(|0>_R |Q0>_S + |1>_R |Q1>_S) / sqrt(2)``, with
   :class:`AncillaChargeSectorEncoding` (``initial: product``): the ancilla
   ``R`` (qubit 0) entangled with the charge ``Q0 = N // 2`` or
   ``Q1 = N // 2 - 1`` of the ``N`` system qubits.
2. Draw a :class:`U1BrickworkCircuit` and scramble the state with its
   all-to-all layers (``U1_circuit.scrambling_layers``). With
   ``U1_circuit.fixed_instance: true`` (the paper's choice) this circuit is
   drawn once and shared by every trajectory, and the seed selects it;
   with ``false`` every trajectory draws its own.
3. For each of the ``samples.L`` trajectories, apply ``U1_circuit.depth``
   brickwork layers (``t = N / 2`` by default, each a full period of the
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
so trajectories are simulated with state vectors of size ``2**N_total``.

Charge conservation is checked along every trajectory (issue #134). The
total charge of the ``N_total`` qubits, ancilla included, is ``N // 2`` in both
branches of Eq. (B1), and every gate and Kraus operator conserves it, so
the state never leaves that sector. After the scrambling, every layer and
every measurement round, :func:`QELM.physics.charge_leakage` gives the
weight outside the sector; the largest one of a trajectory is its
``charge_leakage``, exactly 0 for a correct simulation. With
``charge_conservation.filter_non_conserved_charge=true`` the trajectories
whose leakage exceeds ``charge_conservation.atol`` are dropped before the
statistics; by default every trajectory is kept and the check is only
reported.

Run from the repository root:

    uv run python experiments/charge_sharpening/charge_sharpening.py \
        weak_measurement.gamma=0.4 samples.L=10000

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``charge_sharpening.npz``: ``S_R`` (one entropy per trajectory, shape
  ``(L,)``), ``M`` (measurement record, shape ``(L, rounds, N)``, 1 for
  a click), ``charge_leakage`` (shape ``(L,)``) and ``kept`` (shape
  ``(L,)``, the trajectories used by the statistics), for every
  trajectory, kept or not;
- ``metrics.json``: ``gamma``, ``S_R_mean``, ``S_R_var``, ``binder``, their
  bootstrap errors ``<name>_err``, the click rate (over the kept
  trajectories), ``n_samples`` (kept trajectories) and the charge check:
  ``max_charge_leakage``, ``n_non_conserved``, ``n_discarded`` and
  ``discarded_fraction``;
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
from tqdm import tqdm

from QELM import check_qubit_counts, set_seed
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.evaluation import (
    bootstrap_errors,
    charge_filter,
    sharpening_statistics,
    von_neumann_entropy,
)
from QELM.models.circuits.haar import U1BrickworkCircuit
from QELM.physics import charge_leakage, sharp_charge
from QELM.training import WeakMeasurement
from QELM.utils import save_config

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
    "max_charge_leakage",
    "discarded_fraction",
)
# Integer counts written to metrics.json.
COUNTS = ("n_non_conserved", "n_discarded")


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories at one ``gamma`` and compute the statistics.

    Args:
        cfg: Composed config (``load_config(config_name="charge_sharpening")``
            or Hydra). Reads ``samples.L``, ``ancilla_charge_sector``,
            ``U1_circuit``, ``weak_measurement``, ``charge_sharpening`` and
            ``charge_conservation``.

    Returns:
        ``S_R`` (shape ``(L,)``), ``M`` (shape ``(L, rounds, N)``, with
        ``rounds = depth - 1``, or ``depth`` when measuring after the last
        layer), ``charge_leakage`` and ``kept`` (shape ``(L,)``), for every
        trajectory, and the scalars of :data:`METRICS` and :data:`COUNTS`,
        computed on the kept trajectories.

    Raises:
        ValueError: If the initial state has no sharp total charge, or the
            filter drops every trajectory.
    """
    set_seed(cfg)
    N, Nr, N_total = check_qubit_counts(cfg)
    base = float(cfg.charge_sharpening.entropy_base)

    sector = AncillaChargeSectorEncoding.from_config(cfg)
    measurement = WeakMeasurement.from_config(cfg)
    gamma = measurement.gamma
    # Drawn first, from the seeded state: with a fixed instance, every gamma
    # of a sweep sees the same circuit for a given seed.
    circuit = U1BrickworkCircuit.from_config(cfg)
    psi0 = sector.state()
    # Total charge of R + S, N // 2 in both branches of Eq. (B1).
    Q = sharp_charge(psi0)
    psi_scrambled = circuit.scramble(psi0)
    leak_scrambled = charge_leakage(psi_scrambled, Q)

    L = int(cfg.samples.L)
    depth = circuit.depth
    measure_last = bool(cfg.charge_sharpening.measure_after_last_layer)
    # Layers followed by a measurement round.
    measured = [t for t in range(depth) if t < depth - 1 or measure_last]
    S_R = np.zeros(L)
    M = np.zeros((L, len(measured), N))
    leakage = np.zeros(L)
    for i in tqdm(range(L)):
        if circuit.fixed_instance:
            psi = psi_scrambled
            leak = leak_scrambled
        else:
            circuit.redraw()
            psi = circuit.scramble(psi0)
            leak = charge_leakage(psi, Q)
        for t in range(depth):
            psi = circuit.apply_layer(psi, t)
            leak = max(leak, charge_leakage(psi, Q))
            if t in measured:
                result = measurement.measure(psi)
                psi = result.rho
                M[i, t] = result.outcomes
                leak = max(leak, charge_leakage(psi, Q))
        S_R[i] = von_neumann_entropy(psi, SECTOR_R, base=base)
        leakage[i] = leak

    kept, check = charge_filter(leakage, cfg.charge_conservation)
    stats = sharpening_statistics(S_R[kept])
    errors = bootstrap_errors(S_R[kept], n_boot=int(cfg.charge_sharpening.n_boot))
    return {
        "S_R": S_R,
        "M": M,
        "charge_leakage": leakage,
        "kept": kept,
        "gamma": gamma,
        **stats,
        **errors,
        "click_rate": float(np.mean(M[kept])) if M[kept].size else 0.0,
        **check,
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
    np.savez(
        npz_path,
        S_R=results["S_R"],
        M=results["M"],
        charge_leakage=results["charge_leakage"],
        kept=results["kept"],
    )
    metrics: dict[str, float | int] = {key: float(results[key]) for key in METRICS}
    metrics.update({key: int(results[key]) for key in COUNTS})
    metrics["n_samples"] = int(np.count_nonzero(results["kept"]))
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
    save_config(cfg, out_dir)
    n_kept = int(np.count_nonzero(results["kept"]))
    print(f"gamma: {results['gamma']:.4g}  samples: {n_kept}")
    print(f"S_R: {results['S_R_mean']:.4f} +- {results['S_R_mean_err']:.4f}")
    print(f"var S_R: {results['S_R_var']:.4f} +- {results['S_R_var_err']:.4f}")
    print(f"Binder: {results['binder']:.4f} +- {results['binder_err']:.4f}")
    print(
        f"charge: max leakage {results['max_charge_leakage']:.3g}, "
        f"{results['n_non_conserved']} non-conserving, "
        f"{results['n_discarded']} discarded"
    )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
