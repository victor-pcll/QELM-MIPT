#!/usr/bin/env python3
"""Reference-qubit purification benchmark: Gullans & Huse (2020), Fig. 1.

Reproduces the probe of the measurement-induced transition of M. J. Gullans
and D. A. Huse, "Scalable probes of measurement-induced criticality",
PRL 125, 070606 (2020), arXiv:1910.00020 (issue #185), at one size ``N``
(the paper's ``L``) and one measurement probability ``p``:

1. The reference qubit ``R`` (qubit 0) starts in a Bell pair with the
   middle system site, the other system qubits in ``|0>``
   (:class:`QELM.encoding.AncillaBellPairEncoding`).
2. ``probe.scramble_steps = 2N`` steps of the periodic Clifford brickwork
   without measurement (the paper's ``t0 = 2L``).
3. ``probe.steps = 2N`` more steps, each followed by a Z measurement of
   every system qubit with probability ``p`` (``t - t0 = 2L``).

A step is one half-layer of two-qubit gates drawn uniformly from the
Clifford group, so that every site gets one gate per step: the paper's
time unit. Every trajectory draws new gates (average over circuits). The
state is simulated with stabilizer tableaux
(:mod:`QELM.models.circuits.stabilizer`), which reaches the paper's sizes;
the tests check it against the state vector at small ``N``.

The order parameter is ``S_Q``, the entropy of ``R`` in bits, averaged over
the trajectories at the final time ``t = 4N``. For a stabilizer state it is
0 or 1 in each trajectory, so ``S_Q`` is the probability that ``R`` is
still mixed. The paper finds that the curves of different ``N`` cross at
``p_c = 0.1598(5)``, with ``nu = 1.30(5)``.

With ``probe.dynamics=haar`` (issue #257) the gates are Haar ``U(4)`` on the
same brickwork and the state is a state vector
(:func:`QELM.models.circuits.monitored.haar_reference_entropy`, ``N <= 16``):
only the final ``S_Q`` is kept, with its Renyi-2 value ``S_Q2``.

Run from the repository root:

    uv run python experiments/purification/purification.py N=16 probe.p=0.16

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``purification.npz``: ``S_Q`` (entropy of ``R`` after each step of each
  trajectory, shape ``(L, scramble_steps + steps)``, 0 or 1) and
  ``n_measured`` (number of measurements per trajectory, shape ``(L,)``);
- ``metrics.json``: ``N``, ``p``, ``S_Q_mean`` and its standard error
  ``S_Q_err`` at the final time, ``S_Q_t`` (mean ``S_Q`` after each step),
  ``measurement_rate`` (measurements per site and measured step),
  ``n_samples``, ``scramble_steps`` and ``steps``;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config.
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

from QELM import check_qubit_counts
from QELM.encoding import AncillaBellPairEncoding
from QELM.models.circuits import draw_probe_history, reference_entropy_history
from QELM.models.circuits.monitored import haar_reference_entropy
from QELM.utils import save_config


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories at one ``(N, p)``.

    Args:
        cfg: Composed config (``load_config(config_name=
            "purification_gullans_huse")`` or Hydra). Reads ``seed``,
            ``samples.L``, ``ancilla_bell_pair`` and ``probe``.

    Returns:
        ``S_Q`` (shape ``(L, T)``), ``n_measured`` (shape ``(L,)``) and the
        scalars written to ``metrics.json``.
    """
    N, _, _ = check_qubit_counts(cfg)
    site = AncillaBellPairEncoding.from_config(cfg).site
    p = float(cfg.probe.p)
    scramble_steps = int(cfg.probe.scramble_steps)
    steps = int(cfg.probe.steps)
    L = int(cfg.samples.L)
    # One stream per (seed, N, p): runs of a sweep are independent.
    rng = np.random.default_rng([int(cfg.seed), N, round(p * 1e9)])
    if str(cfg.probe.get("dynamics", "clifford")) == "haar":
        return _run_haar(N, p, scramble_steps, steps, L, rng)
    T = scramble_steps + steps
    S_Q = np.zeros((L, T), dtype=np.uint8)
    n_measured = np.zeros(L, dtype=int)
    for i in tqdm(range(L)):
        history = draw_probe_history(N, p, scramble_steps, steps, rng)
        seed = int(rng.integers(2**31))
        S_Q[i] = reference_entropy_history(history, site=site, seed=seed)
        n_measured[i] = int(history.measured.sum())
    final = S_Q[:, -1].astype(float)
    return {
        "S_Q": S_Q,
        "n_measured": n_measured,
        "N": N,
        "p": p,
        "S_Q_mean": float(final.mean()),
        "S_Q_err": float(final.std(ddof=1) / np.sqrt(L)) if L > 1 else float("nan"),
        "S_Q_t": S_Q.mean(axis=0).tolist(),
        "measurement_rate": float(n_measured.sum() / (L * N * steps)) if steps else 0.0,
        "n_samples": L,
        "scramble_steps": scramble_steps,
        "steps": steps,
    }


def _run_haar(
    N: int, p: float, scramble_steps: int, steps: int, L: int, rng: np.random.Generator
) -> dict[str, Any]:
    """The same probe with Haar gates, on the state vector (issue #257).

    ``S_Q`` is only kept at the final step, with its Renyi-2 value
    ``S_Q2``; ``n_measured`` is not recorded (0).
    """
    values = {"S_Q": np.zeros(L), "S_Q2": np.zeros(L)}
    for i in tqdm(range(L)):
        out = haar_reference_entropy(N, p, scramble_steps, steps, rng)
        for key in values:
            values[key][i] = out[key]
    results: dict[str, Any] = {
        "S_Q": values["S_Q"][:, None],
        "n_measured": np.zeros(L, dtype=int),
        "N": N,
        "p": p,
        "S_Q_t": [],
        "n_samples": L,
        "scramble_steps": scramble_steps,
        "steps": steps,
    }
    for key, data in values.items():
        results[f"{key}_mean"] = float(data.mean())
        results[f"{key}_err"] = (
            float(data.std(ddof=1) / np.sqrt(L)) if L > 1 else float("nan")
        )
    return results


def save(results: dict[str, Any], out_dir: Path) -> Path:
    """Write ``purification.npz`` and ``metrics.json`` to ``out_dir``.

    Args:
        results: Output of :func:`run`.
        out_dir: Destination folder, created if needed.

    Returns:
        Path of the ``.npz`` file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{Path(__file__).stem}.npz"
    np.savez_compressed(npz_path, S_Q=results["S_Q"], n_measured=results["n_measured"])
    metrics = {
        key: value for key, value in results.items() if key not in ("S_Q", "n_measured")
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return npz_path


@hydra.main(
    version_base="1.3",
    config_path="../../conf",
    config_name="purification_gullans_huse",
)
def main(cfg: DictConfig) -> None:
    """Run one ``(N, p)`` and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/purification_gullans_huse.yaml``
            and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    save_config(cfg, out_dir)
    print(f"N: {results['N']}  p: {results['p']:.4g}  samples: {results['n_samples']}")
    print(f"S_Q(t = 4N): {results['S_Q_mean']:.4f} +- {results['S_Q_err']:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
