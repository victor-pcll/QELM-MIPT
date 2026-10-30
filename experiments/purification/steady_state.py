#!/usr/bin/env python3
"""Steady-state entanglement of the monitored Clifford brickwork (issue #257).

The two other probes of the measurement-induced transition, on the same
circuit as the reference-qubit probe of ``purification.py`` (same gates, same
half-layer step, periodic boundaries, gates redrawn for every trajectory):

- ``S_half``, the entropy of half the chain (Li, Chen & Fisher, PRB 100,
  134306, 2019): volume law below ``p_c``, area law above, ``~ log N`` at
  ``p_c``; ``p_c`` comes from the collapse of ``S_half(p, N) - S_half(p_c, N)``;
- ``I3``, the tripartite mutual information of three contiguous quarters
  (Zabalo et al., PRB 101, 060301, 2020): negative and extensive below
  ``p_c``, 0 above, and independent of ``N`` at ``p_c``, so the curves of
  different ``N`` cross there without any fit.

There is no reference qubit: the chain starts in ``|0...0>`` and is monitored
at rate ``p`` from the first step, for ``probe.scramble_steps + probe.steps``
steps (``4N`` by default, the same total time as the reference probe). The
entropies are taken at the final step
(:func:`QELM.models.circuits.stabilizer.steady_state_entropies`).
For a stabilizer state every Renyi entropy is the same. With
``probe.dynamics=haar`` the gates are Haar ``U(4)`` and the state is a state
vector (:mod:`QELM.models.circuits.monitored`); the Renyi-2 values
``S_half2`` and ``I3_2`` are written too.

    uv run python experiments/purification/steady_state.py N=16 probe.p=0.16

Writes to the Hydra run directory ``steady_state.npz`` (``S_half`` and
``I3`` per trajectory), ``metrics.json`` (``N``, ``p``, ``S_half_mean``,
``I3_mean``, their standard errors ``_err``, ``n_samples``, ``steps``) and
the config.
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
from QELM.models.circuits import draw_probe_history
from QELM.models.circuits.monitored import haar_steady_state_entropies
from QELM.models.circuits.stabilizer import quarters, steady_state_entropies
from QELM.utils import save_config

KEYS = ("S_half", "I3")
# With Haar gates the Renyi-2 values differ from the von Neumann ones.
HAAR_KEYS = ("S_half", "I3", "S_half2", "I3_2")


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories at one ``(N, p)``.

    Args:
        cfg: Composed config ``purification_gullans_huse``. Reads ``seed``,
            ``samples.L`` and ``probe``.

    Returns:
        ``S_half`` and ``I3`` (shape ``(L,)``) and the scalars written to
        ``metrics.json``.
    """
    N, _, _ = check_qubit_counts(cfg)
    quarters(N)  # checks that N is divisible by 4
    p = float(cfg.probe.p)
    steps = int(cfg.probe.scramble_steps) + int(cfg.probe.steps)
    L = int(cfg.samples.L)
    # Not the stream of purification.py: the two probes are independent.
    rng = np.random.default_rng([int(cfg.seed), N, round(p * 1e9), 1])
    haar = str(cfg.probe.get("dynamics", "clifford")) == "haar"
    keys = HAAR_KEYS if haar else KEYS
    out = {key: np.zeros(L) for key in keys}
    for i in tqdm(range(L)):
        if haar:
            values = haar_steady_state_entropies(N, p, steps, rng)
        else:
            history = draw_probe_history(N, p, 0, steps, rng)
            values = steady_state_entropies(history, seed=int(rng.integers(2**31)))
        for key in keys:
            out[key][i] = values[key]
    results: dict[str, Any] = {**out, "N": N, "p": p, "n_samples": L, "steps": steps}
    for key in keys:
        results[f"{key}_mean"] = float(out[key].mean())
        results[f"{key}_err"] = (
            float(out[key].std(ddof=1) / np.sqrt(L)) if L > 1 else float("nan")
        )
    return results


def save(results: dict[str, Any], out_dir: Path) -> Path:
    """Write ``steady_state.npz`` and ``metrics.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{Path(__file__).stem}.npz"
    keys = [key for key in HAAR_KEYS if key in results]
    np.savez_compressed(npz_path, **{key: results[key] for key in keys})
    metrics = {k: v for k, v in results.items() if k not in keys}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return npz_path


@hydra.main(
    version_base="1.3",
    config_path="../../conf",
    config_name="purification_gullans_huse",
)
def main(cfg: DictConfig) -> None:
    """Run one ``(N, p)`` and save it to the Hydra run directory."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    save_config(cfg, out_dir)
    print(f"N: {results['N']}  p: {results['p']:.4g}  samples: {results['n_samples']}")
    for key in HAAR_KEYS:
        if f"{key}_mean" in results:
            mean, err = results[f"{key}_mean"], results[f"{key}_err"]
            print(f"{key}: {mean:.4f} +- {err:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
