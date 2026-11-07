"""Fourier expressivity of the monitored QELM on one fixed circuit (issue #309).

For one dynamics, one size ``N``, one depth ``T`` and one seed (one fixed
draw of the gates), compute the transfer matrix ``C(p)`` of the record
features ``M_{t,i}`` and of the final ``Z_i`` on the ``3**N`` Bloch monomials
of the angle encoding, for every ``p`` of a grid, and write its measures
(:func:`QELM.evaluation.fourier.expressivity_measures`) to
``<out>/metrics.json``. These measures are exact: no trajectory is sampled.

With ``--task-samples n`` (default 3000; 0 turns it off), the QELM itself is
also scored, on targets spread over the monomials of one degree ``k``
(:mod:`QELM.evaluation.fourier_tasks`): ``n`` inputs with angles uniform on
``[low, high]``, one single-shot record each, a ridge fitted on the first
``train_frac`` of them, its test R2 and test accuracy; next to it, the R2
predicted from ``C(p)`` and the measured shot noise for ``S`` averaged
shots, and the ceiling ``S = inf``.

    uv run python experiments/fourier_expressivity/fourier_expressivity.py \
        --model haar --N 6 --T 6 --seed 42 --out outputs/fourier/one_run

The launcher ``fourier_expressivity.sh`` runs a grid of these.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from QELM.evaluation.fourier import (
    bloch_monomials,
    expressivity_measures,
    transfer_matrix,
)
from QELM.evaluation.fourier_tasks import (
    degree_target,
    gaussian_accuracy,
    predicted_r2,
    qelm_scores,
    sample_records,
    shot_noise_covariance,
)
from QELM.models.circuits import (
    CliffordBrickworkCircuit,
    HaarBrickworkCircuit,
    U1BrickworkCircuit,
)
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.symmetric import Z2BrickworkCircuit

MODELS: dict[str, type[BrickworkCircuit]] = {
    "haar": HaarBrickworkCircuit,
    "u1": U1BrickworkCircuit,
    "z2": Z2BrickworkCircuit,
    "clifford": CliffordBrickworkCircuit,
}


SHOTS = (1, 10, 100, 1000)


def task_scores(
    layers: list,
    C: np.ndarray,
    N: int,
    p: float,
    theta: np.ndarray,
    targets: dict[int, tuple[np.ndarray, float]],
    n_train: int,
    alpha: float,
    rng: np.random.Generator,
    low: float,
    high: float,
) -> dict[str, dict[str, float]]:
    """Scores of the QELM and their Fourier predictions, per target degree."""
    records = sample_records(layers, N, p, theta, rng)
    noise = shot_noise_covariance(records, C, theta)
    B = bloch_monomials(theta)
    out: dict[str, dict[str, float]] = {}
    for k, (yhat, offset) in targets.items():
        y = B @ yhat - offset
        row: dict[str, float] = dict(qelm_scores(records, y, n_train, alpha))
        for shots in SHOTS:
            row[f"r2_pred_S{shots}"] = predicted_r2(C, yhat, noise, shots, N, low, high)
        row["r2_ideal"] = predicted_r2(C, yhat, noise, np.inf, N, low, high)
        row["accuracy_pred_S1"] = gaussian_accuracy(row["r2_pred_S1"])
        out[f"k{k}"] = row
    return out


def run(
    model: str,
    N: int,
    T: int,
    seed: int,
    p_values: list[float],
    low: float = 0.0,
    high: float = float(np.pi),
    task_samples: int = 0,
    train_frac: float = 0.75,
    degrees: tuple[int, ...] = (1, 2, 3),
    alpha: float = 1e-3,
) -> dict[str, object]:
    """Return the measures of ``C(p)`` for each ``p``, on one circuit.

    With ``task_samples > 0``, each ``p`` also carries ``tasks``: the scores
    of :func:`task_scores` for the degrees ``k <= N`` of ``degrees``. The
    targets and the inputs are drawn once per run (seed ``seed + 10000``),
    the same for every ``p``.
    """
    rng = np.random.default_rng(seed)
    circuit = MODELS[model]({"N": N, "depth": T, "Nr": 0}, rng)
    task_rng = np.random.default_rng(seed + 10_000)
    targets = {k: degree_target(N, k, task_rng, low, high) for k in degrees if k <= N}
    theta = task_rng.uniform(low, high, (task_samples, N))
    n_train = int(round(train_frac * task_samples))
    per_p = []
    for p in p_values:
        C = transfer_matrix(circuit.layers, N, p)
        row: dict[str, object] = {"p": p, **expressivity_measures(C, N, low, high)}
        if task_samples > 0:
            row["tasks"] = task_scores(
                circuit.layers,
                C,
                N,
                p,
                theta,
                targets,
                n_train,
                alpha,
                task_rng,
                low,
                high,
            )
        per_p.append(row)
    return {
        "model": model,
        "N": N,
        "T": T,
        "seed": seed,
        "angle_range": [low, high],
        "n_features": T * N + N,
        "n_monomials": 3**N,
        "task_samples": task_samples,
        "n_train": n_train,
        "alpha": alpha,
        "shots": list(SHOTS),
        "per_p": per_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--p", type=float, nargs="+", default=list(np.round(np.linspace(0, 1, 21), 3))
    )
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=float(np.pi))
    parser.add_argument("--task-samples", type=int, default=3000)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start = time.time()
    result = run(
        args.model,
        args.N,
        args.T,
        args.seed,
        args.p,
        args.low,
        args.high,
        args.task_samples,
        args.train_frac,
        tuple(args.degrees),
        args.alpha,
    )
    result["seconds"] = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'p':>5} {'rank':>5} {'k_mean':>7} {'d_eff':>6} {'odd_x':>6} {'R2_k1':>6}")
    for row in result["per_p"]:  # type: ignore[attr-defined]
        r2 = row.get("tasks", {}).get("k1", {}).get("r2_test", float("nan"))
        print(
            f"{row['p']:5.2f} {row['rank']:5d} {row['mean_degree']:7.3f}"
            f" {row['d_eff']:6.2f} {row['odd_x_weight']:6.3f} {r2:6.3f}"
        )
    print(f"Saved to {args.out} in {result['seconds']:.1f} s")


if __name__ == "__main__":
    main()
