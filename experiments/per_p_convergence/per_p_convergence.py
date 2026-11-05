#!/usr/bin/env python3
"""Statistical convergence of the QELM ridge in ``PER_P`` (issue #267).

Follow-up of PR #261 (``experiments/haar_readouts``). There, the ridge of the
QELM reads the charge sector of a monitored Haar circuit from the
observables of the final state, with ``PER_P`` (``samples_per_p``)
trajectories at each measurement probability ``p``. This experiment asks
how many trajectories are needed for that accuracy to stop moving, at every
``p``, and averages over independent Haar circuits.

For each of ``convergence.n_circuits`` independent circuits ``c`` (one fixed
draw of Haar gates each, the same for every ``p`` and every trajectory) and
each ``p`` of ``convergence.p_values``:

1. Simulate ``max(convergence.per_p_values)`` trajectories, as in
   ``haar_readouts``: start from :class:`AncillaChargeSectorEncoding`, run
   ``convergence.n_rounds`` rounds (one brickwork period of the gates of
   ``c``, then a Z measurement of each system qubit with probability
   ``p``), take the entropy ``S_R`` of ``R``, measure ``R`` with the Born
   rule (the label ``Y = +-1``) and read the QELM features of the system
   state: ``<P_k>`` and ``<P_i P_j>`` (``i < j``) for ``P`` in ``X, Y, Z``
   (the set ``all`` of ``haar_readouts``, ``3 N + 3 N (N - 1) / 2``
   features, 108 at ``N = 8``). The features are exact expectation values
   of the trajectory state: ``PER_P`` counts trajectories (training
   samples), not shots per feature.
2. For each ``PER_P`` of ``convergence.per_p_values``, keep the first
   ``PER_P`` trajectories (the sets are nested, so two values of ``PER_P``
   differ only by the added trajectories), split them 80 % / 20 %
   (``samples.train_fraction``, stratified on ``Y``), fit ``RidgeCV`` (its
   ``alpha`` chosen by leave-one-out over :data:`ALPHAS`) and score the
   sign of its prediction on the test set. Also stored: the training
   accuracy, the chosen ``alpha`` and the eigenvalues of the covariance
   ``X^T X / n`` of the centered training features, with their effective
   rank.

All the trajectories of one ``(c, p)`` are simulated
together, as one array of state vectors (the gates are the same for every
trajectory of a fixed circuit), which is what makes 32 000 trajectories per
point affordable.

Statistics over the circuits, per ``(p, PER_P)``: the mean accuracy
``mu``, its spread ``sigma`` between circuits and the standard error
``sigma / sqrt(n_circuits)``. The saturation threshold ``PER_P*`` of a
``p`` is the smallest ``PER_P`` of the grid from which
``|mu(PER_P) - mu(2 PER_P)| < convergence.tolerance`` holds for every
larger pair of the grid (:func:`saturation_threshold`).

Run from the repository root:

    uv run python experiments/per_p_convergence/per_p_convergence.py \\
        convergence.n_circuits=4 'convergence.per_p_values=[500,1000,2000]'

Everything is written to the Hydra run directory, ``outputs/<date>/<time>/``
by default, which git ignores:

- ``per_p_convergence.npz``: the arrays of :func:`run` (axes
  ``(circuit, p, PER_P)``);
- ``summary.json``: :func:`summarize`, the statistics over the circuits and
  the thresholds;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import RidgeCV

from QELM import check_qubit_counts
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.evaluation.entropy_regression import stratified_split
from QELM.models.circuits.haar import HaarBrickworkCircuit

# Seed tag of the generators of this experiment.
RNG_TAG = 267
# Regularizations tried by the ridge, as in haar_readouts.
ALPHAS = np.logspace(-2, 4, 13)
# Names of the feature blocks, in the order of haar_readouts.OBSERVABLE_SETS.
OBSERVABLE_SETS = ("X", "Y", "Z", "XX", "YY", "ZZ")


# --------------------------------------------------------------------------
# Batched state-vector simulation: psi has shape (B, 2**n), qubit 0 is the
# most significant bit (the leftmost factor), as everywhere in QELM.
# --------------------------------------------------------------------------


def apply_gate_batch(psi: np.ndarray, gate: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply the same two-qubit ``gate`` on qubits ``i`` and ``j`` of every state.

    Args:
        psi: States, shape ``(B, 2**n)``.
        gate: ``4 x 4`` unitary in the basis ``|b_i b_j>``.
        i: First qubit, the most significant bit of the gate index.
        j: Second qubit.

    Returns:
        The states ``G psi``, shape ``(B, 2**n)``.
    """
    B, dim = psi.shape
    n = dim.bit_length() - 1
    T = np.moveaxis(psi.reshape((B,) + (2,) * n), (i + 1, j + 1), (0, 1))
    shape = T.shape
    T = (gate @ T.reshape(4, -1)).reshape(shape)
    return np.ascontiguousarray(np.moveaxis(T, (0, 1), (i + 1, j + 1))).reshape(B, dim)


def measure_batch(
    psi: np.ndarray, qubit: int, mask: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Measure ``qubit`` in Z in the states selected by ``mask``.

    The rule of :func:`~QELM.training.measurements.projective.measure_projective_psi`:
    outcome ``|0>`` when ``u < p0`` (Born probability of ``|0>``), then
    projection and normalization.

    Args:
        psi: States, shape ``(B, 2**n)``.
        qubit: Measured qubit.
        mask: Which states are measured, boolean ``(B,)``.
        u: Uniform draws in ``[0, 1)``, ``(B,)``; only the masked ones are read.

    Returns:
        ``(psi, outcomes)``: the states after the measurement (unchanged where
        ``mask`` is false) and the outcomes, ``+1`` for ``|0>``, ``-1`` for
        ``|1>`` and ``0`` where ``mask`` is false.
    """
    B, dim = psi.shape
    n = dim.bit_length() - 1
    view = psi.reshape(B, 2**qubit, 2, 2 ** (n - qubit - 1))
    weights = np.sum(np.abs(view) ** 2, axis=(1, 3))
    p0 = weights[:, 0] / np.sum(weights, axis=1)
    zero = u < p0
    outcomes = np.where(mask, np.where(zero, 1.0, -1.0), 0.0)
    out = view.copy()
    keep = np.where(zero, 0, 1)
    rows = np.flatnonzero(mask)
    if rows.size:
        kept = out[rows, :, keep[rows], :]
        norm = np.sqrt(np.sum(np.abs(kept) ** 2, axis=(1, 2)))
        out[rows] = 0.0
        out[rows, :, keep[rows], :] = kept / norm[:, None, None]
    return out.reshape(B, dim), outcomes


def apply_pauli_batch(psi: np.ndarray, name: str, qubit: int) -> np.ndarray:
    """Return ``P_qubit psi`` for the Pauli ``name`` in ``X, Y, Z``."""
    B, dim = psi.shape
    n = dim.bit_length() - 1
    view = psi.reshape(B, 2**qubit, 2, 2 ** (n - qubit - 1))
    # z holds the Z eigenvalue of the output bit, +1 for |0>, -1 for |1>.
    z = np.array([1.0, -1.0])[None, None, :, None]
    if name == "Z":
        out = view * z
    elif name == "X":
        out = view[:, :, ::-1, :]
    elif name == "Y":
        # Y|0> = i|1>, Y|1> = -i|0>: the output bit b gets -i z_b.
        out = -1j * z * view[:, :, ::-1, :]
    else:
        raise ValueError(f"Unknown Pauli {name!r}; expected X, Y or Z.")
    return np.asarray(out).reshape(B, dim)


def qelm_features(phi: np.ndarray, N: int) -> np.ndarray:
    """Return the QELM features of the system states ``phi``.

    ``<P_k>`` for ``k = 0 .. N-1``, then ``<P_i P_j>`` for ``i < j``, for the
    sets ``X, Y, Z, XX, YY, ZZ`` in that order: the set ``all`` of
    ``haar_readouts``, ``3 N + 3 N (N - 1) / 2`` columns.

    Args:
        phi: Normalized states of the ``N`` system qubits, ``(B, 2**N)``.
        N: Number of system qubits.

    Returns:
        Real array of shape ``(B, 3 N + 3 N (N - 1) / 2)``.
    """
    columns: list[np.ndarray] = []
    single = {
        name: [apply_pauli_batch(phi, name, k) for k in range(N)] for name in "XYZ"
    }
    for name in OBSERVABLE_SETS:
        if len(name) == 1:
            for k in range(N):
                columns.append(np.real(np.sum(phi.conj() * single[name][k], axis=1)))
        else:
            for i in range(N):
                left = apply_pauli_batch(phi, name[0], i)
                for j in range(i + 1, N):
                    right = single[name[1]][j]
                    columns.append(np.real(np.sum(left.conj() * right, axis=1)))
    return np.stack(columns, axis=1)


def reference_entropy_batch(psi: np.ndarray) -> np.ndarray:
    """Von Neumann entropy (bits) of qubit 0 in every state of ``psi``."""
    B, dim = psi.shape
    A = psi.reshape(B, 2, dim // 2)
    rho = np.einsum("bik,bjk->bij", A, A.conj())
    eig = np.clip(np.linalg.eigvalsh(rho), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(eig > 1e-15, -eig * np.log2(eig), 0.0)
    return np.asarray(np.sum(terms, axis=1))


def simulate(
    psi0: np.ndarray,
    circuit: HaarBrickworkCircuit,
    p: float,
    n_traj: int,
    rng: np.random.Generator,
    batch_size: int = 2000,
) -> dict[str, np.ndarray]:
    """Run ``n_traj`` trajectories of the fixed ``circuit`` at probability ``p``.

    The gates of every trajectory are those of ``circuit`` (a fixed draw);
    only the measured sites and the outcomes differ. The trajectories are
    simulated ``batch_size`` at a time, and the generator is read in the
    same order whatever ``batch_size`` (one batch after the other).

    Args:
        psi0: Initial state of ``R`` and the system, ``(2**N_total,)``.
        circuit: The gates, ``circuit.depth`` rounds.
        p: Measurement probability of each system qubit in each round.
        n_traj: Number of trajectories.
        rng: Generator of the measured sites and the outcomes.
        batch_size: Trajectories simulated together.

    Returns:
        ``Y`` (the Z outcome of ``R``, ``+-1``), ``S_R`` (bits, before that
        measurement), ``n_measured`` (measurements per trajectory) and
        ``features`` (:func:`qelm_features` of the system state after ``R``
        is measured), one row per trajectory.
    """
    N = circuit.N
    out: dict[str, list[np.ndarray]] = {
        "Y": [],
        "S_R": [],
        "n_measured": [],
        "features": [],
    }
    for start in range(0, n_traj, batch_size):
        B = min(batch_size, n_traj - start)
        psi = np.tile(np.asarray(psi0, dtype=complex), (B, 1))
        n_measured = np.zeros(B)
        for t in range(circuit.depth):
            for i, j, gate in circuit.layers[t]:
                psi = apply_gate_batch(psi, gate, i, j)
            for k in range(N):
                mask = rng.random(B) < p
                u = rng.random(B)
                psi, _ = measure_batch(psi, k + circuit.Nr, mask, u)
                n_measured += mask
        S_R = reference_entropy_batch(psi)
        psi, outcome = measure_batch(psi, 0, np.ones(B, dtype=bool), rng.random(B))
        rows = psi.reshape(B, 2, 2**N)
        phi = np.where((outcome > 0)[:, None], rows[:, 0], rows[:, 1])
        out["Y"].append(outcome)
        out["S_R"].append(S_R)
        out["n_measured"].append(n_measured)
        out["features"].append(qelm_features(phi, N))
    return {key: np.concatenate(values) for key, values in out.items()}


# --------------------------------------------------------------------------
# Readout and its diagnostics.
# --------------------------------------------------------------------------


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Participation ratio ``(sum l)^2 / sum l^2`` of non-negative eigenvalues.

    It counts the directions that hold the variance: ``d`` for ``d`` equal
    eigenvalues, 1 when one dominates, 0 for an empty spectrum.
    """
    lam = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    total = float(np.sum(lam**2))
    return 0.0 if total == 0.0 else float(np.sum(lam) ** 2 / total)


def ridge_scores(
    X: np.ndarray,
    Y: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Fit the ridge on a stratified split of ``(X, Y)`` and score it.

    Args:
        X: Features, ``(L, n_features)``.
        Y: Labels, ``+-1``.
        train_fraction: Training share, stratified on ``Y``.
        rng: Generator of the split.

    Returns:
        ``accuracy`` (test), ``accuracy_train``, ``alpha`` (chosen by
        ``RidgeCV``, ``nan`` when nothing is fitted), ``eigenvalues`` (of the
        covariance of the centered training features, descending) and
        ``effective_rank`` (:func:`effective_rank`).
    """
    train, test = stratified_split(Y, train_fraction, rng)
    Xtr, Ytr = X[train], Y[train]
    centered = Xtr - Xtr.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T @ centered / max(1, len(train)))[::-1]
    eig = np.clip(eig, 0.0, None)
    if np.unique(Ytr).size < 2 or np.ptp(Xtr, axis=0).max() < 1e-12:
        # Nothing to learn from: the output is the mean training label.
        pred_te = np.full(test.size, float(np.mean(Ytr)))
        pred_tr = np.full(train.size, float(np.mean(Ytr)))
        alpha = float("nan")
    else:
        model = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr)
        pred_te = model.predict(X[test])
        pred_tr = model.predict(Xtr)
        alpha = float(model.alpha_)
    sign_te = np.where(pred_te >= 0.0, 1.0, -1.0)
    sign_tr = np.where(pred_tr >= 0.0, 1.0, -1.0)
    return {
        "accuracy": float(np.mean(sign_te == Y[test])),
        "accuracy_train": float(np.mean(sign_tr == Ytr)),
        "alpha": alpha,
        "eigenvalues": eig,
        "effective_rank": effective_rank(eig),
    }


def saturation_threshold(
    per_p_values: np.ndarray | list[int], mu: np.ndarray, tolerance: float
) -> int | None:
    """Smallest ``PER_P`` from which the mean accuracy has stopped moving.

    Args:
        per_p_values: Increasing grid, each value twice the previous one.
        mu: Mean accuracy at each value of the grid.
        tolerance: Largest allowed ``|mu(P) - mu(2 P)|``.

    Returns:
        The smallest ``P`` of the grid such that every pair ``(P', 2 P')``
        with ``P' >= P`` moves by less than ``tolerance``; ``None`` when the
        last pair still moves (no threshold inside the grid).
    """
    grid = np.asarray(per_p_values)
    steps = np.abs(np.diff(np.asarray(mu, dtype=float)))
    threshold: int | None = None
    for k in range(steps.size - 1, -1, -1):
        if steps[k] < tolerance:
            threshold = int(grid[k])
        else:
            break
    return threshold


# --------------------------------------------------------------------------
# The experiment.
# --------------------------------------------------------------------------


def _settings(cfg: DictConfig) -> dict[str, Any]:
    """Read and check the ``convergence`` block of ``cfg``."""
    N, Nr, _ = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"per_p_convergence needs Nr=1, got {Nr}.")
    task = cfg.convergence
    p_values = np.asarray([float(v) for v in task.p_values])
    if p_values.size == 0 or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError(f"convergence.p_values must be in [0, 1], got {p_values}.")
    if np.unique(p_values).size != p_values.size:
        raise ValueError(f"convergence.p_values must be distinct, got {p_values}.")
    per_p = np.asarray([int(v) for v in task.per_p_values])
    if per_p.size == 0 or per_p[0] < 10 or np.any(np.diff(per_p) <= 0):
        raise ValueError(
            f"convergence.per_p_values must increase and start at >= 10, got {per_p}."
        )
    n_circuits = int(task.n_circuits)
    n_rounds = int(task.n_rounds)
    if n_circuits < 1 or n_rounds < 1:
        raise ValueError(
            "convergence.n_circuits and convergence.n_rounds must be >= 1, got "
            f"{n_circuits}, {n_rounds}."
        )
    ids = task.get("circuit_ids")
    circuit_ids = list(range(n_circuits)) if ids is None else [int(c) for c in ids]
    if (
        not circuit_ids
        or min(circuit_ids) < 0
        or len(set(circuit_ids)) != len(circuit_ids)
    ):
        raise ValueError(
            f"convergence.circuit_ids must be distinct indices >= 0, got {circuit_ids}."
        )
    return {
        "N": N,
        "seed": int(cfg.seed),
        "circuit_ids": circuit_ids,
        "p_values": p_values,
        "per_p_values": per_p,
        "n_circuits": n_circuits,
        "n_rounds": n_rounds,
        "train_fraction": float(cfg.samples.train_fraction),
        "batch_size": int(task.get("batch_size", 2000)),
        "boundary": str(cfg.haar_circuit.get("boundary", "open")),
        "psi0": AncillaChargeSectorEncoding.from_config(cfg).state(),
    }


def run_circuit(settings: dict[str, Any], c: int) -> dict[str, np.ndarray]:
    """Simulate and score circuit ``c`` at every ``p`` and every ``PER_P``.

    Every generator is seeded by ``(seed, RNG_TAG, c, ...)``, so circuit
    ``c`` gives the same numbers whatever the other circuits, the number of
    processes or the order in which they run.
    """
    seed = settings["seed"]
    circuit = HaarBrickworkCircuit(
        {
            "N": settings["N"],
            "Nr": 1,
            "depth": settings["n_rounds"],
            "boundary": settings["boundary"],
            "fixed_instance": True,
        },
        np.random.default_rng([seed, RNG_TAG, c, 0]),
    )
    per_p = settings["per_p_values"]
    n_p, n_P = len(settings["p_values"]), len(per_p)
    n_features = 3 * settings["N"] + 3 * settings["N"] * (settings["N"] - 1) // 2
    res = {
        "accuracy": np.zeros((n_p, n_P)),
        "accuracy_train": np.zeros((n_p, n_P)),
        "alpha": np.zeros((n_p, n_P)),
        "effective_rank": np.zeros((n_p, n_P)),
        "eigenvalues": np.zeros((n_p, n_P, n_features)),
        "S_R": np.zeros(n_p),
        "n_measured": np.zeros(n_p),
    }
    for a, p in enumerate(settings["p_values"]):
        traj = simulate(
            settings["psi0"],
            circuit,
            float(p),
            int(per_p[-1]),
            np.random.default_rng([seed, RNG_TAG, c, 1, a]),
            settings["batch_size"],
        )
        res["S_R"][a] = traj["S_R"].mean()
        res["n_measured"][a] = traj["n_measured"].mean()
        for b, P in enumerate(per_p):
            scores = ridge_scores(
                traj["features"][:P],
                traj["Y"][:P],
                settings["train_fraction"],
                np.random.default_rng([seed, RNG_TAG, c, 2, a, b]),
            )
            for key in ("accuracy", "accuracy_train", "alpha", "effective_rank"):
                res[key][a, b] = scores[key]
            res["eigenvalues"][a, b] = scores["eigenvalues"]
    return res


def _run_one(args: tuple[dict[str, Any], int]) -> dict[str, np.ndarray]:
    return run_circuit(*args)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Run every circuit and gather the arrays.

    Args:
        cfg: Composed config (``load_config(..., config_name=
            "per_p_convergence")`` or Hydra).

    Returns:
        ``p_values``, ``per_p_values`` and, with axes ``(circuit, p,
        PER_P)``, ``accuracy``, ``accuracy_train``, ``alpha``,
        ``effective_rank`` and ``eigenvalues`` (one more axis, descending),
        plus ``S_R`` and ``n_measured`` of axes ``(circuit, p)``, and
        ``circuit_ids``, the index of each circuit (``convergence.circuit_ids``,
        by default ``0 .. n_circuits - 1``).
    """
    settings = _settings(cfg)
    n_jobs = int(cfg.convergence.get("n_jobs", 1)) or (os.cpu_count() or 1)
    jobs = [(settings, c) for c in settings["circuit_ids"]]
    if n_jobs == 1:
        parts = [_run_one(job) for job in jobs]
    else:
        # spawn, the default on macOS: fork is unsafe in a threaded process.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as pool:
            parts = list(pool.map(_run_one, jobs))
    out: dict[str, Any] = {
        "p_values": settings["p_values"],
        "per_p_values": settings["per_p_values"],
        "circuit_ids": np.asarray(settings["circuit_ids"]),
    }
    for key in parts[0]:
        out[key] = np.stack([part[key] for part in parts])
    return out


def summarize(results: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Statistics over the circuits and the saturation thresholds.

    Args:
        results: Output of :func:`run`.
        tolerance: Criterion of :func:`saturation_threshold`.

    Returns:
        Per ``p`` (key ``f"{p:g}"``): ``mu``, ``sigma``, ``sem`` of the test
        accuracy and ``mu_train`` against ``PER_P``, ``delta`` (``mu(2 P) -
        mu(P)``), ``sem_delta`` (standard error of that difference, from the
        paired differences of each circuit), ``effective_rank``,
        ``alpha_median``, ``S_R`` and ``per_p_star``; and ``per_p_star``,
        the largest threshold over ``p`` (``None`` if one is missing), the
        smallest ``PER_P`` that converges at every ``p``.
    """
    grid = np.asarray(results["per_p_values"])
    acc = np.asarray(results["accuracy"])
    n_c = acc.shape[0]
    by_p: dict[str, Any] = {}
    stars: list[int | None] = []
    for a, p in enumerate(results["p_values"]):
        mu = acc[:, a].mean(axis=0)
        sigma = acc[:, a].std(axis=0, ddof=1) if n_c > 1 else np.zeros_like(mu)
        diffs = np.diff(acc[:, a], axis=1)
        sem_delta = (
            diffs.std(axis=0, ddof=1) / np.sqrt(n_c)
            if n_c > 1
            else np.zeros(diffs.shape[1])
        )
        star = saturation_threshold(grid, mu, tolerance)
        stars.append(star)
        by_p[f"{float(p):g}"] = {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "sem": (sigma / np.sqrt(n_c)).tolist(),
            "mu_train": np.asarray(results["accuracy_train"])[:, a]
            .mean(axis=0)
            .tolist(),
            "delta": np.diff(mu).tolist(),
            "sem_delta": sem_delta.tolist(),
            "effective_rank": np.asarray(results["effective_rank"])[:, a]
            .mean(axis=0)
            .tolist(),
            "alpha_median": np.nanmedian(
                np.asarray(results["alpha"])[:, a], axis=0
            ).tolist(),
            "S_R": float(np.mean(np.asarray(results["S_R"])[:, a])),
            "per_p_star": star,
        }
    known = [s for s in stars if s is not None]
    universal = max(known) if len(known) == len(stars) else None
    return {
        "per_p_values": grid.tolist(),
        "n_circuits": int(n_c),
        "tolerance": float(tolerance),
        "by_p": by_p,
        "per_p_star": universal,
    }


@hydra.main(
    version_base="1.3", config_path="../../conf", config_name="per_p_convergence"
)
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, cfg, out_dir)


def save(results: dict[str, Any], cfg: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Write ``per_p_convergence.npz``, ``summary.json`` and the config.

    Returns:
        The summary of :func:`summarize`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "per_p_convergence.npz", **results)
    summary = summarize(results, float(cfg.convergence.tolerance))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "resolved_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
    )
    print(
        f"{'p':>6} {'S_R':>6} " + " ".join(f"{P:>7d}" for P in results["per_p_values"])
    )
    for key, row in summary["by_p"].items():
        print(
            f"{float(key):6.3f} {row['S_R']:6.3f} "
            + " ".join(f"{m:7.3f}" for m in row["mu"])
            + f"   PER_P* = {row['per_p_star']}"
        )
    print(f"PER_P* (every p): {summary['per_p_star']}")
    print(f"Saved to {out_dir}")
    return summary


# Arrays of run() with one row per circuit.
PER_CIRCUIT = (
    "circuit_ids",
    "accuracy",
    "accuracy_train",
    "alpha",
    "effective_rank",
    "eigenvalues",
    "S_R",
    "n_measured",
)


def merge(folders: list[str | Path], out_dir: str | Path) -> dict[str, Any]:
    """Merge runs of the same sweep over different circuits into one run.

    For a cluster, one job per subset of circuits
    (``convergence.circuit_ids``); this gathers them. The runs must share
    ``N``, the seed, the rounds and the grids, and hold distinct circuits.

    Args:
        folders: Run folders written by :func:`save`.
        out_dir: Folder of the merged run, written by :func:`save`.

    Returns:
        The summary of the merged run.

    Raises:
        ValueError: If the runs are not of the same sweep or share a circuit.
    """
    parts, cfgs = [], []
    for folder in folders:
        with np.load(Path(folder) / "per_p_convergence.npz") as data:
            parts.append({key: data[key] for key in data.files})
        cfgs.append(json.loads((Path(folder) / "resolved_config.json").read_text()))

    def key(cfg: dict[str, Any]) -> tuple[Any, ...]:
        conv = cfg["convergence"]
        return (
            cfg["N"],
            cfg["seed"],
            conv["n_rounds"],
            list(conv["p_values"]),
            list(conv["per_p_values"]),
            cfg["samples"]["train_fraction"],
        )

    if any(key(cfg) != key(cfgs[0]) for cfg in cfgs):
        raise ValueError("The runs to merge must share N, seed, rounds and grids.")
    ids = np.concatenate([part["circuit_ids"] for part in parts])
    if np.unique(ids).size != ids.size:
        raise ValueError(f"The runs to merge share circuits: {sorted(ids)}.")
    order = np.argsort(ids)
    results: dict[str, Any] = {
        "p_values": parts[0]["p_values"],
        "per_p_values": parts[0]["per_p_values"],
    }
    for name in PER_CIRCUIT:
        results[name] = np.concatenate([part[name] for part in parts])[order]
    cfg = OmegaConf.create(cfgs[0])
    cfg.convergence.n_circuits = int(ids.size)
    cfg.convergence.circuit_ids = [int(c) for c in ids[order]]
    return save(results, cfg, Path(out_dir))


if __name__ == "__main__":
    main()
