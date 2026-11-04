#!/usr/bin/env python3
"""Tomography and purity of the reference qubit from the record M (issue #277).

The task of ``experiments/nonlinear_M`` (#269): a reference qubit ``R`` in a
Bell pair with the middle site of the chain, Haar U(4) brickwork gates (one
fixed circuit per seed), Z measurements with probability ``p``. Instead of
the Z outcome of ``R``, the readouts estimate two continuous targets of the
state ``rho_R`` of ``R`` given the record ``M`` and the circuit:

1. the **Bloch vector** ``r = (<X_R>, <Y_R>, <Z_R>)``, that is ``rho_R``
   itself (tomography);
2. the **purity** ``Tr(rho_R^2) = (1 + |r|^2) / 2``, from 1/2 (``R`` still
   maximally entangled) to 1 (``R`` purified by the record).

For each ``p`` of ``bloch_task.p_values`` and each of its
``bloch_task.samples_per_p`` trajectories, :func:`trajectory` runs the
circuit and the measurements of ``nonlinear_M`` and returns the record
``M`` (``(T, N)``, +-1 and 0 for an unmeasured qubit), the exact Bloch
vector ``r(M)`` and the final state. ``R`` is **not** measured: the targets
are the exact conditional quantities, as in ``reference_bloch`` of #275.

The readouts, each fitted on ``samples.train_fraction`` of the trajectories
of one ``p`` and scored on the rest, are ridges (``RidgeCV``, alpha by
leave-one-out over :data:`ALPHAS`, one alpha per target). None of them
simulates the circuit:

- ``M``, the **QELM on M**: the ridge on ``[vec(M), vec(|M|)]`` (mode
  ``raw_abs`` of :data:`QELM.features.MODES`), the one-hot record
  ``{unmeasured, +1, -1}`` of each site and round. ``vec(M)`` alone cannot
  read the purity: it is even in the outcomes, set by *where* the chain was
  measured, which only ``|M|`` sees;
- ``state``, the **QELM on the state**: the ridge on the QELM features of
  the final chain state, ``<P_k>`` and ``<P_i P_j>`` (``P`` in ``X, Y, Z``),
  before ``R`` is measured;
- ``M_state``, the **QELM on M and the state**: both side by side.

The state features carry no coherence between the two branches of ``R``
(``<P>`` of the chain alone), so the QELM on the state is expected near
``R2 = 0`` for the Bloch vector, unlike the Z task of #269 where ``R`` was
measured first and its outcome imprinted on the chain.

And the references:

- ``bayes``, **Bayes optimal on M**: the exact ``r(M)`` and ``Tr(rho_R^2)``
  of an observer who knows the circuit. They are the targets, so their
  ``R2`` is 1 by construction: the ceiling of any readout of ``M``;
- ``chance``: the mean of the training targets, ``R2 = 0`` up to the
  test-set fluctuation.

Scores, per ``p`` (key suffix ``_p<p>``):

- ``r2_bloch_<readout>``: ``R2`` of the vector, ``1 - sum |r_pred - r|^2 /
  sum |r - mean r|^2`` over the three components; ``r2_<c>_<readout>`` for
  each component ``c`` in ``x, y, z``;
- ``r2_purity_<readout>``: ``R2`` of a ridge fitted on the purity directly;
  ``r2_purity_from_bloch_<readout>``: ``R2`` of ``(1 + |r_pred|^2) / 2``,
  ``|r_pred|`` clipped to 1;
- ``purity_mean``, ``purity_std``, ``S_R_mean``: the exact purity and the
  entropy of ``R`` (bits), the purification curve.

Run from the repository root:

    uv run python experiments/bloch_purity/bloch_purity.py N=8 seed=42

Everything is written to the Hydra run directory: ``metrics.json``,
``bloch_purity.npz`` (``M``, ``bloch``, ``purity``, ``p``, unless
``bloch_task.save_arrays=false``) and the config.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from QELM import check_qubit_counts, set_seed
from QELM.features import extract_M_features
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config

HERE = Path(__file__).resolve().parent
# Seed tag of the generators of the gates, the measurements and the split.
RNG_TAG = 277
# Regularizations tried by the ridge.
ALPHAS = np.logspace(-3, 4, 15)
# Readouts fitted by a ridge, in the order of the figures.
READOUTS = ("M", "state", "M_state")
COMPONENTS = ("x", "y", "z")


def _nonlinear_module() -> ModuleType:
    """``experiments/nonlinear_M/nonlinear_M.py``: encodings, depth rule,
    QELM features."""
    path = HERE.parent / "nonlinear_M" / "nonlinear_M.py"
    spec = importlib.util.spec_from_file_location("nonlinear_M_for_bloch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NL = _nonlinear_module()


def reference_rho(psi: np.ndarray) -> np.ndarray:
    """Reduced density matrix of ``R`` (qubit 0, the most significant bit)."""
    half = psi.size // 2
    a, b = psi[:half], psi[half:]
    return np.array(
        [[np.vdot(a, a), np.vdot(b, a)], [np.vdot(a, b), np.vdot(b, b)]],
        dtype=complex,
    )


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """``(<X>, <Y>, <Z>)`` of a one-qubit density matrix."""
    return np.array(
        [
            2.0 * rho[0, 1].real,
            -2.0 * rho[0, 1].imag,
            (rho[0, 0] - rho[1, 1]).real,
        ]
    )


def purity_from_bloch(r: np.ndarray, clip: bool = False) -> np.ndarray:
    """``Tr(rho^2) = (1 + |r|^2) / 2`` along the last axis; ``clip`` caps
    ``|r|`` at 1 (a prediction can leave the Bloch ball)."""
    norm2 = np.sum(np.asarray(r) ** 2, axis=-1)
    if clip:
        norm2 = np.minimum(norm2, 1.0)
    return np.asarray((1.0 + norm2) / 2.0)


def trajectory(
    psi0: np.ndarray,
    circuit: HaarBrickworkCircuit,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one trajectory from ``psi0``; ``R`` is never measured.

    Returns:
        ``(M, r, psi)``: the record ``(depth, N)``, the Bloch vector of ``R``
        given the record, and the final state.
    """
    N, N_total = circuit.N, circuit.N_total
    psi = psi0
    M = np.zeros((circuit.depth, N))
    for t in range(circuit.depth):
        psi = circuit.apply_layer(psi, t)
        for k in range(N):
            if rng.random() < p:
                psi, M[t, k] = measure_projective_psi(psi, k + 1, N_total, rng)
    return M, bloch_vector(reference_rho(psi)), psi


def r2_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``R2`` of a vector target: ``1 - SSE / SST`` summed over the columns.

    NaN if the target does not vary (``p = 0``: ``r = 0`` on every
    trajectory).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    sst = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    if sst < 1e-12 * y_true.size:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / sst


def fit_ridge(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Ridge with one alpha per target, chosen by leave-one-out.

    Constant features (the empty record at ``p = 0``) give the mean target.
    """
    Y_tr = np.asarray(Y_tr, dtype=float)
    if X_tr.shape[1] == 0 or np.ptp(X_tr, axis=0).max() < 1e-12:
        return np.broadcast_to(Y_tr.mean(axis=0), (X_te.shape[0], *Y_tr.shape[1:]))
    model = RidgeCV(alphas=ALPHAS, alpha_per_target=Y_tr.ndim > 1).fit(X_tr, Y_tr)
    return np.asarray(model.predict(X_te)).reshape(X_te.shape[0], *Y_tr.shape[1:])


def score_level(
    features: dict[str, np.ndarray],
    bloch: np.ndarray,
    purity: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    """Scores of every readout on one value of ``p`` (keys without ``_p``)."""
    out: dict[str, float] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (
            fit_ridge(X[train], bloch[train], X[test]),
            fit_ridge(X[train], purity[train], X[test]),
        )
        for name, X in features.items()
    }
    predictions["bayes"] = (bloch[test], purity[test])
    predictions["chance"] = (
        np.broadcast_to(bloch[train].mean(axis=0), bloch[test].shape),
        np.full(test.size, purity[train].mean()),
    )
    for name, (r_pred, P_pred) in predictions.items():
        out[f"r2_bloch_{name}"] = r2_vector(bloch[test], r_pred)
        for c, comp in enumerate(COMPONENTS):
            out[f"r2_{comp}_{name}"] = r2_vector(bloch[test, c], r_pred[:, c])
        out[f"r2_purity_{name}"] = r2_vector(purity[test], P_pred)
        out[f"r2_purity_from_bloch_{name}"] = r2_vector(
            purity[test], purity_from_bloch(r_pred, clip=True)
        )
        out[f"mse_bloch_{name}"] = float(
            np.mean(np.sum((bloch[test] - r_pred) ** 2, axis=1))
        )
        out[f"mae_purity_{name}"] = float(np.mean(np.abs(purity[test] - P_pred)))
    return out


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories of every ``p``, fit and score the readouts.

    Args:
        cfg: Composed config (``load_config(..., config_name="bloch_purity")``
            or Hydra).

    Returns:
        ``M``, ``bloch``, ``purity``, ``p``, ``metrics`` and ``timing``.

    Raises:
        ValueError: If ``Nr`` is not 1, a probability is outside ``[0, 1]``
            or repeated, or ``samples_per_p`` is too small.
    """
    set_seed(cfg)
    N, Nr, _ = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"bloch_purity needs the reference qubit: Nr=1, got {Nr}.")
    task = cfg.bloch_task
    levels = np.asarray([float(v) for v in task.p_values])
    if levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"bloch_task.p_values must be in [0, 1], got {levels}.")
    if np.unique(levels).size != levels.size:
        raise ValueError(f"bloch_task.p_values must be distinct, got {levels}.")
    per_p = int(task.samples_per_p)
    if per_p < 10:
        raise ValueError(f"bloch_task.samples_per_p must be >= 10, got {per_p}.")
    T = NL.depth_from_rule(task.depth_rule, N)
    train_fraction = float(cfg.samples.train_fraction)

    seed = int(cfg.seed)
    gate_rng = np.random.default_rng([seed, RNG_TAG, 0])
    rng = np.random.default_rng([seed, RNG_TAG, 1])
    block = OmegaConf.to_container(cfg.haar_circuit, resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    params["depth"] = T
    circuit = HaarBrickworkCircuit(params, gate_rng)
    fixed = bool(params.get("fixed_instance", True))
    psi0 = NL.initial_state(cfg, str(task.get("encoding", "bell")))

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, T, N))
    bloch = np.zeros((L, 3))
    n_quantum = 3 * N + 3 * N * (N - 1) // 2
    Q = np.zeros((L, n_quantum))
    start = time.perf_counter()
    for i in tqdm(range(L)):
        if not fixed and i > 0:
            circuit.redraw(gate_rng)
        M[i], bloch[i], psi = trajectory(psi0, circuit, float(p[i]), rng)
        Q[i] = NL.quantum_features(psi, N)
    elapsed = time.perf_counter() - start
    purity = purity_from_bloch(bloch)

    vec_M = extract_M_features(M, "raw_abs")
    features = {"M": vec_M, "state": Q, "M_state": np.hstack([vec_M, Q])}
    metrics: dict[str, float] = {
        "N": float(N),
        "T": float(T),
        "fixed_instance": float(fixed),
        "samples_per_p": float(per_p),
        "n_features_M": float(vec_M.shape[1]),
        "n_features_state": float(n_quantum),
    }
    split_rng = np.random.default_rng([seed, RNG_TAG, 2])
    for level in levels:
        idx = np.flatnonzero(p == level)
        order = split_rng.permutation(idx.size)
        n_train = int(round(train_fraction * idx.size))
        tr, te = idx[order[:n_train]], idx[order[n_train:]]
        key = f"p{level:g}"
        metrics[f"purity_mean_{key}"] = float(purity[idx].mean())
        metrics[f"purity_std_{key}"] = float(purity[idx].std())
        metrics[f"S_R_mean_{key}"] = float(np.mean(_entropy_bits(bloch[idx])))
        for name, value in score_level(features, bloch, purity, tr, te).items():
            metrics[f"{name}_{key}"] = value
    return {
        "M": M,
        "bloch": bloch,
        "purity": purity,
        "p": p,
        "metrics": metrics,
        "timing": {"time_simulation_s": elapsed},
    }


def _entropy_bits(bloch: np.ndarray) -> np.ndarray:
    """Von Neumann entropy (bits) of the one-qubit states of Bloch vectors."""
    norm = np.clip(np.linalg.norm(bloch, axis=-1), 0.0, 1.0)
    out = np.zeros(norm.shape)
    for lam in ((1.0 + norm) / 2.0, (1.0 - norm) / 2.0):
        safe = np.where(lam > 1e-15, lam, 1.0)
        out -= np.where(lam > 1e-15, lam * np.log2(safe), 0.0)
    return out


@hydra.main(version_base="1.3", config_path="../../conf", config_name="bloch_purity")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.bloch_task.get("save_arrays", True)):
        np.savez(
            out_dir / f"{Path(__file__).stem}.npz",
            **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
        )
    metrics = {**results["metrics"], **results["timing"]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_config(cfg, out_dir)
    print(f"N: {cfg.N}  T: {int(metrics['T'])}  seed: {cfg.seed}")
    print(f"{'p':>6} {'purity':>7}   R2 Bloch (M, state, M+state)   R2 purity")
    for level in cfg.bloch_task.p_values:
        key = f"p{float(level):g}"
        b = " ".join(f"{metrics[f'r2_bloch_{n}_{key}']:7.3f}" for n in READOUTS)
        q = " ".join(f"{metrics[f'r2_purity_{n}_{key}']:7.3f}" for n in READOUTS)
        print(f"{float(level):6.3f} {metrics[f'purity_mean_{key}']:7.3f}   {b}   {q}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
