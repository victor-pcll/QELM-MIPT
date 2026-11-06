#!/usr/bin/env python3
"""Why the QELM state readout decays with ``p N T``: signal, noise, prediction.

Issue #305, after #268 (PR #274) and #285 (PR #302). On a fixed monitored
Haar circuit, the ridge on the final-state features ``<P_k>``, ``<P_i P_j>``
reads the sector of the reference qubit ``R`` only at small ``p``. A linear
readout sees two things of the feature distribution: the class means and the
within-class covariance. This script computes both, the first exactly:

1. **Signal, exact.** The mean features of class ``Y = +-1`` are
   ``Tr(O rho_+-)``, with ``rho_+-`` the final state of the system averaged
   over the records, given the sector. ``R`` never evolves, so ``rho_+-`` is
   the averaged channel applied to the sector state: each round, the gates
   of the layer, then on each site the Z-dephasing of probability ``p``,
   ``rho -> (1 - p) rho + p (P0 rho P0 + P1 rho P1)``
   (:func:`averaged_states`). The density matrix of the ``N`` system
   qubits is evolved, without sampling. Outputs: the signal vector
   ``s = Tr(O Delta)`` of the features, ``Delta = rho_+ - rho_-``; its
   Hilbert-Schmidt norm ``||Delta||_HS^2`` relative to ``p = 0``; the trace
   distance ``D = ||Delta||_1 / 2``.
2. **Effective volume.** Dephasing site ``k`` multiplies the Pauli
   components of ``Delta`` with ``X`` or ``Y`` on ``k`` by ``1 - p``, so to
   first order ``||Delta||_HS^2 = exp(-p V_eff)`` with
   ``V_eff = sum_t sum_k 2 w_k(t)``, ``w_k(t)`` the weight of those
   components at round ``t`` (:func:`xy_weights`), computed at ``p = 0``.
   For a scrambled ``Delta``, ``w_k = 1/2`` and ``V_eff = N T``.
   The same first-order rate of the signal of the features,
   ``||s||^2 = exp(-p V_feat)``, is ``V_feat`` (``V_Zfeat`` and ``V_XYfeat``
   for the features diagonal in Z and the others), from ``s`` at
   ``p = EPS``. In the Heisenberg picture it counts the measurements inside
   the backward light cone of the observables: ``V_feat < N T``.
3. **Noise, sampled.** ``samples_per_p`` trajectories at each ``p``
   (``experiments/nonlinear_M``: same circuit, features after ``R`` is
   measured), split 70 / 30 (stratified on ``Y``). On the training part:
   the within-class covariance ``Sigma`` of the features (Ledoit-Wolf, the
   features centered by the exact class means) and the ridge readout
   (``RidgeCV``, as in #274). On the test part:

   - ``accuracy_ridge``: the ridge;
   - ``accuracy_oracle``: the linear readout ``w = Sigma^-1 s`` with the
     exact means, the best linear readout up to the estimate of ``Sigma``;
   - ``accuracy_gauss``: the Gaussian prediction
     ``Phi(sqrt(s^T Sigma^-1 s) / 2)``, with no fit to the labels;
   - ``accuracy_ridge_K<k>``: the ridge on the test trajectories with ``k``
     measurements (``K`` = number of nonzero entries of the record).

Run from the repository root:

    uv run python experiments/pnt_decay/pnt_decay.py N=6 pnt_task.T=6 seed=42

Writes ``metrics.json`` (``n_rounds``, ``V_eff`` and, for each ``p`` of the
grid, the keys listed in :func:`run`, suffixed ``_p<p>``) and the Hydra
config to the Hydra run directory.
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
from scipy.special import ndtr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import RidgeCV

from QELM import check_qubit_counts, set_seed
from QELM.evaluation.entropy_regression import stratified_split
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.physics.operators import PAULIS
from QELM.utils import save_config

# Seed tag of the generators of this experiment.
RNG_TAG = 305
# Regularizations tried by the ridge readout (as experiments/haar_readouts).
ALPHAS = np.logspace(-2, 4, 13)
# Largest number of measurements K with its own accuracy key.
K_MAX = 12
# Fewest test trajectories for an accuracy per K.
K_MIN_COUNT = 20
# Step in p of the first-order decay rates V_feat.
EPS = 1e-4


def _load_nonlinear_m() -> ModuleType:
    """Import ``experiments/nonlinear_M/nonlinear_M.py`` (not a package)."""
    path = Path(__file__).resolve().parents[1] / "nonlinear_M" / "nonlinear_M.py"
    spec = importlib.util.spec_from_file_location("nonlinear_M", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NONLINEAR_M = _load_nonlinear_m()


def p_grid(N: int, T: int, pnt_values: list[float]) -> np.ndarray:
    """Probabilities ``p = n / (N T)`` for the ``n`` of ``pnt_values``, at
    most 1, distinct and sorted."""
    p = np.asarray(pnt_values, dtype=float) / (N * T)
    return np.asarray(np.unique(np.round(p[p <= 1.0], 8)))


def sector_states(psi0: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """States of the system paired with ``|0>_R`` and ``|1>_R`` in ``psi0``.

    ``R`` is the most significant qubit: ``psi0 = (|0> a + |1> b) / sqrt 2``
    with the rows of ``psi0.reshape(2, 2**N)``, each normalized here.
    """
    rows = np.asarray(psi0, dtype=complex).reshape(2, 2**N)
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Both branches of psi0 must be nonzero.")
    return rows[0] / norms[0], rows[1] / norms[1]


def dephasing_masks(N: int) -> list[np.ndarray]:
    """Boolean masks of the entries of a ``2**N`` density matrix that are
    off-diagonal in qubit ``k`` (qubit 0 the most significant bit)."""
    idx = np.arange(2**N)
    masks = []
    for k in range(N):
        bit = (idx >> (N - 1 - k)) & 1
        masks.append(bit[:, None] != bit[None, :])
    return masks


def xy_weights(delta: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Weight ``w_k`` of the Pauli components of ``delta`` with ``X`` or ``Y``
    on qubit ``k``, relative to ``||delta||_HS^2``.

    In the Pauli basis, the components with ``X`` or ``Y`` on ``k`` are
    exactly the entries off-diagonal in ``k``, so ``w_k`` is their share of
    the squared Frobenius norm. ``w_k = 1/2`` for a uniformly scrambled
    operator; 0 for a diagonal one.
    """
    total = float(np.sum(np.abs(delta) ** 2))
    if total <= 0.0:
        return np.zeros(len(masks))
    return np.array([float(np.sum(np.abs(delta[m]) ** 2)) / total for m in masks])


def system_layers(circuit: BrickworkCircuit) -> list[list[tuple[int, int, np.ndarray]]]:
    """Gates of the circuit on the system qubits alone (indices shifted by
    ``Nr``)."""
    shift = circuit.Nr
    return [
        [(i - shift, j - shift, g) for i, j, g in layer] for layer in circuit.layers
    ]


def hamming_matrix(N: int) -> np.ndarray:
    """Number of qubits where the row and column bitstrings differ, for
    every entry of a ``2**N`` density matrix."""
    idx = np.arange(2**N)
    x = idx[:, None] ^ idx[None, :]
    count = np.zeros_like(x)
    for k in range(N):
        count += (x >> k) & 1
    return count


def apply_gate_rho(rho: np.ndarray, gate: np.ndarray, i: int, j: int) -> np.ndarray:
    """``G rho G^dagger`` for a two-qubit gate on qubits ``i`` and ``j`` of an
    ``N``-qubit density matrix (as :func:`apply_two_qubit_gate`, faster).

    ``rho`` is reshaped to ``2 N`` qubit axes (rows, then columns); ``G``
    acts on the row axes ``i, j`` and ``G^*`` on the column axes.
    """
    dim = rho.shape[0]
    N = dim.bit_length() - 1
    g = gate.reshape(2, 2, 2, 2)
    t = rho.reshape((2,) * (2 * N))
    t = np.tensordot(g, t, axes=([2, 3], [i, j]))
    t = np.moveaxis(t, (0, 1), (i, j))
    t = np.tensordot(g.conj(), t, axes=([2, 3], [N + i, N + j]))
    t = np.moveaxis(t, (0, 1), (N + i, N + j))
    return np.ascontiguousarray(t).reshape(dim, dim)


def averaged_states(
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
    layers: list[list[tuple[int, int, np.ndarray]]],
    p: float,
    hamming: np.ndarray,
    masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Record-averaged final states of both sectors, and ``V``.

    Each round: the gates of the layer, then on each site the Z-dephasing of
    probability ``p``, which multiplies the entries off-diagonal in that site
    by ``1 - p``: all sites at once, ``(1 - p)**hamming``. With ``masks``,
    the weights :func:`xy_weights` of ``Delta`` are taken after the gates of
    each round, before its dephasing, where the measurements act.

    Returns:
        ``(rho_+, rho_-, V)`` with ``V = sum_t sum_k 2 w_k(t)`` (0 without
        ``masks``): ``V_eff`` when ``p = 0``.
    """
    rp, rm = rho_plus.astype(complex), rho_minus.astype(complex)
    factor = (1.0 - p) ** hamming if p > 0.0 else None
    volume = 0.0
    for layer in layers:
        for i, j, gate in layer:
            rp = apply_gate_rho(rp, gate, i, j)
            rm = apply_gate_rho(rm, gate, i, j)
        if masks is not None:
            volume += 2.0 * float(np.sum(xy_weights(rp - rm, masks)))
        if factor is not None:
            rp = rp * factor
            rm = rm * factor
    return rp, rm, volume


def feature_expectations(rho: np.ndarray, N: int) -> np.ndarray:
    """QELM features ``Tr(P rho)`` of an ``N``-qubit density matrix.

    The order of :func:`nonlinear_M.quantum_features`: ``<X_k>``, ``<Y_k>``,
    ``<Z_k>`` for every site, then ``<X_i X_j>``, ``<Y_i Y_j>``,
    ``<Z_i Z_j>`` for every pair ``i < j``. Computed from the one- and
    two-site reduced states.
    """
    tensor = rho.reshape((2,) * (2 * N))
    names = ("X", "Y", "Z")
    singles = {name: np.zeros(N) for name in names}
    pairs = {name: [] for name in names}
    reduced1 = []
    for k in range(N):
        moved = np.moveaxis(tensor, (k, N + k), (0, 1)).reshape(2, 2, -1)
        side = 2 ** (N - 1)
        moved = moved.reshape(2, 2, side, side)
        reduced1.append(np.einsum("abrr->ab", moved))
    for name in names:
        P = PAULIS[name]
        for k in range(N):
            singles[name][k] = float(np.real(np.trace(P @ reduced1[k])))
    for i in range(N):
        for j in range(i + 1, N):
            moved = np.moveaxis(tensor, (i, j, N + i, N + j), (0, 1, 2, 3))
            side = 2 ** (N - 2)
            moved = moved.reshape(4, 4, side, side)
            rho_ij = np.einsum("abrr->ab", moved)
            for name in names:
                P = PAULIS[name]
                pairs[name].append(float(np.real(np.trace(np.kron(P, P) @ rho_ij))))
    return np.concatenate(
        [singles["X"], singles["Y"], singles["Z"]]
        + [np.asarray(pairs[name]) for name in names]
    )


def feature_groups(N: int) -> dict[str, np.ndarray]:
    """Index sets of the features of :func:`feature_expectations`: ``feat``
    (all), ``Zfeat`` (``<Z_k>``, ``<Z_i Z_j>``, diagonal in Z) and
    ``XYfeat`` (the others)."""
    pairs = N * (N - 1) // 2
    idx = np.arange(3 * N + 3 * pairs)
    z = np.concatenate([np.arange(2 * N, 3 * N), 3 * N + 2 * pairs + np.arange(pairs)])
    return {"feat": idx, "Zfeat": z, "XYfeat": np.setdiff1d(idx, z)}


def oracle_readout(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    mu_plus: np.ndarray,
    mu_minus: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """The linear readout with the exact means and the sampled covariance.

    ``Sigma`` is the Ledoit-Wolf estimate of the within-class covariance of
    the training features, centered by the exact mean of their class;
    ``w = Sigma^-1 s`` with ``s = mu_+ - mu_-`` (Fisher's discriminant).

    Returns:
        ``(score, snr2, noise)``: ``w . (x - (mu_+ + mu_-) / 2)`` on the test
        features, ``s^T Sigma^-1 s`` and the mean within-class variance
        ``tr(Sigma) / d``.
    """
    centered = X_tr - np.where(Y_tr[:, None] > 0, mu_plus, mu_minus)
    sigma = LedoitWolf(assume_centered=True).fit(centered).covariance_
    s = mu_plus - mu_minus
    w = np.linalg.solve(sigma + 1e-12 * np.eye(s.size), s)
    score = (X_te - (mu_plus + mu_minus) / 2.0) @ w
    return score, float(s @ w), float(np.trace(sigma) / s.size)


def _sign(y: np.ndarray) -> np.ndarray:
    return np.where(y >= 0.0, 1.0, -1.0)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Exact signal and sampled readouts at every ``p`` of the grid.

    Args:
        cfg: Config of ``conf/pnt_decay.yaml`` (Hydra or ``load_config``).
            Reads ``N``, ``Nr`` (1), ``seed``, ``haar_circuit`` and
            ``pnt_task``.

    Returns:
        ``metrics`` (deterministic for a seed) and ``timing``. The metrics
        hold ``n_rounds``, ``V_eff``, ``V_feat``, ``V_Zfeat``, ``V_XYfeat``,
        ``w_xy_<t>`` (the mean ``w_k`` at round ``t``, ``p = 0``) and, for
        each ``p``:
        ``hs2`` (``||Delta||_HS^2`` relative to ``p = 0``), ``trace_distance``,
        ``signal2`` (``||s||^2``), ``noise`` (``tr Sigma / d``), ``snr2``,
        ``accuracy_ridge``, ``accuracy_oracle``, ``accuracy_gauss``,
        ``K_mean`` and ``accuracy_ridge_K<k>`` with ``count_K<k>``. With
        ``pnt_task.sample: false``, only the exact keys (no trajectories).

    Raises:
        ValueError: If ``Nr`` is not 1 or a setting is out of range.
    """
    set_seed(cfg)
    N, Nr, _ = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"pnt_decay needs the reference qubit: Nr=1, got {Nr}.")
    task = cfg.pnt_task
    rule = task.get("depth_rule")
    T = int(task.T) if rule is None else NONLINEAR_M.depth_from_rule(rule, N)
    if T < 1:
        raise ValueError(f"pnt_task.T must be >= 1, got {T}.")
    per_p = int(task.samples_per_p)
    if per_p < 40:
        raise ValueError(f"pnt_task.samples_per_p must be >= 40, got {per_p}.")
    levels = p_grid(N, T, [float(v) for v in task.pnt_values])
    if levels.size == 0:
        raise ValueError("pnt_task.pnt_values gives no p in [0, 1].")
    seed = int(cfg.seed)
    gate_rng = np.random.default_rng([seed, RNG_TAG, 0])
    rng = np.random.default_rng([seed, RNG_TAG, 1])
    split_rng = np.random.default_rng([seed, RNG_TAG, 2])
    block = OmegaConf.to_container(cfg.haar_circuit, resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    params["depth"] = T
    circuit = HaarBrickworkCircuit(params, gate_rng)
    psi0 = NONLINEAR_M.initial_state(cfg, str(task.encoding))
    a, b = sector_states(psi0, N)
    rho_plus0, rho_minus0 = np.outer(a, a.conj()), np.outer(b, b.conj())
    layers = system_layers(circuit)
    masks = dephasing_masks(N)
    hamming = hamming_matrix(N)
    want_trace = bool(task.get("trace_distance", True))
    sample = bool(task.get("sample", True))

    metrics: dict[str, float] = {"n_rounds": float(T)}
    timing = {"time_exact_s": 0.0, "time_sampling_s": 0.0}
    # p = 0: the effective volume and the weights round by round.
    start = time.perf_counter()
    rp, rm = rho_plus0.astype(complex), rho_minus0.astype(complex)
    v_eff = 0.0
    for t, layer in enumerate(layers):
        for i, j, gate in layer:
            rp = apply_gate_rho(rp, gate, i, j)
            rm = apply_gate_rho(rm, gate, i, j)
        w = xy_weights(rp - rm, masks)
        metrics[f"w_xy_{t}"] = float(w.mean())
        v_eff += 2.0 * float(w.sum())
    metrics["V_eff"] = v_eff
    hs0 = float(np.sum(np.abs(rp - rm) ** 2))
    # First-order decay of the signal of the features: s(eps) against s(0).
    s0 = feature_expectations(rp, N) - feature_expectations(rm, N)
    rp_e, rm_e, _ = averaged_states(rho_plus0, rho_minus0, layers, EPS, hamming)
    s_e = feature_expectations(rp_e, N) - feature_expectations(rm_e, N)
    for name, sel in feature_groups(N).items():
        a, b = float(s0[sel] @ s0[sel]), float(s_e[sel] @ s_e[sel])
        if a > 0.0 and b > 0.0:
            metrics[f"V_{name}"] = -(np.log(b) - np.log(a)) / EPS
    timing["time_exact_s"] += time.perf_counter() - start

    for p in levels:
        key = f"p{p:g}"
        start = time.perf_counter()
        rp, rm, _ = averaged_states(rho_plus0, rho_minus0, layers, float(p), hamming)
        delta = rp - rm
        mu_plus, mu_minus = feature_expectations(rp, N), feature_expectations(rm, N)
        s = mu_plus - mu_minus
        metrics[f"hs2_{key}"] = float(np.sum(np.abs(delta) ** 2)) / hs0
        if want_trace:
            eig = np.linalg.eigvalsh((delta + delta.conj().T) / 2.0)
            metrics[f"trace_distance_{key}"] = float(np.sum(np.abs(eig)) / 2.0)
        metrics[f"signal2_{key}"] = float(s @ s)
        timing["time_exact_s"] += time.perf_counter() - start
        if not sample:
            continue

        start = time.perf_counter()
        X = np.zeros((per_p, s.size))
        Y = np.zeros(per_p)
        K = np.zeros(per_p, dtype=int)
        for n in range(per_p):
            M, _, _, Y[n], psi = NONLINEAR_M.trajectory(psi0, circuit, float(p), rng)
            X[n] = NONLINEAR_M.quantum_features(psi, N)
            K[n] = int(np.count_nonzero(M))
        tr, te = stratified_split(Y, float(cfg.samples.train_fraction), split_rng)
        metrics[f"K_mean_{key}"] = float(K.mean())
        if np.ptp(X[tr], axis=0).max() < 1e-12 or np.unique(Y[tr]).size < 2:
            ridge = np.full(te.size, float(np.mean(Y[tr])))
        else:
            ridge = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]).predict(X[te])
        hit = _sign(ridge) == Y[te]
        metrics[f"accuracy_ridge_{key}"] = float(np.mean(hit))
        if p > 0.0:
            score, snr2, noise = oracle_readout(X[tr], Y[tr], X[te], mu_plus, mu_minus)
            metrics[f"accuracy_oracle_{key}"] = float(np.mean(_sign(score) == Y[te]))
            metrics[f"snr2_{key}"] = snr2
            metrics[f"noise_{key}"] = noise
            metrics[f"accuracy_gauss_{key}"] = float(ndtr(np.sqrt(max(snr2, 0.0)) / 2))
        else:
            # A fixed circuit maps each sector to one state: no noise.
            metrics[f"accuracy_oracle_{key}"] = 1.0 if s @ s > 0 else 0.5
            metrics[f"accuracy_gauss_{key}"] = 1.0 if s @ s > 0 else 0.5
            metrics[f"noise_{key}"] = 0.0
        for k in range(K_MAX + 1):
            sel = K[te] == k
            if sel.sum() >= K_MIN_COUNT:
                metrics[f"accuracy_ridge_K{k}_{key}"] = float(np.mean(hit[sel]))
                metrics[f"count_K{k}_{key}"] = float(sel.sum())
        timing["time_sampling_s"] += time.perf_counter() - start
    return {"metrics": metrics, "timing": timing, "p": levels}


@hydra.main(version_base="1.3", config_path="../../conf", config_name="pnt_decay")
def main(cfg: DictConfig) -> None:
    """Run the experiment and write ``metrics.json`` to the Hydra run folder."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {**results["metrics"], **results["timing"]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_config(cfg, out_dir)
    N, T = int(cfg.N), int(metrics["n_rounds"])
    print(f"N = {N}, T = {T}, V_eff / (N T) = {metrics['V_eff'] / (N * T):.3f}")
    print(f"V_feat / (N T) = {metrics.get('V_feat', float('nan')) / (N * T):.3f}")
    print(f"{'pNT':>6} {'hs2':>8} {'ridge':>6} {'oracle':>6} {'gauss':>6}")
    nan = float("nan")
    for p in results["p"]:
        key = f"p{p:g}"
        print(
            f"{p * N * T:6.2f} {metrics[f'hs2_{key}']:8.4f}"
            f" {metrics.get(f'accuracy_ridge_{key}', nan):6.3f}"
            f" {metrics.get(f'accuracy_oracle_{key}', nan):6.3f}"
            f" {metrics.get(f'accuracy_gauss_{key}', nan):6.3f}"
        )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
