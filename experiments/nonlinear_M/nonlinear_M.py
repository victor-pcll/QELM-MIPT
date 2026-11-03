#!/usr/bin/env python3
"""Non-linear features of the measurement record M for the QELM-M (issue #269).

Why does a linear readout of the record ``M`` fail, and which light
non-linearity of ``M`` restores what it misses? The setting of the open
pull requests: Haar U(4) gates and the local encoding.

For each measurement probability ``p`` of ``nonlinear_task.p_values`` and
each of its ``nonlinear_task.samples_per_p`` trajectories:

1. Start from :class:`AncillaBellPairEncoding`: the reference qubit ``R``
   (qubit 0) in a Bell pair with the middle system site ``N // 2``, the
   other system qubits in ``|0>``.
2. Run ``T`` rounds (``nonlinear_task.depth_rule``: ``T = N`` by default),
   each one brickwork period of Haar U(4) gates on the system qubits (one
   fixed circuit for the run, set by the seed), then a Z measurement of each
   system qubit with probability ``p``. Row ``t`` of ``M`` holds the
   outcomes of round ``t``, +-1, and 0 for an unmeasured qubit.
3. At the end, the Bloch component ``r_z = <Z_R>`` of ``R`` given the
   record and the circuit, its entropy ``S_R`` (bits), and the label ``Y``,
   the Z outcome of ``R`` drawn with the Born rule. Since ``R`` never
   evolves, measuring it at the end is the same as measuring it first: ``Y``
   is the initial Z state of the middle site, +-1 with probability 1/2.

The readouts, each fitted on 70 % of the trajectories of one ``p`` and
scored on the rest (``accuracy`` of the sign, ``R2`` of the output):

- ``ridge_<mode>``: a ridge (``RidgeCV``, alpha by leave-one-out over
  :data:`ALPHAS`) on the features ``Phi(M)`` of each mode of
  ``nonlinear_task.modes`` (:data:`QELM.features.MODES`): ``raw``, ``abs``,
  ``raw_abs``, ``xor_local``, ``xor_cone``, ``xor_full``;
- ``ridge_quantum`` (``quantum_baseline``): the ridge on the QELM features of
  the final state, ``<P_k>`` and ``<P_i P_j>`` for ``P`` in ``X, Y, Z``,
  after ``R`` is measured, as in ``experiments/haar_readouts``;
- ``mlp_raw`` (``mlp_reference``): a neural network on ``vec(M)``;
- ``bayes``: the sign of ``r_z``, the optimal decoder of an observer who
  knows the circuit (no training). Its accuracy has expectation
  ``E[(1 + |r_z|) / 2]``, reported as ``bayes_expected``.

And the spectrum of the information, order by order:
``corr_rms_<group>_p<p>`` and ``corr_max_<group>_p<p>``, the root mean square
and the largest ``|Corr(phi, Y)|`` over the features of each group
(``m``: the entries; ``abs``: their absolute values; ``pair_space``,
``pair_time``: the nearest-neighbour products; ``pair_all``: every product),
and ``corr_floor_p<p>``, ``1 / sqrt(n)``, the level of a correlation with
shuffled labels. ``corr_map`` (``(n_p, T, N)``, in ``corr_map.json``) holds
``Corr(m_{t,i}, Y)`` entry by entry: where in space-time the record holds
the label.

``|M|`` is independent of ``Y`` by construction: whether a qubit is
measured is drawn with probability ``p`` whatever the state, so
``ridge_abs`` is a null control. With new gates per trajectory
(``haar_circuit.fixed_instance=false``) the Haar average of the first gate
on the paired site forgets its initial state, so every readout of ``M``,
linear or not, is at chance (the observer who does not know the gates of
#275): the second null control.

Run from the repository root:

    uv run python experiments/nonlinear_M/nonlinear_M.py N=8 seed=42

Everything is written to the Hydra run directory: ``metrics.json``,
``corr_map.json``, ``nonlinear_M.npz`` (``M``, ``Y``, ``r_z``, ``S_R``,
``p``, ``corr_map``) and the config.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from QELM import check_qubit_counts, set_seed
from QELM.encoding import AncillaBellPairEncoding
from QELM.evaluation.entropy import von_neumann_entropy
from QELM.evaluation.entropy_regression import stratified_split
from QELM.features import (
    MODES,
    all_pairs,
    extract_M_features,
    feature_label_correlation,
    spatial_pairs,
    temporal_pairs,
)
from QELM.models.circuits.haar import HaarBrickworkCircuit
from QELM.physics.operators import PAULIS
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config

# Seed tag of the generators of the gates and the measurements.
RNG_TAG = 269
# Regularizations tried by the ridge.
ALPHAS = np.logspace(-2, 4, 13)
# Groups of the correlation spectrum.
GROUPS = ("m", "abs", "pair_space", "pair_time", "pair_all")


def depth_from_rule(rule: Any, N: int) -> int:
    """Rounds ``T`` of a depth rule: an integer, ``"N"`` or ``"<k>N"``.

    Raises:
        ValueError: If the rule is not of that form or gives ``T < 1``.
    """
    text = str(rule).strip()
    if text.endswith("N"):
        factor = text[:-1] or "1"
        if not factor.isdigit():
            raise ValueError(f"depth_rule must be an integer, N or kN, got {rule!r}.")
        T = int(factor) * N
    elif text.isdigit():
        T = int(text)
    else:
        raise ValueError(f"depth_rule must be an integer, N or kN, got {rule!r}.")
    if T < 1:
        raise ValueError(f"depth_rule gives T={T}; it must be >= 1.")
    return T


def _apply_pauli(psi: np.ndarray, pauli: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a one-qubit operator to ``qubit`` of an ``n``-qubit state vector."""
    tensor = psi.reshape((2,) * n)
    out = np.tensordot(pauli, tensor, axes=([1], [qubit]))
    return np.moveaxis(out, 0, qubit).reshape(-1)


def quantum_features(psi: np.ndarray, N: int) -> np.ndarray:
    """QELM features of the system: ``<P_k>`` then ``<P_i P_j>`` (``i < j``).

    ``P`` runs over ``X, Y, Z`` (sets ``X, Y, Z, XX, YY, ZZ`` of
    ``experiments/haar_readouts``); qubit 0 is ``R``, system site ``k`` is
    qubit ``k + 1``.

    Returns:
        Array of length ``3 N + 3 N (N - 1) / 2``.
    """
    n = N + 1
    singles: list[float] = []
    pairs: list[float] = []
    for name in ("X", "Y", "Z"):
        P = PAULIS[name]
        applied = [_apply_pauli(psi, P, k + 1, n) for k in range(N)]
        singles.extend(float(np.real(np.vdot(psi, a))) for a in applied)
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append(float(np.real(np.vdot(applied[i], applied[j]))))
    return np.asarray(singles + pairs)


def reference_z(psi: np.ndarray) -> float:
    """``<Z_R>`` of the reference qubit (qubit 0, the most significant bit)."""
    half = psi.size // 2
    return float(np.sum(np.abs(psi[:half]) ** 2) - np.sum(np.abs(psi[half:]) ** 2))


def trajectory(
    psi0: np.ndarray,
    circuit: HaarBrickworkCircuit,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    """Run one trajectory from ``psi0``.

    Returns:
        ``(M, r_z, S_R, Y, psi)``: the record ``(depth, N)``, ``<Z_R>`` and
        the entropy of ``R`` (bits) given the record, the Born outcome of
        ``R``, +-1, and the state after that measurement.
    """
    N, N_total = circuit.N, circuit.N_total
    psi = psi0
    M = np.zeros((circuit.depth, N))
    for t in range(circuit.depth):
        psi = circuit.apply_layer(psi, t)
        for k in range(N):
            if rng.random() < p:
                psi, M[t, k] = measure_projective_psi(psi, k + 1, N_total, rng)
    r_z = reference_z(psi)
    S_R = von_neumann_entropy(psi, [0])
    psi, Y = measure_projective_psi(psi, 0, N_total, rng)
    return M, r_z, S_R, float(Y), psi


def _ridge(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    model = RidgeCV(alphas=ALPHAS).fit(X_tr, Y_tr)
    return np.asarray(model.predict(X_te))


def _mlp(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        early_stopping=True,
        max_iter=500,
        random_state=int(rng.integers(2**31)),
    ).fit(X_tr, Y_tr)
    proba = model.predict_proba(X_te)
    plus = list(model.classes_).index(1.0)
    return np.asarray(2.0 * proba[:, plus] - 1.0)


def _sign(y: np.ndarray) -> np.ndarray:
    return np.where(y >= 0.0, 1.0, -1.0)


def correlation_spectrum(M: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    """``Corr(phi, Y)`` for every feature of each group of :data:`GROUPS`."""
    L = M.shape[0]
    raw = M.reshape(L, -1)
    blocks = {
        "m": raw,
        "abs": np.abs(raw),
        "pair_space": spatial_pairs(M).reshape(L, -1),
        "pair_time": temporal_pairs(M).reshape(L, -1),
        "pair_all": all_pairs(M),
    }
    return {name: feature_label_correlation(X, Y) for name, X in blocks.items()}


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories of every ``p``, fit and score the readouts.

    Args:
        cfg: Composed config (``load_config(..., config_name="nonlinear_M")``
            or Hydra).

    Returns:
        ``M``, ``Y``, ``r_z``, ``S_R``, ``p`` and ``metrics``.

    Raises:
        ValueError: If ``Nr`` is not 1, a probability is outside ``[0, 1]``,
            a count is too small or a mode is unknown.
    """
    set_seed(cfg)
    N, Nr, N_total = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"nonlinear_M needs the reference qubit: Nr=1, got {Nr}.")
    task = cfg.nonlinear_task
    levels = np.asarray([float(v) for v in task.p_values])
    if levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"nonlinear_task.p_values must be in [0, 1], got {levels}.")
    if np.unique(levels).size != levels.size:
        raise ValueError(f"nonlinear_task.p_values must be distinct, got {levels}.")
    per_p = int(task.samples_per_p)
    if per_p < 10:
        raise ValueError(f"nonlinear_task.samples_per_p must be >= 10, got {per_p}.")
    modes = [str(m) for m in task.modes]
    unknown = [m for m in modes if m not in MODES]
    if unknown or not modes:
        raise ValueError(
            f"nonlinear_task.modes must hold modes of {MODES}, got {modes}."
        )
    T = depth_from_rule(task.depth_rule, N)
    want_quantum = bool(task.get("quantum_baseline", True))
    want_mlp = bool(task.get("mlp_reference", True))

    seed = int(cfg.seed)
    gate_rng = np.random.default_rng([seed, RNG_TAG, 0])
    rng = np.random.default_rng([seed, RNG_TAG, 1])
    block = OmegaConf.to_container(cfg.haar_circuit, resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    params["depth"] = T
    circuit = HaarBrickworkCircuit(params, gate_rng)
    fixed = bool(params.get("fixed_instance", True))
    encoding = AncillaBellPairEncoding.from_config(cfg)
    site = encoding.site
    psi0 = encoding.state()

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, T, N))
    r_z = np.zeros(L)
    S_R = np.zeros(L)
    Y = np.zeros(L)
    n_quantum = 3 * N + 3 * N * (N - 1) // 2
    Q = np.zeros((L, n_quantum)) if want_quantum else np.zeros((L, 0))
    start = time.perf_counter()
    for i in tqdm(range(L)):
        if not fixed and i > 0:
            circuit.redraw(gate_rng)
        M[i], r_z[i], S_R[i], Y[i], psi = trajectory(psi0, circuit, float(p[i]), rng)
        if want_quantum:
            Q[i] = quantum_features(psi, N)
    elapsed = time.perf_counter() - start

    split_rng = np.random.default_rng([seed, RNG_TAG, 2])
    fit_rng = np.random.default_rng([seed, RNG_TAG, 3])
    features = {f"ridge_{mode}": extract_M_features(M, mode, site) for mode in modes}
    if want_quantum:
        features["ridge_quantum"] = Q
    metrics: dict[str, float] = {
        "N": float(N),
        "T": float(T),
        "site": float(site),
        "fixed_instance": float(fixed),
        "n_features_quantum": float(n_quantum),
        **{
            f"n_features_{mode}": float(features[f"ridge_{mode}"].shape[1])
            for mode in modes
        },
    }
    corr_map = np.zeros((levels.size, T, N))
    for level in levels:
        idx = np.flatnonzero(p == level)
        train, test = stratified_split(
            Y[idx], float(cfg.samples.train_fraction), split_rng
        )
        tr, te = idx[train], idx[test]
        key = f"p{level:g}"
        metrics[f"S_R_mean_{key}"] = float(S_R[idx].mean())
        metrics[f"abs_rz_mean_{key}"] = float(np.abs(r_z[idx]).mean())
        metrics[f"accuracy_bayes_{key}"] = float(np.mean(_sign(r_z[te]) == Y[te]))
        metrics[f"accuracy_bayes_expected_{key}"] = float(
            np.mean((1.0 + np.abs(r_z[idx])) / 2.0)
        )
        metrics[f"r2_bayes_{key}"] = float(r2_score(Y[te], r_z[te]))
        majority = 1.0 if np.mean(Y[tr]) >= 0.0 else -1.0
        metrics[f"accuracy_baseline_{key}"] = float(np.mean(Y[te] == majority))
        readouts: dict[str, tuple[np.ndarray, str]] = {
            name: (X, "ridge") for name, X in features.items()
        }
        if want_mlp:
            readouts["mlp_raw"] = (features.get("ridge_raw", M.reshape(L, -1)), "mlp")
            if "ridge_raw" not in features:
                readouts["mlp_raw"] = (M.reshape(L, -1), "mlp")
        for name, (X, kind) in readouts.items():
            if np.unique(Y[tr]).size < 2 or np.ptp(X[tr], axis=0).max() < 1e-12:
                # Nothing to learn: one label, or constant features (the
                # empty record at p = 0). The output is the mean label.
                score = np.full(te.size, float(np.mean(Y[tr])))
            elif kind == "ridge":
                score = _ridge(X[tr], Y[tr], X[te])
            else:
                score = _mlp(X[tr], Y[tr], X[te], fit_rng)
            metrics[f"accuracy_{name}_{key}"] = float(np.mean(_sign(score) == Y[te]))
            metrics[f"r2_{name}_{key}"] = float(r2_score(Y[te], score))
        spectrum = correlation_spectrum(M[idx], Y[idx])
        corr_map[list(levels).index(level)] = spectrum["m"].reshape(T, N)
        metrics[f"corr_floor_{key}"] = float(1.0 / np.sqrt(idx.size))
        for group in GROUPS:
            c = np.abs(spectrum[group])
            metrics[f"corr_rms_{group}_{key}"] = float(np.sqrt(np.mean(c**2)))
            metrics[f"corr_max_{group}_{key}"] = float(c.max())
    return {
        "M": M,
        "Y": Y,
        "r_z": r_z,
        "S_R": S_R,
        "p": p,
        "corr_map": corr_map,
        "metrics": metrics,
        "timing": {"time_simulation_s": elapsed},
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="nonlinear_M")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.nonlinear_task.get("save_arrays", True)):
        np.savez(
            out_dir / f"{Path(__file__).stem}.npz",
            **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
        )
    metrics = {**results["metrics"], **results["timing"]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    corr_map = {
        "p": [float(v) for v in cfg.nonlinear_task.p_values],
        "corr_m": results["corr_map"].tolist(),
    }
    (out_dir / "corr_map.json").write_text(json.dumps(corr_map))
    save_config(cfg, out_dir)
    names = ["bayes", *[f"ridge_{m}" for m in cfg.nonlinear_task.modes]]
    if cfg.nonlinear_task.get("quantum_baseline", True):
        names.append("ridge_quantum")
    if cfg.nonlinear_task.get("mlp_reference", True):
        names.append("mlp_raw")
    print(f"N: {cfg.N}  T: {int(metrics['T'])}  seed: {cfg.seed}")
    print(f"{'p':>6} {'S_R':>6} " + " ".join(f"{n[-11:]:>11}" for n in names))
    for level in cfg.nonlinear_task.p_values:
        key = f"p{float(level):g}"
        accs = " ".join(f"{metrics[f'accuracy_{n}_{key}']:11.3f}" for n in names)
        print(f"{float(level):6.3f} {metrics[f'S_R_mean_{key}']:6.3f} {accs}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
