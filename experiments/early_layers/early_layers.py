#!/usr/bin/env python3
"""Does the linear QELM on M only read the first rounds? (issue #288).

The task of ``experiments/nonlinear_M`` (#269): a reference qubit ``R`` in
the state ``(|0>_R |a> + |1>_R |b>) / sqrt(2)`` with the chain (encoding
``early_task.encoding``: ``bell``, ``charge``, ``ghz`` or ``w``), one fixed
brickwork circuit per seed (``early_task.gates``: Haar U(4), or U(1) gates
that conserve the charge, the control where late rounds stay informative),
``T`` rounds (``early_task.depth_rule``, ``2N`` by default), each followed by
a Z measurement of every site with probability ``p``. Row ``t`` of the record
``M`` holds the outcomes of round ``t``: +-1, and 0 for an unmeasured site.
The label ``Y`` is the Z outcome of ``R``.

For each ``p`` of ``early_task.p_values`` this script measures, round by
round (``k = 1 .. T``):

**The trained linear model.** A ridge (``RidgeCV``) on ``vec(M)``, fitted on
70 % of the trajectories:

- ``weight_l2``, ``weight_l1``: the norms of its weights ``w_t`` on round
  ``t``;
- ``share``: ``w_t . c_t / sum_s w_s . c_s``, with ``c_t = E[m_t Y]`` the
  exact population covariance of the record with the label
  (:func:`QELM.evaluation.early_layers.suffix_bounds`). The covariance of the
  score with ``Y`` is the sum over rounds of ``w_t . c_t``, so the shares
  add up to 1: what each round contributes to the signal of the model;
- ``share_bound``: ``||w_t||_1 p D_t / |sum_s w_s . c_s|``, the bound of
  the note on ``|share_t|``;
- ``ablate_late``, ``ablate_early``: test accuracy of the same model with
  the rounds after ``k`` (``ablate_late``), or before ``k``
  (``ablate_early``), replaced by their training mean;
- ``prefix``, ``suffix``: test accuracy of ridges retrained on rounds
  ``1 .. k`` and on rounds ``k .. T``.

**The exact bounds** (any decoder, not only linear):

- ``D``: ``||rho_k^+ - rho_k^-||_1 / 2`` of the trajectory-averaged chain
  states after the gates of round ``k``. Rounds ``k .. T`` alone never give
  more than ``(1 + D_k) / 2`` (``suffix_bound``), and ``D_k`` never grows;
- ``bayes_prefix``: ``(1 + E|<Z_R>_k|) / 2``, the Bayes accuracy of rounds
  ``1 .. k`` (index 0: before any round); ``helstrom``:
  ``(1 + E[H_k]) / 2``, a bound on every decoder of the whole record that
  knows rounds ``1 .. k`` (:func:`reference_helstrom`): rounds after ``k``
  add at most ``helstrom_k - bayes_prefix_k``;
- ``c``, ``c_hat``: the exact and the sampled ``E[m_{t,i} Y]``; ``charge``:
  ``Tr(Q (rho_k^+ - rho_k^-))``, conserved by U(1) gates;
- ``haar_mu``, ``haar_c_rms`` (Haar gates only): the Haar average over
  circuits of the total Pauli mass of ``rho_k^+ - rho_k^-`` (``E[D_k] <=
  sqrt(mu_k) / 2``) and of ``sqrt(mean_i c_{k,i}^2)``, from the exact
  Markov chain of :func:`haar_pauli_chain`, and ``haar_mu_bound``,
  ``mu_0 g(p)^(k - 1)``.

Run from the repository root:

    uv run python experiments/early_layers/early_layers.py N=8 seed=42

Everything is written to the Hydra run directory: ``metrics.json`` and the
config (and ``early_layers.npz``, the records, when ``save_arrays``).
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
from QELM.evaluation.early_layers import (
    haar_mass_rate,
    haar_pauli_chain,
    pauli_type_mass,
    reference_helstrom,
    split_reference_state,
    suffix_bounds,
)
from QELM.evaluation.entropy_regression import stratified_split
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.haar import HaarBrickworkCircuit, U1BrickworkCircuit
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config

HERE = Path(__file__).resolve().parent
# Seed tag of the generators of the measurements and the split.
RNG_TAG = 288
# Seed tag of the gates: that of experiments/nonlinear_M, so a seed draws the
# same Haar circuit there and here.
GATE_TAG = 269
# Regularizations tried by the ridge.
ALPHAS = np.logspace(-2, 4, 13)
GATES = ("haar", "u1")


def _nonlinear_module() -> ModuleType:
    """``experiments/nonlinear_M/nonlinear_M.py``: encodings and depth rule."""
    path = HERE.parent / "nonlinear_M" / "nonlinear_M.py"
    spec = importlib.util.spec_from_file_location("nonlinear_M_for_early", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NL = _nonlinear_module()


def make_circuit(
    gates: str, params: dict[str, Any], rng: np.random.Generator
) -> BrickworkCircuit:
    """The fixed brickwork: Haar U(4) gates or charge-conserving U(1) gates.

    Raises:
        ValueError: If ``gates`` is not one of :data:`GATES`.
    """
    if gates == "haar":
        return HaarBrickworkCircuit(params, rng)
    if gates == "u1":
        return U1BrickworkCircuit(params, rng)
    raise ValueError(f"early_task.gates must be one of {GATES}, got {gates!r}.")


def trajectory(
    psi0: np.ndarray,
    circuit: BrickworkCircuit,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One trajectory, with ``<Z_R>`` and the Helstrom value after each round.

    The protocol of ``nonlinear_M.trajectory``, plus the two diagnostics of
    ``R``, which draw no random number.

    Returns:
        ``(M, r_z, H, Y)``: the record ``(T, N)``; ``r_z`` and ``H``, shape
        ``(T + 1,)``, before any round (index 0) and after the measurements
        of each round; the Born outcome of ``R``, +-1.
    """
    N, N_total = circuit.N, circuit.N_total
    psi = psi0
    T = circuit.depth
    M = np.zeros((T, N))
    r_z = np.zeros(T + 1)
    H = np.zeros(T + 1)
    r_z[0], H[0] = reference_helstrom(psi)
    for t in range(T):
        psi = circuit.apply_layer(psi, t)
        for k in range(N):
            if rng.random() < p:
                psi, M[t, k] = measure_projective_psi(psi, k + 1, N_total, rng)
        r_z[t + 1], H[t + 1] = reference_helstrom(psi)
    psi, Y = measure_projective_psi(psi, 0, N_total, rng)
    return M, r_z, H, float(Y)


def _sign(y: np.ndarray) -> np.ndarray:
    return np.where(y >= 0.0, 1.0, -1.0)


def _accuracy(model: RidgeCV, X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(_sign(np.asarray(model.predict(X))) == Y))


def _fit(X: np.ndarray, Y: np.ndarray) -> RidgeCV | None:
    """A ridge on ``(X, Y)``, or ``None`` when there is nothing to learn."""
    if np.unique(Y).size < 2 or X.shape[1] == 0 or np.ptp(X, axis=0).max() < 1e-12:
        return None
    return RidgeCV(alphas=ALPHAS).fit(X, Y)


def _window_accuracy(
    M_tr: np.ndarray,
    Y_tr: np.ndarray,
    M_te: np.ndarray,
    Y_te: np.ndarray,
    rounds: slice,
) -> float:
    """Test accuracy of a ridge retrained on the rounds ``rounds`` only."""
    X_tr = M_tr[:, rounds].reshape(M_tr.shape[0], -1)
    X_te = M_te[:, rounds].reshape(M_te.shape[0], -1)
    model = _fit(X_tr, Y_tr)
    if model is None:
        majority = 1.0 if np.mean(Y_tr) >= 0.0 else -1.0
        return float(np.mean(Y_te == majority))
    return _accuracy(model, X_te, Y_te)


def linear_analysis(
    M: np.ndarray,
    Y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    c: np.ndarray,
    D: np.ndarray,
    p: float,
) -> dict[str, Any]:
    """Round-by-round analysis of the ridge on ``vec(M)`` (one ``p``).

    Args:
        M: Records, shape ``(L, T, N)``.
        Y: Labels, +-1.
        train, test: Indices of the split.
        c: Exact ``E[m_{t,i} Y]``, shape ``(T, N)``.
        D: Exact ``D_t``, shape ``(T,)``.
        p: Measurement probability.

    Returns:
        The keys ``accuracy``, ``alpha``, ``weight_l2``, ``weight_l1``,
        ``share``, ``share_bound``, ``signal``, ``ablate_late``,
        ``ablate_early``, ``prefix``, ``suffix`` (lists of length ``T`` for
        the per-round ones) described in the module docstring.
    """
    L, T, N = M.shape
    X = M.reshape(L, -1)
    out: dict[str, Any] = {}
    model = _fit(X[train], Y[train])
    if model is None:
        zeros = [0.0] * T
        chance = float(np.mean(Y[test] == (1.0 if Y[train].mean() >= 0 else -1.0)))
        out.update(
            accuracy=chance,
            alpha=float("nan"),
            weight_l2=zeros,
            weight_l1=zeros,
            share=zeros,
            share_bound=zeros,
            signal=0.0,
            ablate_late=[chance] * T,
            ablate_early=[chance] * T,
            prefix=[chance] * T,
            suffix=[chance] * T,
        )
        return out
    w = np.asarray(model.coef_).reshape(T, N)
    per_round = np.sum(w * c, axis=1)
    signal = float(per_round.sum())
    out["accuracy"] = _accuracy(model, X[test], Y[test])
    out["alpha"] = float(model.alpha_)
    out["weight_l2"] = np.linalg.norm(w, axis=1).tolist()
    out["weight_l1"] = np.abs(w).sum(axis=1).tolist()
    out["signal"] = signal
    scale = signal if abs(signal) > 1e-15 else float("nan")
    out["share"] = (per_round / scale).tolist()
    out["share_bound"] = (np.abs(w).sum(axis=1) * p * D / abs(scale)).tolist()
    mean = M[train].mean(axis=0)
    late, early, prefix, suffix = [], [], [], []
    for k in range(1, T + 1):
        kept = M[test].copy()
        kept[:, k:] = mean[k:]
        late.append(_accuracy(model, kept.reshape(test.size, -1), Y[test]))
        kept = M[test].copy()
        kept[:, : k - 1] = mean[: k - 1]
        early.append(_accuracy(model, kept.reshape(test.size, -1), Y[test]))
        prefix.append(
            _window_accuracy(M[train], Y[train], M[test], Y[test], slice(0, k))
        )
        suffix.append(
            _window_accuracy(M[train], Y[train], M[test], Y[test], slice(k - 1, T))
        )
    out.update(ablate_late=late, ablate_early=early, prefix=prefix, suffix=suffix)
    return out


def haar_average(
    a: np.ndarray, b: np.ndarray, circuit: BrickworkCircuit, p: float
) -> dict[str, list[float]]:
    """Haar average over circuits of the Pauli mass and of the linear signal.

    Returns:
        ``haar_mu`` (``E[mu_k]``), ``haar_mu_bound``
        (``mu_0 g(p)^(k - 1)``), ``haar_D_bound`` (``sqrt(E[mu_k]) / 2``, a
        bound on ``E[D_k]``), ``haar_c_rms`` (``sqrt(mean_i E[c_{k,i}^2])``).
    """
    delta = np.outer(a, a.conj()) - np.outer(b, b.conj())
    mass = pauli_type_mass(delta)
    offset = circuit.Nr
    rounds = [
        [(i - offset, j - offset) for i, j, _ in layer] for layer in circuit.layers
    ]
    chain = haar_pauli_chain(mass, rounds, p)
    mu0 = float(mass.sum() - mass[(0,) * mass.ndim])
    g = haar_mass_rate(p)
    T = len(rounds)
    return {
        "haar_mu": chain["mu"].tolist(),
        "haar_mu_bound": [mu0 * g**k for k in range(T)],
        "haar_D_bound": np.minimum(1.0, 0.5 * np.sqrt(chain["mu"])).tolist(),
        "haar_c_rms": (0.5 * p * np.sqrt(chain["single_z"].mean(axis=1))).tolist(),
    }


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories of every ``p`` and measure every round.

    Args:
        cfg: Composed config (``load_config(..., config_name="early_layers")``
            or Hydra).

    Returns:
        ``M``, ``Y``, ``r_z``, ``H``, ``p`` and ``metrics`` (``per_p``: one
        dict per ``p``).

    Raises:
        ValueError: If ``Nr`` is not 1, a probability is outside ``[0, 1]``,
            a count is too small or the gates are unknown.
    """
    set_seed(cfg)
    N, Nr, _ = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"early_layers needs the reference qubit: Nr=1, got {Nr}.")
    task = cfg.early_task
    levels = np.asarray([float(v) for v in task.p_values])
    if levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"early_task.p_values must be in [0, 1], got {levels}.")
    if np.unique(levels).size != levels.size:
        raise ValueError(f"early_task.p_values must be distinct, got {levels}.")
    per_p = int(task.samples_per_p)
    if per_p < 10:
        raise ValueError(f"early_task.samples_per_p must be >= 10, got {per_p}.")
    T = NL.depth_from_rule(task.depth_rule, N)
    gates = str(task.get("gates", "haar"))
    encoding = str(task.get("encoding", "bell"))

    seed = int(cfg.seed)
    gate_rng = np.random.default_rng([seed, GATE_TAG, 0])
    rng = np.random.default_rng([seed, RNG_TAG, 1])
    split_rng = np.random.default_rng([seed, RNG_TAG, 2])
    block = OmegaConf.to_container(cfg.haar_circuit, resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    params["depth"] = T
    circuit = make_circuit(gates, params, gate_rng)
    psi0 = NL.initial_state(cfg, encoding)
    a, b = split_reference_state(psi0)
    want_haar = gates == "haar" and bool(task.get("haar_average", True))

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, T, N))
    r_z = np.zeros((L, T + 1))
    H = np.zeros((L, T + 1))
    Y = np.zeros(L)
    start = time.perf_counter()
    for i in tqdm(range(L)):
        M[i], r_z[i], H[i], Y[i] = trajectory(psi0, circuit, float(p[i]), rng)
    elapsed = time.perf_counter() - start

    per_level: list[dict[str, Any]] = []
    for level in levels:
        idx = np.flatnonzero(p == level)
        train, test = stratified_split(
            Y[idx], float(cfg.samples.train_fraction), split_rng
        )
        bounds = suffix_bounds(circuit.layers, a, b, float(level), offset=Nr)
        entry: dict[str, Any] = {"p": float(level)}
        entry.update(
            linear_analysis(
                M[idx], Y[idx], train, test, bounds["c"], bounds["D"], float(level)
            )
        )
        entry["D"] = bounds["D"].tolist()
        entry["suffix_bound"] = (0.5 * (1.0 + bounds["D"])).tolist()
        entry["c"] = bounds["c"].tolist()
        entry["charge"] = bounds["charge"].tolist()
        entry["c_hat"] = np.mean(M[idx] * Y[idx, None, None], axis=0).tolist()
        entry["bayes_prefix"] = (0.5 * (1.0 + np.abs(r_z[idx]).mean(axis=0))).tolist()
        entry["helstrom"] = (0.5 * (1.0 + H[idx].mean(axis=0))).tolist()
        entry["bayes_accuracy"] = float(np.mean(_sign(r_z[idx, -1]) == Y[idx]))
        if want_haar:
            entry.update(haar_average(a, b, circuit, float(level)))
        per_level.append(entry)
    metrics = {
        "N": N,
        "T": T,
        "seed": seed,
        "gates": gates,
        "encoding": encoding,
        "samples_per_p": per_p,
        "per_p": per_level,
    }
    return {
        "M": M,
        "Y": Y,
        "r_z": r_z,
        "H": H,
        "p": p,
        "metrics": metrics,
        "timing": {"time_simulation_s": elapsed},
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="early_layers")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory."""
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.early_task.get("save_arrays", False)):
        np.savez(
            out_dir / f"{Path(__file__).stem}.npz",
            **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
        )
    metrics = {**results["metrics"], **results["timing"]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics))
    save_config(cfg, out_dir)
    m = results["metrics"]
    print(f"N: {m['N']}  T: {m['T']}  seed: {m['seed']}  {m['gates']} {m['encoding']}")
    print(f"{'p':>6} {'acc':>6} {'bayes':>6} {'share 1-3':>10} {'D_T':>8}")
    for e in m["per_p"]:
        early = float(np.sum(e["share"][:3]))
        print(
            f"{e['p']:6.3f} {e['accuracy']:6.3f} {e['bayes_accuracy']:6.3f} "
            f"{early:10.3f} {e['D'][-1]:8.2e}"
        )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
