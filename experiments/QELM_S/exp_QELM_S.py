#!/usr/bin/env python3
"""QELM experiment: predict the half-chain entanglement entropy (issue #230).

Can a linear readout of local measurements learn a global, nonlinear
property of the state, the entanglement entropy, across the
measurement-induced phase transition (MIPT)? For each measurement
probability ``p`` of ``entropy_task.p_values`` and each of its
``entropy_task.samples_per_p`` trajectories:

1. Start from the Néel product state ``|0101...>`` on the ``N`` qubits (no
   ancilla, ``Nr = 0``) and draw new dynamics: a new TFIM disorder
   (``models=TFIM``, step ``U(dt)`` with ``dt = entropy_task.dt``) or new
   gates of a brickwork circuit (``models=haar_circuit``, ``clifford_circuit``
   or ``U1_circuit``, ``layers_per_step`` layers per step), or the fixed
   Floquet Ising chain of Skinner et al. (2019) (``models=floquet_circuit``,
   issue #239, whose layer is a half-layer of bricks, the time step of the
   paper), or the Ising chain in a tilted field in continuous time
   (``models=continuous_tfim_1d``, issue #241: windows of length ``tau``
   under ``exp(-i H dt)``, with Z measurements as a Poisson process of rate
   ``gamma = -ln(1 - p) / tau``, so that a qubit is measured at least once in
   a window with probability ``p``; row ``j`` of ``M`` holds the last outcome
   of each qubit in window ``j``).
2. Run ``n_rounds`` rounds (``entropy_task.n_rounds``, default ``2 N``, and
   about ``4 N`` half-layers, ``2 N`` periods, for ``floquet_circuit``,
   :func:`floquet_rounds`) of a
   unitary step followed by a projective measurement in Z of each qubit with
   probability ``p``. Row ``j`` of the record ``M`` holds the outcomes of
   round ``j``, +-1, and 0 for an unmeasured qubit.
3. The target is the von Neumann entropy, in bits, of the first ``N // 2``
   qubits in the final state, ``S = -Tr(rho_A log2 rho_A)``, read right after
   the last measurement round. The state is pure along the trajectory, so it
   is followed as a vector and ``S`` comes from its Schmidt values.
4. Three kinds of local features are read from the trajectory: the record
   ``M`` (flattened), the final ``<Z_k>`` (``Z``, expectation values of the
   final state, as in the other QELM experiments), and one single-shot
   measurement of every qubit in Z in the final state (``Zshot``, +-1, what an
   experiment would record; it does not change the state ``S`` is read
   from).

Every trajectory of every ``p`` goes into one dataset, split with the same
training share at each ``p`` (``samples.train_fraction``,
:func:`QELM.evaluation.entropy_regression.stratified_split`). A ridge readout
(``training`` block, bias column added) is fitted on each feature set of
:data:`FEATURE_SETS` to predict ``S``:

- ``p``: the indicators of ``p``, the baseline. It predicts the training mean
  of ``S`` at each ``p``, the best a function of ``p`` alone can do;
- ``n_meas``: the number of measured qubits of the record, a second baseline
  that only knows how many measurements happened;
- ``absM``: ``|M|``, where the measurements happened, without the outcomes;
- ``M``, ``Z``, ``Zshot``, ``M_Z``, ``M_Zshot``: the features of the issue;
- ``M_Z_even``: ``[M, Z]`` and their squares (``|M|``, ``<Z_k>**2``,
  :func:`~QELM.evaluation.entropy_regression.even_features`). ``S`` is even
  in the signs of the outcomes, a linear readout of ``M`` is odd: the squares
  give it even features;
- ``p_n_meas``, ``p_absM``, ``p_M_Z``, ``p_M_Z_even``: the indicators of
  ``p`` next to the features of the set. A readout without them pays at
  fixed ``p`` for every error of its offset at that ``p``, which a readout
  of ``p`` alone never makes; with them, the ``R2`` at fixed ``p`` measures
  only what the features add to ``p``.

Each set ``<name>`` gets ``r2_<name>`` and ``nmse_<name>`` on the test set
pooled over ``p``, and ``r2_within_<name>``, the ``R2`` at fixed ``p``
(:func:`~QELM.evaluation.entropy_regression.r2_within`): the share of the
spread of ``S`` between trajectories of the same ``p`` that the readout
explains. The ``p`` baseline scores about 0 there; a readout that only
learns ``p`` does too. A pooled ``r2`` well above the ``p`` baseline, or a
positive ``r2_within`` of a ``p_<set>`` readout, is the evidence that the
features encode the entanglement of the trajectory. With
``entropy_task.knn=true`` the same scores of a K-NN regression on the same
features are reported (``r2_knn_<name>``, ``r2_within_knn_<name>``, ``K =
round(sqrt(n_train))``, inverse-distance weights); with hundreds of
features, most of them random signs, it does worse than the ridge readout,
so it is not a ceiling here. When ``entropy_task.p_c`` is set, a ridge
readout of the phase label (+1 for ``p < p_c``, volume law; -1 above, area
law) is fitted on each set too, ``accuracy_phase_<name>``: the label is a
function of ``p``, so the ``p`` baseline reaches 1 and this score mostly
tells whether ``p`` can be read from the features.

The unsuffixed ``nmse``, ``nmse_err``, ``r2`` and ``y_pred`` are those of
``M_Z``. Run from the repository root:

    uv run python experiments/QELM_S/exp_QELM_S.py models=haar_circuit N=8

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_S.npz``: ``M`` (record, ``(L, n_rounds, N)``), ``Z``,
  ``Zshot`` (``(L, N)``), ``S`` and ``p`` (``(L,)``), ``train`` and ``test``
  (indices), ``y_test``, ``y_pred_<name>`` (and ``y_pred_knn_<name>``) on
  the test set, ``X`` (the ``M_Z`` features, bias column included);
- ``metrics.json``: the scores above, ``S_mean``, ``S_max`` (``N // 2``, the
  largest possible ``S``), ``n_rounds`` and, per ``p``, ``S_mean_p<p>``,
  ``S_std_p<p>`` and ``r2_within_<name>_p<p>`` for the main sets;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from QELM import TransverseFieldIsingModel, check_qubit_counts, set_seed
from QELM.evaluation.entropy import von_neumann_entropy
from QELM.evaluation.entropy_regression import (
    even_features,
    one_hot,
    phase_labels,
    r2_within,
    stratified_split,
)
from QELM.evaluation.metrics import NMSE, R2, NMSE_err, accuracy
from QELM.models.circuits.base import BrickworkCircuit
from QELM.models.circuits.clifford import CliffordBrickworkCircuit
from QELM.models.circuits.floquet import FloquetBrickworkCircuit
from QELM.models.circuits.haar import HaarBrickworkCircuit, U1BrickworkCircuit
from QELM.models.continuous_tfim_1d import ContinuousTFIMChain
from QELM.training import RidgeReadout
from QELM.training.measurements.poisson import poisson_monitored_batch
from QELM.training.measurements.projective import measure_projective_psi
from QELM.utils import save_config, save_results

# Feature sets fitted with their own readout. "p" and "n_meas" are the
# baselines; the others are read from the trajectory.
FEATURE_SETS = (
    "p",
    "n_meas",
    "absM",
    "M",
    "Z",
    "Zshot",
    "M_Z",
    "M_Zshot",
    "M_Z_even",
    "p_n_meas",
    "p_absM",
    "p_M_Z",
    "p_M_Z_even",
)
# Sets with the indicators of p next to the features: their R2 at fixed p
# measures what the features add to p (name -> set without p).
WITH_P = {
    "p_n_meas": "n_meas",
    "p_absM": "absM",
    "p_M_Z": "M_Z",
    "p_M_Z_even": "M_Z_even",
}
# The set behind the unsuffixed nmse, r2 and y_pred.
MAIN_SET = "M_Z"
# Sets whose R2 at each p is written to metrics.json, for the plots.
PER_P_SETS = ("p", "M_Z", "M_Z_even", "p_absM", "p_M_Z_even")
# The R2 at one p, r2_within_<name>_p<p>, as written by _scores.
PER_P_KEY = re.compile(r"r2_within_.+_p[0-9.e+-]+")
# Random circuits that can replace the TFIM, by models value.
CIRCUITS: dict[str, type[BrickworkCircuit]] = {
    HaarBrickworkCircuit.name: HaarBrickworkCircuit,
    CliffordBrickworkCircuit.name: CliffordBrickworkCircuit,
    U1BrickworkCircuit.name: U1BrickworkCircuit,
    FloquetBrickworkCircuit.name: FloquetBrickworkCircuit,
}
# Seed tag of the generator of the measurements and the single shots.
RNG_TAG = 230
# One unitary step of a trajectory: the state and the round index j.
Step = Callable[[np.ndarray, int], np.ndarray]


def neel_state(N: int) -> np.ndarray:
    """Return ``|0101...>`` on ``N`` qubits, qubit 0 the leftmost factor."""
    index = int("01" * (N // 2) + "0" * (N % 2), 2)
    psi = np.zeros(2**N, dtype=complex)
    psi[index] = 1.0
    return psi


def _tfim_sampler(cfg: DictConfig, N: int, dt: float) -> Callable[[], Step]:
    """Return a function that draws a TFIM and its step ``U(dt)``, per sample."""
    block = OmegaConf.to_container(cfg[TransverseFieldIsingModel.name], resolve=True)
    assert isinstance(block, dict)
    params = {**{str(k): v for k, v in block.items()}, "N": N}

    def draw() -> Step:
        U = TransverseFieldIsingModel(params).propagator(dt=dt)
        return lambda psi, j: U @ psi

    return draw


def _circuit_sampler(cfg: DictConfig, n_rounds: int) -> Callable[[], Step]:
    """Return a function that draws new circuit gates and their steps, per sample.

    Round ``j`` applies brickwork layers ``j * k`` to ``(j + 1) * k - 1``,
    with ``k = layers_per_step``.

    Raises:
        ValueError: If ``layers_per_step < 1``.
    """
    name = str(cfg.models)
    block = OmegaConf.to_container(cfg[name], resolve=True)
    assert isinstance(block, dict)
    params = {str(k): v for k, v in block.items()}
    per_step = int(params.get("layers_per_step", 1))
    if per_step < 1:
        raise ValueError(f"{name}.layers_per_step must be >= 1, got {per_step}.")
    params.setdefault("fixed_instance", False)
    circuit = CIRCUITS[name]({**params, "depth": n_rounds * per_step})

    def step(psi: np.ndarray, j: int) -> np.ndarray:
        for t in range(j * per_step, (j + 1) * per_step):
            psi = circuit.apply_layer(psi, t)
        return psi

    first = True

    def draw() -> Step:
        nonlocal first
        # The constructor drew the gates of the first sample.
        if not (first or circuit.fixed_instance):
            circuit.redraw()
        first = False
        return step

    return draw


def _continuous_trajectories(
    cfg: DictConfig,
    N: int,
    p: np.ndarray,
    n_windows: int,
    psi0: np.ndarray,
    rng: np.random.Generator,
    M: np.ndarray,
) -> np.ndarray:
    """Run every trajectory of the continuous-time chain (issue #241).

    ``n_disorder`` chains are drawn (couplings ``J_i``, from the global NumPy
    state); trajectory ``i`` evolves under chain ``i % n_disorder``, so that
    each chain gets every ``p``. The trajectories of one chain run as one
    batch (:func:`QELM.training.measurements.poisson.poisson_monitored_batch`)
    under ``U(dt) = exp(-i H dt)``, ``dt = tau / n_sub``, built from the
    eigenvectors of ``H``.

    Args:
        M: The record, ``(L, n_windows, N)``, filled in place.

    Returns:
        The final states, ``(2**N, L)``.

    Raises:
        ValueError: If ``tau <= 0``, ``n_sub < 1`` or ``n_disorder < 1``.
    """
    block = OmegaConf.to_container(cfg[ContinuousTFIMChain.name], resolve=True)
    assert isinstance(block, dict)
    params = {**{str(k): v for k, v in block.items()}, "N": N}
    tau = float(params.get("tau", 0.3))
    n_sub = int(params.get("n_sub", 3))
    n_disorder = int(params.get("n_disorder", 12))
    if tau <= 0 or n_sub < 1 or n_disorder < 1:
        raise ValueError(
            "continuous_tfim_1d needs tau > 0, n_sub >= 1 and n_disorder >= 1, got "
            f"{tau}, {n_sub}, {n_disorder}."
        )
    finals = np.zeros((psi0.shape[0], p.size), dtype=complex)
    for b in tqdm(range(n_disorder)):
        H = ContinuousTFIMChain(params).hamiltonian().toarray().real
        E, V = np.linalg.eigh(H)
        U_dt = (V * np.exp(-1j * E * tau / n_sub)) @ V.T
        idx = np.arange(b, p.size, n_disorder)
        if idx.size == 0:
            continue
        M[idx], finals[:, idx] = poisson_monitored_batch(
            U_dt, psi0, p[idx], n_windows, n_sub, rng
        )
    return finals


def floquet_rounds(N: int) -> int:
    """Return the default number of rounds of the Floquet chain on ``N`` qubits.

    About ``2 N`` periods, ``4 N`` half-layers, by when the entropy at
    ``p = 0`` has saturated, plus one when needed so that the last half-layer
    holds the bond ``(N // 2 - 1, N // 2)`` across the cut of ``S``. The
    half-layers alternate, and the entropy read right after the measurements
    depends on whether the cut bond got its gate in the last half-layer or
    one half-layer earlier (with one more round of measurements in between):
    without this choice the curves of ``N / 2`` even and odd split into two
    families (issue #239).
    """
    # Half-layer t holds the bond (i, i + 1) when i and t have the same parity.
    return 4 * N + (N // 2) % 2


def measure_round(
    psi: np.ndarray, p: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Measure each qubit in Z with probability ``p``, in order.

    Returns:
        ``(psi, outcomes)``: the collapsed state and the outcome of each
        qubit, +-1, or 0 when it was not measured.
    """
    N = psi.shape[0].bit_length() - 1
    outcomes = np.zeros(N)
    for k in range(N):
        if rng.random() < p:
            psi, outcomes[k] = measure_projective_psi(psi, k, N, rng)
    return psi, outcomes


def z_expectations(psi: np.ndarray) -> np.ndarray:
    """Return ``<Z_k>`` of every qubit ``k`` of a state vector."""
    N = psi.shape[0].bit_length() - 1
    weights = np.abs(psi) ** 2
    bits = (np.arange(psi.shape[0])[:, None] >> (N - 1 - np.arange(N))) & 1
    return np.asarray(weights @ (1 - 2 * bits), dtype=float)


def single_shot(psi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return one measurement of every qubit in Z, +-1, with the Born rule."""
    N = psi.shape[0].bit_length() - 1
    weights = np.abs(psi) ** 2
    index = int(rng.choice(psi.shape[0], p=weights / weights.sum()))
    bits = (index >> (N - 1 - np.arange(N))) & 1
    return np.asarray(1 - 2 * bits, dtype=float)


def feature_sets(
    M: np.ndarray, Z: np.ndarray, Zshot: np.ndarray, p: np.ndarray, levels: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the features of :data:`FEATURE_SETS`, without the bias column."""
    L = M.shape[0]
    flat = M.reshape(L, -1)
    absM = np.abs(flat)
    indicators = one_hot(p, levels)
    sets = {
        "p": indicators,
        "n_meas": absM.sum(axis=1, keepdims=True),
        "absM": absM,
        "M": flat,
        "Z": Z,
        "Zshot": Zshot,
        "M_Z": np.hstack([flat, Z]),
        "M_Zshot": np.hstack([flat, Zshot]),
        "M_Z_even": even_features(np.hstack([flat, Z])),
    }
    for name, base in WITH_P.items():
        sets[name] = np.hstack([indicators, sets[base]])
    return sets


def _with_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _scores(
    name: str, y_test: np.ndarray, y_pred: np.ndarray, p_test: np.ndarray
) -> dict[str, Any]:
    within, per_p = r2_within(y_test, y_pred, p_test)
    out: dict[str, Any] = {
        f"r2_{name}": R2(y_test, y_pred),
        f"nmse_{name}": NMSE(y_test, y_pred),
        f"r2_within_{name}": within,
    }
    return out | {f"r2_within_{name}_p{level:g}": r2 for level, r2 in per_p.items()}


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the trajectories, fit the readouts and score them.

    Args:
        cfg: Composed config (``load_config(..., config_name="QELM_S")`` or
            Hydra). Reads ``models``, ``N``, ``Nr`` (must be 0), the model
            block, ``entropy_task``, ``samples.train_fraction`` and the
            ``training`` block.

    Returns:
        The arrays and scores listed in the module docstring.

    Raises:
        ValueError: If ``Nr`` is not 0, ``models`` is not supported, a
            probability is outside ``[0, 1]`` or a count is too small.
    """
    set_seed(cfg)
    N, Ns, Nr = check_qubit_counts(cfg)
    if Nr != 0:
        raise ValueError(f"exp_QELM_S runs without ancilla: set Nr=0, got Nr={Nr}.")
    task = cfg.entropy_task
    levels = np.asarray([float(v) for v in task.p_values])
    if levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError(f"entropy_task.p_values must be in [0, 1], got {levels}.")
    if np.unique(levels).size != levels.size:
        raise ValueError(f"entropy_task.p_values must be distinct, got {levels}.")
    per_p = int(task.samples_per_p)
    if per_p < 2:
        raise ValueError(f"entropy_task.samples_per_p must be >= 2, got {per_p}.")
    model_name = str(cfg.models)
    if task.n_rounds is not None:
        n_rounds = int(task.n_rounds)
    elif model_name == FloquetBrickworkCircuit.name:
        n_rounds = floquet_rounds(N)
    elif model_name == ContinuousTFIMChain.name:
        # 4 N windows of length tau, as the 4 N half-layers of the Floquet chain.
        n_rounds = 4 * N
    else:
        n_rounds = 2 * N
    if n_rounds < 1:
        raise ValueError(f"entropy_task.n_rounds must be >= 1, got {n_rounds}.")

    draw_evolution: Callable[[], Step] | None = None
    if model_name == TransverseFieldIsingModel.name:
        draw_evolution = _tfim_sampler(cfg, N, float(task.dt))
    elif model_name in CIRCUITS:
        draw_evolution = _circuit_sampler(cfg, n_rounds)
    elif model_name != ContinuousTFIMChain.name:
        known = ", ".join(
            [TransverseFieldIsingModel.name, *CIRCUITS, ContinuousTFIMChain.name]
        )
        raise ValueError(f"models={model_name} is not supported here. Known: {known}.")

    # The dynamics come from the global NumPy state (set_seed), the
    # measurements and the single shots from their own generator, so a run
    # is reproducible for a given seed.
    seed = int(cfg.seed)
    rng = np.random.default_rng([seed, RNG_TAG])
    psi0 = neel_state(N)
    half = list(range(N // 2))

    L = levels.size * per_p
    p = np.repeat(levels, per_p)
    M = np.zeros((L, n_rounds, N))
    Z = np.zeros((L, N))
    Zshot = np.zeros((L, N))
    S = np.zeros(L)
    finals = (
        None
        if draw_evolution is not None
        else _continuous_trajectories(cfg, N, p, n_rounds, psi0, rng, M)
    )
    for i in tqdm(range(L)):
        if finals is not None:
            psi = finals[:, i]
        else:
            assert draw_evolution is not None
            evolve = draw_evolution()
            psi = psi0
            for j in range(n_rounds):
                psi = evolve(psi, j)
                psi, M[i, j] = measure_round(psi, float(p[i]), rng)
        S[i] = von_neumann_entropy(psi, half)
        Z[i] = z_expectations(psi)
        Zshot[i] = single_shot(psi, rng)

    split_rng = np.random.default_rng([seed, RNG_TAG, 1])
    train, test = stratified_split(p, float(cfg.samples.train_fraction), split_rng)
    y_train, y_test, p_test = S[train], S[test], p[test]
    features = feature_sets(M, Z, Zshot, p, levels)
    p_c = task.get("p_c")
    labels = None if p_c is None else phase_labels(p, float(p_c))
    k = max(1, round(float(np.sqrt(train.size))))

    results: dict[str, Any] = {}
    for name in FEATURE_SETS:
        X = _with_bias(features[name])
        solver = RidgeReadout.from_config(cfg)
        solver.fit(X[train], y_train)
        y_pred = solver.predict(X[test])
        results[f"y_pred_{name}"] = y_pred
        results |= _scores(name, y_test, y_pred, p_test)
        if bool(task.knn) and name != "p":
            knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
            knn.fit(features[name][train], y_train)
            y_knn = np.asarray(knn.predict(features[name][test]))
            results[f"y_pred_knn_{name}"] = y_knn
            results |= _scores(f"knn_{name}", y_test, y_knn, p_test)
        if labels is not None:
            classifier = RidgeReadout.from_config(cfg)
            classifier.fit(X[train], labels[train])
            results[f"accuracy_phase_{name}"] = accuracy(
                labels[test], classifier.predict(X[test])
            )

    # Keep per-p R2 only for the sets that are plotted.
    per_p_keys = {
        f"r2_within_{name}_p{level:g}" for name in PER_P_SETS for level in levels
    }
    results = {
        key: value
        for key, value in results.items()
        if not PER_P_KEY.fullmatch(key) or key in per_p_keys
    }
    y_main = results[f"y_pred_{MAIN_SET}"]
    per_p_stats: dict[str, float] = {}
    for level in levels:
        values = S[p == level]
        per_p_stats[f"S_mean_p{level:g}"] = float(values.mean())
        per_p_stats[f"S_std_p{level:g}"] = float(values.std())
    return {
        "M": M,
        "Z": Z,
        "Zshot": Zshot,
        "S": S,
        "p": p,
        "train": train,
        "test": test,
        "X": _with_bias(features[MAIN_SET]),
        "y_test": y_test,
        "y_pred": y_main,
        "nmse": NMSE(y_test, y_main),
        "nmse_err": NMSE_err(y_test, y_main),
        "r2": R2(y_test, y_main),
        "S_mean": float(S.mean()),
        "S_max": float(N // 2),
        "n_rounds": float(n_rounds),
        **per_p_stats,
        **results,
    }


def scalar_keys(results: dict[str, Any]) -> list[str]:
    """Return the keys of the scalar scores, for metrics.json."""
    skip = {"nmse", "nmse_err", "r2"}
    return [
        key
        for key, value in results.items()
        if isinstance(value, float) and key not in skip
    ]


@hydra.main(version_base="1.3", config_path="../../conf", config_name="QELM_S")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/QELM_S.yaml`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(
        results, out_dir, Path(__file__).stem, extra_metrics=scalar_keys(results)
    )
    save_config(cfg, out_dir)
    print(f"L: {results['S'].shape[0]}  N: {cfg.N}  model: {cfg.models}")
    print(f"S (mean, bits): {results['S_mean']:.4g} of at most {results['S_max']:g}")
    print(f"{'set':>10} {'r2':>8} {'r2_within':>10}")
    for name in FEATURE_SETS:
        r2, within = results[f"r2_{name}"], results[f"r2_within_{name}"]
        print(f"{name:>10} {r2:8.4f} {within:10.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
