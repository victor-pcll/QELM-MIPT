#!/usr/bin/env python3
"""QELM experiment: learn the charge ``Q`` under intermediate Z measurements.

Issue #163. A preparatory noise experiment: strong projective measurements in
the Z basis, applied during the evolution with probability ``p``, disturb the
entanglement of the reservoir, and the readout has to recover the charge from
the final state alone. For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) on the ``Ns = N - 1`` system
   qubits and start from the state of :class:`AncillaChargeSectorEncoding`,
   ``(|0>_R |Q1>_S + |1>_R |Q2>_S) / sqrt(2)`` with ``Q1 = Ns // 2`` and
   ``Q2 = Ns // 2 - 1``. The ancilla ``R`` is measured in Z first, with the
   Born rule: the outcome fixes the charge sector of the system, the label
   ``Y[i] = Q`` (``Q1`` for ``+1``, ``Q2`` for ``-1``, probability 1/2
   each). ``R`` never evolves (``U = 1_R (x) U_S``, issue #104), so measuring
   it first or last gives the same statistics; measuring it first makes the
   features below depend on ``Q``. Without it the reduced state of ``S``
   would be the same mixture of both sectors for every sample, and no
   readout of ``Tr(Z_k rho)`` could tell the two charges apart.
2. Split the total time ``evolution.t`` into ``nbr_time_evolution`` steps of
   ``t / nbr_time_evolution`` (``projective_measurement.nbr_time_evolution``,
   default 5). Between two steps, each system qubit is measured in Z with
   probability ``projective_measurement.p`` and the state collapses. The
   measurements are intermediate by default: there is none after the last
   step, so the final state has evolved for ``t / nbr_time_evolution`` since
   the last collapse, and ``nbr_time_evolution - 1`` rounds are applied in
   total. With ``projective_measurement.final_measurement=true`` a last
   round follows the last step too (``nbr_time_evolution`` rounds): the
   features are then read right after a collapse, so the disruption of the
   entanglement is seen in the final state, which the last unitary step
   otherwise rebuilds. A measured qubit then has ``<Z_k> = +-1``, its
   outcome.
3. The features ``X[i]`` are the final expectation values
   ``<Z_k> = Tr(Z_k rho)`` of the system qubits ``k = 1 .. Ns``, plus a bias
   column. The outcomes of the intermediate measurements are **not**
   features: they only disturb the state.

Which qubits are measured and their outcomes are drawn from their own
generator, seeded with ``(seed, 163)``, while the disorder and the label of
``R`` come from the global NumPy state seeded by ``set_seed``. So for a given
seed every ``p`` sees the same Hamiltonians and the same labels, and at
``p = 0`` the run is exactly the standard QELM without measurements, one
unitary ``exp(-i H t)`` per sample (the baseline of issue #163).

``h = 10 J`` makes the TFIM almost conserve the charge (the ``X_i X_j``
couplings change it by 0 or 2), and a measurement in Z commutes with ``Q``.
So the measurements do not erase ``Q`` itself; what they change is the
state the charge is carried by, and how far the final ``<Z_k>`` spread
around their mean.

The same readout is also fitted, separately, on six feature sets read from
the same final states (issue #163, follow-up): the one-site ``<X_k>``,
``<Y_k>``, ``<Z_k>`` and the two-site correlators ``<X_i X_j>``,
``<Y_i Y_j>``, ``<Z_i Z_j>`` over the pairs ``i < j`` of system qubits, and
the cross correlators ``<X_i Y_j>``, ``<Y_i X_j>`` (set ``XY``). The
TFIM and the Z measurements conserve the parity of ``Q``, while ``X_k`` and
``Y_k`` flip it, so ``<X_k> = <Y_k> = 0`` exactly: these two sets are the
chance level. ``<X_i X_j>`` and ``<Y_i Y_j>`` keep the parity and are
nonzero only through coherences between basis states, which the Z
measurements erase: unlike ``<Z_k>``, they lose the charge as ``p`` grows.
Each set ``<name>`` gets ``accuracy_<name>``, ``balanced_accuracy_<name>``
and ``r2_<name>``; ``accuracy`` and the other unsuffixed scores are those of
``<Z_k>``.

Three more readouts take the outcomes of the intermediate measurements as
features too (issue #179): the record ``M`` of shape
``(n_rounds, Ns)``, +-1 for a measured qubit and 0 for an unmeasured one, as
in ``experiments/QELM_M/exp_QELM_M.py``, with ``n_rounds`` the number of
measurement rounds. The sets of :data:`RECORD_SETS` are ``M`` alone, and
``phi = [M, X]`` with ``X`` the ``<Z_k>`` (``M_Z``) or the ``<X_i X_j>``
(``M_XX``) of the final state; they get the same three scores. The record
is only read by these readouts: it never changes the state, the other
feature sets or the unsuffixed scores. At ``p = 0`` it is all zeros, so
``M`` alone is at chance and ``M_Z``, ``M_XX`` are ``Z``, ``XX``.

The half-system entanglement entropy of the final state, ``S_half[i]``
(von Neumann, in bits, of the first ``Ns // 2`` system qubits; the state of
``S`` is pure once ``R`` is measured), tracks how much the measurements
disentangle the reservoir.

A ridge readout (``training`` block) is fitted on the first
``samples.train_size`` samples to predict ``Q`` and scored on the rest with
NMSE, R2 and, the primary metric, ``accuracy``: a prediction counts as right
when it falls on the side of ``(Q1 + Q2) / 2`` of the true charge. With it,
``accuracy_baseline`` (always the majority charge of the training set, the
chance level) and ``balanced_accuracy`` (0.5 for a readout that ignores the
features), as in ``experiments/QELM_M/exp_QELM_M.py`` (issue #130).

Two K-NN classifiers (:class:`QELM.evaluation.knn.CustomKNN`, ``K =
round(sqrt(train_size))``, inverse-distance votes) estimate how much of ``Q``
a nonlinear rule can read from the same ``<Z_k>`` (bias column dropped) on the
same split (issue #213): ``bayes_estimate_knn`` with the Euclidean distance
and ``bayes_estimate_knn_mahalanobis`` with the Mahalanobis distance, whose
metric is the pseudo-inverse of the covariance ``Sigma`` of the training
``<Z_k>``. ``knn_covariance_rank`` is the number of eigenvalues of ``Sigma``
the pseudo-inverse keeps and ``knn_covariance_cond`` the ratio of the largest
to the smallest kept one: a large ratio means that the Mahalanobis metric
amplifies directions of tiny spread.

Run from the repository root:

    uv run python experiments/QELM_Q/exp_QELM_Q.py N=4 samples.L=500 \
        projective_measurement.p=0.5

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_Q.npz``: ``X`` (the ``<Z_k>`` features, bias column included),
  ``features_<name>`` for each set of :data:`FEATURE_SETS` and
  :data:`RECORD_SETS`, ``M`` (the record, ``(L, n_rounds, Ns)``), ``Y`` (the
  charge), ``y_test``, ``y_pred``, the weights ``W``, ``S_half`` and
  ``n_measured`` (number of collapses per sample), of shape ``(L,)``;
- ``metrics.json``: ``nmse``, ``nmse_err``, ``r2``, ``accuracy``,
  ``accuracy_baseline``, ``balanced_accuracy``, ``S_half_mean``,
  ``n_measured_mean``, the two K-NN accuracies and ``knn_covariance_rank``,
  ``knn_covariance_cond``, the scores of each feature set and the shapes used;
- ``.hydra/config.yaml`` and ``resolved_config.json``: the config of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from QELM import TransverseFieldIsingModel, check_qubit_counts, set_seed
from QELM.encoding import AncillaChargeSectorEncoding
from QELM.evaluation import von_neumann_entropy
from QELM.evaluation.knn import CustomKNN
from QELM.evaluation.metrics import (
    NMSE,
    R2,
    NMSE_err,
    accuracy,
    balanced_accuracy,
    majority_accuracy,
)
from QELM.physics import X as PAULI_X
from QELM.physics import Y as PAULI_Y
from QELM.physics import Z, local_expectation, partial_trace
from QELM.training import ProjectiveMeasurement, RidgeReadout
from QELM.training.measurements.projective import measure_projective_rho
from QELM.utils import save_config, save_results

# Feature sets, each fitted with its own readout: one-site expectation values
# and two-site correlators over the pairs i < j of system qubits; "XY" holds
# both cross correlators <X_i Y_j> and <Y_i X_j> of each pair.
PAULIS = {"X": PAULI_X, "Y": PAULI_Y, "Z": Z}
FEATURE_SETS = ("X", "Y", "Z", "XX", "YY", "ZZ", "XY")
# Readouts on the measurement record M, alone or next to the observables of
# a feature set, phi = [M, X] (issue #179): name -> sets of FEATURE_SETS.
RECORD_SETS: dict[str, tuple[str, ...]] = {"M": (), "M_Z": ("Z",), "M_XX": ("XX",)}
# Scalars written to metrics.json besides nmse, nmse_err and r2.
EXTRA_METRICS = (
    "accuracy",
    "accuracy_baseline",
    "balanced_accuracy",
    "S_half_mean",
    "n_measured_mean",
    "bayes_estimate_knn",
    "bayes_estimate_knn_mahalanobis",
    "knn_covariance_rank",
    "knn_covariance_cond",
    *(
        f"{score}_{name}"
        for name in (*FEATURE_SETS, *RECORD_SETS)
        for score in ("accuracy", "balanced_accuracy", "r2")
    ),
)
# Number of time-evolution steps (issue #132).
NBR_TIME_EVOLUTION = "projective_measurement.nbr_time_evolution"
# Optional measurement round after the last step.
FINAL_MEASUREMENT = "projective_measurement.final_measurement"
# Second word of the seed of the measurement generator: the issue number.
MEASUREMENT_STREAM = 163


def evolve_with_measurements(
    rho: np.ndarray,
    U: np.ndarray,
    n_steps: int,
    P: ProjectiveMeasurement,
    rng: np.random.Generator,
    final: bool = False,
    record: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Apply ``n_steps`` steps ``U``, with a measurement round between two.

    Args:
        rho: Initial density matrix of shape ``(2**N, 2**N)``.
        U: One step of the evolution, same shape.
        n_steps: Number of steps, ``>= 1``.
        P: Measurement of the system qubits, each with its probability ``p``.
        rng: Generator of the measurement choices and outcomes.
        final: Also measure after the last step (issue #163, follow-up).
        record: If given, the outcomes of each round are appended to it, an
            array of length the number of measured sites, ``nan`` for a site
            not measured (issue #179).

    Returns:
        The final state, after the last step (and its round when ``final``),
        and the number of qubits measured on the way.
    """
    U_dag = U.conj().T
    n_measured = 0
    for j in range(n_steps):
        rho = U @ rho @ U_dag
        if final or j < n_steps - 1:
            measure = P.measure(rho, rng=rng)
            rho = measure.rho
            if record is not None:
                record.append(measure.outcomes)
            n_measured += int(np.sum(np.isfinite(measure.outcomes)))
    return rho, n_measured


def observable_features(rho: np.ndarray, system: list[int], name: str) -> np.ndarray:
    """Return the features of one set of :data:`FEATURE_SETS` for ``rho``.

    Args:
        rho: Density matrix of shape ``(2**N, 2**N)``.
        system: The system qubits.
        name: ``"X"``, ``"Y"`` or ``"Z"`` for ``<P_k>`` on each qubit of
            ``system``; ``"XX"``, ``"YY"`` or ``"ZZ"`` for ``<P_i P_j>`` on
            each pair ``i < j``, from the two-qubit reduced state; ``"XY"``
            for ``<X_i Y_j>`` then ``<Y_i X_j>`` on each pair.

    Returns:
        Real array of length ``len(system)``, ``len(system) choose 2``, or
        twice that for ``"XY"``.

    Raises:
        ValueError: If ``name`` is not a set of :data:`FEATURE_SETS`.
    """
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}, expected {FEATURE_SETS}.")
    if len(name) == 1:
        return np.array([local_expectation(rho, PAULIS[name], k) for k in system])
    products = [np.kron(PAULIS[name[0]], PAULIS[name[1]])]
    if name[0] != name[1]:
        products.append(np.kron(PAULIS[name[1]], PAULIS[name[0]]))
    return np.array(
        [
            np.real(np.trace(PP @ partial_trace(rho, [i, j])))
            for n, i in enumerate(system)
            for j in system[n + 1 :]
            for PP in products
        ]
    )


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the readout on ``Q`` and score it.

    Args:
        cfg: Composed config (``load_config`` or Hydra). Reads
            ``evolution.t``, ``samples.L``, ``samples.train_size``, ``seed``,
            the TFIM, encoding, measurement and training blocks, and
            ``projective_measurement.nbr_time_evolution`` (default 5) and
            ``projective_measurement.final_measurement`` (default ``False``).

    Returns:
        The arrays and metrics of the run: ``X`` (with the bias column),
        ``Y`` (the charge ``Q``), ``M`` (the record of the intermediate
        measurements, ``(L, n_rounds, Ns)``, 0 for an unmeasured qubit),
        ``y_test``, ``y_pred``, ``W``, ``S_half``,
        ``n_measured``, ``nmse``, ``nmse_err``, ``r2``, ``accuracy``,
        ``accuracy_baseline``, ``balanced_accuracy``, ``S_half_mean``,
        ``n_measured_mean``, ``bayes_estimate_knn``,
        ``bayes_estimate_knn_mahalanobis``, ``knn_covariance_rank``,
        ``knn_covariance_cond`` and, for each set ``<name>`` of
        :data:`FEATURE_SETS` and :data:`RECORD_SETS`, ``features_<name>``
        (with the bias column),
        ``accuracy_<name>``, ``balanced_accuracy_<name>`` and ``r2_<name>``.
    """
    set_seed(cfg)
    rng = np.random.default_rng([int(cfg.seed), MEASUREMENT_STREAM])

    sector = AncillaChargeSectorEncoding.from_config(cfg)
    solver = RidgeReadout.from_config(cfg)
    N, Ns, Nr = check_qubit_counts(cfg)
    nbr_time_evolution = int(OmegaConf.select(cfg, NBR_TIME_EVOLUTION, default=5))
    if nbr_time_evolution < 1:
        raise ValueError(
            f"{NBR_TIME_EVOLUTION} must be >= 1, got {nbr_time_evolution}."
        )
    dt = float(cfg.evolution.t) / nbr_time_evolution
    final = bool(OmegaConf.select(cfg, FINAL_MEASUREMENT, default=False))
    n_rounds = nbr_time_evolution if final else nbr_time_evolution - 1
    rho0 = sector.rho()
    P = ProjectiveMeasurement.from_config(cfg)
    Q1, Q2 = sector.Q1, sector.Q2

    L = int(cfg.samples.L)
    features: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_SETS}
    Y = np.zeros(L)
    S_half = np.zeros(L)
    n_measured = np.zeros(L)
    M = np.zeros((L, n_rounds, Ns))
    system = list(range(Nr, N))
    half = system[: Ns // 2]

    # The TFIM block with the system size: R does not evolve (issue #104).
    tfim = OmegaConf.to_container(cfg[TransverseFieldIsingModel.name], resolve=True)
    assert isinstance(tfim, dict)
    tfim_params = {**{str(k): v for k, v in tfim.items()}, "N": Ns}
    identity_R = np.eye(2**Nr)

    for i in tqdm(range(L)):
        model = TransverseFieldIsingModel(tfim_params)
        U = np.kron(identity_R, model.propagator(dt=dt))
        # The label first: R measured in Z fixes the charge sector of S.
        rho, r = measure_projective_rho(rho0, 0, N)
        Y[i] = Q1 if r == 1 else Q2
        rounds: list[np.ndarray] = []
        rho, n_measured[i] = evolve_with_measurements(
            rho, U, nbr_time_evolution, P, rng, final, record=rounds
        )
        # An unmeasured qubit counts as 0, as in exp_QELM_M (issue #91).
        if rounds:
            M[i] = np.nan_to_num(np.array(rounds), nan=0.0)
        for name in FEATURE_SETS:
            features[name].append(observable_features(rho, system, name))
        S_half[i] = von_neumann_entropy(rho, half)

    train = int(cfg.samples.train_size)
    y_train, y_test = Y[:train], Y[train:]
    # The two charges are told apart by the side of their midpoint.
    threshold = (Q1 + Q2) / 2

    # One readout per feature set, each with a bias column (RidgeReadout has
    # no intercept). <Z_k> is the main one, saved as X.
    observed = {name: np.array(features[name]).reshape(L, -1) for name in FEATURE_SETS}
    # The record next to the observables, phi = [M, X] (issue #179).
    inputs = {
        **observed,
        **{
            name: np.hstack([M.reshape(L, -1), *(observed[s] for s in obs)])
            for name, obs in RECORD_SETS.items()
        },
    }
    sets: dict[str, Any] = {}
    for name, values in inputs.items():
        phi = np.hstack([values, np.ones((L, 1))])
        readout = RidgeReadout.from_config(cfg)
        readout.fit(phi[:train], y_train)
        pred = readout.predict(phi[train:])
        sets[f"features_{name}"] = phi
        sets[f"accuracy_{name}"] = accuracy(y_test, pred, threshold)
        sets[f"balanced_accuracy_{name}"] = balanced_accuracy(y_test, pred, threshold)
        sets[f"r2_{name}"] = R2(y_test, pred)

    X = sets["features_Z"]
    solver.fit(X[:train], y_train)
    y_pred = solver.predict(X[train:])

    # The K-NN ceilings on the same <Z_k> and split, no bias column (#213).
    Z_k = observed["Z"]
    knn = CustomKNN().fit(Z_k[:train], y_train)
    knn_m = CustomKNN(metric="mahalanobis").fit(Z_k[:train], y_train)
    kept = np.linalg.eigvalsh(knn_m.covariance_)[::-1][: knn_m.rank_]
    knn_metrics = {
        "bayes_estimate_knn": knn.score(Z_k[train:], y_test),
        "bayes_estimate_knn_mahalanobis": knn_m.score(Z_k[train:], y_test),
        "knn_covariance_rank": knn_m.rank_,
        "knn_covariance_cond": float(kept[0] / kept[-1]) if kept.size else np.nan,
    }

    return {
        "X": X,
        "Y": Y,
        "M": M,
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "S_half": S_half,
        "n_measured": n_measured,
        "nmse": NMSE(y_test, y_pred),
        "nmse_err": NMSE_err(y_test, y_pred),
        "r2": R2(y_test, y_pred),
        "accuracy": accuracy(y_test, y_pred, threshold),
        "accuracy_baseline": majority_accuracy(y_train, y_test, threshold),
        "balanced_accuracy": balanced_accuracy(y_test, y_pred, threshold),
        "S_half_mean": float(np.mean(S_half)),
        "n_measured_mean": float(np.mean(n_measured)),
        **knn_metrics,
        **sets,
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(results, out_dir, Path(__file__).stem, extra_metrics=EXTRA_METRICS)
    save_config(cfg, out_dir)
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g} +- {results['nmse_err']:.2g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"accuracy: {results['accuracy']:.6g}")
    print(f"accuracy baseline (majority charge): {results['accuracy_baseline']:.6g}")
    print(f"balanced accuracy: {results['balanced_accuracy']:.6g}")
    print(f"S_half (mean, bits): {results['S_half_mean']:.6g}")
    print(f"K-NN accuracy (Euclidean): {results['bayes_estimate_knn']:.6g}")
    print(
        "K-NN accuracy (Mahalanobis): "
        f"{results['bayes_estimate_knn_mahalanobis']:.6g}"
        f" (rank {results['knn_covariance_rank']},"
        f" cond {results['knn_covariance_cond']:.3g})"
    )
    for name in (*FEATURE_SETS, *RECORD_SETS):
        print(f"accuracy on <{name}>: {results[f'accuracy_{name}']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
