#!/usr/bin/env python3
"""Learnability zone of the unitary QELM: Ridge against Bayes along ``h``.

Issue #194. The canonical QELM, a pure unitary evolution with no measurement
on the way, at one transverse field ``h``. The question is how much of the
charge the local magnetizations ``<Z_k>`` carry, and how much of it the
linear Ridge readout gets. For each sample ``i`` (``samples.L`` in total):

1. Draw a new TFIM (new disorder ``J_ij``) on the ``Ns = N - 1`` system
   qubits, ``H = h sum_i Z_i + sum_{i<j} J_ij X_i X_j``
   (:class:`QELM.TransverseFieldIsingModel`), and start from the state of
   :class:`AncillaChargeSectorEmbedding`,
   ``(|0>_R |Q1>_S + |1>_R |Q2>_S) / sqrt(2)``, ``Q1 = Ns // 2``,
   ``Q2 = Ns // 2 - 1``, the sectors uniform or product states
   (``ancilla_charge_sector.initial``).
2. The ancilla ``R`` is measured in Z, outcome ``0`` with probability 1/2:
   it fixes the charge sector of ``S``, the label ``Y[i] = Q``. The state of
   ``S`` is then the pure sector state ``|Q>_S``, so ``S`` is evolved as a
   state vector, ``|psi> = U |Q>_S``, with the dense ``U = exp(-i H t)``
   and ``t = evolution.t`` (``10 / J`` by default). ``R`` never evolves
   (issue #104), so this is the ``p = 0`` run of
   ``experiments/QELM_Q/exp_QELM_Q.py`` sample by sample: same disorder,
   same labels, same features (checked in the tests).
3. The features ``X[i]`` are the ``<Z_k> = <psi| Z_k |psi>`` of the system
   qubits, and nothing else.

Three classifiers are scored on the test samples (the last
``samples.L - samples.train_size``):

- **chance**: the majority charge of the training set, ``accuracy_chance``
  (:func:`QELM.evaluation.metrics.majority_accuracy`); its mean entropy is
  the entropy of the training shares, ``entropy_chance``, 1 bit for
  balanced classes (:func:`QELM.evaluation.bayes.prior_entropy`);
- **Ridge**: the ridge readout of the ``training`` block on ``[<Z_k>, 1]``
  (``RidgeReadout`` has no intercept), fitted on ``Q``. A prediction counts
  as right when it falls on the side of ``(Q1 + Q2) / 2`` of the true charge:
  ``accuracy_ridge``, ``balanced_accuracy_ridge``. Its output is a number,
  not a probability; its posterior is the Bayes posterior given that number
  only, ``P(Q | y_ridge)``, from a one-dimensional QDA fitted on the
  training outputs. That gives ``entropy_ridge``, the mean entropy of what
  the Ridge output says about ``Q``;
- **Bayes**: a QDA on the ``<Z_k>`` (:func:`QELM.evaluation.bayes.qda_posterior`),
  one Gaussian per charge, its estimate of the Bayes classifier:
  ``accuracy_bayes``, ``balanced_accuracy_bayes``, ``entropy_bayes``. It is
  the ceiling for Gaussian class densities, and an estimate otherwise.

``P_corr`` is the posterior of the true charge, and the entropies are the
mean binary entropy ``E[H(P_corr)]`` of Barratt et al. (2022), in bits by
default (``bayes.entropy_base``). The learnability zone of issue #194 is the
band between the chance and the Bayes curves.

Run from the repository root:

    uv run python experiments/QELM_bayes_h/exp_QELM_bayes_h.py N=6 TFIM.h=2

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``exp_QELM_bayes_h.npz``: ``X`` (the ``<Z_k>``, bias column included,
  as the Ridge sees them), ``Y`` (the charge), ``y_test``, ``y_pred`` (the
  Ridge output), ``W``, ``P_corr_ridge`` and ``P_corr_bayes`` (one per test
  sample);
- ``metrics.json``: ``nmse``, ``nmse_err``, ``r2`` of the Ridge, the scores
  above and the shapes used;
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
from QELM.embedding import AncillaChargeSectorEmbedding
from QELM.evaluation.bayes import (
    correct_probability,
    posterior_scores,
    prior_entropy,
    qda_posterior,
)
from QELM.evaluation.metrics import (
    NMSE,
    R2,
    NMSE_err,
    accuracy,
    balanced_accuracy,
    majority_accuracy,
)
from QELM.physics import Z, local_expectation
from QELM.training import RidgeReadout
from QELM.utils import save_config, save_results

# Scalars written to metrics.json besides nmse, nmse_err and r2.
EXTRA_METRICS = (
    "accuracy_chance",
    "entropy_chance",
    "accuracy_ridge",
    "balanced_accuracy_ridge",
    "entropy_ridge",
    "accuracy_bayes",
    "balanced_accuracy_bayes",
    "entropy_bayes",
)


def sector_states(sector: AncillaChargeSectorEmbedding) -> tuple[np.ndarray, ...]:
    """Return the system states ``|Q1>_S`` and ``|Q2>_S`` of the embedding.

    ``R`` is site 0, the most significant bit, so the first ``2**Ns``
    amplitudes of the embedded state are the branch ``|0>_R |Q1>_S`` and the
    last ``2**Ns`` the branch ``|1>_R |Q2>_S``, each of norm ``1 / sqrt(2)``.

    Args:
        sector: The embedding, one ancilla.

    Returns:
        The two normalized state vectors of shape ``(2**Ns,)``.
    """
    psi = sector.state()
    half = 2**sector.Ns
    return psi[:half] * np.sqrt(2), psi[half:] * np.sqrt(2)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate the samples, fit the classifiers on ``Q`` and score them.

    Args:
        cfg: Composed config (``load_config(config_name="QELM_bayes_h")`` or
            Hydra). Reads ``evolution.t``, ``samples.L``,
            ``samples.train_size``, ``seed``, the TFIM, embedding and
            training blocks and ``bayes.reg_param``, ``bayes.tol`` and
            ``bayes.entropy_base``.

    Returns:
        The arrays and metrics of the run: ``X`` (with the bias column),
        ``Y``, ``y_test``, ``y_pred``, ``W``, ``P_corr_ridge``,
        ``P_corr_bayes``, ``nmse``, ``nmse_err``, ``r2`` and the keys of
        :data:`EXTRA_METRICS`.
    """
    set_seed(cfg)
    sector = AncillaChargeSectorEmbedding.from_config(cfg)
    N, Ns, Nr = check_qubit_counts(cfg)
    if Nr != 1:
        raise ValueError(f"The experiment needs one ancilla, got Nr={Nr}.")
    psi_Q1, psi_Q2 = sector_states(sector)
    Q1, Q2 = sector.Q1, sector.Q2
    t = float(cfg.evolution.t)
    reg_param = float(cfg.bayes.reg_param)
    tol = float(cfg.bayes.tol)
    base = float(cfg.bayes.entropy_base)

    # The TFIM block with the system size: R does not evolve (issue #104).
    tfim = OmegaConf.to_container(cfg[TransverseFieldIsingModel.name], resolve=True)
    assert isinstance(tfim, dict)
    tfim_params = {**{str(k): v for k, v in tfim.items()}, "N": Ns}

    L = int(cfg.samples.L)
    Z_k = np.zeros((L, Ns))
    Y = np.zeros(L)
    for i in tqdm(range(L)):
        # Same draws, in the same order, as exp_QELM_Q: the disorder, then
        # the outcome of R (0 with probability Tr(P_0 rho) = 1/2).
        model = TransverseFieldIsingModel(tfim_params)
        first = np.random.random() < 0.5
        Y[i] = Q1 if first else Q2
        # The dense U = exp(-i H t), as in exp_QELM_Q: model.evolve
        # (expm_multiply) estimates a norm with random vectors drawn from
        # the global NumPy state, which would shift the disorder of the
        # next samples.
        psi = model.propagator(dt=t) @ (psi_Q1 if first else psi_Q2)
        Z_k[i] = [local_expectation(psi, Z, k) for k in range(Ns)]

    train = int(cfg.samples.train_size)
    y_train, y_test = Y[:train], Y[train:]
    threshold = (Q1 + Q2) / 2

    X = np.hstack([Z_k, np.ones((L, 1))])
    solver = RidgeReadout.from_config(cfg)
    solver.fit(X[:train], y_train)
    y_pred = solver.predict(X[train:])
    # Posterior of Q given the Ridge output alone: a 1D QDA on the outputs
    # of the training samples.
    classes, proba_ridge = qda_posterior(
        solver.predict(X[:train]), y_train, y_pred, reg_param, tol
    )
    ridge = posterior_scores(classes, proba_ridge, y_test, base)

    classes, proba_bayes = qda_posterior(
        Z_k[:train], y_train, Z_k[train:], reg_param, tol
    )
    bayes = posterior_scores(classes, proba_bayes, y_test, base)

    return {
        "X": X,
        "Y": Y,
        "y_test": y_test,
        "y_pred": y_pred,
        "W": solver.W,
        "P_corr_ridge": correct_probability(classes, proba_ridge, y_test),
        "P_corr_bayes": correct_probability(classes, proba_bayes, y_test),
        "nmse": NMSE(y_test, y_pred),
        "nmse_err": NMSE_err(y_test, y_pred),
        "r2": R2(y_test, y_pred),
        "accuracy_chance": majority_accuracy(y_train, y_test, threshold),
        "entropy_chance": prior_entropy(y_train, base),
        "accuracy_ridge": accuracy(y_test, y_pred, threshold),
        "balanced_accuracy_ridge": balanced_accuracy(y_test, y_pred, threshold),
        "entropy_ridge": ridge["entropy_mean"],
        "accuracy_bayes": bayes["accuracy"],
        "balanced_accuracy_bayes": bayes["balanced_accuracy"],
        "entropy_bayes": bayes["entropy_mean"],
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="QELM_bayes_h")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/QELM_bayes_h.yaml`` and CLI
            overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(results, out_dir, Path(__file__).stem, extra_metrics=EXTRA_METRICS)
    save_config(cfg, out_dir)
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    for name in ("chance", "ridge", "bayes"):
        print(
            f"{name}: accuracy {results[f'accuracy_{name}']:.4f}  "
            f"entropy {results[f'entropy_{name}']:.4f}"
        )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
