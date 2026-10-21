#!/usr/bin/env python3
"""QELM on the measurement record ``M`` at a fixed sample-to-parameter ratio.

Issue #96. The ridge readout of :mod:`exp_QELM_M` learns one weight per
feature: the flattened record ``M`` of shape ``(nbr_time_evolution, Ns)``
and the bias column, so

    n_features = nbr_time_evolution * Ns + 1,

whatever ``p`` (an unmeasured qubit is still a feature, equal to 0). This
script sets the number of training samples from the ratio

    alpha_ratio = n_train / n_features      (experiment.alpha_ratio)

as ``samples.train_size = max(1, round(alpha_ratio * n_features))`` and adds
a test set of fixed size ``experiment.n_test``, so ``samples.L =
train_size + n_test``. Then it runs :func:`exp_QELM_M.run` unchanged: same
reservoir, same measurements, same scores.

``alpha_ratio < 1`` is the underdetermined regime, fewer training equations
than weights: ``Phi^T Phi`` is singular and only the ridge penalty
``training.alpha > 0`` makes the fit unique. ``alpha_ratio > 1`` is the
overdetermined regime. ``alpha_ratio = 1`` is the interpolation threshold,
where a weakly regularized readout fits the training noise exactly and the
test error peaks (double descent). The ratio is counted on the training set,
not on all ``L`` samples, since only the training samples constrain ``W``.

``experiment.alpha_ratio`` and the ridge penalty ``training.alpha`` are two
independent keys (issue #96); a ratio in the ``training`` block is
rejected. ``experiment.alpha_ratio=null`` keeps ``samples.L`` and
``samples.train_fraction`` as in :mod:`exp_QELM_M`.

Run from the repository root:

    uv run python experiments/QELM_M/exp_QELM_M_ratio.py N=4 experiment.alpha_ratio=0.5

The run directory holds the files of :mod:`exp_QELM_M`; ``metrics.json``
adds ``alpha_ratio`` (requested), ``n_train``, ``n_params`` and
``train_ratio = n_train / n_features`` (achieved, after rounding), next to
``n_samples``, ``n_test`` and ``n_features``. Hydra writes
``.hydra/config.yaml`` before the run, with the default ``samples.L``; the
config with the sizes actually used is ``config_used.yaml``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from QELM import check_qubit_counts
from QELM.utils import save_results

RATIO = "experiment.alpha_ratio"
N_TEST = "experiment.n_test"


def _load_base() -> ModuleType:
    """Import exp_QELM_M.py, next to this file; experiments/ is not a package."""
    path = Path(__file__).resolve().with_name("exp_QELM_M.py")
    spec = importlib.util.spec_from_file_location("exp_QELM_M", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
# Scalars written to metrics.json besides nmse and r2: those of exp_QELM_M
# and the sample counts of the ratio.
EXTRA_METRICS = (
    *base.EXTRA_METRICS,
    "alpha_ratio",
    "n_train",
    "n_params",
    "train_ratio",
)


def n_params(cfg: DictConfig) -> int:
    """Number of weights of the readout, ``nbr_time_evolution * Ns + 1``.

    Computed from the config, before any sample is drawn: one feature per
    measurement step and system qubit, plus the bias column.
    """
    _, Ns, _ = check_qubit_counts(cfg)
    steps = int(OmegaConf.select(cfg, base.NBR_TIME_EVOLUTION, default=5))
    return steps * Ns + 1


def sample_sizes(alpha_ratio: float, n_features: int, n_test: int) -> tuple[int, int]:
    """Return ``(L, train_size)`` for a ratio ``n_train / n_features``.

    Args:
        alpha_ratio: Training samples per weight, > 0.
        n_features: Number of weights of the readout, >= 1.
        n_test: Size of the test set, >= 1.

    Raises:
        ValueError: If one of the arguments is out of range.
    """
    if not alpha_ratio > 0:
        raise ValueError(f"{RATIO} must be > 0, got {alpha_ratio}.")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    if n_test < 1:
        raise ValueError(f"{N_TEST} must be >= 1, got {n_test}.")
    train = max(1, round(alpha_ratio * n_features))
    return train + n_test, train


def run(cfg: DictConfig) -> dict[str, Any]:
    """Set ``samples.L`` from the ratio, then run :func:`exp_QELM_M.run`.

    Args:
        cfg: Composed config (``load_config(config_name="QELM_M_ratio")`` or
            Hydra). Reads ``experiment.alpha_ratio`` and ``experiment.n_test``
            besides the keys of :func:`exp_QELM_M.run`. Changed in place:
            ``samples.L`` and ``samples.train_size`` get the computed sizes,
            so the saved Hydra config holds the sizes actually used.

    Returns:
        The output of :func:`exp_QELM_M.run`, plus ``alpha_ratio``,
        ``n_train``, ``n_params`` and ``train_ratio``.

    Raises:
        ValueError: If the ratio sits in the ``training`` block, or the run
            is underdetermined without a ridge penalty.
        RuntimeError: If the features do not have ``n_params`` columns.
    """
    # The sampling ratio and the ridge penalty stay apart (issue #96).
    if OmegaConf.select(cfg, "training.alpha_ratio") is not None:
        raise ValueError(
            f"training.alpha_ratio is not read: the sample ratio is {RATIO}, "
            "the ridge penalty training.alpha."
        )
    alpha_ratio = OmegaConf.select(cfg, RATIO)
    n_features = n_params(cfg)
    if alpha_ratio is not None:
        L, train = sample_sizes(
            float(alpha_ratio), n_features, int(OmegaConf.select(cfg, N_TEST))
        )
        cfg.samples.L = L
        cfg.samples.train_size = train
    train = int(cfg.samples.train_size)
    if train < n_features and float(cfg.training.alpha) == 0.0:
        raise ValueError(
            f"{train} training samples for {n_features} weights: the fit is "
            "underdetermined and needs a ridge penalty training.alpha > 0."
        )

    results = base.run(cfg)

    # The weight count used for L must be the column count of X (issue #96).
    if results["X"].shape[1] != n_features:
        raise RuntimeError(
            f"X has {results['X'].shape[1]} features, expected {n_features} "
            "from nbr_time_evolution * Ns + 1."
        )
    return {
        **results,
        "alpha_ratio": float("nan") if alpha_ratio is None else float(alpha_ratio),
        "n_train": train,
        "n_params": n_features,
        "train_ratio": train / n_features,
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="QELM_M_ratio")
def main(cfg: DictConfig) -> None:
    """Run the experiment and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/QELM_M_ratio.yaml`` and CLI
            overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save_results(results, out_dir, Path(__file__).stem, extra_metrics=EXTRA_METRICS)
    OmegaConf.save(cfg, out_dir / "config_used.yaml", resolve=True)
    print(
        f"alpha_ratio: {results['alpha_ratio']:.6g}  "
        f"n_train: {results['n_train']}  n_params: {results['n_params']}"
    )
    print(f"X: {results['X'].shape}  Y: {results['Y'].shape}")
    print(f"NMSE: {results['nmse']:.6g}")
    print(f"R2: {results['r2']:.6g}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
