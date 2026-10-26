#!/usr/bin/env python3
"""Double descent of the QELM on MNIST digits, for one seed (issue #181).

The protocol follows the QELM image classification of "Harnessing quantum
extreme learning machines for image classification", Phys. Rev. Applied
(2025): PCA of the images, angle encoding, a fixed quantum reservoir and a
linear readout of local observables.

1. Load the task (:func:`QELM.datasets.load_digit_classes`): MNIST images
   of the digits of ``double_descent.classes`` (3 vs 5 by default; more
   digits make the task harder), image of digit ``classes[k]`` labelled
   ``k``. Draw a training pool of ``n_pool`` images and a test set of
   ``n_test`` images.
2. Reduce each image to ``N`` principal components, fitted on the training
   images of those digits, and map each to an angle in ``[0, pi]``
   (:class:`QELM.datasets.AngleEncoder`).
3. Encode the angles in a product state, qubit ``i`` in
   ``R_y(theta_i)|0>`` (:class:`QELM.encoding.RotationEncoding`), and
   apply one fixed quench ``U = exp(-i H t)``: a single TFIM, drawn once
   for the seed and shared by every image. Measure the one-site observables
   ``<P_i>`` and the correlators ``<A_i B_j>`` of the final state: the
   features, plus a bias column.
4. For each model (``double_descent.models``), fit the ridge readout
   (``training.alpha``, 0 by default: ``W = Phi^+ Y``, no penalty) to the
   one-hot targets ``Y``, ``K`` outputs for ``K`` classes, the predicted
   class being the largest output, on random training sets of
   ``n_train = round(alpha * n_features)`` images of the pool, for every
   ``alpha`` of ``double_descent.alpha_ratios``, ``n_repeats`` times, and
   score it on the test set
   (:func:`QELM.evaluation.double_descent.double_descent_sweep`).
   ``n_features`` is the rank of the model's feature matrix on the pool.

The three sanity checks of the issue are in the same run:

- the classical ELM (``elm``) has as many random ReLU neurons as the QELM
  has observables;
- ``qelm_t0`` measures the same observables at ``t = 0``; its features are
  checked against their closed form on the product state
  (``t0_max_deviation``, 0 up to rounding), so its readout is a linear
  regression on fixed functions of the encoded input;
- ``peak_alpha_<model>`` is the ratio where the median test MSE is largest,
  1 when the unregularized readout and the rank are right.

Run from the repository root:

    uv run python experiments/double_descent/double_descent.py seed=0

Everything is written to the Hydra run directory, by default
``outputs/<date>/<time>/``, which git ignores:

- ``double_descent.npz``: for each model ``m``, ``<m>_alpha``,
  ``<m>_n_train`` and the statistics ``<m>_train_mse``, ``<m>_test_mse``,
  ``<m>_train_error``, ``<m>_test_error``, ``<m>_weight_norm``, of shape
  ``(n_ratios, n_repeats)``; ``confusion_<m>``, the ``(K, K)`` confusion
  matrix on the test set of the readout fitted on the whole pool (entry
  ``[i, j]``: images of ``classes[i]`` predicted as ``classes[j]``);
  ``classes``, the angles ``theta_pool``, ``theta_test``, the class indices
  ``y_pool``, ``y_test`` and the QELM features ``features_qelm_pool``;
- ``metrics.json``: per model the rank ``n_features_<m>``, the number of
  columns ``n_columns_<m>``, the test accuracy of the readout fitted on the
  whole pool ``accuracy_<m>``, ``peak_alpha_<m>`` and the median test MSE
  at ``alpha = 1``; ``accuracy_pixels``, the classical reference (a linear
  readout on every pixel, fitted on all the training images of those
  digits) with its ``confusion_pixels``, and ``t0_max_deviation``;
- ``.hydra/config.yaml`` and ``resolved_config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from QELM import TransverseFieldIsingModel, check_qubit_counts, set_seed
from QELM.datasets import AngleEncoder, load_digit_classes
from QELM.encoding import RotationEncoding
from QELM.evaluation.double_descent import (
    SWEEP_STATISTICS,
    add_bias,
    double_descent_sweep,
    expectations,
    full_pool_fit,
    observable_operators,
    product_state_features,
    random_features,
)
from QELM.training import RidgeReadout
from QELM.utils import save_config

MODELS = ("qelm", "qelm_t0", "elm", "linear")


def _draw(n_available: int, n: int | None, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``n`` samples out of ``n_available``, all of them if ``None``."""
    if n is None:
        return rng.permutation(n_available)
    if n > n_available:
        raise ValueError(f"Asked for {n} samples, only {n_available} available.")
    return rng.choice(n_available, size=n, replace=False)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Build the features of every model and sweep the training-set size.

    Args:
        cfg: Composed config (``load_config(config_name="double_descent")``
            or Hydra). Reads ``N``, ``TFIM``, ``evolution.t``,
            ``rotation_encoding``, ``training.alpha`` and the
            ``double_descent`` block.

    Returns:
        The arrays of the ``.npz`` and the scalars of ``metrics.json``.

    Raises:
        ValueError: If a model is unknown, the config has an ancilla, or
            the pool is too small for the largest ratio.
    """
    seed = set_seed(cfg)
    rng = np.random.default_rng(seed)
    N, _, Nr = check_qubit_counts(cfg)
    if Nr != 0:
        raise ValueError(f"The image task has no ancilla: set Nr=0, got Nr={Nr}.")
    dd = cfg.double_descent
    models = [str(m) for m in dd.models]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Known: {', '.join(MODELS)}.")
    single = [str(label) for label in dd.single]
    pairs = [[str(a), str(b)] for a, b in dd.pairs]
    ridge_alpha = RidgeReadout.from_config(cfg).config_alpha

    # 1-2. The task, the pool, the test set and the angles.
    data = load_digit_classes(
        dataset=str(dd.dataset),
        classes=[int(d) for d in dd.classes],
        root=str(dd.data_dir),
        download=bool(dd.download),
        seed=seed,
    )
    pool = _draw(len(data.y_train), int(dd.n_pool), rng)
    test_n = OmegaConf.select(cfg, "double_descent.n_test")
    test = _draw(len(data.y_test), None if test_n is None else int(test_n), rng)
    encoder = AngleEncoder(N).fit(data.x_train)
    theta_pool = encoder.transform(data.x_train[pool])
    theta_test = encoder.transform(data.x_test[test])
    y_pool, y_test = data.y_train[pool], data.y_test[test]

    # 3. The encoded states and the quench: one TFIM for every image.
    encoding = RotationEncoding.from_config(cfg)
    names, ops = observable_operators(N, single, pairs)
    psi_pool, psi_test = encoding.states(theta_pool), encoding.states(theta_test)
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model = TransverseFieldIsingModel.from_config(cfg)
    t = float(cfg.evolution.t)
    U = model.propagator(dt=t)
    features["qelm"] = (
        add_bias(expectations(psi_pool @ U.T, ops)),
        add_bias(expectations(psi_test @ U.T, ops)),
    )

    # t = 0: the simulated features against their closed form.
    t0_pool = expectations(psi_pool, ops)
    closed = product_state_features(encoding.bloch_vectors(theta_pool), single, pairs)
    t0_max_deviation = float(np.max(np.abs(t0_pool - closed)))
    features["qelm_t0"] = (
        add_bias(t0_pool),
        add_bias(expectations(psi_test, ops)),
    )

    # Classical ELM with as many hidden neurons as measured observables. The
    # same random layer for the pool and the test set; its hyperplanes go
    # through images of the pool, so that every ReLU is active on some images
    # and not on others.
    elm_rng = np.random.default_rng([seed, 181])
    hidden = random_features(
        np.vstack([theta_pool, theta_test]),
        len(ops),
        elm_rng,
        activation=str(dd.elm_activation),
        anchors=theta_pool,
    )
    features["elm"] = (
        add_bias(hidden[: len(pool)]),
        add_bias(hidden[len(pool) :]),
    )
    features["linear"] = (add_bias(theta_pool), add_bias(theta_test))

    # 4. The sweep, one model at a time.
    results: dict[str, Any] = {
        "theta_pool": theta_pool,
        "theta_test": theta_test,
        "y_pool": y_pool,
        "y_test": y_test,
        "classes": np.array(data.classes),
        "n_classes": data.n_classes,
        "features_qelm_pool": features["qelm"][0],
        "observables": np.array(names + ["bias"]),
        "J_matrix": model.J_matrix,
        "t0_max_deviation": t0_max_deviation,
        "ridge_alpha": ridge_alpha,
        "n_pool": len(pool),
        "n_test": len(test),
        "models": models,
    }
    sweep_rng = np.random.default_rng([seed, 1])
    for name in models:
        phi_pool, phi_test = features[name]
        sweep = double_descent_sweep(
            phi_pool,
            y_pool,
            phi_test,
            y_test,
            [float(a) for a in dd.alpha_ratios],
            int(dd.n_repeats),
            sweep_rng,
            ridge_alpha=ridge_alpha,
            n_classes=data.n_classes,
        )
        results[f"{name}_alpha"] = sweep.alpha
        results[f"{name}_n_train"] = sweep.n_train
        for key in SWEEP_STATISTICS:
            results[f"{name}_{key}"] = getattr(sweep, key)
        results[f"n_features_{name}"] = sweep.n_features
        results[f"n_columns_{name}"] = sweep.n_columns
        results[f"peak_alpha_{name}"] = sweep.peak_alpha()
        at_one = np.flatnonzero(np.isclose(sweep.alpha, 1.0))
        results[f"test_mse_at_threshold_{name}"] = (
            float(np.median(sweep.test_mse[at_one[0]])) if at_one.size else float("nan")
        )
        accuracy, confusion = full_pool_fit(
            phi_pool, y_pool, phi_test, y_test, data.n_classes, ridge_alpha
        )
        results[f"accuracy_{name}"] = accuracy
        results[f"confusion_{name}"] = confusion
    if bool(dd.pixel_baseline):
        # Every training image of those digits, not only the pool: with 785
        # weights, a pool of 2000 images sits close to the interpolation peak
        # of the pixel readout itself.
        accuracy, confusion = full_pool_fit(
            add_bias(data.x_train),
            data.y_train,
            add_bias(data.x_test[test]),
            y_test,
            data.n_classes,
            ridge_alpha,
        )
        results["accuracy_pixels"] = accuracy
        results["confusion_pixels"] = confusion
    return results


def save(results: dict[str, Any], out_dir: Path) -> None:
    """Write the arrays to ``double_descent.npz``, the scalars to ``metrics.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    np.savez(out_dir / "double_descent.npz", **arrays)
    metrics: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            metrics[key] = value.item() if isinstance(value, np.generic) else value
    metrics["models"] = list(results["models"])
    metrics["classes"] = [int(d) for d in results["classes"]]
    metrics["ridge_alpha"] = float(results["ridge_alpha"])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


@hydra.main(version_base="1.3", config_path="../../conf", config_name="double_descent")
def main(cfg: DictConfig) -> None:
    """Run one seed and save it to the Hydra run directory.

    Args:
        cfg: Hydra config composed from ``conf/double_descent.yaml`` and CLI
            overrides.
    """
    results = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(results, out_dir)
    save_config(cfg, out_dir)
    classes = [int(d) for d in results["classes"]]
    print(f"classes: {classes}  ridge alpha: {results['ridge_alpha']}")
    print(f"pool: {results['n_pool']}  test: {results['n_test']}")
    print(f"t = 0 features vs closed form: {results['t0_max_deviation']:.2e}")
    for name in results["models"]:
        print(
            f"{name}: rank {results[f'n_features_{name}']}"
            f"/{results[f'n_columns_{name}']}"
            f"  accuracy {results[f'accuracy_{name}']:.4f}"
            f"  peak at alpha = {results[f'peak_alpha_{name}']:.3g}"
        )
    if "accuracy_pixels" in results:
        print(f"pixels: accuracy {results['accuracy_pixels']:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
