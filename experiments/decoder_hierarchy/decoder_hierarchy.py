#!/usr/bin/env python3
"""Three decoders of the charge on the same U(1) trajectories (issue #307).

Agrawal et al. (PRX 2024, Appendix B) compare the optimal decoder that
knows the gates ("postBQP"), the stat-mech decoder that averages over them
(the symmetric exclusion process, SEP) and a trained network. This script
scores them, and the linear QELM readouts, on the same trajectories.

One trajectory, at one size ``N`` and one measurement probability ``p``
(:func:`QELM.evaluation.charge_variance.sample_trajectories`): the chain
starts in ``(|Q0> + |Q1>) / sqrt(2)``, ``Q0 = N // 2``, ``Q1 = Q0 - 1``;
``T = N`` steps of a random U(1) brickwork (open chain, a new circuit per
trajectory); every site measured in Z with probability ``p`` after each
layer. This is the protocol of Barratt et al. (2022) and of
``experiments/learnability``.

The decoders (:data:`DECODERS`), all guessing ``Q0`` or ``Q1``:

- ``bayes``: the posterior ``w`` of an observer who knows the gates (the
  weight left in ``Q0``), the optimal decoder;
- ``sep``: the posterior of an observer who sees the record but not the
  gates, the SEP of ``QELM.evaluation.learnability``; no fit;
- ``ridge_M``: ridge on ``vec(M)`` (+1 / -1 for the outcome, 0 when the
  site is not measured), the QELM-M;
- ``ridge_Z``: ridge on the ``<Z_i>`` of the final state, the QELM state
  readout; it needs the exact state of the trajectory, so it is an upper
  bound on what an experiment could do;
- ``n_ones``: ridge on the number of 0 and of 1 outcomes of the record, a
  control that counts particles without any position or time.

The ridges are trained, per ``p``, on a charge drawn from ``w`` for each
trajectory (the label an experiment sees). Every decoder is scored on the
held-out trajectories by its expected accuracy given ``w``
(:func:`QELM.evaluation.decoder_hierarchy.expected_accuracy`).

Run from the repository root:

    uv run python experiments/decoder_hierarchy/decoder_hierarchy.py N=8

``metrics.json`` and the config land in the Hydra run directory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from QELM.evaluation.charge_variance import exact_features, sample_trajectories
from QELM.evaluation.decoder_hierarchy import (
    bayes_share,
    draw_labels,
    expected_accuracy,
    guess_from_posterior,
    record_features,
    sector_weight,
)
from QELM.evaluation.learnability import UNMEASURED, sep_posterior
from QELM.utils import save_config

DECODERS = ("bayes", "sep", "ridge_M", "ridge_Z", "n_ones")
FITTED = ("ridge_M", "ridge_Z", "n_ones")
RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def simulate(
    N: int,
    p_values: Sequence[float],
    n_samples: int,
    seed: int,
    *,
    depth: int | None = None,
    charges: Sequence[int] | None = None,
    boundary: str = "open",
    measure_every: str = "layer",
    batch: int = 500,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate ``n_samples`` trajectories at each ``p``.

    Args:
        N: Number of sites.
        p_values: Measurement probabilities.
        n_samples: Trajectories per ``p``.
        seed: Seed; each ``p`` has its own stream ``(seed, N, p)``.
        depth: Time steps; ``None`` for ``N``.
        charges: The two charges; ``None`` for ``(N // 2, N // 2 - 1)``.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.
        batch: Trajectories simulated at once.
        progress: Show a progress bar.

    Returns:
        Arrays with a leading axis over ``p``: ``w`` (posterior of ``Q0``
        with the gates known), ``sep`` (SEP posterior of ``Q0``), ``M``
        (``vec(M)``), ``z`` (final ``<Z_i>``), ``counts`` (numbers of 0
        and 1 outcomes), ``leakage``; plus ``p``.
    """
    out: dict[str, list[list[np.ndarray]]] = {}
    P = len(p_values)

    def add(key: str, a: int, value: np.ndarray) -> None:
        out.setdefault(key, [[] for _ in range(P)])[a].append(value)

    for a, p in enumerate(p_values):
        rng = np.random.default_rng([seed, N, round(float(p) * 1e9), 307])
        for start in tqdm(
            range(0, n_samples, batch), desc=f"N={N} p={p}", disable=not progress
        ):
            n = min(batch, n_samples - start)
            tr = sample_trajectories(
                N,
                float(p),
                n,
                depth=depth,
                charges=charges,
                boundary=boundary,
                measure_every=measure_every,
                rng=rng,
            )
            z, _ = exact_features(tr.probs)
            post = sep_posterior(tr.outcomes, tr.charges, boundary, measure_every)
            measured = tr.outcomes != UNMEASURED
            ones = (tr.outcomes == 1).sum(axis=(1, 2))
            zeros = measured.sum(axis=(1, 2)) - ones
            add("w", a, sector_weight(tr.probs, tr.charges[0]))
            add("sep", a, post[:, 0])
            add("M", a, record_features(tr.outcomes))
            add("z", a, z)
            add("counts", a, np.stack([zeros, ones], axis=1).astype(float))
            add("leakage", a, tr.leakage)
    data = {k: np.stack([np.concatenate(r) for r in rows]) for k, rows in out.items()}
    data["p"] = np.asarray(p_values, float)
    return data


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))


def ridge_guess(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Fit a ridge on labels ``+-1`` (``+1`` for ``Q0``) and guess on the test set.

    Returns:
        Guesses 1 (``Q0``), 0 (``Q1``) or 0.5 (tie, or one class only).
    """
    y = 2.0 * np.asarray(y_train, float) - 1.0
    if np.ptp(y) == 0 or np.ptp(X_train) == 0:
        return np.full(len(X_test), 0.5)
    model = _ridge()
    model.fit(X_train, y)
    return guess_from_posterior(0.5 + 0.5 * np.asarray(model.predict(X_test)))


def evaluate(
    data: dict[str, np.ndarray],
    decoders: Sequence[str],
    train_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Score every decoder per ``p`` on held-out trajectories.

    Returns:
        ``p``; ``accuracy[decoder]`` and ``accuracy_err[decoder]`` (mean
        expected accuracy on the test set and its standard error over the
        trajectories); ``share[decoder]``, the share ``g`` of the advantage of
        Bayes; ``bayes_all`` (Bayes on every trajectory); ``max_leakage``.
    """
    w_all = data["w"]
    P, L = w_all.shape
    rng = np.random.default_rng([seed, 307])
    acc: dict[str, list[float]] = {d: [] for d in decoders}
    err: dict[str, list[float]] = {d: [] for d in decoders}
    for a in range(P):
        mask = np.zeros(L, dtype=bool)
        mask[rng.permutation(L)[: int(round(train_fraction * L))]] = True
        labels = draw_labels(w_all[a], rng)
        w = w_all[a][~mask]
        for d in decoders:
            if d == "bayes":
                guess = guess_from_posterior(w)
            elif d == "sep":
                guess = guess_from_posterior(data["sep"][a][~mask])
            else:
                key = {"ridge_M": "M", "ridge_Z": "z", "n_ones": "counts"}[d]
                X = data[key][a]
                guess = ridge_guess(X[mask], labels[mask], X[~mask])
            s = expected_accuracy(w, guess)
            acc[d].append(float(s.mean()))
            err[d].append(float(s.std(ddof=1) / np.sqrt(s.size)))
    bayes = np.asarray(acc["bayes"]) if "bayes" in acc else None
    share = {
        d: (
            bayes_share(np.asarray(acc[d]), bayes).tolist() if bayes is not None else []
        )
        for d in decoders
    }
    bayes_all = expected_accuracy(w_all, guess_from_posterior(w_all)).mean(axis=1)
    return {
        "p": data["p"].tolist(),
        "accuracy": acc,
        "accuracy_err": err,
        "share": share,
        "bayes_all": bayes_all.tolist(),
        "max_leakage": float(data["leakage"].max()),
    }


def run(cfg: DictConfig) -> dict[str, Any]:
    """Simulate and score one size ``N``.

    Raises:
        ValueError: If a decoder is unknown.
    """
    N = int(cfg.N)
    t = cfg.task
    decoders = [str(d) for d in t.decoders]
    unknown = sorted(set(decoders) - set(DECODERS))
    if unknown:
        raise ValueError(f"Unknown decoders {unknown}, expected some of {DECODERS}.")
    charges = None if t.charges is None else [int(q) for q in t.charges]
    data = simulate(
        N,
        [float(p) for p in t.p_values],
        int(t.samples_per_p),
        int(cfg.seed),
        depth=int(t.depth),
        charges=charges,
        boundary=str(t.boundary),
        measure_every=str(t.measure_every),
        batch=int(t.batch),
        progress=bool(t.get("progress", False)),
    )
    metrics = evaluate(data, decoders, float(t.train_fraction), int(cfg.seed))
    metrics.update(
        {
            "N": N,
            "seed": int(cfg.seed),
            "depth": int(t.depth),
            "samples_per_p": int(t.samples_per_p),
            "train_fraction": float(t.train_fraction),
        }
    )
    return metrics


@hydra.main(
    version_base="1.3", config_path="../../conf", config_name="decoder_hierarchy"
)
def main(cfg: DictConfig) -> None:
    """Run one ``N`` and save ``metrics.json`` to the Hydra run directory."""
    metrics = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_config(cfg, out_dir)
    print(f"N: {metrics['N']}  seed: {metrics['seed']}")
    for a, p in enumerate(metrics["p"]):
        row = "  ".join(
            f"{d}={metrics['accuracy'][d][a]:.3f}" for d in metrics["accuracy"]
        )
        print(f"p={p:.3f}  {row}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
