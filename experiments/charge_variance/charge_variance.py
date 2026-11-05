#!/usr/bin/env python3
"""Learn the per-trajectory charge variance ``Var(Q)`` from the final state (#299).

Can a QELM readout of the final state of a monitored U(1) circuit read,
trajectory by trajectory, the variance of the global charge, the order
parameter of the charge-sharpening transition? The identity

    Var(Q) = (1/4) [ sum_ij <Z_i Z_j> - (sum_i <Z_i>)^2 ]

is linear in the two-point functions but quadratic in the one-point
functions, so the features of the readout should matter.

One trajectory, at one size ``N`` and one measurement probability ``p``
(:func:`QELM.evaluation.charge_variance.sample_trajectories`):

1. The chain of ``N`` qubits starts in ``(|Q0> + |Q1>) / sqrt(2)``, the
   uniform states of ``Q0 = N // 2`` and ``Q1 = N // 2 - 1``. A state of
   definite charge would have ``Var(Q) = 0`` at every ``p``.
2. ``T = N`` steps of a random U(1) brickwork (open chain, a new circuit
   per trajectory), every site measured in Z with probability ``p`` after
   each layer: the protocol of Barratt et al. (2022), as in
   ``experiments/learnability``.
3. The target is the exact ``Var(Q)`` of the final state, ``w (1 - w)``
   with ``w`` the weight left in ``Q0``: ``1/4`` while the record hides the
   charge, 0 once it reveals it.
4. The features are ``<Z_i>`` and ``<Z_i Z_j>`` (``i < j``) of the final
   state, exact, or estimated from ``n_shots`` Z-basis bitstrings
   (``task.n_shots``).

The readouts (:data:`READOUTS`):

- ``chance``: the mean of the training targets;
- ``n_meas``: ridge on the number of measurements of the trajectory and its
  square, a control that reads the record, not the state;
- ``sep``: ``P (1 - P)`` with ``P`` the posterior of ``Q0`` of the
  gate-ignorant optimal decoder of the record (the symmetric exclusion
  process of Barratt et al. 2022, ``QELM.evaluation.learnability``); no fit,
  and no state either: what an observer who sees only the record and not
  the gates believes the variance is;
- ``linear_z``: ridge on ``<Z_i>``;
- ``linear_zz``: ridge on ``<Z_i>`` and ``<Z_i Z_j>``, the standard QELM;
- ``quadratic_z``: ridge on ``<Z_i>``, their squares and the products
  ``<Z_i> <Z_j>``: the space of the term ``(sum_i <Z_i>)^2``;
- ``plugin``: the identity above applied to the features, no fit; exact on
  exact features, the plug-in estimator on shots;
- ``direct`` (shots only): the sample variance of the Hamming weight of the
  shots, the estimator an experiment would use without a readout.

Two protocols, as in ``experiments/clifford_state_readout`` (#292): **per
p**, one readout per ``p`` (``R2`` is the information beyond ``p``), and
**pooled**, one readout for every ``p`` that is not told ``p``; its mean
prediction at each ``p`` is its estimate of the order parameter
``<Var(Q)>(p)``.

Access assumption: the features are expectation values of the state of one
trajectory. An experiment prepares that state once, so it would need
post-selection on the record (or many copies) to estimate them, the same
limit as #292: the readout is an upper bound on what a QELM can do.

Run from the repository root:

    uv run python experiments/charge_variance/charge_variance.py N=8

Everything is written to the Hydra run directory: ``metrics.json`` (scores,
order parameters, per ``p`` and feature set), optionally
``trajectories.npz``, and the config.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from QELM.evaluation.charge_variance import (
    charge_moments,
    exact_features,
    sample_trajectories,
    sharpening_ratio,
    shot_estimates,
    variance_from_features,
)
from QELM.evaluation.learnability import sep_posterior
from QELM.utils import save_config

READOUTS = (
    "chance",
    "n_meas",
    "sep",
    "linear_z",
    "linear_zz",
    "quadratic_z",
    "plugin",
    "direct",
)
# Readouts with no fit: their prediction is a function of one trajectory.
UNFITTED = ("sep", "plugin", "direct")
RIDGE_ALPHAS = np.logspace(-6, 4, 21)


def feature_sets(n_shots: Sequence[int]) -> list[str]:
    """Names of the feature sets: ``exact``, then ``shots<n>`` for each ``n``."""
    return ["exact"] + [f"shots{int(s)}" for s in n_shots]


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
    n_shots: Sequence[int] = (),
    batch: int = 500,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate ``n_samples`` trajectories at each ``p`` and their features.

    Args:
        N: Number of sites.
        p_values: Measurement probabilities.
        n_samples: Trajectories per ``p``.
        seed: Seed; each ``p`` has its own stream ``(seed, N, p)``.
        depth: Time steps; ``None`` for ``N``.
        charges: The two charges; ``None`` for ``(N // 2, N // 2 - 1)``.
        boundary: ``"open"`` or ``"periodic"``.
        measure_every: ``"layer"`` or ``"period"``.
        n_shots: Shot counts of the finite-shot feature sets.
        batch: Trajectories simulated at once.
        progress: Show a progress bar.

    Returns:
        Arrays with a leading axis ``P`` (one row per ``p``) then ``L =
        n_samples``: ``var`` (exact ``Var(Q)``), ``mean_Q``, ``n_measured``,
        ``leakage``, ``sep_var`` (``P (1 - P)`` of the SEP posterior), the
        exact features ``z_exact`` ``(P, L, N)`` and ``zz_exact`` ``(P, L,
        N (N - 1) / 2)``, and for each ``n`` in ``n_shots`` the arrays
        ``z_shots<n>``, ``zz_shots<n>`` and ``direct_shots<n>``; plus ``p``.
    """
    P = len(p_values)
    out: dict[str, list[np.ndarray]] = {}

    def add(key: str, a: int, value: np.ndarray) -> None:
        out.setdefault(key, [[] for _ in range(P)])[a].append(value)  # type: ignore[arg-type]

    for a, p in enumerate(p_values):
        rng = np.random.default_rng([seed, N, round(float(p) * 1e9)])
        starts = range(0, n_samples, batch)
        for start in tqdm(starts, desc=f"N={N} p={p}", disable=not progress):
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
            mean_Q, var = charge_moments(tr.probs)
            z, zz = exact_features(tr.probs)
            post = sep_posterior(tr.outcomes, tr.charges, boundary, measure_every)
            add("var", a, var)
            add("mean_Q", a, mean_Q)
            add("n_measured", a, tr.n_measured)
            add("leakage", a, tr.leakage)
            add("sep_var", a, post[:, 0] * post[:, 1])
            add("z_exact", a, z)
            add("zz_exact", a, zz)
            for s in n_shots:
                est = shot_estimates(tr.probs, int(s), rng)
                add(f"z_shots{int(s)}", a, est.z)
                add(f"zz_shots{int(s)}", a, est.zz)
                add(f"direct_shots{int(s)}", a, est.var_direct)
    data = {
        key: np.stack([np.concatenate(rows) for rows in per_p])
        for key, per_p in out.items()
    }
    data["p"] = np.asarray(p_values, float)
    return data


def quadratic_features(z: np.ndarray) -> np.ndarray:
    """``<Z_i>``, their squares and the products ``<Z_i> <Z_j>``, ``i < j``."""
    i, j = np.triu_indices(z.shape[1], k=1)
    return np.concatenate([z, z**2, z[:, i] * z[:, j]], axis=1)


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))


def design(readout: str, z: np.ndarray, zz: np.ndarray, n_meas: np.ndarray) -> Any:
    """Return the design matrix of a fitted readout.

    Raises:
        ValueError: If ``readout`` is not a fitted readout.
    """
    builders: dict[str, Callable[[], np.ndarray]] = {
        "n_meas": lambda: np.stack([n_meas, n_meas**2], axis=1).astype(float),
        "linear_z": lambda: z,
        "linear_zz": lambda: np.concatenate([z, zz], axis=1),
        "quadratic_z": lambda: quadratic_features(z),
    }
    if readout not in builders:
        raise ValueError(f"{readout!r} is not a fitted readout.")
    return builders[readout]()


def predict_unfitted(
    readout: str,
    z: np.ndarray,
    zz: np.ndarray,
    sep_var: np.ndarray,
    direct: np.ndarray | None,
) -> np.ndarray:
    """Prediction of a readout of :data:`UNFITTED`.

    Raises:
        ValueError: If ``readout`` is not unfitted, or is ``direct`` with
            exact features.
    """
    if readout == "sep":
        return sep_var
    if readout == "plugin":
        return variance_from_features(z, zz)
    if readout == "direct":
        if direct is None:
            raise ValueError("direct needs shots.")
        return direct
    raise ValueError(f"{readout!r} is not an unfitted readout.")


def fit_predict(
    readout: str,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> np.ndarray:
    """Fit one readout on ``train`` and predict on ``test``.

    Args:
        readout: Name in :data:`READOUTS`.
        train, test: Dicts with ``var``, ``z``, ``zz``, ``n_measured``,
            ``sep_var`` and ``direct`` (``None`` for exact features).

    Returns:
        Predictions, shape ``(L_test,)``.

    Raises:
        ValueError: If ``readout`` is unknown.
    """
    if readout not in READOUTS:
        raise ValueError(f"Unknown readout {readout!r}, expected one of {READOUTS}.")
    if readout in UNFITTED:
        return predict_unfitted(
            readout, test["z"], test["zz"], test["sep_var"], test["direct"]
        )
    y = train["var"]
    if readout == "chance" or np.ptp(y) <= CONSTANT_TOL:
        return np.full(len(test["var"]), float(np.mean(y)))
    model = _ridge()
    model.fit(design(readout, train["z"], train["zz"], train["n_measured"]), y)
    X = design(readout, test["z"], test["zz"], test["n_measured"])
    return np.asarray(model.predict(X), float)


# Spread of the targets below which they count as constant (``p = 0``, where
# ``Var(Q) = 1/4`` up to rounding): ``R2`` is then undefined.
CONSTANT_TOL = 1e-9


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """``R2`` (NaN when ``y`` is constant) and mean absolute error."""
    r2 = float(r2_score(y, pred)) if np.ptp(y) > CONSTANT_TOL else float("nan")
    return {"R2": r2, "MAE": float(mean_absolute_error(y, pred))}


def split(L: int, train_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean training mask of ``L`` samples, ``round(train_fraction L)`` true."""
    mask = np.zeros(L, dtype=bool)
    mask[rng.permutation(L)[: int(round(train_fraction * L))]] = True
    return mask


def order_parameters(
    var: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    """``4 <Var(Q)>`` (1 at ``p = 0``) and :func:`sharpening_ratio`, with errors.

    The errors are bootstrap standard deviations over the trajectories.
    """
    v = np.asarray(var, float)
    m = 4 * float(v.mean())
    ratio = sharpening_ratio(v)
    boots_m, boots_r = [], []
    for _ in range(n_boot):
        r = v[rng.integers(0, v.size, v.size)]
        boots_m.append(4 * float(r.mean()))
        boots_r.append(sharpening_ratio(r))
    br = np.asarray(boots_r)
    br = br[np.isfinite(br)]
    return {
        "var4": m,
        "var4_err": float(np.std(boots_m, ddof=1)) if n_boot > 1 else float("nan"),
        "ratio": ratio,
        "ratio_err": float(np.std(br, ddof=1)) if br.size > 1 else float("nan"),
    }


def _subset(
    data: dict[str, np.ndarray], fs: str, a: int, rows: np.ndarray
) -> dict[str, Any]:
    return {
        "var": data["var"][a][rows],
        "z": data[f"z_{fs}"][a][rows],
        "zz": data[f"zz_{fs}"][a][rows],
        "n_measured": data["n_measured"][a][rows].astype(float),
        "sep_var": data["sep_var"][a][rows],
        "direct": data[f"direct_{fs}"][a][rows] if fs != "exact" else None,
    }


def _concat(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in parts[0]:
        vals = [part[key] for part in parts]
        out[key] = None if vals[0] is None else np.concatenate(vals)
    return out


def evaluate(
    data: dict[str, np.ndarray],
    readouts: Sequence[str],
    sets: Sequence[str],
    train_fraction: float,
    seed: int,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Score the readouts per ``p`` and pooled, for each feature set.

    Args:
        data: Output of :func:`simulate`.
        readouts: Names in :data:`READOUTS`; ``direct`` is skipped on exact
            features.
        sets: Feature sets, from :func:`feature_sets`.
        train_fraction: Fraction of each ``p`` used for training; the same
            split serves every protocol and feature set.
        seed: Seed of the split and of the bootstrap.
        n_boot: Bootstrap resamples of the order parameters.

    Returns:
        ``oracle`` (order parameters of the exact ``Var(Q)`` on the test
        set, per ``p``), ``oracle_all`` (the same on every trajectory),
        ``measurement_density``, ``max_leakage``, and ``results[set]`` with
        ``per_p[readout]`` (``R2``, ``MAE`` per ``p``) and
        ``pooled[readout]`` (``R2_all``, ``R2`` and ``MAE`` per ``p``,
        ``mean_prediction`` per ``p``, ``p_mean`` the baseline that knows
        ``p``).
    """
    var = data["var"]
    P, L = var.shape
    rng = np.random.default_rng([seed, 299])
    masks = np.stack([split(L, train_fraction, rng) for _ in range(P)])
    boot = np.random.default_rng([seed, 299, 1])
    oracle = [order_parameters(var[a][~masks[a]], n_boot, boot) for a in range(P)]
    oracle_all = [order_parameters(var[a], n_boot, boot) for a in range(P)]
    out: dict[str, Any] = {
        "p": data["p"].tolist(),
        "oracle": {k: [o[k] for o in oracle] for k in oracle[0]},
        "oracle_all": {k: [o[k] for o in oracle_all] for k in oracle_all[0]},
        "measurement_density": (data["n_measured"].mean(axis=1)).tolist(),
        "max_leakage": float(data["leakage"].max()),
        "results": {},
    }
    for fs in sets:
        use = [r for r in readouts if not (r == "direct" and fs == "exact")]
        train = [_subset(data, fs, a, masks[a]) for a in range(P)]
        test = [_subset(data, fs, a, ~masks[a]) for a in range(P)]
        per_p: dict[str, dict[str, list[float]]] = {}
        for r in use:
            rows = [
                scores(test[a]["var"], fit_predict(r, train[a], test[a]))
                for a in range(P)
            ]
            per_p[r] = {k: [row[k] for row in rows] for k in ("R2", "MAE")}
        tr, te = _concat(train), _concat(test)
        p_te = np.concatenate([np.full(len(test[a]["var"]), a) for a in range(P)])
        preds = {r: fit_predict(r, tr, te) for r in use}
        preds["p_mean"] = np.array([train[a]["var"].mean() for a in p_te])
        pooled: dict[str, dict[str, Any]] = {}
        for r, pred in preds.items():
            rows = [scores(te["var"][p_te == a], pred[p_te == a]) for a in range(P)]
            pooled[r] = {
                "R2_all": scores(te["var"], pred)["R2"],
                "R2": [row["R2"] for row in rows],
                "MAE": [row["MAE"] for row in rows],
                "mean_prediction": [float(pred[p_te == a].mean()) for a in range(P)],
            }
        out["results"][fs] = {"per_p": per_p, "pooled": pooled}
    return out


def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Simulate and score one size ``N``.

    Raises:
        ValueError: If a readout is unknown.
    """
    N = int(cfg.N)
    t = cfg.task
    readouts = [str(r) for r in t.readouts]
    unknown = sorted(set(readouts) - set(READOUTS))
    if unknown:
        raise ValueError(f"Unknown readouts {unknown}, expected some of {READOUTS}.")
    n_shots = [int(s) for s in t.n_shots]
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
        n_shots=n_shots,
        batch=int(t.batch),
        progress=bool(t.get("progress", False)),
    )
    sets = feature_sets(n_shots)
    metrics = evaluate(
        data, readouts, sets, float(t.train_fraction), int(cfg.seed), int(t.n_boot)
    )
    metrics.update(
        {
            "N": N,
            "seed": int(cfg.seed),
            "depth": int(t.depth),
            "samples_per_p": int(t.samples_per_p),
            "train_fraction": float(t.train_fraction),
            "n_shots": n_shots,
            "feature_sets": sets,
        }
    )
    return metrics, data


def save(
    metrics: dict[str, Any], data: dict[str, np.ndarray], out_dir: Path, arrays: bool
) -> None:
    """Write ``metrics.json`` and, if ``arrays``, ``trajectories.npz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if arrays:
        np.savez_compressed(out_dir / "trajectories.npz", **data)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="charge_variance")
def main(cfg: DictConfig) -> None:
    """Run one ``N`` and save it to the Hydra run directory."""
    metrics, data = run(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    save(metrics, data, out_dir, bool(cfg.task.save_arrays))
    save_config(cfg, out_dir)
    print(f"N: {metrics['N']}  seed: {metrics['seed']}")
    per_p = metrics["results"]["exact"]["per_p"]
    for a, p in enumerate(metrics["p"]):
        r2 = "  ".join(f"{r}={per_p[r]['R2'][a]:.3f}" for r in per_p)
        print(f"p={p:.3f}  4<Var>={metrics['oracle']['var4'][a]:.3f}  R2: {r2}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
