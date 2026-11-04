#!/usr/bin/env python3
"""Learning curves of the readouts of M for the Bloch vector of R (issue #282).

Is the gap between the QELM on M (a ridge on the raw record) and the Bayes
decoder (``R2 = 1`` on the exact Bloch vector ``r(M)``) a limit of the
*linear* readout, or of the number of trajectories? This script reads the
``bloch_purity.npz`` of one run of ``bloch_purity.py`` (with
``bloch_task.labels=exact`` and many ``samples_per_p``), holds out
``--n-test`` trajectories of each ``p`` and fits, on growing training sets
(``--sizes``):

- ``ridge_M``: the ridge on ``vec(M)`` (the QELM on M);
- ``ridge_M_abs``: the ridge on ``[vec(M), vec(|M|)]``;
- ``mlp``: the MLP of ``bloch_purity.fit_mlp`` on ``[vec(M), vec(|M|)]``.

A readout that rises with the training size towards 1 is limited by the
samples; one that saturates below 1 is limited by its form. Writes
``learning_curve_N<N>.json`` and ``learning_curve_N<N>.png`` (one panel per
``p``):

    uv run python experiments/bloch_purity/readout_learning_curve.py \\
        outputs/bloch_purity/big/bloch_purity.npz --out figs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
READOUTS: tuple[tuple[str, str, str], ...] = (
    ("ridge_M", "ridge on M (QELM)", "#d62728"),
    ("ridge_M_abs", "ridge on [M, |M|]", "#2ca02c"),
    ("mlp", "MLP on [M, |M|]", "#9467bd"),
)


def _bloch_module() -> ModuleType:
    path = HERE / "bloch_purity.py"
    spec = importlib.util.spec_from_file_location("bloch_purity_for_curve", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BP = _bloch_module()


def learning_curve(
    M: np.ndarray,
    bloch: np.ndarray,
    p: np.ndarray,
    sizes: list[int],
    n_test: int,
    seed: int = 282,
) -> dict[str, Any]:
    """Test ``R2`` of the Bloch vector for each ``p``, readout and size.

    Sizes larger than the trajectories left after the test set are skipped.
    """
    L = M.shape[0]
    X = M.reshape(L, -1)
    XA = np.hstack([X, np.abs(X)])
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"sizes": sizes, "n_test": n_test, "p": {}}
    for level in np.unique(p):
        idx = rng.permutation(np.flatnonzero(p == level))
        if idx.size <= n_test:
            raise ValueError(f"p = {level:g}: {idx.size} trajectories <= n_test.")
        test, pool = idx[-n_test:], idx[:-n_test]
        entry: dict[str, list[float | None]] = {name: [] for name, _, _ in READOUTS}
        for n in sizes:
            if n > pool.size:
                for name in entry:
                    entry[name].append(None)
                continue
            tr = pool[:n]
            preds = {
                "ridge_M": BP.fit_ridge(X[tr], bloch[tr], X[test]),
                "ridge_M_abs": BP.fit_ridge(XA[tr], bloch[tr], XA[test]),
                "mlp": BP.fit_mlp(XA[tr], bloch[tr], XA[test]),
            }
            for name, pred in preds.items():
                value = BP.r2_vector(bloch[test], pred)
                entry[name].append(float(value) if np.isfinite(value) else None)
        out["p"][f"{level:g}"] = entry
    return out


def plot(curve: dict[str, Any], N: int, out: Path) -> Path:
    levels = list(curve["p"])
    fig, axes = plt.subplots(
        1, len(levels), figsize=(3.6 * len(levels), 3.6), sharey=True, squeeze=False
    )
    sizes = np.asarray(curve["sizes"], dtype=float)
    for ax, level in zip(axes[0], levels, strict=True):
        for name, legend, color in READOUTS:
            y = np.asarray(
                [np.nan if v is None else v for v in curve["p"][level][name]]
            )
            ax.plot(sizes, y, "o-", color=color, label=legend)
        ax.axhline(1.0, color="black", ls="--", lw=1, label="Bayes optimal on M")
        ax.set_xscale("log")
        ax.set_title(f"p = {level}", fontsize=10)
        ax.set_xlabel("training trajectories")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("test R2, exact Bloch vector r(M)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Readouts of M vs training size, N = {N} "
        f"({curve['n_test']} test trajectories per p, one fixed circuit)",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / f"learning_curve_N{N}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1600, 3200, 6400, 12800, 25600]
    )
    parser.add_argument("--n-test", type=int, default=6400)
    args = parser.parse_args(argv)
    out = args.out or args.npz.parent
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    N = int(data["M"].shape[2])
    curve = learning_curve(data["M"], data["bloch"], data["p"], args.sizes, args.n_test)
    curve["N"] = N
    curve["T"] = int(data["M"].shape[1])
    (out / f"learning_curve_N{N}.json").write_text(json.dumps(curve, indent=2))
    path = plot(curve, N, out)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    main()
