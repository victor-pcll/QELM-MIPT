"""K-NN figures of a QELM_Q sweep, experiments/QELM_Q/plot_knn.py (#214)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/QELM_Q/plot_knn.py"
L, TRAIN = 60, 40


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_knn", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(
    sweep: Path,
    N: int,
    p: float,
    seed: int,
    final: bool,
    X: np.ndarray,
    Y: np.ndarray,
    h: float = 1.0,
) -> None:
    run = sweep / f"N{N}_h{h}_final{final}_seed{seed}_p{p}"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text(
        f"N: {N}\nseed: {seed}\nTFIM:\n  h: {h}\n"
        f"projective_measurement:\n  p: {p}\n"
        f"  final_measurement: {str(final).lower()}\n"
        f"samples:\n  L: {L}\n  train_size: {TRAIN}\n"
    )
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.75}))
    X_bias = np.column_stack([X, np.ones(len(X))])
    np.savez(run / "exp_QELM_Q.npz", X=X_bias, Y=Y)


@pytest.fixture
def sweep(tmp_path: Path) -> Path:
    """Two Gaussian sectors without the final round, ±1 features with it."""
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        for p in (0.0, 1.0):
            Y = rng.integers(0, 2, size=L)
            continuous = rng.normal(size=(L, 3)) + 2.0 * Y[:, None] * (1.0 - p)
            _write_run(tmp_path, 4, p, seed, False, continuous, Y)
            cube = rng.choice([-1.0, 1.0], size=(L, 3))
            _write_run(tmp_path, 4, p, seed, True, cube, Y)
    return tmp_path


def test_resolve_k() -> None:
    module = _load_script()
    assert module.resolve_k("1", 40) == 1
    assert module.resolve_k("sqrt", 40) == 6
    assert module.resolve_k("3sqrt", 40) == 18
    assert module.resolve_k("100", 40) == 40
    assert module.k_label("3sqrt", 40) == "K = 3·round(√n) = 18"
    assert module.k_label("sqrt", 40) == "K = round(√n) = 6"
    for bad in ("0", "x", "sqrt3", "-1"):
        with pytest.raises(ValueError, match="K must be"):
            module.resolve_k(bad, 40)


def test_scores_match_both_classifiers(sweep: Path) -> None:
    from sklearn.neighbors import KNeighborsClassifier

    from QELM.evaluation.knn import CustomKNN

    module = _load_script()
    rows = module.load_knn_sweep(sweep, ["1", "sqrt"])
    assert len(rows) == 8
    for row in rows:
        with np.load(sweep / row["run"] / "exp_QELM_Q.npz") as data:
            X, Y = data["X"][:, :-1], data["Y"]
        ref = KNeighborsClassifier(n_neighbors=6, weights="distance")
        ref.fit(X[:TRAIN], Y[:TRAIN])
        assert row["K"] == 6
        assert row["accuracy_knn_sklearn"] == ref.score(X[TRAIN:], Y[TRAIN:])
        ours = CustomKNN().fit(X[:TRAIN], Y[:TRAIN])
        assert row["accuracy_knn_custom"] == ours.score(X[TRAIN:], Y[TRAIN:])
        assert row["accuracy_knn_sqrt"] == row["accuracy_knn_custom"]
        one = CustomKNN(n_neighbors=1).fit(X[:TRAIN], Y[:TRAIN])
        assert row["accuracy_knn_1"] == one.score(X[TRAIN:], Y[TRAIN:])
        if not row["final"]:
            # Continuous features: no tie, the same labels.
            assert row["tied"] == 0.0
            assert row["agreement"] == 1.0
        else:
            # 40 points on the 8 vertices of the cube: the K-th is tied.
            assert row["tied"] > 0.5


def test_tied_kth() -> None:
    module = _load_script()
    X = np.array([[0.0], [1.0], [-1.0], [3.0]])
    tied = module.tied_kth(X, np.array([[0.0], [2.0]]), 2)
    np.testing.assert_array_equal(tied, [True, False])
    assert not module.tied_kth(X, np.array([[0.0]]), 4).any()


def test_main_writes_both_figures(sweep: Path) -> None:
    module = _load_script()
    module.main([str(sweep), "--K", "1", "sqrt", "2sqrt"])
    for name in ("knn_parity_vs_p", "knn_K_vs_p"):
        assert (sweep / f"{name}.png").is_file()
        assert (sweep / f"{name}.json").is_file()
    parity = json.loads((sweep / "knn_parity_vs_p.json").read_text())
    panels = {(s["N"], s["final"]): s for s in parity["extra"]["panels"]}
    assert panels[(4, False)]["agreement"] == 1.0
    assert panels[(4, False)]["gap"] == 0.0
    table = json.loads((sweep / "knn_runs.json").read_text())
    assert len(table) == 8
    assert {"accuracy_knn_1", "accuracy_knn_sqrt", "accuracy_knn_2sqrt"} <= set(
        table[0]
    )
    k_fig = json.loads((sweep / "knn_K_vs_p.json").read_text())
    assert k_fig["extra"]["K"] == ["1", "sqrt", "2sqrt"]


def test_bad_k_fails_before_reading_the_sweep(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="K must be"):
        _load_script().main([str(tmp_path), "--K", "zero"])


def test_several_fields_are_refused(sweep: Path) -> None:
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(L, 3)), rng.integers(0, 2, size=L)
    _write_run(sweep, 4, 0.0, 0, False, X, Y, h=10.0)
    with pytest.raises(ValueError, match="several fields"):
        _load_script().load_knn_sweep(sweep)
