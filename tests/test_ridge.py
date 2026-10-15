"""Régression ridge analytique ``RidgeReadout`` (issue #53)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge

from QELM import load_config
from QELM.training import RidgeReadout


def _data(n: int = 40, d: int = 5, k: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n, d))
    shape = (d,) if k is None else (d, k)
    y = phi @ rng.normal(size=shape) + 0.1 * rng.normal(size=(n,) + shape[1:])
    return phi, y


# --- alpha : argument, puis config, puis 1.0 ---


def test_alpha_defaults_to_one() -> None:
    assert RidgeReadout().config_alpha == 1.0
    assert RidgeReadout({}).config_alpha == 1.0


def test_alpha_from_params_and_config() -> None:
    assert RidgeReadout({"alpha": 0.3}).config_alpha == 0.3
    assert RidgeReadout.from_config(load_config()).config_alpha == 1.0
    cfg = load_config(["training.alpha=0.25"])
    assert RidgeReadout.from_config(cfg).config_alpha == 0.25


def test_fit_alpha_overrides_config() -> None:
    phi, y = _data()
    a = RidgeReadout({"alpha": 5.0})
    a.fit(phi, y, alpha=0.5)
    b = RidgeReadout({"alpha": 0.5})
    b.fit(phi, y)
    assert a.W is not None and b.W is not None
    np.testing.assert_allclose(a.W, b.W)
    assert a.config_alpha == 5.0


@pytest.mark.parametrize("alpha", [-1.0, "x", True, float("nan")])
def test_invalid_alpha_is_rejected(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        RidgeReadout({"alpha": alpha})
    phi, y = _data()
    with pytest.raises(ValueError, match="alpha"):
        RidgeReadout().fit(phi, y, alpha=alpha)  # type: ignore[arg-type]


# --- Parité avec scikit-learn ---


@pytest.mark.parametrize("alpha", [1e-3, 0.5, 1.0, 10.0])
@pytest.mark.parametrize("k", [None, 3])
def test_matches_sklearn(alpha: float, k: int | None) -> None:
    phi, y = _data(k=k)
    model = RidgeReadout({"alpha": alpha})
    assert model.fit(phi, y) == "fitted"
    ref = Ridge(alpha=alpha, fit_intercept=False).fit(phi, y)
    assert model.W is not None
    np.testing.assert_allclose(model.W, ref.coef_.T, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(model.predict(phi), ref.predict(phi), atol=1e-10)


def test_closed_form() -> None:
    """W = (Phi^T Phi + alpha I)^-1 Phi^T y."""
    phi, y = _data(d=3)
    model = RidgeReadout({"alpha": 2.0})
    model.fit(phi, y)
    expected = np.linalg.inv(phi.T @ phi + 2.0 * np.eye(3)) @ phi.T @ y
    np.testing.assert_allclose(model.W, expected, rtol=1e-10)


# --- Garde-fous ---


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fitted before"):
        RidgeReadout().predict(np.ones((2, 3)))


def test_shape_mismatch_raises() -> None:
    model = RidgeReadout()
    with pytest.raises(ValueError, match="rows"):
        model.fit(np.ones((4, 2)), np.ones(3))
    with pytest.raises(ValueError, match="n_features"):
        model.fit(np.ones(4), np.ones(4))


def test_no_dataset_keeps_the_scaffold_status() -> None:
    """Le pipeline appelle fit(None, None) tant qu'il n'y a pas de dataset."""
    model = RidgeReadout()
    assert model.fit(None, None) == "scaffold"
    assert model.W is None
    assert model.state_dict()["weights"].shape == (0,)


def test_state_dict_holds_the_weights() -> None:
    phi, y = _data()
    model = RidgeReadout()
    model.fit(phi, y)
    np.testing.assert_array_equal(model.state_dict()["weights"], model.W)


def test_from_config_without_training_block() -> None:
    cfg = OmegaConf.create({"seed": 0})
    assert RidgeReadout.from_config(cfg).config_alpha == 1.0
