"""Image datasets and their preprocessing, ``QELM.datasets`` (issue #181)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from QELM.datasets import AngleEncoder, images, load_digit_pair, read_idx


def _write_idx(path: Path, array: np.ndarray, compress: bool) -> None:
    header = bytes([0, 0, 0x08, array.ndim])
    header += b"".join(int(n).to_bytes(4, "big") for n in array.shape)
    raw = header + array.astype(np.uint8).tobytes()
    if compress:
        with gzip.open(path, "wb") as f:
            f.write(raw)
    else:
        path.write_bytes(raw)


@pytest.mark.parametrize("compress", [True, False])
def test_read_idx_round_trip(tmp_path: Path, compress: bool) -> None:
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4) % 256
    path = tmp_path / ("a.gz" if compress else "a")
    _write_idx(path, array, compress)
    np.testing.assert_array_equal(read_idx(path), array)


def test_read_idx_rejects_other_files(tmp_path: Path) -> None:
    (tmp_path / "bad").write_bytes(b"\x01\x02\x03\x04")
    with pytest.raises(ValueError, match="not an IDX"):
        read_idx(tmp_path / "bad")
    _write_idx(tmp_path / "short", np.zeros((2, 2)), compress=False)
    (tmp_path / "short").write_bytes((tmp_path / "short").read_bytes()[:-1])
    with pytest.raises(ValueError, match="shape"):
        read_idx(tmp_path / "short")


def test_load_mnist_reads_the_four_files_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    names = list(images.MNIST_FILES)
    _write_idx(tmp_path / names[0], rng.integers(0, 256, (6, 28, 28)), True)
    _write_idx(tmp_path / names[1], np.array([3, 5, 1, 3, 5, 5]), True)
    _write_idx(tmp_path / names[2], rng.integers(0, 256, (3, 28, 28)), True)
    _write_idx(tmp_path / names[3], np.array([5, 3, 0]), True)
    pair = load_digit_pair("mnist", (3, 5), root=tmp_path, download=False)
    assert pair.x_train.shape == (5, 784)
    np.testing.assert_array_equal(pair.y_train, [1, -1, 1, -1, -1])
    np.testing.assert_array_equal(pair.y_test, [-1, 1])
    assert 0 <= pair.x_train.min() and pair.x_train.max() <= 1
    # A file with the wrong SHA-256 is fetched again from a mirror.
    calls: list[str] = []

    def fail(url: str, timeout: float) -> None:
        calls.append(url)
        raise OSError("offline")

    monkeypatch.setattr(images.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Could not download"):
        images.download_mnist(tmp_path)
    assert calls[0].endswith(names[0])
    assert len(calls) == len(images.MNIST_MIRRORS)


def test_load_mnist_without_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing MNIST"):
        images.load_mnist(tmp_path, download=False)


def test_sklearn_digits_pair() -> None:
    pair = load_digit_pair("digits", (3, 5), seed=0)
    assert pair.dataset == "digits" and pair.digits == (3, 5)
    assert pair.x_train.shape[1] == 64
    n = len(pair.y_train) + len(pair.y_test)
    assert n == 183 + 182  # 3s and 5s of scikit-learn's digits
    assert set(np.unique(pair.y_train)) == {-1.0, 1.0}
    assert 0 <= pair.x_train.min() and pair.x_train.max() <= 1
    again = load_digit_pair("digits", (3, 5), seed=0)
    np.testing.assert_array_equal(pair.x_test, again.x_test)


def test_pair_errors() -> None:
    with pytest.raises(ValueError, match="differ"):
        load_digit_pair("digits", (3, 3))
    with pytest.raises(ValueError, match="dataset"):
        load_digit_pair("cifar")


def test_angle_encoder_maps_the_training_range_to_0_pi() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 20))
    encoder = AngleEncoder(4).fit(x)
    theta = encoder.transform(x)
    assert theta.shape == (200, 4)
    np.testing.assert_allclose(theta.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(theta.max(axis=0), np.pi, atol=1e-12)
    # Out-of-range test points are clipped.
    far = encoder.transform(100 * rng.normal(size=(10, 20)))
    assert far.min() >= 0 and far.max() <= np.pi
    # The components are the PCA's, ordered by explained variance.
    assert encoder.pca is not None
    variance = np.var(encoder.components(x), axis=0)
    assert np.all(np.diff(variance) <= 1e-12)


def test_angle_encoder_errors() -> None:
    with pytest.raises(ValueError, match="n_components"):
        AngleEncoder(0)
    with pytest.raises(ValueError, match="angle_range"):
        AngleEncoder(2, angle_range=0)
    with pytest.raises(RuntimeError, match="fit"):
        AngleEncoder(2).transform(np.zeros((2, 3)))
