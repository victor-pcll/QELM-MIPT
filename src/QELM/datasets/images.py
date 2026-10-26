"""MNIST and the 8x8 digits, reduced to a task between a few chosen digits.

MNIST (LeCun, Cortes and Burges) is not committed: :func:`download_mnist`
fetches its four IDX files once into ``data/mnist/`` (gitignored), checks
their SHA-256 against the published files, and :func:`load_mnist` reads them
from there. scikit-learn's ``load_digits`` (1797 images of 8x8 pixels,
bundled with scikit-learn) needs no download: the tests use it, and so can a
machine without network access (``dataset=digits``).
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASETS = ("mnist", "digits")

# File name -> SHA-256 of the gzipped IDX file.
MNIST_FILES: dict[str, str] = {
    "train-images-idx3-ubyte.gz": (
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
    ),
    "train-labels-idx1-ubyte.gz": (
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
    ),
    "t10k-images-idx3-ubyte.gz": (
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
    ),
}

# Mirrors of the original files, tried in order.
MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

# IDX type byte -> NumPy dtype (big-endian).
_IDX_TYPES: dict[int, np.dtype[Any]] = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


@dataclass(frozen=True)
class DigitClasses:
    """Classification task between a few digits, split into train and test.

    Attributes:
        x_train: Pixels in ``[0, 1]``, of shape ``(n_train, n_pixels)``.
        y_train: Class index ``k`` of each image, its digit being
            ``classes[k]``, of shape ``(n_train,)``.
        x_test: Pixels of the test images, ``(n_test, n_pixels)``.
        y_test: Class indices of the test images, ``(n_test,)``.
        classes: The digits, in the order of the class indices.
        dataset: ``"mnist"`` or ``"digits"``.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: tuple[int, ...]
    dataset: str

    @property
    def n_classes(self) -> int:
        """Number of classes ``K``."""
        return len(self.classes)


def read_idx(path: str | Path) -> np.ndarray:
    """Read an IDX file (the MNIST format), gzipped or not.

    The header is two zero bytes, a type byte, the number of dimensions,
    then one big-endian 32-bit size per dimension; the data follows.

    Args:
        path: File path. A ``.gz`` suffix is decompressed.

    Returns:
        Array of the stored type and shape.

    Raises:
        ValueError: If the header is not that of an IDX file, or the data do
            not fill the announced shape.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0 or raw[2] not in _IDX_TYPES:
        raise ValueError(f"{path} is not an IDX file.")
    dtype, ndim = _IDX_TYPES[raw[2]], raw[3]
    offset = 4 + 4 * ndim
    shape = tuple(
        int.from_bytes(raw[4 + 4 * k : 8 + 4 * k], "big") for k in range(ndim)
    )
    count = int(np.prod(shape))
    if len(raw) - offset != count * dtype.itemsize:
        raise ValueError(f"{path}: the data do not match the shape {shape}.")
    return np.frombuffer(raw, dtype=dtype, offset=offset).reshape(shape)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_mnist(root: str | Path = "data/mnist") -> Path:
    """Download the four MNIST files into ``root``, if not already there.

    A file already present with the right SHA-256 is kept. Each missing one
    is fetched from the first mirror of :data:`MNIST_MIRRORS` that answers,
    and checked.

    Args:
        root: Destination folder, created if needed.

    Returns:
        ``root`` as a :class:`~pathlib.Path`.

    Raises:
        RuntimeError: If a file cannot be fetched from any mirror, or its
            SHA-256 differs from the published one.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in MNIST_FILES.items():
        path = root / name
        if path.exists() and _sha256(path) == digest:
            continue
        errors = []
        for mirror in MNIST_MIRRORS:
            try:
                with urllib.request.urlopen(mirror + name, timeout=60) as response:
                    path.write_bytes(response.read())
                break
            except OSError as exc:  # URLError is an OSError
                errors.append(f"{mirror}: {exc}")
        else:
            raise RuntimeError(
                f"Could not download {name}: " + "; ".join(errors) + ". Copy the "
                f"four MNIST files to {root} by hand, or use dataset=digits."
            )
        if _sha256(path) != digest:
            path.unlink()
            raise RuntimeError(f"{name} does not have the expected SHA-256.")
    return root


def load_mnist(
    root: str | Path = "data/mnist", download: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return MNIST as flat pixels in ``[0, 1]`` and integer labels.

    Args:
        root: Folder of the four IDX files.
        download: Fetch the missing files first (:func:`download_mnist`).

    Returns:
        ``(x_train, y_train, x_test, y_test)``, of shapes ``(60000, 784)``,
        ``(60000,)``, ``(10000, 784)`` and ``(10000,)``.

    Raises:
        FileNotFoundError: If a file is missing and ``download`` is false.
    """
    root = Path(root)
    if download:
        download_mnist(root)
    names = list(MNIST_FILES)
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing MNIST files in {root}: {missing}.")
    x_train, y_train, x_test, y_test = (read_idx(root / name) for name in names)
    return (
        x_train.reshape(len(x_train), -1) / 255.0,
        y_train.astype(np.int64),
        x_test.reshape(len(x_test), -1) / 255.0,
        y_test.astype(np.int64),
    )


def _load_sklearn_digits(
    test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits  # type: ignore[import-untyped]

    bunch = load_digits()
    x = np.asarray(bunch.data, dtype=np.float64) / 16.0
    y = np.asarray(bunch.target, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    test, train = order[:n_test], order[n_test:]
    return x[train], y[train], x[test], y[test]


def load_digit_classes(
    dataset: str = "mnist",
    classes: Sequence[int] = (3, 5),
    root: str | Path = "data/mnist",
    download: bool = True,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> DigitClasses:
    """Return the task "which of ``classes`` is this digit?".

    Image ``n`` of digit ``classes[k]`` gets the label ``k``; the readout
    learns the one-hot target of ``k`` (``QELM.evaluation.double_descent``).
    Two classes give the binary task of issue #181 (3 vs 5); more make it
    harder.

    Args:
        dataset: ``"mnist"`` (its own train and test split) or ``"digits"``
            (scikit-learn's 8x8 digits, split at random).
        classes: Two or more distinct digits, from 0 to 9.
        root: MNIST folder, for ``dataset="mnist"``.
        download: Fetch MNIST if missing.
        test_fraction: Fraction of the 8x8 digits kept for the test set.
        seed: Seed of the train and test split of the 8x8 digits.

    Returns:
        The :class:`DigitClasses`.

    Raises:
        ValueError: If ``dataset`` is unknown, or ``classes`` holds fewer
            than two digits, a repeated digit or one outside 0 to 9.
    """
    digits = tuple(int(d) for d in classes)
    if len(digits) < 2 or len(set(digits)) != len(digits):
        raise ValueError(
            f"classes must hold two or more distinct digits, got {classes!r}."
        )
    if not all(0 <= d <= 9 for d in digits):
        raise ValueError(f"classes must be digits from 0 to 9, got {classes!r}.")
    if dataset == "mnist":
        x_tr, y_tr, x_te, y_te = load_mnist(root, download=download)
    elif dataset == "digits":
        x_tr, y_tr, x_te, y_te = _load_sklearn_digits(test_fraction, seed)
    else:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}.")
    index = np.full(10, -1, dtype=np.int64)
    index[list(digits)] = np.arange(len(digits))

    def pick(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = index[y] >= 0
        return x[keep], index[y[keep]]

    x_train, y_train = pick(x_tr, y_tr)
    x_test, y_test = pick(x_te, y_te)
    return DigitClasses(x_train, y_train, x_test, y_test, digits, dataset)
