"""Static classification datasets for the QELM (issue #181).

- :mod:`QELM.datasets.images`: MNIST (downloaded once to ``data/mnist/``)
  and scikit-learn's 8x8 digits (bundled, no download), reduced to a binary
  task between two digits (:func:`load_digit_pair`);
- :mod:`QELM.datasets.preprocess`: PCA to one component per qubit and the
  mapping of each component to a rotation angle (:class:`AngleEncoder`).
"""

from __future__ import annotations

from QELM.datasets.images import (
    DATASETS,
    MNIST_FILES,
    DigitPair,
    download_mnist,
    load_digit_pair,
    load_mnist,
    read_idx,
)
from QELM.datasets.preprocess import AngleEncoder

__all__ = [
    "DATASETS",
    "MNIST_FILES",
    "AngleEncoder",
    "DigitPair",
    "download_mnist",
    "load_digit_pair",
    "load_mnist",
    "read_idx",
]
