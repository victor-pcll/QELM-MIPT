"""Static classification datasets for the QELM (issue #181).

- :mod:`QELM.datasets.images`: MNIST (downloaded once to ``data/mnist/``)
  and scikit-learn's 8x8 digits (bundled, no download), reduced to a task
  between a few chosen digits (:func:`load_digit_classes`);
- :mod:`QELM.datasets.preprocess`: PCA to one component per qubit and the
  mapping of each component to a rotation angle (:class:`AngleEncoder`).
"""

from __future__ import annotations

from QELM.datasets.images import (
    DATASETS,
    MNIST_FILES,
    DigitClasses,
    download_mnist,
    load_digit_classes,
    load_mnist,
    read_idx,
)
from QELM.datasets.preprocess import AngleEncoder

__all__ = [
    "DATASETS",
    "MNIST_FILES",
    "AngleEncoder",
    "DigitClasses",
    "download_mnist",
    "load_digit_classes",
    "load_mnist",
    "read_idx",
]
