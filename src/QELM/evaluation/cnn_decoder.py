"""A convolutional decoder of the record ``M`` (issue #293).

Dehghani, Lavasani, Hafezi and Gullans, Nat. Commun. 14, 2918 (2023), decode
the reference qubit ``R`` from the record with a convolutional neural
network (CNN): the record ``M`` of one trajectory is an image of shape
``(T, N)``, time by space, and the sign ``<P_R>`` of a purified ``R`` is a
parity of outcomes that sit in the past light cone of the reference. A
convolution reads local patterns of that image with the same weights
everywhere, which is the bias a light-cone-local function asks for.

This module trains such a CNN on the trajectories of
:mod:`QELM.evaluation.dehghani_decoder` and scores it as
:func:`~QELM.evaluation.dehghani_decoder.decode_accuracy` scores the ridge
and the MLP, on the same train / test split.

Architecture (:class:`CNNConfig`):

- input: two channels, the record ``M`` (``+-1`` or 0) and the mask of
  measured sites ``M**2``;
- one block per entry of ``channels``: a 3x3 convolution, periodic in space
  (the brickwork is periodic) and zero-padded in time, a ReLU, and, if
  ``pool``, a 2x2 max pooling (a size of 1 is not pooled). The default has
  three blocks of 32 channels and no pooling: on ``N = 8`` pilots, pooling
  cost 2 to 3 points of accuracy (the parity needs exact positions, which a
  max pooling blurs);
- a dense ReLU layer of ``hidden`` units, then three logits, one per
  purification axis ``X, Y, Z``.

One network serves the three axes: the trunk is shared and each axis has its
own output. A trajectory purified along ``P`` only trains the logit of
``P``, with the logistic loss on its sign; mixed trajectories do not train
the network. This is the per-axis decoder of #289 with a shared trunk, so
the three axes share the data that teaches the trunk the light-cone
structure. Training is AdamW on mini-batches, with early stopping on a
validation tenth of the training set; the parameters with the lowest
validation loss are kept.

The network is written in JAX (CPU), with a hand-written AdamW, so that no
other machine-learning framework is needed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from QELM.evaluation.dehghani_decoder import Records

Params = list[tuple[jax.Array, jax.Array]]


@dataclass(frozen=True)
class CNNConfig:
    """Architecture and training of the CNN.

    Attributes:
        channels: Output channels of each convolution block.
        hidden: Width of the dense ReLU layer.
        lr: Learning rate of AdamW.
        weight_decay: Decoupled weight decay of AdamW.
        batch: Mini-batch size.
        max_epochs: Maximum number of passes over the training set.
        patience: Epochs without a better validation loss before stopping.
        val_fraction: Share of the training set kept for early stopping.
        pool: Whether each block ends with a 2x2 max pooling.
    """

    channels: tuple[int, ...] = (32, 32, 32)
    hidden: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256
    max_epochs: int = 40
    patience: int = 6
    val_fraction: float = 0.1
    pool: bool = False


@dataclass
class TrainLog:
    """What happened during training.

    Attributes:
        epochs: Epochs run.
        best_epoch: Epoch of the kept parameters (1-based, 0 if none).
        val_loss: Validation loss after each epoch.
        val_acc: Validation accuracy after each epoch.
        seconds: Wall time of the training.
        n_params: Number of trainable parameters.
    """

    epochs: int = 0
    best_epoch: int = 0
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    seconds: float = 0.0
    n_params: int = 0


def _pooled(size: int) -> int:
    """Size after a 2x2 max pooling (sizes of 1 stay)."""
    return size // 2 if size >= 2 else size


def init_params(T: int, N: int, config: CNNConfig, key: jax.Array) -> Params:
    """He-initialised weights and zero biases of the CNN for ``(T, N)`` inputs."""
    params: Params = []
    c_in, t, n = 2, T, N
    for c_out in config.channels:
        key, sub = jax.random.split(key)
        std = math.sqrt(2.0 / (9 * c_in))
        params.append(
            (std * jax.random.normal(sub, (3, 3, c_in, c_out)), jnp.zeros(c_out))
        )
        c_in = c_out
        if config.pool:
            t, n = _pooled(t), _pooled(n)
    flat = c_in * t * n
    for d_in, d_out in ((flat, config.hidden), (config.hidden, 3)):
        key, sub = jax.random.split(key)
        std = math.sqrt(2.0 / d_in)
        params.append((std * jax.random.normal(sub, (d_in, d_out)), jnp.zeros(d_out)))
    return params


def _pool(x: jax.Array) -> jax.Array:
    """2x2 max pooling of an ``(n, t, s, c)`` array; a size 1 is kept."""
    wt = 2 if x.shape[1] >= 2 else 1
    ws = 2 if x.shape[2] >= 2 else 1
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, wt, ws, 1), (1, wt, ws, 1), "VALID"
    )


def forward(params: Params, x: jax.Array, pool: bool = False) -> jax.Array:
    """Logits ``(n, 3)`` of the records ``x``, shape ``(n, T, N)``."""
    h = jnp.stack([x, x * x], axis=-1)  # (n, T, N, 2)
    n_conv = len(params) - 2
    for w, b in params[:n_conv]:
        # Periodic in space, zeros in time, then a "valid" 3x3 convolution.
        h = jnp.concatenate([h[:, :, -1:], h, h[:, :, :1]], axis=2)
        h = jnp.pad(h, ((0, 0), (1, 1), (0, 0), (0, 0)))
        h = jax.lax.conv_general_dilated(
            h, w, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        h = jax.nn.relu(h + b)
        if pool:
            h = _pool(h)
    h = h.reshape(h.shape[0], -1)
    (w1, b1), (w2, b2) = params[n_conv:]
    return jax.nn.relu(h @ w1 + b1) @ w2 + b2


def _loss(
    params: Params,
    x: jax.Array,
    axis: jax.Array,
    y: jax.Array,
    w: jax.Array,
    pool: bool,
) -> jax.Array:
    """Weighted logistic loss of the logit of each trajectory's own axis.

    ``axis`` is 0, 1, 2 for X, Y, Z, ``y`` is 1 for ``<P_R> = +1`` and 0
    for ``-1``, ``w`` is 1 for a real sample and 0 for padding.
    """
    z = jnp.take_along_axis(forward(params, x, pool), axis[:, None], axis=1)[:, 0]
    per = jnp.logaddexp(0.0, z) - y * z  # -log sigmoid(+-z)
    return jnp.sum(w * per) / jnp.maximum(jnp.sum(w), 1.0)


@partial(jax.jit, static_argnums=4)
def _eval_batch(
    params: Params, x: jax.Array, axis: jax.Array, y: jax.Array, pool: bool
) -> tuple[jax.Array, jax.Array]:
    """Summed loss and number of correct signs on a batch."""
    z = jnp.take_along_axis(forward(params, x, pool), axis[:, None], axis=1)[:, 0]
    loss = jnp.sum(jnp.logaddexp(0.0, z) - y * z)
    return loss, jnp.sum((z > 0) == (y > 0.5))


def _make_step(config: CNNConfig):  # type: ignore[no-untyped-def]
    """The jitted AdamW update for ``config``."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    grad = jax.value_and_grad(_loss)

    @jax.jit
    def step(params, m, v, t, x, axis, y, w):  # type: ignore[no-untyped-def]
        loss, g = grad(params, x, axis, y, w, config.pool)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_**2, v, g)
        lr = config.lr * jnp.sqrt(1 - b2**t) / (1 - b1**t)
        params = jax.tree_util.tree_map(
            lambda p_, m_, v_: (
                p_
                - lr * m_ / (jnp.sqrt(v_) + eps)
                - config.lr * config.weight_decay * p_
            ),
            params,
            m,
            v,
        )
        return params, m, v, loss

    return step


def _evaluate(
    params: Params,
    x: np.ndarray,
    axis: np.ndarray,
    y: np.ndarray,
    batch: int,
    pool: bool,
) -> tuple[float, float]:
    """Mean loss and accuracy on a whole set, in batches."""
    loss, hits = 0.0, 0
    for i in range(0, x.shape[0], batch):
        lo, hi = _eval_batch(
            params,
            jnp.asarray(x[i : i + batch], jnp.float32),
            jnp.asarray(axis[i : i + batch]),
            jnp.asarray(y[i : i + batch], jnp.float32),
            pool,
        )
        loss += float(lo)
        hits += int(hi)
    n = max(x.shape[0], 1)
    return loss / n, hits / n


def train_cnn(
    M: np.ndarray,
    axis: np.ndarray,
    sign: np.ndarray,
    config: CNNConfig | None = None,
    seed: int = 0,
) -> tuple[Params, TrainLog]:
    """Train the CNN on purified trajectories.

    Args:
        M: Records, shape ``(n, T, N)``.
        axis: Purification axis, 1, 2, 3 for X, Y, Z (0, mixed, is dropped).
        sign: ``<P_R>`` along that axis, ``+-1``.
        config: Architecture and training; :class:`CNNConfig` by default.
        seed: Seed of the initialisation, the validation split and the
            batches.

    Returns:
        The parameters with the lowest validation loss and the
        :class:`TrainLog`.
    """
    config = config or CNNConfig()
    start = time.perf_counter()
    keep = axis > 0
    M, axis, sign = M[keep], axis[keep].astype(np.int32) - 1, sign[keep]
    y = (sign > 0).astype(np.float32)
    n, T, N = M.shape
    rng = np.random.default_rng(seed)
    params = init_params(T, N, config, jax.random.PRNGKey(seed))
    log = TrainLog(
        n_params=int(sum(w.size + b.size for w, b in params)),
    )
    if n == 0:
        log.seconds = time.perf_counter() - start
        return params, log
    perm = rng.permutation(n)
    n_val = int(round(config.val_fraction * n)) if n >= 20 else 0
    val, tr = perm[:n_val], perm[n_val:]
    batch = min(config.batch, tr.size)
    step = _make_step(config)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    best, best_loss, bad, t = params, np.inf, 0, 0
    for epoch in range(1, config.max_epochs + 1):
        order = tr[rng.permutation(tr.size)]
        n_batches = math.ceil(order.size / batch)
        for k in range(n_batches):
            idx = order[k * batch : (k + 1) * batch]
            w = np.ones(batch, np.float32)
            if idx.size < batch:  # pad the last batch: one compiled shape
                w[idx.size :] = 0.0
                idx = np.concatenate([idx, order[: batch - idx.size]])
            t += 1
            params, m, v, _ = step(
                params,
                m,
                v,
                t,
                jnp.asarray(M[idx], jnp.float32),
                jnp.asarray(axis[idx]),
                jnp.asarray(y[idx]),
                jnp.asarray(w),
            )
        log.epochs = epoch
        if n_val == 0:
            best, log.best_epoch = params, epoch
            continue
        vl, va = _evaluate(params, M[val], axis[val], y[val], 2048, config.pool)
        log.val_loss.append(vl)
        log.val_acc.append(va)
        if vl < best_loss - 1e-4:
            best, best_loss, bad, log.best_epoch = params, vl, 0, epoch
        else:
            bad += 1
            if bad >= config.patience:
                break
    log.seconds = time.perf_counter() - start
    return best, log


def predict_sign(
    params: Params, M: np.ndarray, axis: np.ndarray, pool: bool = False
) -> np.ndarray:
    """Predicted ``<P_R>`` (``+-1``) along each trajectory's axis (1, 2, 3)."""
    out = np.empty(M.shape[0], dtype=np.int8)
    a = np.clip(axis.astype(np.int32) - 1, 0, 2)
    for i in range(0, M.shape[0], 2048):
        logits = forward(params, jnp.asarray(M[i : i + 2048], jnp.float32), pool)
        z = np.take_along_axis(np.asarray(logits), a[i : i + 2048, None], axis=1)
        out[i : i + 2048] = np.where(z[:, 0] > 0, 1, -1)
    return out


def split(n: int, train_fraction: float, seed: int) -> np.ndarray:
    """Training mask, the split of
    :func:`~QELM.evaluation.dehghani_decoder.decode_accuracy`."""
    rng = np.random.default_rng(seed)
    train = np.zeros(n, dtype=bool)
    train[rng.permutation(n)[: int(round(train_fraction * n))]] = True
    return train


def decode_cnn_accuracy(
    records: Records,
    train_fraction: float = 0.7,
    seed: int = 0,
    config: CNNConfig | None = None,
    max_train: int | None = None,
) -> tuple[dict[str, float], TrainLog]:
    """Accuracy of the CNN on the purified test trajectories.

    The split is that of
    :func:`~QELM.evaluation.dehghani_decoder.decode_accuracy` with the same
    ``train_fraction`` and ``seed``, so the CNN, the ridge and the MLP are
    scored on the same test trajectories.

    Args:
        records: Output of
            :func:`~QELM.evaluation.dehghani_decoder.simulate_records`.
        train_fraction: Share of the trajectories used for training.
        seed: Seed of the split and of the training.
        config: Architecture and training.
        max_train: If set, train on the first ``max_train`` training
            trajectories only (purified or not), with the test set unchanged:
            the accuracy against training-set size.

    Returns:
        ``{"acc_pure", "acc_all", "n_test_pure", "n_train_pure"}``, as in
        ``decode_accuracy`` (``acc_all = f acc_pure + (1 - f) / 2``), and
        the :class:`TrainLog`.
    """
    n = records.axis.size
    train = split(n, train_fraction, seed)
    test = ~train
    tr_idx = np.flatnonzero(train)
    if max_train is not None:
        tr_idx = tr_idx[:max_train]
    params, log = train_cnn(
        records.M[tr_idx], records.axis[tr_idx], records.sign[tr_idx], config, seed
    )
    te = test & records.purified
    n_pure = int(te.sum())
    f = float(records.purified[test].mean()) if test.any() else float("nan")
    if n_pure:
        pool = (config or CNNConfig()).pool
        pred = predict_sign(params, records.M[te], records.axis[te], pool)
        acc = float(np.mean(pred == records.sign[te]))
    else:
        acc = float("nan")
    return {
        "acc_pure": acc,
        "acc_all": f * acc + (1 - f) / 2 if n_pure else 0.5,
        "n_test_pure": float(n_pure),
        "n_train_pure": float(np.sum(records.axis[tr_idx] > 0)),
    }, log
