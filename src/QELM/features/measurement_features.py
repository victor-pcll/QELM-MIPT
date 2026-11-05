"""Non-linear feature maps of the measurement record ``M`` (issue #269).

A monitored circuit of ``T`` rounds on ``N`` qubits gives, per trajectory,
the record ``M`` of shape ``(T, N)``: ``m_{t,i} = +-1`` for the Z outcome of
qubit ``i`` at round ``t`` and ``0`` when that qubit was not measured. A
ridge on ``vec(M)`` reads only the additive combinations
``sum_a w_a m_a``. The maps below give the ridge explicit non-linear
functions of ``M``, so that it can read what a linear readout misses.

With ``m in {-1, 0, +1}``, ``m**2 = |m|``: every function of one entry is
``a + b m + c |m|``, and every polynomial of degree 2 in the entries is a
combination of ``m_a``, ``|m_a|`` and ``m_a m_b`` (``a < b``). The modes,
for :func:`extract_M_features`:

- ``raw``: ``vec(M)``, the linear QELM-M;
- ``abs``: ``|M|``, which qubits were measured and when (the geometry of the
  measurement pattern, without the signs);
- ``raw_abs``: ``[vec(M), |M|]``, every function of a single entry;
- ``xor_local``: ``[vec(M), m_{t,i} m_{t,i+1}, m_{t,i} m_{t+1,i}]``, the
  record and the products of nearest neighbours in space and in time (the
  XOR of two outcomes, 0 when either is unmeasured);
- ``xor_cone``: ``[vec(M), |M|]`` and the products ``m_a m_b`` of every pair
  of entries in the forward light cone of the reference site;
- ``xor_full``: ``[vec(M), |M|]`` and the products of every pair of
  entries, the full quadratic polynomial of ``M``.

:func:`feature_label_correlation` gives the correlation of each feature with
the label, the order-by-order spectrum of where the information sits.
"""

from __future__ import annotations

import numpy as np

MODES = ("raw", "abs", "raw_abs", "xor_local", "xor_cone", "xor_full")


def _check(M: np.ndarray) -> np.ndarray:
    """Return ``M`` as a float array of shape ``(L, T, N)``.

    Raises:
        ValueError: If ``M`` is not three-dimensional or holds a value other
            than -1, 0, +1.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 3:
        raise ValueError(f"M must have shape (L, T, N), got {M.shape}.")
    if not np.all(np.isin(M, (-1.0, 0.0, 1.0))):
        raise ValueError("M must hold only -1, 0 and +1.")
    return M


def spatial_pairs(M: np.ndarray) -> np.ndarray:
    """Products of spatial neighbours, ``m_{t,i} m_{t,i+1}``.

    Args:
        M: Records of shape ``(L, T, N)``.

    Returns:
        Array of shape ``(L, T, N - 1)`` (open chain).
    """
    M = _check(M)
    return M[:, :, :-1] * M[:, :, 1:]


def temporal_pairs(M: np.ndarray) -> np.ndarray:
    """Products of temporal neighbours, ``m_{t,i} m_{t+1,i}``.

    Args:
        M: Records of shape ``(L, T, N)``.

    Returns:
        Array of shape ``(L, T - 1, N)``.
    """
    M = _check(M)
    return M[:, :-1, :] * M[:, 1:, :]


def light_cone_mask(T: int, N: int, site: int) -> np.ndarray:
    """Entries of the record in the forward light cone of ``site``.

    One round is a full brickwork period (even bonds, then odd bonds), which
    moves information by at most two sites, so the outcome ``m_{t,i}`` of
    round ``t`` (from 0) can depend on the initial state of ``site`` only if
    ``|i - site| <= 2 (t + 1)``.

    Args:
        T: Rounds of the record.
        N: System qubits.
        site: Reference site, from 0 to ``N - 1``.

    Returns:
        Boolean mask of shape ``(T, N)``.

    Raises:
        ValueError: If ``site`` is not a system site or a count is < 1.
    """
    if T < 1 or N < 1:
        raise ValueError(f"T and N must be >= 1, got T={T}, N={N}.")
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    t = np.arange(T)[:, None]
    i = np.arange(N)[None, :]
    return np.abs(i - site) <= 2 * (t + 1)


def all_pairs(M: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Products ``m_a m_b`` of every pair ``a < b`` of entries of the record.

    Args:
        M: Records of shape ``(L, T, N)``.
        mask: Optional boolean ``(T, N)`` mask: only pairs of entries where
            it is true are kept (for example :func:`light_cone_mask`).

    Returns:
        Array of shape ``(L, K (K - 1) / 2)``, with ``K`` the number of kept
        entries, in the row-major order of the upper triangle.
    """
    M = _check(M)
    flat = M.reshape(M.shape[0], -1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != M.shape[1:]:
            raise ValueError(f"mask must have shape {M.shape[1:]}, got {mask.shape}.")
        flat = flat[:, mask.ravel()]
    a, b = np.triu_indices(flat.shape[1], k=1)
    return flat[:, a] * flat[:, b]


def extract_M_features(M: np.ndarray, mode: str, site: int | None = None) -> np.ndarray:
    """Feature vectors ``Phi(M)`` of the records for one mode of :data:`MODES`.

    Args:
        M: Records of shape ``(L, T, N)``, entries -1, 0, +1.
        mode: One of :data:`MODES`.
        site: Reference site, needed by ``xor_cone`` only.

    Returns:
        Array of shape ``(L, n_features)``.

    Raises:
        ValueError: If the mode is unknown or ``xor_cone`` has no site.
    """
    M = _check(M)
    L = M.shape[0]
    raw = M.reshape(L, -1)
    if mode == "raw":
        return raw
    if mode == "abs":
        return np.abs(raw)
    if mode == "raw_abs":
        return np.hstack([raw, np.abs(raw)])
    if mode == "xor_local":
        return np.hstack(
            [raw, spatial_pairs(M).reshape(L, -1), temporal_pairs(M).reshape(L, -1)]
        )
    if mode == "xor_cone":
        if site is None:
            raise ValueError("mode 'xor_cone' needs the reference site.")
        mask = light_cone_mask(M.shape[1], M.shape[2], site)
        return np.hstack([raw, np.abs(raw), all_pairs(M, mask)])
    if mode == "xor_full":
        return np.hstack([raw, np.abs(raw), all_pairs(M)])
    raise ValueError(f"Unknown mode {mode!r}; known: {', '.join(MODES)}.")


def feature_label_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of ``X`` with the label ``Y``.

    A constant column (an entry never measured, or always measured) has
    correlation 0.

    Args:
        X: Features of shape ``(L, d)``.
        Y: Labels of shape ``(L,)``.

    Returns:
        Array of shape ``(d,)`` in ``[-1, 1]``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean()
    sx = np.sqrt(np.mean(Xc**2, axis=0))
    sy = float(np.sqrt(np.mean(Yc**2)))
    cov = Xc.T @ Yc / X.shape[0]
    out = np.zeros(X.shape[1])
    ok = (sx > 1e-12) & (sy > 1e-12)
    out[ok] = cov[ok] / (sx[ok] * sy)
    return out
