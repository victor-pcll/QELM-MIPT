"""Decoding the purified reference qubit from the record (issue #287).

The protocol of Dehghani, Lavasani, Hafezi and Gullans, "Neural-network
decoders for measurement induced phase transitions", Nat. Commun. 14, 2918
(2023), on the periodic Clifford brickwork of
:mod:`QELM.models.circuits.stabilizer`:

1. the reference qubit ``R`` (qubit 0) starts in a Bell pair with the middle
   system site, the other system qubits in ``|0>``;
2. one **fixed** circuit: the gate of every bond and step is drawn once
   (:func:`draw_fixed_circuit`) and shared by every trajectory, while the
   measured sites are drawn anew per trajectory, each site with probability
   ``p`` after the gates of each step (:func:`simulate_records`);
3. the input of a decoder is the record ``M`` of shape ``(T, N)``, ``+1`` or
   ``-1`` for the outcome of a measured site and ``0`` for an unmeasured one;
4. the label is the outcome of ``sigma_P`` on ``R`` at the end, for
   ``P = X, Y, Z``.

The state is a stabilizer state, so ``R`` is either mixed or pure along one
Pauli axis ``P``, with ``<P_R> = +-1``. In the second case the sign is a
parity of the recorded outcomes, ``<P_R> = c * s_j1 * ... * s_jk``: a
deterministic but, for ``k >= 2``, non-linear function of ``M``. In every
other case the outcome of ``sigma_P`` is a fair coin, whatever the decoder.
So the Bayes-optimal decoder is exact: accuracy 1 on the trajectories
purified along ``P``, 1/2 on the others. :func:`parity_weight_one` tests,
by replaying a trajectory with the same measured sites, whether the sign is
fixed by at most one outcome, which is all a linear readout can use.

Each trajectory also ends with a final readout, a single-shot Z measurement
of every system site after the last step (Victor's second case in #287):
:func:`simulate_records` returns the record with and without it. With it,
the system ends in a product state, so ``R`` is always pure and the Bayes
accuracy is 1 at every ``p``: what is left of the transition is how hard
the parity is to learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim  # type: ignore[import-untyped]

from QELM.models.circuits.stabilizer import (
    REFERENCE,
    clifford_gate_programs,
    periodic_bonds,
)

#: Index of the purification axis: 0 for a mixed ``R``, then X, Y, Z.
AXES = ("mixed", "X", "Y", "Z")


def draw_fixed_circuit(N: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Draw the gates of one fixed circuit.

    Args:
        N: Number of system sites, even.
        T: Number of steps (half-layers of the periodic brickwork).
        rng: Random generator.

    Returns:
        Index in :func:`~QELM.models.circuits.clifford.clifford_group_2q` of
        each gate, shape ``(T, N // 2)``, in the order of
        :func:`~QELM.models.circuits.stabilizer.periodic_bonds` at each step.

    Raises:
        ValueError: If ``N`` is odd or ``T < 1``.
    """
    periodic_bonds(N, 0)  # checks N
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}.")
    return rng.integers(len(clifford_gate_programs()), size=(T, N // 2))


def _gate_layers(gates: np.ndarray, N: int) -> list[str]:
    """Return the ``stim`` program of the gates of each step."""
    programs = clifford_gate_programs()
    layers = []
    for t in range(gates.shape[0]):
        bonds = periodic_bonds(N, t)
        layers.append(
            "\n".join(
                programs[g].format(a=a + 1, b=b + 1)
                for g, (a, b) in zip(gates[t], bonds, strict=True)
            )
        )
    return layers


def _bloch_axis(sim: stim.TableauSimulator) -> tuple[int, int]:
    """Return the purification axis of ``R`` (:data:`AXES`) and its sign."""
    bloch = sim.peek_bloch(REFERENCE)
    if bloch == stim.PauliString(1):
        return 0, 0
    axis = int(bloch[0])  # 1 = X, 2 = Y, 3 = Z
    sign = 1 if bloch.sign == 1 else -1
    return axis, sign


def _run(
    layers: list[str], measured: np.ndarray, site: int, seed: int
) -> tuple[np.ndarray, int, int, np.ndarray | None, int, int]:
    """Run one trajectory.

    ``measured`` has one row per step, and optionally one more: the final
    readout, a Z measurement of the sites of that row after the last step,
    without gates.

    Returns:
        The outcomes of the steps (``+-1``, in the order of the record), the
        axis and sign of ``R`` after the last step, then the outcomes of the
        final readout and the axis and sign of ``R`` after it (``None``,
        ``0``, ``0`` without a final readout).
    """
    T = len(layers)
    sim = stim.TableauSimulator(seed=seed)
    sim.h(REFERENCE)
    sim.cnot(REFERENCE, site + 1)
    lines = []
    for t, layer in enumerate(layers):
        lines.append(layer)
        sites = np.flatnonzero(measured[t])
        if sites.size:
            lines.append("M " + " ".join(str(k + 1) for k in sites))
    sim.do_circuit(stim.Circuit("\n".join(lines)))
    bits = np.array(sim.current_measurement_record(), dtype=bool)
    axis, sign = _bloch_axis(sim)
    outcomes = 1 - 2 * bits.astype(np.int8)
    if measured.shape[0] == T:
        return outcomes, axis, sign, None, 0, 0
    sites = np.flatnonzero(measured[T])
    final = (
        1 - 2 * np.array(sim.measure_many(*(sites + 1)), dtype=np.int8)
        if sites.size
        else np.zeros(0, dtype=np.int8)
    )
    axis_f, sign_f = _bloch_axis(sim)
    return outcomes, axis, sign, final, axis_f, sign_f


@dataclass(frozen=True)
class Records:
    """Trajectories of one fixed circuit at one ``p``.

    Attributes:
        M: Records, shape ``(n, T, N)``, ``+1`` / ``-1`` for a measured site
            (outcome ``0`` / ``1``) and ``0`` for an unmeasured one. With a
            final readout, its outcomes are an extra row, ``(n, T + 1, N)``.
        axis: Purification axis of ``R`` at the end (:data:`AXES`), shape
            ``(n,)``: 0 if ``R`` is mixed, 1, 2, 3 for X, Y, Z. With a final
            readout, the axis after it.
        sign: ``<P_R>`` along that axis, ``+-1``, and 0 if mixed, shape
            ``(n,)``.

    The label of a decoder along ``P`` is the outcome of ``sigma_P`` on
    ``R``: ``sign`` where ``axis`` is ``P``, a fair coin elsewhere.
    """

    M: np.ndarray
    axis: np.ndarray
    sign: np.ndarray

    @property
    def purified(self) -> np.ndarray:
        """``True`` where ``R`` ends pure."""
        return np.asarray(self.axis > 0)


def simulate_records(
    gates: np.ndarray,
    N: int,
    p: float,
    n: int,
    rng: np.random.Generator,
    site: int | None = None,
) -> tuple[Records, Records]:
    """Sample ``n`` trajectories of a fixed circuit at measurement rate ``p``.

    Each trajectory ends with a final readout: every system site is measured
    in Z once more after the last step, a single shot. Two views of the same
    trajectories are returned: without the final readout (the record ``M``
    and ``R`` after the last step, the protocol of Dehghani et al.), and
    with it (``M`` and the readout as an extra row, ``R`` after the
    readout). Measuring every site leaves the system in a product state, so
    ``R`` is always pure in the second view.

    Args:
        gates: Output of :func:`draw_fixed_circuit`, shape ``(T, N // 2)``.
        N: Number of system sites.
        p: Probability to measure each site after each step.
        n: Number of trajectories.
        rng: Random generator of the measured sites and the ``stim`` seeds.
        site: System site in the Bell pair with ``R``; ``None`` for
            ``N // 2``.

    Returns:
        ``(without, with_readout)``, two :class:`Records`.

    Raises:
        ValueError: If ``p`` is not a probability or ``site`` is not a site.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    site = N // 2 if site is None else int(site)
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N - 1}], got {site}.")
    T = gates.shape[0]
    layers = _gate_layers(gates, N)
    M = np.zeros((n, T + 1, N), dtype=np.int8)
    axis = np.zeros((2, n), dtype=np.int8)
    sign = np.zeros((2, n), dtype=np.int8)
    for i in range(n):
        measured = np.ones((T + 1, N), dtype=bool)
        measured[:T] = rng.random((T, N)) < p
        out, axis[0, i], sign[0, i], final, axis[1, i], sign[1, i] = _run(
            layers, measured, site, int(rng.integers(2**63))
        )
        assert final is not None
        # Row-major order of the mask = the order of the M lines.
        M[i][measured] = np.concatenate([out, final])
    without = Records(M=M[:, :T], axis=axis[0], sign=sign[0])
    return without, Records(M=M, axis=axis[1], sign=sign[1])


def parity_weight_one(
    gates: np.ndarray,
    N: int,
    measured: np.ndarray,
    rng: np.random.Generator,
    replays: int = 32,
    site: int | None = None,
) -> bool | None:
    """Whether the sign of a purified ``R`` is fixed by at most one outcome.

    The trajectory with these measured sites is replayed ``replays`` times
    with new outcomes. The axis does not depend on the outcomes; the sign is
    ``c * s_j1 * ... * s_jk``. It returns ``True`` if the sign is the same in
    every replay (``k = 0``) or equals ``+-`` one outcome in every replay
    (``k = 1``): then a linear readout of ``M`` can predict it. Otherwise
    (``k >= 2`` for every choice of the parity) it returns ``False``, up to
    a chance coincidence of probability ``2^-replays`` per outcome.

    Args:
        gates: Output of :func:`draw_fixed_circuit`.
        N: Number of system sites.
        measured: Measured sites, boolean, shape ``(T, N)``, or
            ``(T + 1, N)`` with the final readout as the last row (the sign
            is then that of ``R`` after the readout).
        rng: Random generator of the ``stim`` seeds.
        replays: Number of replays.
        site: System site paired with ``R``; ``None`` for ``N // 2``.

    Returns:
        ``None`` if ``R`` ends mixed, else whether the parity has weight
        ``<= 1``.
    """
    site = N // 2 if site is None else int(site)
    layers = _gate_layers(gates, N)
    outs, signs = [], []
    for _ in range(replays):
        o, axis, s, final, axis_f, s_f = _run(
            layers, measured, site, int(rng.integers(2**63))
        )
        if final is not None:
            o, axis, s = np.concatenate([o, final]), axis_f, s_f
        if axis == 0:
            return None
        outs.append(o)
        signs.append(s)
    S = np.array(signs)
    if np.all(S == S[0]):
        return True
    if not outs[0].size:
        return False
    outcomes = np.array(outs)  # (replays, n_meas)
    prod = outcomes * S[:, None]
    return bool(np.any(np.all(prod == prod[:1], axis=0)))


#: Readouts of :func:`decode_accuracy`.
READOUTS = ("ridge", "mlp")


def _fit_predict(
    readout: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, seed: int
) -> np.ndarray:
    """Train ``readout`` on ``(X, y)``, ``y = +-1``; return its test labels."""
    from sklearn.linear_model import RidgeCV  # type: ignore[import-untyped]
    from sklearn.neural_network import (  # type: ignore[import-untyped]
        MLPClassifier,
    )

    classes = np.unique(y)
    if classes.size < 2:  # one label in the training set: predict it
        return np.full(X_test.shape[0], classes[0] if classes.size else 1)
    if readout == "ridge":
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X, y)
        return np.where(ridge.predict(X_test) >= 0, 1, -1)
    if readout == "mlp":
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            max_iter=1000,
            early_stopping=y.size >= 100,
            n_iter_no_change=30,
            random_state=seed,
        ).fit(X, y)
        return np.asarray(mlp.predict(X_test))
    raise ValueError(f"Unknown readout {readout!r}; expected one of {READOUTS}.")


def decode_accuracy(
    records: Records,
    readouts: tuple[str, ...] = READOUTS,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Accuracy of each readout of ``vec(M)`` on the purified trajectories.

    As in Dehghani et al., each axis ``P`` gets its own decoder, trained and
    tested only on the trajectories purified along ``P``: there the label is
    the deterministic sign ``<P_R>``, and the Bayes-optimal accuracy is 1.
    The counts of the axes add up into one accuracy on the purified
    trajectories, ``acc_pure``. On a trajectory where ``R`` is mixed (or
    purified along another axis) every decoder is right with probability
    1/2, so the accuracy along the purification axis over all trajectories
    is ``acc_all = f acc_pure + (1 - f) / 2``, with ``f`` the purified
    fraction; the Bayes ceiling is ``(1 + f) / 2``.

    Args:
        records: Output of :func:`simulate_records`.
        readouts: Readouts to train, among :data:`READOUTS`.
        train_fraction: Share of the trajectories used for training; the
            same split for every axis and readout.
        seed: Seed of the split and of the MLP.

    Returns:
        ``{readout: {"acc_pure", "acc_all", "n_test_pure"}}`` and
        ``"bayes"`` with ``acc_pure = 1`` and ``acc_all = (1 + f) / 2``
        (``acc_pure`` is NaN when no test trajectory is purified).
    """
    n = records.axis.size
    rng = np.random.default_rng(seed)
    train = np.zeros(n, dtype=bool)
    train[rng.permutation(n)[: int(round(train_fraction * n))]] = True
    test = ~train
    X = records.M.reshape(n, -1).astype(float)
    f = float(records.purified[test].mean()) if test.any() else float("nan")
    out: dict[str, dict[str, float]] = {}
    n_pure = int(np.sum(records.purified & test))
    for readout in readouts:
        correct = 0
        for a in (1, 2, 3):
            on_axis = records.axis == a
            tr, te = on_axis & train, on_axis & test
            if not te.any():
                continue
            y_pred = _fit_predict(
                readout, X[tr], records.sign[tr].astype(int), X[te], seed
            )
            correct += int(np.sum(y_pred == records.sign[te]))
        acc = correct / n_pure if n_pure else float("nan")
        out[readout] = {
            "acc_pure": acc,
            "acc_all": f * acc + (1 - f) / 2 if n_pure else 0.5,
            "n_test_pure": float(n_pure),
        }
    out["bayes"] = {
        "acc_pure": 1.0 if n_pure else float("nan"),
        "acc_all": (1 + f) / 2,
        "n_test_pure": float(n_pure),
    }
    return out


def local_paulis(N: int) -> list[stim.PauliString]:
    """The QELM observables ``P_k`` and ``P_i P_j`` (``P = X, Y, Z``).

    Args:
        N: Number of system sites; the strings act on ``N + 1`` qubits, ``R``
            (qubit 0) as the identity.

    Returns:
        ``3 N`` single-site strings (site-major, then X, Y, Z), followed by
        ``9 N (N - 1) / 2`` two-site strings, ``i < j``.
    """
    out = []
    for i in range(N):
        for P in "XYZ":
            s = ["_"] * (N + 1)
            s[i + 1] = P
            out.append(stim.PauliString("".join(s)))
    for i in range(N):
        for j in range(i + 1, N):
            for P in "XYZ":
                for Q in "XYZ":
                    s = ["_"] * (N + 1)
                    s[i + 1], s[j + 1] = P, Q
                    out.append(stim.PauliString("".join(s)))
    return out


@dataclass(frozen=True)
class QELMRecords:
    """Trajectories with the record and the QELM features of the final state.

    Attributes:
        M: Records, shape ``(n, T, N)``, as in :class:`Records`.
        axis: Purification axis of ``R`` after the last step, shape ``(n,)``.
        labels: Outcome of ``sigma_X``, ``sigma_Y``, ``sigma_Z`` on ``R``,
            shape ``(n, 3)``, each measured on its own copy of the final
            state (Born rule).
        features: ``<P_k>`` and ``<P_i P_j>`` (:func:`local_paulis`) of the
            system after that measurement of ``R``, shape ``(n, 3, F)``:
            ``features[:, a]`` goes with ``labels[:, a]``. For a stabilizer
            state each value is ``-1``, ``0`` or ``+1``.
    """

    M: np.ndarray
    axis: np.ndarray
    labels: np.ndarray
    features: np.ndarray


def simulate_qelm_records(
    gates: np.ndarray,
    N: int,
    p: float,
    n: int,
    rng: np.random.Generator,
    site: int | None = None,
) -> QELMRecords:
    """Sample trajectories with the record and the full QELM features.

    The protocol of :func:`simulate_records` without the final readout. At
    the end, for each axis ``P``, a copy of the state measures ``sigma_P``
    on ``R`` (the label) and then reads the exact expectation values of
    :func:`local_paulis` on the system, as the QELM on the state of the
    earlier experiments (#269, #277) does: the features see the system after
    ``R`` is measured, and they are exact (infinitely many shots).

    Args:
        gates: Output of :func:`draw_fixed_circuit`.
        N: Number of system sites.
        p: Probability to measure each site after each step.
        n: Number of trajectories.
        rng: Random generator of the measured sites and the ``stim`` seeds.
        site: System site paired with ``R``; ``None`` for ``N // 2``.

    Returns:
        The :class:`QELMRecords`.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    site = N // 2 if site is None else int(site)
    T = gates.shape[0]
    layers = _gate_layers(gates, N)
    paulis = local_paulis(N)
    M = np.zeros((n, T, N), dtype=np.int8)
    axis = np.zeros(n, dtype=np.int8)
    labels = np.zeros((n, 3), dtype=np.int8)
    features = np.zeros((n, 3, len(paulis)), dtype=np.int8)
    for i in range(n):
        measured = rng.random((T, N)) < p
        sim = stim.TableauSimulator(seed=int(rng.integers(2**63)))
        sim.h(REFERENCE)
        sim.cnot(REFERENCE, site + 1)
        lines = []
        for t, layer in enumerate(layers):
            lines.append(layer)
            sites = np.flatnonzero(measured[t])
            if sites.size:
                lines.append("M " + " ".join(str(k + 1) for k in sites))
        sim.do_circuit(stim.Circuit("\n".join(lines)))
        bits = np.array(sim.current_measurement_record(), dtype=bool)
        M[i][measured] = 1 - 2 * bits.astype(np.int8)
        axis[i] = _bloch_axis(sim)[0]
        for a, gate in enumerate(("MX", "MY", "M")):
            copy = sim.copy()
            copy.do_circuit(stim.Circuit(f"{gate} {REFERENCE}"))
            labels[i, a] = 1 - 2 * int(copy.current_measurement_record()[-1])
            features[i, a] = [
                round(copy.peek_observable_expectation(q)) for q in paulis
            ]
    return QELMRecords(M=M, axis=axis, labels=labels, features=features)


#: Inputs of :func:`decode_qelm_accuracy`.
QELM_INPUTS = ("M", "state", "M+state")


def decode_qelm_accuracy(
    records: QELMRecords,
    readouts: tuple[str, ...] = READOUTS,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Accuracy of each readout of each input on the label ``sigma_P`` of ``R``.

    One decoder per axis ``P``, trained and tested on **all** trajectories
    (the label is then a coin where ``R`` is mixed and only the record is
    read). The inputs are ``vec(M)`` alone, the state features alone and
    both (:data:`QELM_INPUTS`). The accuracy is averaged over the three
    axes, over all test trajectories (``acc``), over those where ``R`` is
    pure after the last step (``acc_pure``) and those where it is mixed
    (``acc_mixed``). ``bayes_M`` is the Bayes ceiling of the record alone,
    the mean over axes of ``(1 + f_P) / 2``.

    Returns:
        ``{"<input>/<readout>": {"acc", "acc_pure", "acc_mixed"}}`` and
        ``{"bayes_M": {"acc"}}``.
    """
    n = records.axis.size
    rng = np.random.default_rng(seed)
    train = np.zeros(n, dtype=bool)
    train[rng.permutation(n)[: int(round(train_fraction * n))]] = True
    test = ~train
    pure = records.axis > 0
    m = records.M.reshape(n, -1).astype(float)
    out: dict[str, dict[str, float]] = {}
    for inp in QELM_INPUTS:
        for readout in readouts:
            hits = np.zeros((3, int(test.sum())), dtype=bool)
            for a in range(3):
                f = records.features[:, a].astype(float)
                X = {"M": m, "state": f, "M+state": np.hstack([m, f])}[inp]
                y = records.labels[:, a].astype(int)
                hits[a] = (
                    _fit_predict(readout, X[train], y[train], X[test], seed) == y[test]
                )
            pt = pure[test]
            out[f"{inp}/{readout}"] = {
                "acc": float(hits.mean()),
                "acc_pure": float(hits[:, pt].mean()) if pt.any() else float("nan"),
                "acc_mixed": float(hits[:, ~pt].mean())
                if (~pt).any()
                else float("nan"),
            }
    f_axis = [float(np.mean(records.axis[test] == a)) for a in (1, 2, 3)]
    out["bayes_M"] = {"acc": float(np.mean([(1 + f) / 2 for f in f_axis]))}
    return out
