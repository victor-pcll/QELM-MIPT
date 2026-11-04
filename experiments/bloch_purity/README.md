# Tomography and purity of the reference qubit (issue #277)

The task of [`nonlinear_M/`](../nonlinear_M/README.md) (#269): a reference
qubit `R` in a Bell pair with the middle site of the chain, purified (or not)
by the measurements of a Haar U(4) brickwork. Predicting the Z outcome of `R`
is capped at 3/4, because the record purifies `R` along a direction set by
the gates (#275). This experiment asks the readouts for the whole state of
`R` instead: its **Bloch vector** `r = (<X_R>, <Y_R>, <Z_R>)` given the
record `M` (tomography), and its **purity** `Tr(rho_R^2) = (1 + |r|^2) / 2`,
the order parameter of the purification transition. Which readout of the
QELM recovers them, and how does that change across the MIPT?

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/bloch_purity/bloch_purity.py`, launcher `bloch_purity.sh`, figures `plot_bloch_purity.py` |
| Config | `conf/bloch_purity.yaml` |
| Physical model | Haar U(4) brickwork on an open chain of `N` qubits, one fixed circuit per seed, Z measurement of each qubit with probability `p` after each round |
| System sizes `N` | 8 (`N = 10, 12` with the cluster command of `experiments/README.md`) |
| Control parameters | `p` from 0.05 to 1 (18 values), `T = 2N = 16` rounds |
| Initial state | `R` (never acted on) in the Bell pair `(|00> + |11>) / sqrt 2` with site `N // 2`, the other sites in `|0>` |
| ML task | regression of `r(M)` (3 targets, exact or single-shot labels) and of `Tr(rho_R^2)`; ridges on `M`, on the state, on both; test `R2` and mean absolute error |
| Issues / PRs | #277; builds on #269 / PR #278 and #270 / PR #275 |

## Methodology

1. One trajectory: `T` rounds of Haar U(4) gates (the same circuit for every
   trajectory of a seed), each followed by a Z measurement of every system
   qubit with probability `p`. Row `t` of the record `M` holds the outcomes,
   +-1, and 0 for an unmeasured qubit.
2. Exact targets: the reduced state `rho_R` of `R` given `M` and the
   circuit, its Bloch vector `r(M)` and its purity `(1 + |r|^2) / 2`.
3. Single-shot labels (`bloch_task.labels=both`, the default): at the end of
   the trajectory, `R` is measured once along an axis `a` drawn uniformly in
   `X, Y, Z`, giving `s = +-1` with `E[s | M, a] = r_a(M)`. This is the label
   an experiment has: one copy of `R` per trajectory. The purity has no
   single-shot label (it needs two copies), so it is always scored on its
   exact value. The shots use their own generator, so the exact scores do not
   depend on this option.
4. Features, none of which needs to simulate the circuit:
   - **QELM on M**: the raw record `vec(M)` (`N T = 128` features).
     `bloch_task.m_features=raw_abs` uses `[vec(M), vec(|M|)]` instead;
   - **QELM on the state**: `<P_k>` and `<P_i P_j>` (`P` in `X, Y, Z`) of the
     final chain state (108 features), before `R` is measured for the exact
     targets, after the single shot for the shot labels (the outcome is
     imprinted on the chain, as in the Z task of #269);
   - **QELM on M and the state**: both side by side.
5. Readout: a ridge (`RidgeCV`, one alpha per target chosen by leave-one-out
   over `10^-3 .. 10^4`), fitted on 70 % of the 8000 trajectories of one `p`
   and scored on the other 30 %. For the shots, one ridge per axis.
6. References: **Bayes optimal on M**, the exact `r(M)` and purity of an
   observer who knows the circuit. Against the exact targets its `R2` is 1
   (the oracle of #277); against single shots it predicts `r_a(M)` and its
   `R2` is `~ E[r_a^2]`, at most about 1/3 (a pure `R` along a random
   direction). **Chance**: the mean of the training targets, `R2 = 0`.
7. Scores: the `R2` of the vector, `1 - sum |r_pred - r|^2 / sum |r - mean
   r|^2` over the three components (and each component), the `R2` against
   the shots pooled over the three axes, the `R2` of the purity, and the mean
   absolute error of the purity, which stays defined when `R` is pure on every
   trajectory.
8. Sweep: 10 seeds (42 to 51), one circuit each; the figures show the mean
   over circuits and the standard error.

How to run it: [Run the tomography and purity of the reference
qubit](../README.md#run-the-tomography-and-purity-of-the-reference-qubit-issue-277).

## Results and discussion

`N = 8`, `T = 16`, 10 fixed circuits, 8000 trajectories per `p`, QELM on M
on the raw record (standard error over circuits about 0.025 on the exact
`R2`, 0.005 on the shot `R2`):

| p | E[Tr rho^2] | Bloch exact, M | Bloch exact, state | Bloch shots, M | Bloch shots, state | Bloch shots, Bayes | purity MAE, every readout |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.56 | 0.44 | 0.00 | 0.001 | 0.027 | 0.035 | 0.048 |
| 0.10 | 0.64 | 0.31 | 0.00 | 0.009 | 0.00 | 0.094 | 0.086 |
| 0.175 | 0.77 | 0.23 | 0.00 | 0.019 | 0.00 | 0.187 | 0.113 |
| 0.25 | 0.88 | 0.21 | 0.00 | 0.027 | 0.00 | 0.250 | 0.093 |
| 0.40 | 0.98 | 0.24 | 0.00 | 0.044 | 0.00 | 0.319 | 0.022 |
| 0.70 | 1.00 | 0.36 | 0.00 | 0.091 | 0.00 | 0.341 | 0.000 |
| 1.00 | 1.00 | 0.50 | 0.00 | 0.132 | 0.00 | 0.330 | 0.000 |

- **Exact targets: the QELM on M reads part of the Bloch vector, with a
  minimum near the transition.** Its `R2` falls from 0.44 at `p = 0.05` to a
  plateau of 0.21 between `p = 0.2` and `0.3`, then rises to 0.50 at
  `p = 1`. The minimum sits where `R` purifies (`E[Tr rho^2]` from 0.8 to
  0.93). The QELM on the state reads nothing (`|R2| < 0.01`): chain
  expectation values before `R` is measured hold no coherence between the
  two branches of `R`. M and the state together equal M alone.
- **Single shots: the Bayes ceiling is the purification curve.** Against one
  shot of `R`, the Bayes decoder reaches `E[r_a^2]`: 0.035 at `p = 0.05`,
  then up to about 1/3 once `R` is pure. The QELM on M follows far below
  (0.02 at `p_c`, 0.13 at `p = 1`): training on ±1 labels with 128 features
  and about 1900 trajectories per axis costs most of what it reads from the
  exact targets. The QELM on the state, which now reads the chain after the
  shot, is close to Bayes at `p = 0.05` (0.027 against 0.035, the outcome is
  copied on the chain while `R` is still entangled with it) and at chance as
  soon as `p >= 0.075`.
- **The purity is not read from the raw record.** On `vec(M)`, its `R2` is
  0 within 0.01 at every `p` and the mean absolute error of every readout
  equals that of chance: a bell that peaks at **0.113 at `p = 0.175`**, where
  the purity varies most over the trajectories, and goes to 0 once `R` is
  pure. The purity is even in the outcomes: it depends on *where* the chain
  was measured, which `vec(M)` does not see linearly. With
  `bloch_task.m_features=raw_abs` (`[M, |M|]`, earlier run of the same
  circuits) the `R2` of the purity rises to 0.56 at `p = 0.05` and 0.22 at
  `p = 0.2`, and the peak of the error falls to 0.095.
- **Gap to Bayes.** On exact targets, a ridge on M recovers at best half of
  the Bloch vector; on single shots, at best 40 % of the Bayes ceiling. The
  larger sizes on the cluster (`N = 10, 12`) tell how the gap scales.

## Figures

![R2 of the Bloch vector against p](figures/bloch_vector_N8.png)

**Figure 1. Tomography of the reference qubit.** Test `R2` of the Bloch
vector of `R` against `p`, `N = 8`, `T = 16`, 10 fixed Haar circuits, 8000
trajectories per `p`. Top: against the exact `(<X_R>, <Y_R>, <Z_R>)` given
`M`. Bottom: against one Pauli measurement of `R` along a random axis per
trajectory. Red: ridge on the raw record `M` (QELM on M); orange: ridge on
`M` and the state (QELM on M and the state, almost on the red curve); blue:
ridge on `<P_k>`, `<P_i P_j>` of the final chain (QELM on the state); black:
Bayes optimal on M (the exact `r(M)`: `R2 = 1` on top, `E[r_a^2]` at the
bottom); dashed grey: chance. Bands: standard error over the circuits.
Produced by `experiments/bloch_purity/plot_bloch_purity.py` (#277).

![R2 and error of the purity against p](figures/purity_N8.png)

**Figure 2. Purity of the reference qubit.** Top: test `R2` of a ridge
fitted on `Tr(rho_R^2)` against `p`, same runs and colors as Figure 1; green
dots, right axis: exact mean purity, the purification curve. Points where the
purity varies by less than 0.01 over the trajectories (`p >= 0.7`) are not
drawn. Bottom: mean absolute error of the predicted purity, defined at every
`p`; Bayes is 0, and the three QELM curves lie on chance. The dotted
horizontal line marks the top of the bell (0.113 at `p = 0.175`). Produced
by `experiments/bloch_purity/plot_bloch_purity.py` (#277).
