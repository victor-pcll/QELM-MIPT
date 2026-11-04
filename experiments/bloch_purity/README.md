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
| ML task | regression of `r(M)` (3 targets) and of `Tr(rho_R^2)`; ridges on `M`, on the state, on both; test `R2` |
| Issues / PRs | #277; builds on #269 / PR #278 and #270 / PR #275 |

## Methodology

1. One trajectory: `T` rounds of Haar U(4) gates (the same circuit for every
   trajectory of a seed), each followed by a Z measurement of every system
   qubit with probability `p`. Row `t` of the record `M` holds the outcomes,
   +-1, and 0 for an unmeasured qubit.
2. Targets: the exact reduced state `rho_R` of `R` given `M` and the
   circuit, its Bloch vector `r(M)` and its purity `(1 + |r|^2) / 2`. `R` is
   not measured.
3. Features, none of which needs to simulate the circuit:
   - **QELM on M**: the one-hot record, `[vec(M), vec(|M|)]` (`2 N T = 256`
     features). `|M|` (which sites were measured) is needed for the purity,
     which is even in the outcomes;
   - **QELM on the state**: `<P_k>` and `<P_i P_j>` (`P` in `X, Y, Z`) of the
     final chain state, as in `haar_readouts` (108 features);
   - **QELM on M and the state**: both side by side.
4. Readout: a ridge (`RidgeCV`, one alpha per target chosen by leave-one-out
   over `10^-3 .. 10^4`), fitted on 70 % of the 8000 trajectories of one `p`
   and scored on the other 30 %.
5. References: **Bayes optimal on M** is the exact `r(M)` and purity of an
   observer who knows the circuit. They are the targets, so their `R2` is 1:
   the ceiling of any readout of `M` (the oracle of #277). **Chance** is the
   mean of the training targets, `R2 = 0`.
6. Score: the `R2` of the vector, `1 - sum |r_pred - r|^2 / sum |r - mean r|^2`
   over the three components (each component is also saved), the `R2` of
   the purity, and its mean absolute error.
7. Sweep: 10 seeds (42 to 51), one circuit each; the figures show the mean
   over circuits and the standard error.

How to run it: [Run the tomography and purity of the reference
qubit](../README.md#run-the-tomography-and-purity-of-the-reference-qubit-issue-277).

## Results and discussion

`N = 8`, `T = 16`, 10 fixed circuits, 8000 trajectories per `p` (standard
error over circuits about 0.025 on the `R2`):

| p | E[Tr rho_R^2] | R2 Bloch, M | R2 Bloch, state | R2 purity, M | R2 purity, state |
|---|---|---|---|---|---|
| 0.05 | 0.56 | 0.44 | 0.00 | 0.56 | 0.01 |
| 0.10 | 0.64 | 0.30 | 0.00 | 0.38 | 0.00 |
| 0.20 | 0.81 | 0.21 | 0.00 | 0.22 | 0.00 |
| 0.25 | 0.88 | 0.20 | 0.00 | 0.17 | 0.00 |
| 0.30 | 0.93 | 0.21 | 0.00 | 0.13 | 0.00 |
| 0.40 | 0.98 | 0.23 | 0.00 | 0.05 | 0.00 |
| 0.70 | 1.00 | 0.35 | 0.00 | (undefined) | (undefined) |
| 1.00 | 1.00 | 0.50 | 0.00 | (undefined) | (undefined) |

- **The QELM on M reads part of the Bloch vector, with a minimum near the
  transition.** Its `R2` falls from 0.44 at `p = 0.05` to a plateau of 0.20
  between `p = 0.2` and `0.3`, then rises to 0.50 at `p = 1`. The minimum
  sits where `R` purifies (`E[Tr rho^2]` from 0.8 to 0.93): at small `p` few
  outcomes carry the information, each one linearly, while near `p_c` the
  information is spread over many outcomes in a non-linear way. The three
  components behave alike (`x` a little lower than `y` and `z`).
- **The QELM on the state reads nothing**, on both targets, at every `p`
  (`|R2| < 0.01`). Its features are expectation values of the chain alone,
  before `R` is measured: they contain no coherence between the two branches
  of `R`, and the purity (a quadratic quantity of the joint state) is not a
  linear function of them. This is unlike the Z task of #269, where `R` was
  measured first and its outcome imprinted on the chain.
- **M and the state together equal M alone** (within 0.01): the state adds
  nothing.
- **The purity is read only in the mixed phase.** The `R2` of the QELM on M
  falls from 0.56 at `p = 0.05` to 0.05 at `p = 0.4`. Above `p ~ 0.5`, `R`
  is pure on almost every trajectory (spread of the purity below 0.01), so
  the `R2` divides by a vanishing variance and is not drawn; the mean error
  is then 0 for every readout. The bottom panel shows that the gain of M
  over chance, in absolute error, is largest (0.02) for `p` from 0.05 to
  0.2.
- **Gap to Bayes.** The Bayes decoder, which knows the circuit, reaches
  `R2 = 1` by definition: a ridge on M recovers at best half of the Bloch
  vector. The gap is the same question as in #269 (a sample-complexity gap,
  or information in high orders of M); the larger sizes on the cluster tell
  how it scales with `N`.

## Figures

![R2 of the Bloch vector against p](figures/bloch_vector_N8.png)

**Figure 1. Tomography of the reference qubit.** Test `R2` of the Bloch
vector `(<X_R>, <Y_R>, <Z_R>)` against `p`, `N = 8`, `T = 16`, 10 fixed Haar
circuits, 8000 trajectories per `p`. Red: ridge on the record `M` (QELM on
M), hidden under orange: ridge on `M` and the state (QELM on M and the
state); blue: ridge on `<P_k>`, `<P_i P_j>` of the final chain (QELM on the
state); black: Bayes optimal on M (the exact `r(M)`, `R2 = 1`); dashed grey:
chance. Bands: standard error over the circuits. Produced by
`experiments/bloch_purity/plot_bloch_purity.py` (#277).

![R2 and error of the purity against p](figures/purity_N8.png)

**Figure 2. Purity of the reference qubit.** Top: test `R2` of a ridge fitted
on `Tr(rho_R^2)` against `p`, same runs and colors as Figure 1; green dots,
right axis: exact mean purity, the purification curve. Points where the
purity varies by less than 0.01 over the trajectories (`p >= 0.7`) are not
drawn. Bottom: mean absolute error of the predicted purity (chance under the
QELM on the state). Produced by `experiments/bloch_purity/plot_bloch_purity.py`
(#277).
