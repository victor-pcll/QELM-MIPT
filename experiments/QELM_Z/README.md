# Simple QELM: the ancilla from the system observables

The baseline of the project: a quantum extreme learning machine (QELM) with
no measurement during the evolution. A disordered transverse-field Ising
model (TFIM) scrambles a charge that the ancilla `R` is entangled with, and a
linear readout of one-site observables of the system learns `<Z_0>` of the
ancilla. Every measured experiment of the repository (`QELM_M`, `QELM_Q`)
reduces to this one at `p = 0`.

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/QELM_Z/exp_QELM_Z.py`, launcher `exp_QELM_Z.sh` |
| Config | `conf/config.yaml` (`models=TFIM`, `training/regression=ridge`) |
| Physical model | all-to-all TFIM `H = h sum_i Z_i + sum_{i<j} J_ij X_i X_j`, `J_ij` uniform in `[-J/2, J/2]`, a new disorder per sample, `h = 10 J` by default |
| System sizes `N` | `4 6` by default (system qubits, plus the ancilla) |
| Control parameters | none (no measurement); ridge penalty `training.alpha` (default `1e-3`) |
| Initial state | `(|0>_R |Q1>_S + |1>_R |Q2>_S) / sqrt(2)` (`AncillaChargeSectorEncoding`), `|Q>_S` the uniform superposition of the configurations of charge `Q`, `Q1 = N // 2`, `Q2 = Q1 - 1` |
| ML task | regression of `<Z_0>` of the ancilla from `<Z_k>, <Y_k>, <X_k>` of the system qubits, ridge readout with a bias column, scored by NMSE and R2 |

## Methodology

For each of the `samples.L` samples (500 by default):

1. Draw a new TFIM and prepare the charge-sector state above.
2. Evolve the state vector for the total time `evolution.t` in one step,
   `U(t) = exp(-i H t)`.
3. Store the one-site expectation values `<O_k>` of the system qubits for
   `O` in `experiment.observables` (default `Z, Y, X`) as features, and
   `<Z_0>` of the ancilla as target.

The ridge readout is fitted on the first `samples.train_size` samples and
scored on the rest. How to run it:
[experiments/README.md, "Run the simple QELM experiment"](../README.md#run-the-simple-qelm-experiment).

## Results and discussion

- At `h >> J` the field term nearly conserves the total charge, so
  `<Z_0> ~ c - sum_k <Z_k>` is close to an affine function of the features,
  which is why the readout carries a bias column: the reservoir maps the
  hidden charge onto local magnetizations that a linear readout can read.
  At `h ~ J` the couplings move the charge and this relation breaks (see the
  `(p, h)` sweeps of [`QELM_Q`](../QELM_Q/README.md)).
- The TFIM conserves the parity of the system and each charge sector has a
  definite parity, so `<X_k>` and `<Y_k>` vanish exactly; only `<Z_k>`
  carries the charge (measured in `notebooks/results/08_QELM_Q_linear_observables.ipynb`,
  issue #191, see [`QELM_Q`](../QELM_Q/README.md)).
- This experiment has no sweep or figure of its own: it is the `p = 0`
  reference point of [`QELM_M`](../QELM_M/README.md) and
  [`QELM_Q`](../QELM_Q/README.md).

## Figures

None: the run writes `metrics.json` (NMSE, R2) and `exp_QELM_Z.npz`
(`X`, `Y`, `y_test`, `y_pred`, `W`). The `p = 0` points of the figures of
[`QELM_Q`](../QELM_Q/README.md) are this experiment.
