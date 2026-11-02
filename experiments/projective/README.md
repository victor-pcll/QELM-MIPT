# Projective readout: the ancilla from a measurement record

The first experiment with measurements during the evolution. The system
qubits of a disordered TFIM are measured projectively in Z after each time
step, and a linear readout learns `<Z_0>` of the ancilla from the record of
outcomes and the final magnetizations. It is the prototype of
[`QELM_M`](../QELM_M/README.md), which replaced it for the sweeps over the
measurement probability `p`.

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/projective/projective_readout.py`, launcher `projective_readout.sh` |
| Config | `conf/config.yaml` (`models=TFIM`, `training/measurements=projective`) |
| Physical model | all-to-all TFIM `H = h sum_i Z_i + sum_{i<j} J_ij X_i X_j`, a new disorder per sample |
| System sizes `N` | `4 6` by default |
| Control parameters | measurement probability `p` (`projective_measurement.p`), back-action on or off (`+experiment.backaction=false`) |
| Initial state | ancilla `R` entangled with the charge of the system (`AncillaChargeSectorEncoding`) |
| ML task | regression of `<Z_0>` of the ancilla from the flattened record `M` and the final `<Z_k>`, ridge readout, scored by NMSE and R2 |

## Methodology

For each sample:

1. Draw a new TFIM and prepare the charge-sector state.
2. Evolve the density matrix for `evolution.n_steps` steps of
   `evolution.dt`. After every step but the last, measure each system qubit
   in Z with probability `p` (Born rule, the state collapses unless
   `backaction=false`) and store the outcome, `+1`, `-1`, or `0` for a qubit
   that is not measured (the `<Z>` of a fully mixed qubit: no information).
3. After the last step, store `<Z_k>` of the system qubits as the last row
   of the features, and `<Z_0>` of the ancilla as the target.

The features `(n_steps, N)` are flattened and a ridge readout is fitted on
the training samples. How to run it:
[experiments/README.md, "Run the projective-readout experiment"](../README.md#run-the-projective-readout-experiment).

## Results and discussion

- The experiment validates the measurement machinery (`ProjectiveMeasurement`,
  Born-rule sampling, back-action) used by every later experiment.
- Its target, `<Z_0>` on the final state, changes with `p` itself, so a
  change of score with `p` mixes what the record reveals with a change of the
  task. [`QELM_M`](../QELM_M/README.md) fixes this with the measured outcome
  of `R` as label (issue #104), and its sweeps hold the results.

## Figures

None: the run writes `projective_readout.npz` (`Phi`, `Y`, `M`, `X`,
`y_test`, `y_pred`, `W`) and `metrics.json` (NMSE, R2).
