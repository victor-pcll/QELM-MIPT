# Experiment scripts

Bash scripts that launch QELM runs from the terminal. Each script fixes a set of
parameters, calls the Hydra entry point once per combination, and writes every
run to its own folder under `outputs/`, which git ignores (see `.gitignore`).

## Run the test experiment

From the repository root, after `uv sync`:

```bash
./experiments/example/test_experiment.sh
```

It runs `main.py` on the TFIM for `N` in `4 6` and `seed` in `0 1`, four runs in
a few seconds. Each run lands in
`outputs/test_experiment/<date_time>/N<N>_seed<seed>/` with:

- `checkpoint.npz`: the checkpoint written by the pipeline;
- `main.log`: the run log;
- `.hydra/config.yaml`: the full composed config, so the run can be reproduced;
- `.hydra/overrides.yaml`: the overrides given on the command line.

Set `OUT_ROOT` to write elsewhere: `OUT_ROOT=/tmp/qelm ./experiments/example/test_experiment.sh`.

## Run the projective-readout experiment

```bash
./experiments/projective/projective_readout.sh
```

It runs `experiments/projective/projective_readout.py` for `N` in `4 6` with `seed=0` and
200 samples (`L=1000 ./experiments/projective/projective_readout.sh` for more).
Each run lands in `outputs/projective_readout/<date_time>/N<N>_seed<seed>/` with
`projective_readout.npz` (`Phi`, `Y`, `M`, `X`, `y_test`, `y_pred`, `W`),
`metrics.json` (NMSE, R2) and `.hydra/config.yaml`.

## Run the simple QELM experiment

```bash
./experiments/QELM_Z/exp_QELM_Z.sh
```

It runs `experiments/QELM_Z/exp_QELM_Z.py` for `N` in `4 6` with `seed=42`, 500 samples and
the default `training.alpha=1e-3` of `conf/training/regression/ridge.yaml` (`L=1000 ALPHA=1e-2 ./experiments/QELM_Z/exp_QELM_Z.sh`
to change them). No measurement during the evolution: the readout learns the
ancilla `<Z_0>` from the system observables `<Z_k>, <Y_k>, <X_k>` after
`U(t)`. Each run lands in `outputs/exp_QELM_Z/<date_time>/N<N>_seed<seed>/` with
`exp_QELM_Z.npz` (`X`, `Y`, `y_test`, `y_pred`, `W`), `metrics.json` (NMSE, R2)
and `.hydra/config.yaml`.

## Run the QELM experiment on the measurement record (work in progress)

```bash
./experiments/QELM_M/exp_QELM_M.sh
```

It runs `experiments/QELM_M/exp_QELM_M.py` for every combination of
`N_VALUES` (default `4`), `SEEDS` (default `42`) and `P_VALUES`
(`projective_measurement.p`, default `0.1 0.3 0.5 0.8 1.0`), with `L=12`
samples per run by default. Each list is an environment variable:

```bash
L=500 SEEDS="42 43 44" P_VALUES="0.0 0.2 0.5 1.0" ./experiments/QELM_M/exp_QELM_M.sh
```

The evolution time is split into `projective_measurement.nbr_time_evolution`
steps (default 5, set in `conf/training/measurements/projective.yaml`,
independent of `N`), and the system
qubits are measured after each step. The features are this measurement record
`M`, of shape `(nbr_time_evolution, Ns)` per
sample, instead of the observables; an unmeasured qubit (`p < 1`) counts as 0.
The target is the outcome ±1 of the ancilla `R` measured in Z at the end.
`R` is a reference qubit that the TFIM does not act on, so this label is the
charge sector the system started in, ±1 with probability 1/2 whatever `p`
(issue #104); `accuracy_optimal` is the accuracy of the Bayes classifier
`sign(<Z_0>)`, the ceiling of the readout.
Each run lands in `outputs/exp_QELM_M/<date_time>/N<N>_seed<seed>_p<p>/`.

`notebooks/results/02_QELM_M_analysis.ipynb` runs this sweep in its own
folder, `outputs/QELM_M/<date_time>/` (issue #151), with one progress bar for the whole sweep (`QELM.utils.run_with_progress`), reads the
run folders back with `QELM.utils.load_sweep` and plots the metrics against
`p`, averaged over the seeds with the standard error as error bars. The
figure and its data go to the sweep folder, `metrics_vs_p.png` and
`metrics_vs_p.json`.

## Run the charge-sharpening benchmark (issue #137)

```bash
./experiments/charge_sharpening/charge_sharpening.sh
```

A test of the library on a published result: it reproduces Fig. 7 of
Agrawal et al., "Observing quantum measurement collapse as a learnability
phase transition", PRX 14, 041012 (2024), at the paper's `L = 6` system
qubits, i.e. `N = 7` here (the ancilla `R` is one of the `N` qubits). The
config is `conf/charge_sharpening.yaml`, which composes `conf/config.yaml`
and changes:

- the initial state, Eq. (B1): `(|0>_R |010101> + |1>_R |010100>) / sqrt(2)`
  (`ancilla_charge_sector.initial=product`), scrambled by 5 all-to-all
  layers of random U(1) gates;
- the dynamics: `U1_circuit.depth = Ns / 2 = 3` brickwork layers of random
  charge-conserving gates (`QELM.models.circuits`), each a full period of
  the brickwork (even then odd bonds), open boundaries, one fixed circuit
  instance per seed (`U1_circuit.fixed_instance=true`);
- the measurement: between two layers (`t - 1 = 2` rounds), every system
  qubit is measured weakly with strength `weak_measurement.gamma`
  (`QELM.training.measurements.weak`), `gamma = 0` learning nothing and
  `gamma = 1` being projective.

The paper does not spell out the timing of the layers and measurements;
this one was chosen against the digitized curves, which it matches within
about 0.01 nats for `S_R` (measuring after every layer, or taking a layer as
half a brickwork period, sharpens the charge about 0.1 too early in `gamma`).

Each run computes the ancilla entropy `S_R` (nats, as in the paper) of `samples.L = 10000`
trajectories and its mean, variance and Binder ratio
`B = 1 - mu_4 / (3 mu_2**2)` (`QELM.evaluation.charge_sharpening`), with
bootstrap errors. The launcher sweeps `GAMMA_VALUES` (default
`0.0 0.1 ... 1.0`), `SEEDS` (default `0`) and `N_VALUES` (default `7`):

```bash
SEEDS="0 1 2 3 4" L=10000 ./experiments/charge_sharpening/charge_sharpening.sh
```

Each run lands in `outputs/charge_sharpening/<date_time>/N<N>_seed<seed>_g<gamma>/`
with `charge_sharpening.npz` (`S_R`, the click record `M`, the charge check
`charge_leakage` and `kept`), `metrics.json` and
`.hydra/config.yaml`. At the end, `plot_charge_sharpening.py` draws the three
panels of Fig. 7 in `charge_sharpening.png`, with the plotted data in
`charge_sharpening.json`, over the curves digitized from the paper in
`experiments/charge_sharpening/reference/` (see the README there), and
writes the deviation at the reference points to `comparison.json`. The
paper counts the system sites only: its curves `L = 6` and `L = 20` are
renamed `N = 7` and `N = 21`, our `N` counting the ancilla.

`notebooks/benchmarks/03_charge_sharpening.ipynb` runs this sweep from
Jupyter, in `outputs/charge_sharpening_fig7/<date_time>/` (issue #151), with one progress bar for the whole sweep, then shows the three
panels over the paper's curves, prints the deviation from its `N = 7` curve,
and plots the distribution of `S_R` over the trajectories at a few values of
`gamma` (`S_R_distribution.png` and `.json` in the sweep folder).

Two exact limits are checked by the tests: `S_R = 1` in every trajectory at
`gamma = 0` (gates on the system cannot change the ancilla's entropy) and
`S_R = 0` at `gamma = 1` (a projective record of every qubit reveals the
charge).

## Charge conservation check (issue #134)

Both U(1) benchmarks (charge sharpening and learnability) check along every
trajectory or record that the global charge is conserved. After each step,
`QELM.physics.charge_leakage` gives the weight of the state outside the
sector of its starting charge (the total charge of `R + S`, `Ns // 2`, for
charge sharpening; the charge `Q` of the record for learnability). The
largest value over the evolution is saved per run as `charge_leakage`; it is
exactly 0 for a correct simulation, because every gate and measurement is
block-diagonal in the charge. `metrics.json` reports `max_charge_leakage`,
`n_non_conserved` (runs with a leakage above `charge_conservation.atol`,
default `1e-10`), `n_discarded` and `discarded_fraction`.

The filter is off by default, so the baseline results do not change. Turn it
on to drop the non-conserving runs before the statistics:

```bash
uv run python experiments/charge_sharpening/charge_sharpening.py \
    charge_conservation.filter_non_conserved_charge=true
```

The `.npz` keeps every run; its boolean `kept` marks the ones the statistics
used.

## Run the charge-learnability benchmark (issue #113)

```bash
./experiments/learnability/learnability.sh
```

A second test of the library on a published result: Fig. 2b of Barratt,
Agrawal, Potter, Gopalakrishnan and Vasseur, "Transitions in the learnability
of global charges from local measurements", PRL 129, 200602 (2022). The
config is `conf/learnability.yaml`; the simulation is
`QELM.evaluation.learnability`.

- A chain of `L = N` qubits (no ancilla, `Nr = 0`) starts in the uniform
  superposition of the bitstrings of charge `Q0 = L // 2` (half of the
  records) or `Q1 = L // 2 - 1` (the other half).
- It evolves for `t_f = L` time steps (`learnability.depth`), each a full
  period of a random U(1) brickwork (`QELM.models.circuits`), with a new
  circuit for every record. After each layer of the brickwork (so twice per
  step, `learnability.measure_every=layer`) every site is measured in Z with
  probability `learnability.p`, by the Born rule. The paper does not say
  when within a step the sites are measured; this choice matches the
  digitized Fig. 2c within 0.02 nats, while measuring once per step
  (`measure_every=period`) learns the charge about twice too slowly in `p`.
- Eve sees the record but not the gates. Her optimal classifier is the Bayes
  posterior `P(Q | m)` computed with the Haar-averaged dynamics, a symmetric
  exclusion process. `P_corr = P(Q* | m)` is the posterior of the true charge.
- The accuracy of Fig. 2b is `P(P_corr > 1/2)`, a tie counting 1/2 so that
  a blind guess scores 1/2 (`accuracy_strict` counts it 0). The inset is the
  Binder ratio `1 - mu_4 / (3 mu_2**2)` of `P_corr`, whose curves for
  several `L` cross at `p_tails ~ 0.1` in the paper. The same run also writes
  the mean binary entropy `H(P_corr)` of Fig. 2c (issue #115).

The launcher sweeps `P_VALUES`, `SEEDS` and `N_VALUES` (default `6`) with
`L = 40000` records per run, as in the paper:

```bash
N_VALUES="6 8 10" SEEDS="0 1 2" ./experiments/learnability/learnability.sh
```

Each run lands in `outputs/learnability/<date_time>/N<N>_seed<seed>_p<p>/`
with `learnability.npz` (`P_corr`, `charge`, the records `outcomes`, the
charge check `charge_leakage` and `kept`),
`metrics.json` and `.hydra/config.yaml`. At the end,
`plot_learnability.py` draws the accuracy and the Binder ratio against `p`
in `learnability.png` (data in `learnability.json`), over the curves
digitized from the paper in
`experiments/learnability/reference/`, and writes to `comparison.json` the
deviation at the reference points and the `p` where the Binder curves of
two sizes cross. `notebooks/benchmarks/01_learnability_accuracy.ipynb` runs
the same sweep, in `outputs/learnability_fig2b/<date_time>/` with
`FIGURES=2b` (issue #151), and plots it.

Two exact limits are checked by the tests: at `p = 0` every `P_corr` is 1/2
and the accuracy is 1/2; at `p = 1` the first record gives the charge and
the accuracy is 1.

### Fig. 2c: the mean entropy (issue #115)

The same runs give Fig. 2c, the success transition: `plot_learnability.py`
also draws the mean entropy `E[H(P_corr)]` of the binary distribution
`{P_corr, 1 - P_corr}` (in nats, as in the paper: `learnability.entropy_base`)
and, from the inset, the Binder ratio of `H(P_corr)`, whose curves for
several sizes `N` cross at `p_success ~ 0.2` in the paper. The figure is
`learnability_fig2c.png` and the comparison `comparison_fig2c.json`, against
`reference/barratt2022_fig2c_entropy.csv` (`N = 6` and `N = 20`) and
`reference/barratt2022_fig2c_binder.csv` when present. The mean entropy is
`ln 2` at `p = 0` (Eve knows nothing) and 0 at `p = 1`.
`notebooks/benchmarks/02_learnability_entropy.ipynb` runs the sweep, in
`outputs/learnability_fig2c/<date_time>/` with `FIGURES=2c`, and plots
Fig. 2c.

## Write your own

1. Copy the test script and make the copy executable:

   ```bash
   cp experiments/example/test_experiment.sh experiments/example/my_sweep.sh
   chmod +x experiments/example/my_sweep.sh
   ```

2. Edit the `Parameters` block: `EXPERIMENT` names the output folder, and the
   bash arrays (`N_VALUES=(4 6 8)`) hold the values to sweep. Add a loop for any
   new swept parameter.
3. Edit the overrides passed to `uv run`. Any key in `conf/` takes a
   `key=value` override, as on the command line, for example
   `"training/regression=linear"`, `"TFIM.J=2.0"` or
   `"projective_measurement.p=0.5"`. Group choices use `/`, values use `.`.
4. To run your own experiment logic instead of the pipeline, copy
   `experiments/example/run_experiment.py`, put the logic in its `main`, and call that
   file instead of `main.py`. It accepts the same overrides.

Rules that keep runs clean:

- Folder names passed in `hydra.run.dir` must not contain `=`: Hydra's override
  parser rejects it inside a value. Use `N4_seed0`, not `N=4_seed=0`.
- `set -euo pipefail` at the top stops the script at the first failing run, so
  a sweep never continues silently past an error.
- A script that should be shared goes in a pull request; a personal one can
  stay untracked.

## Hydra multirun

Hydra can also sweep on its own with `-m`, without a bash loop:

```bash
uv run python main.py -m N=4,6 seed=0,1 \
  hydra.sweep.dir=outputs/multirun
```

Each combination goes to a numbered subfolder of `hydra.sweep.dir`. The bash
loop is easier to read and to extend (for example to chain a plotting step
after the runs), which is why the scripts here use it.
