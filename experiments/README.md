# Experiment scripts

Bash scripts that launch QELM runs from the terminal. Each script fixes a set of
parameters, calls the Hydra entry point once per combination, and writes every
run to its own folder under `outputs/`, which git ignores (see `.gitignore`).

Next to `metrics.json`, every Python experiment script (`projective/`,
`QELM_Z/`, `QELM_M/`, `QELM_Q/`, `charge_sharpening/`, `learnability/`,
`double_descent/`) writes
`resolved_config.json`, the config of the run with every interpolation
evaluated (`QELM.utils.save_config`, issue #85). `.hydra/config.yaml` keeps the
formulas as written in `conf/`, such as `depth: ${int:${div:${Ns},2}}`;
`resolved_config.json` holds the values they gave, here `"depth": 3`, and is
written after the run, so a value the script sets in the config is included:

```python
import json
from pathlib import Path

cfg = json.loads(Path("outputs/.../resolved_config.json").read_text())
cfg["Ns"], cfg["samples"]["train_size"]
```

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

`H_VALUES` sweeps the transverse field `TFIM.h`, in units of `TFIM.J = 1`
(issue #174); each run then lands in `N<N>_h<h>_seed<seed>_p<p>/`. Unset, the
field of the config applies (`h = 10 J`). It needs `MODEL=TFIM`: a circuit has
no field, and the launcher stops before any run.

```bash
H_VALUES="1 10" N_VALUES="4 6" L=500 ./experiments/QELM_M/exp_QELM_M.sh
```

Besides the ancilla and system entropies, every run records `S_half`, the von
Neumann entropy of the first `Ns // 2` system qubits in the final state, and
its mean `S_half_mean` in `metrics.json` (issue #174).

`notebooks/results/02_QELM_M_analysis.ipynb` runs this sweep in its own
folder, `outputs/QELM_M/<date_time>/` (issue #151), with one progress bar for the whole sweep (`QELM.utils.run_with_progress`), reads the
run folders back with `QELM.utils.load_sweep` and plots the metrics against
`p`, averaged over the seeds with the standard error as error bars, in two
figures (issue #174): the performance of the readout (`performance_vs_p.png`
and `.json`) and the entanglement of the final state (`entanglement_vs_p.png`
and `.json`), both in the sweep folder. Its last section sweeps `p` again for
`h = J` and `h = 10 J` and `N = 4, 6`, in `outputs/QELM_M_h/<date_time>/`, and
plots the accuracy, the R2 and `S_half` against `p` (`scaling_vs_p.png`).

### Sample-to-parameter ratio sweep (issue #96)

```bash
./experiments/QELM_M/exp_QELM_M_ratio.sh
```

It runs `experiments/QELM_M/exp_QELM_M_ratio.py`, the same experiment with
the number of training samples set by the ratio
`alpha_ratio = n_train / n_params` (`experiment.alpha_ratio`, config
`conf/QELM_M_ratio.yaml`), where `n_params = nbr_time_evolution * Ns + 1` is
the number of weights of the ridge readout (16 at `N = 4`). The script sets
`samples.train_size = round(alpha_ratio * n_params)` and
`samples.L = train_size + n_test`, with a test set of fixed size
`experiment.n_test` (default 200), checks that `X` has `n_params` columns,
then runs `exp_QELM_M.run`. `alpha_ratio < 1` is underdetermined and needs a
ridge penalty `training.alpha > 0` (the script stops otherwise);
`alpha_ratio` and `training.alpha` are independent keys. The launcher sweeps
`ALPHA_RATIOS` (default `0.5 1.0 1.5 2.0 5.0 10.0`), `P_VALUES`, `SEEDS` and
`N_VALUES`:

```bash
SEEDS="42 43 44" P_VALUES="0.2 0.5 1.0" ./experiments/QELM_M/exp_QELM_M_ratio.sh
```

Each run lands in
`outputs/exp_QELM_M_ratio/<date_time>/N<N>_seed<seed>_p<p>_a<alpha_ratio>/`
with the files of `exp_QELM_M`, `metrics.json` adding `alpha_ratio`,
`n_train`, `n_params` and `train_ratio`, and `config_used.yaml`, the config
with the sample counts actually used (Hydra writes `.hydra/config.yaml`
before the script sets them). `notebooks/results/03_QELM_M_alpha_ratio.ipynb`
runs the sweep in `outputs/QELM_M_ratio/<date_time>/` and plots the test
NMSE, R2 and balanced accuracy against the ratio, one row of panels per `N`,
with the interpolation threshold `alpha_ratio = 1` dashed
(`metrics_vs_alpha_ratio.png` and `.json` in the sweep folder). Its default
grid is `p` from 0 to 1 and `alpha_ratio` from 0.25 to 10 on a log scale, and
its last section draws one heatmap per metric (NMSE, R2, accuracy, balanced
accuracy) and per `N`, `alpha_ratio` on the x axis and `p` on the y axis, with
`alpha_ratio = 1` dashed (`heatmaps_p_alpha_ratio.png` and `.json`, issues
#164 and #167).

A run that fails (a segmentation fault is exit code 139) does not stop the
sweep (issue #168): it is tried again, up to `MAX_ATTEMPTS` times
(default 2), then listed in `failed_runs.txt` in the sweep folder. A run
whose folder already holds `config_used.yaml`, the last file a run writes, is
skipped, so launching again with the same `OUT_DIR` (`RESUME` in the
notebook) completes an interrupted sweep without redoing the finished runs.

### Haar and Clifford dynamics (issue #171)

`models=haar_circuit` and `models=clifford_circuit` replace the TFIM of
`exp_QELM_M` (and so of `exp_QELM_M_ratio`) by a brickwork circuit on the
system qubits: Haar-random two-qubit gates of U(4), or two-qubit Clifford
gates drawn uniformly from the 11 520 elements of the group
(`QELM.models.circuits`). Each time-evolution step applies `layers_per_step`
brickwork layers (default 1, each a full period, even then odd bonds), each
with its own gates, and every sample gets new gates, as every TFIM sample
gets a new disorder (`fixed_instance: false`). The initial state, the
measurements, the record, the target and the scores are those of the TFIM
run. Both launchers take the model from `MODEL`:

```bash
MODEL=clifford_circuit P_VALUES="0.2 0.5 1.0" ./experiments/QELM_M/exp_QELM_M_ratio.sh
```

`notebooks/results/05_QELM_M_circuits.ipynb` runs the ratio sweep once per
model in `outputs/QELM_M_circuits/<date_time>/<model>/`, draws the curves
and heatmaps of notebook 03 for each, and compares the three dynamics
(`comparison_models.png`): the balanced accuracy of the readout and the
Bayes accuracy `accuracy_optimal` against `p`, and the NMSE against the
ratio. Its last section runs the same sweep of the two circuits with one
fixed draw of their gates per seed (`EXTRA_OVERRIDES="<model>.fixed_instance=true"`,
any Hydra overrides added to every run of the ratio launcher), in
`<model>_fixed/`, and compares the two (`comparison_fixed_circuit.png`).

## Run the QELM experiment on the charge Q under intermediate measurements (issue #163)

```bash
./experiments/QELM_Q/exp_QELM_Q.sh
```

It runs `experiments/QELM_Q/exp_QELM_Q.py` for every combination of
`N_VALUES` (default `4`), `H_VALUES` (the transverse field `TFIM.h`, default
`10`, i.e. `10 J`), `T_VALUES` (the total time `evolution.t`, default `10`,
i.e. `10 / J`), `SEEDS` (default `42`) and `P_VALUES`
(`projective_measurement.p`, default `0.0 0.2 0.5 0.8 1.0`) and `FINAL_VALUES`
(`projective_measurement.final_measurement`, default `false`), with `L=200`
samples per run by default. The ancilla `R` is measured in Z first, which
fixes the charge sector of the system: the target is that charge,
`Y = Q` (`Ns // 2` or `Ns // 2 - 1`). The system evolves in
`projective_measurement.nbr_time_evolution` steps (default 5), and **between**
two steps each system qubit is measured in Z with probability `p`; by
default there is no measurement after the last step, and
`final_measurement=true` adds one, so that the features are read right after
a collapse. The features are only the final
`<Z_k> = Tr(Z_k rho)` of the system qubits, never the outcomes. The primary
metric is `accuracy` (the prediction on the right side of the midpoint of the
two charges), with `accuracy_baseline` and `balanced_accuracy` as in
`exp_QELM_M`, plus `S_half_mean`, the mean half-system entanglement entropy of
the final state. The readout is also fitted, separately, on five other
feature sets of the same final states, `<X_k>`, `<Y_k>` and the correlators
`<X_i X_j>`, `<Y_i Y_j>`, `<Z_i Z_j>` and the cross correlators
`<X_i Y_j>`, `<Y_i X_j>` (set `XY`) (`accuracy_<name>`, `r2_<name>` in
`metrics.json`). Three more readouts add the record `M` of the intermediate
measurements (±1 for a measured qubit, 0 otherwise, as in `exp_QELM_M`) to
the input: `M` alone, and `phi = [M, X]` with `X` the `<Z_k>` (`M_Z`) or the
`<X_i X_j>` (`M_XX`) (issue #179); the record is saved as `M` in the `.npz`. The measurements draw from their own generator, so at
`p = 0` the run is exactly the standard QELM, one unitary `exp(-i H t)` per
sample. Each run lands in
`outputs/exp_QELM_Q/<date_time>/N<N>_h<h>_t<t>_final<final>_seed<seed>_p<p>/`.

`notebooks/results/04_QELM_Q_measurements.ipynb` checks the `p = 0` baseline,
runs the sweep in `outputs/QELM_Q/<date_time>/` and plots the accuracy, R2
and `S_half` against `p` for each `N` and `h`, one row of panels without and
one with the final round, then the accuracy of each feature set
(`accuracy_by_observable_<setting>.png`), and finally sweeps the evolution
time at `N = 6` in its own folder `outputs/QELM_Q_time/<date_time>/`
(`accuracy_vs_p_time.png`) (`accuracy_vs_p.png` and
`.json` in the sweep folder).

`notebooks/results/06_QELM_Q_observables_p_h.ipynb` (issue #179) runs the
same launcher over a 2D grid of `P_VALUES` (0 to 1) and `H_VALUES` (0 to
3 J), for `N = 4, 6`, in `outputs/QELM_Q_p_h/<date_time>/`, and plots the
accuracy and R2 of the readouts on `<Z_k>` and on `<X_i X_j>` against `p`
(`metrics_vs_p.png`), against `h` (`metrics_vs_h.png`) and as heatmaps over
the `(p, h)` plane (`heatmaps_p_h.png`), then the same three figures for the
readouts on `M`, `[M, <Z_k>]` and `[M, <X_i X_j>]` (`*_record.png`). Each run is small enough to be
fastest on a single thread: the notebook sets `OMP_NUM_THREADS=1`, which
made a run at `N = 6` about 6 times faster.

### Bayes ceilings of the accuracy on `<Z_k>` (issue #202)

Every run of `exp_QELM_Q.py` also writes two ceilings of the ridge
`accuracy`, on the same split and the same `<Z_k>`
(`src/QELM/evaluation/bayes_optimal.py`): `bayes_estimate_qda`, a quadratic
discriminant analysis written in NumPy (one Gaussian per charge sector,
checked against scikit-learn in `tests/test_bayes_optimal.py`), and
`bayes_estimate_knn`, `sqrt(train_size)` nearest neighbors weighted by
distance. Where the ridge accuracy falls to 0.5, ceilings at 0.5 mean the
two sectors overlap (decoherence), ceilings well above 0.5 mean the charge
is still in `<Z_k>` but not linearly (learnability transition). After a
sweep over `N` and `p`,

```bash
uv run python experiments/QELM_Q/plot_bayes_ceiling.py outputs/exp_QELM_Q/<date_time> --h 1
```

writes `bayes_ceiling.png` (the ridge accuracy against `p` per `N`, the QDA
ceiling dashed and the K-NN one dash-dotted in the same color; `--no-ceilings`
draws the ridge alone) and `qda_contours.png` (per `(N, p)`, the test
`<Z_k>` on their PCA plane colored by sector, with the 1 and 2 sigma ellipses
of the Gaussians fitted by the QDA), each with its `.json`. `--h`, `--t`,
`--final` pick one setting of a sweep that holds several, `--p` the columns
of the contour figure.

## Run the double-descent experiment on MNIST (issue #181)

```bash
./experiments/double_descent/double_descent.sh
```

A static, non-linear task where the readout shows the double descent: MNIST
digits 3 against 5 by default, or any two or more digits (`CLASSES="3 5 8"`)
for a harder task. The protocol follows the QELM image classification of
"Harnessing quantum extreme learning machines for image classification",
Phys. Rev. Applied (2025): PCA, angle encoding, a fixed reservoir and a
linear readout of local observables. The first run downloads MNIST
to `data/mnist/` (about 11 MB, gitignored), checked against the SHA-256 of
the published files; `DATASET=digits` uses scikit-learn's 8x8 digits
instead, with no download. The config is `conf/double_descent.yaml`; each
run of `experiments/double_descent/double_descent.py` is one seed:

1. Each image is reduced to `N` principal components (default `N = 8`),
   fitted on the training images of the chosen digits, and each component is
   mapped to an angle in `[0, pi]` (`QELM.datasets.AngleEncoder`).
2. Qubit `i` starts in `R_y(theta_i)|0>` (`QELM.embedding.RotationEmbedding`),
   with no ancilla (`Nr = 0`).
3. One fixed quench `U = exp(-i H t)` acts on every image: a single TFIM,
   drawn once per seed, with `h = J` (not the `10 J` of the TFIM config, at
   which the field dominates) and `t = 10 / J`.
4. The features are `<X_i>, <Y_i>, <Z_i>` and `<X_i X_j>, <Y_i Y_j>,
   <Z_i Z_j>` of the final state, plus a bias column: 109 columns at
   `N = 8`.
5. The readout is `RidgeReadout` (`training/regression=ridge`) with
   `training.alpha = 0` (`RIDGE_ALPHA`): no penalty, the limit `alpha -> 0`
   of ridge regression, `W = pinv(Phi) @ Y`, least squares above
   `n_train = rank(Phi)` and the minimum-norm interpolation below, so that
   the peak is not smoothed; `RIDGE_ALPHA=0.1` regularizes it. The targets
   are one-hot, `Y` of shape `(n, K)` for `K` digits, and the predicted
   digit is the largest output. It is fitted on
   `n_train = round(alpha * n_features)` images drawn from a pool of
   `n_pool = 2000` training images, `n_repeats = 20` times per `alpha`, for
   `alpha` from 0.1 to 10 (`double_descent.alpha_ratios`), and scored on the
   whole MNIST test split of the chosen digits (1902 images for 3 and 5).
   The MSE is averaged over the samples and the `K` outputs.

`n_features` is the **rank** of the feature matrix of the pool, not its
number of columns: two dependent observables would otherwise move the peak
away from `alpha = 1`. With `alpha = n_train / n_features`, `alpha < 1` is
the over-parameterized side (the readout interpolates the training set,
train error 0) and `alpha > 1` the under-parameterized one (least squares).

The same run computes the three checks of the issue, each with the same
readout and the same images:

- `elm`: a classical ELM, one fixed random ReLU layer with as many neurons
  as the QELM has observables (`QELM.evaluation.double_descent.random_features`;
  the hyperplane of each neuron goes through a training image, otherwise
  many ReLUs are always on, affine in the input, and the rank collapses);
- `qelm_t0`: the same observables at `t = 0`. Its simulated features are
  checked against their closed form on the product state
  (`t0_max_deviation`, about `1e-15`): at `t = 0` the readout is a linear
  regression on fixed functions of the encoded input. `<Y_i>` and
  `<Y_i Y_j>` vanish for a real state, so its rank is 73, not 109;
- `peak_alpha_<model>`: the `alpha` of the largest median test MSE.

`linear` (a readout on the `N` angles) and `accuracy_pixels` (a readout on
the 784 pixels fitted on every training image, the classical reference of
the task) complete the picture. The launcher sweeps `SEEDS` (default
`0 1 2`), `CLASSES` (default `3 5`) and `RIDGE_ALPHA` (default `0.0`):

```bash
SEEDS="0 1 2 3 4" N=8 ./experiments/double_descent/double_descent.sh
CLASSES="3 5 8" ./experiments/double_descent/double_descent.sh
```

Each run lands in
`outputs/double_descent/<date_time>/N<N>_c<classes>_seed<seed>/` (`c35` for
3 and 5) with `double_descent.npz` (per model `<m>_alpha`, `<m>_n_train` and
the train and test MSE and classification error and `||W||`, shape
`(n_ratios, n_repeats)`, and `confusion_<m>`, the `(K, K)` confusion matrix
on the test set of the readout fitted on the whole pool, rows the true
digit, from `sklearn.metrics.confusion_matrix`), `metrics.json` and the
config. At the end, `plot_double_descent.py` draws the test MSE, train MSE,
test error and train error against `alpha` (`double_descent.png` and
`.json`), the confusion matrices summed over the seeds and normalized by row
(`confusion_matrices.png` and `.json`), and writes `summary.json`, in the
sweep folder. `notebooks/results/07_double_descent_mnist.ipynb` runs the
sweep in `outputs/double_descent/<date_time>/`, prints the checks and shows
both figures. Three seeds take about 40 s.

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
