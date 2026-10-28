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
(issue #104). No Bayes ceiling is reported (issue #195): the former
`accuracy_optimal`, `sign(<Z_0>)` on the exact state of `R`, used
`P(Y | rho_R)` while the readout only sees `M`, so it was removed with `<Z_0>`.
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
(`comparison_models.png`): the balanced accuracy of the readout against
`p`, and the NMSE against the ratio. Its last section runs the same sweep of the two circuits with one
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

Every run also logs two K-NN accuracies on the same `<Z_k>` and split
(issue #213): `bayes_estimate_knn_euclidean` (Euclidean, with `CustomKNN`;
the `bayes_estimate_knn` ceiling below is the same rule from scikit-learn) and
`bayes_estimate_knn_mahalanobis` (the metric is the pseudo-inverse of the
covariance `Sigma` of the training `<Z_k>`), with `knn_covariance_rank` and
`knn_covariance_cond` (largest / smallest kept eigenvalue of `Sigma`).
`experiments/QELM_Q/plot_knn_metrics.py` compares them with the ridge readout
on a sweep of one `N`, against `p` (one panel per `h`) and against `h` (one
panel per `p`), and plots `knn_covariance_cond`; the runs with
`final_measurement=true` are open squares:

```bash
OUT_DIR=outputs/QELM_Q_knn N_VALUES=8 H_VALUES="0.5 1 2 5 10" SEEDS="42 43" \
    P_VALUES="0.0 0.2 0.4 0.6 0.8 1.0" L=300 ./experiments/QELM_Q/exp_QELM_Q.sh
OUT_DIR=outputs/QELM_Q_knn N_VALUES=8 H_VALUES="0.5 1 2 5 10" SEEDS="42 43" \
    P_VALUES=1.0 FINAL_VALUES=true L=300 ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/plot_knn_metrics.py outputs/QELM_Q_knn
```

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

`notebooks/results/08_QELM_Q_linear_observables.ipynb` (issue #191) calls
`exp_QELM_Q.run` in its own process at `p = 0` with one evolution step, the
unitary QELM, for datasets of increasing size (`samples.L` = 100, 500, 2000,
5000, one seed per size) at `h = 10 J` and `h = J`, in
`outputs/QELM_Q_linear/<date_time>/h<h>_n<L>/`. It splits the features into
`<Z_k>`, `<Y_k>`, `<X_k>`
(`QELM.evaluation.feature_distribution.linear_observables`), refits the
ridge readout of `Q` on each Pauli alone, and superimposes the histograms of
one `<Z_k>` for the different sizes (`QELM.utils.plots.draw_histograms`),
separately for the samples of charge `Q1`, of charge `Q2` and for both, each
panel with its Gaussian fit. It also measures the convergence of the
empirical laws (Kolmogorov-Smirnov distance to the largest dataset) and
their shape. `H` conserves the parity of the system and each sector has a
definite parity, so `<X_k>` and `<Y_k>` are exactly 0.

`experiments/QELM_Q/plot_knn.py` (issue #214) fits K-NN classifiers of `Q`
on the `<Z_k>` of every run of a sweep, with the split of the ridge readout,
and writes two figures in the sweep folder: `knn_parity_vs_p.png`, the
accuracy of scikit-learn's `KNeighborsClassifier(round(sqrt(n_train)),
weights="distance")` and of the NumPy `CustomKNN` against `p`, with the
fraction of test points they classify the same way and the fraction with a
tie at the `K`-th neighbor; and `knn_K_vs_p.png`, the accuracy of `CustomKNN`
for several `K` (`--K`, default `1 5 sqrt 3sqrt`) with the ridge readout.
The accuracies of every run go to `knn_runs.json`; the sweep must hold a
single field `TFIM.h`. The figures of the PR come from:

```bash
OMP_NUM_THREADS=1 N_VALUES="4 6 8" H_VALUES=1 FINAL_VALUES="false true" \
  SEEDS="42 43 44" L=500 \
  P_VALUES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" \
  ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/plot_knn.py outputs/exp_QELM_Q/<date_time>
```

### Bayes-Ridge on the record `M`, finite-size scaling (issue #233)

Every run of `exp_QELM_Q.py` also reads the charge from the record `M` alone
with a two-step pipeline (`QELM.evaluation.bayes_ridge`): for each charge the
log-likelihood `log P(M | Q)` of a generative model, the fully scrambled limit
of the symmetric exclusion process of Barratt et al. 2022 (before each round
the configuration is uniform on charge `Q`, with a leak `epsilon = 1e-3` per
round), then a ridge on the two log-likelihoods. It logs
`accuracy_M_bayes_ridge` (with its balanced accuracy and R2) and
`accuracy_M_bayes`, the Bayes rule of the model alone on the same test
samples. `BALANCED=true` (`samples.balanced=true`) imposes the outcome of the
ancilla, `Q1, Q2, Q1, ...`, so that each sector gets exactly half of the
samples. `experiments/QELM_Q/plot_bayes_ridge_fss.py` draws the accuracy
against `p` for every `N`, next to the SEP Bayes accuracy of the random U(1)
circuit of `experiments/learnability/` at the same `Ns = N - 1`, and tests an
asymptotic behavior in `N` (fits `A_N = A_inf + c / N`, gaps between sizes,
crossings of the curves, steepest slope against `N`). The figures of the PR
come from:

```bash
OMP_NUM_THREADS=1 OUT_DIR=outputs/fss233/tfim N_VALUES="12 10 8 6" \
  SEEDS="42 43" P_VALUES="0.0 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0" \
  FINAL_VALUES=true BALANCED=true STATE=vector L=2000 MAX_JOBS=4 \
  ./experiments/QELM_Q/exp_QELM_Q.sh
# The SEP reference at Ns = N - 1, with the 5 rounds of the TFIM run
# (sep_matched) and with the t_f = Ns steps of the paper (sep_paper).
for NS in 5 7 9 11; do for P in 0.0 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0; do
  uv run python experiments/learnability/learnability.py N=$NS samples.L=8000 learnability.n_boot=50 \
    learnability.p=$P learnability.depth=5 learnability.measure_every=period \
    hydra.run.dir=outputs/fss233/sep_matched/N${NS}_p$P
  uv run python experiments/learnability/learnability.py N=$NS samples.L=8000 learnability.n_boot=50 \
    learnability.p=$P hydra.run.dir=outputs/fss233/sep_paper/N${NS}_p$P
done; done
uv run python experiments/QELM_Q/plot_bayes_ridge_fss.py outputs/fss233/tfim \
  --sep outputs/fss233/sep_matched outputs/fss233/sep_paper
```

With 5 rounds at every `N` the accuracy falls toward 1/2 as `N` grows, as
`N**-1/2`: the scrambled model predicts `d'^2 ~ 4 R p / ((1 - p) Ns)`. With
`R = Ns` rounds at a fixed step `dt = 2 / J` (so `evolution.t = 2 Ns`),
`d'` no longer depends on `N` and the accuracy reaches a limit,
`Phi(sqrt(p / (1 - p)))`; the follow-up sweep of the PR:

```bash
for N in 12 10 8 6; do NS=$((N - 1)); for SEED in 42 43; do
  for P in 0.0 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0; do
    uv run python experiments/QELM_Q/exp_QELM_Q.py N=$N seed=$SEED \
      samples.L=2000 samples.balanced=true \
      projective_measurement.final_measurement=true \
      projective_measurement.p=$P projective_measurement.nbr_time_evolution=$NS \
      TFIM.h=10 evolution.t=$((2 * NS)) state_representation=vector \
      hydra.run.dir=outputs/fss233/tfim_rN/N${N}_seed${SEED}_p$P
done; done; done   # seed 42 only at N = 12, for time
uv run python experiments/QELM_Q/plot_bayes_ridge_fss.py outputs/fss233/tfim_rN \
  --sep outputs/fss233/sep_paper --out outputs/fss233/figures_rN
```

### A local U(1) reservoir: the XXZ chain (issue #236)

The all-to-all TFIM behaves like a fully scrambled reservoir (issue #233):
between two rounds it erases every spatial memory of the outcomes, so the
accuracy of `Q` from `M` follows the urn prediction `Phi(d' / 2)`. With
`models=XXZ` (`MODEL=XXZ` for the launcher) the reservoir is the
nearest-neighbour chain `H = sum J (X X + Y Y) + Delta Z Z + sum h_i Z_i`
(`conf/models/XXZ.yaml`: Heisenberg `Delta = J`, clean `h = 0`, open
boundaries, a step `dt = 0.25 / J` so that a charge moves about one site per
round). It conserves `Q` exactly, so:

- the vector path evolves only the `C(Ns, Q)` amplitudes of the charge
  sector (`sector_step`), after checking that `H` does not couple it to the
  rest of the space;
- every run checks the charge along every record: `charge_leak_max`, the
  largest weight outside the sector of the label after any step (exactly 0
  for XXZ), and `contradicting_rounds_fraction` (0 for XXZ);
- `accuracy_M_bayes_exact` is the scrambled Bayes rule with no leak
  (`epsilon = 0`), which a conserved charge allows.

A charge moves one bond at a time, so where and when a qubit was measured
carries information that the urn model throws away. The local readout keeps
it: `log P(M | Q)` under a hidden Markov model whose configuration of charge
`Q` moves between two rounds by the nearest-neighbour symmetric exclusion
process, with `lam = gamma dt` exchanges per bond per step fitted on the
training records by maximum likelihood (`local_sep_rate`; `lam -> inf` is the
scrambled model). Its Bayes-Ridge accuracy is
`accuracy_M_local_bayes_ridge`, its Bayes rule `accuracy_M_local_bayes`;
`bayes_ridge.local=false` skips it. `plot_bayes_ridge_fss.py` reads the
reservoir from the configs and draws the local readout when it is there.

Since `sum_k <Z_k> = Ns - 2 Q` exactly for a conserved charge, every readout
that sees the final `<Z_k>` reads `Q` perfectly: only the readouts of the
record alone (`M`, Bayes-Ridge, local Bayes-Ridge) are informative here.

```bash
# R = 5 rounds, dt = 0.25 / J.
OMP_NUM_THREADS=1 MODEL=XXZ OUT_DIR=outputs/xxz236/r5 N_VALUES="12 10 8 6" \
  SEEDS="42 43" P_VALUES="0.0 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0" \
  FINAL_VALUES=true BALANCED=true STATE=vector L=2000 MAX_JOBS=4 \
  ./experiments/QELM_Q/exp_QELM_Q.sh
# R = Ns rounds at the same dt (t = 0.25 Ns), one N at a time.
for N in 12 10 8 6; do NS=$((N - 1)); T=$(awk -v n=$NS 'BEGIN{print 0.25*n}')
  OMP_NUM_THREADS=1 MODEL=XXZ OUT_DIR=outputs/xxz236/rN N_VALUES=$N STEPS=$NS \
    T_VALUES=$T SEEDS="42 43" P_VALUES="0.0 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0" \
    FINAL_VALUES=true BALANCED=true STATE=vector L=2000 MAX_JOBS=4 \
    ./experiments/QELM_Q/exp_QELM_Q.sh
done
uv run python experiments/QELM_Q/plot_bayes_ridge_fss.py outputs/xxz236/r5 \
  --sep outputs/fss233/sep_matched outputs/fss233/sep_paper
```

### Run many runs at once: `MAX_JOBS` (issue #199)

By default the launcher runs one run after the other. `MAX_JOBS=n` puts the
whole grid in a bounded queue (`xargs -P n`): the runs start in the order of
the loops, and a new one starts only when one of the `n` running ones ends,
so there are never more than `n` alive. `JOBS`, the former name, still works
when `MAX_JOBS` is not set.

```bash
OMP_NUM_THREADS=1 MAX_JOBS=6 N_VALUES=12 STATE=vector \
  ./experiments/QELM_Q/exp_QELM_Q.sh
```

- **Memory sets `n`, not the cores.** With the density matrix
  (`STATE=density`, the default), a run at `N = 12` holds `4096 x 4096`
  complex matrices, about 1 to 1.5 GB: on 16 GB of RAM, use at most 6 to 8
  jobs. The state vector (`STATE=vector`) needs far less, and there the
  number of cores is the limit. Set `OMP_NUM_THREADS=1` so that each run
  uses one core.
- **A failed run does not stop the sweep.** A Python error, or a run killed
  by the kernel when the machine runs out of memory (exit 137, `SIGKILL`),
  is written to `<sweep folder>/failed_runs.txt` (`exit <code>: N h t final
  seed p`) and the other runs go on. At the end the launcher prints that
  list and exits with 1; if nothing failed, the file is removed. Rerun only
  the failed points, with a smaller `MAX_JOBS` if they ran out of memory.
- **No two runs write the same file.** Each run writes only into its own
  folder `N<N>_h<h>_t<t>_final<final>_seed<seed>_p<p>/` (`hydra.run.dir`):
  its `.npz`, `metrics.json`, `resolved_config.json`, log and `.hydra/`. The
  only shared file is `failed_runs.txt`, where each failure appends one
  short line, which the system writes in one piece. A grid that lists the
  same value twice (`P_VALUES="0.5 0.5"`) would give two runs one folder,
  so the launcher refuses it.

**On a cluster (Slurm).** The queue maps to a job array: one array task per
line of the grid, and `%n` plays the role of `MAX_JOBS`. A sketch, for one
`N` and `h`, with the grid over `seed x p`:

```bash
#!/usr/bin/env bash
#SBATCH --array=0-104%8          # 5 seeds x 21 p = 105 tasks, 8 at a time
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G                 # per task, N = 12 with the density matrix
SEEDS=(42 43 44 45 46)
P_VALUES=($(seq 0 0.05 1))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 21))]}
p=${P_VALUES[$((SLURM_ARRAY_TASK_ID % 21))]}
OMP_NUM_THREADS=1 N_VALUES=12 H_VALUES=1 SEEDS=$SEED P_VALUES=$p \
  OUT_DIR=outputs/exp_QELM_Q/$SLURM_ARRAY_JOB_ID \
  FAILED_LOG=logs/failed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt \
  ./experiments/QELM_Q/exp_QELM_Q.sh
```

Each task runs the launcher on a grid of one point, into the same
`OUT_DIR`, so the folders land side by side as with `MAX_JOBS` and the
notebooks read them the same way. The tasks share that folder, so each one
gets its own `FAILED_LOG` (by default `<sweep folder>/failed_runs.txt`, which
a task that succeeds removes); Slurm also tracks the failures itself
(`sacct -j <job id>`).

### Bayes ceilings of the accuracy on `<Z_k>` (issue #202)

Every run of `exp_QELM_Q.py` also writes three ceilings of the ridge
`accuracy`, on the same split and the same `<Z_k>`
(`src/QELM/evaluation/bayes_optimal.py`): `bayes_estimate_qda`, a quadratic
discriminant analysis written in NumPy (one Gaussian per charge sector,
checked against scikit-learn in `tests/test_bayes_optimal.py`),
`bayes_estimate_gmm`, the Bayes rule with a mixture of 3 Gaussians per
sector (`GaussianMixtureBayes`, seeded with the seed of the run), and
`bayes_estimate_knn`, `sqrt(train_size)` nearest neighbors weighted by
distance. Where the ridge accuracy falls to 0.5, ceilings at 0.5 mean the
two sectors overlap (decoherence), ceilings well above 0.5 mean the charge
is still in `<Z_k>` but not linearly (learnability transition). At high `p`
the measured qubits split a sector into several clusters, which one
Gaussian blurs: there the QDA can fall below the ridge readout, and the
mixture and the K-NN are the ceilings to read. After a sweep over `N` and
`p`, with 500 samples per run so that these clusters are dense enough to
see,

```bash
N_VALUES="4 6" H_VALUES=1 P_VALUES="0 0.5 1" L=500 ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/plot_bayes_ceiling.py outputs/exp_QELM_Q/<date_time>
```

writes, each with its `.json`:

- `bayes_ceiling.png`: the ridge accuracy against `p` per `N`, with the QDA
  ceiling dashed, the GMM one dotted and the K-NN one dash-dotted in the
  same color (`--no-ceilings` draws the ridge alone);
- on the PCA plane of the test `<Z_k>`, one panel per `(N, p)`, the samples
  colored by sector and: `qda_contours.png`, the 1 and 2 sigma ellipses of
  the QDA Gaussians; `gmm_contours.png`, the contour lines of the mixture
  density of each sector and the 1 sigma ellipse of each component;
  `ridge_regions.png` and `knn_regions.png`, the decision regions of the
  ridge readout and of the K-NN on the plane, with their misclassified test
  samples crossed. The regions are the slice of a classifier of all `Ns`
  coordinates by the plane through the mean of the samples, so a sample far
  from the plane can lie in the region of the other class and still be
  right; the crosses are the true mistakes.

### PCA of the feature spaces `[M]`, `[M, Z]` and `[Z]` (issue #225)

`experiments/QELM_Q/plot_pca.py` shows how the geometry of three feature
spaces of `exp_QELM_Q.py` changes with `p`, for one size (`--N`, default 8):
the measurement record `M` (`features_M`), the record next to the final
`<Z_k>` (`features_M_Z`) and the `<Z_k>` alone (`features_Z`). For each run
the principal axes are fitted on the training samples only
(`QELM.evaluation.pca.fit_pca`, bias column dropped, no standardization) and
the test samples are projected on them (`--samples all` for every sample).
At `p = 0` the record is all zeros: the PCA has no direction at all, and the
panel says so and draws every sample at the origin instead of arbitrary axes.

```bash
N_VALUES=8 P_VALUES="0.0 0.5 1.0" DELTA=200 STATE=vector MAX_JOBS=3 \
    ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/plot_pca.py outputs/exp_QELM_Q/<date_time>
```

writes `pca_feature_spaces.png` and its `.json` (the projected samples of
each panel, the variances, explained shares and rank of its PCA, and the
ridge accuracy of that feature set): one row per feature space, one column
per `p`, colored by charge sector, each panel titled with the ridge accuracy
of its feature set. `--components 3` draws the first three components in 3D
(`pca_feature_spaces_3d.png`). The PCA keeps the directions of largest
variance, not those that separate the sectors: a feature set can be
linearly separable (ridge accuracy 1) along a direction of small variance
that the plane does not show, which is what `[M]` does at `p = 1`, where
the charge is the sum of the outcomes of a round.

Two more figures, from the same command, show where a ridge accuracy of 1
comes from when the PCA plane is a blur. `record_sum_direction.png` draws
`[M]` and `[M, Z]` against the unit vector `u` that sums the record
(`sum_direction`), with the share of the training variance along `u` and
the cosine between `u` and the ridge weights: at `p = 1` every qubit is
measured in every round, each round sums to `Ns - 2 Q`, and the sectors fall
on two vertical lines of a direction holding about 2 % of the variance.
`ridge_outputs.png` draws the histogram of the ridge output `phi . W` on the
test samples, by sector, with the decision threshold: an accuracy of 1 is
two histograms on both sides of the line. The readouts of `[M]` and
`[M, Z]` are refitted from the `.npz` with the `alpha` of the run
(`refit_ridge`), and give the accuracies logged by the run.

### Heatmaps of the ridge accuracy over `(p, h)` (issue #225)

`experiments/QELM_Q/plot_ridge_heatmap.py` draws the test accuracy of the
ridge readout of `Q` over `(p, h)` for one size, one panel per feature space
(`accuracy_M`, `accuracy_M_Z`, `accuracy`), averaged over the seeds, with
the value of each cell and contour lines at 0.6, 0.75 and 0.9. It maps the
wall of the linear readout only, not the Bayes ceilings.

```bash
N_VALUES=8 H_VALUES="0.3 0.5 0.7 1 1.5 2 3 5 7 10" \
    P_VALUES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" \
    DELTA=200 STATE=vector MAX_JOBS=4 ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/plot_ridge_heatmap.py outputs/exp_QELM_Q/<date_time>
```

writes `ridge_heatmap_p_h.png` and its `.json` (`--no-values` leaves the
numbers out of the cells).

`--h`, `--t`, `--final` pick one setting of a sweep that holds several,
`--p` the columns of the PCA figures.

### Is the QDA ceiling enough? QDA against GMM, SVM and K-NN (issue #200)

The QDA models each charge sector by one Gaussian. If the measured
trajectories make a sector multimodal (several bundles of points), the QDA
can miss a separation, and its accuracy is then not the Bayes ceiling.
`experiments/QELM_Q/bayes_benchmark.py` checks it on a sweep of
`exp_QELM_Q.sh`: for every run it reads `<Z_k>` and `Q` back from
`exp_QELM_Q.npz`, keeps the split of the run and fits on it, with the seed of
the run, the QDA, a Bayes classifier with a mixture of 3 and of 5 Gaussians
per sector (`GaussianMixtureBayes`), an SVM with an RBF kernel and the K-NN
(`benchmark_bayes_estimators` in `src/QELM/evaluation/bayes_optimal.py`).

```bash
N_VALUES="8 10" H_VALUES="1 2" SEEDS="42 43" L=300 \
  P_VALUES="0.0 0.1 0.2 0.3 0.5 0.7 1.0" ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/bayes_benchmark.py outputs/exp_QELM_Q/<date_time>
```

writes `bayes_benchmark.json` (accuracy and fit + prediction time of every
estimator per run, and their means per `(N, h)`),
`accuracy_vs_p_bayes_benchmark.pdf` (and `.png`, `.json`: one panel per
`(h, N)`, every estimator against `p` with the ridge readout and chance) and
`time_bayes_benchmark.png` (time of every estimator against `N`).

### The mixture ceiling over the (p, h) plane at N = 8 (issue #215)

`bayes_estimate_gmm` fixes 3 Gaussians per sector. To check that it is a
ceiling everywhere on the plane of the measurement probability `p` and the
field `h`, `gmm_phase_diagram.py` refits every run of one size on its own
split and seed (`src/QELM/evaluation/gmm_scan.py`): the mixture rule for
`K = 1, 2, 3, 4, 6, 8` Gaussians per sector, the mixture whose `K` is chosen
in each sector by BIC (`BICGaussianMixtureBayes`), and the K-NN. It stops if
the refitted `K = 3` and K-NN do not give the logged ceilings.

```bash
N_VALUES=8 H_VALUES="0.5 1 2 5 10" P_VALUES="0 0.1 0.2 0.3 0.5 0.7 1" \
    SEEDS="42 43 44" L=500 ./experiments/QELM_Q/exp_QELM_Q.sh
uv run python experiments/QELM_Q/gmm_phase_diagram.py outputs/exp_QELM_Q/<date_time>
```

writes, averaged over the seeds and each with its `.json`:
`gmm_heatmaps_p_h.png` (maps of the ridge, QDA, BIC mixture and K-NN
accuracies, of the gap mixture minus ridge, the learnability wall, and of
mixture minus K-NN, which should be about 0), `gmm_components_p_h.png` (the
`K` that BIC selects in each sector: how many bundles of trajectories the
data support), `gmm_vs_K.png` (the mixture accuracy against `K`, one panel
per `h`: a ceiling must have saturated in `K`), `gmm_vs_p.png` (every
accuracy against `p`, one panel per `h`) and the table
`gmm_phase_diagram.json`. `--N`, `--t`, `--final` pick one setting of a
sweep that holds several.

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
2. Qubit `i` starts in `R_y(theta_i)|0>` (`QELM.encoding.RotationEncoding`),
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

The state encoding is `double_descent.encoding` (`ENCODING` for the
launcher, issue #196): `rotation_encoding` by default, or one of the
tensor-product encodings of the same paper, `angle_encoding`,
`dense_angle_encoding`, `uniform_bloch_encoding` and `general_encoding`
(`QELM.encoding.ProductEncoding`). The last three put two features on each
qubit, so the image is reduced to `2 N` principal components, each mapped
onto the interval its encoding expects (`[0, pi]` for an angle or a phase,
`[0, 1]` for a population or an amplitude). The notebook runs one sweep
per encoding in `outputs/double_descent_encodings/<date_time>/<encoding>/`,
and `plot_double_descent.py --encodings <that folder>` draws the test MSE
and test error of the QELM against `alpha`, one curve per encoding
(`encodings.png`, `.json` and `encodings_summary.json`):

```bash
ENCODING=dense_angle_encoding ./experiments/double_descent/double_descent.sh
```

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
