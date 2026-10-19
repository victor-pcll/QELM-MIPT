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
`training.alpha=1e-3` (`L=1000 ALPHA=1e-2 ./experiments/QELM_Z/exp_QELM_Z.sh`
to change them). No measurement during the evolution: the readout learns the
ancilla `<Z_0>` from the system observables `<Z_k>, <Y_k>, <X_k>` after
`U(t)`. Each run lands in `outputs/exp_QELM_Z/<date_time>/N<N>_seed<seed>/` with
`exp_QELM_Z.npz` (`X`, `Y`, `y_test`, `y_pred`, `W`), `metrics.json` (NMSE, R2)
and `.hydra/config.yaml`.

## Run the QELM experiment on the measurement record (work in progress)

```bash
./experiments/QELM_M/exp_QELM_M.sh
```

It runs `experiments/QELM_M/exp_QELM_M.py` for `N=4`, `seed=42`, 12 samples
and `projective_measurement.p` in `1.0 0.5`. The evolution time is split into
`experiment.n_measurements` steps (default `N - 1`), and the system qubits are
measured after each step. The features are this measurement record `M`, of
shape `(n_measurements, N - 1)` per sample, instead of the observables; an
unmeasured qubit (`p < 1`) counts as 0.
Each run lands in `outputs/exp_QELM_M/<date_time>/N<N>_seed<seed>_p<p>/`.

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
uv run python main.py -m TFIM.N=4,6 seed=0,1 \
  hydra.sweep.dir=outputs/multirun
```

Each combination goes to a numbered subfolder of `hydra.sweep.dir`. The bash
loop is easier to read and to extend (for example to chain a plotting step
after the runs), which is why the scripts here use it.
