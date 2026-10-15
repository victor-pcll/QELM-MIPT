# Experiment scripts

Bash scripts that launch QELM runs from the terminal. Each script fixes a set of
parameters, calls the Hydra entry point once per combination, and writes every
run to its own folder under `outputs/`, which git ignores (see `.gitignore`).

## Run the test experiment

From the repository root, after `uv sync`:

```bash
./experiments/scripts/test_experiment.sh
```

It runs `main.py` on the TFIM for `N` in `4 6` and `seed` in `0 1`, four runs in
a few seconds. Each run lands in
`outputs/test_experiment/<date_time>/N<N>_seed<seed>/` with:

- `checkpoint.npz`: the checkpoint written by the pipeline;
- `main.log`: the run log;
- `.hydra/config.yaml`: the full composed config, so the run can be reproduced;
- `.hydra/overrides.yaml`: the overrides given on the command line.

Set `OUT_ROOT` to write elsewhere: `OUT_ROOT=/tmp/qelm ./experiments/scripts/test_experiment.sh`.

## Write your own

1. Copy the test script and make the copy executable:

   ```bash
   cp experiments/scripts/test_experiment.sh experiments/scripts/my_sweep.sh
   chmod +x experiments/scripts/my_sweep.sh
   ```

2. Edit the `Parameters` block: `EXPERIMENT` names the output folder, and the
   bash arrays (`N_VALUES=(4 6 8)`) hold the values to sweep. Add a loop for any
   new swept parameter.
3. Edit the overrides passed to `uv run`. Any key in `conf/` takes a
   `key=value` override, as on the command line, for example
   `"training/regression=linear"`, `"TFIM.J=2.0"` or
   `"projective_measurement.p=0.5"`. Group choices use `/`, values use `.`.
4. To run your own experiment logic instead of the pipeline, copy
   `experiments/run_experiment.py`, put the logic in its `main`, and call that
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
