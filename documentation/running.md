# Running

From the repository root, after `uv sync`:

```bash
uv run python main.py models=TFIM N=12 evolution.dt=0.05
```

Hydra composes `conf/config.yaml` and applies those overrides:

- `models=TFIM` selects `conf/models/TFIM.yaml` and `TransverseFieldIsingModel`.
- `N=12` overrides `N` on the TFIM block. `J` and `h` stay at the values in that YAML file.
- `evolution.dt=0.05` overrides the evolution step. `evolution.n_steps` keeps its formula `4 * TFIM.N` from `conf/models/TFIM.yaml`, so it becomes 48.

The process prints two lines: the selection (`models`, each key in the simulation block, and each `evolution` key), then the pipeline status. A checkpoint is written under the Hydra run directory. See [Checkpoints and local data](checkpoints.md).

Other overrides use the same form:

```bash
uv run python main.py models=kitaev training/regression=linear
uv run python main.py training/regression=ridge evaluation=default
uv run python main.py seed=0
uv run python main.py checkpoint.dir=/tmp/qrc-run checkpoint.use_uuid=true
```

`training/regression=ridge` and `evaluation=default` match the defaults in `conf/config.yaml`. `training/regression=linear` selects `LinearReadout`. `evaluation=default` is the only registered evaluator. An unknown `models`, trainer name, or evaluator name raises `KeyError`.

## Notebook

In a notebook, `load_config` composes the same config as the command line. It finds `conf/` from the package location, so it works from any folder of `notebooks/`, and it clears Hydra's global state, so the cell can be re-run:

```python
from QELM import TransverseFieldIsingModel, load_config, set_seed

cfg = load_config(["models=TFIM", "N=6"])
set_seed(cfg)
model = TransverseFieldIsingModel.from_config(cfg)
```

From the repository root, run `uv sync`, then open a notebook of `notebooks/benchmarks/` (reproductions of published figures) or `notebooks/results/` (new results); `notebooks/README.md` lists them. Each notebook saves its figures, with the data they show, in the output folder of the sweep it ran (`outputs/<experiment>/<date_time>/`, gitignored), through `QELM.utils.plots.save_figure`.

### Experimenting without blocking `git pull`

The notebooks in `notebooks/` are templates tracked by git. To experiment, copy one into `notebooks/scratch/`, which is gitignored, and edit the copy:

```bash
mkdir -p notebooks/scratch
cp notebooks/results/01_TFIM_reservoir.ipynb notebooks/scratch/
```

`git pull` never touches that folder, so there is nothing to stash. Code changes in `src/` go on their own branch and become a pull request; `main` stays clean.

The TFIM notebook starts with `%load_ext autoreload` and `%autoreload 2`. `uv sync` installs `QELM` in editable mode, so after `git pull && uv sync` the running kernel picks up the new code in `src/` without a restart.

## Experiment scripts

Standalone experiment scripts start from `experiments/example/run_experiment.py`. It composes `conf/config.yaml` with Hydra like `main.py`, accepts the same overrides, and seeds NumPy from `seed`:

```bash
uv run python experiments/example/run_experiment.py models=TFIM N=6
```

Write temporary outputs (arrays, logs, plots) to `experiments/scratch/`, available in the script as `SCRATCH`. Git ignores everything in that folder except `.gitkeep`, so nothing written there is committed. Copy the script to start a new experiment and keep the template unchanged.

### Bash scripts

Each folder of `experiments/` holds an experiment and the bash script that launches its runs from the terminal. The test experiment sweeps `TFIM.N` and `seed` through `main.py` and writes each run to its own folder under `outputs/`, which git ignores:

```bash
./experiments/example/test_experiment.sh
```

Copy it to write your own sweep. [`experiments/README.md`](../experiments/README.md) explains how.
