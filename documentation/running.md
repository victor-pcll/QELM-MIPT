# Running

From the repository root, after `uv sync`:

```bash
uv run python main.py models=TFIM TFIM.N=12 evolution.dt=0.05
```

Hydra composes `conf/config.yaml` and applies those overrides:

- `models=TFIM` selects `conf/models/TFIM.yaml` and `TransverseFieldIsingModel`.
- `TFIM.N=12` overrides `N` on the TFIM block. `J` and `h` stay at the values in that YAML file.
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

In a notebook, `load_config` composes the same config as the command line. It finds `conf/` from the package location, so it works from `notebooks/`, and it clears Hydra's global state, so the cell can be re-run:

```python
from QELM import TransverseFieldIsingModel, load_config, set_seed

cfg = load_config(["models=TFIM", "TFIM.N=6"])
set_seed(cfg)
model = TransverseFieldIsingModel.from_config(cfg)
```

From the repository root, run `uv sync`, then open `notebooks/series.ipynb`. When `data/series.npy` is a 1D or 2D numeric array, the notebook writes `outputs/series.png`. Both paths are gitignored. That figure is the local file. The pipeline still records the evolution as a scaffold, so that figure is not a plot of the dynamics.

### Experimenting without blocking `git pull`

The notebooks in `notebooks/` are templates tracked by git. To experiment, copy one into `notebooks/scratch/`, which is gitignored, and edit the copy:

```bash
mkdir -p notebooks/scratch
cp notebooks/01-QELM-MIPT.ipynb notebooks/scratch/
```

`git pull` never touches that folder, so there is nothing to stash. Code changes in `src/` go on their own branch and become a pull request; `main` stays clean.

The template notebooks start with `%load_ext autoreload` and `%autoreload 2`. `uv sync` installs `QELM` in editable mode, so after `git pull && uv sync` the running kernel picks up the new code in `src/` without a restart.

## Experiment scripts

Standalone experiment scripts start from `experiments/run_experiment.py`. It composes `conf/config.yaml` with Hydra like `main.py`, accepts the same overrides, and seeds NumPy from `seed`:

```bash
uv run python experiments/run_experiment.py models=TFIM TFIM.N=6
```

Write temporary outputs (arrays, logs, plots) to `experiments/scratch/`, available in the script as `SCRATCH`. Git ignores everything in that folder except `.gitkeep`, so nothing written there is committed. Copy the script to start a new experiment and keep the template unchanged.
