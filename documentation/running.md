# Running

From the repository root, after `uv sync`:

```bash
uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
```

Hydra composes `conf/config.yaml` and applies those overrides:

- `sim_type=ising` selects `conf/sim_type/ising.yaml` and `TransverseFieldIsingModel`.
- `ising.N=12` overrides `N` on the Ising block. `J`, `h` and `learning_rate` stay at the values in that YAML file.
- `test.dt=0.05` overrides the test step. `test.n_steps` stays at the value in `conf/config.yaml`.

The process prints two lines: the selection (`sim_type`, each key in the simulation block, and each `test` key), then the pipeline status. A checkpoint is written under the Hydra run directory. See [Checkpoints and local data](checkpoints.md).

Other overrides use the same form:

```bash
uv run python main.py sim_type=kitaev training=readout
uv run python main.py training=ridge evaluation=default
uv run python main.py seed=0
uv run python main.py checkpoint.dir=/tmp/qrc-run checkpoint.use_uuid=true
```

`training=ridge` and `evaluation=default` match the defaults in `conf/config.yaml`. `training=readout` selects `Readout`. `evaluation=default` is the only registered evaluator. An unknown `sim_type`, trainer name, or evaluator name raises `KeyError`.

## Notebook

From the repository root, run `uv sync`, then open `notebooks/series.ipynb`. When `data/series.npy` is a 1D or 2D numeric array, the notebook writes `outputs/series.png`. Both paths are gitignored. That figure is the local file. `evolve` stays a scaffold and is not a plot of the dynamics.
