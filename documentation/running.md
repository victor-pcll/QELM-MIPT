# Running

From the repository root, after `uv sync`:

```bash
uv run python main.py sim_type=TFIM TFIM.N=12 test.dt=0.05
```

Hydra composes `conf/config.yaml` and applies those overrides:

- `sim_type=TFIM` selects `conf/sim_type/TFIM.yaml` and `TransverseFieldIsingModel`.
- `TFIM.N=12` overrides `N` on the TFIM block. `J`, `h` and `learning_rate` stay at the values in that YAML file.
- `test.dt=0.05` overrides the test step. `test.n_steps` keeps its formula `4 * TFIM.N` from `conf/sim_type/TFIM.yaml`, so it becomes 48.

The process prints two lines: the selection (`sim_type`, each key in the simulation block, and each `test` key), then the pipeline status. A checkpoint is written under the Hydra run directory. See [Checkpoints and local data](checkpoints.md).

Other overrides use the same form:

```bash
uv run python main.py sim_type=kitaev training=readout
uv run python main.py training=ridge evaluation=default
uv run python main.py seed=0
uv run python main.py checkpoint.dir=/tmp/qrc-run checkpoint.use_uuid=true
```

`training=ridge` and `evaluation=default` match the defaults in `conf/config.yaml`. `training=readout` selects `LinearReadout`. `evaluation=default` is the only registered evaluator. An unknown `sim_type`, trainer name, or evaluator name raises `KeyError`.

## Notebook

In a notebook, `load_config` composes the same config as the command line. It finds `conf/` from the package location, so it works from `notebooks/`, and it clears Hydra's global state, so the cell can be re-run:

```python
from QELM import TransverseFieldIsingModel, load_config, set_seed

cfg = load_config(["sim_type=TFIM", "TFIM.N=6"])
set_seed(cfg)
model = TransverseFieldIsingModel.from_config(cfg)
```

From the repository root, run `uv sync`, then open `notebooks/series.ipynb`. When `data/series.npy` is a 1D or 2D numeric array, the notebook writes `outputs/series.png`. Both paths are gitignored. That figure is the local file. `evolve` stays a scaffold and is not a plot of the dynamics.
