# Layout

| Path | Role |
| --- | --- |
| `conf/` | Hydra YAML: `config.yaml` plus groups `sim_type`, `training`, and `evaluation` |
| `main.py` | Entrypoint. Seeds NumPy, loads `.env`, prints the selection, and runs the pipeline |
| `src/QELM/models/` | `TransverseFieldIsingModel` (dense TFIM Hamiltonian), the `KitaevModel` scaffold, and the time-evolution stub |
| `src/QELM/training/` | Trainer scaffolds (`RidgeRegression`, `Readout`) |
| `src/QELM/evaluation/` | Metric and plot scaffolds |
| `src/QELM/pipeline.py` | Routes sim type to a model, a trainer, and an evaluator, then saves a checkpoint |
| `src/QELM/checkpoint.py` | Writes the `.npz` checkpoint |
| `src/QELM/seed.py` | Reads `seed` from the composed config. Seeds NumPy and builds the generators for disorder realisations |
| `src/QELM/selection.py` | Loads `.env` and formats the selection line |
| `data/` | Local datasets. Gitignored, and not created by the package |
| `notebooks/series.ipynb` | Plots a local `data/series.npy` into `outputs/series.png` |
| `tests/` | pytest |

Import the library as `QELM`.

## What one run does

`run_pipeline` resolves names, then calls the selected pieces:

1. `cfg.sim_type` selects `TransverseFieldIsingModel` (`TFIM`) or `KitaevModel` (`kitaev`). `from_config` copies the scalar keys in that simulation block.
2. `cfg.training.name` selects `RidgeRegression` (`ridge`) or `Readout` (`readout`). `from_config` reads `learning_rate` from the simulation block when that key exists, and leaves it unset when it does not (the Kitaev block has no keys).
3. `model.evolve` records `dt` and `n_steps` from `cfg.test` and sets `status` to `scaffold`. It does not allocate a trajectory. `hamiltonian` builds the dense TFIM matrix of shape `(2**N, 2**N)` but the pipeline does not call it yet; the Kitaev scaffold raises `NotImplementedError`.
4. `trainer.fit(features=None, targets=None)` returns `"scaffold"`. No weight vector is estimated. `state_dict` stores a length-0 `float64` array under `weights`. `predict` is the same kind of stub.
5. The `default` evaluator lists the metric names and plot name from `cfg.evaluation` and returns `status=scaffold`. It does not score predictions or write an image.
6. `save_checkpoint` writes the archive described in [Checkpoints and local data](checkpoints.md).

The printed pipeline line names the model, the trainer, `dt`, `n_steps`, the evaluator report, and the checkpoint path.
