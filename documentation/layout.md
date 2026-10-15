# Layout

| Path | Role |
| --- | --- |
| `conf/` | Hydra YAML: `config.yaml` plus groups `sim_type`, `training`, and `evaluation` |
| `main.py` | Entrypoint. Seeds NumPy, loads `.env`, prints the selection, and runs the pipeline |
| `src/QELM/models/` | `TransverseFieldIsingModel` (dense TFIM Hamiltonian), the `KitaevModel` scaffold, and the time-evolution stub |
| `src/QELM/physics/` | Quantum primitives only (`operators.py`): Pauli matrices `I, X, Y, Z`, `pauli(label)`, `get_site_operator(op, site, N, sparse=False)` (dense array by default, CSR with `sparse=True`) and `n_qubits(rho, n_ancilla=0)`, which returns `(N, N_S, n_ancilla)` |
| `src/QELM/embedding/` | Initial-state embeddings. An embedding subclasses `Embedding` (`base.py`): `N` is the total qubit count, so `state(x)` returns `psi` of shape `(2**N,)` (or a CSR column of shape `(2**N, 1)` with `sparse=True`; `density_matrix(x, sparse=True)` likewise returns a CSR matrix) and `density_matrix(x)` returns `rho`, like the models built with the same `N`. Ancillas are the first sites (most significant bits), before the `N_S = N - n_ancilla` system qubits. `AncillaChargeSectorEmbedding` prepares `(|0>_R|Q=N_S//2>_S + |1>_R|Q=N_S//2-1>_S)/sqrt(2)` with one ancilla and `N_S = N - 1`; `get_embedding(name)` returns the class for a config key |
| `src/QELM/training/` | The QELM data pipeline, one subpackage per domain; in each, `base.py` holds the parent class and every child class has its own module. Everything is importable from `QELM.training`. `measurements/`: `base.py` `Readout`, whose `measure(rho, p=1.0, rng=None)` returns `ReadoutResult(outcomes, rho)` with the post-measurement state (each target qubit is measured with probability `p`, outcome `nan` when not measured); `projective.py` `ProjectiveMeasurement`; `get_readout(name)` returns the class for a config key. `readout/`: features computed from the final state without disturbing it, `base.py` `Observables`, `single.py` `SingleObservable`, `double.py` `DoubleObservable`, which return `Tr(rho O_k)`. `regression/`: `base.py` `Trainer`, `linear.py` `LinearReadout` (`readout`), `ridge.py` `RidgeRegression` (`ridge`) |
| `src/QELM/evaluation/` | Post-training analysis only: metrics (`metrics.py`: `MSE`, `NMSE`, `R2` (coefficient of determination); all inputs are checked to be 1-D, non-empty and of the same shape) and plots (`plots.py`, scaffold) |
| `src/QELM/pipeline.py` | Routes sim type to a model, a trainer, and an evaluator, then saves a checkpoint |
| `src/QELM/checkpoint.py` | Writes the `.npz` checkpoint |
| `src/QELM/config.py` | `load_config` composes `conf/` with overrides, from any working directory |
| `src/QELM/resolvers.py` | `mul`, `div` and `int` resolvers for derived values in `conf/` |
| `src/QELM/seed.py` | Reads `seed` from the composed config |
| `src/QELM/selection.py` | Loads `.env` and formats the selection line |
| `data/` | Local datasets. Gitignored, and not created by the package |
| `notebooks/series.ipynb` | Plots a local `data/series.npy` into `outputs/series.png` |
| `tests/` | pytest |

Import the library as `QELM`.

## What one run does

`run_pipeline` resolves names, then calls the selected pieces:

1. `cfg.sim_type` selects `TransverseFieldIsingModel` (`TFIM`) or `KitaevModel` (`kitaev`). `from_config` copies the scalar keys in that simulation block.
2. `cfg.training.name` selects `RidgeRegression` (`ridge`) or `LinearReadout` (`readout`). The trainers take no config parameter: ridge regression will use the closed-form solution.
3. `evolution_request(model, dt=..., n_steps=...)` (in `QELM.models.base`) records `dt` and `n_steps` from `cfg.evolution` and sets `status` to `scaffold`. It does not allocate a trajectory. `hamiltonian` builds the dense TFIM matrix of shape `(2**N, 2**N)` but the pipeline does not call it yet; the Kitaev scaffold raises `NotImplementedError`. `HamiltonianModel.propagator(dt=...)` returns `exp(-i H dt)`, and `model.evolve(state, dt=None)` applies it without forming `U` (with no `dt`, it uses `model.dt`, which `from_config` sets from `evolution.dt`): a state vector `psi` of shape `(dim,)` comes back as `U psi`, a density matrix `rho` of shape `(dim, dim)` as `U rho U^dagger`, and any other shape raises `ValueError`. The pipeline does not call `evolve` yet.
4. `trainer.fit(features=None, targets=None)` returns `"scaffold"`. No weight vector is estimated. `state_dict` stores a length-0 `float64` array under `weights`. `predict` is the same kind of stub.
5. The `default` evaluator lists the metric names and plot name from `cfg.evaluation` and returns `status=scaffold`. It does not score predictions or write an image.
6. `save_checkpoint` writes the archive described in [Checkpoints and local data](checkpoints.md).

The printed pipeline line names the model, the trainer, `dt`, `n_steps`, the evaluator report, and the checkpoint path.
