# Layout

| Path | Role |
| --- | --- |
| `conf/` | Hydra YAML: `config.yaml` plus groups `models`, `training/measurements`, `training/readout`, `training/regression`, and `evaluation`, mirroring `src/QELM/` |
| `main.py` | Entrypoint. Seeds NumPy, loads `.env`, prints the selection, and runs the pipeline |
| `src/QELM/models/` | `TransverseFieldIsingModel` (dense TFIM Hamiltonian), the `KitaevModel` scaffold, and the time-evolution stub |
| `src/QELM/physics/` | Quantum primitives only (`operators.py`): Pauli matrices `I, X, Y, Z`, `pauli(label)`, `get_site_operator(op, site, N, sparse=False)` (dense array by default, CSR with `sparse=True`) and `qubit_counts(rho, Nr=0)`, which returns `(N, Ns, Nr)`; `partial_trace.py`: `partial_trace(state, keep)`, the reduced density matrix `rho_A = Tr_B(rho)` of the qubits `keep` (in that order), from a state vector (via its amplitude matrix, without forming `rho`), a dense or a sparse `rho` |
| `src/QELM/embedding/` | Initial-state embeddings. An embedding subclasses `Embedding` (`base.py`): `N` is the total qubit count, so `state(x)` returns `psi` of shape `(2**N,)` (or a CSR column of shape `(2**N, 1)` with `sparse=True`; `density_matrix(x, sparse=True)` likewise returns a CSR matrix) and `density_matrix(x)`, or its short name `rho(x, sparse=False)`, returns `rho`, like the models built with the same `N`. Ancillas are the first sites (most significant bits), before the `Ns = N - Nr` system qubits. `AncillaChargeSectorEmbedding` prepares `(|0>_R|Q=Ns//2>_S + |1>_R|Q=Ns//2-1>_S)/sqrt(2)` with one ancilla and `Ns = N - 1`; `get_embedding(name)` returns the class for a config key |
| `src/QELM/training/` | The QELM data pipeline, one subpackage per domain; in each, `base.py` holds the parent class and every child class has its own module. Everything is importable from `QELM.training`. `measurements/`: `base.py` `Readout`, whose `measure(rho, p=None, rng=None)` returns `ReadoutResult(outcomes, rho)` with the post-measurement state (each target qubit is measured with probability `p`, outcome `nan` when not measured); `projective.py` `ProjectiveMeasurement`, which reads `p` from its config block when `measure` gets none (1.0 if the block has none); `get_readout(name)` returns the class for a config key. `readout/`: features computed from the final state without disturbing it, `base.py` `Observables`, `single.py` `SingleObservable`, `double.py` `DoubleObservable`, which return `Tr(rho O_k)`. `dataset.py`: `create_qelm_dataset(M, X, y)` builds `QELMDataset(phi, y)` with `phi = [M || X]` (mid-circuit outcomes then final readout of S) and `y` the ancilla outcome; `split_dataset(ds, train_fraction)` gives disjoint train and test sets. `regression/`: `base.py` `Trainer`, `linear.py` `LinearReadout` (`linear`), `ridge.py` `RidgeReadout` (`ridge`) |
| `src/QELM/evaluation/` | Post-training analysis only: metrics (`metrics.py`: `MSE`, `NMSE`, `R2` (coefficient of determination); all inputs are checked to be 1-D, non-empty and of the same shape; `von_neumann_entropy(rho, N)`, the entanglement entropy of the ancilla qubit 0, in bits by default, for a dense or sparse `rho`), entanglement entropies of any subsystem `A` (`entropy.py`, issue #93: `renyi_entropy(state, n, sector_A)` for any order `n >= 0` including `0`, `1`, `2` and `inf`, `von_neumann_entropy(state, sector_A)` and `entanglement_spectrum(state, sector_A)`, in bits by default, for a state vector, a dense or a sparse `rho`; importable from `QELM.evaluation`) and plots (`plots.py`, scaffold) |
| `src/QELM/utils/` | Helpers shared by the experiment scripts. `io.py`: `save_results(results, out_dir, exp_name, features="X")` writes the arrays to `<exp_name>.npz` and the scores to `metrics.json`; each script passes `Path(__file__).stem`, so two experiments never share a file name |
| `src/QELM/pipeline.py` | Routes the `models` key to a model, a trainer, and an evaluator, then saves a checkpoint |
| `src/QELM/checkpoint.py` | Writes the `.npz` checkpoint |
| `src/QELM/config.py` | `load_config` composes `conf/` with overrides, from any working directory |
| `src/QELM/resolvers.py` | `mul`, `div` and `int` resolvers for derived values in `conf/` |
| `src/QELM/seed.py` | Reads `seed` from the composed config |
| `src/QELM/selection.py` | Loads `.env` and formats the selection line |
| `data/` | Local datasets. Gitignored, and not created by the package |
| `notebooks/series.ipynb` | Plots a local `data/series.npy` into `outputs/series.png` |
| `experiments/` | One folder per experiment, each holding its Python script and the bash launcher of its sweep: `example/` (`run_experiment.py`, an executable Hydra template that composes `conf/config.yaml` and seeds NumPy, and `test_experiment.sh`), `projective/` (projective readout), `QELM_Z/` (QELM on the final observables), `QELM_M/` (QELM on the measurement record). `scratch/` is where experiments write temporary outputs; everything in it except `.gitkeep` is gitignored |
| `tests/` | pytest |

Import the library as `QELM`.

## What one run does

`run_pipeline` resolves names, then calls the selected pieces:

1. `cfg.models` selects `TransverseFieldIsingModel` (`TFIM`) or `KitaevModel` (`kitaev`). `from_config` copies the scalar keys in that simulation block.
2. `cfg.training.name` selects `RidgeReadout` (`ridge`) or `LinearReadout` (`linear`). `RidgeReadout` reads `alpha` from `conf/training/regression/ridge.yaml` (default 1.0) and solves `(Phi^T Phi + alpha I) W = Phi^T y`; `fit(phi, y, alpha=...)` overrides `alpha`.
3. `evolution_request(model, dt=..., n_steps=...)` (in `QELM.models.base`) records `dt` and `n_steps` from `cfg.evolution` and sets `status` to `scaffold`. It does not allocate a trajectory. `hamiltonian` builds the dense TFIM matrix of shape `(2**N, 2**N)` but the pipeline does not call it yet; the Kitaev scaffold raises `NotImplementedError`. `HamiltonianModel.propagator(dt=...)` returns `exp(-i H dt)`, and `model.evolve(state, dt=None)` applies it without forming `U` (with no `dt`, it uses `model.dt`, which `from_config` sets from `evolution.dt`): a state vector `psi` of shape `(dim,)` comes back as `U psi`, a density matrix `rho` of shape `(dim, dim)` as `U rho U^dagger`, and any other shape raises `ValueError`. The pipeline does not call `evolve` yet.
4. `trainer.fit(features=None, targets=None)` returns `"scaffold"`, since the pipeline does not build a dataset yet. No weight vector is estimated, so `state_dict` stores a length-0 `float64` array under `weights`. With real features, `RidgeReadout.fit` returns `"fitted"`, `predict` returns `Phi @ W` and `state_dict` stores `W`.
5. The `default` evaluator lists the metric names and plot name from `cfg.evaluation` and returns `status=scaffold`. It does not score predictions or write an image.
6. `save_checkpoint` writes the archive described in [Checkpoints and local data](checkpoints.md).

The printed pipeline line names the model, the trainer, `dt`, `n_steps`, the evaluator report, and the checkpoint path.
