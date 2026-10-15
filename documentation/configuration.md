# Configuration

Scientific parameters and secrets live in different files.

## Hydra YAML

`main.py` loads `conf/config.yaml` with Hydra (`version_base="1.3"`). Command-line overrides are applied on top. `hydra.job.chdir` is `false`, so the process stays in the repository root.

`conf/config.yaml` selects the groups and holds the shared settings:

| Key | Role |
| --- | --- |
| `defaults` | `sim_type: TFIM`, `training: ridge`, `evaluation: default` |
| `seed` | Integer passed to NumPy at startup. See [Code quality](code-quality.md). |
| `ancilla_charge_sector.N` | Total qubit count of the initial state, `${${sim_type}.N}`: it follows the selected Hamiltonian. Read by `AncillaChargeSectorEmbedding.from_config(cfg)` |
| `samples.L`, `samples.train_fraction`, `samples.train_size` | Number of samples and training split, `train_size = int(0.7 * L)`. Not read by the pipeline yet |
| `single_observable.observables`, `double_observable.pairs` | Features measured on the final state: `<O_i>` for each label (default `[Z]`), `<A_i B_j>` for each pair (default `[[Z, Z]]`). Several entries allowed; `[]` or a missing block gives no feature. Read by `SingleObservable.from_config(cfg)` / `DoubleObservable.from_config(cfg)` |
| `dynamic_readout.p` | Dynamic readout parameter `p`. Not read by the pipeline yet |
| `projective_measurement.qubits` | Sites measured by `ProjectiveMeasurement`, in order; `null` (default) measures every qubit. Read by `ProjectiveMeasurement.from_config(cfg)`, which also works without the block |
| `checkpoint` | Where the run writes its `.npz`. See [Checkpoints and local data](checkpoints.md). |
| `hydra.run.dir` | `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` |

Group files:

| File | Contents |
| --- | --- |
| `conf/sim_type/TFIM.yaml` | `sim_type: TFIM`, the `TFIM` block (`N`, `J`, `h = 10 J`, `RUNS` random Hamiltonians) and the evolution block `evolution` (`t = 10 / J`, `n_steps = d = 4 N`, `dt = t / d`) |
| `conf/sim_type/kitaev.yaml` | `sim_type: kitaev`, a `kitaev` block with only `N`, and a fixed `evolution` block (`dt: 0.01`, `n_steps: 100`) |
| `conf/training/ridge.yaml` | `name: ridge` |
| `conf/training/readout.yaml` | `name: readout` |
| `conf/evaluation/default.yaml` | `name: default`, a list of metric names, and a plot name |

`evolution.dt` and `evolution.n_steps` are recorded by `evolution_request` in the pipeline. `TFIM.RUNS` is not read by the code yet.

## Derived values

Some values are computed from others with three resolvers registered by `import QELM` (`src/QELM/resolvers.py`): `${mul:a,b,...}`, `${div:a,b}` and `${int:x}`. For example `h: ${mul:10.0,${TFIM.J}}`. A derived value follows its source, so `TFIM.J=2` gives `h=20`, `t=5` and a smaller `dt`, and `TFIM.N=12` gives `n_steps=48`. It can still be overridden directly (`TFIM.h=3`, `evolution.dt=0.05`); the other derived values then keep their formula, so after `evolution.dt=0.05` the product `n_steps * dt` no longer equals `t`.

`N`, `J`, `dt`, and `seed` are config values. They are not constants in the Python package. Change them in YAML or with a CLI override. See [Running](running.md).

## `.env` secrets

API keys and machine-local paths belong in `.env`. That file is gitignored. Copy the example and fill it in locally:

```bash
cp .env.example .env
```

`.env.example` only lists `WANDB_API_KEY`. `main.py` calls `load_secrets()`, which runs `load_dotenv()`. A missing `.env` is ignored. The pipeline does not read the key and does not upload a run.
