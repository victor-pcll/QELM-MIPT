# Configuration

Scientific parameters and secrets live in different files.

## Hydra YAML

`main.py` loads `conf/config.yaml` with Hydra (`version_base="1.3"`). Command-line overrides are applied on top. `hydra.job.chdir` is `false`, so the process stays in the repository root.

`conf/config.yaml` selects the groups and holds the shared settings:

| Key | Role |
| --- | --- |
| `defaults` | `sim_type: TFIM`, `training: ridge`, `evaluation: default` |
| `seed` | Integer passed to NumPy at startup, and to `make_rng` / `spawn_rngs` for disorder realisations. See [Code quality](code-quality.md#disorder-realisations). |
| `test.dt`, `test.n_steps` | Scalar step size and step count passed to `evolve` |
| `checkpoint` | Where the run writes its `.npz`. See [Checkpoints and local data](checkpoints.md). |
| `hydra.run.dir` | `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` |

Group files:

| File | Contents |
| --- | --- |
| `conf/sim_type/TFIM.yaml` | `sim_type: TFIM` and the `TFIM` block: `N`, `J`, `h`, `learning_rate` |
| `conf/sim_type/kitaev.yaml` | `sim_type: kitaev` and a `kitaev` block with only `N`. No other Hamiltonian parameters are defined there |
| `conf/training/ridge.yaml` | `name: ridge` |
| `conf/training/readout.yaml` | `name: readout` |
| `conf/evaluation/default.yaml` | `name: default`, a list of metric names, and a plot name |

`N`, `J`, `learning_rate`, `dt`, and `seed` are config values. They are not constants in the Python package. Change them in YAML or with a CLI override. See [Running](running.md).

## `.env` secrets

API keys and machine-local paths belong in `.env`. That file is gitignored. Copy the example and fill it in locally:

```bash
cp .env.example .env
```

`.env.example` only lists `WANDB_API_KEY`. `main.py` calls `load_secrets()`, which runs `load_dotenv()`. A missing `.env` is ignored. The pipeline does not read the key and does not upload a run.
