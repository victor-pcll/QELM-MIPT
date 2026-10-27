# QELM-MIPT

A Python library managed with [uv](https://docs.astral.sh/uv/). Requires Python 3.12 or newer.

Longer notes are in [`documentation/`](documentation/README.md): setup, Hydra and `.env`, the project layout, checkpoints, and code quality.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other install options are in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Set up

```bash
uv sync
```

## Layout

| Path | Role |
| --- | --- |
| `conf/` | Hydra YAML: `config.yaml` plus groups `models`, `training/{measurements,readout,regression}`, `evaluation`, mirroring `src/QELM/` |
| `data/` | Local time series and training datasets (gitignored); MNIST is downloaded to `data/mnist/` on first use |
| `src/QELM/models/` | Hamiltonians and time evolution (transverse-field Ising model, Kitaev scaffold), random brickwork circuits in the package `circuits/` (issue #189): `base.py` (`BrickworkCircuit`, gate application), `haar.py` (U(1) and Haar gates), `clifford.py` (Clifford gates) |
| `src/QELM/physics/` | Quantum primitives: Pauli matrices and site operators |
| `src/QELM/encoding/` | Initial-state encodings (renamed from `embedding/`, issue #197) (`Encoding` base, ancilla + charge-sector state, angle encoding `RotationEncoding` of a static input, and the product encodings of issue #196: `AngleEncoding`, `DenseAngleEncoding`, `UniformBlochEncoding`, `GeneralEncoding`) |
| `src/QELM/datasets/` | Static classification datasets: MNIST and scikit-learn's 8x8 digits as a task between a few chosen digits, and their PCA to rotation angles (`AngleEncoder`) |
| `src/QELM/training/` | Measurements, feature extraction and readout fitting (`LinearReadout` scaffold, `RidgeReadout` closed-form ridge regression, `alpha = 0` for the unregularized minimum-norm fit) |
| `src/QELM/evaluation/` | Metrics, entropies, the learnability and charge-sharpening statistics, the double-descent sweep (`double_descent.py`), plot scaffolds |
| `src/QELM/utils/plots/` | Plotting helpers shared by the scripts and the notebooks: one style (no grid, closed box, large titles), sweep curves, reference overlays, `save_figure` (figure and its data in the run's output folder) |
| `notebooks/benchmarks/` | Benchmark notebooks: reproductions of published figures (learnability, Barratt et al. 2022, Fig. 2b and 2c; charge sharpening, Agrawal et al. 2024, Fig. 7) |
| `notebooks/results/` | Notebooks producing new results (TFIM reservoir, QELM on the measurement record M, double descent on MNIST) |
| `experiments/` | One folder per experiment, each with its Python script and its bash launcher: `example/` (the `run_experiment.py` template and `test_experiment.sh`), `projective/`, `QELM_Z/`, `QELM_M/`, `QELM_Q/`, `charge_sharpening/` (benchmark against Agrawal et al. 2024, Fig. 7), `double_descent/` (double descent of the QELM on MNIST digits, with confusion matrices); `scratch/` for local outputs (gitignored) |
| `main.py` | Entrypoint. Routes `models` → model → training → evaluation |

## Run the pipeline

Scientific parameters live in `conf/` and can be overridden from the command line. Secrets and machine-local paths (for example `WANDB_API_KEY`) belong in `.env`, not in YAML. Copy `.env.example` to `.env` for local secrets.

```bash
cp .env.example .env
uv run python main.py models=TFIM N=12 evolution.dt=0.05
```

`models=TFIM` selects `conf/models/TFIM.yaml` and the transverse-field Ising model. `training/regression=ridge` (default) or `training/regression=linear` selects the trainer. `N`, `J`, and `dt` stay in YAML or CLI overrides.

## Notebooks

From the repository root:

```bash
uv sync
```

Then open a notebook of `notebooks/benchmarks/` or `notebooks/results/` (see `notebooks/README.md`). `uv sync` installs `ipykernel` and Matplotlib from the dev group, so the kernel can use `.venv`. Each notebook launches its sweep, reads it back and saves every figure with the data it shows in the sweep's folder under `outputs/` (gitignored).

## Data and checkpoints

Code is committed. Large data is not. `.gitignore` excludes `data/`, `outputs/`, and `wandb/`, plus payload formats such as `*.h5`, `*.csv`, and `*.npy`. Create `data/` locally for datasets; that folder is not committed.

Each run saves a checkpoint at the end of training. `uv run python main.py` writes it in Hydra's timestamped directory from `hydra.run.dir`: `outputs/<YYYY-MM-DD>/<HH-MM-SS>/checkpoint.npz`. Without a Hydra job, the file is `<checkpoint.fallback_root>/<uuid>/checkpoint.npz` (`outputs/<uuid>/checkpoint.npz` by default). Set `checkpoint.dir` to choose the directory, and `checkpoint.use_uuid=true` to nest a UUID under it.

DVC can be added later if large HDF5 or CSV files need strict versioning.

## Run tests

```bash
uv run pytest
```

## Format, lint, and types

```bash
uv run ruff format
uv run ruff check
uv run ruff format --check
uv run mypy
```

Ruff formats and lints. Mypy typechecks the package, `main.py` and `experiments/example/run_experiment.py`. At startup, `main.py` seeds NumPy from `seed` in `conf/config.yaml` (default `42`). Override it with `seed=0`.

## Develop

Package code lives in `src/QELM`. Import it as `QELM`. Add a dependency with `uv add`.
