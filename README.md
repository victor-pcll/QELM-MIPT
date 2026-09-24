# qrc-env-template

A Python library managed with [uv](https://docs.astral.sh/uv/). Requires Python 3.12 or newer.

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
| `conf/` | Hydra YAML: `config.yaml` plus groups `sim_type`, `training`, `evaluation` |
| `data/` | Local time series and training datasets (not committed) |
| `src/qrc_env_template/models/` | Hamiltonians and time evolution (Ising, Kitaev scaffolds) |
| `src/qrc_env_template/training/` | Training algorithms (Readout, Ridge scaffolds) |
| `src/qrc_env_template/evaluation/` | Metrics and plot scaffolds |
| `main.py` | Entrypoint. Routes sim type → model → training → evaluation |

## Run the pipeline

Scientific parameters live in `conf/` and can be overridden from the command line. Secrets and machine-local paths (for example `WANDB_API_KEY`) belong in `.env`, not in YAML. Copy `.env.example` to `.env` for local secrets.

```bash
cp .env.example .env
uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
```

`sim_type=ising` selects `conf/sim_type/ising.yaml` and the Ising model. `training=ridge` (default) or `training=readout` selects the trainer. `N`, `J`, `learning_rate`, and `dt` stay in YAML or CLI overrides.

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

Ruff formats and lints. Mypy typechecks the package and `main.py`. At startup, `main.py` seeds NumPy from `seed` in `conf/config.yaml` (default `42`). Override it with `seed=0`.

## Develop

Package code lives in `src/qrc_env_template`. Import it as `qrc_env_template`. Add a dependency with `uv add`.
