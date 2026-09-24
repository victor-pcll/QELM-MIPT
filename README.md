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

## Run tests

```bash
uv run pytest
```

## Configuration

Scientific parameters live in `config/` and can be overridden from the command line. Secrets and machine-local paths (for example `WANDB_API_KEY`) belong in `.env`, not in YAML. Copy `.env.example` to `.env` for local secrets.

```bash
cp .env.example .env
uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
```

`sim_type=ising` selects `config/sim_type/ising.yaml`. Add another file in that directory to add a simulation type. `test` settings (`dt`, `n_steps`) live in `config/config.yaml`.

## Develop

Package code lives in `src/qrc_env_template`. Import it as `qrc_env_template`. Add a dependency with `uv add`.
