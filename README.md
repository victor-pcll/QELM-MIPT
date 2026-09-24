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

## Develop

Package code lives in `src/qrc_env_template`. Import it as `qrc_env_template`. Add a dependency with `uv add`.
