# Setup

The project requires Python 3.12 or newer. That pin is `.python-version` and `requires-python` in `pyproject.toml`. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other options are in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Sync

From the repository root:

```bash
uv sync
```

`uv sync` creates `.venv`, installs the package, and installs the `dev` group (Ruff, Mypy, and pytest). Run project commands with `uv run` so they use that environment.

Add a dependency with `uv add`.
