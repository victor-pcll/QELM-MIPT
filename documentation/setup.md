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

`uv sync` creates `.venv`, installs the package, and installs the `dev` group (Ruff, Mypy, pytest, ipykernel, and Matplotlib). Run project commands with `uv run` so they use that environment. The series notebook uses that kernel; see [Running](running.md).

Add a dependency with `uv add`.

## After moving or renaming the folder

`.venv` stores the absolute path of `src/`. If the repository folder is moved or renamed, imports such as `import QELM` fail with `ModuleNotFoundError`. Rebuild the environment:

```bash
rm -rf .venv
uv sync
```

## `No module named 'QELM'` on macOS

If `uv run python main.py` fails with `ModuleNotFoundError: No module named
'QELM'` although `.venv` is up to date, check whether macOS has marked the
`.pth` files of the editable install as hidden. Python 3.12+ skips hidden
`.pth` files (`python -v` prints `Skipping hidden .pth file`):

```bash
ls -lO .venv/lib/python3.12/site-packages/*.pth
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
```

The bash launchers in `experiments/scripts/` and the test suite put `src/` on
`PYTHONPATH` themselves, so they work either way.
