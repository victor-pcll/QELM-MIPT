# Code quality

Ruff, Mypy, and pytest are in the `dev` group installed by `uv sync`. From the repository root:

```bash
uv run ruff check
uv run ruff format --check
uv run ruff format
uv run mypy
uv run pytest
```

Ruff lints and formats. `pyproject.toml` sets the target to Python 3.12, a line length of 88, and the rule sets `E`, `F`, `I`, and `UP`. `ruff check` reports lint findings. `ruff format --check` reports format drift and does not rewrite files. `ruff format` rewrites them.

Mypy typechecks `src/QELM` and `main.py` with `disallow_untyped_defs`. It does not typecheck `tests/`.

pytest collects `tests/`. `tests/sparse_checks.py` holds `assert_sparse_hermitian(H, atol=1e-8)`, the standard check that a sparse matrix is Hermitian: it compares `H - H^dagger` in sparse form and never calls `toarray()`, so it also works on matrices too large to hold dense. Import it in a test with `from sparse_checks import assert_sparse_hermitian`.

GitHub Actions runs the four checks on every push and pull request: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, and `uv run pytest`. The workflow is `.github/workflows/code-quality.yml`. It installs uv, installs the Python version in `.python-version`, syncs from `uv.lock`, and runs those commands. It does not start extra services.

## Seed

`main.py` calls `set_seed` before the pipeline. That function reads the integer `seed` from the composed config (`seed` in `conf/config.yaml`, overridable on the command line) and passes it to `numpy.random.seed`. PyTorch and JAX are not dependencies, so only NumPy is seeded. The same value is copied into checkpoint metadata.

```bash
uv run python main.py seed=0
```

`seed.py` does not embed a seed of its own.
