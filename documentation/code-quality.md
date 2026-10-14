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

pytest collects `tests/`.

GitHub Actions runs the four checks on every push and pull request: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, and `uv run pytest`. The workflow is `.github/workflows/code-quality.yml`. It installs uv, installs the Python version in `.python-version`, syncs from `uv.lock`, and runs those commands. It does not start extra services.

## Seed

`main.py` calls `set_seed` before the pipeline. That function reads the integer `seed` from the composed config (`seed` in `conf/config.yaml`, overridable on the command line) and passes it to `numpy.random.seed`. PyTorch and JAX are not dependencies, so only NumPy is seeded. The same value is copied into checkpoint metadata.

```bash
uv run python main.py seed=0
```

`seed.py` does not embed a seed of its own.

### Disorder realisations

The TFIM couplings `J_ij` are random. To average over disorder, every Hamiltonian of a run must get new couplings, and rerunning with the same `seed` must give the same ones. Models therefore take an explicit `numpy.random.Generator` instead of reseeding NumPy:

```python
from QELM import TransverseFieldIsingModel, make_rng

rng = make_rng(cfg)  # one generator for the whole run, seeded by cfg.seed
models = [TransverseFieldIsingModel.from_config(cfg, rng) for _ in range(100)]
```

Each model draws its couplings from `rng` and moves it forward, so model 0, 1, 2… all differ, and the same `seed` gives the same list. The generator does not touch the global `np.random` state, so other random calls between two models cannot shift the realisations.

When realisation `k` must not depend on the others (parallel runs, or a different number of realisations), use one generator per realisation:

```python
from QELM import spawn_rngs

models = [TransverseFieldIsingModel.from_config(cfg, r) for r in spawn_rngs(cfg, 100)]
```

`spawn_rngs` uses `numpy.random.SeedSequence(seed).spawn(n)`, so realisation `k` depends only on `seed` and `k`.

Without a generator (`rng=None`), a model draws from the global `np.random` state seeded by `set_seed`. Calling `set_seed` again before each model then gives the same couplings every time. `run_pipeline` builds its model with `make_rng(cfg)`.
