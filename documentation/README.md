# Documentation

Notes for this repository. Scientific parameters stay in Hydra YAML under `conf/`. Secrets stay in `.env`. Most of the pipeline is still a scaffold: it routes a config and records a status string. The transverse-field Ising Hamiltonian is built (`TransverseFieldIsingModel.hamiltonian`). Time evolution, training, and scoring are not implemented yet.

- [Setup](setup.md) — install uv and sync the project
- [Configuration](configuration.md) — Hydra YAML versus `.env` secrets
- [Layout](layout.md) — `conf/`, `models/`, `training/`, `evaluation/`, and `main.py`
- [Running](running.md) — command-line overrides and the series notebook
- [Checkpoints and local data](checkpoints.md) — `outputs/`, `data/`, and `wandb/`
- [Code quality](code-quality.md) — Ruff, Mypy, pytest, and the seed from config
