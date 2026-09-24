# Documentation

Notes for this repository. Scientific parameters stay in Hydra YAML under `conf/`. Secrets stay in `.env`. The model, training, and evaluation code are scaffolds: they route a config and record a status string. They do not build a Hamiltonian, integrate a trajectory, or compute a score.

- [Setup](setup.md) — install uv and sync the project
- [Configuration](configuration.md) — Hydra YAML versus `.env` secrets
- [Layout](layout.md) — `conf/`, `models/`, `training/`, `evaluation/`, and `main.py`
- [Running](running.md) — command-line overrides
- [Checkpoints and local data](checkpoints.md) — `outputs/`, `data/`, and `wandb/`
- [Code quality](code-quality.md) — Ruff, Mypy, pytest, and the seed from config
