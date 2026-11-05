# Documentation

Notes for this repository. Scientific parameters stay in Hydra YAML under `conf/`. Secrets stay in `.env`. Most of the pipeline is still a scaffold: it routes a config and records a status string. The transverse-field Ising Hamiltonian is built (`TransverseFieldIsingModel.hamiltonian`). Time evolution, training, and scoring are not implemented yet.

- [Setup](setup.md) — install uv and sync the project
- [Configuration](configuration.md) — Hydra YAML versus `.env` secrets
- [Layout](layout.md) — `conf/`, `models/`, `training/`, `evaluation/`, and `main.py`
- [Running](running.md) — command-line overrides and the notebooks
- [Checkpoints and local data](checkpoints.md) — `outputs/`, `data/`, and `wandb/`
- [Code quality](code-quality.md) — Ruff, Mypy, pytest, and the seed from config
- [Contributing](contributing.md) — issues, draft pull requests, and who marks a pull request ready

Scientific notes:

- [The early-layers hypothesis of the linear QELM on M](notes/early-layers-proof.md) — what is proven about which rounds of the record the linear readout uses (#288)
- [État des lieux des sondes de la MIPT](notes/etat-des-lieux-sondes-mipt.md) — note en français : quelles sondes ont été testées, avec quel décodeur, et la suite
- [Le QELM voit-il la MIPT ?](notes/qelm-voit-il-la-mipt.md) — note en français : ce qui est prouvé, mesuré ou conjecturé, le critère de taille finie, le verdict par cas et les calculs manquants (#303)
