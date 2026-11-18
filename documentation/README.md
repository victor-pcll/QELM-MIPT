# Documentation

Notes for this repository. Scientific parameters stay in Hydra YAML under `conf/`. Secrets stay in `.env`. The generic entrypoint `main.py` (`QELM.pipeline`) is still a scaffold: it routes a config and records a status string. The science runs through the library and the scripts of `experiments/`: encodings, Hamiltonian and circuit dynamics, mid-circuit measurements, ridge readout, entropies and finite-size analysis are implemented and tested; the [algorithm guide](algorithmes.md) describes them.

- [Setup](setup.md) — install uv and sync the project
- [Configuration](configuration.md) — Hydra YAML versus `.env` secrets
- [Layout](layout.md) — `conf/`, `models/`, `training/`, `evaluation/`, and `main.py`
- [Running](running.md) — command-line overrides and the notebooks
- [Checkpoints and local data](checkpoints.md) — `outputs/`, `data/`, and `wandb/`
- [Code quality](code-quality.md) — Ruff, Mypy, pytest, and the seed from config
- [Contributing](contributing.md) — issues, draft pull requests, and who marks a pull request ready

Scientific notes:

- [Guide des algorithmes](algorithmes.md) — note en français : chaque algorithme de la bibliothèque (encodages, circuits Haar et Clifford, mesures en milieu de circuit, tableau de stabilisateurs, entropies, Ridge, cadre de Fourier/Bloch et budget de R², décodeurs bayésiens, analyse de taille finie), avec ses conventions, son coût, ses limites et ses tests (#366)
- [The early-layers hypothesis of the linear QELM on M](notes/early-layers-proof.md) — what is proven about which rounds of the record the linear readout uses (#288)
- [État des lieux des sondes de la MIPT](notes/etat-des-lieux-sondes-mipt.md) — note en français : quelles sondes ont été testées, avec quel décodeur, et la suite
- [Le QELM voit-il la MIPT ?](notes/qelm-voit-il-la-mipt.md) — note en français : ce qui est prouvé, mesuré ou conjecturé, le critère de taille finie, le verdict par cas et les calculs manquants (#303)
- [Rang de l'espace des effets : l'algorithme exact](notes/rang-espace-des-effets.md) — note en français : le calcul exact du rang de l'espace des effets, étape par étape avec preuves, pseudo-code et exemples à la main (#317)
- [Le QELM dans le langage des modes de Fourier : le cadre général (Haar), puis Clifford](notes/cadre-fourier-general.md) — note en français : entrées, effets, parités et Ridge dans la base de Pauli pour un circuit quelconque, ce qui est exact en général, ce qui est propre à Clifford et ce qui est conjectural pour Haar, avec des vérifications exactes à petit N (#353)
- [Bibliographie vérifiée et liste de lecture](bibliographie.md) — note en français : vérification de chaque citation du dépôt (arXiv, DOI) et liste de lecture par thème et par priorité
