# Checkpoints and local data

Code is committed. Datasets and run artifacts are not.

## Ignored paths

`.gitignore` excludes:

- `data/` — local time series and training datasets. Create it yourself if you need it. The package does not write there.
- `outputs/` — Hydra run directories and checkpoints.
- `wandb/` — local Weights & Biases files. The pipeline does not create this directory.

`.env` is ignored. `results/` at the repository root is ignored (`notebooks/results/` holds tracked notebooks). Payload suffixes are ignored too, including `*.h5`, `*.hdf5`, `*.csv`, `*.npy`, and `*.npz`.

DVC is not set up.

## Checkpoint path

`run_pipeline` saves one archive at the end of the run. `cfg.checkpoint` in `conf/config.yaml` chooses the path:

| Key | When `dir` is unset |
| --- | --- |
| `checkpoint.dir` | `null` means "do not use a fixed directory" |
| `checkpoint.fallback_root` | `outputs`. Used only when `dir` is unset and no Hydra job is active |
| `checkpoint.use_uuid` | `false`. When `true`, a UUID directory is nested under the chosen directory |
| `checkpoint.filename` | `checkpoint.npz`. This must be a file name, not a path |

`uv run python main.py` starts a Hydra job, so the file is the timestamped run directory from `hydra.run.dir`:

```text
outputs/<YYYY-MM-DD>/<HH-MM-SS>/checkpoint.npz
```

If nothing has initialized Hydra and `checkpoint.dir` is unset, the file is:

```text
outputs/<uuid>/checkpoint.npz
```

`outputs` there is `checkpoint.fallback_root`. Set `checkpoint.dir` to pick the directory. Set `checkpoint.use_uuid=true` to nest a UUID under that directory (or under the Hydra run directory, when `dir` is still unset).

## Archive contents

The file is a NumPy `.npz` with two entries:

- `weights` — `float64` array from the trainer. The scaffold stores shape `(0,)` because `fit` does not estimate weights.
- `metadata` — JSON text. Fields are `run_id`, `models`, `model`, `trainer`, `seed`, `training_status`, `evolution` (`dt`, `n_steps`, `status`), and `params` (the simulation block).

Nothing is uploaded.
