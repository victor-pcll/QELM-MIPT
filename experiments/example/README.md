# Example: the test experiment and a template for new ones

Not a physics experiment: a smoke test of the pipeline and the starting point
for a new sweep.

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/example/test_experiment.sh` (bash sweep of `main.py`), `experiments/example/run_experiment.py` (Hydra entry point for custom logic) |
| Config | `conf/config.yaml` |
| Physical model | TFIM of `conf/models/TFIM.yaml` |
| System sizes `N` | `4 6` |
| Control parameters | `seed` in `0 1` |
| ML task | the default pipeline of `main.py` (`QELM.pipeline.run_pipeline`) |

## Methodology

`test_experiment.sh` runs `main.py` four times (two `N`, two seeds), each run
in `outputs/test_experiment/<date_time>/N<N>_seed<seed>/` with its
checkpoint, log and config. `run_experiment.py` composes the same config and
seeds NumPy; put new logic in its `main` and write temporary outputs to
`experiments/scratch/`, which git ignores.

How to run it and how to write a new sweep:
[experiments/README.md, "Run the test experiment"](../README.md#run-the-test-experiment)
and ["Write your own"](../README.md#write-your-own).

## Results and discussion

The four runs finish in a few seconds; a failure means the environment
(`uv sync`) or the pipeline is broken. A new experiment folder gets its own
`README.md` from [`README_TEMPLATE.md`](../README_TEMPLATE.md).

## Figures

None.
