# Notebooks

| Folder | Notebook | What it does |
| --- | --- | --- |
| `benchmarks/` | `01_learnability_accuracy.ipynb` | Accuracy of Eve's classifier, Barratt et al. PRL 129, 200602 (2022), Fig. 2b (issue #113) |
| `benchmarks/` | `02_learnability_entropy.ipynb` | Mean entropy of the charge posterior, same paper, Fig. 2c (issue #115) |
| `benchmarks/` | `03_charge_sharpening.ipynb` | Charge sharpening, Agrawal et al. PRX 14, 041012 (2024), Fig. 7 (issues #137, #142) |
| `results/` | `01_TFIM_reservoir.ipynb` | The disordered TFIM used as the reservoir |
| `results/` | `02_QELM_M_analysis.ipynb` | QELM on the measurement record `M`, sweep over `p` (issue #94) |

`benchmarks/` reproduces published figures, to validate the library; `results/` holds the new results of the project.

Notation, the same everywhere: `N` is the total number of qubits, ancilla included, and `L` the number of samples per run (`samples.L` in the config). When a paper writes its chain length `L`, the notebook says so and converts it to `N`.

Each notebook moves to the repository root first, launches its sweep (or reads back the latest one with `RUN = False`), and saves every figure with the data it shows in the sweep's folder, `outputs/<experiment>/<date_time>/`, through `QELM.utils.plots.save_figure`: `<name>.png` next to `<name>.json`. The plots share one style (`QELM.utils.plots.subplots`): no grid, a box closed on all four sides, large titles.

To experiment without blocking `git pull`, copy a notebook into `notebooks/scratch/`, which is gitignored.
