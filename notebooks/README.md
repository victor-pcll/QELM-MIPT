# Notebooks

| Folder | Notebook | What it does |
| --- | --- | --- |
| `benchmarks/` | `01_learnability_accuracy.ipynb` | Accuracy of Eve's classifier, Barratt et al. PRL 129, 200602 (2022), Fig. 2b (issue #113) |
| `benchmarks/` | `02_learnability_entropy.ipynb` | Mean entropy of the charge posterior, same paper, Fig. 2c (issue #115) |
| `benchmarks/` | `03_charge_sharpening.ipynb` | Charge sharpening, Agrawal et al. PRX 14, 041012 (2024), Fig. 7 (issues #137, #142) |
| `results/` | `02_QELM_M_analysis.ipynb` | QELM on the measurement record `M`, sweep over `p` (issue #94): one figure for the performance (accuracy, R2), one for the entanglement (ancilla, half-system and single-qubit entropies), then the sweep over `p` for `h = J, 10 J` and `N = 4, 6` (issue #174) |
| `results/` | `03_QELM_M_alpha_ratio.ipynb` | QELM on `M`, metrics against the sample-to-parameter ratio `n_train / n_params` (issue #96), then heatmaps of NMSE, R2, accuracy and balanced accuracy over `p` and the ratio (issue #164) |
| `results/` | `04_QELM_Q_measurements.ipynb` | QELM on the charge `Q` from the final `<Z_k>`, with intermediate Z measurements of probability `p` (issue #163) |
| `results/` | `05_QELM_M_circuits.ipynb` | The sweep of notebook 03 once per unitary dynamics, TFIM, Haar circuit and Clifford circuit: curves against the ratio, heatmaps over `p` and the ratio, and the three dynamics compared, then the same sweep with a fixed draw of the circuit gates (issue #171) |
| `results/` | `06_QELM_Q_observables_p_h.ipynb` | Learnability of the charge `Q` from `<Z_k>` and from `<X_i X_j>` over a 2D grid of `p` and the field `h` (0 to 3 J), for `N = 4, 6`: accuracy and R2 against `p` and against `h`, and heatmaps over the `(p, h)` plane, then the same with the record of the intermediate measurements added to the input, `phi = [M, X]` (issue #179) |
| `results/` | `07_QELM_Z_linear_observables.ipynb` | Distributions of the linear observables `<X_k>`, `<Y_k>`, `<Z_k>` of the unitary QELM (`exp_QELM_Z`, no measurement) for datasets of 100 to 5000 samples: the ridge readout on these features alone, superimposed histograms of `<O_{Ns/2}>`, the convergence of the empirical law with the dataset size, and its shape against a Gaussian and Haar-random states (issue #191) |

`benchmarks/` reproduces published figures, to validate the library; `results/` holds the new results of the project.

Notation, the same everywhere: `N` is the total number of qubits, ancilla included, and `L` the number of samples per run (`samples.L` in the config). When a paper writes its chain length `L`, the notebook says so and converts it to `N`.

Each notebook moves to the repository root first, then launches its sweep in its **own** output folder, so that two notebooks never read each other's runs (issue #151):

| Notebook | Output folder |
| --- | --- |
| `benchmarks/01_learnability_accuracy.ipynb` | `outputs/learnability_fig2b/<date_time>/` |
| `benchmarks/02_learnability_entropy.ipynb` | `outputs/learnability_fig2c/<date_time>/` |
| `benchmarks/03_charge_sharpening.ipynb` | `outputs/charge_sharpening_fig7/<date_time>/` |
| `results/02_QELM_M_analysis.ipynb` | `outputs/QELM_M/<date_time>/`, and `outputs/QELM_M_h/<date_time>/` for its sweep over `h` and `N` |
| `results/03_QELM_M_alpha_ratio.ipynb` | `outputs/QELM_M_ratio/<date_time>/` |
| `results/04_QELM_Q_measurements.ipynb` | `outputs/QELM_Q/<date_time>/`, and `outputs/QELM_Q_time/<date_time>/` for its evolution-time sweep |
| `results/05_QELM_M_circuits.ipynb` | `outputs/QELM_M_circuits/<date_time>/<model>/` |
| `results/06_QELM_Q_observables_p_h.ipynb` | `outputs/QELM_Q_p_h/<date_time>/` |
| `results/07_QELM_Z_linear_observables.ipynb` | `outputs/QELM_Z_linear/<date_time>/` |

The notebook creates the sweep folder (`QELM.utils.new_sweep_dir`), hands it to the launcher as `OUT_DIR` and reads back exactly that folder; `RUN = False` reads back the latest sweep of its own folder. Every figure is saved there with the data it shows, through `QELM.utils.plots.save_figure`: `<name>.png` next to `<name>.json`. The learnability notebooks also pass `FIGURES`, so each sweep folder holds only its own figure (Fig. 2b or 2c). The plots share one style (`QELM.utils.plots.subplots`): no grid, a box closed on all four sides, large titles.

To experiment without blocking `git pull`, copy a notebook into `notebooks/scratch/`, which is gitignored.
