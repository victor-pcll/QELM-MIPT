# Reference data: Agrawal et al. 2024, Fig. 7

Curves digitized with WebPlotDigitizer from Fig. 7 ("Charge sharpening") of
U. Agrawal, J. Lopez-Piqueres, R. Vasseur, S. Gopalakrishnan and A. C. Potter,
"Observing quantum measurement collapse as a learnability phase transition",
Phys. Rev. X 14, 041012 (2024), arXiv:2311.00058. Attached to issue #137.

| File | Panel | WebPlotDigitizer export |
| --- | --- | --- |
| `agrawal2024_fig7_S_R.csv` | left: ancilla entropy `S_R` | `wpd_datasets-4.csv` |
| `agrawal2024_fig7_S_R_var.csv` | center: variance of `S_R` | `wpd_datasets-5.csv` |
| `agrawal2024_fig7_binder.csv` | right: Binder ratio | `wpd_datasets-6.csv` |

The x axis is the measurement strength `gamma`. Read them with
`QELM.utils.load_wpd_csv`; `plot_charge_sharpening.py` overlays them on a
sweep. The `.gitignore` lets `experiments/*/reference/*.csv` through, unlike
every other CSV.
