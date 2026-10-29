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

Each file has two curves, `L = 6` and `L = 20`, the paper's number of
system sites. `plot_charge_sharpening.py` renames them `N = 6` and `N = 20`,
our number of system qubits, the ancilla `R` excluded (issue #235), and
compares each with our curve of the same `N`. The axes were calibrated
the other way round when digitizing, so each pair of columns holds
**(value, gamma)**, not (gamma, value): read them with
`QELM.utils.load_wpd_csv(path, swap_xy=True)`, as `plot_charge_sharpening.py`
does. The entropy is in nats (`S_R <= ln 2 ≈ 0.693`), the variance in
nats². The Binder ratio reaches about `-10` at small `gamma`, which fits the
standard `B = 1 - mu_4 / (3 mu_2**2)` (the paper prints `(1 - mu_4) / (3 mu_2**2)`). The `.gitignore` lets `experiments/*/reference/*.csv` through, unlike
every other CSV.
