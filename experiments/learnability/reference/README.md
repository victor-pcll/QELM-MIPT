# Reference data: Barratt et al. 2022, Fig. 2

Curves digitized with WebPlotDigitizer from Fig. 2 of F. Barratt,
U. Agrawal, A. C. Potter, S. Gopalakrishnan and R. Vasseur, "Transitions in
the learnability of global charges from local measurements",
Phys. Rev. Lett. 129, 200602 (2022), arXiv:2206.12429.

| File | Panel | Source |
| --- | --- | --- |
| `barratt2022_fig2b_accuracy.csv` | Fig. 2b: accuracy of the classifier, `L = 6` and `L = 20` | `wpd_datasets.csv`, attached to issue #113; columns hold **(value, p)**, read with `SWAP_XY = True` in `plot_learnability.py`; the accuracy was digitized on 0..1 and is mapped to `0.564 + 0.436 v` (`VALUE_AXIS`), the two ends read off the figure (0.59 at p = 0.025, 1 at saturation) |
| `barratt2022_fig2c_entropy.csv` | Fig. 2c: mean entropy `E[H(P_corr)]` in nats (`<= ln 2`), `L = 20` and `L = 6` | `wpd_datasets-7.csv`, redigitized for issue #177 (the export of issue #115, `wpd_datasets-2.csv`, had a miscalibrated axis: `L = 20` saturated at 0.033 instead of 0, first point at p = 0.023 instead of 0.025); columns hold **(value, p)** too (`SWAP_XY = True`); values used as digitized |

`plot_learnability.py` reads every file of this folder named after one of
its panels (`PANELS`), with `QELM.utils.load_wpd_csv`. The paper writes
its chain length `L`; there is no ancilla, so a curve named `L = 6` is
renamed `N = 6` (our total number of qubits) and compared with our curve of
the same `N`; the other curves are drawn for reference only. If the axes were calibrated the other
way round when digitizing, so that the columns hold (value, p), set
`SWAP_XY = True` in `plot_learnability.py`, as for the Fig. 7 curves of
issue #137. The `.gitignore` lets `experiments/*/reference/*.csv` through,
unlike every other CSV.
