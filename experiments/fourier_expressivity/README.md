# Fourier expressivity of the monitored QELM

Which functions of the input can the QELM read, and how does the
measurement-induced phase transition (MIPT) change them? The QELM encodes the
input once, with the angle encoding, so every mean the readout can see is a
combination of the `3^N` Bloch monomials
`B_alpha(theta) = prod_{alpha_i = x} sin(theta_i) prod_{alpha_i = z} cos(theta_i)`.
This is a trigonometric polynomial of degree at most one in each angle. The
intermediate measurements, `p` and the symmetry of the gates add no frequency.
They decide which coefficients are in reach, and how many input angles each
feature couples (its degree, the Pauli weight of its Heisenberg operator).
This experiment computes these coefficients exactly, across `p`, `N` and four
dynamics.

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/fourier_expressivity/fourier_expressivity.py` (one fixed circuit, every `p`), launcher `fourier_expressivity.sh`, figures `plot_fourier_expressivity.py` |
| Config | command-line options of the script; environment variables of the launcher |
| Physical model | open brickwork of two-qubit gates, depth `T = N` (full periods), Z measurement of each site with probability `p` after each layer; Haar U(4), U(1), Z2 parity (`prod_i Z_i`) or Clifford gates, one fixed draw per seed |
| System sizes `N` | 4, 6, 8 (10 circuits each) and 10 (2 circuits) by default |
| Control parameters | `p` from 0 to 1 by 0.05, the dynamics |
| Initial state | angle encoding `prod_i (I + sin(theta_i) X_i + cos(theta_i) Z_i) / 2`, angles uniform on `[0, pi]` for `d_eff` |
| ML task | none: the expressivity of the features (record `M_{t,i}` and final `Z_i`), independent of any target |
| Issues / PRs | #309 |

## Methodology

1. Draw the gates of one circuit (`fixed_instance`).
2. For each feature, take the Heisenberg image `O_k` of `Z_i` through the layers. Between layers, apply the Z-dephasing of probability `p` on every site: this is the exact average over the masks and outcomes of the earlier measurements. The mean of `M_{t,i}` is `p Tr[O_k rho(theta)]`, and the mean of the final `Z_i` is `Tr[O_k rho(theta)]`.
3. Project `O_k` on the Pauli strings of `{I, X, Z}^N`. The row `C_{k alpha} = Tr[O_k sigma_alpha] / 2^N` gives the feature mean as `sum_alpha C_{k alpha} B_alpha(theta)`.
4. Read `C(p)` (`QELM.evaluation.fourier.expressivity_measures`):
   - its rank;
   - the degree profile `W_k`, the share of the weight on monomials of `k` angles (norm of angles uniform on the circle), and its mean over `N`;
   - the share `A_k` of the degree-`k` monomials in reach;
   - the participation ratio `d_eff` of the feature covariance for angles uniform on `[0, pi]`;
   - the weight on monomials with an odd number of `sin`.
5. Average over the circuits.

No trajectory is sampled. The cost is the Heisenberg evolution of
`2^N x 2^N` operators. How to run it:
[experiments/README.md, "Run the Fourier expressivity of the QELM"](../README.md#run-the-fourier-expressivity-of-the-qelm-issue-309).
The notebook is `notebooks/results/12_fourier_expressivity.ipynb`.

## Results and discussion

Results pending: the first full sweep is still to run. Exact checks at small `N`, from the tests and a scratch run at `N <= 6`:

- **Spectrum.** Every record and readout mean is reproduced exactly by `C(p)` on the `3^N` monomials, including against a brute-force enumeration of masks and outcomes (`tests/test_fourier.py`).
- **Selection rule.** U(1) and Z2 circuits put no weight on the odd-`sin` monomials, at every `p`. Their rank is bounded by `(3^N + 1) / 2`. With the angle encoding (Bloch vectors in the x-z plane), the two symmetries have the same support; they differ only in the coefficients.
- **Clifford.** Each feature has its weight on a single degree (a single Pauli string), whatever `p < 1`.
- **Measurements.** In the scratch run, the mean degree of the final `Z_i` falls with `p` for Haar and Z2 circuits at `N = 6`. Whether it crosses in `N` near `p_c` is the open question of the sweep.
