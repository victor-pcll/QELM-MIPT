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
| ML task | (1) the expressivity of the features (record `M_{t,i}` and final `Z_i`), independent of any target; (2) regression of targets spread over the Bloch monomials of one degree `k = 1, 2, 3` (unit variance, angles uniform on `[0, pi]`), ridge (`alpha = 1e-3`, bias column) on one single-shot record per input, 3000 inputs, 75 % train, scored by test R2 and test accuracy of the sign of the target |
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
5. QELM scores (`QELM.evaluation.fourier_tasks`), with the same circuit:
   1. Draw one target per degree `k` and 3000 inputs. Both are drawn once per run (seed `seed + 10000`), and the same ones are used for every `p`.
   2. For each input, draw a single-shot record (masks and outcomes with the Born rule on the state vector), then the final Z outcomes.
   3. Fit the ridge on 2250 inputs and score it on 750: test R2 and test accuracy.
   4. Estimate the shot noise `Noise = E[Cov(M | theta)]` from the same records.
   5. From `C(p)`, predict the R2 of the best linear readout of records averaged over `S` shots, `c^T (Sigma + Noise / S)^+ c / Var(y)`, for `S = 1, 10, 100, 1000`. The prediction for `S = inf` (no shot noise) is the ceiling: the share of the target that the mean features can represent. The predicted accuracy is that of a Gaussian target, `1/2 + arcsin(sqrt(R2)) / pi`.
6. Optionally (`PAIRS=same-time` or `PAIRS=all`), add the correlators `F_a F_b` of two features as features: `M_{t,i} M_{t,j}` and the final `Z_i Z_j` (`same-time`), or every pair (`all`). Their means `E[F_a F_b | theta]` are again linear in `rho(theta)`, since the joint outcome distribution is. Their rows come from a backward evolution in which a measured feature acts as `O -> p {Z_i, O} / 2` and the other sites as the dephasing (`product_transfer_matrix`). Steps 3 to 5 are repeated on the linear features plus the correlators, with the same sampled records multiplied pairwise.
7. Average over the circuits.

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
