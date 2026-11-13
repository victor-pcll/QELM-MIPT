# The general Fourier framework, checked exactly at small N (issue #353)

The note [`cadre-fourier-general.md`](../../documentation/notes/cadre-fourier-general.md)
writes the ridge QELM with mid-circuit measurements in the Pauli-mode
("Fourier") language for a **general** circuit, then reduces it to Clifford.
This folder holds the exact small-N table that illustrates which statements
survive outside Clifford. The pytest checks of every general statement are in
`tests/test_fourier_framework.py`.

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/fourier_framework/small_n_table.py` |
| Library | `QELM.evaluation.fourier_framework` (dense, exact, `N <= 4`) |
| Physical model | open brickwork, one layer = even bonds, odd bonds, then a `Z` measurement of each site with probability `p`; depth `T = N`; final `Z` readout of every site (or not, for the "mid" columns) |
| Circuits | for each seed, one Haar and one uniform two-qubit Clifford circuit with the **same** mask; circuits with more than 14 bits skipped |
| System sizes `N` | 4 |
| Control parameters | `p` = 0, 0.1, 0.2, 0.3, 0.5; 10 seeds (9 kept at `p = 0.5`) |
| Inputs | all `4^N` products of `|0>, |1>, |+>, |->` (same first and second moments as an angle uniform on the XZ circle) |
| ML task | population ridge (`lambda -> 0+`, `K = inf`) on the Born probabilities of every record, and on the parities of at most `k = 1, 2` bits |
| Issues / PRs | #353; builds on #334, #337, #339, #341, #344, #349 |

## Methodology

1. Draw the gates and the mask, enumerate every record `m` and its Kraus
   operator `K_m`, build `E_m = K_m^dagger K_m`.
2. Expand each effect on the `4^N` Pauli strings: row `C_m`, with
   `Pr(m | x) = C_m . B(x)`.
3. Structure: dense rank of `C`, its rank on the strings without `Y`, the
   largest commutator on an orthonormal basis of the span, and the
   Born-averaged Renyi entropies `Delta_0, Delta_1, Delta_2` of
   `E_m / Tr E_m` for the mid-circuit record.
4. Ridge: `E_perp` (weighted projection) for `Z` of site 2 and for the sum of
   all `X_i` and `Z_i`; two-filter split for `Z_2` with the parities of at most
   1 and 2 bits; whether the exact features separate the `4^N` inputs.

How to run it: [`experiments/README.md`](../README.md#check-the-general-fourier-framework-at-small-n-issue-353).

## Results and discussion

The two tables are reproduced and read in section 9 of the note. In short:

- **Clifford** reproduces the whole Clifford structure exactly: commuting
  effects, `dim = 2^N` with the final readout, `Delta_0 = Delta_1 = Delta_2`,
  `log2 dim_mid + Delta = N` on every circuit, and features that never
  separate `x` from `Q x Q`.
- **Haar** breaks every part of it: non-commuting effects, a span above
  `2^N` holding up to 70 % of the visible functions, Renyi entropies that
  split more and more as `p` grows (`p = 0.5`: `Delta_0 = 1.0`,
  `Delta_1 = 0.49`, `Delta_2 = 0.39`), `log2 dim_mid + Delta_0 > N`, and
  injective features.
- **Pythagoras holds for Haar** (cross term below `1e-15`), but for the local
  target the readout term (0.25 to 0.63 for `k = 1`) exceeds the circuit term
  from `p = 0.1` on, the opposite of Clifford.

**Caveat.** `N = 4` illustrates structural statements; it says nothing about
`p_c`.
