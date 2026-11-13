# Clifford QELM: the MIPT and its imprint on the ridge at K = infinity (issue #337)

Circuit: the periodic brickwork of random two-qubit Clifford gates (Gullans &
Huse), with a `Z` measurement of each site with probability `p` after every
half-layer. The input is encoded on every site at `t = 0`, then come
`T = 2N` steps and the final `Z` readout of every site. Inputs are products
of random Pauli eigenstates `|0>, |1>, |+>, |->`. Their first and second
moments are those of the angle encoding on the circle, so every linear
quantity of issue #334 is unchanged.

The question: with infinitely many shots (`K = inf`) and `lambda -> 0+`, does
the ridge readout carry the trace of the transition, and how?

## What the ridge is at K = infinity, for a stabilizer circuit

The features of an input are the Born probabilities `Pr(m | x) = Tr[E_m rho(x)]`.
For a Clifford circuit, every effect has the form

    E_m = Tr(E_m) 2^-r  sum_{g in G} s_g(m) g^T .

The group `G` (`2^r` Pauli strings) does not depend on the record `m`. Only
the signs `s_g(m) = +-1` depend on it, and they are characters of the record.
It follows exactly that:

- **`Span{E_m}_m` is the span of `G`, of dimension `2^r`.** With the full
  record, `r = N` at every `p`. With the mid-circuit record alone,
  `r = N - Delta`, where `Delta` is the purification order parameter. The
  effects all commute, so the span can never be complete: it holds `2^N`
  strings out of the `3^N` visible functions.
- **A string containing `Y` is invisible.** It gives `Tr[g rho(x)] = 0` for
  every input in the XZ plane.
- **Spectrum.** The population covariance `Sigma` of the features is
  diagonal in the monomials `B_g` of the visible strings of `G`. Its
  eigenvalues are proportional to `w_g = 2^-|g|`, so **the spectrum of
  `Sigma` is the weight distribution of `G`**.
- **Population ridge.** As `lambda -> 0+`, the ridge is the projector on that
  span. For a monomial target, `E_perp = 1 - Pr[a in G]`, which is 0 or 1
  for each circuit.
- **Locality.** `2^(l - S(l))` elements of `G` are supported in an interval
  of `l` sites, where `S(l)` is the entanglement entropy of those `l`
  reference qubits after the full readout.

The test suite checks all of these statements at `N = 4`
(`tests/test_clifford_ridge.py`). It compares the Pauli support of every
`E_m` with `G`. It runs least squares on the exact Born features of all
`4^4` inputs and finds `R2 = 1` or `0` for each monomial. It checks the
degree-1 `R2` and the eigenvalues `2^-|g|`. A fresh circuit is drawn per
sample.

## Run

    experiments/clifford_ridge/population.sh       # JOBS=3 used, about 1 h
    uv run python experiments/clifford_ridge/plot_population.py outputs/clifford_ridge

The run used `N = 16, 32, 64, 128` with 2000, 1000, 500 and 200 circuits per
`p`, over 29 values of `p` from 0.02 to 0.5. It writes
`population_N*_T*_s0_c*.json`; the plot script merges the chunks. The
`transition` subcommand (`Delta` and memberships, with more samples, at
`T = 2N` and `4N`) was also run, except for `N = 128, T = 4N`. The
finite-shot `ridge` subcommand is kept and tested, but was not run.

## Results

**1. The transition is in the circuit, at the literature value.** The
`Delta` curves of successive `N` cross at `p = 0.162, 0.163, 0.159`
(`T = 2N`), and the transition runs give `0.165, 0.161, 0.159` (`T = 2N`)
and `0.160, 0.163` (`T = 4N`). The mean is `p_c = 0.161`. `Delta` collapses
against `(p - p_c) N^(1/1.3)` (`figures/population_collapse.png`).

**2. No E_perp of a fixed local target sees the transition.** For `Z_i`,
`X_i`, `Z_i Z_i+1`, `X_i X_i+1` and `Z_i X_i+1`, `E_perp(p)` is smooth and
identical for `N = 16` to `128` within errors (`population_eperp.png`). For
`N = 128` and the full record:

| p | E_perp Z_i | E_perp Z_i Z_i+1 | E_perp Z_i Z_i+64 |
|---|---|---|---|
| 0.04 | 0.989 | 0.995 | 1.000 |
| 0.10 | 0.971 | 0.985 | 0.999 |
| 0.16 | 0.948 | 0.970 | 0.997 |
| 0.22 | 0.917 | 0.951 | 0.994 |
| 0.30 | 0.882 | 0.927 | 0.987 |
| 0.50 | 0.830 | 0.898 | 0.970 |

The ridge never becomes complete: below `p_c`, `E_perp` is close to 1, not
to 0. Far-apart targets are not cut off above `p_c` either. They become
*more* learnable as `p` grows, as products of two local elements of `G`
(`Pr[Z_i Z_j in G] ~ Pr[Z_i in G]^2`). Adding the final readout to the mid
record changes nothing for local targets: the extra elements are nonlocal.

**3. The transition is in the geometry of `Span{E_m}`.** The dimension is
`2^N` at every `p`; what changes is the locality of `G`.

- The half-chain entropy `S(N/2)` of the references after the full readout
  grows as `N^0.9` for small `p`. It stays at `O(1)` above `p_c`; for
  example, `S(N/2) = 0.95` for every `N` at `p = 0.3`.
- The growth exponent `d log S / d log N` drops from about 0.8 to 0 between
  `p = 0.1` and `0.2`, and the drop sharpens with `N`
  (`population_order.png`).
- `S(N/2) / log2 N` collapses against the same scaling variable, as at a
  critical point with a logarithm.

For the ridge, this means that the fraction of `G` that fits in a window of
`l` sites, `k(l)/l = 1 - S(l)/l`, is small up to `l ~ N` below `p_c` (volume
law), and close to 1 from small `l` above `p_c` (area law). At `N = 128`
and `l = 8`, it is 0.17 at `p = 0.04`, 0.61 at `p_c` and 0.88 at `p = 0.3`
(`population_range.png`).

**4. The spectrum moves towards large eigenvalues above p_c. It does not
collapse there.**

- At `N = 16`, the visible strings number 705 at `p = 0.04`, 1002 at
  `p_c` and 1848 at `p = 0.5`.
- Their mean weight falls from 10.4 to 7.7, so the typical eigenvalue rises
  from `2^-10` to `2^-8`.
- `d_eff(lambda) = sum w/(w + lambda)` rises monotonically with `p` at
  every `lambda`.
- The effective rank `sum mu / mu_max` rises from about 10 to 65.
- The number of visible strings per site within 6 sites does not depend on
  `N`. It rises smoothly, with an inflection near `p_c`
  (`population_spectrum.png`).

Below `p_c`, the information about the input is read late, by the final
readout, in delocalised strings of weight close to `N`. These strings have
exponentially small eigenvalues, and most of them contain `Y` and are
invisible. Above `p_c`, it is read early and locally.

## Reading

At `K = inf`, the ridge of a Clifford QELM carries the MIPT. It does not do
so as a jump of `E_perp` for a given task, nor as a change in the size of
`Span{E_m}`. It does so as a change in the **geometry** of the dictionary of
`2^N` functions it can represent: delocalised (volume law) below `p_c`,
local (area law) above. The local observables are analytic in `p`, as usual
for the MIPT. The order parameters are global: `Delta`, and the scaling of
`S(l)` with `l` and `N`.

The impossibility of completeness comes from the commuting effects of
Clifford circuits. With Haar gates (issue #334), `Span{E_m}` can be much
larger than `2^N`. At `N = 6`, the full-record `E_perp` of degree 1 was 0.02
at `p = 0.1`.

## Figures

- `population_order.png`: `Delta` (crossing at `p_c`), `S(N/2)` of the
  references, and the growth exponent of `S(N/2)` with `N`.
- `population_collapse.png`: `Delta` and `S(N/2) / log2 N` against
  `(p - p_c) N^(1/nu)`; `E_perp` of `Z_i` against `p` for every `N`.
- `population_eperp.png`: `E_perp(p)` for six families of monomials,
  full record (solid) and mid record (dotted).
- `population_range.png`: at `N = 128`, `R2` against the range `l` of
  `ZZ` and `XX`, and `k(l)/l`.
- `population_spectrum.png`: weight histogram of the visible strings
  (`N = 16`), `d_eff(lambda)`, effective rank, and local visible strings per
  site.
