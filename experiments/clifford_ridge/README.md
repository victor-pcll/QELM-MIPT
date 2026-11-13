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

---

# Follow-up (issue #339): Haar vs Clifford, and the locality of the dictionary

Run (about 1 h 20 min on 4 cores):

    experiments/clifford_ridge/haar_locality.sh
    uv run python experiments/clifford_ridge/plot_locality.py outputs/clifford_ridge
    uv run python experiments/clifford_ridge/plot_haar_span.py outputs/haar_span

## 1. Haar against Clifford at K = infinity, same geometry

Geometry of #334: open brickwork, one layer is even bonds, then odd bonds,
then the `Z` measurements; `T = 2N`; `p_c = 0.29` for Haar in this
convention (#322). For each seed, one Haar circuit and one Clifford circuit
see the same masks. Seeds: 20 (`N = 4, 5`), 10 (`N = 6`) and 3 (`N = 7`,
coarse `p` grid). `haar_span.py` uses the exact span of the effects
(`effect_operator_basis`), its projection on the strings without `Y`, the
population ridge with `lambda -> 0+` (`approximation_bias`) and
`window_capture`, which is `dim(Span ∩ V_A)`. Figure:
`figures/haar_vs_clifford.png`.

Full record, mean over the seeds:

| N | p | Haar: angle rank / 3^N | Haar: E_perp, Z mid / deg 1 / deg 2 | Clifford: angle rank / 3^N | Clifford: E_perp, Z mid / deg 1 / deg 2 |
|---|---|---|---|---|---|
| 6 | 0 | 0.09 | 0.65 / 0.63 / 0.75 | 0.02 | 0.90 / 1.00 / 0.98 |
| 6 | 0.1 | 0.97 | 0.00 / 0.02 / 0.02 | 0.01 | 1.00 / 0.95 / 0.99 |
| 6 | 0.2 | 0.86 | 0.02 / 0.03 / 0.06 | 0.02 | 1.00 / 0.94 / 0.97 |
| 6 | 0.3 | 0.70 | 0.05 / 0.05 / 0.15 | 0.02 | 1.00 / 0.86 / 0.97 |
| 6 | 0.6 | 0.33 | 0.21 / 0.23 / 0.43 | 0.02 | 1.00 / 0.92 / 0.98 |
| 6 | 1.0 | 0.09 | 0.46 / 0.49 / 0.77 | 0.03 | 0.80 / 0.85 / 0.97 |
| 7 | 0.1 | 0.96 | 0.00 / 0.00 / 0.06 | 0.01 | 1.00 / 1.00 / 0.99 |
| 7 | 0.3 | 0.80 | 0.06 / 0.05 / 0.12 | 0.01 | 0.67 / 0.90 / 0.99 |

What this shows:

- **Haar is almost complete in a window of p, Clifford never is.** The
  Haar span holds up to 97 % of the `3^N` functions at `p = 0.1`, and the
  `E_perp` of local, degree-1 and degree-2 targets drops to 0 to 0.06. The
  Clifford span stays at `2^N` strings, of which a few percent are visible,
  and its `E_perp` stays between 0.8 and 1. The lack of completeness in
  #337 is therefore a Clifford property: commuting effects.
- **The Haar window lies below p_c.** At `p = 0` (unitary only) the span is
  the `2^N` diagonal strings moved by `U`, and `E_perp` is about 0.6.
  `E_perp` falls to almost 0 for `p = 0.05` to 0.2. Above that it rises
  steadily: about 0.05 at `p_c`, about 0.2 at `p = 0.6`, about 0.5 at
  `p = 1`. This matches the qualitative picture "complete below `p_c`,
  incompressible bias above".
- **At these sizes, nothing is sharp at p_c.** The `E_perp(p)` curves for
  `N = 4` to 7 lie on top of each other within errors, and the rise starts
  before `p_c`. Telling a transition from a crossover needs larger `N`,
  which the exact span cannot reach (`N = 7` Haar already takes about 4 min
  per `p`).
- **The mid-record purification shows up in the Haar ridge.** With the mid
  record alone, the degree-1 `E_perp` is 0.28 at `p = 0.1` (`N = 6`) and
  0.10 at `N = 7`, against 0.02 and 0.00 with the final readout. From
  `p ≈ 0.3` upwards, the mid record alone reaches the same `E_perp` as the
  full record. Below `p_c` part of the information waits for the final
  readout; above it, it has already been read.
- **Local capture.** The fraction `dim(Span ∩ V_2) / 3^2` of the functions
  of a two-site window peaks at 0.94 (`N = 6`) and 0.98 (`N = 7`) at
  `p = 0.1`. It falls to the Clifford level (about 0.2) above `p ≈ 0.6`.

## 2. Locality of the Clifford dictionary (N = 16 to 256)

`clifford_ridge.py locality` computes the entropy `S(l)` of every interval
of the references after the full readout. It averages over the `N` starting
sites and over 400 circuits (150 at `N = 256`), at `T = 2N`. The
`interval_entropies` function updates a GF(2) rank site by site; it takes
about 1 s per circuit at `N = 256`. Figure:
`figures/dictionary_locality.png`.

How to read it: `2^(l - S(l))` strings of `G` fit in a window of `l` sites.
`S(l)` is therefore the number of input bits of the window that the
dictionary cannot represent with functions confined to the window. In the
clipped gauge, each generator of `G` is one representable function of
extent `l_g`, and the density of generators of length `l` is
`rho(l) = -2 S''(l)`.

- **Map of `1 - S(l)/l` over `(p, l)`** (top left). Below `p_c`, the
  representable fraction stays small up to windows of the size of the
  system. Above `p_c`, it exceeds 0.9 from a few sites on. The iso-lines
  bend around `p_c`.
- **`S(l)`** (top right). At `p = 0.04`, `S(l)` grows linearly with `l`
  (volume law). At `p = 0.16`, it follows `log2 l`. At `p = 0.3`, it
  saturates at 1.3 after 3 sites (area law).
- **Extent of the functions, `rho(l)`** (bottom left). Below `p_c` there is
  a peak at `l ≈ N/2`: a finite fraction of the functions is delocalised
  over the whole chain. At `p ≈ p_c`, `rho(l) ≈ 2 l^-2`, scale free. Above `p_c`
  there is an exponential cut-off, with a typical extent of 2 to 5 sites.
- **`S(N/2)` against N** (bottom right), for `N = 16, 32, 64, 128, 256`:

  | p | 0.04 | 0.10 | 0.14 | 0.16 | 0.18 | 0.22 | 0.30 | 0.50 |
  |---|---|---|---|---|---|---|---|---|
  | N = 16 | 5.85 | 4.14 | 3.25 | 2.85 | 2.50 | 1.97 | 1.25 | 0.66 |
  | N = 256 | 71.0 | 32.7 | 14.4 | 6.89 | 3.87 | 2.25 | 1.31 | 0.67 |

  The growth is close to linear for small `p`, logarithmic at `p_c`, and
  `S(N/2)` stays flat above `p_c`.

---

# Follow-up (issue #341): the ridge on record features is an exact Fourier filter

Run (about 1 h on 4 cores):

    experiments/clifford_ridge/fourier_filter.sh
    uv run python experiments/clifford_ridge/plot_fourier_filter.py outputs/fourier_filter

## Why it is exact in Clifford

Take a record feature: the mean of one bit, or the parity of several bits.
Its value is `Tr[rho(x) O_S]`. `O_S` is the `Z` of the bits, propagated
backwards through the gates (`U^dagger . U`) and through the dephasing of
every earlier measurement. The dephasing at site `k` kills any string that
does not commute with `Z_k`. So `O_S` is **one Pauli string (up to sign) or
zero** (`parity_strings`, `local_strings`). The test suite checks this
against the exact means at `N = 4`, for single bits and for pairs.

The Fourier transfer matrix `C` of the readout is therefore a selection
matrix: one mode per surviving feature. Then:

- `Sigma = C W C^T` is diagonal in the distinct visible modes, with
  eigenvalue `m_a w_a`. Here `w_a = 2^-|a|` and `m_a` is the number of
  features that carry mode `a`. The spectrum of `Sigma` comes from the
  singular values of `C W^(1/2)`, not of `C`: the factor `W` sits inside
  the filter.
- At `K = inf`, the ridge transmits mode `a` with `m w / (m w + lambda)`
  (Wiener filter). Modes outside `C` make up `E_perp`.
- At `K` shots, a mode carried by a single feature is transmitted with
  `K w / (1 + (K - 1) w)`. The noise is `(1 - w)/K` per feature, and on
  average it has no cross terms between features that carry different
  modes. A mode is therefore learned when `w ≳ 1/K`, that is
  `|a| ≲ log2 K`. In amplitude this is `c_a ~ 2^-|a|/2 ≳ 1/sqrt(K)`.

**Validation.** At `N = 16` and 32, for `p = 0.08`, 0.16 and 0.3, with 6
circuits per point, 3000 inputs and 256 shots sampled with stim, the
empirical population `R2` of every mode carried by the local features
follows the prediction. The RMS deviation over 187 (mode, `K`) pairs is
0.009 (`figures/fourier_filter.png`, bottom right).

## Results (T = 2N; bit features for N = 16 to 128, pairs of bits for N = 16 and 32)

- **The local readout keeps almost nothing.** Between 97 % and 99 % of the
  mid-circuit bits are killed: the marginal mean of a bit, averaged over
  the earlier outcomes, is independent of the input. The final bits are
  all killed at `T = 2N`. What survives is mostly the first bits measured
  on each input site.
- **Number of visible modes per site.**
  - Bit features: 0.07 at `p = 0.04`, 0.20 at `p_c` and 0.40 at
    `p = 0.5`, the same for every `N`.
  - Bits plus pairs: 0.11 to 1.7 per site at `N = 16`, and 0.13 to 3.0 at
    `N = 32`. This dictionary grows with `N`.
  - Full record `G` (#337), at `N = 16`: 44 to 112 per site.
  - The information about the input sits in parities of many bits. A linear
    readout on bit means sees only a small part of it.
- **Transfer function `C(p)`.** The bit features couple mostly to modes of
  weight 1 to 4. Below `p_c` there is a thin tail of heavy modes, up to
  weight about 12 at `N = 128`, from late bits whose string grew before it
  was read. The filter removes this tail at any realistic `K`. Above `p_c`
  the tail disappears and the low-weight modes become more frequent.
- **Targets on one or two sites (N = 128).** Below `p ≈ 0.08`, the bit
  features reach almost all of what the full record reaches. Above that,
  the full record pulls ahead: at `p = 0.3`, `E_perp` of `Z_i` is 0.944
  with bit features against 0.880 with the full record. At `p = 0.5` it is
  0.915 against 0.834. The gap opens gradually, with no break at `p_c`.
- **No optimum at p_c for the local readout.** The number of modes
  transmitted at `K` shots, `n(K) = sum K w / (1 + (K - 1) w)`, rises
  monotonically with `p` for every `K` from 1 to 4096. The best `p` is the
  largest of the grid. `n(K)` saturates from `K ≈ 256`, because the
  surviving modes have weight 8 or less.

## Reading

The ridge is a Fourier filter: `C(p)` selects the modes and the filter
`w / (w + lambda)`, or `K w / (1 + (K - 1) w)`, weighs them. In a Clifford
QELM this picture is exact and verified. The MIPT acts on `C` and not on
the filter. Below `p_c`, the few modes that the bit features read include
long strings with exponentially small weight. Above `p_c`, the modes are
short and more numerous. But for the local readout, every one of these
quantities varies smoothly and does not depend on `N`. The trace of the
transition seen in #337 and #339, through `Delta` and the locality of `G`,
requires reading parities of many bits, not bit means.

## Readout by parities of at most k bits

`fourier_filter.py parities` propagates every parity of at most `k` bits
(`parity_strings`, vectorised). The dictionary of order `k` is the union of
the distinct visible strings that these parities reach. The runs use
`N = 8` with `k <= 5` (200 circuits), `N = 12` with `k <= 4` (60 circuits)
and `N = 16` with `k <= 3` (60 circuits), at `T = 2N`, for `p` from 0.04
to 0.3 (`figures/parities.png`).

Fraction of the visible strings of `G` that are reached:

| N | k | p = 0.04 | 0.08 | 0.12 | 0.16 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 0.026 | 0.032 | 0.045 | 0.054 | 0.057 | 0.065 | 0.075 |
| 8 | 3 | 0.136 | 0.108 | 0.138 | 0.165 | 0.204 | 0.241 | 0.297 |
| 8 | 5 | 0.428 | 0.246 | 0.242 | 0.281 | 0.336 | 0.412 | 0.498 |
| 12 | 4 | 0.023 | 0.037 | 0.054 | 0.072 | 0.120 | 0.158 | 0.190 |
| 16 | 3 | 0.003 | 0.008 | 0.014 | 0.020 | 0.029 | 0.032 | 0.047 |

What this shows:

- **Above p_c, each extra order k helps a lot.** At `p = 0.3`, the
  fraction of `G` roughly doubles with each order (`N = 12`: 0.024, 0.063,
  0.120, 0.190 for `k = 1..4`). The number of modes transmitted at
  `K = 64` follows the same trend (2.2 per site at `N = 12`, `k <= 4`,
  against 0.27 for bit features).
- **Below p_c, a second branch appears only once `k` approaches `N/2`.**
  At `N = 8` (`N/2 = 4`), from `k = 3` on, the fraction of `G` stops
  decreasing towards small `p` and rises again: 0.43 at `p = 0.04` for
  `k <= 5`, against a minimum of 0.24 near `p = 0.1`. `n(K = 64)` does the
  same. At `N = 12` (`k <= 4 < N/2`) and `N = 16` (`k <= 3`), everything
  stays monotonic in `p`. This agrees with the picture in which the
  information below `p_c` sits in parities of about `N/2` bits, mostly
  from the final readout, whose bits are all killed one by one. Two
  caveats apply. At `p = 0.04` and `N = 8` there are only about 13 bits,
  so `k <= 5` already covers about 29 % of all parities. And the minimum
  sits near `p ≈ 0.1`, not exactly at `p_c`: we see two branches, not a
  clean optimum at `p_c`.
- **A possible kink just above p_c.** At `N = 12` (`k <= 4`) and `N = 16`
  (`k <= 3`), `n(K = 64)` rises faster between `p = 0.16` and 0.2 than
  elsewhere (`N = 12`: 0.73 to 1.51). For bit features the rise is smooth.
  With 7 values of `p` and 60 circuits this is only an indication.
- **`E_perp` of `Z_i` barely moves beyond `k = 2`** (`N = 16`,
  `p = 0.3`: 0.947, 0.921, 0.908). One-site targets are reached mostly by
  pairs of bits.
- **Cost.** A readout of order `k` has `C(n_bits, k)` features. Reaching
  `k ~ N/2` means an exponential number of features, which is out of reach
  both for the exact enumeration here (`N >= 12`) and for a ridge trained
  on a finite number of inputs. The transition can be read in the ridge
  only with a readout whose complexity matches the entanglement of the
  code. A non-linear decoder, or parities chosen from the structure of
  `G`, would be the realistic way to get there.
