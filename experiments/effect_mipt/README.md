# Reference MIPT probes and Born-weighted effect dimension

Does a Born-weighted version of the effect-space rank see the
measurement-induced transition? The rank of `experiments/effect_rank` is a
min cut of the space-time graph (issue #321): its peak moves to `p = 0` like
`1 / T` and it has no critical point. This folder measures, in the same layer
convention, (i) where the transition is, with two standard probes, and (ii)
the effective dimension `D_2 = 1 / Tr F^2` of the effects, which weights each
direction of the span by the Born rule (issue #322).

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/effect_mipt/effect_mipt.py` (subcommands `clifford`, `haar`, `frame`), `plot_effect_mipt.py` |
| Library | `QELM.evaluation.effect_mipt`, `QELM.evaluation.effect_rank.mincut_bits` |
| Physical model | open brickwork; one layer = even bonds, odd bonds, then a `Z` measurement of each site with probability `p` (the convention of `effect_rank`, one measurement round per **two** half-layers) |
| `clifford` | Clifford gates, `N` = 16, 24, 32, 48, 64, `T = 4N`; entropy of an initially maximally mixed system and of a reference qubit (Bell pair with the middle site, `2N` unmeasured layers first). 200 circuits per point, plus 400 (300 at `N = 48`, 200 at `N = 64`) on `p` = 0.18 to 0.32 |
| `haar` | Haar gates, `N` = 6 to 14, reference qubit, `N` unmeasured layers then `T = 2N`; 400 trajectories per point, plus 1 200 (`N` ≤ 10), 600 (`N = 12`), 500 (`N = 14`) on `p` = 0.175 to 0.35; fresh circuit and mask per trajectory |
| `frame` | Haar gates, `N` = 6, 8, 10, 12, `T = N`, 20 circuits with one fixed mask each (seeds and nested masks of `effect_rank.py`), 600 Born-drawn records per point |
| Control parameter | `p` = 0.02, 0.05, 0.1, ..., 0.5, 0.6 |
| Issues / PRs | #322; depends on #321; plan in the note of #319 |

## Methodology

1. **Mixed-state entropy (Clifford).** Each system qubit starts in a Bell
   pair with an ancilla. After the monitored layers, `S(rho_sys)` is the
   entropy of the ancillas. Its spectrum is that of the mid-circuit effect
   `K_m^dagger K_m` normalised by its trace, so it is the number of bits on
   the input that the final readout still has to supply.
2. **Reference qubit.** One reference qubit in a Bell pair with the middle
   site, unmeasured layers, then the monitored layers; `S(R)` is 1 bit while
   the system remembers the qubit and 0 once the record has purified it.
3. **Frame potential.** With the final readout each effect is
   `|phi_r><phi_r|`, `phi_r = K_m^dagger |z>`. A record is drawn with the
   Born rule from a Haar-random input (which averages to the maximally mixed
   state), then `phi_r` is computed by running `|z>` backwards.
   `Tr F^2 = E |<phi_r|phi_r'>|^4` over pairs of independent records
   (U-statistic over 600 records), and `D_2 = 1 / Tr F^2`: `2^N` for one
   basis, about `4^N / 2` for a Haar POVM. The estimator is checked against
   the enumeration of every record at `N = 3`.
4. **Crossings.** For consecutive sizes, the first sign change of the
   difference of the curves, linearly interpolated; "resolved" when the
   difference exceeds two standard errors on a grid point on each side.

## How to reproduce

```bash
for n in 16 24 32 48 64; do uv run python experiments/effect_mipt/effect_mipt.py clifford --n $n --samples 200; done
for n in 6 8 10 12 14; do uv run python experiments/effect_mipt/effect_mipt.py haar --n $n --samples 400; done
for n in 6 8 10 12; do uv run python experiments/effect_mipt/effect_mipt.py frame --n $n --seeds $(seq 0 19) --samples 600; done
uv run python experiments/effect_mipt/plot_effect_mipt.py outputs/effect_mipt
```

The finer grids are the same commands with `--p ... --seeds 1 --tag fine`.
About 3 hours on 4 cores in total; the Clifford `N = 64` runs dominate.

## Outputs

`outputs/effect_mipt/<probe>_N<n>_*.json` (not versioned; a copy is in the
project folder `effect-mipt-322/runs`). Figures and `summary.json` (every
plotted value, the crossings) in `experiments/effect_mipt/figures/`.

## Results

![Haar reference qubit](figures/haar_reference.png)

**Figure 1. Haar, entropy of the reference qubit at the last layer.** Left:
against `p`, `N = 6` to 14 from light to dark blue, error bars the standard
error. Centre: zoom on the crossing, log scale. Right: crossing of each pair
of consecutive sizes against `1 / sqrt(N N')`; circle resolved, cross not.
Dashed line: the median of the crossings.
- **The curves cross between `p = 0.27` and 0.30:** 0.295 (6/8), 0.289
  (8/10), 0.292 (10/12, not resolved), 0.274 (12/14).
- At `p = 0.275` the five sizes agree within errors (0.30 to 0.33 bit).

![Clifford mixed start](figures/clifford_mixed.png)

**Figure 2. Clifford, entropy of the initially mixed system after `T = 4N`
layers.** Same panels, `N = 16` to 64.
- **The entropy grows with `N` up to `p = 0.28`** (0.07, 0.10, 0.09, 0.17,
  0.22 bit for `N = 16` to 64) **and no longer at `p = 0.30`** (0.03, 0.02,
  0.02, 0.01, 0.03), where it is below 0.04 bit for every size.
- No pairwise crossing is resolved: beyond 0.28 the entropies are a few
  hundredths of a bit and 200 to 600 circuits do not separate the sizes.

![Clifford reference qubit](figures/clifford_reference.png)

**Figure 3. Clifford, entropy of the reference qubit after `T = 4N`
layers.** Same panels. The step sharpens with `N` and ends at
`p ≈ 0.28` to 0.30; crossings not resolved, as in Figure 2.

**Reference value.** In this convention the transition is at
`p_c = 0.29 ± 0.02` for Haar and for Clifford. It is higher than the usual
0.16 to 0.17 because there is one measurement round per two half-layers of
gates. `nu` is not determined: a two-parameter collapse runs to the edge of
any window with these sizes and samples, so none is reported.

![Rank and Born-weighted dimension](figures/frame_dimension.png)

**Figure 4. Rank and Born-weighted dimension of the same 20 Haar circuits,
`T = N`.** Colour: `N = 6` to 12. Vertical lines: the `p_c` of Figure 1
(dashed) and of Figure 2 (dotted). Error bars: standard error over circuits.
- **Left: `log2(rank / 2^N) / N`,** from the min cut of each mask.
- **Second: `log2(D_2 / 2^N) / N`.**
  - `D_2` is well below the rank: at its maximum 0.33, 0.42, 0.50, 0.57 bit
    per site for `N = 6, 8, 10, 12`, against 0.60, 0.66, 0.70, 0.75.
  - **Its maximum moves to small `p` with `N`, like that of the rank:**
    `p = 0.15`, 0.15, 0.10, 0.10.
  - **There is no crossing at `p_c`.** At `p = 0.3` the values are 0.27,
    0.33, 0.33, 0.37; the larger size is above or equal at every `p` below
    0.5.
- **Third: ratio of the two.** For `N >= 8` and `p >= 0.2` the ratio is
  the same for all sizes within a few hundredths (0.63 to 0.66 at
  `p = 0.2`, 0.58 to 0.61 at 0.3, 0.49 to 0.50 at 0.6): `D_2` follows the
  rank up to a smooth factor.
- **Right: `p` of the maximum** (grid points), `D_2` (discs) and rank (open
  squares), with the two `p_c`.

**Reading.** `D_2` behaves like the rank: a peak that follows the number of
measurements, no feature at `p_c ≈ 0.29`. Weighting the span by the Born
rule is not enough to see the transition. Of the candidates of the note of
#319, the one that does see it is the entropy of the mid-circuit effect
(Figure 2), which is the known purification order parameter.

## Per-record Fourier weight (issue #325)

For a mid-circuit record `m`, `Pr(m | x) = sum_a c_a(m) B_a(x)` on the Bloch
monomials, with `c_a` the Pauli coefficients of the effect `E_m`. The weight
outside the constant mode is `W(m) = sum_{a != 0} c_a^2 / c_0^2`. With the
dense encoding, `1 + W = 2^N Tr[E_m^2] / Tr[E_m]^2`, so the missing weight
`Delta = N - log2(1 + W)` is the Renyi-2 entropy of the effect, that is the
purification entropy: `Delta = N` when `x -> Pr(m | x)` is flat, `Delta = 0`
when the record fixes the input as well as a pure state would. The angle
encoding keeps the strings without `Y`. `fourier_weight` computes both from
the Kraus operator of a Born-drawn record (`sample_kraus`), and the `weight`
subcommand runs it on the circuits and masks of the rank runs:

```bash
uv run python experiments/effect_mipt/effect_mipt.py weight --n 8 --ratio 2 \
    --seeds $(seq 0 19) --samples 30
```

![Per-record Fourier weight](figures/fourier_weight.png)

**Figure 5. Per-record Fourier weight, Haar, fixed masks, `T = N`, `2N`,
`4N` (columns).** Colour: `N = 6` to 12. Dashed lines: the two `p_c` of
Figures 1 and 2. Error bars: standard error over circuits. `N = 6, 8`: 20
circuits, 30 records each; `N = 10`: 12 circuits and 20 records (4 and 10 at
`T = 4N`); `N = 12`: 4 circuits and 5 records, `T = N` only.
- **Top: missing weight `Delta` in bits, dense, log scale.**
  - Below `p ≈ 0.2`, `Delta` grows with `N`: at `p = 0.1`, `T = 4N`, 1.26,
    1.92, 2.44 bits for `N = 6, 8, 10`. The record leaves a number of bits
    that grows with the size.
  - Above `p ≈ 0.35`, `Delta` falls below 0.1 bit at `T = 4N` for every
    size, and below 0.25 bit at `T = 2N`.
  - **Between the two the sizes are not separated**: no pairwise crossing is
    resolved (the interpolated ones range from 0.18 to 0.48). `N <= 10` is
    too small for Haar. The same quantity for Clifford at `N = 16` to 64 is
    Figure 2 (for a stabilizer state all Renyi entropies coincide), where
    the change of trend is at `p = 0.28` to 0.30.
  - `Delta` depends on `T`: at `T = N` the pure phase has not finished
    purifying (0.8 to 1.3 bits left at `p = 0.3`).
- **Middle: mean `log2(1 + W_angle) / N`, angle encoding.** It rises as
  `Delta` falls and saturates at 0.64 bit per site, close to the
  `log2(3/2) = 0.585` of a random pure state. The sizes agree within errors:
  the angle encoding sees the same per-record weight, on a smaller scale.
- **Bottom: rank of the same circuits** (min cut). It has its peak at small
  `p`, unrelated to the fall of `Delta`.

**Reading.** In the Fourier framework, the transition is in the weight of
each row of `A` (how far one record's function is from the constant), not
in the number of modes the rows span. At Haar sizes `N <= 12` this is a
trend, not a resolved crossing; the resolved evidence is the Clifford run.

## The Fourier weight recovers the literature Clifford transition (issue #325)

To compare with the literature, the encoded input is reduced to one qubit
and the circuit is the reference-qubit probe of Gullans & Huse (PRL 125,
070606 (2020)), already in `QELM.models.circuits.stabilizer`: periodic
brickwork of random two-qubit Clifford gates, the qubit `x` entangled with
the middle site, `2N` half-layers without measurement, then `2N` half-layers
with a `Z` measurement of each site with probability `p` after every
half-layer. For one encoded qubit, `Pr(m | x) = c_0 (1 + r_m . b(x))` with
`b(x)` the Bloch vector of the input, so `W(m) = |r_m|^2` and
`log2(1 + W) = 1 - S_2` with `S_2` the Renyi-2 entropy of the reference
qubit. For a stabilizer state `W` is 0 or 1 for every record, and
`<W> = 1 - <S_R>`.

```bash
uv run python experiments/effect_mipt/clifford_weight.py --n 64 --samples 4000
uv run python experiments/effect_mipt/plot_clifford_weight.py outputs/effect_mipt
```

![Clifford Fourier weight](figures/clifford_weight.png)

**Figure 6. Fourier weight of the encoded qubit, Clifford, `N = 16, 32, 64,
128`.** 4000 trajectories per point (4500 at `N = 128`), `p = 0.08` to 0.24
in steps of 0.01. Numbers in `figures/clifford_weight_summary.json`.
- **Left: `<W>` against `p`.** Below `p_c` the record carries no weight on
  the input (the information is hidden in the volume-law state), above it
  the record fixes the input completely. The curves sharpen with `N` and
  cross at one point.
- **Middle: crossing of every pair of sizes** (zero of a weighted line
  through the difference on the four grid points around the sign change).
  All six lie between 0.158 and 0.162, with no drift with size; their
  weighted mean is 0.160, the literature value `p_c ≈ 0.16`.
- **Right: collapse against `(p - p_c) N^(1/nu)` with the literature
  `p_c = 0.16` and `nu = 1.3`**, nothing fitted. A free two-parameter
  collapse (polynomial of degree 5, `|x| < 2`) gives `p_c = 0.160`,
  `nu = 1.38`, as a check.

**Reading.** The per-record Fourier weight of the encoded input is an order
parameter of the MIPT: it is the second Renyi purification entropy, and on
the literature circuit it reproduces the known critical point and exponent.
It is the quantity of Figure 5 restricted to one encoded qubit, where
Clifford sizes are large enough to resolve the crossing.

## Dense-encoding Fourier weight at large N, Clifford (issue #327)

Back to the dense encoding on the `N` input qubits, without ancilla:
`Pr(m | x) = Tr[E_m rho(x)]` and `W(m) = sum_{a != 0} c_a^2 / c_0^2` over the
`4^N` Pauli strings, as in Figure 5. With Clifford gates and `Z`
measurements, `E_m` is proportional to a stabilizer projector of rank
`2^(N - k)`, so `1 + W(m) = 2^N Tr[E_m^2] / Tr[E_m]^2 = 2^k` exactly. As
`E_m = K^dagger K` has the spectrum of `K K^dagger`, `k` is the number of
independent stabilizers of the output of the maximally mixed input,
tracked forward without ancilla by `dense_weight_bits`: a gate conjugates
the generators; a measured `Z_j` replaces an anticommuting generator (`k`
unchanged) or else joins the group. `k` does not depend on the outcomes,
so one record per trajectory gives the exact Born average.
`Delta = N - log2(1 + W)` is the Renyi-2 purification entropy. The test
`test_dense_weight_bits_matches_pauli_coefficients` checks `k` against the
explicit Pauli coefficients of the record's effect (`fourier_weight`) for
`N = 4, 6` in both conventions.

```bash
uv run python experiments/effect_mipt/dense_weight.py periodic --n 64 --samples 2000
uv run python experiments/effect_mipt/dense_weight.py open --n 64 --samples 2000
uv run python experiments/effect_mipt/plot_dense_weight.py outputs/effect_mipt
```

Trajectories per point: 4000 (`N = 16, 32`), 2000 (`N = 64`), 800
(`N = 128`); one fresh circuit and measurement pattern each, no final
readout. Numbers in `figures/dense_summary.json`.

![Dense weight, literature circuit](figures/dense_periodic_2N.png)

**Figure 7. Literature circuit** (periodic brickwork, measurement of each
site with probability `p` after every half-layer), after `2N` half-layers.
- `log2(1 + W) / N` rises with `p` and tends to 1: one record fixes almost
  all of the dense input above `p_c`.
- `Delta` grows like `N` below the transition and falls to zero above it,
  faster for larger `N`. The curves cross at one point.
- The six pairwise crossings lie in 0.165 to 0.166, with no drift with size
  (0.165 to 0.168 after `4N` half-layers, `dense_periodic_4N.png`).
- The collapse with the literature `p_c = 0.16`, `nu = 1.3`, nothing
  fitted, puts all sizes on one curve over three decades of `Delta`. A free
  collapse gives `p_c = 0.161`, `nu = 1.30` (`2N`) and `0.161`, `1.38`
  (`4N`).
- The crossings sit 0.005 above the collapse and above the reference-qubit
  crossing of Figure 6 (0.160); their spread is far smaller than that
  offset, which is a finite-time correction of the crossing at fixed
  `t / N` rather than noise.

![Dense weight, #314 convention](figures/dense_open_2N.png)

**Figure 8. Convention of #314** (open brickwork, measurement after every
layer of two half-layers), after `2N` half-layers.
- Same picture. The pairwise crossings lie in 0.299 to 0.307 (0.297 to
  0.311 after `4N`), drifting slightly up with size, inside the
  `0.29 +- 0.02` of the reference probes of #324 (grey band).
- The collapse uses the mean crossing `p_c = 0.305` and the literature
  `nu = 1.3`; a free collapse gives `p_c = 0.302`, `nu = 1.36`.

**Reading.** In the dense encoding of the Fourier framework, the
per-record weight `W(m)` outside the constant mode is an order parameter
of the measurement-induced transition: `log2(1 + W)` is extensive and
`Delta = N - log2(1 + W)` vanishes in the pure phase, with a size-
independent crossing and a collapse at the literature exponent. The rank
of the same circuits (number of modes spanned) has no such crossing.

### Fourier cone

For Clifford gates the Fourier spectrum of one record is flat on the
stabilizer group `S_m` of its effect: `c_a(m) != 0` exactly on its `2^k`
strings, all with `c_a^2 = c_0^2`. `effect_stabilizers` obtains `S_m` by
running the same tracker on the reversed circuit `K^dagger` (Heisenberg
image of the record, no ancilla). For an interval `A` of the input,
`1 + W_A = 2^(k_A)` is the weight of the modes supported in `A`
(`interval_counts`). Both are tested against the explicit Pauli
coefficients and a brute-force count.

```bash
uv run python experiments/effect_mipt/fourier_cone.py --n 64 --p 0.12 --samples 300
uv run python experiments/effect_mipt/plot_fourier_cone.py outputs/effect_mipt
```

![Fourier cone](figures/fourier_cone.png)

**Figure 9. Fourier cone, literature circuit, `N = 64`**, 300 trajectories
per `p`, record lengths `t = 2` to `2N` half-layers, intervals at 8 random
positions per record. `I(l, t) = k(N) - k(l) - k(N - l)` counts the modes
that straddle the two ends of an interval of length `l`, that is the
Fourier weight correlating inputs on both sides.
- **Maps (top):** number of modes per end longer than `l`,
  `n(l, t) = dI/dl / 2`, log colour, and their mean range
  `xi(t) = sum_l n(l, t) / n(0, t)` (orange). At early times modes exist
  only up to a range that grows with `t` (the white wedge at the bottom
  right of each map is outside the cone). In the pure phase the cone stops
  at a finite range; in the mixed phase a tail of long modes keeps filling
  in.
- **Bottom left:** `xi(t)`. All `p` share the same initial growth; it
  saturates at `xi ≈ 2.7` sites (`p = 0.24`) and 3.6 (`p = 0.20`), and is
  still growing at `t = 2N` for `p <= 0.14`.
- **Bottom centre:** `I(N/2, t)`, the modes straddling a cut of the input in
  two halves. Same picture: a plateau of 3.9 and 5.6 modes for
  `p = 0.24, 0.20`, still rising (12 to 14 modes) for `p <= 0.12`.
- **Bottom right:** `I(N/2, 2N)` against `p`, with its steepest fall at
  `p_c = 0.16`.

**Reading.** Below `p_c` the record keeps building Fourier modes that
correlate distant inputs; above `p_c` it only builds modes of a finite
range, set by `p`. One size only: whether the plateau length diverges at
`p_c` needs `N = 32` and 128 as well.

## Caveats

- `D_2` is estimated from 600 records. Its in-sample standard error grows
  with `N` (about 0.05 bit at `N = 6`, 0.1 at `N = 8`, more beyond), and
  `log2` of a noisy mean is biased upwards in dimension. The circuit-to-
  circuit error bars of Figure 4 include this noise but not the bias.
- Sizes `N <= 12` for `D_2`: a feature that appears only at larger sizes is
  not excluded.
- The position of the maxima in Figure 4 is read on a grid of step 0.05.
- `p_c = 0.29 ± 0.02` is read from crossings, with no extrapolation in `N`.
