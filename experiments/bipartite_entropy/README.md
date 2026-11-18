# Bipartite entanglement of the effect dictionary: crossing, collapse and `(p_c, nu)` map (issue #358)

With the full final readout, the effects of the QELM have rank one,
`E_m = |phi_m><phi_m|` with `phi_m = A_m^dagger |m_f>`. The normalised
`psi_m` is the state of the reference qubits after the readout. Its
entanglement profile `S(l)` sets the locality of the dictionary of functions
that the ridge can represent: `2^(l - S(l))` of them fit in a window of `l`
sites in Clifford (#337, #339). #339 showed this profile qualitatively. Here
`S(N/2)` gets the same quantitative analysis that the purification order
parameter `Delta` got in #346 (Clifford) and #348 (Haar): crossings, data
collapse, and a quality map over `(p_c, nu)`. It is a complement, on the
bipartite entropy, to the purification work.

## Experiment overview

| | |
| --- | --- |
| Scripts | `experiments/bipartite_entropy/run.py`, launcher `run_all.sh`, analysis `analyse.py` |
| Module | `QELM.evaluation.bipartite_entropy` |
| Physical model | periodic brickwork of #337 / #346 / #348: half-layer `t` applies two-qubit gates (Haar on U(4), or random Clifford) on `periodic_bonds(N, t)`, then a `Z` measurement of each site with probability `p`; `T = 2N` half-layers, then a `Z` readout of every site |
| System sizes `N` | Clifford 16, 32, 64, 128, 256; Haar 8, 10, 12, 14, 16, 18, 20 |
| Samples per `p` | Clifford 1000, 1000, 600, 400, 150; Haar 2000 (`N <= 14`), 1000, 500, 160 |
| Control parameter | `p`: 19 values from 0.04 to 0.5 (Clifford), 18 from 0.04 to 0.4 (Haar) |
| Observables | `S(l)` of the references, `l = 1 .. N/2`, averaged over the position of the interval (von Neumann in bits; Renyi-2 too for Haar); `I3` of the four quarters |
| Issues / PRs | #358; builds on #337, #339 (`interval_entropies`, PR #340), #346, #348, #352 |

## Methodology

1. **One sample** is one circuit and one record. The record is drawn with the
   probability `Tr(E_m) / 2^N` of the maximally mixed input, as in
   #337 / #348.
2. **Clifford**: the group of the effect with the full record is computed
   with stim (`effect_groups(...).full`). Entropies come from GF(2) ranks:
   `S_A = |A| - k_A`, with `k_A` the rank of the subgroup supported on `A`.
3. **Haar**: `psi_m` is a monitored trajectory run backwards (#352, here for
   the periodic brickwork). It starts from a uniform bit string `m_f`. Then,
   for `t = T - 1 .. 0`, the sites measured after half-layer `t` are
   projected by the Born rule of the backward state, and the adjoint gates of
   half-layer `t` are applied. This gives the record exactly the probability
   `Tr(E_m) / 2^N` (tested against the dense forward Kraus operator at
   `N = 4`). It needs a state vector of `N` qubits, against `2^N x 2^N` for
   the Kraus operator of #348. That is why Haar reaches `N = 20` here, against
   12 for `Delta`.
4. **Profile**: `S(l)` of `[s, s + l)` modulo `N`, averaged over the starting
   site `s`. For Haar at `N >= 16` a subset of starts is used (stride 2 at
   `N = 18`, 3 at `N = 16`, 5 at `N = 20`). **The start parity matters.**
   When `N/2` is even, both ends of the half chain have the parity of `s`.
   The last gate layer of the backward trajectory (the first forward
   half-layer, even bonds) crosses only one type of cut. A first `N = 16` run
   with only even starts gave an area-law `S(N/2)` of 0.6 instead of 1.05
   at `p = 0.4`, so `run.py` now refuses a stride that does not balance the
   two parities.
5. **Order parameters.**
   - `S(N/2)` grows as `log N` at `p_c`. Its scaling form is that of Li, Chen &
     Fisher, PRB 100, 134306 (2019): `S(p, N) - S(p_c, N) = F((p - p_c) N^(1/nu))`.
     The collapse uses a window around `p_c` (0.10 to 0.24 for Clifford, 0.10 to
     0.30 for Haar): deep in the volume law, `S ~ s(p) N` and the tiny error
     bars make a fit over every `p` follow the corrections far from `p_c`.
   - `I3` is scale invariant at `p_c`, so its curves cross there and
     `I3 = F((p - p_c) N^(1/nu))`.
6. **Cost**: the Houdayer-Hartmann quality `Q` of #346 / #348
   (`collapse_quality`, with `subtract=True` for `S`), on a wide grid and on a
   zoom. The best pair is a Nelder-Mead minimum, with 40 parametric bootstrap
   resamples (statistical errors only).
7. **Growth exponent** `alpha(p) = log(S_b / S_a) / log(N_b / N_a)`, between
   neighbouring sizes.

How to run it:

    experiments/bipartite_entropy/run_all.sh                # about 5 h on 4 cores
    uv run python experiments/bipartite_entropy/analyse.py \
        --runs outputs/bipartite_entropy

`run.py` resumes a run from the `p` already written. Each `p` has its own
seed, so a restart gives the same samples. The runs of this report (14 MB,
per-sample values) are kept in the project files, not in git. `analyse.py`
writes `figures/` and `summary.json` (every fit, crossing, profile and
`alpha`).

## Results and discussion

### Clifford (`N = 16` to 256)

- **Profiles** (`profiles.png`, a). At `N = 256`, `S(l)` is linear in `l` at
  `p = 0.04` (`S(N/2) = 70.8`). It bends at `p = 0.1` (32.5) and is almost
  flat beyond a few sites from `p = 0.2` on (2.8). At `p >= 0.3` it saturates
  at the same value for every `N` (1.3 at `p = 0.3`, 0.87 at `p = 0.4`):
  area law.
- **Growth exponent** (`profiles.png`, b). `alpha` goes from 0.9 at
  `p = 0.04` to 0 above `p ≈ 0.19`. The drop sharpens with `N`: at
  `p = 0.17`, `alpha = 0.39, 0.24, 0.20, 0.09` for `16→32` up to `128→256`.
  At `p = 0.16` it is 0.46, 0.34, 0.24, 0.28: it decreases slowly with `N`, as
  `S ~ a log N + b` requires.
- **Raw crossings of `S(N/2)`** (`order.png`, a, c). There is no clean
  crossing: neighbouring sizes touch in the flat area-law region (0.29 and
  0.42 for the smallest pairs), and the crossing drifts down towards `p_c`
  as `N` grows (0.216 for `64/128`, 0.189 for `128/256`). This is expected
  from the `log N` at `p_c`.
- **Crossings of `I3`** (`order.png`, b, c): 0.152(1), 0.159(1),
  0.158(1), 0.160(1) for `16/32` to `128/256`. They lie on the Gullans-Huse
  value 0.1598 from `N = 32` on.
- **Collapse and maps** (`collapse.png`, `collapse_map.png`,
  `collapse_map_variants.png`):

  | data | `p_c` | `nu` | `Q_min` |
  |---|---|---|---|
  | `I3`, `N = 16`-256 | 0.1590(3) | 1.37(1) | 4.2 |
  | `I3`, `N = 32`-256 | 0.1599(4) | 1.42(1) | 4.0 |
  | `I3`, `0.10 <= p <= 0.24` | 0.1588(3) | 1.28(1) | 1.5 |
  | `S(N/2) - S(p_c)`, `0.10 <= p <= 0.24` | 0.1630(11) | 1.15(2) | 4.1 |
  | same, `N = 32`-256 | 0.1641(12) | 1.21(3) | 2.6 |
  | `S(N/2) - S(p_c)`, every `p` | 0.174(1) | 1.31(3) | 55 |

  - `I3` gives `p_c = 0.159 to 0.160` and `nu = 1.28 to 1.42`. This is the
    `Delta` result of #346 (0.161, 1.30) and Gullans & Huse (0.1598, 1.30).
    In the windowed case, `Q` at the Gullans-Huse pair is 2.2, against
    `Q_min = 1.5`.
  - The `S(N/2)` collapse gives the same picture with a slight bias:
    `p_c = 0.163 to 0.164` and `nu = 1.15 to 1.21`. The valley is tilted in
    `(p_c, nu)`, and the `Delta` pair lies on its edge. Over every `p` the
    collapse fails (`Q = 55`), for the reason given in Methodology, step 5.
  - The spread between the data choices is larger than the bootstrap
    errors, so the real uncertainty is systematic: about ±0.001 on `p_c`
    and ±0.1 on `nu` for `I3`.

### Haar (`N = 8` to 20)

- **Profiles.** At `N = 20`, `S(l)` grows almost linearly at `p = 0.04`
  (`S(N/2) = 7.5`, close to the Page value 9.3) and saturates after 3 to 4
  sites from `p = 0.3` on (1.55 at 0.3, 1.08 at 0.4). The area-law value
  still grows slightly with `N` (1.28 at `N = 8`, 1.55 at `N = 20`, at
  `p = 0.3`).
- **Growth exponent.** `alpha` goes from about 1 at `p = 0.04` to 0.1-0.2
  at `p = 0.4`. The drop is broad and the curves of successive pairs do not
  sharpen visibly at these sizes. At `p = 0.17`, `alpha = 0.76` for `8→10`
  and 0.55 for `18→20`.
- **Raw crossings of `S(N/2)`.** Only far above `p_c` (0.29 for `18/20`,
  0.35 to 0.38 for other pairs), in the region where the curves converge.
- **Crossings of `I3`**: 0.187(4) for `8/12`, 0.199(7) for `12/16`,
  0.166(14) for `16/20`. With Renyi-2: 0.183(3), 0.186(11), 0.161(12).
  They lie in the window 0.16 to 0.20, around the Zabalo et al. value
  0.168(5) and the `Delta` crossings of #348 (0.17 to 0.19).
- **Collapse and maps:**

  | data | `p_c` | `nu` | `Q_min` |
  |---|---|---|---|
  | `I3` (`S_1`), `N = 8, 12, 16, 20` | 0.1888(10) | 1.05(2) | 6.7 |
  | `I3` (`S_2`) | 0.1809(9) | 1.03(1) | 7.8 |
  | `S(N/2) - S(p_c)`, `0.10 <= p <= 0.30` | 0.153(4) | 0.63(2) | 1.3 |
  | same, `N >= 10` | 0.241(10) | 1.03(4) | 1.3 |
  | `S_2(N/2) - S_2(p_c)` | 0.153(3) | 0.65(2) | 1.5 |

  - `I3` places `p_c` at 0.18 to 0.19, the same place as `Delta` in #348
    (0.181 at `T = 2N`, `N = 4`-12), slightly above the literature 0.168.
    Its `nu = 1.0 to 1.05` is below `Delta`'s 1.6 and the literature
    1.2 to 1.3. With four sizes, of which only `N = 16, 20` are large, this
    does not settle `nu`.
  - **The `S(N/2)` collapse does not constrain `(p_c, nu)` for Haar at
    `N <= 20`.** The quality map (`collapse_map.png`, bottom left) is a long
    flat valley, with `Q ≈ 1.3` from `(0.15, 0.6)` to `(0.25, 1.0)`.
    Dropping `N = 8` moves the optimum from one end to the other. The reason
    shows in `collapse.png`: in this range of `p` and `N`, `S(N/2) - S(p_c, N)`
    is almost linear in `p` with a slope that grows slowly with `N`, so a
    whole family of `(p_c, nu)` lines the sizes up. The minimum at
    `nu = 0.63` must not be read as a measurement of `nu`.

### Comparison with the purification entropy

| | `p_c` (bipartite) | `nu` (bipartite) | `p_c` (`Delta`) | `nu` (`Delta`) | literature |
|---|---|---|---|---|---|
| Clifford | 0.159-0.160 (`I3`), 0.163 (`S`) | 1.3-1.4 (`I3`), 1.15-1.2 (`S`) | 0.161 (#346) | 1.30 | 0.1598, 1.30 (Gullans & Huse) |
| Haar | 0.18-0.19 (`I3`), not constrained (`S`) | ~1.0 (`I3`) | 0.181 (#348) | 1.59 | 0.168(5), 1.2(2) (Zabalo et al.) |

- **Clifford**: the bipartite entropy of the dictionary and the
  purification of the references give the same transition. The `p_c`
  values agree to 0.002 and the `nu` values to 0.1. The locality transition
  of the dictionary is the MIPT: same point, same universality class.
- **Haar**: the `p_c` agree (0.18 to 0.19 from both at `N <= 20` and
  `N <= 12`), with the same small-size drift above the literature value.
  `nu` is not determined at these sizes by either probe. They bracket the
  literature value (about 1.0 from `I3`, 1.6 from `Delta`).

## Figures

- `figures/summary_table.png` (`summary_table.py`): `p_c` and `nu` from the
  collapses of `Delta` (#346, #348), `S(N/2)` and `I3`, with the literature,
  for Clifford and Haar. Each cell is the main collapse, value(bootstrap
  error), and below it the range over the data variants.
- `figures/order.png`: (a) `S(N/2)` against `p` for every `N` (crosses:
  crossings of neighbouring sizes; orange band: literature `p_c`). (b) `I3`
  of the quarters against `p`. (c) Crossing of neighbouring sizes against
  `1 / mean(N)`, for `I3` and for `S(N/2)`. Top row Clifford, bottom row
  Haar.
- `figures/collapse.png`: `S(N/2) - S(p_c, N)` (left) and `I3` (right)
  against `(p - p_c) N^(1/nu)` at the best pair of each.
- `figures/collapse_map.png`: `Q` over `(p_c, nu)`, log colour scale, for
  `S(N/2)` (wide, zoom) and `I3` (wide, zoom). White contours: `1.5` and
  `2 Q_min` (a visual guide, not a confidence region). Red dot: best pair
  with bootstrap errors. Orange markers: literature and `Delta` pairs.
- `figures/collapse_map_variants.png`: zoomed maps for other data choices
  (smallest size dropped, every `p`, Renyi-2 for Haar, `I3` window for
  Clifford).
- `figures/profiles.png`: (a) `S(l)` against `l`, `l = 1 .. N/2`, at the
  largest `N`, for eight values of `p` (blue: small `p`, red: large `p`).
  (b) `alpha(p)` for each pair of neighbouring sizes.
