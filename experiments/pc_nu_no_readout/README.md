# (p_c, nu) without final readout, from the Fourier weights (#378)

New runs of the three observables Delta, S_2(N/2) and I3, closer to the
literature circuits than #346, #348 and #358:

- **Circuit**: periodic brickwork, a Z measurement of each site with
  probability p after every half-layer, **no final readout**.
- **Clifford**: N = 32, 64, 128, 256 at T = 4N (Gullans & Huse PRX 2020 read
  at t = 4L).
- **Haar**: N = 4, 6, 8, 10, 12 at T = 4N (Zabalo et al. 2020).
- T = N and 2N are recorded on the way.

## Observables

Every observable is the Renyi-2 entropy of a region A of the references,
`S_2(A) = |A| - log2(1 + W_A)`. Here `W_A` is the Fourier (Pauli) weight of
the effect `E = K^dagger K` on the strings supported in A
(`src/QELM/evaluation/no_readout.py`). W gives purities only, so only n = 2 is
available.

| Key | Quantity |
| --- | --- |
| `delta` | S_2 of all the references: the Delta of #314, #346, #348 |
| `s_half` | S_2 of N/2 consecutive references, mean over the N starts |
| `i3` | I3 of the quarters A, B, C (Renyi-2), mean over the N starts; only when 4 divides N |
| `s_one` | S_2 of one reference, mean over the sites: a variant of S_Q (Gullans & Huse PRL 2020), not the same protocol |

### Differences from the literature

- S_2(N/2) and I3 are those of the mixed state of the references, which
  starts from I / 2^N. Zabalo et al. compute them on the pure state of the
  system, which starts from a product state. The same p_c and nu are
  expected, but not the same curves.
- The records are drawn with the Born rule of the maximally mixed input.
- The entropy is computed per circuit and then averaged; W is never
  averaged first.

## Statistics and cost

Each sample takes one circuit and one Born record, and its seed is
`(seed, model, N, p, index)`. The per-sample values are stored, so a
bootstrap over realisations can be run later.

| Model | N | Circuits per p | CPU per circuit (T up to 4N) |
| --- | --- | --- | --- |
| Clifford | 32, 64, 128, 256 | 4000, 2000, 1000, 500 | 3.8 s at N = 256 |
| Haar | 4, 6, 8, 10, 12 | 8000, 8000, 6000, 2000, 300 | 1.0 s at N = 10, 75 s at N = 12 |

These times were measured on one core of the cloud container. At N = 12,
the run takes about 20 h on 4 cores.

## How to run

From the repository root:

    uv run python experiments/pc_nu_no_readout/run.py --model clifford --n 64
    uv run python experiments/pc_nu_no_readout/run.py --model haar --n 12

The default p grid and number of circuits are those of the table. Each run
writes `runs/<model>_N<N>.json` after every p. Launching the same command
again skips the p that are already written.
