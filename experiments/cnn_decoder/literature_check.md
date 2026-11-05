# Literature check: can a learned readout of M see the MIPT?

A critique of our summary ("the MIPT only shows in exact-state quantities;
neither the MLP, the CNN nor a readout of the state recovers it at the sizes
tested") cites five sources. This note checks each claim against the papers
themselves and against our runs (PR #294 for the CNN, PR #295 for the state
readout). Issue #297.

The PDFs were not in the project folder (only group-meeting slides were), so
the papers were read on arXiv / ar5iv:

- Dehghani, Lavasani, Hafezi, Gullans, Nat. Commun. 14, 2918 (2023),
  [arXiv:2204.10904](https://arxiv.org/abs/2204.10904).
- Agrawal et al., PRX 14, 041012 (2024),
  [arXiv:2311.00058](https://arxiv.org/abs/2311.00058).
- Barratt, Agrawal, Potter, Gopalakrishnan, Vasseur, PRL 129, 200602 (2022),
  [arXiv:2206.12429](https://arxiv.org/abs/2206.12429).
- Kobayashi & Motome, Nat. Commun. (2025), doi 10.1038/s41467-025-58751-0,
  [arXiv:2402.07097](https://arxiv.org/abs/2402.07097) (phase transitions), and
  [arXiv:2308.00898](https://arxiv.org/abs/2308.00898) (scrambling).
- "Rudolph et al., PRX Quantum 2024": **not found**. No paper by that name
  on learnability or planted XOR turned up. The planted-XOR model is in Kim,
  von Keyserlingk, Lamacraft, [arXiv:2504.08888](https://arxiv.org/abs/2504.08888) (2025).

## Verdicts

| # | Claim of the critique | Verdict | What the source says |
| --- | --- | --- | --- |
| 1 | The MIPT lives in `P(M \| Q)`, so a decoder of `M` can see it; "only in the exact state" is wrong | **Verified**, with a caveat | Barratt 2022 and Agrawal 2024 decode the global charge from `M`; the transition they see is the **charge-sharpening** transition of U(1) circuits (`p# ≈ 0.09` in Barratt), not the entanglement / purification MIPT. For our Clifford task the same holds in our own data: Bayes on `M` has accuracy `(1 + f)/2`, whose curves cross near `p ≈ 0.12` (#294). The caveat: Bayes needs the gates `U`; Agrawal writes the decoder `D(Q \| M, U)`. |
| 2a | Dehghani's CNN on `M` recovers the transition, `L = 16` to 128, `p_c ≈ 0.16` and universal exponents | **Partly verified** | Yes, "the purification phase transition shows itself as a learnability phase transition". Sizes run from 16 to 256 overall, but the scaling collapse uses `L = 32, 48, 64` and gives `p_c ≈ 0.13` (joint fit `0.14 ± 0.03`, `ν = 1.5 ± 0.3`). 0.16 is the literature value they compare with, not their learned value. |
| 2b | Dehghani train on one fixed circuit, while ours mixes circuits | **Verified for them, does not apply to #294** | One network per fixed circuit, `N_c` random circuits per `(p, L)`. #294 does the same: one fixed circuit per `(N, seed)`, measured sites redrawn per trajectory. It does apply to #295, where the gates are new for each trajectory, but #295 reads the final state, not `M`. |
| 2c | Dehghani mask `M` to the light cone of the reference, and that is what lets the CNN converge | **Contradicted** | The phase analysis (unconditional scheme) feeds the full record `M_T`. The light-cone box is a separate test of scalability, and the conditional scheme drops outcomes after the purification time `t_p` to study complexity. They present the box as reducing the cost, not as required. |
| 2d | Training-set sizes, conditioning on `t_p` | **Checked** | `5·10³` trajectories per circuit (scalability tests); `10³` circuits for Fig. 3. The conditional scheme selects circuits by `t_p`; the phase analysis does not post-select. |
| 3 | Agrawal: a supervised network needs `\|S\| ~ e^L` to reach the optimal decoder (Fig. 2b), with RNN / LSTM / Transformers | **Partly verified** | One recurrent network (RNN), no LSTM or Transformer. Fig. 2b: the training set needed to match PostBQP "appears to scale exponentially with system size" (numerics at `L = 6, 10, 14`, not a proof). Matching the stat-mech decoder needs only a polynomial training set. The optimal (PostBQP) decoder simulates the circuit post-selected on `M`; it is not described as a Markovian Bayes filter. |
| 3' | Dehghani also find `\|S\| ~ e^L` | **Contradicted** | Dehghani find a roughly exponential growth with the **purification time** `t_p`, and a sample complexity "nearly independent of the system size" (×2 for ×8 in `L`). |
| 4 | Kobayashi & Motome: a linear readout of local 1- or 2-body observables detects nonequilibrium phase transitions (Fig. 8), so a state readout is not blind in principle | **Contradicted as applied to us** | The QRP paper detects **ground-state** quantum phase transitions (TFIM, ANNNI, spin-1 XXZ, extended SSH) after a local quench, under closed unitary dynamics, reading single-site `σ^x`. The target is the quench parameter, not a trajectory quantity. There are no monitored circuits, no mid-circuit measurements, and no MIPT. Its Fig. 8 does not exist; Fig. 8 of the scrambling paper is about memory and propagation, not transitions. #295 also explains its null result without "the wrong observable": the linear `R² = 0` is exact, because `S_R` and every `\|<P>\|` depend only on the gates and measured sites while the signs of `<P>` are random. The nonlinear readouts (`\|<P>\|`, quadratic, MLP) were also tried. |
| 5 | Barratt / Rudolph: symmetric circuits map to planted XOR; one-point averages vanish and the information is in products of signs; ridge on raw bits cannot solve it | **Source contradicted, conclusion verified for our circuits by another route** | Barratt 2022 has no planted XOR, no parity argument and no linear decoder; its mechanism is the correlation of outcomes induced by charge conservation. "Rudolph et al." was not found. Planted XOR appears in Kim, von Keyserlingk, Lamacraft (2025), on a tree. For our Clifford circuits the statement holds exactly, without any symmetry: the sign of a purified `R` is a parity of outcomes, and the ridge only reads the weight-1 ones (#289). |

## How #294 differs from Dehghani

| | Dehghani et al. 2023 | #294 |
| --- | --- | --- |
| Circuits | one network per fixed circuit, `10²` to `10³` circuits | one network per fixed circuit, 4 to 6 circuits per `N` |
| Depth | `T ~ O(10)`; the collapse uses `t_d = L/16` (2, 3, 4 at `L = 32, 48, 64`) | `T = 2N` (12 to 32) |
| Input | full `M_T` (light-cone box only for the scalability test) | full `M`, plus the mask `M²`; local 3×3 convolutions, **no light-cone crop** |
| Sizes | `L = 32` to 64 for the collapse | `N = 6` to 16 |
| Score | fraction of circuits learnt to accuracy ≥ 0.98, `S̃_Q = 1 − R_l` | mean accuracy on purified test trajectories |
| Training data | `5·10³` per circuit | `4.2·10⁴` per circuit (sweep), up to `2.1·10⁵` (scaling) |

The two differences that matter most are the **depth** and the **score**.
At `t_d = L/16` most purified trajectories purify early, through short
parities, and Dehghani's own data say the cost grows exponentially with
`t_p`. At `T = 2N` our labels include late, long parities. And `R_l` counts
the circuits a network learns almost perfectly, which tracks the purified
fraction without needing every trajectory to be decoded. The regime
`T ≈ L/16` at `L ≥ 32` was not tested here.

## Corrected conclusion

In English:

> In our benchmarks the transition is resolved only by the Bayes-optimal
> decoder of the record `M`, which knows the gates. It is therefore a
> property of `M`, not only of the exact state. At `T = 2N`, `N ≤ 16`, 4 to 6
> circuits per size and up to `2·10⁵` training trajectories, the decoders
> trained as black boxes on `M` (ridge, MLP, CNN) do not recover it. The
> ridge fails for an exact reason: the label is a parity of outcomes and a
> linear readout only reads the weight-1 ones. The CNN and the MLP gain only
> 0.02 to 0.03 per decade of data. This agrees with the exponential sample
> complexity reported by Agrawal et al. (in system size, for a recurrent
> network against the optimal decoder) and by Dehghani et al. (in
> purification time). Dehghani et al. do recover the transition with a CNN,
> at short depth (`t ≈ L/16`), `L = 32` to 64, and with the fraction of
> circuits learnt to 98 %; we did not test that regime. The linear and
> quadratic readouts of the final state (#295) do not see it either, for an
> exact reason in Clifford circuits; the quantum reservoir probing results
> of Kobayashi & Motome concern ground-state transitions under unitary
> dynamics and do not bear on this.

En français :

> Dans nos benchmarks, la transition n'est résolue que par le décodeur
> Bayes-optimal du registre `M`, qui connaît les portes. Elle est donc une
> propriété de `M`, et pas seulement de l'état exact. À `T = 2N`, `N ≤ 16`,
> 4 à 6 circuits par taille et jusqu'à `2·10⁵` trajectoires d'entraînement,
> les décodeurs entraînés en boîte noire sur `M` (ridge, MLP, CNN) ne la
> retrouvent pas. L'échec du ridge a une raison exacte : l'étiquette est une
> parité des résultats de mesure et une lecture linéaire ne lit que les
> parités de poids 1. Le CNN et le MLP ne gagnent que 0,02 à 0,03 par décade
> de données, ce qui est cohérent avec la complexité d'échantillonnage
> exponentielle rapportée par Agrawal et al. (en taille, pour un réseau
> récurrent face au décodeur optimal) et par Dehghani et al. (en temps de
> purification). Dehghani et al. retrouvent bien la transition avec un CNN,
> mais à faible profondeur (`t ≈ L/16`), pour `L = 32` à 64, et avec la
> fraction de circuits appris à 98 % ; nous n'avons pas testé ce régime. Les
> lectures linéaires et quadratiques de l'état final (#295) ne la voient pas
> non plus, pour une raison exacte dans les circuits de Clifford ; les
> résultats de quantum reservoir probing de Kobayashi & Motome portent sur
> des transitions d'état fondamental en dynamique unitaire et ne
> s'appliquent pas ici.
