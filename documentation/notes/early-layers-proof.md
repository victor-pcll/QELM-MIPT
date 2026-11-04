# The early-layers hypothesis of the linear QELM on M: what is proven (issue #288)

**Hypothesis.** To guess the label of the reference qubit from the measurement record `M`, the linear readout (ridge on `vec(M)`) only uses the first rounds (`t = 1, 2, 3`), before Haar scrambling hides the initial state, and ignores the late ones (`t = 15, 16` at `N = 8`, `T = 2N`).

**Short answer.** For Haar U(4) gates the hypothesis holds in the sense that matters, and that part is a theorem: no decoder of the late rounds alone, linear or not, can beat chance by more than `D_k / 2`, where `D_k` is a trace distance that never grows with `k` and that we compute exactly. For the ten circuits studied, it is below `10^-3` at `k = 15` for every `p >= 0.2`. The covariance of each late entry of `M` with the label is bounded by `p D_k`, so the late rounds carry at most a `10^-5` to `10^-10` share of the linear model's signal. But the literal claim, that the trained ridge puts **small weights** on the late rounds, is false: their weight norm is 15 to 50 % of that of round 1. These weights fit sampling noise. Blanking them changes nothing, but they are not small. Also, the late rounds do help the **optimal** decoder, together with the early ones (`p = 0.2`: Bayes goes from 0.60 with 3 rounds to 0.69 with 16). Only the linear readout cannot use that help. With charge-conserving U(1) gates, the hypothesis fails, and that is also a theorem.

Code: `src/QELM/evaluation/early_layers.py` (tested in `tests/test_early_layers.py`), experiment and figures in [`experiments/early_layers/`](../../experiments/early_layers/README.md).

## Setting

- Chain of `N` sites and a reference qubit `R` in `|Psi_0> = (|0>_R |a> + |1>_R |b>) / sqrt 2`, with `<a|b> = 0`. The encodings are `bell` (`|a>`, `|b>` differ on the middle site), `ghz`, `charge` and `w`.
- One fixed circuit: in round `t` the gates `U_t` act on the chain, then each site `i` is measured in Z with probability `p`, by an independent coin. `m_{t,i}` is `+-1`, or `0` if the site is not measured.
- The label `Y` is the Z outcome of `R`.
- The ridge's score is `s(M) = beta + sum_{t,i} w_{t,i} m_{t,i}`, with `w_t` the weights of round `t`.

**Lemma 0 (the label can be read first).** `Z_R` commutes with every operation on the chain, so measuring `R` at the start or at the end gives the same joint law of `(M, Y)`. Given `Y = +1` the chain starts in `|a>`, given `Y = -1` in `|b>`, each with probability 1/2.

**Lemma 1 (averaged states).** Let `rho_t^±` be the chain state right after the gates of round `t`, averaged over every coin and outcome of rounds `1 .. t-1`, starting from `|a>` or `|b>`. Write `D_p` for the dephasing `rho -> (1 - p) rho + p sum_m Pi_m rho Pi_m` on every site. Then `rho_{t+1}^± = U_{t+1} D_p(rho_t^±) U_{t+1}^dagger`. Summing a Z measurement over its outcomes gives the dephasing, and averaging over the coin gives `D_p`. Because the later coins are independent, **the marginal law of the rows `k .. T` of `M` given `Y` is that of the same protocol started at round `k` from `rho_k^Y`.** Write `Delta_t = rho_t^+ - rho_t^-` and `D_t = ||Delta_t||_1 / 2`.

## 1. Rounds `k .. T` alone: a bound for every decoder (proven, every circuit, every `p`)

**Theorem 1.** For every decoder `f` (linear or not, randomized or not) that sees only the rows `k .. T` of `M`:

    P(f = Y) <= (1 + D_k) / 2,    with 1 = D_1 >= D_2 >= ... >= D_T.

*Proof.* By Lemma 1 the rows `k .. T` come from a fixed instrument, plus independent randomness, applied to `rho_k^Y`. Any decision built on them is therefore a two-outcome POVM on the chain at round `k`. By Helstrom's theorem with equal priors, its success is at most `1/2 + ||rho_k^+ - rho_k^-||_1 / 4`. The monotonicity is data processing: `rho_{k+1}^±` are images of `rho_k^±` under one CPTP map, and the trace norm contracts under CPTP maps. `D_1 = 1` because `U_1` keeps `|a>` and `|b>` orthogonal. ∎

`D_k` is computed exactly, not sampled: `suffix_bounds` propagates two `2^N x 2^N` matrices. For the Bell encoding, the ten Haar circuits, `N = 8` and `T = 16`, `D_15` averaged over the circuits is:

| p | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.6 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| `D_15` | 0.21 | 2.8e-2 | 4.1e-3 | 8.8e-4 | 1.0e-4 | 2.0e-5 | 9.6e-7 | 2.2e-8 |
| `D_4` | 0.87 | 0.72 | 0.58 | 0.45 | 0.27 | 0.16 | 0.07 | 0.02 |

So for `p >= 0.2`, no function of rounds 15 and 16 beats `0.5005`, and the ridge retrained on rounds `k .. 16` is at chance from `k = 3` or `4` on (figure `windows_*`). At `p = 0.05` the bound is weak (`D_15 = 0.21`): at small `p` the theorem does not exclude a role for the late rounds.

**Corollary 1 (the linear signal of each entry).** Write `c_{t,i} = E[m_{t,i} Y]`. Then

    c_{t,i} = (p / 2) Tr(Z_i Delta_t),    |c_{t,i}| <= p D_t.

*Proof.* The coin is independent and the earlier Z measurements of the same round commute with `Z_i`, so `E[m_{t,i} | Y] = p Tr(Z_i rho_t^Y)`. Also `|Tr(Z Delta)| <= ||Z||_inf ||Delta||_1`. ∎

So, for **any** linear score, `E[Y] = 0` gives `Cov(s, Y) = sum_t w_t . c_t` exactly, and

    |w_t . c_t| <= p D_t ||w_t||_1.

We call `share_t = w_t . c_t / Cov(s, Y)` the share of round `t` in the signal of the model. The shares add up to 1. For the ridges trained here (Bell, Haar), the share of rounds 1 to 3, the share of rounds 15 and 16, and the bound above on the latter are:

| p | 0.05 | 0.1 | 0.2 | 0.4 | 1.0 |
|---|---|---|---|---|---|
| share of rounds 1-3 | 0.80 | 0.90 | 0.95 | 0.99 | 1.00 |
| share of rounds 15-16 | 2e-3 | -1e-4 | 4e-6 | 2e-7 | -1e-10 |
| proven bound on the latter | 0.85 | 0.09 | 2e-3 | 3e-5 | 3e-8 |

The bound uses the measured `||w_t||_1` and the exact `D_t`. It is a theorem about the trained model, not an estimate.

**What this does not say: the weights themselves.** The population ridge is `w = (Sigma + alpha)^-1 c`. Through the covariance `Sigma` of the record, a late entry can receive weight without carrying any signal, for example to cancel noise that it shares with early entries. Theorem 1 does not bound `||w_t||`. In practice the weight norm of the late rounds is **15 to 50 % of that of round 1** (figure `weights_*`, left). These are weights fitted to the sampling noise of a finite training set. Their share of the signal is zero up to `10^-4`, and blanking rounds `> 3` (replacing them by their training mean) never lowers the test accuracy: it raises it by 0.003 to 0.023 depending on `p` (Bell, `p = 1`: 0.656 instead of 0.643). The weight figure alone would therefore *not* support the hypothesis. The right quantities are the signal share, which is bounded, and the ablation, which is measured.

## 2. What the late rounds add to the optimal decoder (proven, estimated by sampling)

**Theorem 2.** Let `r_k = <Z_R>` given the first `k` rows and the circuit. Write `A_k = <0|_R Psi_k>` and `B_k = <1|_R Psi_k>` for the two branches of the joint state after round `k`, and `H_k = sqrt(1 - 4 |<A_k|B_k>|^2)`. Then:

- the Bayes accuracy of the first `k` rounds is `(1 + E|r_k|) / 2`;
- **every** decoder of the whole record, and even any final measurement of the chain, has accuracy at most `(1 + E[H_k]) / 2`.

So the rounds after `k` add at most `E[H_k - |r_k|] / 2`.

*Proof.* Given the prefix `m`, the chain carries the unnormalized branches `|A><A|` (for `Y = +1`) and `|B><B|` (for `Y = -1`). Everything that follows is a POVM on the chain, so its success is at most `(1 + || |A><A| - |B><B| ||_1) / 2` (Helstrom with priors `|A|^2`, `|B|^2`). This rank-2 operator has eigenvalues `(|A|^2 - |B|^2)/2 ± sqrt(1/4 - |<A|B>|^2)`, so its trace norm is `H`. Averaging over `m` gives the bound. The optimal decision from the prefix alone is `sign(r_k)`. ∎

The expectations are sample means over the trajectories, so this bound comes with a statistical error. At `p = 1` every site is measured in round 1, and `E[H_1] = E|r_1|`: **after round 1 nothing can be added**, by any decoder. At `p = 0.4`, rounds 4 to 16 can add at most 0.21. At `p = 0.2`, the Bayes decoder does use the late rounds: it goes from 0.60 (3 rounds) to 0.69 (16 rounds). The late rounds are useless alone (Theorem 1) but informative *together with* the early ones. That joint information lives in correlations between rounds that the ridge cannot represent (#278 found that the products `m_a m_b` do not capture it either). This is one more piece of the gap between the linear QELM and Bayes.

## 3. Haar average: exponential decay (proven, average over circuits)

Expand `Delta_t = 2^-N sum_P a_P(t) P` in Pauli strings, and let `mu_t = sum_P a_P(t)^2` (so `mu_1 = 2^N Tr(Delta^2) = 2^{N+1}`).

**Lemma 2 (Markov chain).** Let every gate be independent and Haar on U(4) (a 2-design). Then the second moments `E[a_P^2]` evolve by a closed linear map:

- a gate keeps the mass on the identity of its pair and spreads the rest uniformly over the 15 non-identity Paulis of the pair;
- `D_p` multiplies the mass of each `X` or `Y` factor by `(1 - p)^2`.

The map only depends on the type (`I`, `Z`, `X/Y`) on each site, so it is an exact chain on `3^N` configurations (`haar_pauli_chain`). It is checked against 600 sampled circuits in the tests. The cross moments vanish because, for a 2-design, `E[(U x U)(P x Q)(U x U)^dagger] = 0` for `P != Q`.

**Theorem 3.** For an even `N` on an open chain, `E[mu_{t+1}] <= g(p) E[mu_t]` with

    g(p) = (1 + 2 (1 - p)^2) / 3,

hence

    E[D_t] <= sqrt(E[mu_t]) / 2 <= 2^{(N-1)/2} g(p)^{(t-1)/2},
    E[c_{t,i}^2] <= (p^2 / 4) 2^{N+1} g(p)^{t-1}.

*Proof.* The gates keep the total mass. It is enough to follow one unit of mass, a random string that the gates resample pair by pair, and to bound the average of its dephasing factor after the last half-layer of a round.

- If the string meets a gate of the odd half-layer, that pair is uniform over the 15 Paulis. Its average factor is `(3 + 8q + 4q^2) / 15`, with `q = (1 - p)^2`, and the other sites only lower it.
- Otherwise its support lies on the two edge sites. Each of them came out of an even-half-layer gate with its partner at `I`, so it is uniform over `Z, X, Y`, with factor `(1 + 2q) / 3`, which is the larger of the two.

The first inequality on `D_t` is `||Delta||_1 <= 2^{N/2} ||Delta||_2` plus Jensen. ∎

This bound is rigorous but loose by the factor `2^{N/2}`: at `N = 8` it falls below 1 at `t = 16` only for `p > 0.24`. The exact chain is tight: it reproduces the RMS of `c_{t,i}` of the ten circuits round by round (figure `bounds_*`, right, dashed lines).

## 4. With a conserved charge the hypothesis fails (proven)

**Proposition 4.** Suppose every gate commutes with `Q = sum_i (1 - Z_i)/2` (U(1) gates). Then `Tr(Q Delta_t) = <a|Q|a> - <b|Q|b> =: dQ` for every `t`, and

    sum_i c_{t,i} = -p dQ  for every round t,    D_t >= |dQ| / (2N).

*Proof.* `Q` commutes with the gates, and `D_p` leaves `Tr(Q .)` unchanged because `Q` is diagonal. Moreover `sum_i Z_i = N - 2Q` and `Tr(Delta) = 0`. Finally, `|Tr(Q Delta)| <= ||Q||_inf ||Delta||_1 = 2 N D`. ∎

For the Bell and charge encodings `|dQ| = 1`. Every round, the 16th as much as the 1st, then carries the same total linear signal `p`, and Theorem 1 gives no decay. Figures `*_u1_*` show what the ridge does with it. The early-layers property is therefore not a general property of monitored circuits. It is a consequence of the absence of a conservation law in the Haar U(4) dynamics.

## What is proven and what is only numerical

| Statement | Status |
|---|---|
| No decoder of rounds `k .. T` beats `(1 + D_k) / 2`; `D_k` decreasing (Th. 1) | proven, every circuit and `p` |
| `D_k` of the ten circuits (table above) | exact computation, no sampling |
| `\|E[m_{t,i} Y]\| <= p D_t`; signal share of late rounds `<= p D_t \|\|w_t\|\|_1 / Cov` (Cor. 1) | proven for any linear readout |
| What late rounds add to Bayes `<= E[H_k - \|r_k\|] / 2` (Th. 2) | proven; the expectation is a sample mean |
| Haar average: decay at least like `g(p)^{t/2}` (Th. 3) | proven, loose; the exact chain is tight |
| U(1) gates: every round keeps signal `p`, `D_t >= 1/(2N)` (Prop. 4) | proven |
| The trained ridge has small weights on late rounds | **false**: 15 to 50 % of round 1, fitted to noise |
| Blanking rounds `> 3` never lowers the ridge's accuracy (it gains 0.003 to 0.023); the ridge retrained on rounds 1..3 is as good as the full one | numerical (10 circuits, 4000 trajectories per `p`) |
| Small `p` (`<= 0.1`): rounds 4 to 10 still carry 10 to 20 % of the signal | numerical; Theorem 1 is weak there |
