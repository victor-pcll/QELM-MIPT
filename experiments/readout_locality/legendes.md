# Légendes : localité du readout, `E_readout(k)` en fonction de `p` et `k` (issue #386)

Cadre commun : limite de population (`K = ∞`, `λ → 0+`). Toutes les figures
montrent l'erreur de readout relative
`ε(k) = E_readout(k) / (1 − E_circuit) = 1 − R²(k)/R²(∞)`. C'est la part de
l'information transmise par le circuit que le Ridge sur les parités d'au plus
`k` bits ne lit pas.

Pour Clifford (brickwork périodique de #337, lecture finale `Z`), `ε(k)` se
lit sur le poids minimal de parité `w(g)` de chaque chaîne visible `g` du
groupe d'effets `G` : `ε(k)` est la part des chaînes avec `w > k`. Prior
flat : chaque chaîne compte pareil. Prior iid : chaque chaîne compte
`2^{-|g|}`. `w` est calculé exactement par programmation dynamique.

Les masques sont emboîtés : pour un circuit donné, mêmes portes et même tirage
`u` pour tous les `p`. Ligne pointillée verticale : `p_c = 0.161` (Clifford,
`Delta`, #346).

## `readout_profile.png` — profil en `k`

Un panneau par `N` (6 à 12, `T = 2N`). Abscisse : `k / N`. Ordonnée : `ε(k)`
moyen sur les circuits. Trait plein : prior flat ; tirets : prior iid. Une
couleur par `p`. À `p = 0`, on voit une marche en `k = N/2` (`w` binomial). Le
profil s'étale vers les grands `k/N` pour `p` proche de 0.1 à 0.15, et
s'étale davantage quand `N` croît. À `p = 0.5`, il redescend vers `k/N ≈ 0.5`.

## `readout_vs_p.png` — `ε(k)` en fonction de `p`

Colonnes : `k = 1, 2, 3, N/2, N`. Lignes : prior flat, puis prior iid. Une
couleur par `N`. Barres : erreur standard sur les circuits. À `k` fixe,
`ε(k)` est proche de 1 et croît avec `N`. Pour `k ∝ N`, `ε` présente un
maximum vers `p ≈ 0.1` qui grandit avec `N`. Les courbes se croisent, mais
les croisements dérivent avec `N` (pour `k = N`, prior iid : 0.30, 0.24,
0.18).

## `mean_weight.png` — poids minimal moyen

**Gauche.** `⟨w⟩/N` (moyenne sur les chaînes visibles de `G`, puis sur les
circuits) en fonction de `p`. Le maximum est à `p_max` = 0.165, 0.152 et
0.141 pour `N` = 8, 10 et 12, avec une hauteur 1.03, 1.22 et 1.39.

**Milieu.** Exposant local `d ln⟨w⟩ / d ln N` entre tailles consécutives. Il
vaut 1.6 à 1.85 sous `p_c` et environ 1 ou moins à grand `p`. Tirets :
exposants 1 et 2.

**Droite.** `⟨w | |g|⟩` en fonction du support `|g|` à `N = 12`. À
`p ≤ 0.2`, il sature dès `|g| ≈ 4` : la lecture est non locale. À
`p ≥ 0.3`, il croît avec `|g|` : la lecture est locale.

## `depth.png` — contrôle en profondeur

`⟨w⟩` en fonction de `p` pour plusieurs `T` (`N = 8` : `T = N, 2N, 4N, 8N` ;
`N = 10` : `T = N, 2N, 4N`). 200 circuits par courbe (400 et 300 pour
`T = 2N`). Sous `p_c`, `⟨w⟩` croît avec `T` : il faut annuler le déphasage
de chaque mesure sur l'histoire de l'opérateur. Pour `p ≥ 0.3`, `⟨w⟩` ne
dépend plus de `T`.

## `haar_vs_clifford.png` — Haar et Clifford à `N = 4`

Brickwork ouvert de `fourier_framework`, `T = 8`. Haar (bleu) et Clifford
(orange) sur les mêmes masques. `ε(k)` pour `k = 1` à 4 (styles de trait),
calculé par les traces des projecteurs : rapport des rangs pour le prior
flat, trace pondérée par `w_a` pour le prior iid. 48 circuits par point. Les
spans Haar saturent dès `k = 3` (`ε(4) ≤ 0.03`), alors que Clifford garde
`ε(3)` entre 0.2 et 0.36.
