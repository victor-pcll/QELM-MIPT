# Légendes : budget R² en fonction de p, Clifford et Haar (issue #360)

Cadre commun : limite de population (`K = ∞`, ensemble d'entraînement infini,
`λ → 0+`, donc `E_estimation = 0`). Cibles synthétiques construites dans la
base des monômes de Bloch, `y = Σ_{a ≠ I} β_a B_a(x)` sur les `3^N − 1`
chaînes sans `Y` (entrées XZ, poids `w_a = 2^{-|a|}`), donc
`E_encoding = 0`. Encodage dense initialisé de façon homogène :
`β_a ~ N(0, 1)` i.i.d. (fichiers `_iid`). Variante `_flat` :
`√w_a β_a ~ N(0, 1)`, même puissance pour chaque chaîne. Le budget est
`1 = E_circuit + E_readout(k) + R²_pop(k)`, par Pythagore pondéré avec les
projecteurs `W`-orthogonaux `Π_C` (toutes les features du record) et
`Π_C^(k)` (parités d'au plus `k` bits).

## `budget_hist_k2_iid.png` — décodeur Ridge `k = 2`

`N = 4`, brickwork ouvert, `T = 4`, lecture finale `Z`. Une colonne par `p`
(0 à 0.5), une ligne par terme : `E_circuit`, `E_readout(2)`, `R²_pop(2)`.
Haar (bleu) et Clifford (orange) sur le même histogramme : mêmes masques de
mesure et mêmes cibles, seules les portes changent. Chaque histogramme
regroupe 300 masques × 40 cibles ; ordonnée : fraction des échantillons ;
trait vertical : moyenne. Clifford : `E_circuit` est un pic étroit vers
0.93, qui ne bouge pas avec `p`. Haar : `E_circuit` passe de 0.75 (`p = 0`)
à 0.21 (`p = 0.3`) puis remonte à 0.30 ; `E_readout(2)` est maximal
(≈ 0.34) vers `p = 0.2` ; `R²(2)` monte de 0.17 à 0.55.

## `budget_hist_kinf_iid.png` — décodeur Ridge `k = ∞`

Même chose avec le décodeur sur toutes les parités du record : sa span est
celle de toutes les features, donc `E_readout(∞) = 0` exactement (pas
d'erreur de readout) et `R²(∞) = 1 − E_circuit`. Deux lignes : `E_circuit`
et `R²(∞)`. Haar atteint `R²(∞) ≈ 0.79` en moyenne à `p = 0.3`, Clifford
reste vers 0.06–0.08. L'histogramme Haar est multimodal : chaque mode
correspond à un rang de l'espace des effets (15, 31, 63, 71 ou 80 = toutes
les chaînes visibles à `N = 4`).

## `budget_mean_iid.png` — budget moyen empilé

Moyenne des trois termes en fonction de `p`, barres empilées (rouge
`E_circuit`, jaune `E_readout(k)`, vert `R²_pop(k)`) ; à chaque `p`, Haar à
gauche et Clifford (hachuré) à droite. Trois panneaux : `k = 1`, `k = 2`,
`k = ∞`. Le passage de `k = 1` à `k = ∞` ne change que la répartition entre
jaune et vert : `E_circuit` ne dépend pas du décodeur.

## `clifford_sizes_iid.png` — Clifford à plus grand `N`

Modèle de #337 (brickwork périodique, mesure après chaque demi-couche,
`T = 2N`), `N = 8` (200 circuits) et `N = 12` (100 circuits), 40 cibles par
circuit. Les parts sont minuscules, donc axe logarithmique : médiane (ligne)
et bande 5–95 %. Panneaux : `R²(∞) = 1 − E_circuit`, `R²(2)`,
`E_readout(2)`. Calcul exact par comptage des chaînes dans `G` et `D_k`.
`R²(∞)` décroît avec `N` (moyenne `iid` à `p = 0.5` : 1.8 % à `N = 8`,
0.33 % à `N = 12`) et croît doucement avec `p`, sans signature à
`p_c = 0.161`.

## Fichiers `_flat`

Mêmes figures pour `√w_a β_a ~ N(0, 1)`. Pour Clifford, `R²(∞)` vaut alors en
moyenne la fraction de chaînes visibles dans `G`, au plus
`(2^N − 1)/(3^N − 1)`.

## JSON

`dense_N4_<prior>.json`, `clifford_N<N>_<prior>.json` : configuration, et
pour chaque `p` l'histogramme (40 classes sur `[0, 1]`), la moyenne,
l'écart-type et les quantiles 5/25/50/75/95 % de chaque terme, ainsi que
le plus grand terme croisé et le plus grand écart `|somme − 1|` (au plus
`3.8e-10`).

## `rank_levels_iid.png` — structure des rangs en Haar (issue #362)

Mêmes 300 circuits Haar × 11 valeurs de `p` que les histogrammes (rejoués
depuis les seeds, vérifiés sur les échantillons stockés). (a) Fraction des
circuits à chaque rang visible de l'espace des effets (15, 31, 63, 71, 80),
en fonction de `p`. (b) `E_circuit` de chaque circuit (moyenne sur ses 40
cibles), tous `p` confondus, en fonction de son rang ; trait noir : moyenne
par niveau (0.75, 0.52, 0.16, 0.09, 0) ; losanges : prédiction
`1 − Σ_a w_a [Π_C]_aa / Σ_a w_a`. (c) `E_circuit` moyen mesuré (noir) et
reconstruit comme Σ fraction(p) × valeur du niveau (orange) : les deux
courbes se superposent. (d) Coupe minimale `C` du graphe espace-temps
(`log2` du rang dense, exact pour chaque circuit Haar), et ses deux bornes :
`N + K_eff` (nombre de bits de record utiles) et `2N − n_1` (mesures de la
première couche). `C` moyen culmine à 6.0 à `p = 0.3`. Lecture : le rang
plein (80) n'est atteint que par au plus 11 % des circuits ; l'optimum
vient de la distribution des niveaux, poussée vers le haut par les records
puis rabattue par les mesures précoces de la première couche.

## `ecirc_hist_by_rank_iid.png` — histogrammes colorés par rang

Les histogrammes Haar de `E_circuit` (`budget_hist_k2_iid.png`, première
ligne), chaque échantillon coloré par le rang de son circuit : chaque bosse
est un niveau de rang.

Fichiers `_flat` : mêmes figures pour `√w_a β_a ~ N(0, 1)` ; losanges
`1 − rang / (3^N − 1)`.
