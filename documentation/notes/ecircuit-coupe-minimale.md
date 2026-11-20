# E_circuit et coupe minimale : la démonstration, avec ses hypothèses

*Note du 20 novembre 2026, issue #383. Elle part de la démonstration de Victor
et précise chaque hypothèse. Les chiffres viennent de
`experiments/mincut_pedagogy` (figure 5), de #362 et de #321.*

**Résultat.** Pour un circuit de Haar à masque fixe et une cible isotrope en
coordonnées blanchies :

```
E[R²(∞)] = d / M,     E_circuit = 1 − d / M,     d ≤ min(2^C − 1, M),
```

où `C` est la coupe minimale du graphe espace-temps et `M` le nombre de modes non
constants visibles. En dense, et quand le rang vaut `2^C`, on a
`d = 2^C − 1`, donc `log₂ R²(∞) ≈ C − 2N`.

## 1. Le problème d'apprentissage est une projection

L'entrée `x` est encodée dans `ρ(x)`. Le record `m` a la probabilité
`Pr(m | x) = Tr[E_m ρ(x)]`. Une readout linéaire du record prédit

```
f̂(x) = Σ_m w_m Pr(m | x) = Tr[(Σ_m w_m E_m) ρ(x)],
```

donc toute prédiction est `Tr[O ρ(x)]`, avec `O` dans `𝒮 = span{E_m}`. Avec une
infinité de tirs (`K → ∞`), une infinité d'entrées d'apprentissage et `λ → 0⁺`,
la ridge donne la projection orthogonale de la cible sur `𝒮` (#360).

**Le produit scalaire.** C'est celui de `L²` pour la loi des entrées :
`⟨f, g⟩ = E_x[f(x) g(x)]`. Les monômes de Bloch (une chaîne de Pauli `a` par
monôme) y sont orthogonaux, mais pas normés : `E[B_a²] = w_a`, avec
`w_a = 3^(−|a|)` en dense (sphère) et `2^(−|a|)` en XZ (`|a|` est le poids de la
chaîne). On travaille donc en coordonnées blanchies `g_a = √w_a β_a`, où la base
est orthonormée. Avec `Π` le projecteur orthogonal (dans ces coordonnées) sur
l'image de `𝒮` :

```
R²(∞) = ‖Π g‖² / ‖g‖²,      E_circuit = 1 − R²(∞).
```

## 2. Cible isotrope : `E[R²] = d/M`

On tire `g` gaussien isotrope sur les `M` modes non constants visibles : c'est le
prior `flat` de `r2_budget`, où chaque mode porte la même variance. Si `Π` est de
rang `d` sur ces modes, `‖Πg‖² / ‖g‖²` suit une loi `Beta(d/2, (M − d)/2)`, de
moyenne `d/M`. Le résultat est exact, et pas seulement à l'ordre dominant. Sa
variance vaut `2d(M − d) / (M²(M + 2))` : elle est petite quand `M` est grand.
C'est pourquoi les niveaux de la figure 5 sont si fins (dispersion d'au plus
0,006 en dense à `N = 4, 5`).

**Ce n'est plus vrai pour une cible non isotrope.** Avec le prior `iid`
(`β_a ~ N(0, 1)`, donc une variance `w_a` par mode), `E[R²]` dépend de **quels**
modes sont dans `𝒮`, et pas seulement de leur nombre. Par exemple, à `N = 4` en
XZ, au niveau `C = 4` : `E_circuit = 0,749` en `iid`, contre `0,812 = 1 − 15/80`
en `flat` (#362). Une cible structurée, par exemple de bas degré, peut être bien
mieux ou bien moins bien servie que `d/M`.

## 3. Le décompte des dimensions

**Dénominateur `M`.** La variance ignore le mode constant (la chaîne identité,
qui donne la fonction 1). Il reste donc `M = 4^N − 1` en dense, et
`M = 3^N − 1` en XZ, parce que l'encodage d'angle ne voit pas `Y`.

**Numérateur `d`.** Les effets forment un POVM complet, `Σ_m E_m = I`, donc
l'identité est toujours dans `𝒮`. En dense, toutes les chaînes sont visibles et
`d = dim 𝒮 − 1 = rang_dense − 1`. En XZ, `d` est le rang de la **projection** de
`𝒮` sur les chaînes sans `Y`, moins 1. Il vaut au plus `min(rang_dense − 1, M)`,
mais il peut être en dessous des deux :

| `N` | `C` | `2^C − 1` | `M` (XZ) | `d` observé |
| --- | --- | --- | --- | --- |
| 4 | 7 | 127 | 80 | 71 ou 80 |
| 5 | 7 | 127 | 242 | 95 ou 127 |
| 5 | 8 | 255 | 242 | 143, 191, 215 ou 242 |

## 4. Le rang et la coupe minimale

**Démontré : `rang_dense ≤ 2^C`.** La matrice des records contre les monômes est
un réseau de tenseurs sur le graphe espace-temps. Un fil libre porte 2 bits, un
fil qui touche une mesure en porte 1, et chaque record ou lecture finale en
porte 1. Sans lecture finale, il faut retirer ce qui est hors du passé causal
des records, parce qu'une porte dont les sorties sont toutes tracées est
l'identité (`QELM.evaluation.mincut_graph`). Donc
`R²(∞) ≤ (2^C − 1)/M` pour toute cible isotrope.

**Observé, pas démontré : `rang_dense = 2^C`.**
- Avec lecture finale, l'égalité tient dans tous les cas testés de Haar, soit
  plus de 6 000 (#321, #362, #383).
- Elle est fausse pour les portes structurées : Clifford reste à `2^N`, Z₂ tombe
  à environ la moitié de `2^C`, et U(1) est plus bas (#321).
- Sans lecture finale, même après l'élagage, 1 % (`T = 2N`) à 3,5 % (`T = N`)
  des masques donnent un rang sous `2^C`, qui n'est alors pas une puissance de 2
  (60 au lieu de 64, 12 au lieu de 16). Tous ont des mesures dans la dernière
  couche mesurée.

Quand l'égalité tient, en dense : `d = 2^C − 1` et
**`E_circuit = 1 − (2^C − 1)/(4^N − 1)`**. La figure 5 le vérifie niveau par
niveau, à 0,005 près (`N = 4, 5`).

## 5. `log₂ R²(∞) ≈ C − 2N` et ses limites

En dense :

```
R²(∞) = (2^C − 1)/(2^(2N) − 1),
log₂ R²(∞) − (C − 2N) = log₂(1 − 2^(−C)) − log₂(1 − 4^(−N)).
```

L'écart vaut −0,42 à `C = 2` et −0,19 à `C = 3`, puis tend vers 0 quand `C`
grandit. Perdre un bit multiplie `R²` par `(2^(C−1) − 1)/(2^C − 1)`. Ce facteur
est un peu **moins** que 1/2, et vaut 1/3 entre `C = 2` et `C = 1` : la
division par 2 n'est exacte qu'asymptotiquement.

**La coupe maximale `C = 2N` (`R² = 1`) n'est pas toujours atteignable.** Elle
demande qu'aucun site ne soit mesuré à la première couche (`n₁ = 0`, sinon
`C ≤ 2N − n₁`), et assez de records : `K ≥ 2N` sans lecture finale,
`K_eff ≥ N` avec. Dans les données de la figure 5 (`p ≤ 0,5`, `T = 2N`, sans
lecture finale), le maximum observé est `2N − 1`, d'où le plancher
`E_circuit ≈ 0,5`.

## 6. Ce qu'il faut en retenir

1. `E_circuit` est fixé par la géométrie du masque, au travers de `C` : c'est une
   fonction de comptage, du type Rényi 0 (voir la comparaison avec Skinner,
   Ruhman et Nahum dans `experiments/mincut_pedagogy`).
2. La dépendance en `p` de `E_circuit` vient entièrement de la distribution de
   `C` sur les masques : en dense, la prédiction suit la moyenne à 0,003 près.
3. Les écarts ont trois sources connues : la troncature XZ, les rares masques
   sans lecture finale où le rang reste sous `2^C`, et les cibles non isotropes.
4. Comme le rang, ce bilan est à nombre de tirs infini et à masque fixe. La MIPT
   est à chercher dans les quantités pondérées par la règle de Born (note
   `transition-rang-mipt.md`).
