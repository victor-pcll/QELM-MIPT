# Le spectre du dictionnaire G(t=0) à travers la MIPT

*Note du 16 novembre 2026, issue #356. Code : `src/QELM/evaluation/dictionary_spectrum.py`,
scripts et figures dans [`experiments/dictionary_spectrum/`](../../experiments/dictionary_spectrum/README.md).
Chaque conclusion porte une étiquette : **exact** (démontré et testé),
**observé** (mesuré, aux tailles indiquées) ou **conjecture**.*

## La question

Une proposition extérieure suggérait de tracer le « spectre » du dictionnaire,
c'est-à-dire de l'ensemble des fonctions de l'entrée que la lecture du record
peut représenter à `K = ∞`, pour voir d'un seul coup d'œil :

1. sa taille (la somme du spectre donnerait le rang) ;
2. sa localité (la forme du spectre : piqué vers `l ≈ 3N/4` sous `p_c`,
   replié vers les petits `l` au-dessus de `p_c` en Clifford ; plat sous
   `p_c` et à décroissance exponentielle au-dessus en Haar).

Nous avons tracé les deux, sur nos circuits. Le point 1 est juste. Le point 2
est faux dans la forme annoncée, en Clifford comme en Haar.

## 1. Clifford : l'histogramme N(l)

**Définition.** Pour un circuit de Clifford,
`E_m = Tr(E_m) 2^-r Σ_{g ∈ G} s_g(m) g^T`. Le groupe `G` (`2^r` chaînes de
Pauli) ne dépend pas du record, seuls les signes en dépendent (#337). Pour
chaque circuit, on énumère les `2^r` chaînes de `G` et on compte leurs poids
de Hamming `l = |g|` (nombre de sites où `g ≠ I`). `N(l)` est la moyenne sur
les circuits. On trace aussi `N_vis(l)`, les chaînes sans `Y`, seules vues par
l'encodage angulaire.

Circuit de #337 : brickwork périodique, une demi-couche par pas, mesure `Z`
de chaque site avec probabilité `p` après chaque pas, `T = 2N`,
`p_c = 0,161`. Tailles : `N = 12, 16` (1000 circuits), `20` (400), `24` (60).

**Résultats.**

- **Exact.** `Σ_l N(l) = 2^N` avec la lecture finale, à tout `p` ;
  `Σ_l N(l) = 2^(N − Δ)` sans elle. Le script le vérifie circuit par circuit.
  La marche en `1 − Δ/N` de la figure (panneau en bas à gauche) est donc la
  courbe de purification elle-même. Attention : c'est la dimension de
  l'espace engendré (chaînes avec `Y` comprises). La somme de `N_vis(l)` est
  le rang vu par l'encodage angulaire, bien plus petit (695 sur 65 536 à
  `N = 16`, `p = 0,04`).
- **Exact.** `N_vis(l)` est le spectre de la covariance `F` du record (#334),
  niveau par niveau : `F` est diagonale sur les chaînes visibles de `G`, de
  valeur propre `2^-l`. Le test le vérifie (profils de span et de Born
  égaux et entiers).
- **Observé : sous `p_c`, l'histogramme est celui de chaînes aléatoires.**
  À `p = 0,04`, `N(l)` suit `2^N C(N,l) 3^l / 4^N` (pic à `3N/4`,
  `<l>/N = 0,742` pour `N = 16` à `24`). Les chaînes visibles suivent
  `2^l` au lieu de `3^l` (pic à `2N/3`).
- **Observé : au-dessus de `p_c`, le volume ne se replie pas vers les petits
  `l`.** Le pic passe de `3N/4` à environ `0,66 N` (`p = 0,3`) puis `0,62 N`
  (`p = 0,5`). La limite où tout est mesuré, les `2^N` chaînes `Z`, aurait
  un pic à `N/2`. La quasi-totalité des `2^N` chaînes reste de poids
  extensif.

  | p | <l>/N, N = 16 | N = 20 | N = 24 | pic, N = 20 |
  |---|---|---|---|---|
  | 0,04 | 0,742 | 0,743 | 0,743 | 15 |
  | 0,16 | 0,706 | 0,708 | 0,706 | 15 |
  | 0,30 | 0,660 | 0,662 | 0,664 | 14 |
  | 0,50 | 0,623 | 0,627 | 0,621 | 13 |

- **Observé : `<l>/N` ne voit pas la transition.** Les courbes de `N = 12`
  à `24` sont superposées à tout `p`. Rien ne s'aiguise à `p_c`.
- **Observé : ce qui change, c'est une queue locale.** Le nombre de chaînes
  d'un seul site, `N(1)/N`, passe de 0,03 (`p = 0,04`) à 0,17 (`p_c`) puis
  0,35 (`p = 0,3`), sans dépendre de `N`. Ce sont `O(N)` chaînes sur `2^N` :
  invisibles sur l'histogramme linéaire, visibles seulement en échelle log
  (colonne du milieu de la figure). Le nombre de chaînes de poids `≤ 3` par
  site croise entre tailles près de `p_c`. **Conjecture** : ce croisement
  vient du mélange de deux effets qui n'ont rien de critique (la queue du
  modèle aléatoire qui décroît avec `N` sous `p_c`, et les produits de deux
  chaînes locales éloignées, en `N²`, au-dessus) ; nous ne le présentons pas
  comme un paramètre d'ordre.

**Lecture.** L'histogramme `N(l)` montre bien la taille (exact) et, en
échelle log, la naissance d'une queue locale. Mais la localité du
dictionnaire n'est pas une propriété du *volume* de `G` : un groupe engendré
par `N` générateurs locaux contient surtout des produits de beaucoup de
générateurs, donc de poids extensif. Le bon objet pour la localité reste le
profil `S(l)` des générateurs (#339), où la loi de volume et la loi d'aire
se lisent directement.

## 2. Haar : le spectre de F

**Définition.** Sans structure de groupe, on prend la matrice du dictionnaire
pondérée par Born : lignes = records tirés selon Born, colonnes = modes de
Pauli `a`, coefficients `v_a(m) = sqrt(w_a) c_a(m) / c_0(m)`. Ses valeurs
singulières au carré sont les valeurs propres `μ_k` de `F = E_m[v vᵀ]`
(#334) : les corrélations canoniques au carré entre un tir du record et
l'entrée. On trace `μ_k` triées. On calcule aussi le rang exact du span, le
nombre de `μ_k > λ` pour `λ = 10^-2, 10^-3, 10^-4`, le rapport de
participation `(Tr F)² / Tr F²` et la part de `Tr F` dans les 10 premiers
modes. Deux profils en degré prolongent `N(l)` :

- `P(l) = Σ_{|a| = l} Π_aa`, avec `Π` le projecteur sur le span ; il somme
  au rang ;
- `N_F(l) = Σ_{|a| = l} E_m[(c_a / c_0)²]`, où chaque mode compte pour son
  amplitude de lecture au carré, dans `[0, 1]`.

**Exact.** Pour des portes de Clifford, `P(l) = N_F(l) = N_vis(l)`, et le
spectre de `F` est un escalier `2^-l` de multiplicités `N_vis(l)`.

Géométrie de #334 : brickwork ouvert, une couche = liens pairs, liens
impairs, puis mesures ; `T = 2N` ; `p_c = 0,29` dans cette convention
(#322), donc `p = 0,3` est ici au point critique, et nous ajoutons
`p = 0,5`. Les mêmes masques avec des portes de Clifford servent de
référence. Tailles : `N = 4` (12 graines, 2000 records) et `N = 6`
(4 graines, 6000 records).

**Résultats (`N = 6`, lecture finale).**

| p | rang / 728 | μ_1 | μ > 10^-3 | rapport de participation | part des 10 premiers |
|---|---|---|---|---|---|
| 0,04 | 0,31 | 0,02 | 0,26 | 123 | 0,17 |
| 0,10 | 0,92 | 0,16 | 0,34 | 60 | 0,29 |
| 0,16 | 0,87 | 0,16 | 0,32 | 53 | 0,32 |
| 0,30 | 0,61 | 0,20 | 0,26 | 36 | 0,43 |
| 0,50 | 0,53 | 0,28 | 0,22 | 26 | 0,49 |

- **Observé : le spectre se concentre quand `p` augmente, mais sans
  transition.** À `p = 0,04` il est plat (`μ ≈ 10^-2`) jusqu'à son rang
  `2^N = 256` puis tombe à zéro. En montant `p`, le haut monte et la queue
  s'incline : c'est le pivot déjà vu en #334. La part des 10 premiers modes
  croît de façon monotone et lisse ; l'essentiel du changement a lieu sous
  `p = 0,1`, loin de `p_c`. Rien de particulier à `p_c = 0,29`.
- **Observé : pas de chute exponentielle sur quelques modes.** À `p = 0,5`,
  38 % des modes ont encore `μ > 10^-4`. Le contraste « plat sous `p_c`,
  exponentiel au-dessus » n'apparaît pas sous cette forme. À `N = 4`, la
  tendance est faible et non monotone.
- **Observé : le rang a son maximum à `p ≈ 0,1`**, pas à `p_c` (le min-cut
  de #321). Un seuil `λ` ne change pas ce constat.
- **Observé : `N_F(l)` passe d'un profil plat en degré (`p = 0,04` : environ
  1,6 % des modes lus à chaque degré) à un profil penché vers les bas degrés
  (`p = 0,5` : 12 % au degré 1, 1,2 % au degré 6).** C'est l'analogue Haar de
  la queue locale de Clifford, et il évolue lui aussi continûment.
- **Observé : Haar et Clifford vont en sens inverse pour le rapport de
  participation.** Il baisse avec `p` en Haar (le spectre se concentre) et
  monte en Clifford (de nouvelles chaînes locales de grande valeur propre
  apparaissent).

**Ce que cela confirme.** Nous avions déjà trouvé que la ridge ne lit pas la
transition de Haar (#334, #352). Le spectre du dictionnaire est cohérent avec
cela : il change de forme, mais de façon lisse et sans signature à `p_c`, aux
tailles accessibles (`N ≤ 6`). Ces tailles ne permettent pas de distinguer
une transition d'un crossover.

## Ce qu'il faut retenir pour le manuscrit

- **Exact** : la somme de `N(l)` est `2^(N − Δ)` sans lecture finale et
  `2^N` avec. La figure de la taille du dictionnaire est la courbe de
  purification.
- **Exact** : en Clifford, l'histogramme des chaînes visibles est le spectre
  de `F`.
- **Observé** : la forme de `N(l)` ne se replie pas vers les petits `l`
  au-dessus de `p_c`. Le volume reste à poids extensif. Seule une queue
  locale de `O(N)` chaînes apparaît, et elle croît sans singularité à `p_c`.
- **Observé** : en Haar (`N ≤ 6`), le spectre de `F` se concentre de façon
  lisse avec `p`, sans signature à `p_c`, et sans décroissance exponentielle.
- **À ne pas écrire** : « la distribution est piquée sur `l ≤ ξ(p)` au-dessus
  de `p_c` » et « le spectre de Haar chute exponentiellement au-dessus de
  `p_c` ». Nos données les contredisent.
- La localité qui distingue les deux phases se lit sur `S(l)` (#339), pas
  sur l'histogramme des poids de tout le groupe.
