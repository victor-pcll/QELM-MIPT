# Le QELM dans le langage des modes de Fourier : le cadre général (Haar), puis Clifford

*Note du 13 novembre 2026, issue #353. Elle rassemble dans un seul cadre les
résultats de #334 (PR #335), #337 (PR #338), #339 (PR #340), #341 (PR #342),
#344 (PR #345), #346 (PR #347) et #349 (PR #351), et les corrige là où ils
avaient été énoncés trop largement. Les énoncés généraux sont vérifiés
exactement à petit N par `tests/test_fourier_framework.py` (module
`QELM.evaluation.fourier_framework`) ; le tableau de la section 9 vient de
`experiments/fourier_framework/small_n_table.py`.*

**Comment lire cette note.** On construit d'abord le cadre pour un circuit
monitoré quelconque : Haar est le cas générique. On ne spécialise à Clifford
qu'à la section 7. Chaque énoncé porte une étiquette :

| Étiquette | Sens |
| --- | --- |
| **[Exact]** | démontré pour tout circuit (Haar, Clifford, U(1), …) |
| **[Clifford]** | démontré pour les circuits de Clifford seulement ; faux en général (le test donne un contre-exemple Haar) |
| **[Observé]** | mesuré numériquement, à petites tailles ; pas démontré |
| **[Conjecture]** | attendu pour des raisons physiques, ni démontré ni mesuré proprement |

---

## 0. Résumé

Le QELM à K = ∞ est une chaîne de quatre objets, et chaque objet a une
écriture simple dans la base des chaînes de Pauli (les « modes de Fourier » de
l'entrée) :

```
entrée  ──►  circuit  ──►  lecture  ──►  filtre
 w_a        Span{E_m}      parités       λ, K
 (poids)    (dictionnaire)  |S| ≤ k     (Wiener / corrélations canoniques)
```

1. **L'entrée** fixe un produit scalaire : les monômes de Bloch $B_a$ sont
   orthogonaux, de poids $w_a = 2^{-|a|}$ (entrées dans le plan XZ, chaînes
   sans Y) ou $3^{-|a|}$ (sphère de Bloch). [Exact]
2. **Le circuit** fixe l'espace $\mathrm{Span}\{E_m\}$ des fonctions que le
   record peut porter. En Heisenberg, une porte agit sur les modes par une
   matrice orthogonale $R$ et une mesure tue les modes qui anticommutent avec
   $Z$. [Exact]
3. **La lecture** choisit des parités du record. La parité $S$ a pour moyenne
   $\mathrm{Tr}[\rho(x)\,O_S]$ avec $O_S = \sum_m (-1)^{m_S} E_m$, et toutes
   les parités ensemble engendrent exactement $\mathrm{Span}\{E_m\}$
   (dualité Walsh–Pauli). [Exact]
4. **Le filtre** pondère : à K = ∞ c'est un filtre de Wiener
   $s/(s+\lambda)$ sur le spectre de $\Sigma = C W C^\top$ ; à K tirs, c'est
   $K\mu/(1+(K-1)\mu)$ sur les corrélations canoniques $\mu_k$ d'un tir avec
   l'entrée. [Exact]

Ce qui est spécial à Clifford, c'est que **chaque objet devient diagonal dans
les modes** : une porte envoie une chaîne sur une chaîne, chaque parité est
une seule chaîne, $\mathrm{Span}\{E_m\}$ est un groupe de Pauli $G$, les effets
commutent, $\dim = 2^{N-\Delta}$, et chaque erreur est une somme
d'indicatrices sur les modes. En Haar rien de cela n'est vrai : les effets ne
commutent pas, le span peut dépasser $2^N$ (jusqu'à $4^N$), et une erreur de
monôme prend n'importe quelle valeur entre 0 et 1.

**Où est la MIPT ?** La quantité centrale du Ridge à K = ∞, $\lambda \to 0^+$
($E_\perp$), ne dépend que du *span* : c'est une quantité de **rang**. En
Clifford le spectre des états est plat, donc le rang (Rényi 0) et les
entropies de Rényi 1, 2 coïncident, et le rang voit $p_c$. En Haar le rang
suit la **coupe minimale** du graphe espace-temps (égalité observée sur tous
les circuits de #321), une quantité géométrique sans transition à $p_c$, tandis que la MIPT vit dans les
entropies de Rényi $\geq 1$, c'est-à-dire dans le **spectre**. On s'attend donc
à ce que le $E_\perp$ du Ridge Haar ne voie pas $p_c$, et que la trace de la
transition, s'il y en a une, soit dans les quantités spectrales (λ > 0,
K fini). C'est une conjecture (section 6.3), compatible avec tout ce qu'on a
mesuré en Haar jusqu'ici.

---

## 1. Le cadre : entrées, modes de Fourier, features

### 1.1 Entrées produit et monômes de Bloch

L'entrée est un état produit

```math
\rho(x) = \bigotimes_{i=1}^{N} \frac{I + n_i \cdot \sigma}{2}, \qquad n_i = (n_i^X, n_i^Y, n_i^Z).
```

Une chaîne de Pauli $a = (a_1, \dots, a_N) \in \{I, X, Y, Z\}^N$ définit le
**monôme de Bloch** $B_a(x) = \prod_i n_i^{a_i}$, avec $n^I = 1$. Son poids
$|a|$ est le nombre de sites où $a_i \neq I$.

**Ensemble d'entrées.** On tire les sites indépendamment, chacun centré
($\mathbb{E}[n_i] = 0$) et de covariance diagonale
($\mathbb{E}[n_i^\alpha n_i^\beta] = c_\alpha \delta_{\alpha\beta}$). Alors

```math
\mathbb{E}[B_a B_b] = w_a\, \delta_{ab}, \qquad w_a = \prod_{i:\, a_i \ne I} c_{a_i}. \qquad \textbf{[Exact]}
```

| Ensemble | $c_X, c_Y, c_Z$ | $w_a$ | Modes visibles |
| --- | --- | --- | --- |
| angle uniforme sur le cercle XZ, ou les 4 états $\lvert 0\rangle, \lvert 1\rangle, \lvert +\rangle, \lvert -\rangle$ | 1/2, 0, 1/2 | $2^{-\lvert a\rvert}$ | $3^N$ chaînes sans Y |
| sphère de Bloch uniforme, ou les 6 états propres de Pauli | 1/3, 1/3, 1/3 | $3^{-\lvert a\rvert}$ | les $4^N$ chaînes |

Une chaîne avec $w_a = 0$ est **invisible** : $B_a = 0$ sur tout l'ensemble.
Pour des entrées dans le plan XZ, toute chaîne contenant un Y est invisible.
Ce n'est pas propre à Clifford : c'est une propriété de l'ensemble d'entrées.

*Vérifié par* `test_monomials_are_orthogonal_with_weights` (matrice de Gram
exacte sur les $4^N$ ou $6^N$ entrées, $N = 3$).

**Remarque : les monômes n'épuisent pas $L^2$.** L'espace $L^2(\mu)$ des
fonctions de l'entrée contient aussi des fonctions non multilinéaires, par
exemple $x_0^2 - z_0^2$ (dans quelle base le site 0 a été préparé). Les
features, elles, sont toujours multilinéaires (section 1.2). Une lecture
linéaire ne voit donc que la partie multilinéaire d'une cible ; le reste est
une erreur de la lecture *linéaire*, pas du circuit (#349, panneau f).

### 1.2 Effets, features et matrice de transfert

Un record $m$ est la liste des résultats d'un passage dans le circuit. Si
$K_m$ est le produit ordonné des portes et des projecteurs le long de $m$,
l'**effet** du record est $E_m = K_m^\dagger K_m$ et

```math
\Pr(m \mid x) = \mathrm{Tr}[E_m\, \rho(x)] = \sum_a C_{m a}\, B_a(x), \qquad C_{m a} = \frac{\mathrm{Tr}[E_m P_a]}{2^N}. \qquad \textbf{[Exact]}
```

La ligne $C_m$ est donc la **transformée de Fourier** (de Pauli) de la feature
$x \mapsto \Pr(m \mid x)$, et $C$ est la **matrice de transfert** du QELM.
La complétude $\sum_m E_m = I$ donne $\sum_m C_{ma} = \delta_{a, I}$.

Conséquences immédiates, vraies pour tout circuit :

- une feature est **multilinéaire** en les vecteurs de Bloch (affine en chaque
  $\rho_i$) ;
- seules les colonnes **visibles** de $C$ comptent ;
- l'espace des fonctions que le record peut porter est la projection de
  $\mathrm{Span}\{E_m\}$ sur les chaînes visibles (le « rang angle » de #313,
  #317).

*Vérifié par* `test_features_are_the_fourier_transform_of_the_effects`
(probabilités de Born exactes contre $C\,B(x)$, Haar et Clifford, $N = 3$).

### 1.3 La dynamique des modes, en Heisenberg

On calcule $E_m$ à rebours depuis la fin. Deux opérations suffisent.

**Une porte.** $U^\dagger P_a U = \sum_b R_{ab} P_b$. La matrice de transfert
de Pauli $R$ est réelle et orthogonale pour tout unitaire. **[Exact]**
Pour une porte de Clifford, $R$ est une permutation signée : une chaîne va sur
une seule chaîne. **[Clifford]** Pour une porte de Haar, chaque chaîne non
triviale s'étale sur plusieurs chaînes.

*Vérifié par* `test_pauli_transfer_matrix`.

**Une mesure Z au site k.** Avec $\Pi_\pm = (I \pm Z_k)/2$ et une chaîne
$\sigma$ :

```math
\Pi_\pm\, \sigma\, \Pi_\pm =
\begin{cases}
\tfrac12(\sigma \pm \sigma Z_k) & \text{si } [\sigma, Z_k] = 0,\\[2pt]
0 & \text{si } \{\sigma, Z_k\} = 0.
\end{cases}
\qquad \textbf{[Exact]}
```

Une mesure **tue** les modes qui anticommutent avec $Z_k$ (X ou Y au site k)
et **mélange** $\sigma$ et $\sigma Z_k$ avec le signe du résultat. Sommée sur
le résultat, c'est le déphasage $D_k(\sigma) = \sigma$ ou 0. Pondérée par
$(-1)^{m_k}$, c'est $\sigma \mapsto \sigma Z_k$ ou 0 (section 3.2).

Toute la physique du circuit est là : les portes **étalent** les modes, les
mesures **coupent** ceux qui ne commutent pas avec $Z$. La différence entre
Haar et Clifford est seulement dans l'étalement : un mode Clifford reste une
seule chaîne ; un mode Haar devient une somme de chaînes, qui interfèrent.

---

## 2. Le Ridge en population, dans la base de Fourier

On note $W = \mathrm{diag}(w_a)$, et on écrit une cible
$y = \sum_{a \ne I} \beta_a B_a$ (partie multilinéaire). Tout ce qui suit est
**[Exact]** pour tout circuit ; c'est de l'algèbre linéaire dans $L^2(\mu)$.

### 2.1 Covariance et spectre

```math
\Sigma = C\, W\, C^\top, \qquad b = \mathrm{Cov}(f, y) = C\, W \beta, \qquad \mathrm{Var}\, y = \sum_{a \ne I} w_a \beta_a^2 .
```

(colonnes $a \ne I$ seulement : la constante n'a pas de variance.) Le spectre
de $\Sigma$ est celui des valeurs singulières au carré de $C W^{1/2}$, **pas**
de $C$ : le poids de l'entrée est à l'intérieur du filtre. C'est pourquoi le
nombre de conditionnement de $X^\top X$ ou le spectre de
$\mathrm{Tr}(E_m E_{m'})$ ne sont pas les bonnes quantités (#334).

*Vérifié par* `test_ridge_spectrum_is_that_of_c_w_half_not_of_c`.

### 2.2 λ → 0⁺ : une projection pondérée

Le Ridge de population avec $\lambda \to 0^+$ est la projection
$W$-orthogonale de la cible sur l'espace des lignes de $C$. Son erreur, en
unités de $\mathrm{Var}\, y$, est

```math
E_\perp = \frac{\lVert (1 - \Pi_C)\, \beta \rVert_W^2}{\mathrm{Var}\, y} = 1 - \frac{b^\top \Sigma^+ b}{\mathrm{Var}\, y}.
```

**$E_\perp$ ne dépend que de l'espace engendré.** Deux circuits qui ont le
même span ont le même $E_\perp$, quelle que soit la façon dont le poids se
répartit dans ce span. C'est un fait simple, mais c'est le point qui sépare
Haar et Clifford (section 6.3) : **$E_\perp$ est une quantité de rang.**

*Vérifié par* `test_population_ridge_matches_brute_force` : moindres carrés
avec intercept sur les $4^N$ entrées exactes, Haar et Clifford, λ = 0 et
λ = 0,05, accord à $10^{-9}$.

### 2.3 λ > 0 : un filtre de Wiener

Dans la base propre de $\Sigma$ (valeurs propres $s_k$), le Ridge transmet la
direction $k$ avec le facteur $s_k / (s_k + \lambda)$. À λ > 0 l'erreur dépend
donc du **spectre** et plus seulement du span.

### 2.4 Ce qu'aucune lecture linéaire ne dépasse, et ce qu'une lecture non linéaire peut dépasser

- Toute statistique moyenne $\mathbb{E}[g(m) \mid x] = \sum_m g(m) \Pr(m \mid x)$
  d'une fonction du record **par tir** est dans le span. Une telle lecture, à
  n'importe quel nombre de tirs, ne fait jamais mieux que $E_\perp$. **[Exact]**
- Une fonction **non linéaire** des probabilités estimées (qui suppose
  beaucoup de tirs de la même entrée) peut faire mieux. À K = ∞, l'erreur
  minimale de tout décodeur est l'erreur de la moyenne conditionnelle sur les
  fibres $\{x : f(x) = f(x')\}$.
  - En Clifford, pour les modes de Pauli, la réponse est le théorème
    d'invisibilité : on ne fait pas mieux (section 7.4). **[Clifford]**
  - En Haar, sur un ensemble fini d'entrées, les features séparent en
    général toutes les entrées, donc un décodeur non linéaire atteint
    l'erreur 0 sur **toute** cible, alors que $E_\perp$ du Ridge reste
    positif. **[Observé]** (N = 3, 4 ; `test_haar_features_separate_inputs_beyond_the_linear_span`)

Conséquence : en Haar, $E_\perp$ est une limite de la lecture **linéaire**,
pas une limite d'information. La phrase « une lecture quelconque du record,
linéaire ou non, peut reproduire exactement Span » de
[`rang-espace-des-effets.md`](rang-espace-des-effets.md) est juste si « non
linéaire » porte sur la fonction $g(m)$ d'un tir, pas sur l'histogramme.

---

## 3. La lecture : parités du record et dualité Walsh–Pauli

### 3.1 Les parités engendrent le span

À masque fixé, un record est un mot $m \in \{0,1\}^{n_b}$. Pour un
sous-ensemble $S$ des bits, la parité $\chi_S(m) = (-1)^{\sum_{i \in S} m_i}$
a pour moyenne

```math
\mathbb{E}[\chi_S \mid x] = \mathrm{Tr}[\rho(x)\, O_S], \qquad O_S = \sum_m \chi_S(m)\, E_m. \qquad \textbf{[Exact]}
```

La transformation de Walsh $m \leftrightarrow S$ est inversible, donc
$\mathrm{Span}\{O_S\}_S = \mathrm{Span}\{E_m\}_m$ : les parités et les
probabilités des records portent exactement le même espace de fonctions. Mieux,
la matrice de Walsh $H$ vérifie $H H^\top = 2^{n_b}$, donc le Ridge sur
l'histogramme complet avec $\lambda$ est **identique** au Ridge sur toutes les
parités avec $2^{n_b}\lambda$. **[Exact]**

*Vérifié par* `test_all_parities_span_the_effects` et
`test_ridge_on_histogram_equals_ridge_on_all_parities`.

### 3.2 La forme de Heisenberg d'une parité

En reprenant la section 1.3 :

- un bit **dans** $S$ agit à rebours comme
  $A \mapsto \Pi_0 A \Pi_0 - \Pi_1 A \Pi_1 = \tfrac12\{Z_k, A\}$, soit
  $\sigma \mapsto Z_k \sigma$ si $\sigma$ commute avec $Z_k$, et 0 sinon ;
- un bit **hors** de $S$ agit comme le déphasage $D_k$ : $\sigma \mapsto \sigma$
  ou 0 ;
- une porte agit par $R$.

Donc $O_S$ se calcule en partant de $Z_{S \cap \text{final}}$ à la fin, et en
remontant : on multiplie par $Z_k$ aux bits de $S$, on tue les modes
anticommutants à chaque mesure, et on étale par les portes. **[Exact]**

En Haar, $O_S = \sum_a T_{S a} P_a$ est une somme de chemins, avec des
interférences. En Clifford chaque chemin reste une seule chaîne, donc
$O_S = \pm P_g$ ou 0. **[Clifford]** Le test mesure le nombre de chaînes de
$O_S$ (rapport de participation) : exactement 1 en Clifford, jusqu'à plus de 2
en Haar à N = 3.

*Vérifié par* `test_parity_operators_single_string_only_for_clifford`.

### 3.3 Lecture d'ordre k et décomposition en deux filtres

La lecture d'ordre $k$ garde les parités de $|S| \le k$ ; ses lignes sont
$C^{(k)} = \chi\, C$. Comme
$\mathrm{Span}\, C^{(k)} \subset \mathrm{Span}\, C$, les deux projections sont
emboîtées et Pythagore donne, en population, à $\lambda \to 0^+$ et K = ∞ :

```math
E_\text{total}(k) = \underbrace{\frac{\lVert (1-\Pi_C)\beta \rVert_W^2}{\mathrm{Var}\,y}}_{E_\text{circuit}}
+ \underbrace{\frac{\lVert (\Pi_C-\Pi_{C^{(k)}})\beta \rVert_W^2}{\mathrm{Var}\,y}}_{E_\text{readout}(k)} . \qquad \textbf{[Exact]}
```

C'est vrai **pour tout circuit**, Haar compris : seules l'inclusion des espaces
et le même produit scalaire servent. *Vérifié par*
`test_pythagoras_of_nested_readouts_in_haar` (terme croisé $< 10^{-9}$ sur un
circuit Haar, lecture $k = 1$).

Ce qui est **[Clifford]**, c'est la forme diagonale (#349) : chaque parité est
une chaîne, donc les projections sont diagonales dans les modes et

```math
E_\text{circuit} = \frac{\sum_{a \notin G,\ a \text{ visible}} w_a \beta_a^2}{\mathrm{Var}\, y}, \qquad
E_\text{readout}(k) = \frac{\sum_{a \in G \setminus D_k} w_a \beta_a^2}{\mathrm{Var}\, y},
```

où $D_k$ est l'**ensemble des chaînes atteintes** par les parités de $|S|\le k$
(et pas l'espace des features). Pour une cible monôme, les deux termes sont
des indicatrices 0/1. En Haar, $E_\perp$ d'un monôme prend des valeurs
intermédiaires. *Vérifié par*
`test_clifford_split_is_a_sum_of_indicators_haar_is_not`.

**Hors de la population, le budget à trois termes.** Avec λ > 0, le terme
croisé n'est plus nul (il est positif, de l'ordre de $E_\text{readout}$,
#349 ; *vérifié* dans le même test). Avec un ensemble d'entraînement fini ou K
fini, on garde $E_\text{circuit}$ et $E_\text{readout}(k)$ comme quantités de
population, et on définit

```math
E_\text{total} = E_\text{circuit} + E_\text{readout}(k) + E_\text{estimation}, \qquad
E_\text{estimation} = E_\text{total} - E_\text{total}^\text{pop}(k) \ge 0 .
```

$E_\text{estimation} \ge 0$ est **[Exact]** pour l'erreur moyennée sur le
bruit de test : à poids entraînés fixés, la prédiction moyenne est dans
$\mathrm{Span}\, C^{(k)}$, et rien dans cet espace ne bat la projection. Cet
argument n'utilise pas Clifford. Deux précisions (#349) :

- $E_\text{estimation}$ **ne tend pas vers 0 quand K → ∞** : il reste
  l'erreur due au nombre fini d'entrées d'entraînement (0,004 à 0,007 avec 256
  entrées à N = 4) ;
- il dépend des deux filtres à la fois (du nombre de features par $k$, et des
  poids des modes par le circuit), ce n'est donc pas un troisième filtre.

---

## 4. Tirs finis : le filtre des corrélations canoniques

Avec $K$ tirs de la même entrée, l'histogramme $h_K$ a pour moyenne $f(x)$ et
pour covariance totale

```math
\mathrm{Cov}(h_K) = \Big(1 - \tfrac1K\Big)\Sigma + \tfrac1K M, \qquad M = \mathrm{diag}(\bar f) - \bar f \bar f^\top ,
```

où $M$ est la covariance d'un seul tir. Avec $\mu_k \in [0,1]$ les valeurs
propres de $M^{+1/2} \Sigma M^{+1/2}$ (corrélations canoniques au carré entre
un tir et l'entrée), et $z_k^2$ le poids de la cible sur la direction $k$,

```math
R^2(K) = \sum_k z_k^2\, \frac{K \mu_k}{1 + (K-1)\mu_k}, \qquad R^2(\infty) = 1 - E_\perp . \qquad \textbf{[Exact]}
```

(meilleure lecture linéaire de l'histogramme, une infinité d'entrées
d'entraînement, $\lambda \to 0^+$.) Une direction est apprise quand
$\mu_k \gtrsim 1/K$. *Vérifié par* `test_shot_formula` (formule spectrale
contre la covariance directe, K = 1 à 4096, Haar et Clifford).

En Clifford, pour une feature qui porte un seul mode $a$, $\mu = w_a$ et le
facteur devient $K w_a / (1 + (K-1) w_a)$ : un mode est appris quand
$|a| \lesssim \log_2 K$ (#341). **[Clifford]** En Haar les $\mu_k$ ne sont pas
les $w_a$ : une direction canonique mélange des modes.

---

## 5. Ce que fait le circuit : le cas général (Haar)

### 5.1 Dimension du span

- **Bornes.** $\dim \mathrm{Span}\{E_m\} \le \min(\#\text{records},\, 4^N,\, 2^{C})$,
  où $C$ est la coupe minimale du graphe espace-temps (une jambe d'opérateur
  porte 2 bits, une mesure en laisse passer 1, #321). **[Exact]**
- **Haar sature la coupe.** Sur les 1 050 circuits et masques de #321,
  $\dim = 2^C$ exactement. **[Observé]** C'est le comportement attendu d'un
  réseau de tenseurs générique, mais ce n'est pas démontré ici.
- **Le span Haar dépasse $2^N$.** Avec la lecture finale et $p > 0$, on trouve
  par exemple $\dim = 64 > 2^4$ à N = 4 (tableau section 9). Les effets ne
  commutent pas. **[Observé]** ; *vérifié par*
  `test_haar_breaks_the_clifford_structure`.
- **Le span Haar est presque complet dans une fenêtre de p.** À N = 6 et
  p = 0,1, il contient 97 % des $3^N$ fonctions visibles, et $E_\perp$ des
  cibles de degré 1 et 2 tombe à 0,02 (#339). **[Observé]**

### 5.2 Purification : les entropies de Rényi ne coïncident pas

Partant de l'état maximalement mélangé, l'état conditionné au record
intermédiaire $m$ a le spectre de $E_m/\mathrm{Tr}\,E_m$. On note
$\Delta_\alpha$ ses entropies de Rényi, moyennées avec le poids de Born
$\mathrm{Tr}\,E_m / 2^N$.

- $\Delta_0 = \log_2 \mathrm{rang}\, E_m$ est fixé par la géométrie (rang d'un
  produit de projecteurs et de portes génériques) ; c'est une quantité de
  coupe minimale. **[Exact]** pour la borne, **[Observé]** pour l'égalité.
- $\Delta_1$ et $\Delta_2$ sont les paramètres d'ordre de la transition de
  purification Haar ($p_c \approx 0{,}29$ dans la convention du dépôt, #322,
  #350).
- En Haar ils diffèrent : à N = 4, T = 4 et p = 0,4, un circuit (seed 3) donne
  $\Delta_0 = 1$ bit mais $\Delta_1 = 0{,}05$ et $\Delta_2 = 0{,}02$ (le
  spectre est très inégal). **[Observé]**

### 5.3 La conséquence pour le Ridge

Le Ridge à K = ∞, λ → 0⁺ ne voit que le span (section 2.2). En Haar, la
dimension du span suit la coupe minimale ; et la relation Clifford
« $\dim \mathrm{Span}_\text{mid} = 2^{N - \Delta}$ » est fausse
(tableau section 9 : le produit $\dim \mathrm{Span}_\text{mid} \cdot 2^{\Delta_0}$
n'est pas $2^N$). Tout ce qu'on a mesuré en Haar va dans ce sens :

- $E_\perp(p)$ ne montre rien de net à $p_c$ pour N = 4 à 7 (#339) ;
- sans lecture finale, le seuil d'échec du Ridge Haar descend avec N et
  semble gouverné par le nombre de mesures ($2pN^2 > N$), pas par $p_c$
  (#344) ; c'est une hypothèse compatible avec N = 4 à 7, pas un résultat
  établi ;
- la capture locale du span culmine vers p ≈ 0,1, sans rien de net à $p_c$
  (#339).

---

## 6. Ce qui est général, ce qui ne l'est pas : la MIPT dans le cadre

### 6.1 Les deux couches

Il faut séparer **la physique du circuit** et **sa lecture par le Ridge**.

- La physique est universelle : $\Delta$ (purification) et l'entropie
  d'intrication des trajectoires existent pour Haar, Clifford, U(1). Avec la
  lecture finale complète, chaque effet est de rang 1,
  $E_m = |\psi_m\rangle\langle\psi_m|$, où $\psi_m$ est la trajectoire jouée à
  rebours depuis l'état produit mesuré : son entropie suit la transition loi
  de volume / loi d'aire quelles que soient les portes.
- La lecture par le Ridge à K = ∞ est une quantité de span. Elle lit
  exactement ces paramètres d'ordre en Clifford (section 7), et pas en
  général.

### 6.2 Pourquoi Clifford est exceptionnel

En Clifford tous les spectres sont plats : un état stabilisateur a toutes ses
valeurs propres non nulles égales. Donc $\Delta_0 = \Delta_1 = \Delta_2$, et une
quantité de rang (span, dimension, nombre de fonctions locales) contient la
même information qu'une entropie. C'est ce qui permet au Ridge à K = ∞ de
« voir » la MIPT Clifford par $\dim = 2^{N-\Delta}$ et par $2^{l - S(l)}$.
*Vérifié par* `test_clifford_effects_commute_and_span_is_2_to_n_minus_delta`
($\Delta_0 = \Delta_1 = \Delta_2$ à $10^{-9}$ et $\dim = 2^{N-\Delta}$, six
circuits).

### 6.3 Conjecture pour Haar

**[Conjecture]** Dans un circuit Haar, les quantités de *rang* du Ridge
($E_\perp$ à λ → 0⁺ et K = ∞, dimension du span, capture locale exacte) sont
gouvernées par la géométrie (coupe minimale, percolation) et ne montrent pas
de transition à $p_c$. Si le Ridge porte la trace de la MIPT Haar, c'est dans
ses quantités **spectrales** : le spectre $\{s_k\}$ de $\Sigma$, les
corrélations canoniques $\{\mu_k\}$, donc $d_\text{eff}(\lambda)$ et
$n_\text{learn}(K)$, qui pondèrent les directions comme $\Delta_1$, $\Delta_2$
pondèrent les états.

Ce qu'on sait déjà, et qui ne tranche pas : à N = 6 et 8, $n_\text{learn}$ et
le spectre de $F$ varient de façon lisse avec p, avec un pivot du spectre et
un plateau près de $p_c$ (#334). Les tailles sont trop petites pour séparer
une transition d'un crossover.

---

## 7. Réduction à Clifford

Dans cette section seulement, les portes sont de Clifford. Chaque énoncé est
**[Clifford]** ; les contre-exemples Haar sont dans les tests cités.

### 7.1 Une chaîne reste une chaîne

$R$ est une permutation signée (section 1.3), et une mesure envoie une chaîne
sur une chaîne ou sur 0. Donc chaque parité est $O_S = \pm P_g$ ou 0
(section 3.2), et la matrice de transfert d'une lecture est une **matrice de
sélection** : une ligne, un mode (#341).

### 7.2 Le span est un groupe

L'ensemble des chaînes $g$ atteintes par toutes les parités est un groupe
abélien de Pauli $G$, de $2^r$ éléments, qui ne dépend pas du record. Chaque
effet s'écrit

```math
E_m = \mathrm{Tr}(E_m)\, 2^{-r} \sum_{g \in G} s_g(m)\, g^\top, \qquad s_g(m) = \pm 1,
```

et les signes sont des caractères du record (#337). Conséquences :

- les effets **commutent** ; $\mathrm{Span}\{E_m\} = \mathrm{Span}\, G$, de
  dimension $2^r \le 2^N$ ;
- avec la lecture finale, $r = N$ à tout p ; sans elle, $r = N - \Delta$ ;
- $\Sigma$ est diagonale dans les modes visibles de $G$, de valeurs propres
  $w_g$ (fois la multiplicité dans la base des parités) : **le spectre est la
  distribution des poids de $G$** ;
- $E_\perp$ d'un monôme vaut 0 ou 1.

Pour des entrées XZ, seule une petite fraction de $G$ est visible : environ 1 %
à N = 16, 0,3 % à N = 20, moins sous $p_c$ qu'au-dessus (#349). Contre les
$3^N$ fonctions visibles, $G$ en contient 690 sur 43 046 720 à N = 16,
p = 0,04.

### 7.3 La lecture d'ordre k et les tirs

Les deux filtres sont diagonaux (section 3.3). Le filtre de tirs devient
$K w_a / (1 + (K-1) w_a)$ par mode (section 4). La lecture locale
($k = 1, 2$) garde surtout des modes de poids 1 à 4 ; pour une cible locale,
c'est le terme circuit qui domine : pour $Z_i$ à N = 16,
$E_\text{circuit}$ va de 0,83 à 0,99 selon p (#349).

### 7.4 Le théorème d'invisibilité, avec ses conditions

**Énoncé.** Hypothèses :

1. le circuit est de Clifford (le span est un groupe de Pauli $G$) ;
2. l'ensemble d'entrées est invariant par conjugaison par toute chaîne de
   Pauli (états propres de Pauli, angle uniforme sur le cercle XZ, sphère
   uniforme) ;
3. la cible est un mode de Pauli $B_a$ avec $a \notin G$.

Alors il existe $Q$ dans le centralisateur de $G$ qui anticommute avec $a$
(le double centralisateur d'un groupe de Pauli est le groupe). La
conjugaison $x \mapsto QxQ$ préserve l'ensemble d'entrées, laisse **la loi du
record** inchangée, et change le signe de $B_a$. Donc aucune statistique des
tirs, à **n'importe quel K** et pour n'importe quel décodeur, n'est corrélée à
la cible.

*Vérifié par* `test_invisibility_theorem_in_clifford` : pour chaque monôme
hors de $G$, on trouve $Q$, on vérifie que les probabilités de Born sont
identiques pour $x$ et $QxQ$, et que le meilleur décodeur quelconque (moyenne
sur les fibres) a une erreur exactement 1.

**Ce que le théorème ne dit pas.**

- Rien pour Haar : il n'y a pas de groupe, et sur un ensemble fini les
  features Haar séparent les entrées (section 2.4).
- Rien pour une cible non multilinéaire ($x_0^2 - z_0^2$ : erreur linéaire 1,
  meilleur décodeur 0,4 à 0,7, #349).
- $E_\text{readout}(k)$ n'est pas une limite pour un décodeur non linéaire :
  des produits de features de supports disjoints récupèrent 72 % à 100 % des
  modes du groupe engendré par $D_k$ (#349).

### 7.5 Localité du dictionnaire

Le nombre d'éléments de $G$ supportés dans une fenêtre de $l$ sites est
$2^{l - S(l)}$, où $S(l)$ est l'entropie des qubits de référence de cette
fenêtre après la lecture complète (#337, #339). Elle suit la loi de volume
sous $p_c$, $\log l$ à $p_c$, la loi d'aire au-dessus. Attention : ce sont
les **générateurs** (jauge coupée) qui deviennent locaux au-dessus de $p_c$ ;
les **éléments** de $G$ restent non locaux à tout p (à N = 16, la plupart des
chaînes visibles ont une extension de 12 à 15 sites, #349).

### 7.6 Les deux paramètres d'ordre du Ridge Clifford

- **Sans lecture finale :** $\log_2 \dim / N = 1 - \Delta/N$, croisement à
  $p_c = 0{,}161$, collapse avec $\nu = 1{,}30$ (#346, #351).
- **Avec lecture finale :** la localité $1 - S(l)/l$, et l'exposant
  $d\log S(N/2)/d\log N$ qui chute de 0,9 à 0 autour de $p_c$ (#339, #351).

Les deux demandent K = ∞ et toutes les parités du record (un nombre
exponentiel de features). Avec des bits ou des paires de bits, tout redevient
lisse en p et indépendant de N (#341).

---

## 8. Cas spéciaux

| Cas | Ce qui devient simple | Statut |
| --- | --- | --- |
| **p = 0** (portes seules, lecture finale) | $E_m = U^\dagger \lvert m\rangle\langle m\rvert U$ : effets qui commutent, $\dim = 2^N$, pour tout circuit. La fraction visible dépend de U (Haar N = 6 : $E_\perp \approx 0{,}6$, #339) | [Exact], `test_p_zero_is_commutative_for_every_circuit` |
| **p = 1** (tous les sites mesurés à chaque couche) | après la première couche l'état est un état de base déterminé par le premier record ; la suite n'ajoute que du bruit classique indépendant de l'entrée. Span = celui de la première couche : $\dim = 2^N$, effets qui commutent, pour tout circuit | [Exact], `test_p_one_is_commutative_for_every_circuit` |
| **Sans lecture finale** | effets de rang $2^{\Delta_0}$. Clifford : $\dim = 2^{N - \Delta}$. Haar : pas de relation simple | [Clifford] / [Observé] |
| **Lecture locale** (k = 1, 2) | sous-espace $\mathrm{Span}\, C^{(k)}$ ; Pythagore pour tout circuit ; somme d'indicatrices en Clifford | [Exact] / [Clifford] |
| **Entrées XZ contre sphère** | $w = 2^{-\lvert a\rvert}$, chaînes sans Y, contre $w = 3^{-\lvert a\rvert}$, toutes les chaînes | [Exact] |
| **États propres de Pauli contre angles continus** | mêmes moments d'ordre 1 et 2, donc mêmes $\Sigma$, $E_\perp$, $\mu_k$ ; le théorème d'invisibilité vaut pour les deux. Les fibres d'un ensemble fini, elles, dépendent de l'ensemble | [Exact] |

---

## 9. Petits N exacts : Haar contre Clifford, même masque

`experiments/fourier_framework/small_n_table.py --n 4 --seeds 10` : N = 4,
T = 4, brickwork ouvert, pour chaque seed un circuit Haar et un circuit
Clifford avec le **même masque** ; entrées : les $4^4$ produits de
$\lvert 0\rangle, \lvert 1\rangle, \lvert +\rangle, \lvert -\rangle$ ;
circuits de plus de 14 bits écartés. Moyennes sur les circuits gardés.

**Structure du dictionnaire.** $\dim$ : dimension de
$\mathrm{Span}\{E_m\}$ avec la lecture finale ; « visible / $3^N$ » : sa
projection sur les chaînes sans Y ; $\max\lVert[A,B]\rVert$ : plus grand
commutateur sur une base orthonormée du span ; « mid » : record intermédiaire
seul ; $\Delta_\alpha$ : entropies de Rényi de purification (section 5.2).

| circuit | p | circuits | bits | $\log_2\dim$ (dense) | visible / $3^N$ | $\max\lVert[A,B]\rVert$ | $\log_2\dim_\text{mid}$ | $\Delta_0$ | $\Delta_1$ | $\Delta_2$ | $\log_2\dim_\text{mid} + \Delta_0$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Haar | 0.0 | 10 | 4.0 | 4.00 | 0.198 | 0.00 | 0.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| Haar | 0.1 | 10 | 5.3 | 4.80 | 0.395 | 0.62 | 1.30 | 3.00 | 2.90 | 2.86 | 4.30 |
| Haar | 0.2 | 10 | 6.9 | 5.60 | 0.672 | 0.91 | 2.86 | 2.20 | 2.03 | 1.92 | 5.06 |
| Haar | 0.3 | 10 | 8.7 | 5.80 | 0.701 | 1.09 | 4.58 | 1.60 | 1.19 | 1.04 | 6.18 |
| Haar | 0.5 | 9 | 12.1 | 5.67 | 0.658 | 1.32 | 5.44 | 1.00 | 0.49 | 0.39 | 6.44 |
| Clifford | 0.0 | 10 | 4.0 | 4.00 | 0.065 | 0.00 | 0.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| Clifford | 0.1 | 10 | 5.3 | 4.00 | 0.058 | 0.00 | 1.20 | 2.80 | 2.80 | 2.80 | 4.00 |
| Clifford | 0.2 | 10 | 6.9 | 4.00 | 0.063 | 0.00 | 2.10 | 1.90 | 1.90 | 1.90 | 4.00 |
| Clifford | 0.3 | 10 | 8.7 | 4.00 | 0.062 | 0.00 | 3.30 | 0.70 | 0.70 | 0.70 | 4.00 |
| Clifford | 0.5 | 9 | 12.1 | 4.00 | 0.073 | 0.00 | 3.89 | 0.11 | 0.11 | 0.11 | 4.00 |

**Ridge en population.** Cibles : $Z$ du site 2, et la somme des $X_i$ et
$Z_i$ (degré 1). $E_\text{readout}(k)$ et le terme croisé : décomposition de la
section 3.3 pour la cible $Z_2$, lecture des parités de $k$ bits au plus
(terme croisé : maximum sur les circuits). Dernière colonne : nombre de
circuits dont les features exactes séparent les $4^4$ entrées.

| circuit | p | $E_\perp(Z_2)$ | $E_\perp$ deg. 1 | $E_\perp$ deg. 1, mid | $E_\text{readout}(1)$, $Z_2$ | $E_\text{readout}(2)$, $Z_2$ | terme croisé | features injectives |
|---|---|---|---|---|---|---|---|---|
| Haar | 0.0 | 0.566 | 0.554 | 1.000 | 0.250 | 0.090 | 5e-16 | 10/10 |
| Haar | 0.1 | 0.356 | 0.354 | 0.913 | 0.456 | 0.179 | 1e-15 | 10/10 |
| Haar | 0.2 | 0.154 | 0.165 | 0.676 | 0.629 | 0.217 | 1e-15 | 10/10 |
| Haar | 0.3 | 0.109 | 0.140 | 0.382 | 0.623 | 0.167 | 1e-15 | 10/10 |
| Haar | 0.5 | 0.085 | 0.125 | 0.167 | 0.405 | 0.070 | 1e-15 | 9/9 |
| Clifford | 0.0 | 1.000 | 0.963 | 1.000 | 0.000 | 0.000 | 4e-29 | 0/10 |
| Clifford | 0.1 | 1.000 | 0.950 | 0.975 | 0.000 | 0.000 | 4e-29 | 0/10 |
| Clifford | 0.2 | 1.000 | 0.912 | 0.950 | 0.000 | 0.000 | 6e-29 | 0/10 |
| Clifford | 0.3 | 1.000 | 0.912 | 0.912 | 0.000 | 0.000 | 5e-32 | 0/10 |
| Clifford | 0.5 | 1.000 | 0.847 | 0.861 | 0.000 | 0.000 | 1e-46 | 0/9 |

**Ce que montrent ces tableaux.**

- **Clifford : toute la structure de la section 7, exactement.** Effets qui
  commutent, $\dim = 2^N$ avec la lecture finale à tout p,
  $\Delta_0 = \Delta_1 = \Delta_2$, et $\log_2 \dim_\text{mid} + \Delta = N$
  sur chaque circuit. Seuls 6 % à 7 % des fonctions visibles sont dans le span.
  $Z_2$ n'est jamais dans $G$ sur ces 49 circuits, donc $E_\perp = 1$ et
  $E_\text{readout} = 0$.
- **Haar : rien de cette structure.** Les effets ne commutent pas dès p > 0.
  Le span dépasse $2^N$ (jusqu'à $2^{5{,}8}$ en moyenne) et contient jusqu'à
  70 % des fonctions visibles. Les entropies de Rényi se séparent, et de plus
  en plus avec p : à p = 0,5, $\Delta_0 = 1$ bit, alors que $\Delta_1 = 0{,}49$
  et $\Delta_2 = 0{,}39$. La somme $\log_2\dim_\text{mid} + \Delta_0$ dépasse
  $N$ et croît avec p : la relation de Clifford ne tient pas.
- **Pythagore tient en Haar** (terme croisé $\le 10^{-15}$), mais la
  répartition est inverse de celle de Clifford : pour la cible locale $Z_2$,
  le terme de lecture $E_\text{readout}(1)$ (0,25 à 0,63) est plus grand que le
  terme circuit (0,09 à 0,57) dès p ≥ 0,1. En Haar, la lecture locale perd une
  grande partie de ce que le circuit a gardé ; en Clifford, elle ne perd rien
  parce que le circuit n'a rien gardé de local. **[Observé]**, N = 4.
- **Les features Haar séparent toutes les entrées** (49 circuits sur 49), les
  features Clifford jamais (le théorème d'invisibilité apparie $x$ et $QxQ$).
- **N = 4 est trop petit pour parler de $p_c$.** Ce tableau illustre les
  énoncés structurels, il ne dit rien de la transition.

---

## 10. Tableau de statut

| Énoncé | Tout circuit | Clifford | Haar |
| --- | --- | --- | --- |
| $\mathbb{E}[B_aB_b] = w_a\delta_{ab}$, Y invisible pour XZ | Exact | — | — |
| features multilinéaires, $\Pr(m\mid x) = C_m \cdot B(x)$ | Exact | — | — |
| porte = $R$ orthogonale ; mesure tue les modes anticommutants | Exact | $R$ permutation signée | $R$ dense |
| spectre du Ridge = $\mathrm{sv}^2(CW^{1/2})$ | Exact | diagonal, $w_g$ | non diagonal |
| $E_\perp$ = projection pondérée, ne dépend que du span | Exact | 0 ou 1 par monôme | valeurs intermédiaires |
| toutes les parités engendrent $\mathrm{Span}\{E_m\}$ | Exact | — | — |
| $O_S$ = une seule chaîne | non | Clifford | faux (plusieurs chaînes) |
| Pythagore $E_\text{circuit} + E_\text{readout}(k)$ (pop., λ→0⁺, K=∞) | Exact | sommes d'indicatrices | projections non diagonales |
| terme dominant pour une cible locale | — | circuit ($E_\text{circuit} \ge 0{,}83$ pour $Z_i$, Observé N = 16) | lecture ($E_\text{readout}(1)$ 0,25 à 0,63, Observé N = 4) |
| $E_\text{estimation} \ge 0$, ne s'annule pas à K = ∞ (entrées finies) | Exact / Observé | — | — |
| $R^2(K) = \sum z_k^2 K\mu_k/(1+(K-1)\mu_k)$ | Exact | $\mu = w_a$ par mode | $\mu_k$ mélangent les modes |
| effets qui commutent, $\dim \le 2^N$ | non | Clifford | faux |
| $\dim \mathrm{Span}_\text{mid} = 2^{N-\Delta}$, $\Delta_0 = \Delta_1 = \Delta_2$ | non | Clifford | faux |
| $\dim = 2^{\text{coupe min}}$ | borne exacte | bien en dessous | Observé (égalité) |
| théorème d'invisibilité (entrées invariantes par Pauli, tout K) | non | Clifford | ne s'applique pas ; features injectives (Observé) |
| fonctions locales $= 2^{l - S(l)}$ | non | Clifford | inconnu (#352 en cours) |
| $E_\perp$ ne voit pas $p_c$ | — | vrai pour les cibles locales (Observé, N ≤ 128) | Conjecture (cohérent avec N ≤ 7) |
| la MIPT est dans le spectre ($\mu_k$, $s_k$) | — | — | Conjecture |

---

## 11. Corrections intégrées

Cette note remplace des formulations antérieures trop larges :

1. Le complémentaire apporté par la lecture finale ne contient **aucune des
   cibles locales testées** ($Z_i$, $Z_iZ_{i+1}$, $Z_iX_{i+1}$ à N = 128) ; on
   n'a pas montré qu'il est « exclusivement non local ».
2. Le seuil Haar sans lecture finale **semble** gouverné par $2pN^2 > N$ :
   hypothèse compatible avec N = 4 à 7.
3. La branche à bas p de N = 8 ($k \le 5$) n'est pas un artefact : elle vient
   entièrement des bits de la lecture finale.
4. Le théorème d'invisibilité demande un circuit de Clifford et un ensemble
   d'entrées invariant par conjugaison de Pauli. Il ne dit rien pour Haar. Il
   vaut à K fini.
5. Les sommes sur les modes de $E_\text{circuit}$ et $E_\text{readout}$ ne sont
   exactes qu'en Clifford ; elles sont divisées par $\mathrm{Var}\, y$,
   restreintes aux chaînes visibles, et $D_k$ est l'ensemble des chaînes
   atteintes. Pythagore, lui, vaut pour tout circuit.
6. « ≈ 90 % de la variance dans $E_\text{circuit}$ » vient de la cible dense à
   N = 4 ; pour $Z_i$ à N = 16, $E_\text{circuit}$ va de 0,83 à 0,99.
7. $E_\text{estimation}$ ne tend pas vers 0 quand K → ∞ (entrées
   d'entraînement finies) ; $E_\text{estimation} \ge 0$ vaut pour l'erreur
   moyennée sur le bruit de test.

---

## 12. Questions ouvertes pour Haar

1. **Une quantité spectrale du Ridge voit-elle $p_c$ ?** Mesurer
   $d_\text{eff}(\lambda)$, $n_\text{learn}(K)$ ou l'entropie du spectre
   normalisé de $\Sigma$ (un « Rényi du dictionnaire »), à N = 8 à 12, et
   chercher un croisement à $p_c \approx 0{,}29$.
2. **La localité du span Haar suit-elle l'entropie des trajectoires
   $\psi_m$ ?** En cours dans #352.
3. **Démontrer $\dim = 2^{\text{coupe min}}$** pour des portes de Haar
   (rang générique d'un réseau de tenseurs), ou trouver le contre-exemple.
4. **Information non linéaire à K fini.** En Haar, les features exactes
   séparent les entrées ; à quel K un décodeur non linéaire bat-il
   significativement le Ridge, et ce K dépend-il de p de façon critique ?
