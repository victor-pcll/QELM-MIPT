# Guide des algorithmes de la bibliothèque

*Note d'ensemble, issue #366. Elle décrit, étape par étape, les algorithmes
qu'emploie `src/QELM/` : ce que chacun calcule, comment, avec quelles
conventions, à quel coût et avec quelles limites. Elle ne remplace pas les
notes détaillées de [`notes/`](notes/), qui démontrent les résultats ; elle
dit où se trouve chaque morceau et comment les morceaux s'emboîtent. Chaque
section renvoie au module et aux tests qui vérifient ce qui est écrit.*

Le fil conducteur est celui d'une expérience QELM complète :

1. une entrée classique `x` est **encodée** dans l'état initial du réservoir
   (section 2) ;
2. le réservoir évolue sous une **dynamique unitaire**, hamiltonienne ou
   circuit aléatoire (section 3), entrecoupée de **mesures en milieu de
   circuit** (section 4) ;
3. la trajectoire est simulée par vecteur d'état, matrice densité ou tableau
   de stabilisateurs (sections 5 et 6), et l'on en tire des **entropies**
   (section 7) ;
4. le record de mesure et la lecture finale forment les **features** d'une
   **régression Ridge** (section 8) ;
5. le **cadre de Fourier / Bloch** dit exactement ce que ces features peuvent
   représenter, et le **budget de R²** sépare ce que perd le circuit, la
   lecture et l'estimation (section 9) ;
6. des **décodeurs bayésiens** donnent des plafonds de référence (section 10) ;
7. l'**analyse de taille finie** estime `p_c` et `ν` à partir des courbes de
   plusieurs tailles (section 11).

La section 12 récapitule les coûts et les tests.

---

## 1. Conventions communes

Ces conventions valent dans tout le dépôt ; la plupart des erreurs de lecture
d'un résultat viennent de l'une d'elles.

- **Ordre des qubits.** Le qubit 0 est le facteur le plus à gauche du produit
  tensoriel, donc le bit **le plus significatif** de l'indice d'un état de
  base : l'état `|b_0 b_1 … b_{n-1}>` a l'indice `Σ_i b_i 2^(n-1-i)`
  (`QELM.physics.operators.get_site_operator`, `QELM.physics.partial_trace`).
  Une porte posée sur `(i, j)` a `i` comme bit de poids fort de son indice à
  deux qubits.
- **`N`, `Nr`, `N_total`.** `N` est le nombre de qubits du système, `Nr` le
  nombre d'ancillas (qubits de référence), placées **en tête** (qubits
  `0 … Nr-1`), et `N_total = N + Nr` le nombre de qubits de l'état simulé
  (issue #235, `qubit_counts`). Les portes et les mesures n'agissent que sur
  les qubits `Nr … N_total - 1`.
- **Résultats de mesure.** Un qubit mesuré en `Z` donne la valeur propre
  `m = +1` pour `|0>` et `m = -1` pour `|1>`, de sorte que la moyenne de `m`
  est `<Z>`. Dans un `ReadoutResult`, un qubit non mesuré vaut `nan` ; dans le
  record `M` qui sert de features, il vaut `0`.
- **Entropies en bits.** Toutes les entropies sont en bits (logarithme en base
  2) sauf mention contraire (`base=np.e` pour des nats, utilisé par la
  reproduction de Barratt et al.).
- **Pas de temps d'un circuit.** Deux conventions coexistent, et elles ne
  comptent pas le temps de la même façon :
  - `BrickworkCircuit` (`models/circuits/base.py`), `effect_rank`,
    `fourier`, `ridge_spectrum` : une **couche** = une période complète
    (liens pairs puis liens impairs), bords **ouverts** par défaut ;
  - `stabilizer`, `monitored`, `haar_mipt`, `clifford_ridge`,
    `dehghani_decoder` : un **pas** = une **demi-couche** du brickwork
    **périodique** (`periodic_bonds(N, t)`), ce qui est l'unité de temps des
    articles de Gullans & Huse et de Zabalo et al. (« nombre de portes qui
    ont agi sur chaque site »).

  Une profondeur `T` dans la première convention vaut donc `2T` pas dans la
  seconde. Comparer un `p_c` entre les deux demande de convertir.
- **Graine.** `QELM.seed.set_seed` initialise l'état global de NumPy à partir
  de `seed` (défaut 42). Les fonctions qui acceptent `rng=None` tirent alors un
  générateur de cet état global, si bien qu'un run est reproductible à graine
  fixée même sans `rng` explicite. Les simulations de trajectoires passent en
  général un `np.random.Generator` dédié.

---

## 2. Encodage de l'entrée

Module : `src/QELM/encoding/`. Tests : `test_encoding.py`,
`test_encoding_rho.py`, `test_product_encodings.py`,
`test_rotation_encoding.py`.

Un encodage (`Encoding`) prépare un état pur `|ψ(x)>` et sa matrice densité
`ρ(x) = |ψ(x)><ψ(x)|` (`state`, `density_matrix` ou son alias `rho`,
éventuellement en CSR avec `sparse=True`).

### 2.1 Encodages produit d'une entrée continue

`ProductEncoding` découpe l'entrée `x = (x_1, …, x_M)` en groupes de
`features_per_qubit` composantes consécutives ; le groupe `i` fixe l'état
`a_i|0> + b_i|1>` du qubit `i`, et l'état total est le produit tensoriel. Avec
deux composantes par qubit et `M` impair, le dernier groupe est complété par
un 0. Toute composante hors de ses bornes lève `OutOfBoundsError` : un
prétraitement mal normalisé ne passe pas inaperçu.

Ce qui compte pour la suite (section 9) est le **vecteur de Bloch**
`r_i = (<X_i>, <Y_i>, <Z_i>)` de chaque qubit, car
`ρ(x) = Π_i (I + r_i · σ_i) / 2`.

| Classe (clé de config) | Composantes par qubit | État d'un qubit | Vecteur de Bloch `r_i` |
| --- | --- | --- | --- |
| `AngleEncoding` (`angle_encoding`) | 1, `θ ∈ [0, π]` | `cos(θ/2)|0> + sin(θ/2)|1>` | `(sin θ, 0, cos θ)` |
| `DenseAngleEncoding` (`dense_angle_encoding`) | 2, `θ, φ ∈ [0, π]` | `cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>` | `(sin θ cos φ, sin θ sin φ, cos θ)` |
| `UniformBlochEncoding` (`uniform_bloch_encoding`) | 2, `u ∈ [0, 1]`, `φ ∈ [0, π]` | `√u|0> + e^{iφ}√(1-u)|1>` | `(2√(u(1-u)) cos φ, 2√(u(1-u)) sin φ, 2u - 1)` |
| `GeneralEncoding` (`general_encoding`) | 2, `x_1, x_2 ∈ [0, 1]` | `(x_1|0> + x_2|1>) / ‖x‖` | celui de l'angle `θ = 2 atan2(x_2, x_1)` |

Trois remarques qui ont des conséquences :

- **Angle et général restent dans le plan `x-z`** (`<Y> = 0`). Une fonction de
  l'entrée lue par n'importe quel record est alors une combinaison des `3^N`
  monômes de Bloch construits sur `{1, sin θ_i, cos θ_i}` ; les chaînes de
  Pauli contenant `Y` sont invisibles (section 9.1 et
  [`notes/rang-espace-des-effets.md`](notes/rang-espace-des-effets.md)).
- **Le dense couvre la sphère**, et donne accès aux `4^N` chaînes de Pauli.
  Avec `φ ∈ [0, π]` seulement, la composante `<Y> = sin θ sin φ` est
  toujours `≥ 0` : l'encodage couvre une demi-sphère. Ce n'est pas une
  restriction de rang (les monômes restent linéairement indépendants comme
  fonctions), mais c'est une restriction de la distribution des entrées, qui
  change les covariances de la section 9.
- **`UniformBlochEncoding`** rend `<Z> = 2u - 1` linéaire dans la composante :
  des pas égaux de `u` découpent des aires égales de la sphère (Archimède), là
  où l'angle donne `<Z> = cos θ`.

`RotationEncoding` (`rotation.py`) est l'encodage historique d'une entrée
statique : `R_a(x_0)|0> ⊗ … ⊗ R_a(x_{N-1})|0>`, `a = x` ou `y`, avec
`bloch_vectors(x)` en forme close. Avec `a = y` c'est l'`AngleEncoding` ; avec
`a = x`, `R_x(θ)|0> = cos(θ/2)|0> - i sin(θ/2)|1>` place le vecteur de Bloch
dans le plan `y-z`, `r = (0, -sin θ, cos θ)`.

Le prétraitement des images (`QELM.datasets.preprocess.AngleEncoder`) est une
ACP ajustée sur les images d'entraînement seules, chaque composante étant
ensuite envoyée linéairement sur `[0, π]`.

### 2.2 États avec qubit de référence

- `AncillaChargeSectorEncoding` (`charge_sector.py`) :
  `(|0>_R |Q1>_S + |1>_R |Q2>_S) / √2`, avec `Q1 = N // 2`, `Q2 = N // 2 - 1`
  et `|Q>_S` la superposition uniforme des `C(N, Q)` états de base de charge
  `Q = Σ_i b_i`. Mesurer `R` en `Z` fixe le secteur de charge du système :
  c'est l'étiquette des tâches `QELM_M` et `QELM_Q`. Comme `R` n'évolue
  jamais (`U = 1_R ⊗ U_S`), le mesurer au début ou à la fin donne la même
  statistique.
- `AncillaBellPairEncoding` (`bell_pair.py`) :
  `(|0>_R |0…0>_S + |1>_R |0…1_site…0>_S) / √2`, une paire de Bell entre `R`
  et un seul site, les autres sites dans `|0>`. C'est la sonde de
  purification de Gullans & Huse : `S_R = 1` bit au départ.

---

## 3. Dynamique unitaire

### 3.1 Hamiltoniens et évolution en temps continu

Modules : `models/base.py`, `models/TFIM.py`, `models/continuous_tfim_1d.py`.
Tests : `test_TFIM_model.py`, `test_continuous_tfim_1d.py`.

- **TFIM à couplages aléatoires tous-à-tous.** Avec `basis="X"` (défaut),
  `H = h Σ_i Z_i + Σ_{i<j} J_ij X_i X_j` ; avec `basis="Z"`,
  `H = -h Σ_i X_i - Σ_{i<j} J_ij Z_i Z_j`. Chaque `J_ij` est tiré une fois,
  uniforme dans `[-J/2, J/2]`, à la construction. Avec `h = 10 J` (défaut des
  expériences sur la charge), les termes `X_i X_j` changent la charge de 0 ou
  2 : la parité de `Q` est conservée exactement, `Q` lui-même
  approximativement.
- **Chaîne d'Ising en champ incliné** (`ContinuousTFIMChain`) :
  `H = -Σ_i J_i Z_i Z_{i+1} - h_x Σ_i X_i - h_z Σ_i Z_i`, bords ouverts,
  `h_x = 1.05`, `h_z = 0.5`, `J_i` uniforme dans `[0.5, 1.5]` : le modèle
  non intégrable de Kim & Huse (2013), contrepartie hamiltonienne de la chaîne
  de Floquet de la section 3.3.

**Algorithme d'évolution.** `H` est construit en CSR. `evolve(state, dt)`
applique `U = exp(-i H dt)` **sans jamais former `U`** :
`scipy.sparse.linalg.expm_multiply` calcule `exp(-i H dt) v` par une série de
Taylor tronquée qui ne demande que des produits `H @ v`. Pour un vecteur, le
coût est quelques dizaines de produits creux, `O(nnz(H))` chacun. Pour une
matrice densité, l'algorithme est appliqué deux fois :
`U ρ U† = U (U ρ†)†`. `propagator(dt)` forme au contraire la matrice dense
`expm(-i H dt)` (coût `O(8^N)`), utile seulement pour les tests et les petites
tailles.

### 3.2 Circuits brickwork aléatoires

Modules : `models/circuits/base.py`, `haar.py`, `clifford.py`. Tests :
`test_circuits.py`.

**Géométrie.** `BrickworkCircuit.bonds(half)` renvoie les paires disjointes
`(i, i+1)` d'une demi-couche : celles qui commencent au premier site du
système pour `half` pair, au second pour `half` impair, plus le lien
`(N-1, 0)` en bords périodiques avec `N` pair. Une couche est la demi-couche
paire suivie de l'impaire. Les `scrambling_layers` optionnelles apparient les
qubits du système au hasard (permutation aléatoire), tous-à-tous. Toutes les
portes sont tirées une fois par `redraw` ; `fixed_instance` dit au script s'il
faut les garder pour toutes les trajectoires (moyenne sur les records d'un
circuit fixe) ou les retirer pour chaque échantillon (moyenne sur les
circuits).

**Application d'une porte** (`apply_two_qubit_gate`). L'état n'est jamais
multiplié par la matrice `2^N × 2^N` de la porte. L'indice de ligne est
remodelé en `N` axes de taille 2 ; les axes des qubits `i` et `j` sont amenés
en tête, contractés avec la matrice `4 × 4`, puis remis en place. Coût :
`O(2^N)` pour un vecteur d'état, `O(4^N)` pour une matrice densité, qui
reçoit `G ρ G†` en deux passes (`A = G ρ`, puis `(G A†)†`).

**Portes de Haar.** `haar_unitary(n)` suit Mezzadri (2007) : décomposition QR
d'une matrice gaussienne complexe `Z`, puis multiplication de chaque colonne
de `Q` par la phase `d_k / |d_k|` du coefficient diagonal `R_kk`. Sans cette
correction, la QR de LAPACK fixe une convention de phase et la loi de `Q`
n'est **pas** la mesure de Haar.

**Portes U(1)** (`random_u1_gate`) : une phase uniforme sur `|00>`, une sur
`|11>` et un bloc de Haar `2 × 2` sur `{|01>, |10>}`. Elles conservent le
nombre de bits à 1, donc la charge globale (Agrawal et al. 2024).

**Groupe de Clifford à deux qubits** (`clifford_group_2q`). Énumération
exhaustive par parcours en largeur depuis l'identité, avec les générateurs
`H ⊗ I`, `I ⊗ H`, `S ⊗ I`, `I ⊗ S` et `CNOT`. Deux matrices qui ne diffèrent
que d'une phase globale représentent le même élément : la clé de hachage
divise la matrice par la phase de sa première entrée non nulle et arrondit à
`1e-9`. Le parcours s'arrête sur exactement `|C_2| = 11 520` éléments, mis en
cache. `random_clifford_gate` en tire un uniformément ; c'est la mesure
uniforme sur le groupe, qui est un 3-design.

### 3.3 Autres dynamiques

- **Chaîne de Floquet de Skinner, Ruhman & Nahum** (`floquet.py`, test
  `test_floquet_circuit.py`) : chaque brique est la même porte
  `exp{-i[φ_xx X X + θ_x (X + X)]} exp{-i[φ_zz Z Z + θ_z (Z + Z)]}`, avec les
  valeurs de l'article. Ici une couche est une demi-couche
  (`layers_per_period = 2`).
- **Circuits à symétrie diagonale** (`symmetric.py`) : `Z2BrickworkCircuit`
  conserve la parité `Π_i Z_i`, `DiagonalBrickworkCircuit` conserve chaque
  `Z_i`. Ils servent de contre-exemples (issue #288) : une symétrie qui
  commute avec les mesures `Z` change ce que le record peut apprendre.
- **Pas U(1) de Haar global** (`exp_QELM_Q.py`, `unitary_step=haar_u1`) : un
  unitaire de Haar indépendant sur chaque secteur de charge, le modèle
  « complètement brouillé » de la section 10.2 réalisé exactement.

---

## 4. Mesures en milieu de circuit

Modules : `training/measurements/`. Tests : `test_projective.py`,
`test_weak_measurement.py`, `test_exp_QELM_*.py`.

### 4.1 Mesure projective `Z`, matrice densité ou vecteur d'état

`measure_projective_rho(ρ, i)` :

1. **Règle de Born** : `p_0 = Tr(P_0 ρ)`, somme des éléments diagonaux dont le
   bit `i` vaut 0 (masque `((k >> (n-1-i)) & 1) == 0`), coupée dans `[0, 1]`.
2. **Tirage** : un nombre uniforme `u` ; résultat 0 si `u < p_0`.
3. **Projection** : `P_m ρ P_m / p_m`, en ne gardant que le bloc
   `(masque_m, masque_m)` ; les cohérences entre les deux blocs sont mises à
   zéro.

`measure_projective_psi(ψ, i)` fait la même chose sur un vecteur :
`p_0 = Σ_{b_i = 0} |ψ_b|²`, même tirage, puis `P_m ψ / √p_m`. Pour
`ρ = |ψ><ψ|` les deux fonctions tirent le **même** résultat avec le même
générateur et rendent le même état, au coût `O(2^N)` au lieu de `O(4^N)` :
c'est ce qui rend `N_total = 12` abordable dans `QELM_Q`
(`state_representation=vector`, issue #183).

`ProjectiveMeasurement.measure(ρ, p)` mesure chaque qubit cible, dans
l'ordre, avec probabilité `p`. L'ordre des tirages est fixé et fait partie de
la reproductibilité :

- avec `p = 1`, aucun tirage supplémentaire : un seul uniforme par qubit, pour
  le résultat ;
- avec `0 < p < 1`, un premier uniforme décide si le qubit est mesuré
  (`u < p`), puis, seulement s'il l'est, un second tire le résultat.

Les ancillas (`Nr` premiers qubits) ne sont jamais mesurées quand `qubits`
n'est pas donné.

**Pourquoi l'état reste pur.** Une évolution unitaire suivie de projections
envoie un état pur sur un état pur ; c'est pourquoi les trajectoires peuvent
être suivies par vecteur d'état. La matrice densité n'est nécessaire que pour
des canaux non purs, ou pour vérifier l'équivalence dans les tests.

### 4.2 Mesure faible de la charge

`weak.py` (Agrawal et al. 2024, issue #137). Le qubit contrôle une rotation
`exp(-i γ π X / 2)` d'un détecteur frais, lu ensuite projectivement. Sur le
qubit, cela donne deux opérateurs de Kraus diagonaux :

```
K_0 = |0><0| + cos(γπ/2) |1><1|     (pas de clic)
K_1 =          sin(γπ/2) |1><1|     (clic)
```

avec `K_0†K_0 + K_1†K_1 = 1`. `γ = 0` n'apprend rien ; `γ = 1` est la mesure
projective de la charge. Comme les deux opérateurs sont diagonaux, la mise à
jour est une simple remise à l'échelle des amplitudes dont le bit vaut 1,
sans construire de matrice ; l'état reste pur.

### 4.3 Mesures en temps continu (processus de Poisson)

`poisson.py` (issue #241). Chaque qubit est mesuré aux instants d'un processus
de Poisson de taux `γ`. Le temps est coupé en fenêtres `τ`, chacune en
`n_sub` micro-pas `dt = τ / n_sub` : évolution `U(dt)`, puis mesure de chaque
qubit avec la probabilité `1 - exp(-γ dt)` d'au moins un événement. Le taux est
calibré sur le `p` des protocoles discrets, `γ = -ln(1 - p) / τ`, de sorte
qu'un qubit est mesuré au moins une fois par fenêtre avec probabilité
exactement `p` ; par micro-pas cela fait `1 - (1 - p)^(1/n_sub)`. Les
trajectoires sont évoluées par lots (colonnes d'une matrice `(dim, B)`), un
seul produit matriciel par micro-pas.

---

## 5. Simulation de Clifford par tableau de stabilisateurs

Module : `models/circuits/stabilizer.py`, et `evaluation/clifford_ridge.py`,
`effect_mipt.py`, `dictionary_spectrum.py` pour les usages QELM. Tests :
`test_stabilizer_probe.py`, `test_clifford_ridge.py`,
`test_clifford_collapse_map.py`.

Un circuit de portes de Clifford et de mesures de Pauli envoie un état
stabilisateur sur un état stabilisateur. L'état de `n` qubits est décrit par
`n` générateurs de son groupe stabilisateur, soit un tableau de `2n` bits par
générateur, et chaque porte ou mesure coûte `O(n)` à `O(n²)` opérations
binaires (Gottesman-Knill, Aaronson-Gottesman). La bibliothèque délègue cette
simulation à `stim.TableauSimulator`, ce qui atteint `N = 256` là où le
vecteur d'état s'arrête vers `N = 20`.

**Pont entre la bibliothèque et `stim`.** `clifford_gate_programs` convertit
une fois chacun des 11 520 éléments de `clifford_group_2q` en tableau
`stim` (`Tableau.from_unitary_matrix(u, endian="big")`, la convention de poids
fort de la section 1), puis en une suite de portes `H`, `S`, `CX` par
élimination. Une porte du circuit est donc désignée par son **indice** dans
le groupe.

**Reproductibilité.** `draw_probe_history` tire avec NumPy les indices des
portes et le masque des sites mesurés (`ProbeHistory`), et
`reference_entropy_history` les rejoue dans `stim`. Le tirage ne dépend donc
pas de `stim`, et les tests rejouent la même histoire sur le vecteur d'état
avec les portes de la bibliothèque : ils retrouvent la même entropie à chaque
pas.

**Entropie du qubit de référence.** Pour un état stabilisateur, l'état réduit
d'un qubit est pur ou `I/2`. `R` est mélangé si et seulement si aucun
stabilisateur n'agit sur lui seul, c'est-à-dire si `peek_bloch(R)` est
l'identité. `S_Q` vaut donc 0 ou 1 bit dans chaque trajectoire et **ne dépend
pas des résultats de mesure**, seulement des portes et du masque : la graine
des résultats de `stim` n'a aucune influence.

**Entropie d'une région.** Pour `n` générateurs indépendants d'un état pur,
écrits en bits `[x | z]` (matrice `n × 2n`, `stabilizer_matrix`),

```
S_A = rang_GF(2)(G_A) - |A|,
```

où `G_A` ne garde que les colonnes `x` et `z` des qubits de `A` (Fattal et al.
2004). Toutes les entropies de Rényi prennent cette valeur. `gf2_rank` est une
élimination de Gauss modulo 2 : pour chaque colonne, chercher un pivot sous la
ligne courante, l'échanger, faire le XOR de la ligne pivot sur les lignes du
dessous. `steady_state_entropies` en déduit `S_half` et l'information mutuelle
tripartite `I_3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC` sur quatre
quarts contigus de la chaîne périodique : `I_3 < 0` en loi de volume, `0` en
loi d'aire.

**Le groupe de l'effet (QELM Clifford).** Dans `clifford_ridge`, chaque site
`i` part en paire de Bell avec sa propre référence `N + i`. Après le circuit,
l'état réduit des références est `E_m^T / Tr E_m`, avec `E_m` l'effet du
record (section 9.1). Pour un état stabilisateur, cet effet s'écrit
`E_m ∝ Σ_{g ∈ G} s_g(m) g^T` avec `G` un groupe de `2^r` chaînes de Pauli qui
**ne dépend pas des résultats**, `r = N` avec la lecture finale et
`r = N - Δ` sans elle. Tout ce que le QELM peut lire à `K = ∞` se réduit donc
à des questions d'appartenance à `G`, résolues par des rangs sur GF(2).
`dictionary_spectrum` énumère `G` par doublement sur des masques de bits (un
`uint64` pour `x`, un pour `z`), ce qui atteint `N = 20`. Les démonstrations
sont dans [`notes/cadre-fourier-general.md`](notes/cadre-fourier-general.md),
section 7.

---

## 6. Simulation de Haar par vecteur d'état

Modules : `models/circuits/monitored.py`, `evaluation/haar_mipt.py`,
`evaluation/effect_mipt.py`. Tests : `test_monitored_haar.py`,
`test_haar_mipt.py`, `test_effect_mipt.py`.

Sans structure de stabilisateur, l'état est un vecteur de `2^n` amplitudes,
et la taille utile s'arrête vers `N ≈ 20`.

**Trajectoire mesurée** (`monitored._evolve`). Pour chaque pas : une porte de
Haar `U(4)` **neuve** sur chaque lien de la demi-couche périodique, puis une
mesure `Z` de chaque site avec probabilité `p`, le résultat tiré par la règle
de Born (`measure_z` : remodelage en `(2^q, 2, 2^(n-q-1))`, poids `p_1` du
bloc central, projection et renormalisation). `haar_reference_entropy`
applique ce protocole à la sonde de Gullans & Huse ; contrairement au cas de
Clifford, `S_vN` et `S_2` diffèrent et dépendent des résultats.

**Opérateur de Kraus exact d'un record** (`haar_mipt.haar_record`, issue
#348). Plutôt que de suivre un état, on suit l'opérateur
`K_m = P_{m_T} U_T … P_{m_1} U_1` entier : ses `2^N` colonnes sont les images
des `2^N` états de base, évoluées ensemble (coût `O(4^N)` par porte). Chaque
résultat est tiré avec la loi de Born de l'entrée **maximalement mélangée**,
`Pr(m) = Tr[K_m† K_m] / 2^N`. Avec `E = K†K` l'effet du record,

```
1 + W(m) = 2^N Tr[E²] / Tr[E]²,     Δ(m) = N - log2(1 + W(m)),
```

où `W` est le poids de Fourier dense hors du mode constant (#314). Comme
`Tr[E²] / Tr[E]²` est la pureté de `K K† / Tr[K K†]`, l'état atteint depuis
`I / 2^N`, `Δ` est exactement son entropie de Rényi-2 en bits : le paramètre
d'ordre de purification de Gullans & Huse (PRX 10, 041020), sans
approximation.

---

## 7. Entropies d'intrication

Modules : `physics/partial_trace.py`, `evaluation/entropy.py`. Tests :
`test_entropy.py`, `test_von_neumann_entropy.py`.

**Spectre de `ρ_A`** (`entanglement_spectrum`). Deux chemins selon l'entrée :

- **vecteur d'état** : les amplitudes sont remodelées en matrice
  `2^|A| × 2^|B|` (qubits de `A` en lignes, dans l'ordre donné), et les valeurs
  propres de `ρ_A` sont les carrés de ses valeurs singulières (décomposition
  de Schmidt). Ni `ρ` ni `ρ_A` ne sont formés : coût de la SVD,
  `O(2^N · 2^min(|A|, |B|))` ;
- **matrice densité** (dense ou creuse) : `partial_trace(ρ, A)`, puis
  `eigvalsh`.

Les valeurs propres `≤ eps = 1e-12` sont écartées, pour qu'un bruit d'arrondi
comme `-1e-16` n'atteigne jamais un logarithme.

**Entropies.** `S_n = log(Σ λ_i^n) / (1 - n)` pour tout `n ≥ 0` : `n = 0`
donne `log rang ρ_A`, `n = 1` la limite de von Neumann `-Σ λ log λ`, `n = 2`
`-log Tr ρ_A²`, `n = ∞` `-log λ_max`. `tripartite_mutual_information` combine
sept entropies comme en section 5.

**Ce que mesurent les expériences.** L'entropie du qubit de référence
(sonde de purification : 1 bit tant que le système cache l'information, 0 une
fois que les mesures l'ont révélée), l'entropie de demi-chaîne `S_half`
(paramètre d'ordre loi de volume / loi d'aire), et `I_3`. Pour un état global
pur, `S_n(ρ_R) = S_n(ρ_S)` à tout ordre : dans `QELM_M`, `S_vN_sys` répète
`S_vN`.

**Statistiques sur les trajectoires** (`charge_sharpening.py`). Pour des
entropies `S^(m)` tirées avec leur probabilité de Born : moyenne, moments
centrés `μ_k` et rapport de Binder `B = 1 - μ_4 / (3 μ_2²)`, mis à `nan`
quand `μ_2 ≤ 1e-14` (le rapport n'a pas de sens pour des valeurs toutes
égales).

---

## 8. Jeu de données QELM et régression Ridge

Modules : `training/dataset.py`, `training/regression/ridge.py`,
`evaluation/metrics.py`, `evaluation/double_descent.py`. Tests :
`test_dataset.py`, `test_ridge.py`, `test_metrics.py`,
`test_double_descent.py`.

### 8.1 Features

Un échantillon est une trajectoire. Ses features sont, selon l'expérience :

- le **record** `M`, de forme `(T, N)` : `±1` pour un site mesuré au tour
  `t`, `0` sinon, aplati en vecteur ;
- la **lecture finale** `X` : résultats `±1` d'une mesure `Z` finale, ou
  valeurs moyennes exactes `<Z_k> = Tr(Z_k ρ)` (et corrélateurs
  `<O_i O_j>`) quand on lit l'état sans le perturber (`training/readout/`) ;
- leur concaténation `Φ = [M ‖ X]` (`create_qelm_dataset`).

`split_dataset` sépare des ensembles d'entraînement et de test disjoints.
`features/measurement_features.py` ajoute des fonctions non linéaires
explicites du record (produits d'entrées, utilisant `m² = |m|` pour
`m ∈ {-1, 0, 1}`), pour donner au Ridge ce qu'une lecture linéaire manque.

### 8.2 Le Ridge en forme close

`RidgeReadout.fit(Φ, y)` résout

```
(Φ^T Φ + α I) W = Φ^T y
```

par `numpy.linalg.solve` (jamais d'inversion explicite), pour une ou
plusieurs sorties. Points à connaître :

- **Pas d'intercept.** Le Ridge n'en ajuste pas : les scripts ajoutent une
  colonne de 1 à `Φ`. Cette colonne est pénalisée comme les autres, ce qui
  explique le petit `α` par défaut (`1e-3`, issue #92 : `α = 1` coûtait du
  R²).
- **`α = 0`.** La matrice `Φ^T Φ` est singulière dès que
  `n_samples < n_features`. Le cas `α = 0` passe donc par
  `numpy.linalg.lstsq`, soit `W = Φ⁺ y`, la limite `α → 0⁺` de la solution :
  moindres carrés ordinaires au-dessus de `n_samples = rang Φ`, interpolation
  de norme minimale en dessous. C'est le régime de la double descente
  (`double_descent_sweep`), dont l'erreur de test culmine à
  `n_train = rang Φ`.
- **Ordre de résolution de `α`** : argument de `fit`, puis bloc `training` de
  la config, puis `1e-3`.
- **Échelle de `α`.** `RidgeReadout` pénalise la somme des carrés (non
  normalisée par le nombre d'échantillons `S`). Deux modules d'analyse
  emploient une autre échelle : `two_filters.ridge_fit` écrit le Ridge avec la
  Gram moyenne, `(Φ^T Φ / S + λ I)`, qui coïncide avec `RidgeReadout` pour
  `λ = α / S` ; `ridge_spectrum.ridge_sweep` prend un `λ` **relatif**, dont la
  valeur absolue est `λ · S · tr(Σ̂) / d` (`Σ̂` la covariance empirique, `d`
  le nombre de features), et n'y pénalise pas l'intercept. Il faut convertir
  avant de comparer des valeurs de régularisation entre modules.

### 8.3 Scores

- `R2 = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²`, comme `sklearn.metrics.r2_score`. Une
  cible est considérée constante si `ss_tot ≤ 1e-12 · max(1, Σ y²)`, pas
  seulement si `ss_tot = 0` : une cible comme `<Z_0> = ±1` calculée par une
  trace porte un bruit de `1e-16`, qui donnait des R² de `-5e32` (issue #103).
- `NMSE` et son erreur standard ; `accuracy` du signe (ou du côté du seuil
  `(Q1 + Q2)/2`) ; `accuracy_baseline`, la classe majoritaire de
  l'entraînement, qui est le vrai niveau de hasard d'un ensemble de test fini ;
  `balanced_accuracy`, qui vaut 0.5 pour une lecture qui ignore les features
  (issue #130).

---

## 9. Cadre de Fourier / Bloch et budget de R²

Modules : `evaluation/fourier.py`, `fourier_tasks.py`, `effect_rank.py`,
`ridge_spectrum.py`, `two_filters.py`, `fourier_framework.py`. Tests :
`test_fourier.py`, `test_fourier_tasks.py`, `test_effect_rank.py`,
`test_ridge_spectrum.py`, `test_two_filters.py`, `test_fourier_framework.py`.
Notes détaillées : [`notes/cadre-fourier-general.md`](notes/cadre-fourier-general.md)
et [`notes/rang-espace-des-effets.md`](notes/rang-espace-des-effets.md).

### 9.1 L'idée : tout est linéaire en `ρ(x)`

Un record `m` a la probabilité `Pr(m | x) = Tr[E_m ρ(x)]`, avec
`K_m = P_{m_T} U_T … P_{m_1} U_1` (produit des portes et des projecteurs le
long du record) et `E_m = K_m† K_m` son **effet**. Pour un encodage produit,
`ρ(x) = Π_i (I + r_i · σ_i)/2`, donc `Pr(m | x)` est une combinaison linéaire
des **monômes de Bloch** `B_α(x) = Π_{i : α_i ≠ 0} r_{i, α_i}`,
`α ∈ {0, x, y, z}^N`, avec pour coefficients les composantes de Pauli de
l'effet, `Tr[E_m σ_α] / 2^N`. Une moyenne `E[g(record) | x]`, quelle que soit
la fonction `g` (linéaire ou non), est donc un polynôme trigonométrique de
degré au plus 1 en chaque angle. Le circuit, `p` et les symétries ne fixent
que les coefficients.

Le **degré** d'un monôme est `|α|`, le nombre de sites où `α_i ≠ 0` : le poids
de Pauli de `σ_α`. En encodage angle, `α_i = y` est impossible : il reste
`3^N` monômes (ordre en base 3, site 0 chiffre de poids fort, `0, 1, 2` pour
`I, X, Z`, `bloch_monomials`).

### 9.2 Matrice de transfert des features moyennes

`fourier.transfer_matrix(layers, N, p)` donne la matrice `C(p)` dont la ligne
`k` contient les coefficients de la moyenne de la feature `k` sur les `3^N`
monômes : `E[feature_k | θ] = Σ_α C_kα B_α(θ)`.

**Pourquoi un déphasage.** On moyenne sur les masques et les résultats de
toutes les mesures **antérieures** à la feature lue. Un site mesuré avec
probabilité `p`, résultat moyenné, subit le canal
`ρ ↦ (1-p) ρ + p (P_0 ρ P_0 + P_1 ρ P_1)` : un déphasage `Z` de force `p`, qui
multiplie par `(1-p)` chaque élément de matrice dont les bits de ligne et de
colonne diffèrent sur ce site. Ce canal est auto-adjoint, donc il agit de la
même façon en point de vue de Heisenberg (`dephase`).

**Algorithme**, pour la feature `M_{t,i}` :

1. partir de `Z_i` ;
2. le remonter à travers la couche `t` : `O ← U_t† O U_t`
   (`heisenberg_layer`, portes appliquées en ordre inverse) ;
3. pour `s = t-1, …, 0` : déphaser `O` avec la force `p`, puis le remonter à
   travers la couche `s` ;
4. lire les coefficients `Tr[O σ_α] / 2^N` (`pauli_xz_coefficients`, qui
   contracte site par site avec `{I, X, Z}` sans former les `3^N` matrices de
   Pauli) ;
5. multiplier par `p`, car `M_{t,i}` vaut 0 quand le site n'est pas mesuré.

La lecture finale `Z_i` donne la même ligne que `M_{T-1,i}` sans le facteur
`p` : la mesure du dernier tour commute avec `Z_i` et ne change pas sa
moyenne. Coût : `O(T² N)` passages de Heisenberg sur des matrices
`2^N × 2^N`, réservé à `N ≲ 8`.

`accessible_rank(C)` est la dimension de l'espace des fonctions représentables
par une lecture linéaire des features moyennes ; `degree_profile` répartit le
poids de `C` par degré ; `odd_x_mask` repère les monômes inaccessibles quand
une symétrie diagonale en `Z` retourne tous les `sin θ_i` à la fois.

### 9.3 Rang exact de l'espace des effets

`effect_rank.effect_rank(transfers, masks)` calcule le rang de la matrice des
`Pr(m | x)` (une ligne par record, une colonne par monôme), c'est-à-dire ce
qu'**aucune** lecture du record, même non linéaire et avec une infinité de
tirs, ne peut dépasser, **sans lister les `2^(N+K)` records**. Il construit à
rebours une base orthonormée de `Span{E_m}` dans la base de Pauli réelle :

1. après la lecture finale, le span est celui des opérateurs diagonaux
   `{I, Z}^N` (dimension `2^N`) ;
2. une mesure `Z` du site `i` envoie le span `S` sur `P_0 ⊗ S_0 + P_1 ⊗ S_1`,
   avec `S_0 = span{A + B}` et `S_1 = span{A - B}` quand
   `E = I ⊗ A + Z ⊗ B + X ⊗ C + Y ⊗ D` (`measure_site`) ;
3. une porte agit par la matrice orthogonale réelle `16 × 16`
   `R[a,b] = Tr[σ_a U† σ_b U] / 4` (`heisenberg_transfer`), qui garde une base
   orthonormée orthonormée.

Chaque réorthonormalisation (`_orthonormal_rows`) diagonalise la matrice de
Gram, coupe les valeurs propres sous `tol² · λ_max` (`tol = 1e-6`), puis
fait une passe de QR de Cholesky pour corriger la perte d'orthogonalité de la
voie de Gram. Le rang dense est la dimension du span ; le rang angle est
celui de sa projection sur les chaînes sans `Y`. Preuves et exemples à la
main : [`notes/rang-espace-des-effets.md`](notes/rang-espace-des-effets.md).

### 9.4 Le R² idéal et le budget de R²

**R² idéal** (`fourier_tasks.ideal_r2`). Pour une cible
`y(θ) = Σ_α ŷ_α B_α(θ)` et des angles i.i.d. uniformes sur `[low, high]`, la
meilleure lecture linéaire des features **moyennes** (infinité d'échantillons,
aucun bruit de tir) atteint

```
R²_idéal = c^T Σ⁺ c / Var(y),
```

avec `Σ = C G C^T - (C μ)(C μ)^T` la covariance des features moyennes,
`c = Cov(features, y)`, `μ = E[B]` et `G = E[B B^T]`. Les moments des monômes
se factorisent site par site (`angle_moments`), donc `G` n'est jamais formée :
`_gram_apply` la contracte site par site.

**Bruit de tir** (`predicted_r2`). Un seul record est une réalisation bruitée
de sa moyenne. Avec `noise = E_θ[Cov(record | θ)]`, estimé sans biais par
`shot_noise_covariance` comme la moyenne de `r r^T - φ φ^T` (`φ = C B(θ)` la
moyenne exacte), la lecture de `K` tirs moyennés atteint

```
R²(K) = c^T (Σ + noise / K)⁺ c / Var(y),
```

et `K = ∞` redonne le R² idéal. Pour une cible gaussienne, la précision du
signe s'en déduit par la formule de Sheppard,
`1/2 + arcsin(√R²) / π` (`gaussian_accuracy`).

**Spectre des corrélations canoniques** (`ridge_spectrum`, issue #334). Avec
des angles uniformes sur le **cercle entier**, les monômes sont orthogonaux,
`E[B_a B_b] = w_a δ_ab` avec `w_a = 2^(-|a|)`, et l'entrée moyenne est l'état
maximalement mélangé, si bien que `Pr(m) = c_0(m)`. Les moyennes
conditionnelles de la base orthonormée `f_a = B_a / √w_a` sachant le record
sont `v_a(m) = √w_a c_a(m) / c_0(m)`, et

```
F = E_m[v(m) v(m)^T]
```

a ses valeurs propres `μ_k ∈ [0, 1]` : les carrés des corrélations canoniques
entre un tir et l'entrée. Avec `K` tirs de la même entrée, la direction `k`
atteint `R²_k(K) = K μ_k / (1 + (K-1) μ_k)` ; elle est apprenable
(`R²_k > 1/2`) si et seulement si `μ_k > 1/(K+1)` (`n_learn`). La part d'une
cible hors du span, `E_perp` (`approximation_bias`), est un biais strict
qu'aucun nombre de tirs ni d'échantillons ne supprime.

**Budget d'erreur en trois termes** (`two_filters`, issue #349). Sur la
population uniforme des `4^N` produits d'états propres de Pauli, avec
`F ⊂ S` (lecture par parités d'au plus `k` bits, contenue dans le span du
circuit `S = Span G`) et les projecteurs orthogonaux du même produit scalaire,

```
y - P_F y = (y - P_S y) + (P_S y - P_F y),
```

et les deux termes sont orthogonaux, d'où par Pythagore

```
E_total = E_circuit + E_lecture(k)
```

(normalisés par `E[y²]`, `decompose`). Le module calcule chaque terme et le
**terme croisé** séparément : l'identité est vérifiée numériquement, et sa
violation mesurée quand `λ > 0`, quand l'ensemble d'entraînement est fini ou
quand `K` est fini. Le budget tracé par
`experiments/clifford_ridge/plot_two_filters.py` ajoute un troisième terme,
`E_estimation = E_total(entraîné) - (E_circuit + E_lecture)`, l'excès du Ridge
réellement entraîné ; il est positif ou nul car aucun prédicteur de `Span F`
ne bat `P_F`. Pour une cible centrée, le R² atteignable se lit comme
`1 - E`, terme par terme. `any_decoder_error` donne en regard l'erreur du
meilleur décodeur de forme quelconque des features exactes,
`E[Var(y | φ)] / Var(y)`, par moyenne conditionnelle sur chaque fibre
`{x : φ(x) = v}`.

---

## 10. Décodeurs bayésiens et plafonds de référence

Un score de Ridge ne se lit qu'en regard d'un plafond. Trois familles, qui ne
conditionnent pas sur les mêmes variables : il ne faut pas les comparer entre
elles sans le dire (issue #195).

### 10.1 Apprenabilité de la charge (Barratt et al. 2022)

`evaluation/learnability.py`, tests `test_learnability*.py`. Une chaîne de `L`
qubits part de la superposition uniforme des chaînes de charge `Q ∈ {Q0, Q1}`
et évolue sous un brickwork U(1), chaque site mesuré avec probabilité `p`
après chaque couche. Un observateur qui voit le record mais **pas les portes**
utilise la loi de Born **moyennée sur Haar**, qui est un processus d'exclusion
symétrique (SEP) classique : chaque porte devient la matrice stochastique qui
garde la configuration avec probabilité 1/2 et échange les deux bits avec
probabilité 1/2 ; chaque mesure annule les configurations incompatibles avec
son résultat. `sep_log_likelihood` évolue ce vecteur de probabilités sur les
`2^L` configurations (`v ↦ (v + v ∘ swap_ij) / 2`), renormalise après chaque
tour et somme les logarithmes, pour que les longs records ne sous-débordent
pas. Le postérieur `P(Q | m)` donne `P_corr = P(Q* | m)`, d'où la précision
`P(P_corr > 1/2)` (Fig. 2b) et l'entropie moyenne `E[H(P_corr)]` (Fig. 2c).
La moyenne de Haar est exacte : les records tirés des circuits quantiques
suivent exactement la loi du SEP (vérifié dans les tests). Coût dense
`O(2^L)`, jusqu'à `L ≈ 14`.

### 10.2 Bayes-Ridge : la charge à partir du record seul

`evaluation/bayes_ridge.py`, test `test_bayes_ridge.py`. Pour le TFIM, qui n'a
pas de SEP exact, le modèle génératif prend la limite **complètement
brouillée** : juste avant chaque tour, la configuration est uniforme sur la
charge `Q`. Un tour qui mesure `k` sites et en trouve `n` à `|1>` a alors la
probabilité hypergéométrique `C(N-k, Q-n) / C(N, Q)`. Comme le TFIM ne
conserve `Q` qu'approximativement, chaque tour est mélangé avec une fuite de
poids `ε` (résultats uniformes, `2^(-k)`) :

```
P(m_r | Q) = (1 - ε) C(N-k, Q-n) / C(N, Q) + ε 2^(-k),
log P(M | Q) = Σ_r log P(m_r | Q).
```

Sans `ε`, un seul tour incompatible donnerait `-∞`, une feature infinie
qu'aucune lecture linéaire ne prend. Le modèle oublie la mémoire de la
dynamique entre deux tours et surcompte donc l'évidence des tours corrélés ;
un Ridge sur les log-vraisemblances standardisées (plus un biais) corrige
cette surconfiance en les remettant à l'échelle et en déplaçant le seuil.

### 10.3 Purification et plafonds sur des features continues

- **Purification** (`purification.py`) : sachant l'état complet conditionné
  au record, le meilleur pari sur la mesure finale de `R` est `sign(z)`,
  juste avec probabilité `(1 + |z|) / 2`, `z = <Z_R>` : c'est le plafond de
  tout décodeur du record (`bayes_accuracy`). `expected_accuracy` note un
  décodeur contre `z` sans le bruit des tirs uniques.
- **Plafonds empiriques sur `<Z_k>`** (`bayes_optimal.py`, `gmm_scan.py`,
  `knn.py`) : QDA (une gaussienne par classe ; la règle de Bayes si les
  features sont gaussiennes), mélange de `K` gaussiennes par classe (`K`
  choisi par BIC dans `BICGaussianMixtureBayes`), K-NN à
  `K = round(√n_train)` voisins pondérés par l'inverse de la distance,
  euclidienne ou de Mahalanobis (métrique : pseudo-inverse de la covariance
  d'entraînement). Ce sont des **estimations** du plafond de Bayes sur un
  ensemble fini, pas des bornes : un ajustement trop riche surapprend. Quand
  le Ridge tombe à 0.5, un plafond resté haut signale un mur
  d'apprenabilité linéaire, un plafond à 0.5 une perte réelle de
  l'information (décohérence).

---

## 11. Analyse de taille finie

Deux modules, pour deux usages. Tests : `test_finite_size.py`,
`test_finite_size_scaling.py`.

Près d'une transition continue en `p_c`, d'exposant de longueur de
corrélation `ν`, un paramètre d'ordre obéit à la forme d'échelle

```
O(p, N) = F((p - p_c) N^(1/ν)),
```

donc les courbes de tailles différentes se croisent en `p_c` (aux
corrections près) et se superposent après remise à l'échelle.

### 11.1 Croisements

`finite_size_scaling.crossing` (et `finite_size.curve_crossings`, qui rend
tous les croisements) cherche un changement de signe de la différence de deux
courbes sur leurs `p` communs et le place par interpolation linéaire ; un zéro
exact sur un point de grille compte comme croisement. `crossing_errors`
donne l'erreur par **bootstrap paramétrique** : chaque point est retiré selon
une gaussienne de son erreur standard, les croisements sont relocalisés, et
l'erreur est la demi-largeur de l'intervalle 16–84 % des rééchantillons qui
croisent encore. C'est un écart-type robuste : un rééchantillon peut croiser
une seconde fois loin du premier croisement.

### 11.2 Collapse des courbes

**Pour l'entropie de demi-chaîne** (`finite_size.py`, issues #239 et #251),
méthode de Skinner, Ruhman & Nahum (PRX 9, 031009, sec. IV) :

1. `rescale` : `x = (p - p_c) N^(1/ν)` et `y = S(p, N) - S(p_c, N)`, avec
   `S(p_c, N)` interpolé en `p`. Soustraire la valeur critique retire la
   croissance en `log N` au point critique ;
2. `collapse_cost` : pour chaque paire ordonnée de tailles, les points de
   l'une qui tombent dans l'étendue de l'autre sont comparés à l'autre
   interpolée linéairement ; le coût est l'écart quadratique moyen divisé par
   la variance de tous les `y` (indépendant des unités). Une fenêtre
   `|p - p_c| ≤ window` restreint la comparaison à la zone où la forme
   d'échelle vaut ;
3. `fit_collapse` : recherche sur une grille `41 × 41` de `(p_c, ν)` (`ν` en
   échelle logarithmique), puis Nelder-Mead depuis le meilleur point, borné à
   la même boîte ; le résultat de la grille est gardé si Nelder-Mead fait
   pire ;
4. `bootstrap_collapse` : erreur **statistique** par bootstrap paramétrique,
   chaque ajustement repartant du meilleur `(p_c, ν)`.

**Pour un paramètre d'ordre avec barres d'erreur** (`finite_size_scaling.py`,
issue #185) : fonction de qualité de Houdayer & Hartmann (PRB 70, 014418).
Chaque point `(x_i, y_i, dy_i)` est comparé à chaque autre taille dont la
courbe remise à l'échelle encadre `x_i`, interpolée linéairement en
`(Y, dY)` ; la qualité est la moyenne de
`(y_i - Y)² / (dy_i² + dY²)`. Elle vaut environ 1 pour un bon collapse, et
l'option `subtract=True` collapse `O(p, N) - O(p_c, N)` comme ci-dessus.
`fit_collapse` lance Nelder-Mead depuis `(p_c0, ν0)` et quatre départs
voisins (`p_c0 ± 0.02`, `ν0 × 0.7` ou `× 1.4`), garde le meilleur minimum, et
estime les erreurs par bootstrap paramétrique. `quality_map` trace la qualité
sur une grille, ce qui montre la forme de la vallée.

**Limites.** Les erreurs de bootstrap sont statistiques. À petit `N`,
l'erreur **systématique** (choix des tailles, de la fenêtre en `p`,
corrections d'échelle) est en général plus grande : le coût a une longue
vallée dans le plan `(p_c, ν)`, le long de laquelle `p_c` et `ν` se
compensent. Un `(p_c, ν)` ne se rapporte qu'avec les tailles et la fenêtre
qui l'ont produit.

---

## 12. Récapitulatif

| Algorithme | Module | Coût dominant | Tailles atteintes | Tests |
| --- | --- | --- | --- | --- |
| Évolution hamiltonienne `exp(-iHt) v` | `models/base.py` | produits creux `H @ v` | `N_total ≈ 12–14` | `test_TFIM_model.py` |
| Porte à deux qubits | `models/circuits/base.py` | `O(2^N)` (vecteur), `O(4^N)` (`ρ`) | `N ≈ 20` (vecteur) | `test_circuits.py` |
| Mesure projective | `training/measurements/projective.py` | `O(2^N)` (vecteur), `O(4^N)` (`ρ`) | idem | `test_projective.py` |
| Tableau de stabilisateurs | `models/circuits/stabilizer.py` (`stim`) | `O(n²)` par opération | `N = 256` | `test_stabilizer_probe.py` |
| Groupe de l'effet Clifford | `evaluation/clifford_ridge.py`, `dictionary_spectrum.py` | rangs GF(2), énumération `2^r` | `N = 20` (énumération) | `test_clifford_ridge.py`, `test_dictionary_spectrum.py` |
| Kraus exact d'un record Haar | `evaluation/haar_mipt.py` | `O(4^N)` par porte | `N ≈ 12` | `test_haar_mipt.py` |
| Spectre d'intrication | `evaluation/entropy.py` | SVD `2^|A| × 2^|B|` | `N ≈ 20` | `test_entropy.py` |
| Ridge en forme close | `training/regression/ridge.py` | `O(S F² + F³)` | — | `test_ridge.py` |
| Matrice de transfert `C(p)` | `evaluation/fourier.py` | `O(T² N)` passages de Heisenberg `4^N` | `N ≲ 8` | `test_fourier.py` |
| Rang de l'espace des effets | `evaluation/effect_rank.py` | Gram `R × 4^N` par mesure | `N ≲ 8` | `test_effect_rank.py` |
| Spectre `F` et budget de R² | `evaluation/ridge_spectrum.py`, `two_filters.py` | `4^N` modes | `N ≈ 4–10` | `test_ridge_spectrum.py`, `test_two_filters.py` |
| SEP de Barratt et al. | `evaluation/learnability.py` | `O(2^L)` par porte | `L ≈ 14` | `test_learnability.py` |
| Collapse de taille finie | `evaluation/finite_size*.py` | grille + Nelder-Mead + bootstrap | — | `test_finite_size*.py` |

`S` est le nombre d'échantillons, `F` le nombre de features, `R` la dimension
du span courant. Les tailles atteintes sont des ordres de grandeur sur une
machine de bureau, pas des limites du code.
