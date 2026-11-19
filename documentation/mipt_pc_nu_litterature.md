# MIPT : couples (p_c, ν) de la littérature et comparaison avec Δ

Compilation du 19 novembre 2026 (issue #374). Version de travail partagée : document Claude « MIPT : couples (p_c, ν) de la littérature et comparaison ». Données brutes : [`mipt_pc_nu_litterature.csv`](mipt_pc_nu_litterature.csv).

## Résumé et conventions

En 1+1D avec qubits et mesures projectives Z, la littérature converge vers **p_c = 0,1593 à 0,1598 et ν = 1,28 à 1,30 pour Clifford** (Gullans et Huse, L jusqu'à 512) et **p_c = 0,168(5), ν = 1,2(2) pour Haar** (Zabalo et al., L ≤ 24). Pour Δ, qui est une Rényi-2, la référence Haar retenue est leur valeur n = 2 : p_c = 0,162(3), ν = 1,3(1). Nos résultats Δ reproduisent le couple Clifford à 0,001 près sur p_c et 0,02 sur ν. En Haar, ils retrouvent p_c (0,17 ± 0,01) mais pas ν (1,6 ± 0,3).

Les valeurs ci-dessous viennent du texte intégral des PDF de la collection Zotero « 05 MIPT / Fondements » (7 articles distincts). S'y ajoutent les articles des sous-collections « Théorie », « Apprenabilité et décodeurs » et « Expériences » qui publient un couple (p_c, ν). Chaque valeur a été relue dans le PDF, avec la table ou la figure indiquée dans les notes.

Quatre conventions décident si deux couples sont comparables :

- **Définition de p.** Presque tous les articles mesurent chaque site avec probabilité p après chaque couche de portes. C'est notre convention sur la brique périodique (#346, #348, #358), où une « demi-couche » est un pas de temps de Zabalo. Koh et al. et notre QELM ouvert (#322) ne mesurent qu'une fois toutes les deux couches : p y correspond à 1 − (1 − p_couche)².
- **Observable.** Les entropies bipartites S_n (n = 0, 1, 2, ∞) ont de fortes dérives de taille. I3 et les qubits de référence sont invariants d'échelle à p_c. Notre Δ = N − log2(1 + W(m)) est l'entropie de Rényi-2 de l'état purifié depuis I/2^N : c'est le paramètre d'ordre de purification de Gullans et Huse.
- **Bords.** Périodiques chez Zabalo, Gullans-Huse, Bao-Choi-Altman, Dehghani et chez nous (#346, #348, #358). Ouverts chez Skinner, Li-Chen-Fisher et Koh.
- **Fit.** Les barres d'erreur ne mesurent pas la même chose. Zabalo prend la région O ≤ 1,3 O_min de la fonction de Kawashima-Ito. Skinner retire les grandes tailles et extrapole. Nous donnons un bootstrap paramétrique, purement statistique, et l'écart entre variantes de données comme erreur systématique.

## Tableau principal

Une ligne par couple publié. « F » = collection Fondements, « T / A / E » = Théorie, Apprenabilité, Expériences. Les erreurs sont celles des auteurs, dans leur notation : 0,168(5) = 0,168 ± 0,005. « sans erreur » signale qu'aucune barre n'est publiée. Sauf mention, le circuit est une brique 1+1D de portes à 2 qubits, et p est la probabilité de mesurer Z sur chaque site après chaque couche.

| Papier | Circuit, bords, tailles | Observable | p_c | ν | Fit et erreur |
| --- | --- | --- | --- | --- | --- |
| [Zabalo et al. 2020](https://arxiv.org/abs/1911.00008) (F) | Haar, périodique, L = 8-24, t = 4L | I3 (quarts), S1 | 0,168(5) | 1,2(2) | Kawashima-Ito, erreur = région O ≤ 1,3 O_min |
| Zabalo et al. 2020 (F) | Haar, idem | I3, Rényi n = 2 | 0,162(3) | 1,3(1) | idem |
| Zabalo et al. 2020 (F) | Haar, idem | I3, n = 5 | 0,168(4) | 1,4(1) | idem |
| Zabalo et al. 2020 (F) | Haar, idem | I3, n = ∞ | 0,170(4) | 1,4(1) | idem |
| Zabalo et al. 2020 (F) | Haar, idem, cutoff 2,2·10⁻¹⁶ | I3, n = 0 | 0,45(3) | 1,5(3) | idem (exact : 1/2, 4/3) |
| Zabalo et al. 2020 (F) | Clifford, périodique, L = 8-24 | I3 (n indépendant) | 0,154(4) | 1,24(7) | idem |
| [Gullans et Huse, PRX 2020](https://link.aps.org/doi/10.1103/PhysRevX.10.041020) (F) | Clifford, périodique, L = 128-512 | I3 (L/4), état initial pur | 0,1593(5) | 1,28(2) | p_c au croisement, ν par collapse à p_c fixé ; méthode d'erreur non décrite |
| [Gullans et Huse, PRL 2020](https://arxiv.org/abs/1910.00020) (A) | Clifford, périodique, L = 64-256 | Entropie S_Q d'un qubit de référence | 0,1598(5) | 1,30(5) | idem |
| Gullans et Huse, PRL 2020 (A) | Clifford, périodique, L = 4-16 | S_Q | 0,16(1) | 1,3(2) | idem, petites tailles |
| [Choi, Bao, Qi, Altman 2020](https://arxiv.org/abs/1903.05124) (T) | Clifford, modèle en blocs (m = 11), profondeur d = 1, périodique | I3 (quarts) | 0,162, sans erreur | 1,28 ± 0,03 | coût en 2 étapes, bootstrap (80 p sur 100, 100 tirages) |
| [Dehghani et al. 2023](https://doi.org/10.1038/s41467-023-37902-1) (A) | Clifford, périodique, L = 32, 48, 64 | Taux de décroissance d'un qubit de référence, appris par réseau | 0,14 ± 0,03 | 1,5 ± 0,3 | carte d'erreur inverse, plage à 10 % du maximum ; z = 0,9 ± 0,15 |
| [Turkeshi 2022](https://doi.org/10.1103/PhysRevB.106.144313) (A) | Clifford, périodique, L ≥ 64 | 1re composante principale du tableau stabilisateur | 0,159(4) | 1,35(5) | collapse w̄₁ = L^(ζ/ν) f, ζ = 0,51(3) |
| Turkeshi 2022 (A) | idem | idem, forme soustraite | 0,165(6) | 1,30(7) | collapse de w̄₁(p) − w̄₁(p_c) |
| Turkeshi 2022 (A) | idem | Dimension intrinsèque (2NN) | 0,16(2) | 1,3(1) | collapse, α = 0,3(1) |
| [Skinner, Ruhman, Nahum 2019](https://arxiv.org/abs/1808.05953) (F) | Haar, ouvert, L = 6-24, t = 2L | S1 demi-chaîne | 0,26 ± 0,08 | 2,01 ± 0,10 | moindres carrés à la courbe moyenne, erreur par retrait des grandes tailles |
| Skinner et al. 2019 (F) | idem | S2 | 0,27 ± 0,03 | 2,31 ± 0,75 | idem |
| Skinner et al. 2019 (F) | idem | S∞ | 0,26 ± 0,08 | 2,25 ± 0,25 | idem |
| Skinner et al. 2019 (F) | Floquet (porte fixe), ouvert, t = 8L | S1 | 0,21 ± 0,16 | 1,74 ± 0,45 | idem |
| Skinner et al. 2019 (F) | idem | S2 | 0,22 ± 0,05 | 2,07 ± 0,22 | idem |
| Skinner et al. 2019 (F) | idem | S∞ | 0,21 ± 0,05 | 2,03 ± 0,07 | idem |
| Skinner et al. 2019 (F) | Haar, ouvert | S0 | 0,50 ± 0,05 | 1,36 ± 0,10 | idem (exact : 1/2, 4/3) |
| Skinner et al. 2019 (F) | Floquet, ouvert | S0 | 0,51 ± 0,01 | 1,36 ± 0,21 | idem |
| Skinner et al. 2019 (F) | Percolation classique (coupe minimale), t = 4L | S0 | 0,51 ± 0,01 | 1,24 ± 0,13 | idem |
| [Bao, Choi, Altman 2020](https://arxiv.org/abs/1908.04305) (T) | Haar, périodique, N = 10-30 | S1 demi-chaîne | 0,26 ± 0,02 | 2,9, sans erreur | collapse sur p ∈ [0,2 ; 0,35], figure à p_c = 0,27 |
| [Li, Chen, Fisher 2018](https://arxiv.org/abs/1808.06134) (F) | Clifford, ouvert, L = 16-512, mesure Z des deux qubits avant une porte (proba p par porte) | S_A, L_A = L/4 | 0,15, sans erreur | 1,85, sans erreur | p_c lu sur S(L), puis collapse S = L^γ F, γ = 0,30, p ∈ [0,05 ; 0,3] |
| Li, Chen, Fisher 2018 (F) | idem, mesure de rang 2 (parité ZZ) | S_A, L_A = L/4 | 0,68, sans erreur | 1,75, sans erreur | idem, γ = 0,33, p ∈ [0,3 ; 1] |
| [Koh et al. 2023](https://arxiv.org/abs/2203.04338) (E) | Processeur IBM, CX + rotations aléatoires, ouvert, L = 5-8, **une mesure par 2 couches** | S1 demi-chaîne | 0,25, imposé | 2,1 ± 0,3 | p* au pic de Var(S1) à L = 5, γ = 1,9 ± 0,4, moindres carrés |
| Koh et al. 2023 (E) | idem | S1 | 0,22, extrapolé (numérique) | 2,3 ± 0,3 | idem |
| [Turkeshi, Fazio, Dalmonte 2020](https://arxiv.org/abs/2007.02970) (F) | **2+1D** Clifford, plaquettes, périodique, L = 4-48, Z sur un site | S, bande L_A = L/2 | 0,54(1) (erratum), 0,650(5) (publié) | 0,67(1) (erratum), 0,56(1) (publié) | collapse de S(p) − S(p_c), fits polynomiaux |
| Turkeshi et al. 2020 (F) | **2+1D**, idem, ZZ voisins (rang 2) | idem | 0,84(1) (erratum), 0,890(3) (publié) | 0,68(1) (erratum), 0,54(1) (publié) | idem |

## Notes par papier

### Collection Fondements

**Zabalo, Gullans, Wilson, Gopalakrishnan, Huse, Pixley, PRB 101, 060301(R) (2020).** C'est la référence Haar. Un pas de temps est une couche de portes suivie d'une couche de mesures Z, chaque site étant mesuré avec probabilité p. L'état initial est un produit, l'évolution est exacte (Haar) ou stabilisatrice (Clifford), avec des bords périodiques. Les mesures sont prises à t = 4L, en moyennant les deux parités de couche (Supplément, « Numerical methods »). Les croisements de I3 sont lus sur L = 16, 20, 24. Le fit minimise la fonction O(p_c, ν) de Kawashima et Ito : chaque point est comparé à la droite de ses deux voisins, et O = 1 pour un collapse idéal. L'erreur est la région O ≤ 1,3 O_min. Tous les couples viennent de la Table I. Le résumé arrondit p_c à 0,17(1). ν va de 1,2(2) à 1,4(1) quand n passe de 0,7 à ∞. Autres exposants : z = 1,06(4) (L = 12, 16, 20 à p = 0,17), η = 0,19(1) en Haar et 0,22(1) en Clifford. I2 entre régions antipodales de taille L/4 croise aussi à p_c, mais avec des dérives plus fortes.

**Gullans et Huse, PRX 10, 041020 (2020).** Brique Clifford périodique, mesures Z avec probabilité p entre chaque couche. Le croisement de ⟨I3⟩, à partir d'un état pur, donne p_c = 0,1593(5). Le collapse de L = 128 à 512 à ce p_c donne ν = 1,28(2). La densité d'entropie de l'état parti de I/2^N, à t = 4L et L = 256-512, suit A(p_c − p)^ν avec A ≈ 7,3 et les mêmes p_c et ν (Fig. 2b). Le collapse de la dynamique de purification, sur L = 64-512 avec z = 1, réutilise ces valeurs. **C'est l'article le plus proche de notre Δ** : même circuit, même paramètre d'ordre de purification. L'article ne décrit pas la méthode qui donne les erreurs entre parenthèses.

**Skinner, Ruhman, Nahum, PRX 9, 031009 (2019).** Chaîne ouverte en MPS (ITensor), mesures Z avec probabilité p sur chaque spin après chaque couche, état initial produit. Deux dynamiques : des portes Haar indépendantes (t = 2L), ou une porte Floquet fixe non intégrable (t = 8L). L = 6 à 24. Le collapse porte sur S_n(L, p) − S_n(L, p_c) contre (p − p_c)L^(1/ν), sans préfacteur. Le coût est la somme des résidus au carré par rapport à la courbe moyenne, minimisée par descente de gradient depuis plusieurs points de départ (Annexe A). L'erreur est obtenue en retirant les tailles L > L0, pour L0 de Lmax/2 à Lmax, puis en extrapolant linéairement en 1/L0 : c'est l'écart entre L0 = Lmax/2 et l'extrapolation. Toutes les valeurs viennent de la Table I. La moyenne pondérée des six ν à n > 0 est 2,03 ± 0,05 (éq. 19). Les limites t → ∞ et L → ∞ donnent ν_L = 1,99 ± 0,20 et ν_t = 2,00 ± 0,37. L'information mutuelle de deux spins donne p_c = 0,267 ± 0,027 (Fig. 15). Ces p_c ≈ 0,26 et ν ≈ 2 ont ensuite été révisés à la baisse par Zabalo et al., qui les attribuent à des effets de taille.

**Li, Chen, Fisher, PRB 98, 205136 (2018).** Brique Clifford ouverte. Chaque porte est, avec probabilité p, précédée d'une mesure projective sur la paire. Modèle B1 : projecteurs de rang 1, soit Z sur les deux qubits. Modèle B2 : projecteurs de rang 2, soit la parité ZZ. L = 16 à 512, sous-système L_A = L/4. p_c est lu sur S(L) en échelle log (Fig. 3), sans croisement. Puis γ et ν sont choisis pour le meilleur collapse de S_A = L^γ F((p − p_c)L^(1/ν)) (Fig. 4 et 5). Aucune barre d'erreur n'est donnée. Les modèles Haar A1 et A2 ne sont étudiés qu'à p = 1, L ≤ 16, donc sans couple (p_c, ν).

**Turkeshi, Fazio, Dalmonte, arXiv:2007.02970 (2020).** Réseau carré L × L en 2+1D, portes Clifford sur des plaquettes décalées, bords périodiques, 10⁴ circuits par point. Entropie d'une bande N_A = L × L/2, L = 4 à 48. **La version arXiv de la bibliothèque contient un erratum.** Le rang calculé par SVD était faux, il est refait par élimination de Gauss sur F₂. Les valeurs corrigées sont 0,54(1) et 0,67(1) au rang 1, 0,84(1) et 0,68(1) au rang 2. Les conclusions qualitatives tiennent : classe différente de la percolation 3D (ν = 0,87). Hors de notre géométrie, à ne citer que pour la dimension.

**Fisher, Khemani, Nahum, Vijay, Annu. Rev. CMP 14, 335 (2023).** C'est une revue, sans fit propre. Elle cite p_c ≈ 0,17 pour Clifford en 1+1D avec z = 1, et la limite q → ∞ (percolation, p_c = 1/2, ν = 4/3). Attention, ce 0,17 attribué à Clifford est plus haut que les 0,159-0,160 des articles dédiés. Il ne faut pas l'utiliser comme référence.

**Kelson-Packer, Sukeno, Miyake, arXiv:2610.36528 (2026).** Circuits de jauge-Higgs à mesures seules, sur réseaux euclidiens et hyperboliques, qui se ramènent à de la percolation. Les p_c (Table II, par exemple 0,248(7) contre 0,2488 en percolation 3D sur {4,4}) sont obtenus par collapse de la probabilité de croisement, avec ν proche des valeurs de percolation (4/3 en 2D, 0,88 en 3D). Aucun ν n'est ajusté numériquement, et il n'y a pas de dynamique unitaire. Ce papier n'est pas comparable à Δ.

### Autres sous-collections MIPT

**Gullans et Huse, PRL 125, 070606 (2020), « Scalable probes ».** Même circuit Clifford périodique. Un qubit de référence est intriqué au système, et le paramètre d'ordre S_Q est son information cohérente. p_c = 0,1598(5) au croisement pour 64 ≤ L ≤ 256, ν = 1,30(5) par collapse. **Sur 4 ≤ L ≤ 16, ils trouvent déjà (0,16(1), 1,3(2))** : c'est la référence à citer pour la qualité d'un estimateur à petite taille.

**Choi, Bao, Qi, Altman, PRL 125, 030505 (2020).** Modèle Clifford en blocs de m qubits, de profondeur locale d, périodique. d = 1 correspond à peu près à la brique. La Table SI (I3 des quarts, m = 11) donne p_c = 0,162 et ν = 1,28 ± 0,03 à d = 1. ν reste entre 1,12 et 1,30 de d = 3 à 31, pour p_c de 0,41 à 0,89. L'erreur vient d'un bootstrap : 80 des 100 valeurs de p sont tirées, 100 fois. p_c n'a pas de barre. Le collapse de l'entropie bipartite donne ν ≈ 1, que les auteurs jugent moins fiable. Les tailles exactes de la Table SI n'y sont pas écrites (la Fig. S6 va jusqu'à L = 64).

**Bao, Choi, Altman, PRB 101, 104301 (2020).** Théorie : limite q → ∞ vers le modèle de Potts, p_c = 1/2. Simulation exacte Haar à q = 2 pour 10 ≤ N ≤ 30, bords périodiques, mesure projective avec probabilité p après chaque couche. Le collapse de S(p, N) − S(p_c, N), sur la fenêtre p ∈ [0,2 ; 0,35], donne p_c = 0,26 ± 0,02 pour l'entropie de von Neumann. La figure montre le collapse à p_c = 0,27 et ν = 2,9, sans erreur sur ν. Ce sont les mêmes dérives d'entropie bipartite que Skinner.

**Dehghani, Lavasani, Hafezi, Gullans, Nat. Commun. 14, 2918 (2023).** Circuit Clifford périodique. Un réseau de neurones apprend l'état d'un qubit de référence à partir des résultats de mesure. Le paramètre d'ordre est le taux de décroissance appris, sur L = 32, 48, 64. Le collapse L^z λ contre (p − p_c)L^(z/ν) donne d'abord p_c ≃ 0,13 et ν ≃ 1,5 à z = 1. La recherche sur trois paramètres donne p_c = 0,14 ± 0,03, ν = 1,5 ± 0,3 et z = 0,9 ± 0,15, plage à 10 % du maximum de l'inverse de l'erreur. Les simulations exactes croisent vers 0,13 aux mêmes tailles, et vers 0,16 à plus grande taille. C'est un analogue proche de notre approche par l'apprenabilité.

**Turkeshi, PRB 106, 144313 (2022).** Apprentissage non supervisé sur le tableau stabilisateur, brique Clifford périodique, tailles L ≥ 64 pour le collapse. La première composante principale quantifiée donne (0,159(4), 1,35(5)) en forme multiplicative et (0,165(6), 1,30(7)) en forme soustraite. Un choix de tableau aléatoire donne (0,159(7), 1,35(8)), et la dimension intrinsèque (0,16(2), 1,3(1)). La référence citée est (0,1599(1), 1,27(1)). C'est le précédent le plus proche d'un paramètre d'ordre dans l'espace des données, comme notre W(m).

**Koh, Sun, Motta, Minnich, Nature Physics 19 (2023).** Processeurs IBM, portes à 2 qubits faites de rotations Haar à un qubit et d'un CX orienté au hasard. **Un pas de temps compte deux couches de portes puis une couche de mesures** : c'est notre convention QELM ouverte de #322, pas celle de Zabalo. Entropie S1 de la demi-chaîne par tomographie, avec correction de lecture et RE. p* ≈ 0,25 est imposé au pic de Var(S1) à L = 5. Le fit par moindres carrés sur L = 5-8 donne γ ≈ 1,9 ± 0,4 et ν ≈ 2,1 ± 0,3. Avec p* ≈ 0,22 extrapolé numériquement, ν ≈ 2,3 ± 0,3. Avec la correction RE simplifiée, ν ≈ 3,3 ± 0,5 (Suppl. Note 5). Ramené à une couche, p* = 0,22-0,25 vaut environ 0,12-0,13, avec des portes moins intriquantes que Haar.

**Sans couple (p_c, ν) exploitable.** Jian et al., PRB 101, 104302 (2020) est analytique (Haar de dimension q, percolation quand q → ∞). Li, Vasseur, Fisher, Ludwig, PRB 109, 174307 (2024) traite le modèle stat-méca Clifford sans fit. Barratt et al., PRL 129, 200602 (2022) donne le partage de charge U(1) à p# ≃ 0,09, sans ν. Agrawal et al., PRX 14, 041012 (2024) donne γ_c ≈ 0,4 pour une force de mesure. Kim et al. 2025 donne μ_c = 0,39 à L = 12 pour une force de mesure. Les articles de mesures continues ne publient pas de couple dans cette convention.

## Comparaison avec nos résultats

Notre Δ = N − log2(1 + W(m)), calculé à partir du poids de Fourier dense W(m) du record, est l'entropie de Rényi-2 du système parti de I/2^N. **Son équivalent publié exact est l'entropie de purification de Gullans et Huse (PRX 2020)**, sur le même circuit Clifford périodique avec la même définition de p. En Haar, aucun article de la collection ne fitte la purification. La référence la plus proche est Zabalo et al. en Rényi n = 2, sur un état pur et avec I3.

Nos valeurs viennent des expériences #346 (Clifford Δ), #348 (Haar Δ), #358 (entropie bipartite et I3 du dictionnaire) et #322 (QELM ouvert). Elles sont prises sur la brique périodique, avec une mesure Z par site et par demi-couche, sauf la dernière ligne. Les erreurs sont statistiques (bootstrap, 40 tirages), suivies de l'erreur systématique entre variantes quand elle est donnée.

| Notre résultat | Données | p_c | ν | Référence comparable | Verdict |
| --- | --- | --- | --- | --- | --- |
| Δ, Clifford (#346) | T = 2N, N = 16-128, Q_min = 4,45 | 0,1610(2), syst. ± 0,001 | 1,299(5), syst. ± 0,05 | Gullans-Huse PRX : 0,1593(5) ; 1,28(2) | **Comparable.** ν identique, p_c plus haut de 0,0017. L'écart tombe à 0,0013 sans N = 16 |
| Δ, Clifford (#346) | T = 4N, N = 16-64 | 0,1611(2) | 1,340(8) | Idem, t = 4L dans les deux cas | Comparable, ν à environ 3 σ de 1,28(2) |
| I3 du dictionnaire, Clifford (#358) | N = 16-256, p ∈ [0,10 ; 0,24] | 0,1588(3) | 1,28(1) | Gullans-Huse PRX (I3) : 0,1593(5) ; 1,28(2) | **Comparable**, accord dans les barres |
| S(N/2) du dictionnaire, Clifford (#358) | N = 32-256, forme soustraite | 0,1641(12) | 1,21(3) | Li-Chen-Fisher 2018 : 0,15 ; 1,85 | Peu comparable : bords ouverts, L_A = L/4, forme L^γ F, pas d'erreur publiée |
| Δ, Haar (#348) | T = 2N, N = 4-12, Q_min = 1,98 | 0,1805(8) | 1,59(2) | Zabalo n = 2 : 0,162(3) ; 1,3(1) | Partiel : autre observable, et p_c tiré vers le haut par N = 4 |
| Δ, Haar (#348) | T = 4N, N = 6-10 | 0,1690(16) | 1,88(3) | Idem, t = 4L dans les deux cas | p_c compatible, ν non (+0,6) |
| Δ, Haar, lecture globale (#348) | Toutes variantes | 0,17 ± 0,01 | 1,6 ± 0,3 | Zabalo n = 1 : 0,168(5) ; 1,2(2) | p_c retrouvé, ν à environ 1 σ |
| I3 du dictionnaire, Haar (#358) | N = 8, 12, 16, 20, Rényi 1 | 0,1888(10) | 1,05(2) | Zabalo n = 1 (I3) : 0,168(5) ; 1,2(2) | Même observable. p_c plus haut de 0,02, ν dans la barre de Zabalo |
| I3 du dictionnaire, Haar (#358) | Idem, Rényi 2 | 0,1809(9) | 1,03(1) | Zabalo n = 2 : 0,162(3) ; 1,3(1) | Même observable, p_c plus haut de 0,02, ν plus bas de 0,3 |
| Qubit de référence, QELM ouvert (#322) | Haar et Clifford, N = 6-14, croisements | 0,29 ± 0,02 | non déterminé | 1 − (1 − 0,168)² = 0,31 ; Koh et al. : 0,22-0,25 | Même convention de p que Koh (une mesure par 2 couches). Il faut convertir avant toute comparaison |

**Ce qui est comparable.**

- **Clifford Δ et Gullans-Huse** : même circuit, même p, même bord, même paramètre d'ordre, et toutes les entropies de Rényi sont égales en Clifford. C'est notre comparaison la plus forte, et elle tient.
- **Le temps.** Notre T compte des demi-couches, comme le t de Zabalo et de Gullans-Huse. La variante T = 4N est donc la bonne pour comparer à leur t = 4L. T = 2N est plus proche de Skinner (t = 2L, mais bords ouverts).
- **Haar.** On compare Δ à Zabalo n = 2 plutôt qu'à n = 1, puisque Δ est une Rényi-2. Le p_c passe alors de 0,168 à 0,162, et ν de 1,2(2) à 1,3(1).
- **Petites tailles.** Gullans-Huse PRL obtiennent déjà (0,16(1), 1,3(2)) en Clifford sur 4 ≤ L ≤ 16, avec un qubit de référence. C'est la meilleure référence pour juger notre Haar à N ≤ 12 : leur estimateur ne gonfle pas ν à ces tailles, alors que notre Δ Haar le fait.

**Ce qui ne l'est pas.**

- Skinner et Bao-Choi-Altman (p_c ≈ 0,26, ν ≈ 2-2,9) : entropie bipartite, bords ouverts pour Skinner, L ≤ 24-30. Ces valeurs sont dépassées pour le p_c Haar. Elles ne valent que comme exemple de dérive de S_n.
- Notre Floquet de #333 (croisements de Δ à 0,12-0,14, collapse à 0,108(6) et ν = 1,10) contre le Floquet de Skinner (S2 : 0,22 ± 0,05 ; 2,07 ± 0,22) : l'observable et les bords diffèrent, et la porte Floquet doit être vérifiée. Ce n'est pas une comparaison directe.
- Li-Chen-Fisher, Turkeshi 2+1D et Kelson-Packer : autre géométrie de mesure, autre dimension, ou autre modèle.

**Pour aligner les barres d'erreur.** Nos erreurs bootstrap (± 0,0002 sur p_c) sont 10 fois plus petites que l'écart entre nos variantes. Pour une comparaison à armes égales, deux options. On peut donner la région Q ≤ 1,3 Q_min, qui est l'équivalent direct de l'erreur de Zabalo. On peut aussi reprendre la procédure de Skinner : retirer les plus grandes tailles et extrapoler en 1/N_max. Nos cartes ont déjà les contours 1,5 et 2 Q_min, donc le contour 1,3 ne demande qu'un recalcul.

## Valeurs non vérifiées, limites et sources

**Barres d'erreur absentes dans l'article.** Li-Chen-Fisher 2018 (tous les couples), le ν = 2,9 de Bao-Choi-Altman, le p_c = 0,162 de Choi et al., et le p* de Koh, imposé et non fitté. Gullans et Huse donnent des erreurs entre parenthèses sans décrire comment elles sont obtenues.

**Points à vérifier.**

- Turkeshi, Fazio, Dalmonte : le PDF de la bibliothèque est la version arXiv avec erratum. La version publiée n'a pas été lue. Les valeurs « publié » sont celles du corps de cette version arXiv, et l'erratum dit qu'elles sont fausses.
- Choi et al. : les tailles utilisées pour la Table SI ne sont pas écrites près de la table.
- Fisher et al. : le p_c ≈ 0,17 attribué à Clifford contredit les articles dédiés (0,159-0,160). Il n'est pas utilisé.
- Li, Chen, Fisher, PRB 100, 134306 (2019), qui donne le couple Clifford à partir de l'entropie bipartite, n'est pas dans la bibliothèque Zotero. Ses valeurs n'ont pas été relues.
- Notre Floquet (#333) : il faut vérifier que la porte fixe est bien celle de Skinner avant toute comparaison.

**Limites de cette compilation.** Le texte des PDF a été extrait avec pdftotext. Les tables ont été relues ligne à ligne, mais une valeur lue seulement sur une figure n'est pas reprise ici. Les couples hors convention (force de mesure γ, μ, partage de charge p#) sont listés dans les notes, pas dans le tableau.

**Sources.** Les PDF de la bibliothèque Zotero, collections « 05 MIPT / Fondements », « Théorie », « Apprenabilité et décodeurs » et « Expériences ». Les liens du tableau principal mènent aux pages arXiv ou DOI enregistrées dans Zotero.
