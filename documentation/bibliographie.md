# Bibliographie vérifiée et liste de lecture

*Note du 12 novembre 2026. Elle recense les articles cités dans le dépôt, vérifie
chaque référence, et propose une liste de lecture classée par thème et par
priorité pour comprendre le projet : un QELM (ridge sur le registre de mesures
`M` et sur les valeurs moyennes de l'état final) lu à travers la transition de
phase induite par la mesure (MIPT).*

## Méthode

- **Ce qui a été vérifié.** Pour chaque article : titre, liste des auteurs,
  date de dépôt, identifiant arXiv et DOI, sur la page `arxiv.org/abs/<id>` de
  l'article. Le journal, le volume et le numéro d'article sont pris dans le DOI
  quand celui-ci les contient (APS : `10.1103/PhysRevX.9.031009` donne PRX 9,
  031009). Pour les DOI qui ne contiennent pas le volume (Nature, npj, Springer),
  seuls le journal et l'année sont donnés.
- **Ce qui n'a pas été vérifié.** Le contenu scientifique (valeurs de `p_c`, de
  `ν`, numéros de figure ou de table) n'a pas été relu, sauf mention contraire.
  Les affirmations de contenu sur Dehghani et al., Agrawal et al. (2024) et
  Barratt et al. (2022) ont été relues dans la PR #298.
- **Marquage.** ✅ vérifié sur arXiv ; ⚠️ partiellement vérifié (voir la
  remarque) ; ❌ introuvable ou faux.

## 1. Vérification des citations du dépôt

Recherche faite dans `README.md`, `documentation/`, `conf/`, `src/`,
`experiments/`, `notebooks/`, `tests/`, dans les issues ouvertes et dans la PR
#298. Toutes les références bibliographiques trouvées existent et
correspondent à ce qui est cité. Il n'y a **aucune erreur de volume, de numéro
d'article ou d'identifiant arXiv**. Les seuls problèmes sont des références
incomplètes et une référence fantôme, déjà signalée dans #298.

| Citation dans le dépôt | Où | Statut | Référence vérifiée |
|---|---|---|---|
| Skinner, Ruhman, Nahum, PRX 9, 031009 (2019) | `conf/models/floquet_circuit.yaml`, `src/QELM/models/circuits/floquet.py`, `experiments/QELM_R`, `QELM_S`, notes | ✅ | arXiv:1808.05953 |
| Gullans & Huse, PRL 125, 070606 (2020), arXiv:1910.00020 | `conf/purification_gullans_huse.yaml`, `src/QELM/encoding/bell_pair.py`, `stabilizer.py`, notebook 04 | ✅ | « Scalable probes of measurement-induced criticality » |
| Gullans & Huse, PRX 10, 041020 (2020), arXiv:1905.05195 | `src/QELM/evaluation/purification.py`, `effect_mipt.py`, `operator_cones.py` | ✅ | « Dynamical purification phase transitions induced by quantum measurements » |
| Zabalo et al., PRB 101, 060301 (2020) | `conf/QELM_S.yaml`, `entropy.py`, `operator_cones`, `QELM_S` | ✅ | arXiv:1911.00008 (article de type Rapid Communication, 060301(R)) |
| Li, Chen & Fisher, PRB 100 (2019) | `experiments/purification/steady_state.py`, `stabilizer.py` | ✅ | PRB 100, 134306, arXiv:1901.08092. Le numéro d'article manque dans le code. |
| Agrawal et al., PRX 12, 041002 (2022) | `src/QELM/evaluation/charge_variance.py`, `experiments/charge_variance` | ✅ | arXiv:2107.10279 |
| Barratt et al., PRL 129, 120604 (2022) | idem | ✅ | « Field theory of charge sharpening in symmetric monitored quantum circuits », arXiv:2111.09336 |
| Barratt et al., PRL 129, 200602 (2022), arXiv:2206.12429 | `conf/learnability.yaml`, `learnability.py`, notebooks 01 et 02 | ✅ | auteurs F. Barratt, U. Agrawal, A. C. Potter, S. Gopalakrishnan, R. Vasseur, conformes aux notebooks |
| Agrawal et al., PRX 14, 041012 (2024), arXiv:2311.00058 | `conf/charge_sharpening.yaml`, `weak.py`, `charge_sector.py`, notebook 03 | ✅ | « Observing quantum measurement collapse as a learnability phase transition » |
| Dehghani et al., Nat. Commun. 14, 2918 (2023) | `dehghani_decoder`, `haar_readouts`, notebook 09 | ✅ | arXiv:2204.10904, DOI 10.1038/s41467-023-37902-1. Le volume et le numéro viennent de la vérification de #298. |
| Kim & Huse, PRL 111, 127205 (2013) | `conf/models/continuous_tfim_1d.yaml`, `chaos_locality.py` | ✅ | « Ballistic spreading of entanglement in a diffusive nonintegrable system », arXiv:1306.4306 |
| Atas et al., PRL 110, 084101 (2013) | `chaos_locality.py` | ✅ | arXiv:1212.5611 |
| Parker et al., PRX 9, 041017 (2019) | `operator_cones` | ✅ | « A universal operator growth hypothesis », arXiv:1812.08657 |
| Nahum, Vijay, Haah, PRX 8, 021014 (2018) | `operator_cones.py` | ✅ | arXiv:1705.08975 |
| von Keyserlingk et al., PRX 8, 021013 (2018) | `operator_cones.py` | ✅ | arXiv:1705.08910 |
| Houdayer & Hartmann, PRB 70, 014418 (2004) | `finite_size_scaling.py` | ✅ | arXiv:cond-mat/0402036 |
| Feldmeier, Sala, De Tomasi, Pollmann, Knap, PRL 125, 245303 (2020) | `src/QELM/models/circuits/symmetric.py` | ✅ | arXiv:2004.00635 |
| Fattal et al. 2004 | `stabilizer.py` | ✅ | « Entanglement in the stabilizer formalism », arXiv:quant-ph/0406168. Pas de journal indiqué sur arXiv. |
| Ledoit-Wolf | `experiments/pnt_decay` | ✅ | J. Multivariate Anal. 88, 365–411 (2004) (vérifié sur RePEc) |
| Fisher, Khemani, Nahum, Vijay, Annu. Rev. Condens. Matter Phys. 14 (2023) | `documentation/notes/qelm-voit-il-la-mipt.md` (C4, « cité de mémoire ») | ✅ | « Random Quantum Circuits », vol. 14, p. 335–379, arXiv:2207.14280. La note peut retirer la mention « cité de mémoire » pour la référence ; le contenu de C4 n'a pas été relu. |
| « Harnessing quantum extreme learning machines for image classification », Phys. Rev. Applied (2025) | `experiments/double_descent`, `src/QELM/encoding/base.py`, `conf/double_descent.yaml`, notebook 07 | ⚠️ incomplète | De Lorenzis, Casado, Estarellas, Lo Gullo, Lux, Plastina, Riera, Settino, **Phys. Rev. Applied 23, 044024 (2025)**, arXiv:2409.00998. Ajouter les auteurs et le numéro. |
| « Xiong et al. » (concentration exponentielle) | `experiments/operator_cones/README.md` | ⚠️ incomplète | Xiong, Facelli, Sahebi, Agnel, Chotibut, Thanasilp, Holmes, « On fundamental aspects of quantum extreme learning machines », arXiv:2312.15124, DOI 10.1007/s42484-025-00239-7 (Quantum Mach. Intell., 2025). L'abstract parle bien de « exponential concentration ». |
| Dehghani et al. 2023 « deep reference » | `conf/nonlinear_M.yaml` | ✅ | même article que ci-dessus |
| Innocenti et al. 2023 | issue #193 | ✅ | Commun. Phys. 6, 118 (2023), arXiv:2210.00780 |
| Turkeshi, PRB 106, 144313 (2022) | issue #272 | ✅ | « Measurement-induced criticality as a data-structure transition », arXiv:2101.06245 |
| De Lorenzis et al. 2024/2025 | issue #279 | ✅ | voir la ligne « Harnessing… » |
| Kobayashi & Motome (bord du chaos, SYK) | issue #86 | ⚠️ incomplète | « Edge of many-body quantum chaos in quantum reservoir computing », arXiv:2506.17547, DOI 10.1103/j2qj-vwcl. Le journal (PRL) vient du communiqué de l'université de Tokyo, pas de la page arXiv. |
| Kobayashi & Motome, Nat. Commun. (2025) et arXiv:2308.00898 | PR #298 | ✅ | arXiv:2402.07097 (DOI 10.1038/s41467-025-58751-0) ; arXiv:2308.00898 est publié dans SciPost Phys. 18, 198 (2025) d'après son DOI |
| Kim, von Keyserlingk, Lamacraft, arXiv:2504.08888 | PR #298 | ✅ | « Measurement-induced phase transitions in quantum inference problems and quantum hidden Markov models », préprint |
| « Rudolph et al., PRX Quantum 2024 » | critique citée dans #297 / PR #298 | ❌ | Introuvable, déjà signalé par #298. Ne pas citer. |

### Points de contenu à nuancer (relevés en passant)

- **Issue #264, `ν ≈ 4/3`.** `4/3` est l'exposant de la percolation 2D, qui
  décrit la limite de grande dimension locale (et l'entropie de Hartley `S_0`
  des circuits Haar, Skinner et al. 2019). Pour les qubits, Zabalo et al. (2020)
  et Gullans & Huse (2020) donnent `ν ≈ 1,2–1,3`, la valeur déjà utilisée dans le
  notebook 04 (`ν = 1,3`). À relire dans les deux articles avant de fixer `ν`.
- **Issue #270, borne d'erreur.** `P_err ≥ (1/4) e^{-D_KL}` est l'inégalité de
  Bretagnolle–Huber (erreur bayésienne à priors égaux), pas une borne de
  Chernoff. La borne est juste, son nom est à corriger.

## 2. Liste de lecture

Priorités : **P1** à lire en premier, indispensable pour défendre les
résultats du projet ; **P2** important pour un chapitre ou une expérience
précise ; **P3** contexte, à lire si le sujet devient central. Chaque entrée
donne la raison de la lire pour ce projet. Tous les identifiants ont été
vérifiés sur arXiv le 12 novembre 2026.

### 2.1 La MIPT : fondements

| P | Référence | Pourquoi la lire |
|---|---|---|
| P1 | M. P. A. Fisher, V. Khemani, A. Nahum, S. Vijay, « Random Quantum Circuits », Annu. Rev. Condens. Matter Phys. 14, 335 (2023), [arXiv:2207.14280](https://arxiv.org/abs/2207.14280) | La revue de référence. Elle pose le vocabulaire (loi de volume/d'aire, purification, répliques, problème de la post-sélection) et le fait qu'une quantité linéaire en `ρ` moyennée sur les trajectoires ne voit pas la MIPT, base de la conjecture C4. |
| P1 | B. Skinner, J. Ruhman, A. Nahum, « Measurement-Induced Phase Transitions in the Dynamics of Entanglement », PRX 9, 031009 (2019), [arXiv:1808.05953](https://arxiv.org/abs/1808.05953) | Le circuit Floquet du dépôt (`floquet_circuit`), l'image de la coupe minimale (utile pour #319, rang = `2^{min cut}`) et le collapse de taille finie de `finite_size.py`. |
| P1 | Y. Li, X. Chen, M. P. A. Fisher, « Quantum Zeno effect and the many-body entanglement transition », PRB 98, 205136 (2018), [arXiv:1808.06134](https://arxiv.org/abs/1808.06134) | Un des deux articles fondateurs, avec les circuits Clifford simulés comme dans `stabilizer.py`. |
| P1 | Y. Li, X. Chen, M. P. A. Fisher, « Measurement-driven entanglement transition in hybrid quantum circuits », PRB 100, 134306 (2019), [arXiv:1901.08092](https://arxiv.org/abs/1901.08092) | Le `S_half` stationnaire de `steady_state.py` et les grandes tailles en Clifford. |
| P2 | A. Chan, R. M. Nandkishore, M. Pretko, G. Smith, « Unitary-projective entanglement dynamics », PRB 99, 224307 (2019), [arXiv:1808.05949](https://arxiv.org/abs/1808.05949) | Le troisième article fondateur, paru le même jour que Skinner et al. ; à citer avec eux. |
| P1 | M. J. Gullans, D. A. Huse, « Dynamical purification phase transitions induced by quantum measurements », PRX 10, 041020 (2020), [arXiv:1905.05195](https://arxiv.org/abs/1905.05195) | La MIPT vue comme une transition de purification : c'est exactement le label « `R` purifié ou non » des expériences QELM, et l'entropie de l'effet `Δ` de #325/#327 est une entropie de purification. |
| P1 | M. J. Gullans, D. A. Huse, « Scalable probes of measurement-induced criticality », PRL 125, 070606 (2020), [arXiv:1910.00020](https://arxiv.org/abs/1910.00020) | Le qubit de référence, `p_c ≈ 0,16` en Clifford, reproduit dans le notebook 04. C'est la sonde que le QELM essaie d'apprendre. |
| P1 | A. Zabalo, M. J. Gullans, J. H. Wilson, S. Gopalakrishnan, D. A. Huse, J. H. Pixley, « Critical properties of the measurement-induced transition in random quantum circuits », PRB 101, 060301(R) (2020), [arXiv:1911.00008](https://arxiv.org/abs/1911.00008) | `p_c ≈ 0,17` et `ν` du brickwork Haar de qubits, l'information mutuelle tripartite `I_3` (`plot_estimators.py`). Référence pour toute comparaison de `p_c` et de `ν`. |
| P2 | S. Choi, Y. Bao, X.-L. Qi, E. Altman, « Quantum error correction in scrambling dynamics and measurement-induced phase transition », PRL 125, 030505 (2020), [arXiv:1903.05124](https://arxiv.org/abs/1903.05124) | La phase de loi de volume comme code correcteur : pourquoi les mesures locales `M` ne contiennent presque rien sur l'information encodée sous `p_c` (lien avec T4 et la décroissance de `D_t`). |
| P2 | Y. Bao, S. Choi, E. Altman, « Theory of the phase transition in random unitary circuits with measurements », PRB 101, 104301 (2020), [arXiv:1908.04305](https://arxiv.org/abs/1908.04305) | Le modèle de physique statistique à répliques ; explique pourquoi la transition dépend de l'indice de Rényi et la limite percolation (`ν = 4/3`). |
| P2 | C.-M. Jian, Y.-Z. You, R. Vasseur, A. W. W. Ludwig, « Measurement-induced criticality in random quantum circuits », PRB 101, 104302 (2020), [arXiv:1908.08051](https://arxiv.org/abs/1908.08051) | Même construction, vue du côté théorie conforme ; à lire avec Bao et al. |
| P2 | Y. Li, X. Chen, A. W. W. Ludwig, M. P. A. Fisher, « Conformal invariance and quantum nonlocality in critical hybrid circuits », PRB 104, 104305 (2021), [arXiv:2003.12721](https://arxiv.org/abs/2003.12721) | Exposants critiques des circuits Clifford, utiles pour les collapses à grande taille (#327). |
| P2 | Y. Li, R. Vasseur, M. P. A. Fisher, A. W. W. Ludwig, « Statistical mechanics model for Clifford random tensor networks and monitored quantum circuits », PRB 109, 174307 (2024), [arXiv:2110.02988](https://arxiv.org/abs/2110.02988) | Pourquoi la transition Clifford diffère de la transition Haar ; à lire avant de comparer les deux classes de portes. |
| P2 | A. C. Potter, R. Vasseur, « Entanglement dynamics in hybrid quantum circuits », chapitre de livre (Springer, 2022), [arXiv:2111.08018](https://arxiv.org/abs/2111.08018), DOI 10.1007/978-3-031-03998-0_9 | Revue pédagogique, plus courte que Fisher et al., avec les circuits symétriques U(1). |
| P3 | A. Nahum, S. Roy, B. Skinner, J. Ruhman, « Measurement and entanglement phase transitions in all-to-all quantum circuits, on quantum trees, and in Landau-Ginsburg theory », PRX Quantum 2, 010352 (2021), [arXiv:2009.11311](https://arxiv.org/abs/2009.11311) | Circuits tout-à-tout, comme le TFIM tout-à-tout de `QELM_S`. |
| P3 | M. Ippoliti, M. J. Gullans, S. Gopalakrishnan, D. A. Huse, V. Khemani, « Entanglement phase transitions in measurement-only dynamics », PRX 11, 011030 (2021), [arXiv:2004.09560](https://arxiv.org/abs/2004.09560) | Transitions sans portes unitaires, utile pour la limite `p → 1`. |
| P3 | L. Fidkowski, J. Haah, M. B. Hastings, « How dynamical quantum memories forget », Quantum 5, 382 (2021), [arXiv:2008.10611](https://arxiv.org/abs/2008.10611) | La phase « intriquée » ne protège l'information que pendant un temps fini à taille finie : important pour le choix de la profondeur `T` (`T = N`, `2N`). |

### 2.2 Symétrie U(1), sharpening de charge et apprenabilité

| P | Référence | Pourquoi la lire |
|---|---|---|
| P1 | U. Agrawal, A. Zabalo, K. Chen, J. H. Wilson, A. C. Potter, J. H. Pixley, S. Gopalakrishnan, R. Vasseur, « Entanglement and charge-sharpening transitions in U(1) symmetric monitored quantum circuits », PRX 12, 041002 (2022), [arXiv:2107.10279](https://arxiv.org/abs/2107.10279) | Le circuit U(1) du dépôt et la transition de sharpening, distincte de la MIPT d'intrication. C'est le seul cas où le QELM voit une transition (charge dans les features). |
| P1 | F. Barratt, U. Agrawal, S. Gopalakrishnan, D. A. Huse, R. Vasseur, A. C. Potter, « Field theory of charge sharpening in symmetric monitored quantum circuits », PRL 129, 120604 (2022), [arXiv:2111.09336](https://arxiv.org/abs/2111.09336) | La théorie du sharpening (`charge_variance.py`). |
| P1 | F. Barratt, U. Agrawal, A. C. Potter, S. Gopalakrishnan, R. Vasseur, « Transitions in the learnability of global charges from local measurements », PRL 129, 200602 (2022), [arXiv:2206.12429](https://arxiv.org/abs/2206.12429) | L'apprenabilité de la charge depuis `M`, le décodeur SEP (processus d'exclusion) et `p_# ≈ 0,09`, reproduits dans les notebooks 01 et 02. Le cadre exact de la question « un décodeur de `M` voit-il la transition ? ». |
| P1 | U. Agrawal, J. Lopez-Piqueres, R. Vasseur, S. Gopalakrishnan, A. C. Potter, « Observing quantum measurement collapse as a learnability phase transition », PRX 14, 041012 (2024), [arXiv:2311.00058](https://arxiv.org/abs/2311.00058) | Hiérarchie des décodeurs (postBQP, stat-mech, réseau de neurones), Fig. 7 reproduite (notebook 03), point de départ de #307. |
| P2 | M. Ippoliti, V. Khemani, « Learnability transitions in monitored quantum dynamics via eavesdropper's classical shadows », PRX Quantum 5, 020304 (2024), [arXiv:2307.15011](https://arxiv.org/abs/2307.15011) | Un observateur qui apprend à partir de `M` et d'ombres classiques : très proche de la lecture `<P_k>` + `M` du QELM. |
| P2 | S. W. P. Kim, C. von Keyserlingk, A. Lamacraft, « Measurement-induced phase transitions in quantum inference problems and quantum hidden Markov models », [arXiv:2504.08888](https://arxiv.org/abs/2504.08888) (2025) | L'inférence à partir des résultats de mesure comme problème de XOR planté : la vraie source de l'argument sur les parités (T5). |
| P3 | J. Feldmeier, P. Sala, G. De Tomasi, F. Pollmann, M. Knap, « Anomalous diffusion in dipole- and higher-moment conserving systems », PRL 125, 245303 (2020), [arXiv:2004.00635](https://arxiv.org/abs/2004.00635) | Cité par `symmetric.py` pour les circuits à moments conservés. |

### 2.3 Décodeurs, post-sélection et observabilité

| P | Référence | Pourquoi la lire |
|---|---|---|
| P1 | H. Dehghani, A. Lavasani, M. Hafezi, M. J. Gullans, « Neural-network decoders for measurement induced phase transitions », Nat. Commun. 14, 2918 (2023), [arXiv:2204.10904](https://arxiv.org/abs/2204.10904) | Le seul décodeur appris qui retrouve la MIPT d'intrication depuis `M` (CNN, circuit fixe). Comparaison directe avec le ridge du QELM (`dehghani_decoder`, `cnn_decoder`). |
| P1 | Y. Li, Y. Zou, P. Glorioso, E. Altman, M. P. A. Fisher, « Cross entropy benchmark for measurement-induced phase transitions », PRL 130, 220404 (2023), [arXiv:2209.00609](https://arxiv.org/abs/2209.00609) | Une sonde sans post-sélection qui utilise une simulation classique, comme le décodeur « Bayes sur `M` » du projet ; elle dit ce qu'un observateur qui connaît le circuit peut voir. |
| P2 | S. J. Garratt, E. Altman, « Probing post-measurement entanglement without post-selection », PRX Quantum 5, 030311 (2024), [arXiv:2305.20092](https://arxiv.org/abs/2305.20092) | Le problème de la post-sélection posé proprement ; borne ce que peut faire une lecture classique. |
| P2 | M. McGinley, « Postselection-free learning of measurement-induced quantum dynamics », PRX Quantum 5, 020347 (2024), [arXiv:2310.04156](https://arxiv.org/abs/2310.04156) | Apprendre des propriétés des états post-mesure par optimisation, avec des bornes des deux côtés. |
| P2 | W. Hou, S. J. Garratt, N. M. Eassa, E. Rosenberg, P. Roushan, Y.-Z. You, E. Altman, « Machine learning the effects of many quantum measurements », [arXiv:2509.08890](https://arxiv.org/abs/2509.08890) (2025) | Réseaux de neurones entraînés sans supervision sur des résultats de mesure d'un processeur supraconducteur ; la capacité d'apprendre a une transition qui coïncide avec une MIPT. Le travail récent le plus proche de la question du projet. |
| P2 | H. Yu, J. Hu, S.-X. Zhang, « Post-selection-free decoding of measurement-induced area-law phases via neural networks », [arXiv:2604.03550](https://arxiv.org/abs/2604.03550) (2026) | Décodeur CNN + attention sur les seuls records ; montre ce qu'un décodeur non linéaire ajoute à une lecture linéaire. |
| P2 | M. Ippoliti, V. Khemani, « Postselection-free entanglement dynamics via spacetime duality », PRL 126, 060501 (2021), [arXiv:2010.15840](https://arxiv.org/abs/2010.15840) | Une autre façon de contourner la post-sélection, utilisée dans l'expérience Google. |
| P3 | S. J. Garratt, Z. Weinstein, E. Altman, « Measurements conspire nonlocally to restructure critical quantum states », PRX 13, 021026 (2023), [arXiv:2207.09476](https://arxiv.org/abs/2207.09476) | Effet non local des mesures sur un état critique. |
| P3 | M. J. Gullans, S. Krastanov, D. A. Huse, L. Jiang, S. T. Flammia, « Quantum coding with low-depth random circuits », PRX 11, 031066 (2021), [arXiv:2010.09775](https://arxiv.org/abs/2010.09775) | Profondeur nécessaire pour qu'un circuit aléatoire encode bien l'information ; utile pour discuter `T` contre `N`. |

### 2.4 Expériences sur processeurs quantiques

| P | Référence | Pourquoi la lire |
|---|---|---|
| P2 | C. Noel et al. (11 auteurs), « Observation of measurement-induced quantum phases in a trapped-ion quantum computer », Nat. Phys. (2022), [arXiv:2106.05881](https://arxiv.org/abs/2106.05881), DOI 10.1038/s41567-022-01619-7 | Première observation, avec le qubit de référence de Gullans & Huse. |
| P2 | J. C. Hoke, M. Ippoliti, E. Rosenberg et al. (Google Quantum AI), « Measurement-induced entanglement and teleportation on a noisy quantum processor », Nature (2023), [arXiv:2303.04792](https://arxiv.org/abs/2303.04792), DOI 10.1038/s41586-023-06505-7 | Décodage classique des records sur un vrai processeur, avec bruit. |
| P2 | J. M. Koh, S.-N. Sun, M. Motta, A. J. Minnich, « Experimental realization of a measurement-induced entanglement phase transition on a superconducting quantum processor », Nat. Phys. (2023), [arXiv:2203.04338](https://arxiv.org/abs/2203.04338), DOI 10.1038/s41567-023-02076-6 | Mesures en milieu de circuit, comme le QELM du projet. |
| P3 | X. Feng, J. Côté, S. Kourtis, B. Skinner, « Postselection-free experimental observation of the measurement-induced phase transition in circuits with universal gates », [arXiv:2502.01735](https://arxiv.org/abs/2502.01735) (2025) | Arbres Haar avec mesures faibles, décodage classique linéaire en `N`, sur Quantinuum H1-1. |

### 2.5 QELM et calcul par réservoir quantique

| P | Référence | Pourquoi la lire |
|---|---|---|
| P1 | L. Innocenti, S. Lorenzo, I. Palmisano, A. Ferraro, M. Paternostro, G. M. Palma, « Potential and limitations of quantum extreme learning machines », Commun. Phys. 6, 118 (2023), [arXiv:2210.00780](https://arxiv.org/abs/2210.00780) | Le QELM comme POVM effectif : seules les fonctions linéaires de `ρ` dans l'espace engendré par les effets sont apprenables. C'est la base du rang de l'espace des effets (#313, #317, #319). |
| P1 | W. Xiong, G. Facelli, M. Sahebi, O. Agnel, T. Chotibut, S. Thanasilp, Z. Holmes, « On fundamental aspects of quantum extreme learning machines », [arXiv:2312.15124](https://arxiv.org/abs/2312.15124), DOI 10.1007/s42484-025-00239-7 | Analyse de Fourier du QELM et quatre sources de concentration exponentielle des observables ; cadre du poids de Fourier `W(m)` (#325, #327) et du « Xiong et al. » de `operator_cones`. |
| P1 | P. Mujal, R. Martínez-Peña, J. Nokkala, J. García-Beni, G. L. Giorgi, M. C. Soriano, R. Zambrini, « Opportunities in quantum reservoir computing and extreme learning machines », Adv. Quantum Technol. (2021), [arXiv:2102.11831](https://arxiv.org/abs/2102.11831), DOI 10.1002/qute.202100027 | Revue du domaine QRC/QELM, pour situer le projet. |
| P2 | K. Fujii, K. Nakajima, « Harnessing disordered-ensemble quantum dynamics for machine learning », Phys. Rev. Applied 8, 024030 (2017), [arXiv:1602.08159](https://arxiv.org/abs/1602.08159) | Article fondateur du réservoir quantique (titre arXiv : « Harnessing disordered quantum dynamics for machine learning »). |
| P2 | S. Ghosh, A. Opala, M. Matuszewski, T. Paterek, T. C. H. Liew, « Quantum reservoir processing », npj Quantum Inf. (2019), [arXiv:1811.10335](https://arxiv.org/abs/1811.10335), DOI 10.1038/s41534-019-0149-8 | Réservoir qui traite des états quantiques en entrée, comme le qubit `R` du projet. |
| P2 | A. De Lorenzis, M. P. Casado, M. P. Estarellas, N. Lo Gullo, T. Lux, F. Plastina, A. Riera, J. Settino, « Harnessing quantum extreme learning machines for image classification », Phys. Rev. Applied 23, 044024 (2025), [arXiv:2409.00998](https://arxiv.org/abs/2409.00998) | Protocole de `double_descent` et encodages de #279. |
| P2 | R. Martínez-Peña, G. L. Giorgi, J. Nokkala, M. C. Soriano, R. Zambrini, « Dynamical phase transitions in quantum reservoir computing », PRL 127, 100502 (2021), [arXiv:2103.05348](https://arxiv.org/abs/2103.05348) | Performance d'un réservoir à travers une transition (MBL/ergodique) : le pendant unitaire de la question du projet. |
| P2 | P. Mujal, R. Martínez-Peña, G. L. Giorgi, M. C. Soriano, R. Zambrini, « Time series quantum reservoir computing with weak and projective measurements », npj Quantum Inf. (2023), [arXiv:2205.06809](https://arxiv.org/abs/2205.06809), DOI 10.1038/s41534-023-00682-z | Mesures faibles et projectives dans un réservoir, comme `training/measurements/weak.yaml`. |
| P2 | K. Kobayashi, Y. Motome, « Edge of many-body quantum chaos in quantum reservoir computing », [arXiv:2506.17547](https://arxiv.org/abs/2506.17547), DOI 10.1103/j2qj-vwcl | SYK étendu, performance maximale au bord du chaos et temps de Thouless : la référence de l'issue #86. |
| P2 | K. Kobayashi, Y. Motome, « Quantum reservoir probing of quantum phase transitions », Nat. Commun. (2025), [arXiv:2402.07097](https://arxiv.org/abs/2402.07097) | Un réservoir comme sonde de transition (état fondamental, dynamique unitaire, pas de circuit mesuré, voir #298). |
| P3 | K. Kobayashi, Y. Motome, « Quantum reservoir probing: an inverse paradigm of quantum reservoir computing for exploring quantum many-body physics », SciPost Phys. 18, 198 (2025), [arXiv:2308.00898](https://arxiv.org/abs/2308.00898) | Le même programme appliqué au scrambling. |
| P2 | F. Hu, G. Angelatos, S. A. Khan, M. Vives, E. Türeci, L. Bello, G. E. Rowlands, G. J. Ribeill, H. E. Türeci, « Tackling sampling noise in physical systems for machine learning applications: fundamental limits and eigentasks », PRX 13, 041020 (2023), [arXiv:2307.16083](https://arxiv.org/abs/2307.16083) | Capacité expressive à nombre de tirs fini et « eigentasks » : la version à tirs finis du rang de l'espace des effets. |
| P3 | F. Hu, S. A. Khan, N. T. Bronn, G. Angelatos, G. E. Rowlands, G. J. Ribeill, H. E. Türeci, « Overcoming the coherence time barrier in quantum machine learning on temporal data », Nat. Commun. (2024), [arXiv:2312.16165](https://arxiv.org/abs/2312.16165) | Réservoir à mesures en milieu de circuit, testé sur processeur supraconducteur, pour des données temporelles plus longues que le temps de cohérence. |
| P3 | B. Zhang, F. Hu, R. Mo, T. Chen, H. E. Türeci, Q. Zhuang, « Scaling laws of quantum information lifetime in monitored quantum dynamics », [arXiv:2506.22755](https://arxiv.org/abs/2506.22755) (2025) | Avec les résultats enregistrés, l'information survit un temps exponentiel en `N` ; lien direct entre records et mémoire d'un réservoir. |
| P3 | T. Yasuda et al. (11 auteurs), « Quantum reservoir computing with repeated measurements on superconducting devices », [arXiv:2310.06706](https://arxiv.org/abs/2310.06706) (2023) | Réservoir avec mesures répétées sur du matériel. |
| P3 | A. Suprano et al. (13 auteurs), « Experimental property-reconstruction in a photonic quantum extreme learning machine », PRL 132, 160802 (2024), [arXiv:2308.04543](https://arxiv.org/abs/2308.04543) | Mise en œuvre expérimentale du QELM d'Innocenti et al. |
| P3 | A. Sakurai, M. P. Estarellas, W. J. Munro, K. Nemoto, « Quantum reservoir computation utilising scale-free networks » (titre arXiv), Phys. Rev. Applied 17, 064044 (2022), [arXiv:2108.12131](https://arxiv.org/abs/2108.12131) | QELM sur un réseau d'interactions fixe sans invariance par translation. |
| P3 | A. H. Abbas, I. S. Maksymov, « Reservoir computing using measurement-controlled quantum dynamics », [arXiv:2403.01024](https://arxiv.org/abs/2403.01024) (2024) | Réservoir piloté par la mesure (atome en cavité). Peu lié aux circuits, à garder pour une revue exhaustive. |

### 2.6 Expressivité, encodage et analyse de Fourier

| P | Référence | Pourquoi la lire |
|---|---|---|
| P1 | M. Schuld, R. Sweke, J. J. Meyer, « The effect of data encoding on the expressive power of variational quantum machine learning models », PRA 103, 032430 (2021), [arXiv:2008.08605](https://arxiv.org/abs/2008.08605) | Un modèle quantique est une série de Fourier dont le spectre est fixé par l'encodage : base du cadre de Fourier (#309) et des encodages angle/dense. |
| P2 | A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre, « Data re-uploading for a universal quantum classifier », Quantum 4, 226 (2020), [arXiv:1907.02085](https://arxiv.org/abs/1907.02085) | Pourquoi une seule injection limite le degré des fonctions apprenables (pureté avec deux copies, #193). |
| P2 | S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes, « Exponential concentration in quantum kernel methods », [arXiv:2208.11060](https://arxiv.org/abs/2208.11060) (2022) | Concentration exponentielle des noyaux : même mécanisme que dans Xiong et al. Journal non vérifié ici. |
| P2 | H.-Y. Huang, R. Kueng, J. Preskill, « Predicting many properties of a quantum system from very few measurements », Nat. Phys. (2020), [arXiv:2002.08953](https://arxiv.org/abs/2002.08953), DOI 10.1038/s41567-020-0932-7 | Ombres classiques : coût en tirs d'une lecture des `<P_k>` et `<P_k P_l>`. |
| P2 | X. Turkeshi, « Measurement-induced criticality as a data-structure transition », PRB 106, 144313 (2022), [arXiv:2101.06245](https://arxiv.org/abs/2101.06245) | La MIPT vue par la dimension intrinsèque des données de mesure : base de l'issue #272. |
| P3 | E. Facco, M. d'Errico, A. Rodriguez, A. Laio, « Estimating the intrinsic dimension of datasets by a minimal neighborhood information », Sci. Rep. (2017), [arXiv:1803.06992](https://arxiv.org/abs/1803.06992), DOI 10.1038/s41598-017-11873-y | L'estimateur TWO-NN de #272. |
| P3 | M. Belkin, D. Hsu, S. Ma, S. Mandal, « Reconciling modern machine learning practice and the bias-variance trade-off », PNAS (2019), [arXiv:1812.11118](https://arxiv.org/abs/1812.11118), DOI 10.1073/pnas.1903070116 | Double descente, pour lire `experiments/double_descent`. |
| P3 | O. Ledoit, M. Wolf, « A well-conditioned estimator for large-dimensional covariance matrices », J. Multivariate Anal. 88, 365 (2004) | Estimateur de covariance de `pnt_decay`. |

### 2.7 Croissance d'opérateurs, chaos et outils numériques

| P | Référence | Pourquoi la lire |
|---|---|---|
| P2 | A. Nahum, J. Ruhman, S. Vijay, J. Haah, « Quantum entanglement growth under random unitary dynamics », PRX 7, 031016 (2017), [arXiv:1608.06950](https://arxiv.org/abs/1608.06950) | Croissance de l'intrication et coupe minimale dans les circuits aléatoires, base de l'image de Skinner et al. |
| P2 | A. Nahum, S. Vijay, J. Haah, « Operator spreading in random unitary circuits », PRX 8, 021014 (2018), [arXiv:1705.08975](https://arxiv.org/abs/1705.08975) | La chaîne de Markov sur les chaînes de Pauli de `operator_cones.py`. |
| P2 | C. von Keyserlingk, T. Rakovszky, F. Pollmann, S. L. Sondhi, « Operator hydrodynamics, OTOCs, and entanglement growth in systems without conservation laws », PRX 8, 021013 (2018), [arXiv:1705.08910](https://arxiv.org/abs/1705.08910) | OTOC et vitesse papillon (`v_B ≈ 0,6` reproduit). |
| P3 | D. E. Parker, X. Cao, A. Avdoshkin, T. Scaffidi, E. Altman, « A universal operator growth hypothesis », PRX 9, 041017 (2019), [arXiv:1812.08657](https://arxiv.org/abs/1812.08657) | Coefficients de Lanczos `b_n` de `operator_cones`. |
| P3 | H. Kim, D. A. Huse, « Ballistic spreading of entanglement in a diffusive nonintegrable system », PRL 111, 127205 (2013), [arXiv:1306.4306](https://arxiv.org/abs/1306.4306) | Point chaotique `(h_x, h_z) = (1,05 ; 0,5)` du TFIM continu. |
| P3 | Y. Y. Atas, E. Bogomolny, O. Giraud, G. Roux, « The distribution of the ratio of consecutive level spacings in random matrix ensembles », PRL 110, 084101 (2013), [arXiv:1212.5611](https://arxiv.org/abs/1212.5611) | `<r>` de référence de `chaos_locality.py`. |
| P2 | S. Aaronson, D. Gottesman, « Improved simulation of stabilizer circuits », PRA 70, 052328 (2004), [arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196) | Le tableau stabilisateur derrière les simulations Clifford à grand `N`. |
| P3 | D. Fattal, T. S. Cubitt, Y. Yamamoto, S. Bravyi, I. L. Chuang, « Entanglement in the stabilizer formalism », [arXiv:quant-ph/0406168](https://arxiv.org/abs/quant-ph/0406168) (2004) | Entropie d'un état stabilisateur, utilisée dans `stabilizer.py` et pour `1 + W = 2^k` (#327). |
| P3 | J. Houdayer, A. K. Hartmann, « Low-temperature behavior of two-dimensional Gaussian Ising spin glasses », PRB 70, 014418 (2004), [arXiv:cond-mat/0402036](https://arxiv.org/abs/cond-mat/0402036) | Fonction de qualité du collapse de `finite_size_scaling.py`. |
| P3 | M. Szyniszewski, A. Romito, H. Schomerus, « Entanglement transition from variable-strength weak measurements », PRB 100, 064204 (2019), [arXiv:1903.05452](https://arxiv.org/abs/1903.05452) | MIPT avec mesures faibles, à comparer avec `weak.yaml`. |
| P3 | X. Cao, A. Tilloy, A. De Luca, « Entanglement in a fermion chain under continuous monitoring », SciPost Phys. 7, 024 (2019), [arXiv:1804.04638](https://arxiv.org/abs/1804.04638) | Surveillance continue, pendant des mesures poissoniennes de `QELM_S`. |

## 3. Parcours de lecture conseillé

1. **Une semaine pour la base.** Fisher et al. (revue), Skinner et al., Gullans
   & Huse (les deux), Zabalo et al. On sait alors définir la MIPT, la sonde du
   qubit de référence, `p_c` et `ν` pour Haar et Clifford.
2. **Le cœur du projet.** Barratt et al. (apprenabilité), Agrawal et al. 2024,
   Dehghani et al., Li et al. (cross entropy). C'est la littérature sur « peut-on
   lire la transition dans `M` ? ». Puis Hou et al. 2025, le travail récent le
   plus proche.
3. **Le côté QELM.** Innocenti et al., Xiong et al., Schuld et al., Hu et al.
   (eigentasks). Ils donnent le cadre du rang de l'espace des effets et du poids
   de Fourier.
4. **Selon l'expérience en cours.** Sections 2.2 (U(1)), 2.4 (matériel), 2.7
   (opérateurs, outils).

## 4. Ce qui reste à faire

- Compléter les références incomplètes du tableau 1 dans le code et les
  README (De Lorenzis et al., Xiong et al., Li, Chen & Fisher PRB 100).
- Relire le contenu des articles pour les valeurs numériques citées de mémoire
  (par exemple `p_c` et `ν` de Skinner et al., Table I, dans `conf/QELM_S.yaml`)
  et pour la conjecture C4.
- Le journal de Thanasilp et al. et de Yasuda et al. n'a pas été vérifié.
