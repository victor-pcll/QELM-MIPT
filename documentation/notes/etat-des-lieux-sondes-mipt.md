# État des lieux : les sondes de la MIPT dans le QELM

*Synthèse au 4 novembre 2026 des PR #255 à #280 (et des expériences plus anciennes
qu'elles réutilisent). Les nombres viennent des descriptions de PR et des README
d'expérience ; chaque ligne renvoie à sa source.*

## Cadre

La MIPT est une propriété de la **dynamique** : un circuit brickwork (Haar U(4),
Clifford ou U(1)) mesuré au taux `p`. Pour Haar, `p_c ≈ 0.18` (purification, #258) ;
pour Clifford, `p_c = 0.160` (trois estimateurs d'accord jusqu'à N = 128, #258).
Les observables ne *définissent* pas la transition : ce sont des **sondes**. Une
sonde « voit » la MIPT si ses courbes à différents `N` se croisent près de `p_c`
(ou se rassemblent par une loi d'échelle autour de `p_c`). La question du projet
est de savoir quelles sondes, lues par quel décodeur, gardent cette signature.

## Tableau des sondes

| Sonde | Décodeur | Testé sur | Croisement près de `p_c` ? | Dépendance en N | Verdict |
| --- | --- | --- | --- | --- | --- |
| Intrication `S_half`, `I3` (#258, #239, #251) | aucun (état exact) | Clifford N ≤ 128, Haar N ≤ 16, Floquet N ≤ 16 | Oui à grand N (Clifford 0.160). Haar : `S_half` donne 0.256, `I3` dérive de 0.19 à 0.27 | Convergence lente, biais de taille finie fort | Sonde valide mais coûteuse ; inutilisable seule aux tailles Haar accessibles |
| Purification `S_R` du qubit de référence (#255, #258, #274) | aucun (état exact) | Haar, Clifford ; tâche QELM avec T = 10, N, 2N | Oui : 0.184 / 0.178 (#258, N = 8–16). Dans la tâche QELM à T = 2N : 0.23, 0.22, 0.27 (#274). T = 10 : aucun croisement | Croisement stable dans #258 ; dérive dans #274 (10 circuits seulement) | **Meilleure sonde de référence** ; il faut T ∝ N |
| Apprenabilité de la charge (#113, #275) | **oracle bayésien** (gates inconnues ou connues) | U(1), Haar, N = 6–10 | Pas de marche nette. Binder de `H(P)` : 0.12 → 0.14 → 0.18. Signe de la dérive de `I(Q;M)` change entre 0.15 et 0.175 ; vers 0.10 pour l'observateur qui connaît le circuit | Signature dans la **dérive** avec N, pas dans les courbes elles-mêmes | Voit la transition, faiblement ; l'observateur qui connaît le circuit n'est pas encore résolu (il faut N = 10, 12) |
| Charge depuis l'état final (#259, #261, #274, #276) | **QELM linéaire** (ridge, logistique, quadratique, réservoir de Haar…) | Haar, Clifford, N = 6–12, T = 5 à 2N | **Non.** Tous les lecteurs tombent au hasard dès `p ≈ 0.2`, bien avant la purification | Toutes les courbes se rassemblent en `p·N·T` : `p_0 N T = 4.0 ± 0.3` | **Négatif** : le QELM compte les mesures, il ne voit pas la phase |
| Charge depuis le record M (#259) | ridge, k-NN, **MLP**, gradient boosting | Haar, N = 8 | Non | — | Négatif : au hasard à tout `p`, même là où R est purifié |
| Paire de Bell locale depuis M (#278, ouverte) | ridge sur M et sur `|M|`, XOR, paires, **MLP** ; oracle bayésien | Haar, N = 6, 8, 4 encodages | Non étudié (deux tailles). Ridge : 0.52 → 0.64 ; oracle : 0.56 → 0.72 (N = 8) | Inconnue ; indépendant de T (seuls les deux premiers rounds comptent) | Aucune non-linéarité n'aide ; l'écart à l'oracle est un écart d'échantillons et de connaissance du circuit |
| Record M du TFIM (#234, #253) | Bayes-Ridge | TFIM tout-à-tout, N = 5–11 | Non : collapse sur `Φ(d'/2)` sans paramètre | Limite lisse `Φ(√(p/(1−p)))` | Négatif : le TFIM tout-à-tout se comporte comme le modèle complètement brouillé, pas de MIPT visible |
| Vecteur de Bloch et pureté de R depuis M (#280, ouverte) | ridge (QELM sur M, sur l'état, les deux) ; oracle | Haar, **N = 8 seulement** | Signature molle : minimum de `R²` (≈ 0.21) entre `p = 0.2` et 0.3 ; pic de l'erreur sur la pureté (0.113) à `p = 0.175` | Non testée (#282 la prévoit) | Pas de signature nette ; la pureté est au hasard en linéaire (paire en M), `R² = 0.56` avec `[M, |M|]` |
| Prédiction de `S_A` (#232) | ridge sur `[M, ⟨Z⟩]`, avec ou sans carrés | TFIM, Floquet, Haar, U(1) | Sans objet | — | Le `R²` vient presque entièrement de `p` et de la position des mesures ; linéaire impossible (sonde paire) |
| Sharpening de charge (#137) | aucun (état exact) | U(1), L = 6 | Crossover lisse vers `γ ≈ 0.6` | Une seule taille | Reproduit Agrawal et al. ; **jamais testé comme sonde lue par un décodeur** |

## Points ouverts

1. **Le croisement de `S_R` dérive avec N dans la tâche QELM** (0.22–0.27 à T = 2N,
   #274) alors qu'il est stable à 0.18 dans le benchmark de purification (#258). La
   différence de protocole (10 circuits fixes, encodage de charge, 2000 trajectoires)
   n'est pas encore expliquée. Les runs de taille finie annoncés (#278 au cluster,
   #282 pour Bloch et pureté à N = 6, 8, 10) **sont en attente** ; aucun résultat
   n'était disponible à la date de cette note.
2. **La précision du QELM ne dépend que de `p·N·T`.** Un lecteur fixe de l'état final
   compte combien de mesures ont eu lieu ; il perd l'information après environ quatre
   mesures, quelle que soit la phase. Tant que le décodeur est entraîné ainsi, il ne
   peut pas montrer de croisement.
3. **La régression de Bloch et de pureté ne montre pas de signature nette** à N = 8 :
   des minima et des pics proches de `p_c`, mais sans balayage en N pour savoir s'ils
   se resserrent.
4. **Un lecteur non linéaire retrouve-t-il le croisement de l'oracle ?** Les MLP
   testés (#259, #278) font au mieux comme le ridge. Mais l'oracle lui-même n'a pas
   encore de croisement propre pour la tâche locale (N = 6, 8 seulement), donc la
   question n'est pas encore bien posée.
5. **La sonde de sharpening (variance d'une quantité trajectoire par trajectoire)
   n'a pas été testée** avec un décodeur. #137 ne l'a mesurée que sur l'état exact,
   à une seule taille.

## Direction recommandée (trois étapes)

1. **Fixer la référence.** Sur la tâche locale (paire de Bell, Haar, T = 2N, circuits
   fixes), mesurer au cluster pour N = 6 à 12 et sur les **mêmes** circuits : `S_R`,
   l'information `I(Y;M)` et la précision de l'oracle qui connaît le circuit. But : un
   croisement de l'oracle comparable à celui de `S_R`, et l'explication de l'écart
   entre #258 et #274.
2. **Mesurer l'écart QELM–oracle en fonction de N**, de part et d'autre de `p_c`, avec
   un nombre de trajectoires qui suit N (`PER_P` ≥ 16 000, #276) et un lecteur non
   linéaire (MLP) à côté du ridge. Si l'écart reste constant, le QELM ne voit pas la
   transition ; s'il s'ouvre en phase volumique et se ferme en phase d'aire, c'est
   l'écart lui-même qui est la sonde.
3. **Tester la sonde de variance.** À partir des données de #280, tracer la variance
   sur les trajectoires de la pureté de R (oracle) et de la sortie du QELM, puis
   suivre la position et la hauteur de son pic en N. C'est l'analogue décodé du
   sharpening de #137, et c'est peu coûteux : les tableaux existent déjà.
