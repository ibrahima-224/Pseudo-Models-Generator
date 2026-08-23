# CAHIER DE PLAN DE DÉVELOPPEMENT
## SPRINTS 6 À 11

### Pseudo-Models Generator — PMG

**Version du document :** 1.0  
**Période couverte :** Sprint 6 → Sprint 11  
**Responsable du développement :** Ibrahima-224  
**Équipe de développement :** 1 développeur  
**Langage :** Rust  
**Édition Rust cible :** 2021  
**Licence du projet :** GPL-3.0  
**Statut :** Plan de développement approuvé  
**Dépôt :** Pseudo-Models-Generator

---

# TABLE DES MATIÈRES

1. Vision des Sprints 6 à 11
2. Principes d'organisation
3. Sprint 6 — Moteur de génération tensorielle
4. Sprint 7 — Injection des distributions réalistes
5. Sprint 8 — Injection des structures et corrélations
6. Sprint 9 — Injection des super-poids et anomalies critiques
7. Sprint 10 — Génération complète des pseudo-modèles
8. Sprint 11 — Validation, comparaison et stabilisation
9. Matrice des dépendances
10. Stratégie de tests
11. Stratégie de performance
12. Stratégie de validation scientifique
13. Critères de sortie
14. Références techniques

---

# 1. VISION DES SPRINTS 6 À 11

Les Sprints 0 à 5 ont pour fonction de construire les fondations du logiciel.

Les Sprints 6 à 11 constituent le **cœur scientifique et génératif de PMG**.

La progression est volontairement séquentielle :

```text
SPRINT 6
Moteur de génération tensorielle
        │
        ▼
SPRINT 7
Distributions statistiques réalistes
        │
        ▼
SPRINT 8
Corrélations et structures bas-rang
        │
        ▼
SPRINT 9
Super-poids et anomalies critiques
        │
        ▼
SPRINT 10
Assemblage du pseudo-modèle complet
        │
        ▼
SPRINT 11
Validation + comparaison + stabilisation
```

L'objectif n'est donc pas simplement de produire des fichiers `.safetensors`.

PMG doit produire des **pseudo-modèles structurellement et statistiquement plausibles**, tout en permettant de contrôler explicitement les caractéristiques injectées.

---

# 2. PRINCIPES D'ORGANISATION

## 2.1 Une seule responsabilité par Sprint

Chaque Sprint possède une responsabilité scientifique ou logicielle principale.

| Sprint | Responsabilité |
|---|---|
| 6 | Générer les valeurs de base des tenseurs |
| 7 | Reproduire des distributions statistiques réalistes |
| 8 | Reproduire les structures et corrélations |
| 9 | Reproduire les super-poids et anomalies critiques |
| 10 | Assembler et écrire un pseudo-modèle complet |
| 11 | Valider et comparer le résultat |

Cette séparation est importante.

Par exemple, le Sprint 7 ne doit pas décider comment les super-poids sont injectés. Cette responsabilité appartient au Sprint 9.

---

# 2.2 Une seule responsabilité par étape

Chaque étape possède :

- un seul objectif ;
- un fichier principal ;
- une responsabilité précise ;
- des tests associés ;
- un critère d'acceptation.

Exemple :

```text
Étape 7.3
Responsabilité :
implémenter la distribution Student-t.

Fichier principal :
crates/pmg-math/src/distributions/student_t.rs

Ne fait PAS :
- l'injection des outliers ;
- l'écriture Safetensors ;
- la génération du CLI.
```

Cette règle évite que le code devienne difficile à maintenir.

---

# 2.3 Organisation pour un développeur unique

Puisque Ibrahima-224 est seul à développer PMG, il ne faut pas organiser le travail comme une équipe parallèle.

Le modèle de travail est :

```text
Étape
  ↓
Implémentation
  ↓
Tests
  ↓
Documentation
  ↓
Validation
  ↓
Commit
  ↓
Étape suivante
```

Une étape ne doit être considérée comme terminée que lorsque les tests et la documentation sont également terminés.

---

# 2.4 Règle des 500 lignes

Aucun fichier Rust ne doit dépasser 500 lignes de code.

Si un module devient trop important :

```text
generator.rs
    │
    ├── base.rs
    ├── distribution.rs
    ├── structure.rs
    └── anomalies.rs
```

Le développeur doit privilégier les responsabilités cohérentes plutôt que simplement réduire artificiellement le nombre de lignes.

---

# 2.5 Déterminisme

Le moteur PMG doit être déterministe lorsqu'une seed est fournie.

Formellement :

\[
G(C,S)=M
\]

où :

- \(C\) = configuration ;
- \(S\) = seed ;
- \(G\) = générateur ;
- \(M\) = pseudo-modèle.

Ainsi :

\[
G(C,S)=G(C,S)
\]

doit produire exactement le même résultat dans les mêmes conditions.

Deux seeds différentes doivent normalement produire deux réalisations différentes :

\[
S_1 \neq S_2
\Rightarrow
G(C,S_1)\neq G(C,S_2)
\]

---

# SPRINT 6 — MOTEUR DE GÉNÉRATION TENSORIELLE

## 6.1 Informations générales

**Responsabilité unique :**

> Construire le moteur capable de générer les valeurs numériques de base des tenseurs à partir de leurs métadonnées.

**Durée indicative :** 2 à 3 semaines.

**Responsable :** Ibrahima-224.

**Dépendances :**

- Sprint 1 : types fondamentaux ;
- Sprint 2 : lecture des métadonnées ;
- Sprint 3 : mathématiques de base ;
- Sprint 4 : I/O ;
- Sprint 5 : orchestration préparatoire.

---

## 6.2 Objectifs

Le Sprint 6 doit permettre de :

- générer des tenseurs déterministes ;
- gérer différentes dimensions ;
- gérer les différents DType supportés ;
- travailler par blocs ;
- éviter de charger inutilement un modèle complet en RAM ;
- préparer l'intégration des distributions ;
- préparer l'injection structurelle ;
- préparer l'injection des anomalies.

---

## 6.3 Architecture

```text
TensorMetadata
      │
      ▼
GenerationPlan
      │
      ▼
TensorGenerator
      │
      ├── BaseGenerator
      │
      ├── RNG
      │
      └── ChunkGenerator
      │
      ▼
TensorChunk
      │
      ▼
pmg-io
```

---

# Étape 6.1 — Plan de génération

### Responsabilité

Transformer les métadonnées d'un tenseur en plan de génération.

### Fichier

```text
crates/pmg-core/src/generation_plan.rs
```

### Objectif

Créer une structure décrivant ce qui doit être généré sans générer immédiatement les données.

Exemple conceptuel :

```rust
pub struct GenerationPlan {
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub seed: u64,
    pub chunk_elements: usize,
}
```

### Attentes

Le plan doit être :

- sérialisable ;
- inspectable ;
- déterministe ;
- indépendant de l'écriture disque.

### Point critique

Ne pas mélanger :

```text
planification
```

et :

```text
génération physique
```

---

# Étape 6.2 — Générateur de base

### Fichier

```text
crates/pmg-math/src/generator.rs
```

### Responsabilité

Produire des valeurs pseudo-aléatoires de base.

Le modèle initial peut être :

\[
x_i \sim \mathcal{N}(\mu,\sigma^2)
\]

avec :

\[
x_i=\mu+\sigma z_i
\]

où :

\[
z_i\sim\mathcal{N}(0,1)
\]

### Exemple

Pour :

```text
mean = 0
std = 0.02
```

on génère :

```text
0.013
-0.004
0.021
...
```

### Points forts

- simple ;
- rapide ;
- déterministe ;
- excellente base pour les transformations ultérieures.

### Point faible

Une simple loi normale n'est pas suffisante pour reproduire la complexité statistique d'un modèle réel.

C'est précisément pourquoi le Sprint 7 existe.

---

# Étape 6.3 — Gestion déterministe du RNG

### Fichier

```text
crates/pmg-math/src/rng.rs
```

### Responsabilité

Centraliser la génération pseudo-aléatoire.

Le développeur doit éviter de créer plusieurs RNG indépendants sans stratégie de seed.

Une stratégie possible :

\[
S_{tensor}=H(S_{global}, tensor\_id)
\]

où \(H\) est une fonction de dérivation déterministe.

### Exemple

```text
Seed globale : 42

tensor 0 → seed A
tensor 1 → seed B
tensor 2 → seed C
```

Ainsi, l'ajout d'un tenseur ne doit pas nécessairement modifier toutes les autres générations.

### Test

Deux générations :

```text
seed = 42
```

doivent produire exactement les mêmes valeurs.

---

# Étape 6.4 — Génération par chunks

### Fichier

```text
crates/pmg-math/src/chunk_generator.rs
```

### Responsabilité

Générer un tenseur par morceaux.

Si un tenseur possède :

\[
N=10^9
\]

éléments, il ne faut pas nécessairement créer un :

```rust
Vec<f32>
```

contenant tout le tenseur.

On utilise :

\[
N = N_1 + N_2 + \dots + N_k
\]

où chaque \(N_i\) correspond à un chunk.

### Exemple

```text
Tenseur :
10 000 000 éléments

Chunk :
1 000 000 éléments

Nombre de chunks :
10
```

### Point critique

La génération par chunks doit être mathématiquement cohérente avec une génération complète.

---

# Étape 6.5 — Tests du générateur

### Fichier

```text
crates/pmg-math/tests/generator_tests.rs
```

Tests :

- déterminisme ;
- taille correcte ;
- seed différente ;
- génération vide ;
- dimensions invalides ;
- chunk boundaries.

### Critère de sortie

Le moteur doit générer plusieurs millions d'éléments sans erreur et sans consommation mémoire inutile.

---

# SPRINT 7 — DISTRIBUTIONS STATISTIQUES RÉALISTES

## 7.1 Responsabilité

> Implémenter le moteur de distributions permettant de représenter des comportements statistiques plus réalistes que la seule loi normale.

Ce Sprint est fondamental pour PMG.

---

# 7.2 Pourquoi plusieurs distributions ?

Une distribution normale possède des queues relativement légères.

Pour certaines grandeurs observées dans des systèmes complexes, les valeurs extrêmes peuvent être plus fréquentes.

Une distribution à queues lourdes peut être plus appropriée.

PMG doit donc pouvoir représenter notamment :

- Normal ;
- Student-t ;
- Laplace ;
- Log-normal ;
- Weibull ;
- Pareto.

Le choix exact doit rester configurable.

---

# Étape 7.1 — Abstraction Distribution

### Fichier

```text
crates/pmg-math/src/distributions/mod.rs
```

### Responsabilité

Définir l'interface commune.

Conceptuellement :

```rust
pub trait Distribution {
    fn sample(&mut self) -> f64;
}
```

Une abstraction plus riche peut également exposer :

```rust
fn mean(&self) -> Option<f64>;
fn variance(&self) -> Option<f64>;
fn name(&self) -> &'static str;
```

### Point critique

Certaines distributions ont des moments qui n'existent pas pour certains paramètres.

Il ne faut donc pas supposer que :

```text
variance = toujours définie
```

---

# Étape 7.2 — Distribution normale

### Fichier

```text
crates/pmg-math/src/distributions/normal.rs
```

### Formule

\[
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\]

### Paramètres

- \(\mu\) : moyenne ;
- \(\sigma\) : écart-type.

### Tests

Vérifier approximativement :

\[
\bar{x}\approx\mu
\]

et :

\[
s^2\approx\sigma^2
\]

sur un échantillon suffisamment grand.

---

# Étape 7.3 — Student-t

### Fichier

```text
crates/pmg-math/src/distributions/student_t.rs
```

### Responsabilité

Implémenter une distribution à queues lourdes.

Sa densité est :

\[
f(x)=
\frac{
\Gamma((\nu+1)/2)
}{
\sqrt{\nu\pi}\Gamma(\nu/2)
}
\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}
\]

où \(\nu\) est le nombre de degrés de liberté.

### Intuition débutant

Plus \(\nu\) est petit :

```text
plus les queues sont lourdes
```

Plus \(\nu\) augmente :

```text
Student-t → Normal
```

### Utilisation PMG

Elle peut être utilisée pour générer une base présentant davantage de valeurs extrêmes avant injection explicite des super-poids.

---

# Étape 7.4 — Laplace

### Fichier

```text
crates/pmg-math/src/distributions/laplace.rs
```

### Densité

\[
f(x)=
\frac{1}{2b}
e^{-\frac{|x-\mu|}{b}}
\]

Elle est intéressante pour représenter des distributions plus concentrées autour du centre avec des queues différentes de la normale.

---

# Étape 7.5 — Log-normal

### Fichier

```text
crates/pmg-math/src/distributions/log_normal.rs
```

### Définition

Si :

\[
Y\sim N(\mu,\sigma^2)
\]

alors :

\[
X=e^Y
\]

suit une loi log-normale.

Elle est particulièrement intéressante lorsqu'une quantité doit rester positive.

---

# Étape 7.6 — Weibull

### Fichier

```text
crates/pmg-math/src/distributions/weibull.rs
```

### Densité

\[
f(x)=
\frac{k}{\lambda}
\left(\frac{x}{\lambda}\right)^{k-1}
e^{-(x/\lambda)^k}
\]

pour \(x\geq0\).

### Utilisation

La Weibull peut servir à générer certaines amplitudes positives ou certains facteurs d'échelle.

---

# Étape 7.7 — Pareto

### Fichier

```text
crates/pmg-math/src/distributions/pareto.rs
```

### Densité

\[
f(x)=
\frac{\alpha x_m^\alpha}{x^{\alpha+1}}
\]

pour :

\[
x\geq x_m
\]

### Importance

La Pareto permet de modéliser des queues extrêmement lourdes.

Elle doit cependant être utilisée avec prudence.

### Point critique

Il ne faut pas utiliser Pareto partout.

PMG doit permettre :

```text
distribution = normal
distribution = student_t
distribution = weibull
distribution = pareto
```

mais aussi leur utilisation localisée.

---

# Étape 7.8 — Configuration des distributions

### Fichier

```text
crates/pmg-core/src/distribution_config.rs
```

Exemple conceptuel :

```json
{
  "distribution": {
    "type": "student_t",
    "degrees_of_freedom": 5.0,
    "location": 0.0,
    "scale": 0.02
  }
}
```

### Point fort

Le moteur mathématique reste indépendant du format de configuration.

---

# Étape 7.9 — Validation statistique

### Fichier

```text
crates/pmg-math/tests/distribution_tests.rs
```

Tests :

- moyenne ;
- variance lorsque définie ;
- quantiles ;
- asymétrie ;
- kurtosis ;
- comportement des queues.

Le test ne doit pas exiger une égalité exacte.

On utilise des tolérances statistiques.

Exemple :

\[
|\hat{\mu}-\mu|<\epsilon
\]

---

# SPRINT 8 — STRUCTURES, CORRÉLATIONS ET BAS-RANG

## 8.1 Responsabilité

> Introduire une structure non indépendante entre les valeurs des tenseurs.

Un tenseur réel n'est pas nécessairement une collection de nombres indépendants.

PMG doit donc pouvoir représenter :

- corrélations ;
- structure par blocs ;
- facteurs latents ;
- composantes bas-rang ;
- covariance structurée.

---

# Étape 8.1 — Modèle de base indépendant

### Fichier

```text
crates/pmg-math/src/structure/base_structure.rs
```

### Responsabilité

Définir le modèle :

\[
W=E
\]

où chaque élément est principalement issu du générateur statistique.

C'est notre référence zéro.

---

# Étape 8.2 — Structure bas-rang

### Fichier

```text
crates/pmg-math/src/structure/low_rank.rs
```

### Modèle

Une matrice structurée peut être représentée :

\[
W=L+E
\]

où :

- \(L\) = composante bas-rang ;
- \(E\) = bruit résiduel.

Avec :

\[
L=UV^T
\]

où :

\[
U\in\mathbb{R}^{m\times r}
\]

et :

\[
V\in\mathbb{R}^{n\times r}
\]

avec :

\[
r\ll\min(m,n)
\]

### Exemple

Une matrice :

\[
4096\times4096
\]

peut avoir une composante de rang :

\[
r=16
\]

Cela permet d'introduire une structure importante avec relativement peu de paramètres.

---

# Étape 8.3 — Générateur de facteurs

### Fichier

```text
crates/pmg-math/src/structure/factors.rs
```

### Responsabilité

Générer \(U\) et \(V\).

Les facteurs peuvent eux-mêmes utiliser les distributions du Sprint 7.

Exemple :

\[
U_{ij}\sim Student_t(\nu,\sigma)
\]

\[
V_{ij}\sim Normal(0,\sigma)
\]

---

# Étape 8.4 — Corrélation

### Fichier

```text
crates/pmg-math/src/structure/correlation.rs
```

### Responsabilité

Introduire une corrélation contrôlée.

Une construction générique est :

\[
x = Az
\]

avec :

\[
z\sim N(0,I)
\]

et :

\[
\Sigma=AA^T
\]

Ainsi :

\[
Cov(x)=\Sigma
\]

### Point critique

La matrice de covariance doit être valide.

Elle doit notamment être symétrique :

\[
\Sigma=\Sigma^T
\]

et, dans le cas classique, semi-définie positive.

---

# Étape 8.5 — Corrélation locale

### Fichier

```text
crates/pmg-math/src/structure/local_correlation.rs
```

### Responsabilité

Introduire des corrélations sans créer une énorme matrice globale.

Exemple conceptuel :

```text
bloc 0 → corrélation forte
bloc 1 → corrélation moyenne
bloc 2 → faible corrélation
```

Cette approche est plus adaptée au streaming.

---

# Étape 8.6 — Structure par blocs

### Fichier

```text
crates/pmg-math/src/structure/block_structure.rs
```

### Modèle

\[
W=
\begin{bmatrix}
W_1 & 0 & 0\\
0 & W_2 & 0\\
0 & 0 & W_3
\end{bmatrix}
\]

ou avec des interactions contrôlées entre blocs.

### Utilisation

Permettre à PMG de simuler différentes structures sans stocker une covariance complète.

---

# Étape 8.7 — Paramètre de force structurelle

### Fichier

```text
crates/pmg-core/src/structure_config.rs
```

PMG doit permettre un paramètre du type :

```text
structure_strength = 0.0
```

pour une génération indépendante.

Puis :

```text
structure_strength = 0.5
```

pour une structure intermédiaire.

Et :

```text
structure_strength = 1.0
```

pour une structure dominante.

Le mapping exact doit être défini mathématiquement et testé.

---

# Étape 8.8 — Tests structurels

### Fichier

```text
crates/pmg-math/tests/structure_tests.rs
```

Tests :

- rang approximatif ;
- corrélation ;
- covariance ;
- stabilité numérique ;
- reproductibilité ;
- structure par blocs.

Un test important consiste à comparer :

```text
structure_strength = 0
```

et :

```text
structure_strength > 0
```

afin de vérifier que le paramètre produit effectivement une différence mesurable.

---

# SPRINT 9 — SUPER-POIDS ET ANOMALIES CRITIQUES

## 9.1 Responsabilité

> Implémenter l'injection contrôlée de valeurs extrêmes et de structures d'outliers inspirées des phénomènes observables dans les poids de modèles.

Ce Sprint ne doit pas simplement faire :

```text
x *= 100
```

au hasard.

Il doit produire des anomalies **contrôlables, reproductibles et mesurables**.

---

# Étape 9.1 — Modèle d'outlier

### Fichier

```text
crates/pmg-math/src/outliers/model.rs
```

### Modèle conceptuel

On peut définir :

\[
W'=W+O
\]

où \(O\) est une composante d'anomalie.

Une autre approche :

\[
W'=W\odot M
\]

où \(M\) est un masque multiplicatif.

---

# Étape 9.2 — Masque d'outliers

### Fichier

```text
crates/pmg-math/src/outliers/mask.rs
```

### Responsabilité

Déterminer quelles positions sont affectées.

Types possibles :

```text
élément-wise
row-wise
column-wise
block-wise
channel-wise
```

### Exemple

Pour une matrice :

```text
1000 × 1000
```

on peut choisir :

```text
0.1 % des éléments
```

ou :

```text
2 % des colonnes
```

---

# Étape 9.3 — Super-poids

### Fichier

```text
crates/pmg-math/src/outliers/super_weight.rs
```

### Responsabilité

Créer des valeurs extrêmes contrôlées.

On peut définir :

\[
x_{outlier}=s\cdot x
\]

avec \(s>1\).

Mais une stratégie plus riche peut être :

\[
x_{outlier}
=
sign(x)
\left(
|x|+\Delta
\right)
\]

où \(\Delta\) dépend d'une distribution à queue lourde.

### Exemple

Valeur normale :

```text
0.018
```

Super-poids :

```text
2.4
```

La valeur n'est donc plus seulement un nombre aléatoire : elle devient un événement statistique identifiable.

---

# Étape 9.4 — Amplitude des super-poids

### Fichier

```text
crates/pmg-math/src/outliers/amplitude.rs
```

### Responsabilité

Déterminer l'amplitude des anomalies.

Plusieurs stratégies peuvent être supportées :

```text
fixed
relative_to_std
quantile_based
heavy_tail
```

### Exemple quantile

Si :

\[
q_{0.999}
\]

est le 99.9e percentile, une anomalie peut être créée au-delà de ce seuil.

---

# Étape 9.5 — Outliers structurés

### Fichier

```text
crates/pmg-math/src/outliers/structured.rs
```

### Responsabilité

Créer des anomalies corrélées.

Exemple :

```text
colonne 125
████████████████
    ↑
super-poids
```

plutôt que :

```text
. . . X . . X . X .
```

Cela permet de distinguer :

```text
outlier aléatoire
```

et :

```text
outlier structurel
```

---

# Étape 9.6 — Super-poids par couche

### Fichier

```text
crates/pmg-math/src/outliers/layer_policy.rs
```

### Responsabilité

Définir une politique différente selon la couche.

Exemple :

```text
layers 0-10   → faible
layers 11-20  → moyen
layers 21-30  → élevé
```

La politique peut être décrite par :

\[
p(l)
\]

où \(l\) est l'indice de couche.

---

# Étape 9.7 — Métadonnées d'anomalies

### Fichier

```text
crates/pmg-core/src/outlier_metadata.rs
```

PMG doit conserver les informations nécessaires à la validation.

Exemple :

```json
{
  "count": 1250,
  "fraction": 0.00012,
  "max_abs": 8.42,
  "strategy": "quantile_based",
  "seed": 42
}
```

---

# Étape 9.8 — Tests des super-poids

### Fichier

```text
crates/pmg-math/tests/outlier_tests.rs
```

Tests :

- nombre d'outliers ;
- position ;
- amplitude ;
- déterminisme ;
- signe ;
- distribution ;
- absence d'outliers lorsque fréquence = 0.

Test essentiel :

```text
outlier_frequency = 0
```

doit produire exactement le même résultat que le générateur sans injection.

---

# SPRINT 10 — GÉNÉRATION DU PSEUDO-MODÈLE COMPLET

## 10.1 Responsabilité

> Assembler les composants mathématiques et I/O afin de générer un pseudo-modèle complet compatible avec la structure attendue.

C'est le Sprint d'intégration du moteur de génération.

---

# Architecture

```text
ModelConfig
     │
     ▼
GenerationPlan
     │
     ▼
TensorGenerator
     │
     ├── Distribution
     │
     ├── Structure
     │
     ├── Low Rank
     │
     └── Outliers
     │
     ▼
TensorStream
     │
     ▼
SafetensorsWriter
     │
     ▼
Pseudo-Model
```

---

# Étape 10.1 — Configuration globale

### Fichier

```text
crates/pmg-core/src/generator_config.rs
```

Exemple :

```json
{
  "seed": 42,
  "distribution": {
    "type": "student_t",
    "degrees_of_freedom": 5.0
  },
  "structure": {
    "type": "low_rank",
    "rank": 16,
    "strength": 0.35
  },
  "outliers": {
    "enabled": true,
    "frequency": 0.0001,
    "strategy": "quantile_based"
  }
}
```

---

# Étape 10.2 — Pipeline de génération

### Fichier

```text
crates/pmg-core/src/generation_pipeline.rs
```

### Responsabilité

Orchestrer les transformations dans un ordre explicite.

Pipeline recommandé :

\[
X_0
\rightarrow
X_1
\rightarrow
X_2
\rightarrow
X_3
\]

avec :

\[
X_0 = \text{base distribution}
\]

\[
X_1 = \text{structure}(X_0)
\]

\[
X_2 = \text{correlation}(X_1)
\]

\[
X_3 = \text{outliers}(X_2)
\]

### Point critique

L'ordre doit être documenté.

Changer :

```text
structure → outliers
```

en :

```text
outliers → structure
```

peut produire des propriétés statistiques différentes.

---

# Étape 10.3 — Génération d'un tenseur

### Fichier

```text
crates/pmg-core/src/tensor_generation.rs
```

### Responsabilité

Générer un tenseur complet à partir de son plan.

Exemple :

```text
model.layers.0.self_attn.q_proj.weight
```

devient :

```text
shape = [4096, 4096]
dtype = BF16
```

Puis le moteur génère les chunks correspondants.

---

# Étape 10.4 — Génération par streaming

### Fichier

```text
crates/pmg-core/src/streaming_generation.rs
```

### Responsabilité

Connecter :

```text
math engine
```

à :

```text
Safetensors writer
```

sans conserver inutilement tout le tenseur en mémoire.

Flux :

```text
Generate chunk
      ↓
Transform chunk
      ↓
Encode chunk
      ↓
Write chunk
      ↓
Discard chunk
      ↓
Next chunk
```

---

# Étape 10.5 — Génération du fichier de configuration

### Fichier

```text
crates/pmg-io/src/config_writer.rs
```

### Responsabilité

Écrire :

```text
config.json
```

avec les paramètres cohérents avec le pseudo-modèle.

---

# Étape 10.6 — Génération des métadonnées

### Fichier

```text
crates/pmg-io/src/metadata_writer.rs
```

### Responsabilité

Produire les métadonnées permettant d'expliquer comment le pseudo-modèle a été généré.

Exemple :

```json
{
  "generator": "PMG",
  "version": "1.0.0",
  "seed": 42,
  "distribution": "student_t",
  "structure": "low_rank",
  "outliers": true
}
```

---

# Étape 10.7 — Manifest du pseudo-modèle

### Fichier

```text
crates/pmg-core/src/manifest.rs
```

### Responsabilité

Décrire le contenu du pseudo-modèle.

Exemple :

```text
manifest.json
```

avec :

```text
model type
architecture
number of tensors
number of parameters
dtype
generation seed
generation strategy
```

---

# Étape 10.8 — Test complet de génération

### Fichier

```text
crates/pmg-core/tests/full_generation.rs
```

Scénario :

```text
configuration minimale
        ↓
génération
        ↓
écriture
        ↓
lecture
        ↓
validation
```

### Critère

Le modèle produit doit pouvoir être relu par PMG sans incohérence.

---

# Étape 10.9 — Test de déterminisme global

### Fichier

```text
crates/pmg-core/tests/determinism_tests.rs
```

Scénario :

```text
generate(seed=42) → model A
generate(seed=42) → model B
```

Puis comparer les données générées.

Résultat attendu :

```text
A == B
```

Puis :

```text
generate(seed=43) → model C
```

Résultat attendu :

```text
A != C
```

---

# SPRINT 11 — VALIDATION, COMPARAISON ET STABILISATION

## 11.1 Responsabilité

> Construire le système permettant de déterminer si un pseudo-modèle respecte les propriétés attendues et de comparer ses statistiques avec celles d'un modèle original sans télécharger ses poids complets.

Ce Sprint transforme PMG d'un simple générateur en **outil scientifique de validation**.

---

# 11.2 Principes

PMG doit distinguer :

```text
VALID
```

```text
INVALID
```

et :

```text
WARNING
```

Une différence statistique modérée ne signifie pas nécessairement que le modèle est invalide.

---

# Étape 11.1 — Collecteur de statistiques

### Fichier

```text
crates/pmg-analysis/src/statistics.rs
```

### Responsabilité

Calculer :

- moyenne ;
- variance ;
- écart-type ;
- minimum ;
- maximum ;
- quantiles ;
- médiane ;
- skewness ;
- kurtosis ;
- norme L1 ;
- norme L2 ;
- norme infinie.

---

# Étape 11.2 — Statistiques de queues

### Fichier

```text
crates/pmg-analysis/src/tail_statistics.rs
```

### Responsabilité

Analyser les valeurs extrêmes.

Exemples :

\[
q_{0.99}
\]

\[
q_{0.999}
\]

\[
q_{0.9999}
\]

et :

\[
\frac{\max |x|}{\sigma}
\]

Cette dernière mesure peut être particulièrement utile pour suivre les super-poids.

---

# Étape 11.3 — Détection d'outliers

### Fichier

```text
crates/pmg-analysis/src/outlier_analysis.rs
```

### Responsabilité

Détecter les valeurs dépassant des seuils configurés.

Exemple :

\[
|x|>k\sigma
\]

avec :

```text
k = 6
```

ou une approche par quantile.

---

# Étape 11.4 — Analyse de corrélation

### Fichier

```text
crates/pmg-analysis/src/correlation_analysis.rs
```

### Responsabilité

Mesurer les corrélations présentes.

Pour deux variables :

\[
\rho_{X,Y}
=
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
\]

PMG peut produire :

```text
corrélation moyenne
corrélation maximale
corrélation par bloc
```

---

# Étape 11.5 — Analyse bas-rang

### Fichier

```text
crates/pmg-analysis/src/low_rank_analysis.rs
```

### Responsabilité

Estimer la concentration de l'énergie matricielle.

Si les valeurs singulières sont :

\[
\sigma_1\geq\sigma_2\geq\dots\geq\sigma_n
\]

on peut mesurer :

\[
E_r=
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{n}\sigma_i^2}
\]

Cela donne la fraction d'énergie expliquée par les \(r\) premières composantes.

---

# Étape 11.6 — Validateur de modèle

### Fichier

```text
crates/pmg-validation/src/validator.rs
```

### Responsabilité

Combiner les différents contrôles.

Architecture :

```text
Validator
   │
   ├── StructuralValidator
   ├── StatisticalValidator
   ├── DistributionValidator
   ├── OutlierValidator
   └── MetadataValidator
```

---

# Étape 11.7 — Niveaux de validation

### Fichier

```text
crates/pmg-validation/src/severity.rs
```

Définir :

```text
INFO
WARNING
ERROR
CRITICAL
```

Exemple :

```text
INFO:
distribution conforme.

WARNING:
variance légèrement différente.

ERROR:
shape incompatible.

CRITICAL:
offset Safetensors invalide.
```

---

# Étape 11.8 — Comparateur

### Fichier

```text
crates/pmg-analysis/src/comparator.rs
```

### Responsabilité

Comparer :

```text
modèle original
```

avec :

```text
pseudo-modèle
```

sans télécharger les fichiers de poids complets du modèle original.

Le comparateur travaille principalement avec :

- configuration ;
- métadonnées ;
- headers ;
- index ;
- statistiques disponibles ;
- propriétés structurelles.

---

# Étape 11.9 — Comparaison des configurations

### Fichier

```text
crates/pmg-analysis/src/config_comparator.rs
```

Comparer :

```text
hidden_size
num_layers
num_heads
num_kv_heads
intermediate_size
num_experts
vocab_size
dtype
```

Exemple :

```text
Original:
num_layers = 80

PMG:
num_layers = 80

Résultat:
MATCH
```

---

# Étape 11.10 — Comparaison des structures de tenseurs

### Fichier

```text
crates/pmg-analysis/src/tensor_comparator.rs
```

Comparer :

- noms ;
- dimensions ;
- types ;
- nombre de paramètres ;
- offsets ;
- organisation des fichiers.

---

# Étape 11.11 — Comparaison statistique

### Fichier

```text
crates/pmg-analysis/src/statistical_comparator.rs
```

Si les statistiques sont disponibles pour les deux modèles, comparer :

\[
\Delta_\mu
=
|\mu_A-\mu_B|
\]

\[
\Delta_\sigma
=
|\sigma_A-\sigma_B|
\]

et éventuellement :

\[
D_{KS}
\]

pour comparer deux distributions empiriques.

### Point critique

Une distance statistique n'est pas une preuve d'équivalence fonctionnelle.

PMG doit afficher cette distinction explicitement.

---

# Étape 11.12 — Rapport de validation

### Fichier

```text
crates/pmg-validation/src/report.rs
```

Le rapport doit être exploitable par :

```text
CLI
```

et éventuellement :

```text
JSON
```

Exemple :

```text
╭──────────────────────────────────────────────╮
│ PMG — RAPPORT DE VALIDATION                  │
├──────────────────────────────────────────────┤
│ Modèle             : pseudo-model            │
│ Tenseurs           : 1 284                   │
│ Paramètres         : 70.2 B                  │
│ Distribution       : Student-t               │
│ Structure          : Low-rank r=16           │
│ Super-poids        : activés                 │
├──────────────────────────────────────────────┤
│ Structure          : ✓ PASS                  │
│ Statistiques       : ✓ PASS                  │
│ Distribution       : ✓ PASS                  │
│ Outliers           : ⚠ WARNING               │
│ Métadonnées        : ✓ PASS                  │
├──────────────────────────────────────────────┤
│ RESULTAT GLOBAL : PASS WITH WARNINGS         │
╰──────────────────────────────────────────────╯
```

---

# Étape 11.13 — Intégration de la commande `validate`

### Fichier

```text
crates/pmg-cli/src/commands/validate.rs
```

Commande :

```bash
pmg validate model.safetensors
```

Options :

```bash
pmg validate model.safetensors --verbose
pmg validate model.safetensors --debug
pmg validate model.safetensors --dry-run
```

---

# Étape 11.14 — Intégration de `compare`

### Fichier

```text
crates/pmg-cli/src/commands/compare.rs
```

Commande :

```bash
pmg compare original/ pseudo/
```

### Règle fondamentale

La commande `compare` ne doit pas télécharger automatiquement les poids du modèle original.

Elle doit inspecter les informations disponibles.

---

# Étape 11.15 — Intégration de `espec`

### Fichier

```text
crates/pmg-cli/src/commands/espec.rs
```

Responsabilité :

Afficher :

```text
configuration
architecture
dimensions
dtype
nombre de paramètres
statistiques
structure
distribution
anomalies
```

Exemple :

```bash
pmg espec pseudo-model/
```

---

# Étape 11.16 — Intégration de `generate`

### Fichier

```text
crates/pmg-cli/src/commands/generate.rs
```

Responsabilité :

Orchestrer la génération depuis le CLI.

Exemple :

```bash
pmg generate \
    --template llama \
    --distribution student-t \
    --rank 16 \
    --outliers \
    --seed 42 \
    --output ./pseudo-model
```

---

# Étape 11.17 — Commande `help`

### Fichier

```text
crates/pmg-cli/src/help.rs
```

La documentation doit être orientée débutant.

Exemple :

```bash
pmg help
```

doit expliquer :

```text
1. ce qu'est PMG ;
2. les commandes ;
3. les options ;
4. des exemples simples ;
5. les erreurs courantes.
```

---

# Étape 11.18 — `version`

### Fichier

```text
crates/pmg-cli/src/commands/version.rs
```

Exemple :

```bash
pmg version
```

Affichage :

```text
PMG — Pseudo-Models Generator
Version : 1.0.0
Rust    : 1.xx
Licence : GPL-3.0
Build   : release
```

---

# Étape 11.19 — Tests CLI

### Fichier

```text
crates/pmg-cli/tests/commands.rs
```

Tests :

```text
pmg help
pmg generate --help
pmg espec --help
pmg validate --help
pmg compare --help
pmg version
```

---

# Étape 11.20 — Tests E2E

### Fichier

```text
crates/pmg-cli/tests/e2e.rs
```

Scénario complet :

```text
generate
   ↓
espec
   ↓
validate
   ↓
compare
```

Le test doit vérifier que les quatre opérations fonctionnent sur un petit modèle de test.

---

# 12. MATRICE DES DÉPENDANCES

```text
                 ┌─────────────┐
                 │  Sprint 6   │
                 │ Génération  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Sprint 7   │
                 │Distributions│
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Sprint 8   │
                 │ Structures  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Sprint 9   │
                 │Super-poids  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Sprint 10   │
                 │ Génération  │
                 │ complète    │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Sprint 11   │
                 │ Validation  │
                 └─────────────┘
```

---

# 13. STRATÉGIE DE TESTS

## 13.1 Tests unitaires

Chaque module mathématique possède ses propres tests.

```text
distribution
   → tests distribution

structure
   → tests structure

outlier
   → tests outlier
```

---

## 13.2 Tests statistiques

Les tests statistiques doivent éviter les seuils trop stricts.

Mauvais :

```text
assert_eq!(mean, 0.0);
```

Meilleur :

\[
|\bar{x}-0|<\epsilon
\]

avec une tolérance justifiée.

---

# 13.3 Tests de propriétés

PMG doit progressivement introduire des tests de propriétés :

```text
seed identique → sortie identique
seed différente → sortie différente
fréquence outlier = 0 → aucun outlier
rank = 0 → aucune composante bas-rang
strength = 0 → structure désactivée
```

---

# 13.4 Tests de non-régression

Chaque bug scientifique ou mathématique corrigé doit devenir un test.

Exemple :

```text
Bug :
collision de seed entre deux couches.

Correction :
nouvelle dérivation de seed.

Test :
deux couches différentes doivent avoir deux séquences différentes.
```

---

# 14. STRATÉGIE DE PERFORMANCE

PMG doit privilégier :

```text
streaming
```

plutôt que :

```text
chargement intégral
```

---

## 14.1 Mémoire

Pour un tenseur de taille \(N\), l'objectif est que la mémoire de travail reste approximativement :

\[
O(B)
\]

où \(B\) est la taille du chunk.

et non :

\[
O(N)
\]

---

# 14.2 Parallélisme

Le parallélisme ne doit être introduit qu'après validation du chemin séquentiel.

Ordre recommandé :

```text
correct
   ↓
testé
   ↓
benchmarké
   ↓
parallélisé
```

Rayon peut ensuite être utilisé pour certaines opérations indépendantes.

---

# 14.3 Benchmark

Les benchmarks doivent mesurer notamment :

```text
éléments/seconde
octets/seconde
mémoire maximale
temps par tenseur
temps par chunk
temps total de génération
```

---

# 15. STRATÉGIE DE VALIDATION SCIENTIFIQUE

PMG ne doit jamais déclarer :

```text
"ce pseudo-modèle est identique au modèle réel"
```

sur la seule base de statistiques.

Il doit utiliser des formulations précises :

```text
structure compatible
```

```text
distribution statistiquement similaire
```

```text
configuration compatible
```

```text
présence d'une structure bas-rang similaire
```

---

# 15.1 Niveaux de similarité

Un rapport peut utiliser :

### Niveau 0 — Structure

```text
architecture compatible
```

### Niveau 1 — Statistique

```text
moments et quantiles compatibles
```

### Niveau 2 — Structure statistique

```text
corrélations et concentration spectrale compatibles
```

### Niveau 3 — Anomalies

```text
queues et super-poids compatibles
```

### Niveau 4 — Fonctionnel

Ce niveau ne doit pas être revendiqué par PMG uniquement à partir des fichiers.

Il nécessiterait des évaluations fonctionnelles externes.

---

# 16. CRITÈRES DE SORTIE DES SPRINTS

## Sprint 6 terminé si :

- [ ] générateur de base fonctionnel ;
- [ ] RNG déterministe ;
- [ ] génération par chunks ;
- [ ] tests unitaires ;
- [ ] tests de déterminisme ;
- [ ] documentation API ;
- [ ] benchmark initial.

---

## Sprint 7 terminé si :

- [ ] abstraction Distribution ;
- [ ] Normal ;
- [ ] Student-t ;
- [ ] Laplace ;
- [ ] Log-normal ;
- [ ] Weibull ;
- [ ] Pareto ;
- [ ] configuration ;
- [ ] validation statistique.

---

## Sprint 8 terminé si :

- [ ] structure bas-rang ;
- [ ] facteurs ;
- [ ] corrélations ;
- [ ] structures par blocs ;
- [ ] paramètres de force ;
- [ ] tests spectraux ;
- [ ] tests de covariance.

---

## Sprint 9 terminé si :

- [ ] masque d'outliers ;
- [ ] super-poids ;
- [ ] amplitudes contrôlées ;
- [ ] outliers structurés ;
- [ ] politique par couche ;
- [ ] métadonnées ;
- [ ] tests statistiques.

---

## Sprint 10 terminé si :

- [ ] pipeline complet ;
- [ ] génération streaming ;
- [ ] configuration ;
- [ ] métadonnées ;
- [ ] manifest ;
- [ ] génération Safetensors ;
- [ ] déterminisme global ;
- [ ] test E2E.

---

## Sprint 11 terminé si :

- [ ] statistiques ;
- [ ] analyse des queues ;
- [ ] analyse des outliers ;
- [ ] corrélations ;
- [ ] analyse bas-rang ;
- [ ] validation ;
- [ ] comparaison ;
- [ ] `generate` ;
- [ ] `espec` ;
- [ ] `validate` ;
- [ ] `compare` ;
- [ ] `version` ;
- [ ] `help` ;
- [ ] tests CLI ;
- [ ] tests E2E ;
- [ ] documentation.

---

# 17. ORDRE DE TRAVAIL RECOMMANDÉ POUR IBRAHIMA-224

Pour un développeur unique, l'ordre exact recommandé est :

```text
1. Sprint 6
   ├── générateur
   ├── RNG
   └── streaming

2. Sprint 7
   ├── Normal
   ├── Student-t
   ├── Laplace
   ├── Log-normal
   ├── Weibull
   └── Pareto

3. Sprint 8
   ├── low-rank
   ├── facteurs
   ├── covariance
   └── corrélation

4. Sprint 9
   ├── masque
   ├── amplitude
   ├── super-poids
   └── anomalies structurées

5. Sprint 10
   ├── pipeline
   ├── streaming
   ├── metadata
   └── modèle complet

6. Sprint 11
   ├── analyse
   ├── validation
   ├── comparaison
   └── CLI
```

Il est fortement déconseillé de commencer le Sprint 10 avant que les trois composants mathématiques suivants soient individuellement validés :

```text
Distribution
Structure
Outlier
```

---

# 18. RÉFÉRENCES TECHNIQUES ET GUIDES

## Rust

- [The Rust Programming Language](https://doc.rust-lang.org/book/?utm_source=chatgpt.com)
- [Rust Standard Library](https://doc.rust-lang.org/std/?utm_source=chatgpt.com)
- [Cargo Book](https://doc.rust-lang.org/cargo/?utm_source=chatgpt.com)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/?utm_source=chatgpt.com)

## CLI

- [clap documentation](https://docs.rs/clap/?utm_source=chatgpt.com)

## Sérialisation

- [Serde documentation](https://serde.rs/?utm_source=chatgpt.com)
- [serde_json documentation](https://docs.rs/serde_json/?utm_source=chatgpt.com)

## Génération aléatoire

- [rand documentation](https://docs.rs/rand/?utm_source=chatgpt.com)
- [rand_chacha documentation](https://docs.rs/rand_chacha/?utm_source=chatgpt.com)

## Parallélisme

- [Rayon documentation](https://docs.rs/rayon/?utm_source=chatgpt.com)

## Format Safetensors

- [Hugging Face Safetensors documentation](https://huggingface.co/docs/safetensors/?utm_source=chatgpt.com)

## Statistiques

Les implémentations doivent être vérifiées contre des références mathématiques reconnues concernant :

- distributions de probabilité ;
- moments statistiques ;
- quantiles ;
- corrélations ;
- covariance ;
- décomposition spectrale ;
- valeurs singulières ;
- matrices de faible rang.

---

# 19. CONCLUSION

Les Sprints 6 à 11 constituent le **cœur scientifique de PMG**.

Le chemin de développement est volontairement progressif :

\[
\boxed{
Génération
\rightarrow
Distribution
\rightarrow
Structure
\rightarrow
Super\text{-}poids
\rightarrow
Pseudo\text{-}modèle
\rightarrow
Validation
}
\]

Le principe fondamental est de ne jamais masquer la complexité scientifique derrière une simple fonction de génération.

Un pseudo-modèle PMG doit être construit à partir de plusieurs dimensions indépendantes :

\[
M =
f(
Architecture,
Distribution,
Structure,
Corrélation,
Anomalies,
Seed
)
\]

Cela permet ensuite à PMG de répondre à des besoins différents :

```text
génération simple
       ↓
génération statistique réaliste
       ↓
génération structurelle
       ↓
génération avec super-poids
       ↓
génération complète
       ↓
validation scientifique
```

Le Sprint 11 marque ainsi la transition entre un **moteur de génération** et un véritable **système PMG complet**, capable de générer, inspecter, valider et comparer des pseudo-modèles de manière reproductible.

**Fin du Cahier de Plan de Développement — Sprints 6 à 11.**