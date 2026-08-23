# CAHIER DE PLAN DE DÉVELOPPEMENT
# PSEUDO-MODELS GENERATOR — PMG

**Partie III — Sprints 12 à 17**

---

## 0. IDENTIFICATION DU DOCUMENT

| Élément | Valeur |
|---|---|
| Projet | Pseudo-Models Generator |
| Acronyme | PMG |
| Version du plan | 1.0 |
| Période | Sprints 12 à 17 |
| Responsable du développement | Ibrahima-224 |
| Équipe de développement | 1 développeur |
| Langage | Rust |
| Licence | GPL-3.0 |
| Interface | CLI riche en français |
| Format principal | Safetensors / métadonnées de modèles |
| Architecture | Workspace Cargo multi-crates |
| Philosophie | Déterministe, modulaire, testable, reproductible |
| Document précédent | Cahier de Plan de Développement — Sprints 0 à 5 puis 6 à 11 |

---

# 1. POSITIONNEMENT DES SPRINTS 12 À 17

Les Sprints 0 à 11 ont pour rôle de construire progressivement les fondations de PMG :

```text
Sprints 0–5
    ↓
Fondations du projet
    ↓
Core / Blueprint / Mathématiques fondamentales
    ↓
Sprints 6–11
    ↓
Génération avancée / I/O / structures statistiques
    ↓
Sprints 12–17
    ↓
Validation scientifique
    ↓
Inspection
    ↓
Comparaison
    ↓
CLI complète
    ↓
Robustesse
    ↓
Release candidate
```

Les Sprints 12 à 17 constituent donc une phase de **consolidation et d'intégration**.

L'objectif n'est plus simplement de produire des composants isolés, mais de transformer ces composants en un logiciel PMG cohérent.

---

# 2. OBJECTIFS GLOBAUX DES SPRINTS 12 À 17

Cette phase doit permettre d'obtenir un PMG capable de :

1. générer un pseudo-modèle à partir d'un blueprint ;
2. appliquer les distributions statistiques configurées ;
3. appliquer les structures de corrélation ;
4. appliquer les structures de bas-rang ;
5. injecter les super-poids et outliers ;
6. préserver les contraintes structurelles du modèle ;
7. produire des fichiers exploitables ;
8. inspecter un modèle sans charger tous ses poids ;
9. comparer un modèle original et un pseudo-modèle au niveau des métadonnées ;
10. valider mathématiquement et structurellement un modèle ;
11. exposer toutes ces fonctions à travers une CLI française ;
12. produire des diagnostics compréhensibles par un débutant ;
13. gérer proprement les erreurs ;
14. assurer la reproductibilité grâce aux seeds ;
15. préparer une première version release candidate.

---

# 3. PRINCIPES DE DÉVELOPPEMENT DE CETTE PHASE

## 3.1. Un Sprint = une responsabilité principale

Chaque Sprint possède une seule responsabilité dominante.

```text
Sprint 12 → Orchestration de génération
Sprint 13 → Validation scientifique et statistique
Sprint 14 → Inspection des modèles
Sprint 15 → Comparaison des modèles
Sprint 16 → Interface CLI complète
Sprint 17 → Durcissement et Release Candidate
```

Cela évite qu'un Sprint devienne un mélange de fonctionnalités difficile à tester.

---

# 4. SPRINT 12 — ORCHESTRATION DE LA GÉNÉRATION

**Durée indicative : 2 à 3 semaines**

**Responsabilité unique :**

> Transformer les composants mathématiques et les blueprints en pipeline cohérent de génération d'un pseudo-modèle.

---

## 4.1. Objectifs

Le Sprint 12 doit créer le moteur qui coordonne :

```text
Blueprint
   ↓
Configuration
   ↓
Seed
   ↓
Distribution
   ↓
Structure
   ↓
Corrélation
   ↓
Bas-rang
   ↓
Outliers
   ↓
Super-poids
   ↓
Contraintes
   ↓
Tensor final
   ↓
Writer
```

Le moteur d'orchestration ne doit pas réimplémenter les mathématiques.

Il doit uniquement les **ordonner correctement**.

---

## 4.2. Attentes

À la fin du Sprint :

- une génération complète doit être possible ;
- chaque étape doit être déterministe ;
- les étapes doivent être activables/désactivables ;
- le pipeline doit produire des statistiques intermédiaires ;
- les erreurs doivent être propagées proprement ;
- un modèle identique doit être reproductible avec la même seed.

---

## 4.3. Points forts

- architecture modulaire ;
- séparation entre mathématiques et orchestration ;
- possibilité d'ajouter ultérieurement de nouveaux générateurs ;
- forte testabilité ;
- reproductibilité.

---

## 4.4. Points faibles

- beaucoup d'interactions entre crates ;
- risque d'API trop complexe ;
- risque de dépendances circulaires ;
- pipeline potentiellement difficile à comprendre si mal documenté.

---

## 4.5. Points critiques

### Critique 1 — ordre des opérations

L'ordre n'est pas arbitraire.

Par exemple :

\[
W = T(O(B(C(D(Z))))
\]

où :

- \(Z\) = bruit initial ;
- \(D\) = distribution ;
- \(C\) = corrélation ;
- \(B\) = structure bas-rang ;
- \(O\) = outliers ;
- \(T\) = transformations finales.

L'implémentation exacte dépend du modèle statistique choisi, mais le principe est fondamental :

> l'ordre du pipeline doit être explicite et documenté.

---

# 4.6. Structure du Sprint

```text
SPRINT 12
│
├── Étape 12.1 : Générateur principal
├── Étape 12.2 : Contexte de génération
├── Étape 12.3 : Gestion de la seed
├── Étape 12.4 : Pipeline des transformations
├── Étape 12.5 : Génération par tenseur
├── Étape 12.6 : Génération par couche
├── Étape 12.7 : Génération par modèle
├── Étape 12.8 : Statistiques intermédiaires
└── Étape 12.9 : Tests d'intégration du pipeline
```

---

# 4.7. ÉTAPE 12.1 — GÉNÉRATEUR PRINCIPAL

### Fichier

```text
crates/pmg-generator/src/generator.rs
```

### Responsabilité

Définir le point d'entrée du moteur de génération.

### API conceptuelle

```rust
pub struct ModelGenerator {
    blueprint: ModelBlueprint,
    config: GenerationConfig,
}
```

Puis :

```rust
impl ModelGenerator {
    pub fn generate(&self) -> Result<GeneratedModel, GeneratorError>;
}
```

### Objectif débutant

Le `ModelGenerator` joue le rôle du **chef d'orchestre**.

Il ne fabrique pas lui-même chaque note.

Il demande :

```text
distribution → génère les valeurs
structure → impose la structure
outlier → injecte les anomalies
writer → écrit les données
```

---

# 4.8. ÉTAPE 12.2 — CONTEXTE DE GÉNÉRATION

### Fichier

```text
crates/pmg-generator/src/context.rs
```

### Responsabilité

Centraliser les paramètres nécessaires pendant la génération.

Exemple :

```rust
pub struct GenerationContext {
    pub seed: u64,
    pub model_name: String,
    pub layer_index: usize,
    pub tensor_index: usize,
}
```

### Critique

Le contexte ne doit pas devenir un "sac à variables".

Chaque champ doit avoir une justification.

---

# 4.9. ÉTAPE 12.3 — GESTION DE LA SEED

### Fichier

```text
crates/pmg-generator/src/seed.rs
```

### Responsabilité

Garantir la reproductibilité.

Principe :

\[
S_i = H(S_0, i)
\]

où :

- \(S_0\) est la seed globale ;
- \(i\) identifie une couche ou un tenseur ;
- \(H\) est une fonction de dérivation déterministe.

Exemple conceptuel :

```text
seed globale
     ↓
seed modèle
     ↓
seed couche
     ↓
seed tenseur
     ↓
seed distribution
```

### Test critique

Deux générations avec :

```text
seed = 42
```

doivent produire exactement les mêmes résultats.

---

# 4.10. ÉTAPE 12.4 — PIPELINE DES TRANSFORMATIONS

### Fichier

```text
crates/pmg-generator/src/pipeline.rs
```

### Responsabilité

Définir l'ordre d'exécution des transformations.

Exemple :

```rust
pub struct GenerationPipeline {
    steps: Vec<PipelineStep>,
}
```

Avec :

```rust
pub enum PipelineStep {
    Distribution,
    Correlation,
    LowRank,
    Outliers,
    SuperWeights,
}
```

### Point critique

L'ordre doit être testé.

Le test doit notamment vérifier qu'une configuration :

```text
distribution → correlation → outlier
```

n'est pas silencieusement transformée en :

```text
outlier → distribution → correlation
```

---

# 4.11. ÉTAPE 12.5 — GÉNÉRATION PAR TENSEUR

### Fichier

```text
crates/pmg-generator/src/tensor_generator.rs
```

### Responsabilité

Générer un tenseur individuel.

Interface :

```rust
pub trait TensorGenerator {
    fn generate_tensor(
        &self,
        metadata: &TensorMetadata,
        context: &GenerationContext,
    ) -> Result<GeneratedTensor, GeneratorError>;
}
```

### Exemple

Pour :

```text
shape = [4096, 4096]
dtype = BF16
```

le générateur doit savoir :

- combien d'éléments produire ;
- quelle distribution utiliser ;
- quelles structures appliquer ;
- comment convertir vers le dtype final.

---

# 4.12. ÉTAPE 12.6 — GÉNÉRATION PAR COUCHE

### Fichier

```text
crates/pmg-generator/src/layer_generator.rs
```

### Responsabilité

Regrouper les tenseurs appartenant à une couche.

Exemple :

```text
layer.0
 ├── q_proj
 ├── k_proj
 ├── v_proj
 ├── o_proj
 ├── gate_proj
 ├── up_proj
 └── down_proj
```

Le générateur doit conserver les relations structurelles entre ces tenseurs.

---

# 4.13. ÉTAPE 12.7 — GÉNÉRATION DU MODÈLE

### Fichier

```text
crates/pmg-generator/src/model_generator.rs
```

### Responsabilité

Parcourir le blueprint complet.

Schéma :

```text
ModelBlueprint
      ↓
Embedding
      ↓
Layer 0
      ↓
Layer 1
      ↓
...
      ↓
Layer N
      ↓
Final Norm
      ↓
LM Head
```

### Point critique

Ne jamais supposer que tous les modèles ont exactement la même architecture.

Le blueprint doit rester la source de vérité.

---

# 4.14. ÉTAPE 12.8 — STATISTIQUES INTERMÉDIAIRES

### Fichier

```text
crates/pmg-generator/src/generation_stats.rs
```

### Responsabilité

Collecter :

- moyenne ;
- variance ;
- écart-type ;
- minimum ;
- maximum ;
- quantiles ;
- nombre d'outliers ;
- nombre de super-poids ;
- paramètres générés.

Variance :

\[
\sigma^2 =
\frac{1}{N}
\sum_{i=1}^{N}
(x_i-\mu)^2
\]

Écart-type :

\[
\sigma = \sqrt{\sigma^2}
\]

---

# 4.15. ÉTAPE 12.9 — TESTS DU PIPELINE

### Fichier

```text
crates/pmg-generator/tests/pipeline_tests.rs
```

Tests obligatoires :

```text
test_same_seed_same_output
test_different_seed_different_output
test_pipeline_order
test_tensor_generation
test_layer_generation
test_model_generation
test_statistics_collection
```

---

# 5. SPRINT 13 — VALIDATION SCIENTIFIQUE ET STATISTIQUE

**Durée indicative : 2 à 3 semaines**

**Responsabilité unique :**

> Déterminer si un pseudo-modèle respecte les propriétés statistiques, structurelles et numériques attendues.

---

# 5.1. Objectifs

Le validateur doit répondre à :

> "Le modèle généré ressemble-t-il suffisamment au modèle statistique et structurel décrit par son blueprint ?"

Il ne doit pas prétendre prouver que le pseudo-modèle est fonctionnellement équivalent au modèle original.

---

# 5.2. Structure

```text
SPRINT 13
│
├── Étape 13.1 : Moteur de validation
├── Étape 13.2 : Validation des shapes
├── Étape 13.3 : Validation des dtypes
├── Étape 13.4 : Validation des statistiques
├── Étape 13.5 : Validation des distributions
├── Étape 13.6 : Validation des corrélations
├── Étape 13.7 : Validation des structures bas-rang
├── Étape 13.8 : Validation des outliers
├── Étape 13.9 : Score global
└── Étape 13.10 : Rapport de validation
```

---

# 5.3. ÉTAPE 13.1 — MOTEUR DE VALIDATION

### Fichier

```text
crates/pmg-validation/src/validator.rs
```

### Responsabilité

Coordonner toutes les validations.

```rust
pub struct Validator {
    rules: ValidationRules,
}
```

---

# 5.4. ÉTAPE 13.2 — VALIDATION DES SHAPES

### Fichier

```text
crates/pmg-validation/src/shape_validation.rs
```

Vérifier :

\[
shape_{observé} = shape_{attendu}
\]

Exemple :

```text
attendu : [4096, 11008]
obtenu  : [4096, 11008]
→ PASS
```

Mais :

```text
attendu : [4096, 11008]
obtenu  : [4096, 11000]
→ FAIL
```

---

# 5.5. ÉTAPE 13.3 — VALIDATION DES DTYPES

### Fichier

```text
crates/pmg-validation/src/dtype_validation.rs
```

Vérifier la cohérence :

```text
Blueprint → dtype attendu
Model     → dtype trouvé
```

---

# 5.6. ÉTAPE 13.4 — VALIDATION STATISTIQUE

### Fichier

```text
crates/pmg-validation/src/statistical_validation.rs
```

Comparer :

\[
\mu_{observé}
\]

avec :

\[
\mu_{cible}
\]

et :

\[
\sigma_{observé}
\]

avec :

\[
\sigma_{cible}
\]

Une erreur relative peut être calculée par :

\[
E =
\frac{|\theta_{obs}-\theta_{target}|}
{\max(|\theta_{target}|,\epsilon)}
\]

---

# 5.7. ÉTAPE 13.5 — VALIDATION DES DISTRIBUTIONS

### Fichier

```text
crates/pmg-validation/src/distribution_validation.rs
```

Le système doit pouvoir vérifier les distributions utilisées :

- normale ;
- Student-t ;
- Weibull ;
- Pareto ;
- log-normale ;
- distributions configurées ultérieurement.

Selon le cas, on peut utiliser des statistiques adaptées ou des tests d'adéquation.

Pour une comparaison empirique :

\[
D =
\sup_x |F_n(x)-F(x)|
\]

où :

- \(F_n\) = CDF empirique ;
- \(F\) = CDF théorique.

---

# 5.8. ÉTAPE 13.6 — VALIDATION DES CORRÉLATIONS

### Fichier

```text
crates/pmg-validation/src/correlation_validation.rs
```

Pour deux variables :

\[
\rho_{X,Y}
=
\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
\]

Le validateur vérifie que les corrélations produites respectent les contraintes du blueprint.

---

# 5.9. ÉTAPE 13.7 — VALIDATION BAS-RANG

### Fichier

```text
crates/pmg-validation/src/low_rank_validation.rs
```

Pour :

\[
W = UV^T + E
\]

le validateur estime si la composante structurée possède effectivement une dimension réduite.

Une métrique utile est le rapport d'énergie :

\[
R_k =
\frac{
\sum_{i=1}^{k}\sigma_i^2
}{
\sum_i\sigma_i^2
}
\]

où \(\sigma_i\) sont les valeurs singulières.

---

# 5.10. ÉTAPE 13.8 — VALIDATION DES OUTLIERS

### Fichier

```text
crates/pmg-validation/src/outlier_validation.rs
```

Vérifier :

- fréquence ;
- magnitude ;
- localisation ;
- concentration par lignes/colonnes ;
- cohérence avec le profil prévu.

Une règle simple peut utiliser :

\[
|x-\mu| > k\sigma
\]

mais PMG ne doit pas dépendre exclusivement de ce seuil.

---

# 5.11. ÉTAPE 13.9 — SCORE GLOBAL

### Fichier

```text
crates/pmg-validation/src/score.rs
```

Exemple :

\[
S =
w_sS_s +
w_dS_d +
w_cS_c +
w_rS_r +
w_oS_o
\]

avec :

- \(S_s\) = score structurel ;
- \(S_d\) = score distributionnel ;
- \(S_c\) = score corrélation ;
- \(S_r\) = score bas-rang ;
- \(S_o\) = score outliers.

Les poids \(w_i\) doivent être configurables.

---

# 5.12. ÉTAPE 13.10 — RAPPORT

### Fichier

```text
crates/pmg-validation/src/report.rs
```

Exemple utilisateur :

```text
╔══════════════════════════════════════════╗
║          RAPPORT DE VALIDATION PMG       ║
╠══════════════════════════════════════════╣
║ Structure             PASS               ║
║ Shapes                PASS               ║
║ DTypes                PASS               ║
║ Distribution          PASS               ║
║ Corrélation           WARN               ║
║ Bas-rang              PASS               ║
║ Outliers              PASS               ║
╠══════════════════════════════════════════╣
║ Score global          94.7 %              ║
╚══════════════════════════════════════════╝
```

---

# 6. SPRINT 14 — INSPECTION DES MODÈLES

**Durée indicative : 2 semaines**

**Responsabilité unique :**

> Fournir une vue détaillée d'un modèle sans charger inutilement les poids complets.

---

# 6.1. Objectif fondamental

La commande :

```bash
pmg espec model/
```

doit permettre de comprendre un modèle à partir de :

- configuration ;
- headers ;
- index ;
- metadata ;
- shapes ;
- dtypes ;
- tailles ;
- architecture.

Elle ne doit pas charger systématiquement tous les fichiers de poids.

---

# 6.2. Structure

```text
SPRINT 14
│
├── Étape 14.1 : Inspecteur principal
├── Étape 14.2 : Lecture de configuration
├── Étape 14.3 : Lecture des headers Safetensors
├── Étape 14.4 : Indexation des tenseurs
├── Étape 14.5 : Statistiques structurelles
├── Étape 14.6 : Statistiques physiques
├── Étape 14.7 : Résumé architectural
├── Étape 14.8 : Formatage CLI
└── Étape 14.9 : Tests d'inspection
```

---

# 6.3. ÉTAPE 14.1 — INSPECTEUR PRINCIPAL

### Fichier

```text
crates/pmg-inspect/src/inspector.rs
```

Responsabilité :

```rust
pub struct ModelInspector;
```

Il coordonne les différentes sources de métadonnées.

---

# 6.4. ÉTAPE 14.2 — CONFIGURATION

### Fichier

```text
crates/pmg-inspect/src/config_inspector.rs
```

Extraire :

```text
architecture
hidden_size
num_layers
num_attention_heads
num_key_value_heads
intermediate_size
vocab_size
dtype
```

Le système doit accepter les différences de noms entre architectures.

---

# 6.5. ÉTAPE 14.3 — HEADERS SAFETENSORS

### Fichier

```text
crates/pmg-inspect/src/safetensors_inspector.rs
```

Le but est de lire le header sans charger les données.

Pour un tenseur :

```text
name
dtype
shape
data_offsets
```

On peut calculer :

\[
N = \prod_i shape_i
\]

et estimer :

\[
Size \approx N \times bytes(dtype)
\]

avec les précautions nécessaires pour les formats sous-octets.

---

# 6.6. ÉTAPE 14.4 — INDEXATION

### Fichier

```text
crates/pmg-inspect/src/index_inspector.rs
```

Responsabilité :

Construire :

```text
tensor → fichier shard
```

Exemple :

```text
model.layers.0.self_attn.q_proj.weight
        ↓
model-00001-of-00008.safetensors
```

---

# 6.7. ÉTAPE 14.5 — STATISTIQUES STRUCTURELLES

### Fichier

```text
crates/pmg-inspect/src/structural_stats.rs
```

Produire :

- nombre de tenseurs ;
- nombre de couches ;
- nombre de shards ;
- nombre d'experts ;
- dimensions ;
- paramètres théoriques.

---

# 6.8. ÉTAPE 14.6 — STATISTIQUES PHYSIQUES

### Fichier

```text
crates/pmg-inspect/src/physical_stats.rs
```

Calculer :

\[
Memory \approx \sum_i Size_i
\]

et éventuellement :

```text
taille brute
taille théorique
dtype
densité
répartition par couche
répartition par shard
```

---

# 6.9. ÉTAPE 14.7 — RÉSUMÉ ARCHITECTURAL

### Fichier

```text
crates/pmg-inspect/src/architecture.rs
```

Produire une représentation humaine :

```text
Architecture : Transformer
Couches      : 80
Hidden size  : 8192
Attention    : 64 heads
KV heads     : 8
Experts      : 64
```

Les champs non disponibles doivent être explicitement indiqués comme inconnus.

---

# 6.10. ÉTAPE 14.8 — FORMATAGE CLI

### Fichier

```text
crates/pmg-inspect/src/display.rs
```

Prévoir :

```text
--brief
--verbose
--debug
```

Exemple :

```bash
pmg espec model/
```

et :

```bash
pmg espec model/ --verbose
```

---

# 6.11. ÉTAPE 14.9 — TESTS

### Fichier

```text
crates/pmg-inspect/tests/inspection_tests.rs
```

Tests :

```text
test_config_inspection
test_header_inspection
test_shard_index
test_tensor_count
test_parameter_estimation
test_no_weight_loading
```

Le dernier test est particulièrement important.

---

# 7. SPRINT 15 — COMPARAISON DES MODÈLES

**Durée indicative : 2 semaines**

**Responsabilité unique :**

> Comparer un modèle original et un pseudo-modèle uniquement à partir des informations accessibles sans téléchargement des poids complets.

---

# 7.1. Règle fondamentale

La commande :

```bash
pmg compare original/ pseudo/
```

ne doit **pas** faire une comparaison profonde des poids.

PMG compare :

```text
configuration
headers
metadata
shapes
dtypes
noms de tenseurs
structure
sharding
statistiques disponibles
```

---

# 7.2. Structure

```text
SPRINT 15
│
├── Étape 15.1 : Modèle de comparaison
├── Étape 15.2 : Comparaison configuration
├── Étape 15.3 : Comparaison architecture
├── Étape 15.4 : Comparaison tenseurs
├── Étape 15.5 : Comparaison shapes
├── Étape 15.6 : Comparaison dtypes
├── Étape 15.7 : Comparaison sharding
├── Étape 15.8 : Détection des différences
├── Étape 15.9 : Score de similarité structurelle
└── Étape 15.10 : Rapport de comparaison
```

---

# 7.3. ÉTAPE 15.1 — MODÈLE DE COMPARAISON

### Fichier

```text
crates/pmg-compare/src/comparison.rs
```

Définir :

```rust
pub struct ComparisonReport {
    pub configuration: ComparisonStatus,
    pub architecture: ComparisonStatus,
    pub tensors: ComparisonStatus,
    pub shapes: ComparisonStatus,
    pub dtypes: ComparisonStatus,
    pub sharding: ComparisonStatus,
}
```

---

# 7.4. ÉTAPE 15.2 — CONFIGURATION

### Fichier

```text
crates/pmg-compare/src/config_compare.rs
```

Comparer :

```text
vocab_size
hidden_size
num_layers
num_heads
num_experts
intermediate_size
```

---

# 7.5. ÉTAPE 15.3 — ARCHITECTURE

### Fichier

```text
crates/pmg-compare/src/architecture_compare.rs
```

Déterminer :

```text
IDENTIQUE
COMPATIBLE
DIFFÉRENTE
INCONNUE
```

---

# 7.6. ÉTAPE 15.4 — TENSEURS

### Fichier

```text
crates/pmg-compare/src/tensor_compare.rs
```

Comparer les noms.

Exemple :

```text
original:
  layer.0.q_proj.weight
  layer.0.k_proj.weight

pseudo:
  layer.0.q_proj.weight
  layer.0.k_proj.weight
```

Résultat :

```text
2/2 tenseurs présents
```

---

# 7.7. ÉTAPE 15.5 — SHAPES

### Fichier

```text
crates/pmg-compare/src/shape_compare.rs
```

Vérification :

\[
shape_{original} = shape_{pseudo}
\]

---

# 7.8. ÉTAPE 15.6 — DTYPES

### Fichier

```text
crates/pmg-compare/src/dtype_compare.rs
```

Exemple :

```text
original : BF16
pseudo    : BF16
→ MATCH
```

---

# 7.9. ÉTAPE 15.7 — SHARDING

### Fichier

```text
crates/pmg-compare/src/shard_compare.rs
```

Comparer :

```text
nombre de shards
mapping tensor → shard
taille des shards
```

Le nombre de shards peut être différent sans que la structure logique soit nécessairement incorrecte.

---

# 7.10. ÉTAPE 15.8 — DIFFÉRENCES

### Fichier

```text
crates/pmg-compare/src/diff.rs
```

Exemple :

```text
+ tensor présent uniquement dans pseudo
- tensor absent du pseudo
~ shape différente
~ dtype différent
```

---

# 7.11. ÉTAPE 15.9 — SCORE STRUCTUREL

### Fichier

```text
crates/pmg-compare/src/score.rs
```

Exemple :

\[
S =
\frac{
N_{match}
}{
N_{total}
}
\times 100
\]

Mais le score ne doit jamais masquer les erreurs critiques.

Un modèle peut obtenir :

```text
98 %
```

tout en ayant une dimension critique incorrecte.

Le rapport doit donc toujours afficher les anomalies bloquantes séparément.

---

# 7.12. ÉTAPE 15.10 — RAPPORT

### Fichier

```text
crates/pmg-compare/src/report.rs
```

Exemple :

```text
╔══════════════════════════════════════════╗
║           COMPARAISON PMG                ║
╠══════════════════════════════════════════╣
║ Configuration       MATCH                ║
║ Architecture        MATCH                ║
║ Tenseurs             100 %               ║
║ Shapes               100 %               ║
║ DTypes               100 %               ║
║ Sharding              92 %               ║
╠══════════════════════════════════════════╣
║ Similarité structurelle : 97.8 %         ║
╚══════════════════════════════════════════╝

Aucune lecture profonde des poids effectuée.
```

---

# 8. SPRINT 16 — INTERFACE CLI COMPLÈTE

**Durée indicative : 2 à 3 semaines**

**Responsabilité unique :**

> Transformer toutes les capacités internes de PMG en une interface CLI française cohérente et accessible.

---

# 8.1. Commandes officielles

Le CLI final doit exposer :

```text
pmg help
pmg generate
pmg espec
pmg validate
pmg compare
pmg version
```

---

# 8.2. Structure

```text
SPRINT 16
│
├── Étape 16.1 : Structure CLI
├── Étape 16.2 : Commande help
├── Étape 16.3 : Commande generate
├── Étape 16.4 : Commande espec
├── Étape 16.5 : Commande validate
├── Étape 16.6 : Commande compare
├── Étape 16.7 : Commande version
├── Étape 16.8 : Gestion des flags
├── Étape 16.9 : Affichage des erreurs
├── Étape 16.10 : Codes de sortie
└── Étape 16.11 : Tests E2E CLI
```

---

# 8.3. ÉTAPE 16.1 — STRUCTURE CLI

### Fichier

```text
crates/pmg-cli/src/cli.rs
```

Définir la structure Clap.

Conceptuellement :

```rust
#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}
```

---

# 8.4. ÉTAPE 16.2 — HELP

### Fichier

```text
crates/pmg-cli/src/commands/help.rs
```

Le mode débutant doit expliquer :

```text
1. ce qu'est PMG ;
2. ce qu'est un pseudo-modèle ;
3. comment générer ;
4. comment inspecter ;
5. comment valider ;
6. comment comparer ;
7. comment utiliser --dry-run.
```

Exemple :

```bash
pmg help generate
```

---

# 8.5. ÉTAPE 16.3 — GENERATE

### Fichier

```text
crates/pmg-cli/src/commands/generate.rs
```

Exemple :

```bash
pmg generate \
  --template model.json \
  --output ./output
```

Avec :

```bash
pmg generate ... --dry-run
```

Le dry-run doit montrer ce qui serait fait sans produire le modèle final.

---

# 8.6. ÉTAPE 16.4 — ESPEC

### Fichier

```text
crates/pmg-cli/src/commands/espec.rs
```

Responsabilité :

Inspection du modèle.

Exemple :

```bash
pmg espec ./model
```

---

# 8.7. ÉTAPE 16.5 — VALIDATE

### Fichier

```text
crates/pmg-cli/src/commands/validate.rs
```

Exemple :

```bash
pmg validate ./pseudo-model
```

Résultat :

```text
Validation terminée.

Structure : PASS
Statistiques : PASS
Distribution : PASS
Corrélation : WARN
Outliers : PASS

Score : 94.7 %
```

---

# 8.8. ÉTAPE 16.6 — COMPARE

### Fichier

```text
crates/pmg-cli/src/commands/compare.rs
```

Exemple :

```bash
pmg compare ./original ./pseudo
```

Avec garantie explicite :

```text
Mode de comparaison :
métadonnées + headers + configuration

Lecture profonde des poids :
NON
```

---

# 8.9. ÉTAPE 16.7 — VERSION

### Fichier

```text
crates/pmg-cli/src/commands/version.rs
```

Exemple :

```text
PMG — Pseudo-Models Generator

Version       : 1.0.0
Build         : release
Rust          : stable
Architecture  : x86_64
Licence       : GPL-3.0
```

---

# 8.10. ÉTAPE 16.8 — FLAGS

### Fichier

```text
crates/pmg-cli/src/options.rs
```

Flags officiels :

```text
-h, --help
-d, --dry-run
--debug
-b, --verbose
```

### Correction importante

Dans la spécification initiale, `-h` était attribué deux fois :

```text
-h, --help
-h, --debug
```

Cela est impossible proprement avec Clap.

La version définitive doit donc utiliser :

```text
-h, --help
--debug
-b, --verbose
-d, --dry-run
```

---

# 8.11. ÉTAPE 16.9 — ERREURS CLI

### Fichier

```text
crates/pmg-cli/src/error_display.rs
```

Objectif :

Transformer une erreur interne :

```text
HeaderReserveExceeded(10485760)
```

en message compréhensible :

```text
Erreur : la réserve de l'en-tête est insuffisante.

Taille nécessaire : 10.5 MiB
Taille disponible : 10.0 MiB

Conseil :
utilisez une réserve d'en-tête plus grande.
```

---

# 8.12. ÉTAPE 16.10 — CODES DE SORTIE

### Fichier

```text
crates/pmg-cli/src/exit_codes.rs
```

Exemple :

```text
0 → succès
1 → erreur générale
2 → argument invalide
3 → modèle invalide
4 → erreur d'I/O
5 → validation échouée
6 → comparaison incompatible
```

---

# 8.13. ÉTAPE 16.11 — TESTS E2E

### Fichier

```text
crates/pmg-cli/tests/cli_tests.rs
```

Tester :

```text
pmg --help
pmg help
pmg generate --help
pmg espec --help
pmg validate --help
pmg compare --help
pmg version
pmg generate --dry-run
```

---

# 9. SPRINT 17 — DURCISSEMENT, AUDIT ET RELEASE CANDIDATE

**Durée indicative : 2 à 3 semaines**

**Responsabilité unique :**

> Transformer l'ensemble du projet en version stable candidate à la publication.

---

# 9.1. Objectif

Le Sprint 17 n'ajoute pas de grosse fonctionnalité.

Il cherche à répondre à :

> "PMG est-il suffisamment robuste pour être distribué ?"

---

# 9.2. Structure

```text
SPRINT 17
│
├── Étape 17.1 : Audit des erreurs
├── Étape 17.2 : Audit des panics
├── Étape 17.3 : Audit unsafe
├── Étape 17.4 : Audit des dépendances
├── Étape 17.5 : Audit licence
├── Étape 17.6 : Audit des fichiers > 500 lignes
├── Étape 17.7 : Audit documentation
├── Étape 17.8 : Test reproductibilité
├── Étape 17.9 : Test grandes configurations
├── Étape 17.10 : Test corruption fichiers
├── Étape 17.11 : Test performances
├── Étape 17.12 : Build multi-plateformes
├── Étape 17.13 : Release Candidate
└── Étape 17.14 : Validation finale
```

---

# 9.3. ÉTAPE 17.1 — AUDIT DES ERREURS

### Fichier

```text
docs/audit_erreurs.md
```

Chercher :

```text
unwrap()
expect()
panic!()
todo!()
unimplemented!()
```

Chaque occurrence doit être justifiée.

---

# 9.4. ÉTAPE 17.2 — AUDIT DES PANICS

### Fichier

```text
docs/audit_panics.md
```

Objectif :

Un fichier utilisateur corrompu ne doit pas provoquer un crash brutal.

Mauvais :

```text
thread 'main' panicked at ...
```

Préféré :

```text
Erreur PMG : le fichier Safetensors est invalide.

Cause :
l'offset de fin dépasse la taille du fichier.

Action :
vérifiez l'intégrité du fichier.
```

---

# 9.5. ÉTAPE 17.3 — AUDIT UNSAFE

### Fichier

```text
docs/audit_unsafe.md
```

Commande utile :

```bash
grep -R "unsafe" crates/
```

Chaque `unsafe` doit disposer :

- d'une justification ;
- d'un invariant ;
- d'une API sûre autour ;
- d'un test.

---

# 9.6. ÉTAPE 17.4 — AUDIT DES DÉPENDANCES

### Fichier

```text
docs/audit_dependances.md
```

Vérifier :

```bash
cargo tree
cargo audit
cargo outdated
```

Objectifs :

- détecter les vulnérabilités ;
- supprimer les dépendances inutilisées ;
- vérifier les licences ;
- limiter les dépendances lourdes.

---

# 9.7. ÉTAPE 17.5 — AUDIT GPL-3.0

### Fichier

```text
LICENSE
```

Vérifier que :

```text
Cargo.toml
README
LICENSE
documentation
package metadata
```

indiquent correctement :

```text
GPL-3.0
```

### Point critique

La documentation précédente contenait parfois une incohérence entre MIT et GPL-3.0.

La décision définitive du projet est :

> **PMG est distribué sous GPL-3.0.**

Cette décision doit être uniforme dans tout le dépôt.

---

# 9.8. ÉTAPE 17.6 — AUDIT LIMITE 500 LIGNES

### Fichier

```text
scripts/check_file_size.sh
```

Le script doit détecter les fichiers Rust dépassant la limite.

Conceptuellement :

```text
pour chaque *.rs :
    compter les lignes de code
    si > 500 :
        erreur
```

La règle de 500 lignes reste une règle architecturale de PMG.

---

# 9.9. ÉTAPE 17.7 — AUDIT DOCUMENTATION

### Fichier

```text
docs/audit_documentation.md
```

Vérifier :

```text
README
Cahier des Besoins
Cahier Fonctionnel
Cahier Technique
Cahier des Charges
Cahier des Piliers
Cahier de Développement
Cahier du Plan de Développement
```

et :

```bash
cargo doc --no-deps
```

---

# 9.10. ÉTAPE 17.8 — TEST DE REPRODUCTIBILITÉ

### Fichier

```text
crates/pmg-generator/tests/reproducibility_tests.rs
```

Test :

```text
Génération A
seed = 12345

Génération B
seed = 12345
```

On vérifie :

\[
A = B
\]

au niveau des données générées ou d'une représentation déterministe définie.

Avec :

```text
seed = 12345
```

puis :

```text
seed = 54321
```

on doit obtenir une sortie différente avec une probabilité correspondant au générateur et au pipeline utilisés.

---

# 9.11. ÉTAPE 17.9 — GRANDES CONFIGURATIONS

### Fichier

```text
tests/stress/large_model_tests.rs
```

Tester des configurations importantes :

```text
nombre élevé de couches
nombre élevé de tenseurs
MoE
nombre élevé d'experts
grandes dimensions
plusieurs shards
```

Le test doit vérifier la stabilité sans nécessiter nécessairement de générer plusieurs centaines de gigaoctets.

---

# 9.12. ÉTAPE 17.10 — FICHIERS CORROMPUS

### Fichier

```text
tests/corruption/corrupted_files_tests.rs
```

Cas :

```text
header tronqué
JSON invalide
offset invalide
shape invalide
dtype inconnu
fichier vide
fichier incomplet
index incohérent
```

Attendu :

```text
Erreur contrôlée
```

et non :

```text
panic
```

---

# 9.13. ÉTAPE 17.11 — PERFORMANCES

### Fichier

```text
benches/final_benchmarks.rs
```

Mesurer notamment :

```text
inspection
parsing
génération
packing
écriture
validation
comparaison
```

Les benchmarks ne doivent pas seulement mesurer le temps.

Ils doivent également surveiller :

```text
mémoire
allocation
débit
taille des sorties
```

---

# 9.14. ÉTAPE 17.12 — BUILD MULTI-PLATEFORME

### Fichier

```text
.github/workflows/release.yml
```

Cibles minimales :

```text
Linux x86_64
Windows x86_64
macOS x86_64
macOS ARM64
```

Le build release doit être reproductible autant que possible.

---

# 9.15. ÉTAPE 17.13 — RELEASE CANDIDATE

### Fichier

```text
CHANGELOG.md
```

Créer :

```text
v1.0.0-rc.1
```

Le processus :

```text
develop
   ↓
release/v1.0.0-rc.1
   ↓
CI
   ↓
Tests
   ↓
Audit
   ↓
Build
   ↓
RC
```

---

# 9.16. ÉTAPE 17.14 — VALIDATION FINALE

### Fichier

```text
docs/release_checklist.md
```

Checklist :

```text
[ ] cargo fmt --all -- --check
[ ] cargo clippy --all-targets --all-features -- -D warnings
[ ] cargo test --workspace
[ ] cargo doc --workspace --no-deps
[ ] cargo audit
[ ] cargo build --release
[ ] tests E2E
[ ] tests de corruption
[ ] tests de reproductibilité
[ ] tests de grandes configurations
[ ] benchmarks
[ ] vérification GPL-3.0
[ ] vérification README
[ ] vérification CHANGELOG
[ ] vérification version
[ ] vérification CLI
[ ] vérification des fichiers > 500 lignes
[ ] vérification absence de secrets
```

---

# 10. MATRICE GLOBALE DES SPRINTS 12 À 17

| Sprint | Responsabilité unique | Résultat |
|---|---|---|
| 12 | Orchestration | Pipeline complet de génération |
| 13 | Validation | Validation scientifique/statistique |
| 14 | Inspection | Analyse sans chargement complet |
| 15 | Comparaison | Comparaison structurelle |
| 16 | CLI | Interface utilisateur complète |
| 17 | Durcissement | Release Candidate |

---

# 11. CHAÎNE TECHNIQUE COMPLÈTE

À la fin du Sprint 17, la chaîne principale doit être :

```text
                    ┌──────────────────┐
                    │    Blueprint     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ GenerationConfig │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Seed Management  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Distribution    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Corrélation    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Bas-rang     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Outliers      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Super-poids    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Tensor Generator │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Safetensors I/O  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Pseudo-Modèle    │
                    └────────┬─────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        ┌─────────┐    ┌───────────┐    ┌──────────┐
        │ Inspect │    │ Validate  │    │ Compare  │
        └─────────┘    └───────────┘    └──────────┘
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                       ┌──────────┐
                       │   CLI    │
                       └──────────┘
```

---

# 12. DÉFINITION DES CRITÈRES DE FIN DE PHASE

Les Sprints 12 à 17 ne sont considérés comme terminés que si PMG respecte simultanément les conditions suivantes.

## 12.1. Fonctionnel

```text
[✓] generate
[✓] espec
[✓] validate
[✓] compare
[✓] version
[✓] help
```

---

## 12.2. Scientifique

```text
[✓] distributions
[✓] corrélations
[✓] structures bas-rang
[✓] outliers
[✓] super-poids
[✓] statistiques
```

---

## 12.3. Structurel

```text
[✓] shapes
[✓] dtypes
[✓] tenseurs
[✓] couches
[✓] shards
[✓] configuration
```

---

## 12.4. Qualité logicielle

```text
[✓] tests unitaires
[✓] tests intégration
[✓] tests E2E
[✓] benchmarks
[✓] clippy
[✓] rustfmt
[✓] documentation
```

---

## 12.5. Sécurité

```text
[✓] cargo audit
[✓] audit unsafe
[✓] audit unwrap/expect
[✓] gestion des fichiers corrompus
[✓] aucun secret dans le dépôt
```

---

# 13. GUIDE DE TRAVAIL POUR IBRAHIMA-224

Puisqu'il n'y a qu'un seul développeur, le principal risque n'est pas le manque de coordination entre développeurs.

Le principal risque est la **complexité cognitive**.

Il faut donc travailler en petites unités.

Pour chaque étape :

```text
1. Lire la responsabilité de l'étape.
2. Lire les interfaces des crates dépendantes.
3. Écrire les tests.
4. Implémenter.
5. Formater.
6. Linter.
7. Tester.
8. Documenter.
9. Vérifier la limite de 500 lignes.
10. Commit.
```

Cycle recommandé :

```bash
cargo fmt --all
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
git status
git diff
git commit
```

---

# 14. RÈGLE IMPORTANTE : NE PAS SAUTER LES ÉTAPES

Le développeur ne doit pas faire :

```text
Sprint 12 → coder rapidement → Sprint 16 → corriger les problèmes
```

Mais :

```text
Sprint 12
   ↓
tests
   ↓
validation
   ↓
Sprint 13
   ↓
tests
   ↓
validation
   ↓
Sprint 14
   ↓
...
```

Cela réduit considérablement le risque d'accumuler une dette technique.

---

# 15. RÈGLE DE COMMIT

Chaque étape doit idéalement produire un commit identifiable.

Exemples :

```text
feat(pmg-generator): add generation orchestrator
feat(pmg-generator): add deterministic seed context
feat(pmg-generator): add tensor generation pipeline
feat(pmg-validation): add statistical validator
feat(pmg-validation): add correlation validation
feat(pmg-inspect): add safetensors header inspector
feat(pmg-compare): add structural comparison
feat(pmg-cli): add generate command
feat(pmg-cli): add validate command
chore: prepare v1.0.0-rc.1
```

---

# 16. GESTION DES RISQUES

| Risque | Probabilité | Impact | Réponse |
|---|---:|---:|---|
| Pipeline trop complexe | Élevée | Élevé | Interfaces minimales |
| Mauvais ordre mathématique | Moyenne | Très élevé | Tests dédiés |
| Génération non reproductible | Moyenne | Élevé | Seed hiérarchique |
| Validation trop permissive | Moyenne | Très élevé | Tests négatifs |
| Lecture mémoire excessive | Moyenne | Élevé | Streaming |
| CLI trop complexe | Moyenne | Moyen | Commandes simples |
| Dépendances lourdes | Faible | Moyen | Audit Cargo |
| Fichiers >500 lignes | Moyenne | Moyen | Refactorisation |
| Régression | Élevée | Élevé | CI obligatoire |
| Modèle architectural non supporté | Moyenne | Élevé | Blueprint explicite |
| Corruption Safetensors | Moyenne | Élevé | Tests de corruption |
| Différences de configuration | Élevée | Moyen | Normalisation |

---

# 17. JALON MAJEUR — PMG V1.0 RC

À la fin du Sprint 17, PMG doit être considéré comme :

> **Feature Complete + Test Complete + Documentation Complete + Release Candidate**

et non simplement comme :

> "Le code compile."

La compilation n'est qu'un premier niveau de validation.

La définition de terminé est :

\[
DefinitionOfDone =
Code +
Tests +
Documentation +
Validation +
Reproductibilité +
Sécurité +
Packaging
\]

---

# 18. PHASE SUIVANTE

Après le Sprint 17, le développement peut entrer dans une phase distincte :

```text
SPRINT 18+
──────────

Amélioration scientifique
        ↓
Nouvelles architectures
        ↓
Calibration sur modèles réels
        ↓
Optimisation performances
        ↓
Support de nouveaux formats
        ↓
Amélioration UX
        ↓
Publications / releases
```

Ces travaux ne doivent pas être mélangés avec la stabilisation de PMG v1.0.

---

# 19. RÉFÉRENCES TECHNIQUES PRINCIPALES

Les références utilisées pour l'implémentation doivent prioritairement être les documentations officielles des technologies concernées :

- **Rust Book** — apprentissage et conception idiomatique Rust.
- **Rust Reference** — sémantique du langage.
- **Cargo Reference** — workspaces, packages et dépendances.
- **Clippy** — analyse statique Rust.
- **rustfmt** — formatage.
- **Clap** — construction de l'interface CLI.
- **Serde** — sérialisation/désérialisation.
- **Safetensors** — structure des fichiers de poids et headers.
- **Rand** — génération pseudo-aléatoire.
- **Rayon** — parallélisme CPU.
- **Criterion** — benchmarks.
- **GitHub Actions** — CI/CD.
- **GPL-3.0** — licence du projet.

Pour les parties scientifiques :

- théorie des probabilités ;
- statistiques mathématiques ;
- analyse multivariée ;
- théorie des matrices ;
- décomposition en valeurs singulières ;
- distributions à queues lourdes ;
- tests d'adéquation ;
- génération pseudo-aléatoire déterministe.

---

# 20. CONCLUSION

Les Sprints 12 à 17 constituent la transition entre un ensemble de composants techniques et un **logiciel PMG utilisable**.

La progression est volontairement linéaire :

```text
S12
Orchestrer
   ↓
S13
Valider
   ↓
S14
Inspecter
   ↓
S15
Comparer
   ↓
S16
Exposer
   ↓
S17
Durcir
```

À l'issue de cette séquence, PMG doit disposer d'un cycle logiciel complet :

```text
          ┌──────────────┐
          │  Blueprint   │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │   Generate   │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │   Validate   │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │    Espec     │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │   Compare    │
          └──────┬───────┘
                 ↓
          ┌──────────────┐
          │   Release    │
          └──────────────┘
```

**Le Sprint 17 constitue ainsi le point de passage vers PMG v1.0 Release Candidate.**