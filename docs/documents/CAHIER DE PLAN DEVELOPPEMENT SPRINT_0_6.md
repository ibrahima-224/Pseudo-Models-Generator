# CAHIER DE PLAN DE DÉVELOPPEMENT — PMG

## Partie I — Sprints 0 à 5

**Projet :** Pseudo-Models Generator — PMG
**Version du cahier :** 1.0
**Périmètre :** Sprints 0 à 5
**Développeur unique :** Ibrahima-224
**Licence du logiciel :** GPL-3.0
**Langage :** Rust
**Interface :** CLI riche, entièrement francophone
**Philosophie :** CPU-first, faible empreinte mémoire, aucune dépendance ML lourde
**Statut :** Plan directeur de développement

---

# 1. OBJET DU PRÉSENT CAHIER

Ce document décrit **comment PMG doit être construit**, et non simplement ce qu'il doit faire.

La période couverte comprend les six premiers sprints :

| Sprint | Responsabilité unique           | Résultat principal                                       |
| ------ | ------------------------------- | -------------------------------------------------------- |
| **0**  | Initialisation du projet        | Workspace Rust propre et reproductible                   |
| **1**  | Fondations du Core              | Types et invariants fondamentaux                         |
| **2**  | Modélisation Blueprint          | Description abstraite d'un pseudo-modèle                 |
| **3**  | Moteur mathématique/statistique | RNG, distributions et statistiques                       |
| **4**  | Moteur d'injection              | Super-poids, corrélation, bas-rang et structures         |
| **5**  | Génération déterministe         | Première chaîne complète Blueprint → tenseurs → artefact |

L'objectif n'est **pas** d'obtenir dès le Sprint 5 un PMG final complet.

L'objectif est d'obtenir un **socle techniquement fiable** sur lequel les fonctionnalités avancées des Sprints 6–11 pourront être construites.

---

# 2. RÈGLE FONDAMENTALE DU PLAN

## 2.1. Un développeur = une séquence de responsabilités

Le projet est développé par une seule personne :

> **Ibrahima-224**

Il ne faut donc pas organiser artificiellement le projet comme si plusieurs développeurs travaillaient simultanément.

Chaque sprint possède :

* une seule responsabilité principale ;
* un objectif mesurable ;
* un ordre strict d'implémentation ;
* des étapes atomiques ;
* un fichier principal par étape ;
* des tests associés ;
* une validation de sortie.

### Principe

```text
SPRINT
   │
   ├── Responsabilité unique
   │
   ├── Étape 1 → fichier 1
   ├── Étape 2 → fichier 2
   ├── Étape 3 → fichier 3
   ├── ...
   │
   └── Validation du Sprint
```

Une étape ne doit pas essayer de résoudre deux problèmes indépendants.

---

# 3. RÈGLE « UNE ÉTAPE = UNE RESPONSABILITÉ = UN FICHIER »

Cette règle devient obligatoire.

Par exemple :

### Mauvais découpage

```text
Étape : créer le moteur mathématique

math.rs
 ├── RNG
 ├── Student-t
 ├── Weibull
 ├── Pareto
 ├── covariance
 ├── SVD
 └── outliers
```

C'est trop large.

### Bon découpage

```text
3.1 → rng.rs
3.2 → distribution.rs
3.3 → student_t.rs
3.4 → weibull.rs
3.5 → pareto.rs
3.6 → statistics.rs
3.7 → covariance.rs
3.8 → low_rank.rs
```

Chaque fichier possède une responsabilité clairement identifiable.

Cette organisation facilite également le respect de la limite de **500 lignes par fichier**.

---

# 4. ARCHITECTURE CIBLE DES SPRINTS 0–5

À la fin du Sprint 5, l'architecture visée est :

```text
Pseudo-Models-Generator/
│
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── README.md
├── CHANGELOG.md
├── .gitignore
├── .rustfmt.toml
├── .clippy.toml
├── .editorconfig
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docs/
│   ├── cahier_besoins.md
│   ├── cahier_fonctionnel.md
│   ├── cahier_technique.md
│   ├── cahier_piliers.md
│   ├── cahier_charges.md
│   ├── cahier_developpement.md
│   └── cahier_plan_developpement.md
│
└── crates/
    │
    ├── pmg-core/
    │   └── src/
    │
    ├── pmg-blueprint/
    │   └── src/
    │
    ├── pmg-math/
    │   └── src/
    │
    ├── pmg-injector/
    │   └── src/
    │
    ├── pmg-io/
    │   └── src/
    │
    └── pmg-cli/
        └── src/
```

### Remarque importante

Un éventuel `pmg-gpu` ne doit **pas** être développé pendant les Sprints 0–5.

Il serait prématuré d'introduire une couche GPU avant d'avoir :

1. défini les structures ;
2. validé les mathématiques ;
3. mesuré les performances CPU ;
4. identifié les véritables hotspots.

---

# 5. ORDRE GLOBAL DE CONSTRUCTION

```text
SPRINT 0
Fondations
   ↓
SPRINT 1
Core
   ↓
SPRINT 2
Blueprint
   ↓
SPRINT 3
Mathématiques
   ↓
SPRINT 4
Injection
   ↓
SPRINT 5
Génération déterministe
   ↓
────────────────────────────
Sprints 6–11
Fonctionnalités avancées
```

La dépendance est volontairement descendante :

```text
CLI
 ↓
Generation
 ↓
Injection
 ↓
Math
 ↓
Blueprint
 ↓
Core
```

Le Core ne doit dépendre d'aucune couche supérieure.

---

# SPRINT 0 — INITIALISATION ET INFRASTRUCTURE

## Responsabilité unique

> **Construire un environnement Rust reproductible, propre et contrôlable.**

**Durée cible :** 2 semaines maximum.

---

## 0.1 Objectif du Sprint

À la fin du Sprint 0 :

* Cargo doit reconnaître le workspace ;
* les crates doivent compiler ;
* Git doit être correctement configuré ;
* rustfmt doit fonctionner ;
* Clippy doit fonctionner ;
* la CI minimale doit fonctionner ;
* la licence GPL-3.0 doit être présente ;
* les conventions du projet doivent être matérialisées dans les fichiers.

Cargo fournit notamment le mécanisme officiel de workspace permettant de gérer plusieurs packages Rust au sein d'un même projet. ([Documentation Rust][2])

---

## 0.2 Structure

```text
SPRINT 0
│
├── 0.1 → Cargo.toml
├── 0.2 → .gitignore
├── 0.3 → .rustfmt.toml
├── 0.4 → .clippy.toml
├── 0.5 → .editorconfig
├── 0.6 → LICENSE
├── 0.7 → README.md
├── 0.8 → ci.yml
├── 0.9 → pmg-core/Cargo.toml
├── 0.10 → pmg-blueprint/Cargo.toml
├── 0.11 → pmg-math/Cargo.toml
├── 0.12 → pmg-injector/Cargo.toml
├── 0.13 → pmg-io/Cargo.toml
└── 0.14 → pmg-cli/Cargo.toml
```

---

# ÉTAPE 0.1 — WORKSPACE CARGO

**Fichier :**

```text
Cargo.toml
```

### Responsabilité

Définir exclusivement le workspace.

### Objectifs

Le fichier doit :

* déclarer les crates ;
* définir l'édition Rust ;
* définir la licence ;
* centraliser les versions communes ;
* éviter les dépendances dupliquées.

### Architecture logique

```toml
[workspace]
members = [
    "crates/pmg-core",
    "crates/pmg-blueprint",
    "crates/pmg-math",
    "crates/pmg-injector",
    "crates/pmg-io",
    "crates/pmg-cli",
]

resolver = "2"
```

### Dépendances prévues

Le principe est :

> **Une dépendance doit résoudre un problème concret.**

Par exemple :

* `serde` → sérialisation ;
* `serde_json` → JSON ;
* `thiserror` → erreurs des bibliothèques ;
* `anyhow` → erreurs du CLI ;
* `rand` → génération aléatoire ;
* `clap` → CLI.

Pas de framework ML.

### Point critique

Ne pas introduire immédiatement toutes les dépendances envisagées.

Le Cargo manifest doit rester minimal.

### Attente

```bash
cargo check --workspace
```

doit fonctionner.

### Référence

[Cargo Workspaces — documentation officielle](https://doc.rust-lang.org/cargo/reference/workspaces.html?utm_source=chatgpt.com)

---

# ÉTAPE 0.2 — GITIGNORE

**Fichier :**

```text
.gitignore
```

### Responsabilité

Empêcher les artefacts locaux et compilés d'entrer dans Git.

### Doit notamment ignorer

```text
/target/
*.log
*.tmp
*.swp
.env
.vscode/
.idea/
```

### Point critique

Ne jamais ignorer :

```text
Cargo.toml
Cargo.lock
src/
docs/
tests/
```

### Validation

```bash
git status --ignored
```

---

# ÉTAPE 0.3 — RUSTFMT

**Fichier :**

```text
.rustfmt.toml
```

### Responsabilité

Définir le formatage Rust.

Configuration cible :

```toml
edition = "2021"
max_width = 100
use_small_heuristics = "Default"
fn_args_layout = "Tall"
match_block_trailing_comma = true
```

### Validation

```bash
cargo fmt --all
cargo fmt --all -- --check
```

### Critère d'acceptation

Aucun changement produit par :

```bash
cargo fmt --all -- --check
```

---

# ÉTAPE 0.4 — CLIPPY

**Fichier :**

```text
.clippy.toml
```

### Responsabilité

Configurer le linting du projet.

### Règle

Le projet doit viser :

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Point critique

Les exceptions doivent être rares et justifiées.

Une exception ne doit jamais être utilisée simplement pour masquer une mauvaise conception.

---

# ÉTAPE 0.5 — EDITORCONFIG

**Fichier :**

```text
.editorconfig
```

### Responsabilité

Uniformiser les paramètres d'édition.

Exemple :

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
indent_style = space
indent_size = 4
trim_trailing_whitespace = true
```

---

# ÉTAPE 0.6 — LICENCE

**Fichier :**

```text
LICENSE
```

### Responsabilité

Déclarer la licence **GPL-3.0** du projet.

### Point critique

Le cahier de développement précédent contenait une incohérence entre GPL-3.0 et certaines formulations indiquant que les licences MIT/Apache seraient simplement « compatibles avec GPL-3.0 ».

Pour PMG, la règle opérationnelle doit être :

> **Toute nouvelle dépendance doit être vérifiée individuellement pour sa licence et ses obligations de redistribution.**

---

# ÉTAPE 0.7 — README

**Fichier :**

```text
README.md
```

### Responsabilité

Présenter le projet à un nouvel utilisateur ou développeur.

Le README doit expliquer :

```text
PMG
 │
 ├── Qu'est-ce que PMG ?
 ├── Pourquoi ?
 ├── Installation
 ├── Première commande
 ├── Architecture
 ├── Développement
 ├── Tests
 └── Licence
```

### Exemple débutant

```bash
cargo run -- generate --help
```

doit permettre à un débutant de comprendre immédiatement la commande.

---

# ÉTAPE 0.8 — CI

**Fichier :**

```text
.github/workflows/ci.yml
```

### Responsabilité

Automatiser les contrôles qualité.

### Pipeline minimal

```text
Push / PR
   │
   ├── cargo fmt --check
   ├── cargo check
   ├── cargo clippy
   ├── cargo test
   └── cargo doc
```

Les tests Rust utilisent le mécanisme intégré de test et les exemples `rustdoc` peuvent également être exécutés automatiquement comme tests. ([Documentation Rust][2])

### Point critique

La CI ne doit jamais être considérée comme décorative.

Un code qui ne passe pas la CI est un code non intégrable.

---

# ÉTAPES 0.9 À 0.14 — CRÉATION DES CRATES

Chaque crate possède son propre `Cargo.toml`.

| Étape | Fichier                    | Responsabilité             |
| ----- | -------------------------- | -------------------------- |
| 0.9   | `pmg-core/Cargo.toml`      | Configuration Core         |
| 0.10  | `pmg-blueprint/Cargo.toml` | Configuration Blueprint    |
| 0.11  | `pmg-math/Cargo.toml`      | Configuration mathématique |
| 0.12  | `pmg-injector/Cargo.toml`  | Configuration injection    |
| 0.13  | `pmg-io/Cargo.toml`        | Configuration I/O          |
| 0.14  | `pmg-cli/Cargo.toml`       | Configuration CLI          |

### Règle

Les dépendances doivent respecter :

```text
pmg-core
   ↑
pmg-blueprint
   ↑
pmg-math
   ↑
pmg-injector
   ↑
pmg-io
   ↑
pmg-cli
```

Le diagramme représente l'ordre conceptuel, pas nécessairement toutes les dépendances Cargo exactes.

---

# VALIDATION DU SPRINT 0

Le Sprint 0 est terminé uniquement lorsque :

```bash
cargo check --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

fonctionnent.

### Point fort

Infrastructure propre dès le début.

### Point faible

Peu de fonctionnalité visible.

### Risque critique

Passer trop rapidement à l'implémentation mathématique avant de stabiliser l'architecture.

---

# SPRINT 1 — FONDATIONS DU CORE

## Responsabilité unique

> **Définir les types fondamentaux et les invariants communs de PMG.**

**Durée cible :** 2 semaines.

---

# 1.1 Architecture

```text
SPRINT 1
│
├── 1.1 → dtype.rs
├── 1.2 → shape.rs
├── 1.3 → tensor_metadata.rs
├── 1.4 → model_config.rs
├── 1.5 → tensor_role.rs
├── 1.6 → error.rs
├── 1.7 → validation.rs
└── 1.8 → lib.rs
```

---

# ÉTAPE 1.1 — DTYPE

**Fichier :**

```text
crates/pmg-core/src/dtype.rs
```

### Responsabilité

Représenter les types numériques utilisés par PMG.

Exemples :

```rust
pub enum DType {
    F32,
    BF16,
    F16,
    F8E4M3,
    F8E8M0,
    I8,
    I64,
    FP4,
}
```

### Attention

`size_bytes()` ne doit pas être conçu naïvement pour tous les types.

Pour un format 4 bits :

[
\text{bytes} = \frac{N \times 4}{8}
]

soit :

[
\text{bytes} = \frac{N}{2}
]

avec gestion correcte des dimensions impaires.

### Test minimal

```rust
assert_eq!(DType::F32.size_bytes(), 4);
assert_eq!(DType::BF16.size_bytes(), 2);
```

---

# ÉTAPE 1.2 — SHAPE

**Fichier :**

```text
crates/pmg-core/src/shape.rs
```

### Responsabilité

Représenter les dimensions d'un tenseur.

Exemple :

```text
[4096, 4096]
```

correspond à :

[
N = 4096 \times 4096
]

éléments.

### API envisagée

```rust
pub struct Shape {
    dimensions: Vec<u64>,
}
```

### Invariants

Une dimension ne doit pas être négative.

Une shape vide doit être explicitement définie comme scalaire ou interdite selon le modèle retenu.

---

# ÉTAPE 1.3 — TENSOR METADATA

**Fichier :**

```text
crates/pmg-core/src/tensor_metadata.rs
```

### Responsabilité

Décrire un tenseur sans contenir ses données.

Exemple conceptuel :

```text
Nom :
model.layers.0.self_attn.q_proj.weight

DType :
BF16

Shape :
[4096, 4096]

Nombre d'éléments :
16 777 216
```

PMG doit conserver cette distinction fondamentale :

```text
TensorMetadata
       ≠
TensorData
```

Cela sera essentiel pour permettre l'inspection sans charger les poids.

Le format Safetensors stocke notamment les métadonnées nécessaires à l'identification des tenseurs et leurs données binaires séparément ; sa documentation officielle décrit également la possibilité d'accéder à des parties de tenseurs. ([Hugging Face][1])

---

# ÉTAPE 1.4 — MODEL CONFIG

**Fichier :**

```text
crates/pmg-core/src/model_config.rs
```

### Responsabilité

Définir la configuration architecturale abstraite.

Exemple :

```text
hidden_size = 4096
num_layers = 32
num_attention_heads = 32
intermediate_size = 11008
vocab_size = 128256
```

### Formule

Le nombre approximatif de paramètres d'une matrice :

[
P = \prod_{i=1}^{k} d_i
]

Exemple :

[
4096 \times 4096 = 16,777,216
]

---

# ÉTAPE 1.5 — TENSOR ROLE

**Fichier :**

```text
crates/pmg-core/src/tensor_role.rs
```

### Responsabilité

Identifier le rôle fonctionnel d'un tenseur.

Exemples :

```rust
pub enum TensorRole {
    Embedding,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    MlpUp,
    MlpDown,
    Router,
    Expert,
    Norm,
    Output,
    Unknown,
}
```

Cela permettra plus tard d'appliquer des distributions différentes selon le rôle.

---

# ÉTAPE 1.6 — ERREURS

**Fichier :**

```text
crates/pmg-core/src/error.rs
```

### Responsabilité

Centraliser les erreurs fondamentales.

Exemple :

```rust
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Dimension invalide")]
    InvalidDimension,

    #[error("Configuration de modèle invalide")]
    InvalidModelConfig,

    #[error("Type numérique non supporté")]
    UnsupportedDType,
}
```

---

# ÉTAPE 1.7 — VALIDATION

**Fichier :**

```text
crates/pmg-core/src/validation.rs
```

### Responsabilité

Vérifier les invariants du Core.

Exemples :

```text
num_heads > 0
num_layers > 0
hidden_size > 0
vocab_size > 0
```

Pour l'attention :

[
    d_{\text{head}}

\frac{d_{\text{model}}}{h}
]

Si :

[
d_{\text{model}}=4096
]

et :

[
h=32
]

alors :

[
d_{\text{head}}=128
]

Si la division n'est pas entière, la configuration doit être rejetée pour les architectures qui imposent cette contrainte.

---

# ÉTAPE 1.8 — LIB

**Fichier :**

```text
crates/pmg-core/src/lib.rs
```

### Responsabilité

Exposer l'API publique du Core.

Le `lib.rs` ne doit pas devenir un fichier métier gigantesque.

Il doit principalement organiser :

```rust
pub mod dtype;
pub mod error;
pub mod model_config;
pub mod shape;
pub mod tensor_metadata;
pub mod tensor_role;
pub mod validation;
```

---

# VALIDATION SPRINT 1

### Attentes

Le Core doit permettre de faire :

```text
Configuration
      ↓
Validation
      ↓
TensorMetadata
      ↓
TensorRole
      ↓
Calculs de dimensions
```

### Point fort

Tous les autres modules disposent enfin d'un langage commun.

### Point faible

Aucun vrai modèle n'est encore généré.

### Point critique

**Ne jamais laisser les crates supérieures inventer leurs propres structures.**

---

# SPRINT 2 — BLUEPRINT DU PSEUDO-MODÈLE

## Responsabilité unique

> **Transformer les métadonnées architecturales en une représentation exploitable du modèle à générer.**

**Durée cible :** 2 à 3 semaines.

---

# 2.1 Architecture

```text
SPRINT 2
│
├── 2.1 → blueprint.rs
├── 2.2 → layer.rs
├── 2.3 → tensor_spec.rs
├── 2.4 → architecture.rs
├── 2.5 → moe.rs
├── 2.6 → naming.rs
├── 2.7 → planner.rs
└── 2.8 → validation.rs
```

---

# ÉTAPE 2.1 — BLUEPRINT

**Fichier :**

```text
crates/pmg-blueprint/src/blueprint.rs
```

### Responsabilité

Représenter le pseudo-modèle complet.

Exemple :

```text
PseudoModelBlueprint
│
├── architecture
├── layers
├── embeddings
├── normalization
└── output
```

### Principe

Le Blueprint ne contient **aucun poids réel**.

Il contient uniquement :

```text
quoi générer
comment le générer
avec quelles dimensions
avec quelles règles
```

---

# ÉTAPE 2.2 — LAYER

**Fichier :**

```text
crates/pmg-blueprint/src/layer.rs
```

### Responsabilité

Décrire une couche Transformer.

Exemple :

```text
Layer 0
├── attention.q
├── attention.k
├── attention.v
├── attention.o
├── mlp.up
├── mlp.down
└── norm
```

---

# ÉTAPE 2.3 — TENSOR SPEC

**Fichier :**

```text
crates/pmg-blueprint/src/tensor_spec.rs
```

### Responsabilité

Décrire précisément comment un tenseur doit être généré.

Exemple :

```text
TensorSpec
├── name
├── shape
├── dtype
├── role
├── distribution
├── structure
└── injection_policy
```

Cette structure est particulièrement importante pour les futurs sprints.

---

# ÉTAPE 2.4 — ARCHITECTURE

**Fichier :**

```text
crates/pmg-blueprint/src/architecture.rs
```

### Responsabilité

Décrire les familles architecturales.

Exemple :

```rust
pub enum ArchitectureKind {
    DenseTransformer,
    MoETransformer,
}
```

---

# ÉTAPE 2.5 — MOE

**Fichier :**

```text
crates/pmg-blueprint/src/moe.rs
```

### Responsabilité

Décrire uniquement les composants Mixture-of-Experts.

Exemple :

```text
num_experts = 128
experts_per_token = 8
```

Le Blueprint doit pouvoir représenter :

[
E = \text{nombre total d'experts}
]

et :

[
K = \text{nombre d'experts activés}
]

sans générer leurs poids à ce stade.

---

# ÉTAPE 2.6 — NAMING

**Fichier :**

```text
crates/pmg-blueprint/src/naming.rs
```

### Responsabilité

Produire les noms de tenseurs.

Exemple :

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
```

### Point critique

Les conventions de nommage doivent être déterministes.

Une même configuration + même architecture doit produire exactement les mêmes noms.

---

# ÉTAPE 2.7 — PLANNER

**Fichier :**

```text
crates/pmg-blueprint/src/planner.rs
```

### Responsabilité

Transformer le Blueprint en plan d'émission.

Exemple :

```text
Blueprint
   ↓
TensorPlan
   ↓
Tensor 1
Tensor 2
Tensor 3
...
Tensor N
```

### Pourquoi ?

Le générateur ne doit pas découvrir dynamiquement ce qu'il doit faire pendant l'écriture.

Il doit recevoir un plan explicite.

---

# ÉTAPE 2.8 — VALIDATION BLUEPRINT

**Fichier :**

```text
crates/pmg-blueprint/src/validation.rs
```

### Responsabilité

Vérifier que le Blueprint est cohérent.

Exemples :

```text
hidden_size compatible avec num_heads
nombre d'experts > 0
expert_top_k <= num_experts
shape de chaque tenseur cohérente
noms uniques
```

---

# VALIDATION SPRINT 2

Le Sprint est terminé lorsque PMG peut produire :

```text
Configuration
     ↓
Blueprint
     ↓
TensorPlan
```

sans générer un seul poids.

### Point fort

Séparation extrêmement nette entre :

```text
architecture
```

et :

```text
génération numérique
```

### Point faible

Le Blueprint peut sembler abstrait pour un débutant.

### Point critique

Ne pas mettre de logique de génération aléatoire dans `pmg-blueprint`.

---

# SPRINT 3 — MOTEUR MATHÉMATIQUE ET STATISTIQUE

## Responsabilité unique

> **Fournir les primitives mathématiques déterministes nécessaires à PMG.**

**Durée cible :** 3 semaines.

C'est un sprint particulièrement important parce qu'il constitue le socle scientifique du générateur.

---

# 3.1 Architecture

```text
SPRINT 3
│
├── 3.1 → rng.rs
├── 3.2 → statistics.rs
├── 3.3 → normal.rs
├── 3.4 → student_t.rs
├── 3.5 → weibull.rs
├── 3.6 → pareto.rs
├── 3.7 → mixture.rs
├── 3.8 → covariance.rs
└── 3.9 → low_rank.rs
```

---

# ÉTAPE 3.1 — RNG

**Fichier :**

```text
crates/pmg-math/src/rng.rs
```

### Responsabilité

Fournir une génération pseudo-aléatoire reproductible.

Concept :

[
X_{seed}=f(seed)
]

Pour une même seed :

```text
seed = 42
```

PMG doit produire exactement la même séquence.

### Exemple

```text
PMG
 ├── seed = 42
 ├── couche = 10
 └── tenseur = q_proj
```

doit toujours produire la même séquence.

### Critique

La seed doit être propagée de façon structurée.

Éviter :

```rust
seed ^ layer_id
```

comme unique stratégie si cela crée des collisions conceptuelles.

Préférer une dérivation documentée et testée.

---

# ÉTAPE 3.2 — STATISTICS

**Fichier :**

```text
crates/pmg-math/src/statistics.rs
```

### Responsabilité

Calculer les statistiques fondamentales :

[
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
]

[
\sigma^2 =
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\mu)^2
]

[
\sigma = \sqrt{\sigma^2}
]

ainsi que :

* min ;
* max ;
* moyenne ;
* variance ;
* écart-type ;
* quantiles ;
* asymétrie ;
* kurtosis.

Ces métriques seront indispensables à `espec` et `validate`.

---

# ÉTAPE 3.3 — NORMALE

**Fichier :**

```text
crates/pmg-math/src/normal.rs
```

### Responsabilité

Implémenter :

[
X\sim\mathcal N(\mu,\sigma^2)
]

avec :

[
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
]

### Utilisation

Distribution de base pour les tenseurs sans structure particulière.

---

# ÉTAPE 3.4 — STUDENT-T

**Fichier :**

```text
crates/pmg-math/src/student_t.rs
```

### Responsabilité

Implémenter la distribution Student-t.

[
f(x)=
\frac{
\Gamma\left(\frac{\nu+1}{2}\right)
}{
\sqrt{\nu\pi},
\Gamma\left(\frac{\nu}{2}\right)
}
\left(
1+\frac{x^2}{\nu}
\right)^{-\frac{\nu+1}{2}}
]

### Pourquoi ?

Elle permet de modéliser des queues plus lourdes qu'une loi normale.

C'est important pour la représentation de valeurs extrêmes.

---

# ÉTAPE 3.5 — WEIBULL

**Fichier :**

```text
crates/pmg-math/src/weibull.rs
```

### Responsabilité

Implémenter :

[
f(x)=
\frac{k}{\lambda}
\left(\frac{x}{\lambda}\right)^{k-1}
e^{-(x/\lambda)^k}
]

pour :

[
x\geq0
]

### Utilisation potentielle

Modélisation de certaines amplitudes positives et distributions asymétriques.

---

# ÉTAPE 3.6 — PARETO

**Fichier :**

```text
crates/pmg-math/src/pareto.rs
```

### Responsabilité

Implémenter une distribution à queue lourde :

[
f(x)=
\frac{\alpha x_m^\alpha}{x^{\alpha+1}}
]

pour :

[
x\geq x_m
]

### Utilisation PMG

Particulièrement intéressante pour expérimenter des comportements extrêmes contrôlés.

---

# ÉTAPE 3.7 — MIXTURE

**Fichier :**

```text
crates/pmg-math/src/mixture.rs
```

### Responsabilité

Combiner plusieurs distributions.

Exemple :

[
X\sim
\begin{cases}
\mathcal N(0,\sigma^2), & 99%\
t_\nu, & 1%
\end{cases}
]

Cela constitue une primitive importante pour les futures politiques de super-poids.

---

# ÉTAPE 3.8 — COVARIANCE

**Fichier :**

```text
crates/pmg-math/src/covariance.rs
```

### Responsabilité

Calculer et exploiter les relations entre variables.

Pour deux variables :

[
\operatorname{Cov}(X,Y)
=======================

E[(X-\mu_X)(Y-\mu_Y)]
]

et :

[
\rho_{XY}
=========

\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
]

### Pourquoi ?

Un pseudo-modèle réaliste ne doit pas être seulement :

```text
beaucoup de nombres aléatoires indépendants
```

Il doit pouvoir présenter des dépendances structurées.

---

# ÉTAPE 3.9 — LOW RANK

**Fichier :**

```text
crates/pmg-math/src/low_rank.rs
```

### Responsabilité

Créer des perturbations de rang contrôlé.

Une structure simple :

[
\Delta W = UV^T
]

avec :

[
U\in\mathbb R^{m\times r}
]

et :

[
V\in\mathbb R^{n\times r}
]

où :

[
r\ll\min(m,n)
]

### Exemple

Pour une matrice :

[
4096\times4096
]

on peut créer :

[
r=8
]

au lieu de générer directement une structure pleine.

Cela réduit considérablement la complexité de construction.

---

# VALIDATION SPRINT 3

PMG doit pouvoir :

```text
seed
 ↓
distribution
 ↓
échantillons
 ↓
statistiques
 ↓
validation statistique
```

et :

```text
U + V
 ↓
UVᵀ
 ↓
structure bas-rang
```

### Point fort

Le moteur mathématique devient indépendant du modèle.

### Point faible

Les distributions ne garantissent pas à elles seules le réalisme.

### Point critique

**Ne jamais prétendre qu'une distribution particulière représente automatiquement les poids d'un vrai LLM.**

Les distributions doivent être traitées comme des **modèles statistiques paramétrables**, validés empiriquement.

---

# SPRINT 4 — MOTEUR D'INJECTION STRUCTURELLE

## Responsabilité unique

> **Introduire volontairement les structures statistiques et anomalies contrôlées nécessaires aux pseudo-modèles.**

**Durée cible :** 3 semaines.

C'est ici que les trois idées fondamentales du PMG deviennent opérationnelles :

1. super-poids ;
2. corrélation ;
3. structure bas-rang.

---

# 4.1 Architecture

```text
SPRINT 4
│
├── 4.1 → injection_policy.rs
├── 4.2 → outlier_mask.rs
├── 4.3 → super_weight.rs
├── 4.4 → correlated.rs
├── 4.5 → low_rank.rs
├── 4.6 → sparse_structure.rs
├── 4.7 → layer_pattern.rs
├── 4.8 → tensor_injector.rs
└── 4.9 → injection_validator.rs
```

---

# ÉTAPE 4.1 — POLITIQUE D'INJECTION

**Fichier :**

```text
crates/pmg-injector/src/injection_policy.rs
```

### Responsabilité

Décrire **quoi injecter**, sans encore effectuer l'injection.

Exemple :

```text
InjectionPolicy
├── outlier_frequency
├── outlier_scale
├── correlation_strength
├── low_rank_probability
├── low_rank_rank
└── heavy_tail_probability
```

---

# ÉTAPE 4.2 — MASQUE D'OUTLIERS

**Fichier :**

```text
crates/pmg-injector/src/outlier_mask.rs
```

### Responsabilité

Déterminer les positions qui seront affectées.

Exemple :

```text
[0 0 0 1 0 0]
[0 0 1 0 0 0]
[0 0 0 0 0 0]
```

Le masque ne modifie encore aucune valeur.

---

# ÉTAPE 4.3 — SUPER-POIDS

**Fichier :**

```text
crates/pmg-injector/src/super_weight.rs
```

### Responsabilité

Transformer certaines valeurs ordinaires en valeurs extrêmes contrôlées.

Une stratégie simplifiée :

[
w' = s\cdot w
]

où :

[
s\gg1
]

Mais PMG doit progressivement évoluer vers une stratégie statistique plus riche :

[
w' \sim T(\theta)
]

où (T) est une distribution de queue lourde ou un mélange contrôlé.

### Exemple conceptuel

Distribution normale :

```text
-0.2  0.1  -0.4  0.3
```

Après injection :

```text
-0.2  0.1  -8.7  0.3
```

Le point important est que l'anomalie doit être :

* rare ;
* contrôlée ;
* reproductible ;
* statistiquement mesurable.

---

# ÉTAPE 4.4 — CORRÉLATION

**Fichier :**

```text
crates/pmg-injector/src/correlated.rs
```

### Responsabilité

Introduire une dépendance contrôlée entre variables.

Une construction simple :

[
X = \rho Z + \sqrt{1-\rho^2}\epsilon
]

avec :

[
Z,\epsilon\sim\mathcal N(0,1)
]

permet d'obtenir approximativement :

[
\operatorname{Corr}(X,Z)=\rho
]

### Exemple

Avec :

[
\rho=0.8
]

les variables deviennent fortement corrélées.

---

# ÉTAPE 4.5 — BAS-RANG

**Fichier :**

```text
crates/pmg-injector/src/low_rank.rs
```

### Responsabilité

Appliquer :

[
W'=W+\alpha UV^T
]

avec :

[
r\ll\min(m,n)
]

### Paramètres

```text
alpha
rank
seed
distribution
```

---

# ÉTAPE 4.6 — STRUCTURE SPARSE

**Fichier :**

```text
crates/pmg-injector/src/sparse_structure.rs
```

### Responsabilité

Créer des structures localisées.

Par exemple :

```text
████░░░░
████░░░░
░░░░░░░░
░░░░████
░░░░████
```

Le but n'est pas seulement de produire des zéros, mais de pouvoir représenter une structure contrôlée.

---

# ÉTAPE 4.7 — PATTERN DE COUCHE

**Fichier :**

```text
crates/pmg-injector/src/layer_pattern.rs
```

### Responsabilité

Faire varier les injections selon la profondeur du réseau.

Exemple :

[
p_l = p_0 + \Delta p\frac{l}{L-1}
]

où :

* (l) = index de couche ;
* (L) = nombre total de couches.

Cela permet de créer des profils non uniformes.

---

# ÉTAPE 4.8 — TENSOR INJECTOR

**Fichier :**

```text
crates/pmg-injector/src/tensor_injector.rs
```

### Responsabilité

Orchestrer les différentes transformations sur un tenseur.

Pipeline :

```text
Base Tensor
    ↓
Distribution
    ↓
Structure
    ↓
Corrélation
    ↓
Low-rank
    ↓
Super-weights
    ↓
Tensor final
```

### Critique

L'ordre des opérations doit être explicitement défini.

Changer :

```text
low-rank → outliers
```

en :

```text
outliers → low-rank
```

peut produire des résultats statistiques différents.

---

# ÉTAPE 4.9 — VALIDATION

**Fichier :**

```text
crates/pmg-injector/src/injection_validator.rs
```

### Responsabilité

Mesurer si l'injection a réellement produit l'effet demandé.

Exemples :

```text
outlier_ratio
mean
std
max_abs
quantiles
correlation
estimated_rank
```

### Principe fondamental

Une politique :

```text
outlier_frequency = 0.01
```

ne doit pas simplement être acceptée parce qu'elle est configurée.

PMG doit pouvoir mesurer :

[
\hat p =
\frac{N_{\text{outliers}}}{N}
]

et comparer :

[
|\hat p-p|<\epsilon
]

---

# VALIDATION SPRINT 4

Le Sprint est terminé lorsque PMG sait produire :

```text
Tensor statistique
       +
structure
       +
corrélation
       +
bas-rang
       +
super-poids
```

et **mesurer** le résultat.

### Point fort

C'est le premier sprint qui transforme réellement PMG en générateur statistiquement structuré.

### Point faible

Les paramètres sont encore essentiellement définis manuellement.

### Point critique

Il faut éviter le piège :

> « Plus il y a d'anomalies, plus le modèle est réaliste. »

C'est faux.

Le réalisme doit être évalué par des métriques.

---

# SPRINT 5 — PREMIER PIPELINE COMPLET DE GÉNÉRATION

## Responsabilité unique

> **Assembler Core + Blueprint + Math + Injection pour obtenir une génération déterministe de pseudo-modèle.**

**Durée cible :** 3 semaines.

---

# 5.1 Architecture

```text
SPRINT 5
│
├── 5.1 → generator.rs
├── 5.2 → tensor_generator.rs
├── 5.3 → chunk.rs
├── 5.4 → seed_plan.rs
├── 5.5 → generation_report.rs
├── 5.6 → generation_validator.rs
├── 5.7 → deterministic.rs
└── 5.8 → lib.rs
```

---

# ÉTAPE 5.1 — GENERATOR

**Fichier :**

```text
crates/pmg-generator/src/generator.rs
```

### Responsabilité

Orchestrer la génération.

Pipeline :

```text
ModelConfig
     ↓
Blueprint
     ↓
TensorPlan
     ↓
TensorGenerator
     ↓
Injector
     ↓
Tensor
```

---

# ÉTAPE 5.2 — TENSOR GENERATOR

**Fichier :**

```text
crates/pmg-generator/src/tensor_generator.rs
```

### Responsabilité

Générer les valeurs initiales d'un seul tenseur.

Exemple :

```text
TensorSpec
    ↓
distribution
    ↓
RNG
    ↓
values
```

---

# ÉTAPE 5.3 — CHUNK

**Fichier :**

```text
crates/pmg-generator/src/chunk.rs
```

### Responsabilité

Découper la génération en blocs.

Pourquoi ?

Une matrice :

[
4096\times4096
]

contient :

[
16,777,216
]

valeurs.

Pour un modèle entier, la mémoire pourrait exploser si PMG tentait de tout conserver.

La génération doit donc être pensée comme :

```text
Tensor
 ↓
Chunk 0
 ↓
Chunk 1
 ↓
Chunk 2
 ↓
...
```

---

# ÉTAPE 5.4 — SEED PLAN

**Fichier :**

```text
crates/pmg-generator/src/seed_plan.rs
```

### Responsabilité

Déterminer la seed de chaque élément logique.

Exemple conceptuel :

[
S_{tensor}
==========

H(S_{global}, layer, tensor_id)
]

puis :

[
S_{chunk}
=========

H(S_{tensor}, chunk_id)
]

### Objectif

Permettre :

```text
génération complète
```

ou :

```text
génération par chunks
```

avec des résultats identiques.

---

# ÉTAPE 5.5 — RAPPORT

**Fichier :**

```text
crates/pmg-generator/src/generation_report.rs
```

### Responsabilité

Produire un résumé de génération.

Exemple :

```text
PMG — Rapport de génération

Modèle       : ExempleTransformer
Couches      : 32
Tenseurs     : 418
Paramètres   : 7.1B
Seed         : 42

Distribution :
  Normale     94.1 %
  Student-t    4.2 %
  Pareto       0.7 %
  Autres       1.0 %

Injection :
  Outliers    : 0.83 %
  Low-rank    : 12 couches
  Corrélation : activée
```

---

# ÉTAPE 5.6 — VALIDATION GÉNÉRATION

**Fichier :**

```text
crates/pmg-generator/src/generation_validator.rs
```

### Responsabilité

Vérifier que la génération est cohérente.

Tests :

```text
nombre de tenseurs
nombre de paramètres
shapes
dtype
seed
statistiques
injections
```

---

# ÉTAPE 5.7 — DÉTERMINISME

**Fichier :**

```text
crates/pmg-generator/src/deterministic.rs
```

### Responsabilité

Garantir le déterminisme.

Test fondamental :

```text
seed = 42
generation A

seed = 42
generation B
```

doit produire :

```text
A == B
```

à spécification identique.

Et :

```text
seed = 42
generation A

seed = 43
generation B
```

doit normalement produire :

```text
A != B
```

---

# ÉTAPE 5.8 — API GÉNÉRATEUR

**Fichier :**

```text
crates/pmg-generator/src/lib.rs
```

### Responsabilité

Exposer l'API publique du générateur.

---

# 6. PIPELINE COMPLET À LA FIN DU SPRINT 5

Le système doit désormais suivre :

```text
                    ┌──────────────────┐
                    │ ModelConfig      │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Blueprint        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ TensorPlan       │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Distribution     │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ RNG déterministe │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Structure        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Corrélation      │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Low-rank         │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Super-weights    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Validation       │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Tensor final     │
                    └──────────────────┘
```

---

# 7. MATRICE DES RESPONSABILITÉS

| Couche         | S0 | S1 | S2 | S3 | S4 | S5 |
| -------------- | -: | -: | -: | -: | -: | -: |
| Infrastructure |  ✓ |    |    |    |    |    |
| Core           |    |  ✓ |    |    |    |    |
| Blueprint      |    |    |  ✓ |    |    |    |
| Mathématiques  |    |    |    |  ✓ |    |    |
| Injection      |    |    |    |    |  ✓ |    |
| Génération     |    |    |    |    |    |  ✓ |

Cette séparation est volontaire.

---

# 8. MATRICE DES DÉPENDANCES

```text
                    pmg-cli
                       │
                       ↓
                pmg-generator
                 ↙     ↓      ↘
        pmg-injector  pmg-io  pmg-math
              ↓         ↓       ↓
         pmg-blueprint ────────┘
                ↓
             pmg-core
```

La direction exacte des dépendances Cargo devra être validée pendant l'implémentation pour éviter toute dépendance circulaire.

---

# 9. CRITÈRES DE QUALITÉ COMMUNS AUX SPRINTS 0–5

Chaque étape doit respecter :

### Compilation

```bash
cargo check
```

### Tests

```bash
cargo test
```

### Formatage

```bash
cargo fmt --all -- --check
```

### Clippy

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Documentation

```bash
cargo doc --workspace --no-deps
```

### Règle 500 lignes

```text
Fichier > 500 lignes
       ↓
REFactorisation obligatoire
```

---

# 10. RÈGLE DE TEST POUR CHAQUE ÉTAPE

Même si chaque étape possède **un seul fichier de responsabilité**, le fichier Rust peut contenir son module :

```rust
#[cfg(test)]
mod tests {
    // tests unitaires liés exclusivement à ce fichier
}
```

Cela permet de respecter le principe :

```text
1 responsabilité
1 fichier principal
1 unité logique
```

tout en gardant les tests proches de leur implémentation.

Les tests d'intégration inter-crates seront ajoutés aux **portes de validation des sprints**, et non utilisés pour mélanger les responsabilités des étapes.

---

# 11. MÉTHODE DE TRAVAIL D'IBRAHIMA-224

Puisqu'il n'y a qu'un développeur, le risque principal n'est pas le conflit entre développeurs.

Le risque est :

> **accumuler trop de travail simultanément et perdre la traçabilité.**

La méthode recommandée est donc :

```text
Lire l'étape
   ↓
Lire ses références
   ↓
Définir les invariants
   ↓
Écrire le fichier
   ↓
Écrire les tests
   ↓
cargo fmt
   ↓
cargo check
   ↓
cargo test
   ↓
cargo clippy
   ↓
Commit
   ↓
Étape suivante
```

---

# 12. FORMAT DE COMMIT RECOMMANDÉ

Exemple :

```bash
git commit -m "feat(pmg-core): add tensor metadata"
```

Puis :

```bash
git commit -m "test(pmg-core): validate tensor metadata invariants"
```

Puis :

```bash
git commit -m "refactor(pmg-core): simplify metadata validation"
```

---

# 13. CHECKPOINT À LA FIN DE CHAQUE SPRINT

Avant de commencer le sprint suivant, Ibrahima-224 doit répondre **oui** à toutes les questions :

```text
[ ] Le code compile-t-il ?
[ ] Les tests passent-ils ?
[ ] Clippy est-il propre ?
[ ] rustfmt est-il propre ?
[ ] Les API publiques sont-elles documentées ?
[ ] Les invariants sont-ils testés ?
[ ] Les erreurs sont-elles explicites ?
[ ] Aucun fichier ne dépasse-t-il 500 lignes ?
[ ] Les responsabilités sont-elles correctement séparées ?
[ ] Les dépendances sont-elles minimales ?
[ ] Le résultat est-il déterministe lorsque requis ?
[ ] La documentation correspond-elle au code ?
```

Si une réponse est **non**, le sprint n'est pas considéré comme terminé.

---

# 14. RISQUES MAJEURS DES SPRINTS 0–5

## Risque 1 — Sur-architecture

Créer trop tôt :

```text
GPU
distributed generation
multi-node
CUDA
parallel scheduler
```

### Réponse

Reporter ces fonctionnalités.

---

## Risque 2 — Dépendances excessives

Le projet doit rester léger.

### Réponse

Chaque crate externe doit répondre à une question :

> « Pourquoi avons-nous réellement besoin de cette crate ? »

---

## Risque 3 — Confusion entre Blueprint et données

Erreur :

```text
Blueprint = poids
```

Correct :

```text
Blueprint = description de ce qui doit être généré
```

---

## Risque 4 — Aléatoire non reproductible

Erreur :

```text
thread_rng()
```

partout.

Correct :

```text
seed global
   ↓
seed modèle
   ↓
seed couche
   ↓
seed tenseur
   ↓
seed chunk
```

---

## Risque 5 — Faux réalisme statistique

Une génération :

```text
Normal(0,1)
```

avec quelques valeurs énormes n'est pas automatiquement un pseudo-modèle réaliste.

Le futur PMG devra pouvoir comparer plusieurs propriétés :

[
\mu,\sigma,\text{quantiles},\text{kurtosis},\text{corrélations},\text{structure},\text{queues}
]

et non seulement :

```text
min / max
```

---

# 15. RÉSULTAT ATTENDU À LA FIN DU SPRINT 5

À ce stade, PMG doit disposer d'un **prototype scientifique et logiciel cohérent**, capable conceptuellement de réaliser :

```text
Configuration
      ↓
Blueprint
      ↓
Plan de tenseurs
      ↓
Distribution
      ↓
Génération déterministe
      ↓
Structures statistiques
      ↓
Super-poids
      ↓
Validation
```

Mais **pas encore** :

* l'interface CLI complète ;
* `generate` final ;
* `espec` complet ;
* `validate` complet ;
* `compare` complet ;
* le support exhaustif Safetensors ;
* le streaming final ;
* les benchmarks industriels ;
* les profils de modèles avancés ;
* la calibration automatique sur des modèles réels.

Ces éléments appartiennent aux Sprints 6–11.

---

# 16. RÉFÉRENCES TECHNIQUES DE TRAVAIL

Pour le développement Rust :

* [The Rust Programming Language — Rust Book](https://doc.rust-lang.org/book/?utm_source=chatgpt.com)
* [Cargo — Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html?utm_source=chatgpt.com)
* [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/?utm_source=chatgpt.com)
* [Clippy — documentation officielle](https://doc.rust-lang.org/clippy/?utm_source=chatgpt.com)
* [Rustdoc — tests de documentation](https://doc.rust-lang.org/rustdoc/documentation-tests.html?utm_source=chatgpt.com)

Pour le format de modèles :

* [Hugging Face Safetensors — documentation officielle](https://huggingface.co/docs/safetensors/main/index?utm_source=chatgpt.com)

La documentation Safetensors confirme notamment que le format est conçu pour un stockage sûr et rapide des tenseurs et qu'il permet des accès ciblés à des tenseurs ou à des tranches, ce qui correspond bien à la philosophie PMG d'éviter le chargement intégral des poids lorsqu'il s'agit uniquement d'inspection. ([Hugging Face][1])

---

# 17. SYNTHÈSE EXÉCUTIVE

Les Sprints 0–5 doivent donc être compris comme **six fondations successives** :

```text
S0
INFRASTRUCTURE
     │
     ▼
S1
CORE
     │
     ▼
S2
BLUEPRINT
     │
     ▼
S3
MATH / STATISTIQUES
     │
     ▼
S4
INJECTION STRUCTURELLE
     │
     ▼
S5
GÉNÉRATION DÉTERMINISTE
```

Le point essentiel est que **PMG ne doit pas être développé comme un simple générateur de nombres aléatoires**.

L'architecture des Sprints 0–5 prépare déjà le véritable objectif scientifique du projet :

[
\boxed{
\text{Pseudo-Modèle}

\text{Architecture}
+
\text{Distributions}
+
\text{Structures}
+
\text{Corrélations}
+
\text{Queues lourdes}
+
\text{Super-poids}
+
\text{Déterminisme}
}
]

Et surtout, chaque mécanisme introduit devra être **mesurable, testable et désactivable individuellement**. C'est cette propriété qui permettra dans les Sprints 6–11 de passer d'un prototype mathématique à un véritable logiciel PMG exploitable.

**Étape suivante logique :** le cahier **Sprints 6 à 11** devra reprendre exactement cette granularité, mais couvrir notamment l'I/O Safetensors, le streaming, l'inspection `espec`, la validation avancée, `compare` sans téléchargement des poids, la CLI française, les profils architecturaux, la calibration statistique, les benchmarks et la release.

[1]: https://huggingface.co/docs/safetensors/main/index?utm_source=chatgpt.com "Safetensors · Hugging Face"
[2]: https://doc.rust-lang.org/rustc/tests/index.html?utm_source=chatgpt.com "Tests - The rustc book"
