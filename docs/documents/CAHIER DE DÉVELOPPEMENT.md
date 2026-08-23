# CAHIER DE DÉVELOPPEMENT
# Pseudo-Models Generator — PMG

**Version :** 1.0  
**Statut :** Approuvé  
**Projet :** Pseudo-Models Generator  
**Licence :** GPL-3.0  
**Langage principal :** Rust  
**Interface :** CLI riche en français  
**Cibles V1 :**
- GLM-5.2
- DeepSeek-V4-Flash

**Équipe d’architecture :**
- Ibrahima-224 — Product Owner / Software Engineering
- Boubacar Diop — Développeur senior
- Abdoulaye Baldé — Scrum Master

---

# 1. Objet du document

Ce document définit les règles techniques, architecturales et organisationnelles utilisées pour développer **Pseudo-Models Generator (PMG)**.

Il constitue la référence destinée aux développeurs travaillant sur le projet.

Le document explique notamment :

1. l'architecture logicielle ;
2. l'organisation du dépôt ;
3. les responsabilités de chaque crate ;
4. les conventions Rust ;
5. la gestion des fichiers de modèles ;
6. la représentation interne des tenseurs ;
7. le moteur mathématique de génération ;
8. la génération des pseudo-poids ;
9. la génération des outliers ;
10. la génération des structures corrélées et bas-rang ;
11. les distributions statistiques ;
12. la gestion des dtypes ;
13. la quantification ;
14. la génération de modèles à taille cible ;
15. la génération des fichiers Safetensors ;
16. la validation ;
17. les tests ;
18. les benchmarks ;
19. la sécurité ;
20. le workflow Git ;
21. la CI/CD.

---

# 2. Principe fondamental de PMG

## 2.1 Définition

PMG est un générateur de **pseudo-modèles de langage** destiné à fournir à des logiciels de :

- quantification ;
- compression ;
- optimisation ;
- inférence ;
- conversion ;
- inspection ;
- benchmarking ;
- expérimentation ;

un modèle suffisamment réaliste structurellement et statistiquement pour permettre de tester leurs mécanismes sans nécessiter le téléchargement complet des poids originaux.

---

# 3. Règle scientifique fondamentale

PMG ne doit jamais prétendre qu'un pseudo-modèle est mathématiquement identique au modèle original lorsque les poids originaux n'ont pas été lus.

Cette distinction est fondamentale.

Soit le véritable modèle :

\[
M=(A,W,T,C)
\]

avec :

- \(A\) : architecture ;
- \(W\) : poids ;
- \(T\) : tokenizer ;
- \(C\) : configuration.

Les fichiers de configuration permettent d'observer une partie de :

\[
O(M)=\{A,T,C,S\}
\]

où \(S\) représente certaines métadonnées des fichiers de poids, notamment :

- noms de tenseurs ;
- dimensions ;
- dtypes ;
- offsets ;
- répartition des shards ;
- nombre de paramètres déductible.

Mais ils ne permettent généralement pas de connaître exactement :

\[
W
\]

ni les statistiques internes détaillées des valeurs de \(W\).

Ainsi PMG doit produire :

\[
\hat W \sim P(W\mid O(M),R)
\]

où :

- \(\hat W\) = pseudo-poids ;
- \(P\) = modèle statistique ;
- \(O(M)\) = informations observables ;
- \(R\) = règles de génération ;
- \(\sim\) = génération selon la distribution définie.

Le pseudo-modèle est donc un **modèle synthétique contraint par les informations observables**, et non une reconstruction exacte des poids.

---

# 4. Principe "aucune information inventée"

PMG possède trois catégories d'informations.

## 4.1 Informations certaines

Informations directement observées dans les fichiers de configuration ou métadonnées autorisées.

Exemples :

```text
hidden_size = 4096
num_layers = 32
dtype = BF16
vocab_size = 128000
```

Ces informations peuvent être utilisées directement.

---

## 4.2 Informations déduites

Informations obtenues mathématiquement à partir d'informations certaines.

Exemple :

Si :

\[
shape=(4096,4096)
\]

alors :

\[
N=4096\times4096=16\,777\,216
\]

éléments.

Si chaque élément utilise 2 octets :

\[
size=16\,777\,216\times2
\]

\[
size=33\,554\,432\ bytes
\]

Ces informations doivent être marquées comme **dérivées**.

---

## 4.3 Informations synthétiques

Informations générées par PMG parce qu'elles ne sont pas disponibles directement.

Exemples :

- distribution des valeurs ;
- corrélations ;
- rang effectif ;
- super-poids ;
- outliers ;
- structure locale ;
- spectre synthétique ;
- matrices bas-rang ;
- dépendances statistiques.

Elles doivent être explicitement marquées :

```text
source = synthetic
confidence = estimated
```

PMG ne doit jamais présenter une information synthétique comme une mesure réelle.

---

# 5. Lecture des fichiers du modèle

PMG V1 travaille principalement avec :

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.json
template_jinja.json
model.safetensors.index.json
*.safetensors
```

Cependant, les poids ne doivent pas être lus pour leur contenu dans le mode normal de PMG.

---

# 6. Règle concernant les fichiers .safetensors

## 6.1 Interdiction principale

PMG ne doit jamais télécharger ou analyser l'intégralité des données de poids simplement pour construire un pseudo-modèle.

Il ne doit donc jamais effectuer :

```text
download(model.safetensors)
```

pour ensuite analyser plusieurs gigaoctets de poids.

---

## 6.2 Inspection distante des métadonnées

Une exception contrôlée est autorisée pour les modèles distants :

```text
HTTP Range
```

L'objectif est de récupérer uniquement l'en-tête Safetensors.

Le format Safetensors place au début du fichier une longueur d'en-tête puis un en-tête JSON contenant notamment les informations relatives aux tenseurs. La documentation Hugging Face décrit explicitement l'utilisation de requêtes Range pour récupérer ces métadonnées sans télécharger les poids complets.

PMG peut donc effectuer conceptuellement :

```text
Range: bytes=0-7
```

puis :

```text
Range: bytes=8-(7+header_length)
```

Mais PMG ne doit jamais récupérer les régions :

```text
data_offsets[0] .. data_offsets[1]
```

correspondant aux données des tenseurs.

---

# 7. Architecture générale

PMG est organisé en workspace Cargo.

Structure recommandée :

```text
Pseudo-Models-Generator/
│
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── README.md
├── CONTRIBUTING.md
│
├── crates/
│   │
│   ├── pmg-cli/
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── cli.rs
│   │   │   ├── commands/
│   │   │   ├── output.rs
│   │   │   └── mod.rs
│   │   └── tests/
│   │
│   ├── pmg-core/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── model.rs
│   │   │   ├── tensor.rs
│   │   │   ├── layer.rs
│   │   │   ├── architecture.rs
│   │   │   └── error.rs
│   │
│   ├── pmg-config/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── config.rs
│   │   │   ├── tokenizer.rs
│   │   │   ├── index.rs
│   │   │   └── template.rs
│   │
│   ├── pmg-safetensors/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── header.rs
│   │   │   ├── index.rs
│   │   │   ├── writer.rs
│   │   │   ├── reader.rs
│   │   │   └── packing.rs
│   │
│   ├── pmg-math/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── distributions.rs
│   │   │   ├── statistics.rs
│   │   │   ├── correlation.rs
│   │   │   ├── low_rank.rs
│   │   │   ├── outliers.rs
│   │   │   ├── spectral.rs
│   │   │   └── rng.rs
│   │
│   ├── pmg-generator/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── generator.rs
│   │   │   ├── tensor_generator.rs
│   │   │   ├── layer_generator.rs
│   │   │   ├── budget.rs
│   │   │   └── deterministic.rs
│   │
│   ├── pmg-models/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── glm52.rs
│   │   │   ├── deepseek_v4_flash.rs
│   │   │   └── registry.rs
│   │
│   ├── pmg-inspect/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── inspector.rs
│   │   │   ├── report.rs
│   │   │   └── statistics.rs
│   │
│   └── pmg-validate/
│       ├── src/
│       │   ├── lib.rs
│       │   ├── validator.rs
│       │   ├── structural.rs
│       │   └── semantic.rs
│
├── tests/
│
├── benches/
│
├── docs/
│
└── .github/
    └── workflows/
```

---

# 8. Responsabilités des crates

## 8.1 pmg-cli

Responsabilité :

- parser les commandes ;
- parser les options ;
- afficher les résultats ;
- gérer les codes de sortie ;
- connecter le CLI au moteur PMG.

Il ne doit pas contenir les algorithmes mathématiques.

---

# 9. pmg-core

Cette crate contient les abstractions fondamentales.

Exemples :

```rust
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}
```

Et :

```rust
pub struct Layer {
    pub index: usize,
    pub tensors: Vec<TensorMetadata>,
}
```

---

# 10. pmg-config

Responsable de :

- `config.json` ;
- tokenizer ;
- templates ;
- index Safetensors ;
- métadonnées du dépôt.

Elle transforme les fichiers externes en structures Rust typées.

---

# 11. pmg-safetensors

Cette crate gère :

- parsing de headers ;
- indexation ;
- calcul des tailles ;
- génération des headers ;
- génération des fichiers ;
- découpage en shards ;
- packing quantifié ;
- vérification de cohérence.

Le format Safetensors associe aux tenseurs des informations telles que `dtype`, `shape` et `data_offsets`.

---

# 12. pmg-math

C'est le moteur mathématique de PMG.

Il contient :

- RNG déterministe ;
- distributions ;
- statistiques ;
- matrices ;
- corrélations ;
- structures bas-rang ;
- valeurs extrêmes ;
- spectres ;
- transformations de tenseurs.

Cette crate ne doit connaître ni le CLI ni les chemins de fichiers.

---

# 13. pmg-generator

Elle combine :

```text
Architecture
      +
Tensor metadata
      +
Statistiques
      +
Distributions
      +
Corrélations
      +
Outliers
      +
Low-rank
      +
Budget mémoire
      ↓
Pseudo-poids
```

---

# 14. pmg-models

Cette crate contient les profils des modèles supportés.

V1 :

```text
GLM-5.2
DeepSeek-V4-Flash
```

Chaque profil décrit :

- architecture ;
- types de couches ;
- conventions de noms ;
- tenseurs attendus ;
- règles de génération ;
- règles MoE éventuelles ;
- règles d'attention ;
- règles spécifiques aux modèles.

Les valeurs exactes doivent provenir des artefacts/documentations validés du modèle, jamais être inventées dans le code.

---

# 15. pmg-inspect

Responsable de la commande :

```bash
pmg espec
```

Elle produit :

- architecture ;
- nombre de couches ;
- nombre de paramètres ;
- tailles ;
- dtypes ;
- répartition des tenseurs ;
- statistiques dérivées ;
- statistiques synthétiques ;
- budget mémoire ;
- configuration mathématique ;
- configuration de génération.

---

# 16. pmg-validate

Responsable de :

```bash
pmg validate
```

La validation comporte plusieurs niveaux.

### Niveau 1 — fichiers

Vérifier :

```text
config.json
tokenizer.json
model.safetensors.index.json
*.safetensors
```

---

### Niveau 2 — JSON

Vérifier :

- syntaxe ;
- types ;
- champs obligatoires ;
- cohérence.

---

### Niveau 3 — Safetensors

Vérifier :

\[
offset_{start}<offset_{end}
\]

et :

\[
offset_{end}-offset_{start}
=
N_{elements}\times bytes(element)
\]

lorsque le dtype possède une taille élémentaire directement applicable.

Pour les formats packés, la formule doit utiliser la convention de packing correspondante.

---

### Niveau 4 — architecture

Vérifier que :

```text
configuration
        ↕
tensor names
        ↕
tensor shapes
```

sont cohérents.

---

# 17. Interface CLI

Le CLI officiel est en français.

Commandes :

```text
pmg help
pmg generate
pmg espec
pmg validate
pmg compare
pmg version
```

---

# 18. Commande help

Exemple :

```bash
pmg help
```

Affiche un guide pour débutant.

Exemple :

```text
PMG — Pseudo-Models Generator

COMMANDES :

generate    Générer un pseudo-modèle
espec       Inspecter un modèle
validate    Valider un modèle
compare     Comparer les métadonnées de deux modèles
version     Afficher la version
help        Afficher l'aide

EXEMPLE :

pmg generate --model glm-5.2 --size 1GB --dtype bf16
```

---

# 19. Commande generate

Syntaxe conceptuelle :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GB \
    --dtype bf16 \
    --output ./glm52-pseudo
```

---

# 20. Sélection du modèle

Exemple :

```bash
pmg generate --model deepseek-v4-flash
```

PMG doit utiliser un registre :

```rust
pub enum ModelFamily {
    Glm52,
    DeepSeekV4Flash,
}
```

---

# 21. Taille cible

PMG doit permettre :

```bash
--size 1GB
```

ou :

```bash
--size 1024MB
```

Le générateur doit alors construire un budget.

Soit :

\[
B=1\,073\,741\,824
\]

octets pour 1 GiB.

Le budget final doit respecter :

\[
S_{metadata}+S_{weights}+S_{overhead}\leq B
\]

PMG doit annoncer clairement si la taille demandée est :

- une limite stricte ;
- une taille cible ;
- une taille approximative.

Pour V1, la politique recommandée est :

```text
--size = budget maximal du package de poids + fichiers générés selon une tolérance documentée
```

La définition exacte doit être stabilisée dans la spécification CLI avant implémentation.

---

# 22. Changement de dtype

PMG sépare :

```text
StorageDType
```

de :

```text
QuantizationScheme
```

C'est indispensable.

Exemple :

```text
StorageDType:
    F32
    F16
    BF16
    F8_...
    I8
    U8
```

et :

```text
QuantizationScheme:
    None
    Int8
    Int4Packed
    NF4Packed
    GPTQLike
    AWQLike
```

Un format de quantification 4 bits n'est pas automatiquement un dtype natif Safetensors. Il faut donc distinguer le type physique de stockage et la convention de quantification. La documentation Safetensors montre notamment que les métadonnées de tenseurs exposent des dtypes comme F32, F16, BF16, I8, U8, etc.; les extensions de dtype doivent être traitées selon la version réellement supportée par le générateur/lecteur.

---

# 23. Exemple de commande dtype

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GB \
    --dtype bf16
```

Ou :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GB \
    --quantization int4
```

---

# 24. Règle de cohérence dtype

Pour un tenseur :

\[
T\in \mathbb{R}^{m\times n}
\]

le nombre d'éléments est :

\[
N=m\times n
\]

Pour un stockage à \(b\) bits par élément :

\[
S=\left\lceil\frac{Nb}{8}\right\rceil
\]

hors métadonnées supplémentaires.

Pour BF16 :

\[
b=16
\]

donc :

\[
S=2N
\]

octets.

Pour INT8 :

\[
S=N
\]

octets.

Pour INT4 packé :

\[
S=\left\lceil\frac{N}{2}\right\rceil
\]

octets, avant les éventuels scales, zero-points, groupes ou métadonnées.

---

# 25. Génération des pseudo-poids

PMG ne génère pas simplement :

```rust
random()
```

pour chaque poids.

Cette approche produirait un tenseur structurellement pauvre.

Le générateur doit utiliser plusieurs composantes.

---

# 26. Modèle mathématique du générateur

Un pseudo-tenseur peut être modélisé :

\[
W=\alpha W_{base}
+\beta W_{corr}
+\gamma W_{lr}
+\delta W_{outlier}
+\epsilon W_{local}
\]

avec :

- \(W_{base}\) : composante principale ;
- \(W_{corr}\) : corrélations ;
- \(W_{lr}\) : structure bas-rang ;
- \(W_{outlier}\) : valeurs extrêmes ;
- \(W_{local}\) : variations locales.

Les coefficients doivent être contrôlés par le profil du modèle.

---

# 27. Distributions

Le moteur peut fournir plusieurs familles :

```text
Normal
StudentT
Laplace
LogNormal
Weibull
Pareto
Mixture
```

Mais PMG ne doit pas utiliser toutes les distributions arbitrairement.

Chaque distribution doit avoir une justification dans le profil statistique.

---

# 28. Distribution de base

Exemple :

\[
X\sim\mathcal N(0,\sigma^2)
\]

Puis :

\[
W_{base}=X
\]

---

# 29. Student-t

La Student-t peut être utilisée lorsqu'une composante à queues plus lourdes est nécessaire.

\[
X\sim t_\nu
\]

Lorsque :

\[
\nu\rightarrow\infty
\]

la Student-t converge vers une loi normale.

PMG peut donc utiliser \(\nu\) comme paramètre contrôlant l'épaisseur des queues.

---

# 30. Mélanges statistiques

Une meilleure approche pour certains tenseurs est :

\[
P(W)=
\sum_{i=1}^{k}\pi_iP_i(W)
\]

avec :

\[
\sum_i\pi_i=1
\]

Exemple conceptuel :

```text
95 % : distribution principale
4.5 % : composante à queues lourdes
0.5 % : composante outlier
```

Ces valeurs sont des exemples de configuration et ne doivent pas être considérées comme les statistiques réelles d'un modèle donné.

---

# 31. Injection des outliers

Les outliers constituent une partie importante du générateur.

PMG doit distinguer :

```text
élément outlier
ligne outlier
colonne outlier
canal outlier
bloc outlier
tenseur outlier
```

---

# 32. Modèle d'outlier

Pour un poids normal :

\[
w\sim P_{base}
\]

Un outlier peut être généré :

\[
w'=s\cdot w
\]

avec :

\[
s>1
\]

ou directement depuis une distribution à queues lourdes.

PMG doit contrôler :

- fréquence ;
- magnitude ;
- position ;
- corrélation ;
- regroupement.

---

# 33. Super-poids

PMG utilise le terme interne :

```text
SuperWeight
```

pour désigner un petit ensemble de valeurs extrêmement importantes dans le pseudo-modèle.

Un super-poids n'est pas nécessairement un phénomène universellement présent dans tous les modèles.

Il doit donc être traité comme une **hypothèse synthétique configurable**, sauf lorsqu'une source mesurée permet de l'établir.

Structure :

```rust
pub struct SuperWeight {
    pub tensor_id: TensorId,
    pub index: usize,
    pub magnitude: f64,
    pub confidence: Confidence,
}
```

---

# 34. Corrélations

Un tenseur ne doit pas systématiquement être constitué de valeurs indépendantes.

PMG peut générer :

\[
W=LU^T+\sigma E
\]

où :

- \(L\) = matrice latente ;
- \(U\) = matrice latente ;
- \(E\) = bruit.

Cette construction crée une structure de corrélation contrôlée.

---

# 35. Structure bas-rang

Pour un tenseur matriciel :

\[
W\in\mathbb{R}^{m\times n}
\]

on peut créer :

\[
W_{lr}=UV^T
\]

avec :

\[
U\in\mathbb{R}^{m\times r}
\]

et :

\[
V\in\mathbb{R}^{n\times r}
\]

où :

\[
r\ll\min(m,n)
\]

Le nombre de paramètres passe alors de :

\[
mn
\]

à :

\[
r(m+n)
\]

pour la composante latente.

---

# 36. Structure complète

Le modèle synthétique peut donc être :

\[
W =
UV^T
+
\sigma E
+
O
\]

où :

- \(UV^T\) = structure bas-rang ;
- \(E\) = composante statistique ;
- \(O\) = composante outlier.

Cette architecture est particulièrement intéressante pour tester les algorithmes de compression et de quantification.

---

# 37. Spectre synthétique

Pour certains tenseurs, PMG doit pouvoir produire une structure spectrale.

Si :

\[
W=U\Sigma V^T
\]

alors :

\[
\Sigma=
diag(\sigma_1,\sigma_2,\dots,\sigma_r)
\]

PMG peut contrôler :

\[
\sigma_1\geq\sigma_2\geq\dots\geq\sigma_r
\]

afin de créer différents profils :

```text
faiblement compressible
modérément compressible
fortement compressible
```

Cela permet notamment de tester des méthodes de réduction de rang.

---

# 38. Génération par type de tenseur

PMG ne doit pas utiliser le même générateur pour :

```text
embedding
attention
Q/K/V
O projection
MLP
gate
up projection
down projection
normalization
router
expert
lm_head
```

Chaque catégorie doit posséder un profil.

---

# 39. Embeddings

Exemple conceptuel :

\[
E\in\mathbb{R}^{V\times H}
\]

avec :

- \(V\) = vocabulaire ;
- \(H\) = dimension cachée.

Le générateur doit respecter exactement :

```text
shape
dtype
tensor name
```

---

# 40. Normalisation

Les tenseurs de normalisation sont généralement beaucoup plus petits que les matrices principales.

PMG doit éviter d'appliquer mécaniquement les mêmes distributions aux paramètres de normalisation.

---

# 41. Matrices linéaires

Pour :

\[
W\in\mathbb{R}^{m\times n}
\]

PMG peut combiner :

```text
base distribution
+
low-rank component
+
row/column correlation
+
outliers
```

---

# 42. MoE

Si un modèle contient des experts, PMG doit conserver :

```text
nombre d'experts
dimensions
noms
structure
routing tensors
```

et ne doit pas transformer arbitrairement une architecture MoE en architecture dense.

---

# 43. Génération déterministe

Chaque génération doit pouvoir être reproduite.

Commande :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GB \
    --seed 42
```

Doit produire exactement la même génération lorsqu'elle est exécutée avec :

- même version PMG ;
- même profil ;
- mêmes paramètres ;
- même seed ;
- même plateforme lorsque les opérations numériques le permettent.

---

# 44. Dérivation des seeds

Le générateur ne doit pas utiliser simplement :

```text
seed + layer_index
```

pour toutes les structures.

Une dérivation déterministe doit être utilisée :

\[
seed_i=H(seed,model,tensor,layer,component)
\]

où \(H\) est une fonction de dérivation déterministe.

Cela évite que deux composants différents utilisent accidentellement la même séquence pseudo-aléatoire.

---

# 45. Budget mémoire

PMG doit disposer d'un :

```text
BudgetPlanner
```

Exemple :

```text
Budget total : 1 GiB

Metadata       : 2 MiB
Headers        : 1 MiB
Weights        : 1019 MiB
Safety margin  : 2 MiB
```

Le générateur ne doit jamais allouer tout le modèle en RAM si cela n'est pas nécessaire.

---

# 46. Génération streaming

Le moteur doit privilégier :

```text
générer
→ encoder
→ écrire
→ libérer
→ tenseur suivant
```

plutôt que :

```text
générer tout le modèle
→ stocker en RAM
→ écrire à la fin
```

Cette architecture permet de générer des modèles beaucoup plus volumineux que la mémoire disponible.

---

# 47. Génération par chunks

Un tenseur peut être divisé :

```text
Tensor
 ├── chunk 0
 ├── chunk 1
 ├── chunk 2
 └── ...
```

La taille d'un chunk doit être configurable.

Exemple :

```text
64 KiB
256 KiB
1 MiB
4 MiB
```

---

# 48. Safetensors Writer

Le writer doit produire :

```text
header_size
header_json
tensor_data
```

Le header doit contenir les informations nécessaires au lecteur.

PMG doit contrôler :

\[
offset_{i+1}\geq offset_i
\]

et :

\[
offset_{final}\leq file\_size
\]

---

# 49. Sharding

PMG doit pouvoir produire :

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors

model.safetensors.index.json
```

Le système d'indexation doit être cohérent avec les noms réellement générés.

---

# 50. Structure du dossier généré

Exemple :

```text
glm52-pseudo/
│
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── chat_template.json
│
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
│
└── pmg-manifest.json
```

---

# 51. pmg-manifest.json

PMG peut ajouter son propre fichier :

```json
{
  "generator": "PMG",
  "version": "1.0.0",
  "model_family": "glm-5.2",
  "synthetic": true,
  "seed": 42
}
```

Ce fichier permet d'identifier clairement qu'il s'agit d'un pseudo-modèle.

Il ne doit pas être utilisé à la place des fichiers attendus par les moteurs d'inférence.

---

# 52. Compatibilité avec les moteurs

PMG doit viser :

```text
syntactic compatibility
+
structural compatibility
+
shape compatibility
+
dtype compatibility
+
naming compatibility
```

Mais PMG ne doit pas promettre :

```text
behavioral equivalence
```

sans validation réelle contre les poids originaux.

---

# 53. Commande espec

Exemple :

```bash
pmg espec ./glm52-pseudo
```

Sortie :

```text
╭──────────────────────────────╮
│      PMG — INSPECTION        │
╰──────────────────────────────╯

Modèle          : GLM-5.2
Type            : Pseudo-model
Architecture    : détectée
Couches         :  ...
Paramètres      :  ...
Dtype           : BF16

Tenseurs
  Embeddings     : ...
  Attention      : ...
  MLP            : ...
  Experts        : ...

Statistiques
  Distribution   : synthétique
  Outliers       : activés
  Low-rank       : activé
  Corrélation    : activée

Confiance
  Architecture  : CERTAIN
  Shapes        : CERTAIN
  Dtypes        : CERTAIN
  Statistiques  : SYNTHÉTIQUE
```

---

# 54. Commande validate

Exemple :

```bash
pmg validate ./glm52-pseudo
```

Résultat :

```text
[OK] config.json
[OK] tokenizer.json
[OK] tokenizer_config.json
[OK] model.safetensors.index.json
[OK] shards
[OK] tensor names
[OK] tensor shapes
[OK] offsets
[OK] dtypes
[OK] budget
```

---

# 55. Commande compare

PMG ne doit pas comparer les poids en profondeur.

Commande :

```bash
pmg compare ./original ./pseudo
```

Elle compare :

```text
architecture
tensor names
tensor shapes
dtypes
number of tensors
number of parameters
sharding
configuration
tokenizer metadata
```

Elle peut aussi comparer les headers Safetensors lorsque ceux-ci sont accessibles.

La commande ne doit pas télécharger les payloads des tenseurs.

---

# 56. Commande dry-run

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GB \
    --dry-run
```

Aucun fichier de poids final ne doit être produit.

Le programme affiche :

```text
Architecture
Budget
Nombre de tenseurs
Taille estimée
Nombre de shards
Dtype
Quantification
Seed
Méthodes statistiques
```

---

# 57. Verbose

Le flag officiel :

```text
-v
--verbose
```

Affiche des informations détaillées destinées à l'utilisateur.

Exemple :

```text
Génération couche 12/80
Tensor : model.layers.12.attention.q_proj.weight
Shape  : [4096, 4096]
Dtype  : BF16
Taille : 32 MiB
```

---

# 58. Debug

Le flag officiel :

```text
--debug
```

Le mode debug affiche les informations internes utiles au développement :

```text
seed dérivé
distribution parameters
budget calculations
chunk allocation
packing information
internal state
```

Les logs de debug ne doivent pas être mélangés avec la sortie normale.

---

# 59. Flags officiels

PMG utilise :

```text
-h, --help
-v, --verbose
--debug
-d, --dry-run
```

Le conflit du document initial où `-h` était attribué simultanément à `--help` et `--debug` est supprimé.

---

# 60. Gestion des erreurs

Chaque crate importante doit posséder ses erreurs spécialisées.

Exemple :

```rust
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("Erreur d'écriture : {0}")]
    Write(#[from] std::io::Error),

    #[error("Header Safetensors invalide")]
    InvalidHeader,

    #[error("Dtype non supporté")]
    UnsupportedDType,

    #[error("Budget mémoire dépassé")]
    BudgetExceeded,
}
```

---

# 61. anyhow dans pmg-cli

Le binaire CLI peut utiliser :

```rust
anyhow::Result
```

pour présenter des erreurs propres à l'utilisateur.

Les crates de bibliothèque doivent conserver des erreurs typées.

---

# 62. unsafe

`unsafe` est interdit par défaut.

Une exception peut être autorisée lorsqu'elle est nécessaire à une API système ou à une optimisation clairement isolée.

Exemple :

```rust
unsafe {
    /* opération strictement encapsulée */
}
```

Toute utilisation doit :

1. être minimale ;
2. être documentée ;
3. être encapsulée ;
4. posséder des invariants vérifiables ;
5. être testée.

---

# 63. Conventions de code

Rust :

```text
snake_case
```

pour :

- fonctions ;
- variables ;
- modules.

```text
CamelCase
```

pour :

- structs ;
- enums ;
- traits.

```text
SCREAMING_SNAKE_CASE
```

pour les constantes.

---

# 64. Formatage

PMG utilise :

```bash
cargo fmt --all
```

avec une largeur maximale de 100 colonnes.

---

# 65. Clippy

Commande :

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Aucun warning n'est accepté sans justification.

---

# 66. Limite de 500 lignes

Aucun fichier Rust ne doit dépasser 500 lignes de code hors commentaires et lignes blanches.

Un fichier trop grand doit être décomposé selon ses responsabilités.

Exemple :

```text
writer.rs
```

devient :

```text
writer/
├── mod.rs
├── header.rs
├── data.rs
├── metadata.rs
└── shards.rs
```

---

# 67. Documentation

Tout élément public doit avoir :

```rust
///
```

Exemple :

```rust
/// Génère un tenseur synthétique selon le profil statistique spécifié.
pub fn generate_tensor(...) -> Result<Tensor> {
    ...
}
```

---

# 68. Commentaires

Les commentaires internes sont en français.

Ils expliquent principalement :

```text
pourquoi
```

et non simplement :

```text
quoi
```

Mauvais :

```rust
// Incrémente i
i += 1;
```

Bon :

```rust
// On avance d'un tenseur afin de conserver l'ordre déterministe
// utilisé pour la dérivation des seeds.
i += 1;
```

---

# 69. Tests unitaires

Chaque algorithme mathématique important doit posséder des tests.

Exemple :

```text
test_normal_distribution
test_student_t_distribution
test_low_rank_generation
test_outlier_generation
test_seed_determinism
test_budget_calculation
test_int4_packing
```

---

# 70. Test de déterminisme

Le test doit vérifier :

\[
G(seed,x)=G(seed,x)
\]

sur deux exécutions.

Exemple :

```rust
let a = generate(config, 42)?;
let b = generate(config, 42)?;

assert_eq!(a, b);
```

---

# 71. Test de budget

Si :

\[
B=1\,073\,741\,824
\]

alors :

\[
S_{generated}\leq B
\]

doit être vérifié automatiquement.

---

# 72. Tests statistiques

Un générateur statistique ne doit pas seulement être testé avec :

```text
assert!(value != 0)
```

Il doit être testé avec des propriétés.

Exemple pour une distribution centrée :

\[
|\hat\mu-\mu|<\epsilon
\]

pour une taille d'échantillon suffisante.

Pour une variance :

\[
|\hat\sigma^2-\sigma^2|<\epsilon
\]

dans les limites attendues.

---

# 73. Tests d'outliers

Le test doit contrôler :

- fréquence ;
- amplitude ;
- déterminisme ;
- position ;
- absence de corruption du reste du tenseur.

---

# 74. Tests de corrélation

Pour deux variables :

\[
\rho(X,Y)=
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
\]

PMG peut vérifier que la corrélation générée se trouve dans une tolérance :

\[
|\hat\rho-\rho_{target}|<\epsilon
\]

---

# 75. Tests low-rank

Si une matrice est générée avec un rang cible \(r\), PMG doit vérifier son rang effectif selon une tolérance numérique.

Il faut éviter d'exiger un rang mathématique exact lorsque les opérations flottantes rendent cette mesure instable.

---

# 76. Tests d'intégration

Les tests d'intégration doivent vérifier :

```text
configuration
→ génération
→ écriture
→ lecture
→ validation
```

---

# 77. Tests E2E

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 64MB \
    --output /tmp/pmg-test
```

Puis :

```bash
pmg validate /tmp/pmg-test
```

Le test doit réussir.

Les tailles réduites sont préférables aux tailles réelles pour les tests CI.

---

# 78. Tests de compatibilité

Pour chaque modèle V1, un jeu de fixtures doit être maintenu :

```text
fixtures/
├── glm52/
│   ├── config.json
│   ├── tokenizer_config.json
│   └── model.safetensors.index.json
│
└── deepseek_v4_flash/
    ├── config.json
    ├── tokenizer_config.json
    └── model.safetensors.index.json
```

Les fixtures ne doivent pas contenir les gigaoctets de poids.

---

# 79. Benchmarks

Les composants critiques doivent être benchmarkés :

```text
header parsing
JSON parsing
tensor generation
outlier injection
low-rank generation
packing
streaming writer
sharding
```

---

# 80. Benchmark mémoire

PMG doit surveiller :

```text
RAM
CPU
I/O
temps de génération
taille finale
```

Un générateur de 1 GiB ne doit pas nécessiter arbitrairement plusieurs dizaines de gigaoctets de RAM.

---

# 81. Parallélisme

Le parallélisme peut être utilisé pour les tenseurs indépendants.

Mais l'ordre d'écriture et la génération doivent rester déterministes.

Le parallélisme ne doit donc pas modifier :

```text
tensor ordering
seed
metadata ordering
shard assignment
```

---

# 82. Rayon

Si Rayon est utilisé :

```text
rayon
```

doit être encapsulé dans les parties où le parallélisme est réellement bénéfique.

Il ne doit pas être utilisé partout automatiquement.

---

# 83. Gestion des dépendances

Chaque dépendance doit répondre à :

```text
Est-elle nécessaire ?
Est-elle maintenue ?
Quel est son coût ?
Quelle est sa licence ?
Présente-t-elle des vulnérabilités ?
```

Commandes :

```bash
cargo tree
cargo audit
cargo outdated
```

---

# 84. Licence

PMG est distribué sous :

```text
GPL-3.0
```

La compatibilité de licence des dépendances doit être vérifiée avant leur intégration.

La règle n'est pas simplement :

```text
"la dépendance doit être GPL"
```

mais :

```text
"la combinaison de licences doit être juridiquement compatible
avec la distribution GPL-3.0 de PMG."
```

---

# 85. Git

Branches principales :

```text
main
develop
```

Branches fonctionnelles :

```text
feature/*
bugfix/*
hotfix/*
refactor/*
release/*
```

---

# 86. Exemple

```bash
git checkout develop
git pull origin develop

git checkout -b feature/outlier-engine
```

---

# 87. Conventional Commits

Format :

```text
type(scope): subject
```

Exemple :

```text
feat(pmg-math): add student-t generator
```

ou :

```text
fix(pmg-safetensors): prevent offset overflow
```

---

# 88. Pull Request

Avant toute PR :

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps
```

La PR doit également vérifier :

```text
[ ] Tests
[ ] Documentation
[ ] Formatage
[ ] Clippy
[ ] Sécurité
[ ] Aucun fichier > 500 lignes
[ ] Commentaires français
```

---

# 89. CI

Pipeline :

```text
Git Push / Pull Request
        │
        ▼
     Build
        │
        ├── Tests
        ├── Clippy
        ├── Format
        ├── Documentation
        └── Audit
```

---

# 90. CD

Lorsqu'un tag est créé :

```text
v1.0.0
```

pipeline :

```text
Build
  ↓
Tests
  ↓
Packaging
  ↓
Release
```

Les packages peuvent être produits selon les plateformes supportées.

---

# 91. Validation avant release

Une release PMG ne doit être publiée que si :

\[
Build=OK
\]

\[
Tests=OK
\]

\[
Clippy=OK
\]

\[
Fmt=OK
\]

\[
Docs=OK
\]

\[
Audit=OK
\]

---

# 92. Sécurité

PMG traite des fichiers externes.

Il doit donc considérer :

```text
JSON malveillant
header corrompu
offset overflow
shape overflow
integer overflow
path traversal
fichier extrêmement volumineux
metadata abuse
```

comme des entrées potentiellement hostiles.

---

# 93. Limitation du header

PMG doit imposer une limite raisonnable sur la taille de l'en-tête JSON avant allocation.

Le lecteur ne doit jamais accepter aveuglément :

```text
header_size
```

fourni par un fichier externe.

Il doit vérifier :

\[
0<header\_size\leq MAX\_HEADER\_SIZE
\]

avant d'allouer.

---

# 94. Protection contre les overflow

Les calculs :

\[
N=\prod_i shape_i
\]

doivent utiliser des opérations vérifiées.

Exemple conceptuel :

```rust
checked_mul()
```

et jamais une multiplication non contrôlée.

---

# 95. Protection des chemins

Lors de la génération :

```text
output/
```

PMG doit éviter toute écriture en dehors du répertoire prévu.

Les noms de fichiers provenant d'un index externe doivent être validés.

---

# 96. Architecture du moteur

Pipeline principal :

```text
INPUT
  │
  ▼
Configuration Loader
  │
  ▼
Model Profile
  │
  ▼
Tensor Metadata
  │
  ▼
Budget Planner
  │
  ▼
Statistical Profile
  │
  ▼
Tensor Generator
  │
  ├── Base distribution
  ├── Correlation
  ├── Low-rank
  ├── Outliers
  └── SuperWeights
  │
  ▼
Quantization / Encoding
  │
  ▼
Safetensors Writer
  │
  ▼
Validator
  │
  ▼
Pseudo Model
```

---

# 97. Principe de séparation

Le générateur ne doit pas connaître directement :

```text
File
Path
CLI
stdout
```

Il produit des structures et flux abstraits.

Le writer s'occupe de la persistance.

Cette séparation facilite :

- les tests ;
- les benchmarks ;
- le streaming ;
- les futures interfaces.

---

# 98. Architecture de configuration

PMG doit avoir une configuration interne :

```rust
pub struct GenerationConfig {
    pub model: ModelFamily,
    pub target_size: Option<u64>,
    pub dtype: StorageDType,
    pub quantization: QuantizationScheme,
    pub seed: u64,
    pub outliers: OutlierConfig,
    pub low_rank: LowRankConfig,
}
```

---

# 99. Profil de modèle

Exemple conceptuel :

```rust
pub trait ModelProfile {
    fn model_family(&self) -> ModelFamily;
    fn architecture(&self) -> &ArchitectureSpec;
    fn tensor_rules(&self) -> &[TensorRule];
    fn generation_policy(&self) -> &GenerationPolicy;
}
```

---

# 100. Règles de tenseurs

Une règle peut définir :

```text
tensor name pattern
shape constraints
dtype
distribution
outlier policy
low-rank policy
```

Exemple conceptuel :

```text
*.q_proj.weight
    → linear_weight
    → low_rank = enabled
    → outliers = enabled
```

---

# 101. Manifest de génération

Chaque génération doit être reproductible.

PMG peut stocker :

```json
{
  "pmg_version": "1.0.0",
  "model": "glm-5.2",
  "seed": 42,
  "dtype": "BF16",
  "target_size": 1073741824,
  "synthetic_statistics": true
}
```

---

# 102. Reproductibilité

Deux machines doivent idéalement produire :

```text
mêmes paramètres
+
même seed
+
même version
=
même pseudo-modèle
```

Lorsque des différences de représentation flottante ou de plateforme empêchent une égalité binaire stricte, PMG doit documenter cette limite.

---

# 103. Gestion des versions des profils

Un profil doit être versionné.

Exemple :

```text
glm52-v1
deepseek-v4-flash-v1
```

Ainsi :

```text
PMG 1.0
+
profil glm52-v1
```

peut être reproduit même si le profil évolue ultérieurement.

---

# 104. Compatibilité ascendante

Les profils existants ne doivent pas être modifiés silencieusement.

Une modification statistique importante doit produire :

```text
profile_version++
```

---

# 105. Philosophie de validation scientifique

PMG doit distinguer quatre niveaux :

```text
EXACT
DERIVED
SYNTHETIC
UNKNOWN
```

Exemple :

```text
num_layers      EXACT
tensor_shape    EXACT
parameter_count DERIVED
weight_mean     UNKNOWN
weight_std      SYNTHETIC
outlier_rate    SYNTHETIC
```

Cette distinction est obligatoire dans `espec`.

---

# 106. Interdiction des faux benchmarks

PMG ne doit jamais dire :

```text
"Le vrai modèle possède exactement 0.13 % d'outliers."
```

si cette valeur n'a pas été mesurée.

Il doit dire :

```text
"Taux d'outliers synthétique : 0.13 %"
```

ou :

```text
"Taux d'outliers : UNKNOWN"
```

---

# 107. Objectif des pseudo-modèles

L'objectif principal n'est pas :

```text
reproduire la sortie exacte du LLM
```

mais :

```text
reproduire suffisamment fidèlement
la structure physique et statistique des poids
pour tester des logiciels de traitement des modèles.
```

---

# 108. Catégories de fidélité

PMG doit mesurer sa fidélité selon plusieurs axes :

```text
F1 — Architecture
F2 — Tensor shapes
F3 — Tensor naming
F4 — Dtype
F5 — Memory layout
F6 — Distribution
F7 — Outliers
F8 — Correlation
F9 — Low-rank structure
F10 — Spectral structure
F11 — Quantization behavior
F12 — Compression behavior
```

---

# 109. Score de fidélité

Un score global peut être défini :

\[
F=
\sum_{i=1}^{n}w_iF_i
\]

avec :

\[
\sum_iw_i=1
\]

Mais ce score ne doit être utilisé que lorsque les valeurs de référence sont disponibles.

Sans données réelles, PMG doit afficher :

```text
FIDELITY SCORE = NOT AVAILABLE
```

plutôt qu'un nombre inventé.

---

# 110. Critère de réussite V1

PMG V1 est considéré fonctionnel lorsque :

1. GLM-5.2 peut être inspecté ;
2. DeepSeek-V4-Flash peut être inspecté ;
3. un pseudo-modèle peut être généré ;
4. les fichiers de configuration sont produits ;
5. les headers Safetensors sont valides ;
6. les shapes sont cohérentes ;
7. les dtypes sont cohérents ;
8. la taille cible est respectée selon la politique définie ;
9. les modèles sont validables ;
10. les générations sont reproductibles ;
11. les moteurs de traitement ciblés peuvent ouvrir le pseudo-modèle lorsque leur contrat de format est satisfait.

---

# 111. Première implémentation recommandée

L'ordre de développement est :

```text
PHASE 1
├── workspace Cargo
├── pmg-core
├── pmg-config
└── pmg-cli

PHASE 2
├── Safetensors header parser
├── index parser
└── validator

PHASE 3
├── model profiles
├── GLM-5.2
└── DeepSeek-V4-Flash

PHASE 4
├── RNG
├── distributions
├── tensor generator
└── deterministic generation

PHASE 5
├── low-rank
├── correlation
├── outliers
└── super-weights

PHASE 6
├── dtype encoder
├── quantization
└── streaming writer

PHASE 7
├── target-size planner
├── sharding
└── generated package

PHASE 8
├── espec
├── validate
└── compare

PHASE 9
├── tests
├── benchmarks
└── CI/CD
```

---

# 112. Règle absolue pour l'équipe

Lorsqu'une information concernant un modèle n'est pas connue :

```text
NE PAS INVENTER.
```

Il faut choisir entre :

```text
EXACT
DERIVED
SYNTHETIC
UNKNOWN
```

C'est l'une des règles scientifiques fondamentales de PMG.

---

# 113. Règle absolue sur les poids

PMG ne doit jamais prétendre reconstruire :

\[
W_{original}
\]

à partir de simples fichiers de configuration.

La cible mathématique est :

\[
\hat W
\sim
P(W\mid O)
\]

et non :

\[
\hat W=W
\]

sauf si les poids originaux ont réellement été fournis et analysés — ce qui est explicitement hors du mode normal de PMG.

---

# 114. Règle absolue sur la taille

Si l'utilisateur demande :

```bash
--size 1GB
```

PMG doit adapter :

```text
dtype
quantification
sharding
chunking
génération
```

au budget.

Il ne doit jamais simplement tronquer arbitrairement les tenseurs.

Un tenseur tronqué détruirait la cohérence architecturale.

---

# 115. Règle absolue sur la qualité

Un pseudo-modèle PMG doit être :

```text
structurellement cohérent
+
mathématiquement valide
+
statistiquement contrôlé
+
reproductible
+
compatible avec son format
+
explicitement identifié comme synthétique.
```

---

# 116. Définition finale du rôle de PMG

PMG n'est pas un :

```text
poids-récupérateur
```

ni un :

```text
modèle de substitution comportemental
```

Il est un :

```text
Synthetic Model Artifact Generator
```

dont la fonction est de construire des artefacts de modèles suffisamment réalistes pour exercer les logiciels qui manipulent les modèles.

---

# 117. Référence d'architecture finale

```text
                         ┌──────────────────┐
                         │      PMG CLI      │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
          generate              espec             validate
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                         ┌────────▼────────┐
                         │   PMG Core      │
                         └────────┬────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
       Configuration          Model Profiles       Metadata
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Budget Planner │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Math Generator  │
                         └────────┬────────┘
                                  │
            ┌─────────────┬───────┼───────┬─────────────┐
            │             │       │       │             │
        Distribution  Low-rank  Corr.  Outliers   SuperWeights
            │             │       │       │             │
            └─────────────┴───────┼───────┴─────────────┘
                                  │
                         ┌────────▼────────┐
                         │ Encoder / DType │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Safetensors I/O │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Pseudo-Model    │
                         └─────────────────┘
```

---

# 118. Conclusion

Le développement de PMG V1 doit être guidé par quatre principes :

### 1. Vérité

Une donnée inconnue reste inconnue.

### 2. Mathématiques

Toute transformation importante doit être définie par un modèle mathématique ou algorithmique testable.

### 3. Compatibilité

Le pseudo-modèle doit respecter les contrats attendus par les outils qui manipulent les modèles.

### 4. Reproductibilité

Une même entrée et une même configuration doivent produire la même génération dans les limites numériques documentées.

**PMG V1 est donc défini comme un générateur déterministe de pseudo-modèles synthétiques, piloté par les métadonnées architecturales des modèles GLM-5.2 et DeepSeek-V4-Flash, avec génération statistique structurée, injection contrôlée d'outliers, structures corrélées et bas-rang, gestion des dtypes et quantifications, génération streaming et production d'artefacts Safetensors compatibles.**

**Statut du présent cahier : BASE DE DÉVELOPPEMENT V1.0**