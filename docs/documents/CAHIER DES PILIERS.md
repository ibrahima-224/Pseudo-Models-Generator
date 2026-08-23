# CAHIER DES PILIERS
# Pseudo-Models Generator — PMG

**Version : 1.0**  
**Statut : Approuvé — Architecture de référence V1**  
**Licence : GPL-3.0**  
**Langage principal : Rust**  
**Interface : CLI riche en français**

**Équipe d'architecture :**
- Ibrahima-224 — Product Owner, Software Engineering & Development
- Boubacar Diop — Développeur senior
- Abdoulaye Baldé — Scrum Master

---

# 1. Objet du cahier des piliers

Ce document définit les **principes fondamentaux et non négociables** du projet Pseudo-Models Generator (PMG).

Il ne décrit pas seulement les fonctionnalités du logiciel. Il définit surtout les règles qui doivent guider :

- l'architecture ;
- le développement ;
- les algorithmes ;
- la génération des pseudo-modèles ;
- la gestion des données ;
- la sécurité ;
- les performances ;
- les tests ;
- la validation scientifique ;
- la compatibilité avec les moteurs de traitement des modèles ;
- la maintenance du projet.

Pour un débutant, on peut comparer les piliers d'un logiciel à ceux d'un bâtiment :

> Si les murs sont bien construits mais que les fondations sont mauvaises, le bâtiment finira par avoir des problèmes.

PMG doit donc être construit autour de principes solides avant de multiplier les fonctionnalités.

---

# 2. Vision de PMG

PMG signifie :

**Pseudo-Models Generator**

Son objectif est de générer un **pseudo-modèle synthétique** représentant aussi fidèlement que possible certaines propriétés structurelles, statistiques, numériques et organisationnelles d'un modèle réel, sans avoir besoin de télécharger ou de lire l'ensemble de ses poids.

Le modèle généré doit pouvoir être utilisé comme **modèle mannequin** par différents logiciels et outils tels que :

- moteurs d'inférence ;
- outils de compression ;
- outils de quantification ;
- outils d'optimisation ;
- analyseurs de modèles ;
- convertisseurs ;
- pipelines de traitement ;
- systèmes de benchmark structurel ;
- logiciels de validation de formats.

---

# 3. Principe scientifique fondamental

PMG ne doit jamais prétendre reconstruire exactement les poids originaux lorsqu'il ne possède pas les valeurs de ces poids.

Soit le modèle réel :

\[
W
\]

et les informations accessibles :

\[
M=f(W)
\]

où \(M\) représente les métadonnées, la configuration, les dimensions, les types, les offsets, etc.

Il peut exister deux modèles différents :

\[
W_1\neq W_2
\]

tels que :

\[
f(W_1)=f(W_2)=M.
\]

Il est donc mathématiquement impossible, en général, de déterminer exactement \(W\) à partir de \(M\) uniquement.

PMG vise donc :

\[
P(W_{PMG})\approx P(W_{réel})
\]

pour des propriétés définies et mesurables.

Par exemple :

\[
\mu_{PMG}\approx\mu_{réel}
\]

\[
\sigma_{PMG}\approx\sigma_{réel}
\]

\[
P(|W_{PMG}|>T)\approx P(|W_{réel}|>T)
\]

\[
\rho_{PMG}\approx\rho_{réel}
\]

et, lorsque cela est possible :

\[
\lambda_i(W_{PMG})\approx\lambda_i(W_{réel})
\]

pour les propriétés spectrales étudiées.

**Cette distinction est fondamentale.**

PMG produit un **pseudo-modèle représentatif**, pas une copie exacte des poids secrets ou absents.

---

# 4. Les piliers de PMG

PMG V1 repose sur les piliers suivants :

1. **Pilar de fidélité structurelle**
2. **Pilier Metadata-First**
3. **Pilier Safetensors**
4. **Pilier Zero-Payload**
5. **Pilier HTTP Range**
6. **Pilier Model Blueprint**
7. **Pilier Tensor Atlas**
8. **Pilier Synthetic Tensor**
9. **Pilier distributions statistiques**
10. **Pilier Outliers et Super-Poids**
11. **Pilier corrélation et structure bas-rang**
12. **Pilier spectral**
13. **Pilier architecture du modèle**
14. **Pilier MoE et routing**
15. **Pilier dtype et quantification**
16. **Pilier génération déterministe**
17. **Pilier streaming et mémoire**
18. **Pilier compatibilité**
19. **Pilier validation scientifique**
20. **Pilier sécurité**
21. **Pilier observabilité**
22. **Pilier performance**
23. **Pilier reproductibilité**
24. **Pilier modularité**
25. **Pilier qualité logicielle**

---

# 5. PILIER 1 — Fidélité structurelle

## 5.1 Principe

Le premier objectif de PMG n'est pas de produire des nombres aléatoires.

Il doit produire une structure qui ressemble réellement à celle du modèle cible.

Un modèle réel peut posséder :

- embeddings ;
- attention ;
- projections Q/K/V ;
- MLP ;
- experts ;
- router ;
- normalisations ;
- matrices de sortie ;
- tensors auxiliaires ;
- paramètres spécifiques à l'architecture.

PMG doit préserver ces structures lorsqu'elles sont connues.

## 5.2 Exemple débutant

Un mauvais générateur pourrait faire :

```text
tensor_001
tensor_002
tensor_003
...
```

avec des matrices aléatoires.

Un générateur PMG doit plutôt comprendre :

```text
model
 ├── embeddings
 ├── layer 0
 │    ├── attention
 │    │    ├── q_proj
 │    │    ├── k_proj
 │    │    └── v_proj
 │    └── MLP
 ├── layer 1
 ├── layer 2
 └── lm_head
```

La structure est donc une information de premier ordre.

---

# 6. PILIER 2 — Metadata-First

PMG doit commencer par les informations disponibles **avant toute génération**.

Sources possibles :

```text
config.json
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
generation_config.json
special_tokens_map.json
template_jinja.json
autres fichiers de configuration
```

PMG construit ensuite une représentation interne :

```text
ModelMetadata
      ↓
ModelBlueprint
      ↓
PseudoModelPlan
```

---

# 7. PILIER 3 — Safetensors

PMG doit comprendre le format Safetensors.

Pour chaque tensor, les informations importantes comprennent notamment :

```text
nom
dtype
shape
offset
taille
nombre d'éléments
```

Si :

\[
shape=(d_1,d_2,\ldots,d_n)
\]

alors :

\[
N=\prod_{i=1}^{n}d_i
\]

est le nombre d'éléments.

Si chaque élément occupe \(b\) octets :

\[
S=N\times b.
\]

Cette relation permet de vérifier la cohérence des métadonnées.

---

# 8. PILIER 4 — Zero-Payload

PMG ne doit pas télécharger le contenu complet des fichiers :

```text
*.safetensors
```

lorsqu'il fonctionne en mode d'analyse d'un modèle réel.

La règle fondamentale est :

> Les poids réels ne constituent jamais une source de données pour la génération V1.

PMG peut connaître :

```text
shape
dtype
offset
size
tensor name
```

mais ne doit pas récupérer les gigaoctets de valeurs correspondants.

---

# 9. PILIER 5 — HTTP Range

Lorsque les modèles sont distants, PMG peut utiliser HTTP Range lorsque cela est nécessaire pour récupérer les informations autorisées sans télécharger le fichier entier.

Conceptuellement :

```text
HTTP server
      │
      │ Range request
      ▼
[portion nécessaire]
      │
      ▼
PMG
```

Au lieu de :

```text
10 GB file
     ↓
10 GB download
```

on vise :

```text
10 GB file
     ↓
small requested region
     ↓
metadata/header
```

La quantité transférée doit être contrôlée et enregistrée.

---

# 10. PILIER 6 — Model Blueprint

PMG ne doit pas générer directement à partir des fichiers JSON.

Il doit d'abord construire un modèle intermédiaire.

Conceptuellement :

```text
Fichiers source
      ↓
Parser
      ↓
Normalizer
      ↓
ModelBlueprint
      ↓
Generator
```

Le `ModelBlueprint` décrit le modèle cible sous une forme indépendante du format de fichier.

Il peut contenir :

```text
architecture
hidden_size
num_layers
num_heads
num_kv_heads
intermediate_size
vocab_size
dtype
tensor layout
MoE configuration
routing information
normalization configuration
```

Les champs exacts dépendent du modèle.

---

# 11. PILIER 7 — Tensor Atlas

PMG doit construire un inventaire des tensors.

Exemple :

```text
TensorAtlas
├── embedding.weight
├── layers.0.attention.q_proj.weight
├── layers.0.attention.k_proj.weight
├── layers.0.attention.v_proj.weight
├── layers.0.mlp.gate_proj.weight
├── layers.0.mlp.up_proj.weight
├── layers.0.mlp.down_proj.weight
└── ...
```

Chaque entrée doit contenir les informations nécessaires à la génération.

Cela permet notamment :

- l'analyse ;
- la validation ;
- la planification ;
- la génération ;
- le calcul de taille ;
- la quantification.

---

# 12. PILIER 8 — Synthetic Tensor

PMG distingue explicitement :

```text
RealTensorMetadata
```

et :

```text
SyntheticTensor
```

Le premier décrit le tensor réel.

Le second contient des valeurs générées par PMG.

Cette séparation empêche de confondre :

> « nous savons comment le tensor est structuré »

avec :

> « nous connaissons ses valeurs ».

---

# 13. PILIER 9 — Distributions statistiques

Les valeurs synthétiques ne doivent pas être uniformément aléatoires.

PMG doit pouvoir utiliser différents modèles statistiques.

Exemples :

### Gaussienne

\[
X\sim\mathcal N(\mu,\sigma^2)
\]

### Student-t

\[
T=\frac{Z}{\sqrt{V/\nu}}
\]

où :

\[
Z\sim\mathcal N(0,1)
\]

et :

\[
V\sim\chi^2_\nu.
\]

### Log-normal

\[
X=e^Y
\]

avec :

\[
Y\sim\mathcal N(\mu,\sigma^2).
\]

### Weibull

\[
F(x)=1-e^{-(x/\lambda)^k}
\]

pour \(x\geq0\).

### Pareto

\[
P(X>x)=\left(\frac{x_m}{x}\right)^\alpha.
\]

PMG doit sélectionner une distribution sur la base d'un profil justifié, et non simplement parce qu'elle « semble réaliste ».

---

# 14. PILIER 10 — Outliers et Super-Poids

Les valeurs extrêmes peuvent avoir une importance considérable dans les modèles neuronaux.

PMG doit donc posséder un mécanisme d'injection d'outliers.

Conceptuellement :

\[
W'=W\odot M
\]

ou, selon le modèle d'injection :

\[
W'_{ij}=W_{ij}s_{ij}
\]

où \(s_{ij}\) est différent de 1 sur les positions sélectionnées.

Exemple :

```text
valeurs normales :
0.02
-0.11
0.07
0.14

outliers :
4.8
-6.1
9.2
```

Mais PMG ne doit pas inventer arbitrairement :

> « 1 % des poids sont des super-poids ».

La fréquence et l'amplitude doivent provenir :

- de données documentées ;
- de mesures disponibles ;
- d'une hypothèse explicitement marquée comme telle ;
- ou d'une calibration expérimentale.

---

# 15. PILIER 11 — Corrélation et structure bas-rang

Une matrice réelle n'est pas nécessairement un ensemble de valeurs indépendantes.

PMG doit pouvoir modéliser :

\[
W\approx UV^T+R
\]

où :

\[
U\in\mathbb R^{m\times r}
\]

\[
V\in\mathbb R^{n\times r}
\]

et \(R\) représente le résidu.

Si :

\[
r\ll\min(m,n),
\]

alors une structure bas-rang est présente.

Cela permet de reproduire des dépendances entre dimensions.

---

# 16. PILIER 12 — Structure spectrale

PMG doit pouvoir représenter les propriétés spectrales des matrices.

Pour une matrice \(W\), on peut étudier :

\[
W^TW
\]

et ses valeurs propres :

\[
\lambda_1,\lambda_2,\ldots,\lambda_n.
\]

Les valeurs singulières sont :

\[
\sigma_i=\sqrt{\lambda_i}.
\]

PMG peut utiliser des profils spectraux pour éviter de produire une matrice totalement aléatoire dont le comportement numérique serait très différent.

---

# 17. PILIER 13 — Architecture du modèle

PMG V1 est spécialisé sur :

1. **DeepSeek-V4-Flash**
2. **GLM-5.2**

Les profils doivent être séparés.

```text
profiles/
├── deepseek-v4-flash/
│   ├── architecture
│   ├── tensor rules
│   ├── statistical profile
│   └── quantization rules
│
└── glm-5.2/
    ├── architecture
    ├── tensor rules
    ├── statistical profile
    └── quantization rules
```

Une propriété connue pour un modèle ne doit pas être automatiquement transférée à l'autre.

---

# 18. PILIER 14 — MoE et routing

Si le modèle utilise une architecture Mixture-of-Experts, PMG doit représenter :

```text
Experts
Router
Expert dimensions
Number of experts
Selection mechanism
Routing metadata
```

Un modèle MoE ne doit pas être réduit à un simple MLP dense.

Conceptuellement :

\[
y=\sum_{i\in S(x)}g_i(x)E_i(x)
\]

où :

- \(E_i\) est un expert ;
- \(g_i(x)\) est son poids de routing ;
- \(S(x)\) est l'ensemble des experts sélectionnés.

PMG doit donc pouvoir générer les structures nécessaires à ce comportement.

---

# 19. PILIER 15 — Dtypes et quantification

PMG doit gérer séparément :

### Dtype

```text
F64
F32
F16
BF16
FP8
I8
U8
I16
...
```

### Schéma de quantification

```text
FP8
INT8
INT4
NF4
GPTQ
AWQ
...
```

Ces concepts ne doivent pas être confondus.

Pour une quantification affine :

\[
q=\operatorname{round}\left(\frac{x}{s}\right)+z
\]

où :

- \(s\) = scale ;
- \(z\) = zero-point.

La reconstruction est :

\[
\hat{x}=s(q-z).
\]

---

# 20. PILIER 16 — Génération déterministe

PMG doit utiliser des seeds explicites.

Exemple :

```bash
pmg generate \
    --model deepseek-v4-flash \
    --size 1G \
    --seed 42
```

Avec la même configuration, le même profil et le même seed, PMG doit viser :

\[
G(M,P,S)=G(M,P,S)
\]

c'est-à-dire une génération reproductible.

Les algorithmes pseudo-aléatoires utilisés doivent donc être déterministes et documentés.

---

# 21. PILIER 17 — Streaming et mémoire

La taille du modèle produit ne doit pas imposer une consommation RAM équivalente.

L'objectif architectural est :

\[
RAM=O(B)
\]

où \(B\) représente les buffers/chunks utilisés.

Exemple :

```text
Pseudo-modèle :
100 GB

RAM disponible :
8 GB

PMG :
génération par blocs
```

Il ne faut pas faire :

```text
100 GB
↓
RAM
↓
écriture
```

mais :

```text
bloc
↓
génération
↓
transformation
↓
quantification
↓
écriture
↓
bloc suivant
```

---

# 22. PILIER 18 — Taille cible

L'utilisateur doit pouvoir demander une taille cible.

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1G
```

PMG doit alors produire un pseudo-modèle respectant le budget de stockage demandé.

Si :

\[
S_{target}=1\,GB
\]

PMG doit déterminer une configuration de stockage telle que :

\[
S_{output}\approx S_{target}.
\]

La taille exacte dépendra :

- du nombre de tensors ;
- des headers ;
- des dtypes ;
- du packing ;
- des métadonnées ;
- du nombre de fichiers.

PMG doit donc distinguer :

```text
target size
estimated size
actual size
```

---

# 23. PILIER 19 — Changement de précision

L'utilisateur doit pouvoir modifier la représentation numérique.

Exemple conceptuel :

```bash
pmg generate \
    --model deepseek-v4-flash \
    --dtype bf16
```

ou :

```bash
pmg generate \
    --model deepseek-v4-flash \
    --dtype fp8
```

ou une configuration hybride :

```text
embeddings → BF16
attention  → FP8
MLP        → INT8
```

PMG doit toutefois refuser les combinaisons incompatibles avec le format ou l'architecture cible.

---

# 24. PILIER 20 — Compatibilité des fichiers de sortie

PMG ne doit pas seulement produire :

```text
model.safetensors
```

Il doit produire un **répertoire modèle complet**.

Exemple :

```text
pseudo-model/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── template_jinja.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
└── model-00004-of-00004.safetensors
```

Les noms exacts dépendent du modèle et de l'organisation utilisée.

---

# 25. PILIER 21 — Validation

La commande :

```bash
pmg validate
```

doit vérifier plusieurs niveaux.

### Niveau format

```text
JSON valide
Safetensors valide
headers cohérents
offsets cohérents
tailles cohérentes
```

### Niveau structure

```text
tensors attendus présents
shapes cohérentes
dtype cohérent
configuration cohérente
```

### Niveau mathématique

Par exemple :

\[
N=\prod_i shape_i
\]

et :

\[
bytes=N\times sizeof(dtype)
\]

pour les formats à taille fixe.

### Niveau modèle

```text
architecture cohérente
nombre de couches cohérent
dimensions compatibles
```

---

# 26. PILIER 22 — Compare

La commande :

```bash
pmg compare
```

ne doit pas comparer les poids complets si ceux-ci ne sont pas téléchargés.

Elle compare les informations disponibles :

```text
configuration
architecture
tensor names
shape
dtype
sizes
metadata
index
header
```

Exemple :

```text
Modèle A
  layers = 80

Modèle B
  layers = 80

Résultat :
  structure compatible

dtype :
  A = BF16
  B = BF16
```

PMG doit clairement afficher :

```text
Comparaison metadata-only
Aucune comparaison des valeurs des poids
```

---

# 27. PILIER 23 — Espec

La commande :

```bash
pmg espec
```

doit produire une analyse technique.

Exemple :

```text
Architecture :
  MoE

Couches :
  80

Dtype :
  BF16

Nombre estimé de paramètres :
  ...

Taille estimée :
  ...

Tensor families :
  ...

Distribution profile :
  ...

Outlier profile :
  ...

Structure low-rank :
  ...

Spectral profile :
  ...
```

Les valeurs inconnues doivent être indiquées comme :

```text
unknown
estimated
inferred
measured
assumed
```

PMG ne doit jamais présenter une estimation comme une mesure réelle.

---

# 28. PILIER 24 — Transparence des informations

Chaque information utilisée par PMG doit idéalement avoir un statut.

Exemple :

```text
SOURCE = metadata
```

```text
SOURCE = documented
```

```text
SOURCE = measured
```

```text
SOURCE = inferred
```

```text
SOURCE = synthetic
```

```text
SOURCE = assumed
```

C'est essentiel pour distinguer :

> ce que PMG sait

de :

> ce que PMG estime.

---

# 29. PILIER 25 — Sécurité

PMG doit être conçu pour minimiser les risques.

Principes :

- pas d'exécution de code provenant des fichiers modèle ;
- validation stricte des JSON ;
- contrôle des tailles ;
- contrôle des offsets ;
- prévention des dépassements d'entiers ;
- contrôle des chemins ;
- protection contre les allocations gigantesques ;
- limites explicites ;
- gestion des erreurs ;
- pas de `unsafe` sans justification.

---

# 30. PILIER 26 — Unsafe

`unsafe` est interdit par défaut.

Il peut être utilisé exceptionnellement lorsqu'une fonctionnalité l'exige.

Exemple potentiel :

```text
memory mapping
SIMD spécialisé
FFI
```

Mais :

```text
unsafe
```

doit être encapsulé derrière une API sûre.

L'utilisateur du module ne doit pas avoir besoin de manipuler directement les invariants `unsafe`.

---

# 31. PILIER 27 — Performance

PMG doit être performant sans sacrifier la correction.

Les optimisations doivent être mesurées.

On ne doit jamais dire :

> « Cette implémentation est plus rapide. »

sans benchmark.

La règle est :

\[
\text{optimisation} \Rightarrow \text{mesure}.
\]

Les fonctions critiques doivent être benchmarkées avec Criterion.

Exemples :

```text
tensor generation
random generation
distribution sampling
quantization
INT4 packing
header generation
streaming write
```

---

# 32. PILIER 28 — Parallélisme

PMG pourra paralléliser les opérations indépendantes.

Conceptuellement :

```text
Layer 0 ─┐
Layer 1 ─┤
Layer 2 ─┼──→ workers
Layer 3 ─┤
Layer 4 ─┘
```

Mais le parallélisme ne doit pas casser la reproductibilité.

Pour cela, les seeds doivent être dérivées de manière déterministe :

\[
seed_{tensor}=H(seed_{global},tensor\_id)
\]

par exemple.

Ainsi, l'ordre d'exécution des workers ne doit pas modifier les données produites.

---

# 33. PILIER 29 — Modèle de mémoire

PMG doit éviter les allocations inutiles.

Les opérations doivent préférer :

```text
streaming
chunking
buffer reuse
bounded allocation
```

plutôt que :

```text
allocation gigantesque
copie
allocation
copie
```

Pour un tensor de taille :

\[
S=100GB,
\]

PMG ne doit pas nécessiter automatiquement :

\[
RAM\geq100GB.
\]

---

# 34. PILIER 30 — Architecture modulaire

Le projet doit être organisé par responsabilités.

Exemple :

```text
crates/
├── pmg-core
├── pmg-io
├── pmg-math
├── pmg-models
├── pmg-validation
└── pmg-cli
```

Une fonctionnalité ne doit pas devenir un énorme fichier central.

La limite :

\[
LOC_{file}\leq500
\]

est obligatoire.

---

# 35. PILIER 31 — Gestion des erreurs

Les erreurs internes doivent être explicites.

Exemple :

```text
InvalidHeader
UnsupportedDType
InvalidShape
OffsetOverflow
SizeMismatch
InvalidConfiguration
UnsupportedArchitecture
TargetSizeImpossible
```

Un message d'erreur doit expliquer :

1. ce qui s'est produit ;
2. pourquoi ;
3. lorsque possible, comment corriger le problème.

---

# 36. PILIER 32 — CLI française

PMG doit être utilisable par un débutant.

Commandes V1 :

```text
pmg help
pmg generate
pmg espec
pmg validate
pmg compare
pmg version
```

Options globales :

```text
-h, --help
-d, --dry-run
-D, --debug
-b, --verbose
```

Exemple :

```bash
pmg generate --model glm-5.2 --size 1G --dtype bf16
```

---

# 37. PILIER 33 — Dry Run

Le mode :

```text
--dry-run
```

ne doit produire aucun modèle final.

Il doit calculer et afficher le plan.

Exemple :

```text
Modèle : GLM-5.2
Taille demandée : 1 GiB
Dtype : BF16

Analyse...
✓ Architecture identifiée
✓ 80 couches planifiées
✓ 1250 tensors planifiés

Taille estimée :
  1.02 GiB

Aucun fichier généré (--dry-run)
```

C'est particulièrement important pour les modèles gigantesques.

---

# 38. PILIER 34 — Verbose et Debug

`--verbose` :

```text
informations supplémentaires
```

`--debug` :

```text
informations internes
logs
étapes détaillées
```

Le debug ne doit pas être nécessaire à l'utilisateur normal.

---

# 39. PILIER 35 — Documentation pour débutants

Tout module public doit être documenté.

Les exemples doivent être exécutables lorsque possible.

Exemple :

```rust
/// Représente le type numérique utilisé pour un tensor.
///
/// # Exemple
///
/// ```
/// use pmg_core::DType;
///
/// let dtype = DType::Bf16;
/// assert_eq!(dtype.size_bytes(), 2);
/// ```
```

Les commentaires internes doivent être en français.

Les noms de code restent en anglais selon les conventions Rust du projet.

---

# 40. PILIER 36 — Tests

Chaque fonctionnalité doit posséder des tests adaptés.

## Tests unitaires

```text
fonction
 ↓
test
```

## Tests d'intégration

```text
module A
 ↓
module B
 ↓
résultat
```

## Tests E2E

```text
commande PMG
 ↓
programme
 ↓
fichiers
 ↓
validation
```

## Benchmarks

```text
fonction critique
 ↓
mesure
 ↓
comparaison
```

---

# 41. PILIER 37 — Tests scientifiques

PMG doit également tester les propriétés statistiques.

Exemple :

Supposons qu'un générateur produise \(N\) valeurs.

La moyenne empirique est :

\[
\hat{\mu}=\frac{1}{N}\sum_{i=1}^{N}x_i.
\]

La variance empirique :

\[
\hat{\sigma}^2=
\frac{1}{N-1}
\sum_{i=1}^{N}(x_i-\hat{\mu})^2.
\]

PMG doit pouvoir vérifier que :

\[
|\hat{\mu}-\mu_{cible}|<\epsilon_\mu
\]

et :

\[
|\hat{\sigma}-\sigma_{cible}|<\epsilon_\sigma
\]

lorsque ces contraintes sont définies.

---

# 42. PILIER 38 — Validation des outliers

Si la probabilité cible d'outlier est :

\[
p
\]

et que nous générons :

\[
N
\]

valeurs, le nombre attendu est :

\[
E[K]=Np.
\]

PMG peut vérifier que le nombre observé reste dans un intervalle statistiquement acceptable.

Cela évite un générateur qui annonce :

```text
outlier_rate = 0.1 %
```

mais qui produit réellement 3 %.

---

# 43. PILIER 39 — Validation de la taille

Avant génération :

\[
S_{estimated}
\]

doit être calculé.

Après génération :

\[
S_{actual}
\]

doit être mesuré.

PMG doit afficher les deux.

Exemple :

```text
Taille demandée : 1.000 GiB
Taille estimée : 1.004 GiB
Taille produite : 1.003 GiB
Écart : +0.3 %
```

---

# 44. PILIER 40 — Reproductibilité

Un résultat PMG doit pouvoir être reproduit.

Une génération doit idéalement être définie par :

\[
G=
f(
model,
profile,
configuration,
dtype,
size,
seed,
version
).
\]

La version PMG doit donc participer à l'identité du résultat.

Exemple :

```text
PMG version : 1.0.x
Model profile : glm-5.2
Seed : 42
Dtype : BF16
Target : 1 GiB
```

---

# 45. PILIER 41 — Profil scientifique

Les paramètres scientifiques doivent être séparés du code.

Exemple conceptuel :

```text
profiles/glm-5.2/statistics.json
```

pourrait contenir :

```text
distribution family
distribution parameters
outlier policy
low-rank parameters
correlation parameters
spectral parameters
```

Le code Rust devient alors le moteur générique.

Le profil devient le modèle scientifique.

---

# 46. PILIER 42 — Pas d'hypothèse cachée

Un développeur ne doit pas écrire :

```text
let outlier_rate = 0.001;
```

simplement parce que :

> « Cela semble réaliste. »

Il faut pouvoir répondre :

> Pourquoi 0.001 ?

La réponse doit être :

- donnée mesurée ;
- résultat publié ;
- résultat expérimental ;
- hypothèse explicitement déclarée ;
- valeur par défaut documentée.

---

# 47. PILIER 43 — Séparation connaissance / hypothèse

PMG doit distinguer :

```text
KNOWN
```

```text
MEASURED
```

```text
INFERRED
```

```text
ASSUMED
```

```text
SYNTHETIC
```

Exemple :

```text
hidden_size = 8192
source = configuration
confidence = exact
```

mais :

```text
outlier_probability = 0.002
source = assumed
confidence = low
```

C'est beaucoup plus honnête scientifiquement.

---

# 48. PILIER 44 — Compatibilité moteur

Le pseudo-modèle doit être conçu pour être consommable par les logiciels ciblés.

Cela signifie que PMG doit respecter :

```text
noms des tensors
shapes
dtypes
configuration
index
format Safetensors
architecture metadata
tokenizer metadata
```

Un fichier `.safetensors` valide mais contenant des noms incorrects n'est pas nécessairement utilisable par un moteur.

La compatibilité doit donc être considérée au niveau :

\[
Format + Structure + Convention + Architecture.
\]

---

# 49. PILIER 45 — Aucun téléchargement implicite

Une commande d'analyse ne doit pas déclencher silencieusement le téléchargement d'un modèle gigantesque.

Exemple :

```bash
pmg espec model/
```

doit rester une opération metadata-first.

Si une opération nécessite une ressource distante, PMG doit l'indiquer.

Exemple :

```text
Requête distante :
  URL : ...
  Méthode : HTTP Range
  Volume prévu : 16 KiB

Continuer ? 
```

selon le mode d'utilisation.

---

# 50. PILIER 46 — Observabilité

PMG doit fournir des informations sur ce qu'il fait.

Exemple :

```text
[INFO] Lecture de config.json
[INFO] Architecture détectée
[INFO] Lecture de l'index Safetensors
[INFO] Construction du Tensor Atlas
[INFO] Planification de 80 couches
[INFO] Génération streaming
[INFO] Écriture du shard 1/8
```

En mode normal, les logs restent concis.

En mode debug, ils deviennent détaillés.

---

# 51. PILIER 47 — Git et collaboration

Branches :

```text
main
develop
feature/*
bugfix/*
hotfix/*
release/*
```

Une fonctionnalité doit être développée dans :

```text
feature/nom-fonctionnalite
```

puis fusionnée vers :

```text
develop
```

après revue.

---

# 52. PILIER 48 — Conventional Commits

Exemples :

```text
feat(pmg-io): add safetensors header parser
```

```text
feat(pmg-math): add student-t generator
```

```text
fix(pmg-io): prevent tensor offset overflow
```

```text
perf(pmg-math): optimize int4 packing
```

```text
test(pmg-validation): add shape consistency tests
```

```text
docs: update development guide
```

---

# 53. PILIER 49 — CI obligatoire

Une Pull Request doit au minimum vérifier :

```text
cargo fmt
cargo clippy
cargo test
cargo doc
build
```

et selon la configuration :

```text
cargo audit
coverage
benchmarks
```

Une PR qui ne respecte pas les règles qualité ne doit pas être fusionnée.

---

# 54. PILIER 50 — Qualité et refactorisation

Une fonction ou un module doit être refactorisé lorsque :

- sa responsabilité devient excessive ;
- il contient trop de duplication ;
- il devient difficile à tester ;
- ses invariants sont difficiles à comprendre ;
- il dépasse la limite de 500 lignes ;
- sa complexité devient excessive.

La règle n'est pas :

> « faire le moins de fichiers possible ».

La règle est :

> **Une responsabilité claire par module.**

---

# 55. PILIER 51 — Dépendances minimales

PMG ne doit pas devenir dépendant d'un écosystème ML massif.

Sont interdits comme dépendances obligatoires du Core :

```text
PyTorch
TensorFlow
JAX
Transformers
vLLM
CUDA
cuDNN
SciPy
NumPy
```

sauf décision architecturale ultérieure explicitement approuvée.

Les crates Rust doivent être ajoutées seulement lorsqu'elles apportent une valeur réelle.

---

# 56. PILIER 52 — Crates principales envisagées

La sélection finale sera validée lors de la conception du workspace, mais les familles envisagées sont :

```text
clap
serde
serde_json
thiserror
anyhow
rand
rayon
criterion
```

Éventuellement :

```text
HTTP/TLS
compression
memory mapping
```

si réellement nécessaires.

Aucune dépendance ne doit être ajoutée uniquement pour éviter quelques dizaines de lignes de code simples.

---

# 57. PILIER 53 — Workspace Rust

PMG doit utiliser un workspace Cargo.

Structure cible :

```text
Pseudo-Models-Generator/
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── pmg-core/
│   ├── pmg-io/
│   ├── pmg-math/
│   ├── pmg-models/
│   ├── pmg-validation/
│   └── pmg-cli/
├── profiles/
├── tests/
├── benches/
├── docs/
├── research/
├── .github/
└── LICENSE
```

Cette structure pourra être affinée lors de la conception détaillée.

---

# 58. PILIER 54 — Research Python

Python 3.13 est autorisé pour la recherche.

Mais :

```text
PMG Core ≠ Python
```

Python peut être utilisé pour :

```text
analyse statistique
visualisation
calibration
recherche
expérimentation
validation indépendante
```

Puis les résultats validés peuvent être transformés en profils utilisables par Rust.

---

# 59. PILIER 55 — Validation indépendante

Lorsque cela est possible, une propriété importante doit être vérifiée par deux implémentations indépendantes.

Exemple :

```text
Rust PMG
     ↓
résultat
     ↑
Python Research
```

Si les deux obtiennent les mêmes statistiques dans les tolérances prévues, la confiance augmente.

Ce principe est particulièrement utile pour :

- distributions ;
- quantification ;
- packing ;
- calculs statistiques ;
- génération déterministe.

---

# 60. PILIER 56 — Architecture extensible

PMG V1 supporte principalement :

```text
DeepSeek-V4-Flash
GLM-5.2
```

Mais le moteur ne doit pas être codé comme :

```text
if model == "deepseek" { ... }
else if model == "glm" { ... }
```

partout dans le code.

Il faut plutôt avoir :

```text
ModelProfile
     │
     ├── DeepSeekV4FlashProfile
     │
     └── Glm52Profile
```

avec des interfaces communes.

Ainsi V2 pourra ajouter d'autres modèles sans réécrire le moteur.

---

# 61. PILIER 57 — Séparation moteur / profil

Le moteur doit savoir :

```text
comment générer
```

Le profil doit savoir :

```text
quoi générer
```

Exemple :

```text
GeneratorEngine
      +
Glm52Profile
      ↓
Pseudo GLM-5.2
```

et :

```text
GeneratorEngine
      +
DeepSeekV4FlashProfile
      ↓
Pseudo DeepSeek-V4-Flash
```

---

# 62. PILIER 58 — Contrats mathématiques

Chaque algorithme non trivial doit être documenté avec :

```text
Entrées
Sorties
Hypothèses
Invariants
Complexité
Limites
```

Exemple :

```text
Algorithme : low-rank synthesis

Entrées :
m, n, r, seed

Sortie :
W ∈ R^(m×n)

Construction :
W = UVᵀ + R

Complexité :
O(mnr) pour la génération directe

Mémoire :
O((m+n)r + B)
avec génération par blocs.
```

---

# 63. PILIER 59 — Pas de pseudo-science

PMG ne doit jamais utiliser des termes comme :

```text
"physiquement identique"
"exactement identique"
"poids reconstruits"
"vrais poids"
```

si aucune démonstration ne permet de les justifier.

Les termes acceptables sont :

```text
représentatif
synthétique
statistiquement calibré
structurellement compatible
approximativement similaire
estimé
inféré
```

---

# 64. PILIER 60 — Principe ultime

Le principe directeur de PMG est :

\[
\boxed{
\text{Ne jamais inventer une connaissance que PMG ne possède pas.}
}
\]

Lorsqu'une donnée est connue :

```text
la mesurer / la lire.
```

Lorsqu'elle est calculable :

```text
la calculer.
```

Lorsqu'elle est inférable :

```text
l'inférer et le signaler.
```

Lorsqu'elle est inconnue :

```text
la modéliser et le signaler comme hypothèse.
```

Lorsqu'elle est synthétique :

```text
la marquer comme synthétique.
```

---

# 65. Architecture conceptuelle finale

Le fonctionnement global de PMG V1 peut être résumé ainsi :

```text
                 MODÈLE RÉEL
                      │
                      │
          ┌───────────┴───────────┐
          │                       │
     Configurations          Safetensors
          │                       │
          │                Header/Metadata
          │                  autorisé
          │                       │
          └───────────┬───────────┘
                      ▼
               Metadata Parser
                      │
                      ▼
              Model Blueprint
                      │
                      ▼
               Tensor Atlas
                      │
                      ▼
             Statistical Profile
                      │
          ┌───────────┼────────────┐
          │           │            │
      Distributions Outliers   Low-Rank
          │           │            │
          └───────────┼────────────┘
                      ▼
               Tensor Generator
                      │
                      ▼
                Quantization
                      │
                      ▼
                 Streaming
                      │
                      ▼
              Safetensors Writer
                      │
                      ▼
              Pseudo-Model Folder
```

---

# 66. Les trois grandes couches de PMG

Pour simplifier l'ensemble du projet, PMG peut être considéré comme trois grandes couches.

## Couche A — Observation

```text
config
index
metadata
header
tokenizer
architecture
```

Elle répond :

> Que savons-nous du modèle ?

## Couche B — Modélisation

```text
statistics
distribution
correlation
outliers
low-rank
spectral
routing
quantization
```

Elle répond :

> Comment représenter synthétiquement ce que nous savons ?

## Couche C — Synthèse

```text
tensor generation
packing
streaming
safetensors
configuration
validation
```

Elle répond :

> Comment produire le pseudo-modèle ?

---

# 67. Critère de réussite de PMG V1

PMG V1 ne sera pas considéré comme réussi simplement parce qu'il peut produire un fichier de plusieurs gigaoctets.

Il devra satisfaire simultanément :

\[
\boxed{
Structure
+
Format
+
Cohérence
+
Statistiques
+
Numérique
+
Reproductibilité
+
Compatibilité
}
\]

Un pseudo-modèle de 1 Go qui ne peut être chargé par les logiciels ciblés n'est pas un succès.

Un pseudo-modèle parfaitement valide mais statistiquement trivial n'est pas un succès complet.

Un modèle statistiquement réaliste mais contenant des shapes incorrectes n'est pas un succès.

---

# 68. Définition finale du PMG V1

PMG V1 est donc :

> **un générateur de pseudo-modèles synthétiques, écrit en Rust, capable d'analyser les métadonnées et structures accessibles d'un modèle cible, d'en construire une représentation intermédiaire, de générer des tensors synthétiques selon des profils statistiques et structurels documentés, de les encoder dans les formats appropriés, et de produire un répertoire de modèle cohérent et exploitable par les outils compatibles.**

Les deux profils officiellement ciblés pour V1 sont :

```text
DeepSeek-V4-Flash
GLM-5.2
```

La génération doit pouvoir être contrôlée par :

```text
modèle
taille cible
dtype
quantification
seed
profil
configuration
```

tout en respectant le principe fondamental :

\[
\boxed{
\text{PMG ne prétend jamais connaître ce qu'il n'a pas observé.}
}
\]

---

# 69. Checklist officielle avant chaque fonctionnalité

Avant de fusionner une fonctionnalité PMG, le développeur doit pouvoir répondre :

```text
[ ] Quelle est sa responsabilité ?
[ ] Quelle est son entrée ?
[ ] Quelle est sa sortie ?
[ ] Quels sont ses invariants ?
[ ] Quelle est sa complexité ?
[ ] Existe-t-il une démonstration mathématique si nécessaire ?
[ ] Existe-t-il un test ?
[ ] Existe-t-il un benchmark si la performance est importante ?
[ ] Les erreurs sont-elles correctement gérées ?
[ ] Les commentaires sont-ils en français ?
[ ] La documentation publique existe-t-elle ?
[ ] Le fichier fait-il ≤ 500 lignes ?
[ ] Clippy passe-t-il sans warning ?
[ ] rustfmt passe-t-il ?
[ ] cargo test passe-t-il ?
[ ] La fonctionnalité respecte-t-elle la politique Zero-Payload ?
[ ] Les hypothèses sont-elles explicitement identifiées ?
[ ] Aucune information inventée n'est-elle présentée comme réelle ?
```

---

# 70. Conclusion

Les piliers de PMG imposent une philosophie simple :

```text
OBSERVATION
     ↓
COMPRÉHENSION
     ↓
MODÉLISATION
     ↓
SYNTHÈSE
     ↓
VALIDATION
```

et non :

```text
CONFIGURATION
     ↓
NOMBRES ALÉATOIRES
     ↓
FICHIER .SAFETENSORS
```

La différence entre ces deux approches constitue précisément la différence entre un **générateur de gros fichiers** et un véritable **Pseudo-Models Generator**.

PMG doit donc être considéré simultanément comme :

- un logiciel système ;
- un moteur de génération tensorielle ;
- un moteur statistique ;
- un analyseur de modèles ;
- un générateur de formats de modèles ;
- un laboratoire de modélisation synthétique ;
- et un outil de test pour les systèmes de traitement des LLM.

**Ce cahier constitue la base normative de l'architecture PMG V1.**