# CAHIER TECHNIQUE
# Pseudo-Models Generator — PMG

**Version : 1.0**  
**Statut : Document technique de référence**  
**Licence : GPL-3.0**  
**Langue de l'interface : Français**  
**Langage d'implémentation : Rust**  
**Cibles V1 : DeepSeek-V4-Flash et GLM-5.2**

---

# 1. Objet du cahier technique

PMG — **Pseudo-Models Generator** — est un logiciel Rust destiné à générer des modèles de substitution (*pseudo-models*) à partir des informations structurelles et métadonnées d'un modèle LLM réel.

PMG doit permettre à des logiciels de :

- compression ;
- quantification ;
- optimisation ;
- inspection ;
- conversion ;
- chargement ;
- validation ;
- benchmarking structurel ;
- expérimentation de moteurs d'inférence ;

de travailler sur un modèle mannequin sans nécessiter le téléchargement des gigaoctets ou téraoctets de poids réels.

La V1 prend exclusivement en charge :

1. **DeepSeek-V4-Flash**
2. **GLM-5.2**

---

# 2. Principe scientifique fondamental

## 2.1 Ce que PMG peut connaître

Les informations suivantes peuvent être déterminées sans lire les valeurs des poids :

\[
\mathcal{M}_{structure}
=
\{
architecture,
dimensions,
noms,
shapes,
dtypes,
shards,
offsets,
nombre\ de\ paramètres,
métadonnées
\}
\]

Les configurations Hugging Face exposent notamment des propriétés comme `hidden_size`, `num_attention_heads`, `num_hidden_layers` et `vocab_size`.

Pour Safetensors, l'en-tête contient notamment :

```text
dtype
shape
data_offsets
```

et éventuellement :

```text
__metadata__
```

Le format commence par 8 octets indiquant la taille du header, suivis du JSON du header, puis du buffer binaire contenant les données des tenseurs.

---

# 3. Limite mathématique fondamentale

Soit un modèle réel :

\[
f_\theta(x)
\]

où :

- \(x\) = entrée ;
- \(\theta\) = ensemble des poids ;
- \(f\) = architecture.

Les fichiers de configuration permettent principalement de connaître :

\[
A = architecture(\theta)
\]

mais pas :

\[
\theta
\]

Deux modèles peuvent posséder exactement la même architecture :

\[
A_1=A_2
\]

tout en ayant :

\[
\theta_1\neq\theta_2
\]

et donc :

\[
f_{\theta_1}(x)\neq f_{\theta_2}(x)
\]

pour certaines entrées \(x\).

**Conclusion :**

PMG ne doit jamais prétendre que son pseudo-model est numériquement identique au modèle original.

La terminologie technique officielle de PMG sera :

> **Surrogate Model / Pseudo-Model structurel et statistique**

et non :

> copie exacte des poids.

---

# 4. Modèle de fidélité PMG

PMG doit distinguer plusieurs niveaux de fidélité.

## 4.1 Fidélité structurelle

Objectif :

\[
F_s \rightarrow 1
\]

Elle mesure la correspondance entre :

- architecture ;
- nombre de couches ;
- dimensions ;
- tenseurs ;
- shapes ;
- noms ;
- dtypes ;
- shards ;
- paramètres déclarés.

Cette fidélité doit être **exacte** lorsque les données sources sont disponibles.

---

## 4.2 Fidélité de distribution

On cherche :

\[
P_{PMG}(W)\approx P_{real}(W)
\]

où \(W\) représente les valeurs des poids.

PMG ne connaît pas directement \(P_{real}\). Il doit donc construire une approximation contrôlée.

---

## 4.3 Fidélité des moments

Pour chaque famille de tenseurs :

\[
\mu = E[X]
\]

\[
\sigma^2 = E[(X-\mu)^2]
\]

PMG doit pouvoir générer une distribution possédant :

- moyenne ;
- variance ;
- asymétrie ;
- kurtosis ;
- queues de distribution ;
- proportion d'outliers.

---

## 4.4 Fidélité structurelle des matrices

Pour une matrice :

\[
W\in\mathbb{R}^{m\times n}
\]

PMG doit pouvoir reproduire approximativement :

\[
rank(W)
\]

ou plus précisément son **rang numérique effectif**.

Une mesure utile est :

\[
r_{eff}(\epsilon)
=
\min
\left\{
r:
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}
\geq 1-\epsilon
\right\}
\]

où \(\sigma_i\) sont les valeurs singulières.

---

# 5. Architecture générale de PMG

Architecture recommandée :

```text
PMG
│
├── pmg-cli
│
├── pmg-core
│
├── pmg-models
│   ├── deepseek_v4_flash
│   └── glm_5_2
│
├── pmg-config
│
├── pmg-io
│   ├── safetensors
│   ├── http_range
│   └── filesystem
│
├── pmg-math
│   ├── distributions
│   ├── statistics
│   ├── correlation
│   ├── low_rank
│   ├── outliers
│   └── quantization
│
├── pmg-generator
│
├── pmg-validator
│
└── pmg-comparator
```

---

# 6. Responsabilité des crates

## 6.1 `pmg-cli`

Responsable de :

- parsing CLI ;
- affichage français ;
- progression ;
- erreurs utilisateur ;
- logs ;
- `--dry-run` ;
- `--verbose` ;
- `--debug`.

---

## 6.2 `pmg-core`

Contient les types fondamentaux :

```rust
ModelSpec
TensorSpec
LayerSpec
ShardSpec
DType
GenerationConfig
ValidationReport
```

---

## 6.3 `pmg-models`

Implémentations spécifiques :

```text
deepseek_v4_flash/
glm_5_2/
```

Chaque architecture doit avoir un adaptateur indépendant.

---

# 7. DeepSeek-V4-Flash

Les informations actuellement publiées indiquent notamment :

- 43 couches ;
- `hidden_size = 4096` ;
- 256 experts routés ;
- 1 expert partagé ;
- top-6 experts activés ;
- vocabulaire de 129280 tokens ;
- tête d'attention de dimension 512 ;
- mécanismes d'attention hybride ;
- hyper-connections ;
- FP8/FP4 dans la configuration publiée.

La configuration officielle publiée sur Hugging Face contient également `hc_mult = 4`, `hc_sinkhorn_iters = 20`, `index_topk = 512` et une configuration RoPE jusqu'à 1M de tokens.

NVIDIA décrit également DeepSeek-V4-Flash comme un MoE de 284B paramètres totaux avec environ 13B paramètres activés, et décrit son backbone comme ayant 43 couches, 256 experts routés, un expert partagé et un top-6.

PMG doit donc représenter explicitement :

```text
Embedding
│
├── 43 × Transformer Block
│      │
│      ├── Hybrid Attention
│      ├── Hyper-Connection
│      └── MoE
│           ├── 256 routed experts
│           └── 1 shared expert
│
├── MTP
│
└── LM Head
```

---

# 8. GLM-5.2

La configuration publiée de GLM-5.2 indique notamment :

- architecture `GlmMoeDsaForCausalLM` ;
- `hidden_size = 6144` ;
- 78 couches ;
- 256 experts routés ;
- 1 expert partagé ;
- top-8 experts par token ;
- vocabulaire de 154880 ;
- attention à 64 têtes ;
- mécanisme DSA ;
- indexation sparse ;
- IndexShare ;
- 1M de contexte annoncé.



Le dépôt officiel contient également un `chat_template.jinja`, `generation_config.json` et 282 shards Safetensors, avec une taille de dépôt d'environ 1,51 To.

PMG doit donc représenter :

```text
Embedding
│
├── 78 × Transformer Block
│      │
│      ├── DSA
│      ├── Sparse Indexer
│      └── MoE
│           ├── 256 routed experts
│           └── 1 shared expert
│
├── MTP
│
└── LM Head
```

---

# 9. Analyse Safetensors

## 9.1 Structure du fichier

Un fichier Safetensors suit conceptuellement :

```text
+-----------------------+
| header_size : u64 LE  |
+-----------------------+
| JSON header           |
+-----------------------+
| tensor byte buffer    |
+-----------------------+
```

Pour chaque tenseur :

```json
{
  "dtype": "BF16",
  "shape": [4096, 4096],
  "data_offsets": [123456, 789012]
}
```

La taille du tenseur est :

\[
S = END-BEGIN
\]

La taille théorique peut également être calculée :

\[
S =
\left(
\prod_i shape_i
\right)
\times bytes(dtype)
\]

Pour un dtype traditionnel à taille fixe, ces deux valeurs doivent être cohérentes.

---

# 10. Lecture HTTP Range

PMG doit exploiter les requêtes HTTP Range lorsque la source est distante.

La documentation Hugging Face décrit explicitement la méthode :

1. requête `Range: bytes=0-7` ;
2. lecture du `u64` little-endian ;
3. requête du header ;
4. parsing JSON.



Exemple conceptuel :

```text
GET /model-00001-of-00282.safetensors
Range: bytes=0-7
```

Puis :

```text
GET /model-00001-of-00282.safetensors
Range: bytes=8-(7+header_size)
```

PMG ne télécharge alors pas le buffer des poids.

---

# 11. Règle absolue de sécurité PMG

Le moteur d'analyse doit fonctionner avec deux niveaux :

```text
METADATA_ONLY
WEIGHTS_DATA
```

En V1 :

```text
METADATA_ONLY = autorisé
WEIGHTS_DATA   = interdit
```

Même si un utilisateur fournit accidentellement un fichier `.safetensors`, PMG ne doit pas lire son payload.

Il peut éventuellement :

- lire le chemin ;
- identifier le fichier ;
- vérifier son existence ;
- demander son header via Range si distant ;

mais il ne doit jamais charger les valeurs des tenseurs.

---

# 12. Extraction du modèle

PMG construit un modèle intermédiaire :

```rust
pub struct ModelSpec {
    pub architecture: Architecture,
    pub config: ModelConfig,
    pub tokenizer: TokenizerSpec,
    pub tensors: Vec<TensorSpec>,
    pub shards: Vec<ShardSpec>,
}
```

Chaque tenseur :

```rust
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub byte_size: u64,
    pub shard: String,
    pub offset_start: u64,
    pub offset_end: u64,
}
```

---

# 13. Construction du pseudo-tenseur

PMG ne doit pas représenter un pseudo-tenseur comme une simple suite de nombres aléatoires.

Il doit être généré selon :

\[
W =
W_{base}
+
W_{corr}
+
W_{lr}
+
W_{outlier}
+
W_{tail}
\]

où :

- \(W_{base}\) = composante principale ;
- \(W_{corr}\) = corrélations ;
- \(W_{lr}\) = structure bas-rang ;
- \(W_{outlier}\) = anomalies structurées ;
- \(W_{tail}\) = comportement de queue.

---

# 14. Distribution de base

Pour une matrice \(W\), PMG peut commencer avec :

\[
X_{ij}\sim\mathcal{N}(0,\sigma^2)
\]

mais cette approche seule est insuffisante.

Une distribution gaussienne possède une queue légère et ne reproduit pas nécessairement les distributions observées dans les poids réels.

PMG doit donc supporter plusieurs familles :

```text
Gaussian
StudentT
Laplace
LogNormal
Weibull
Pareto
Mixture
Custom
```

---

# 15. Distribution Student-t

Pour reproduire des queues plus lourdes :

\[
X\sim t_\nu
\]

où \(\nu\) contrôle les degrés de liberté.

Lorsque \(\nu\) diminue :

\[
P(|X|>x)
\]

diminue plus lentement qu'une gaussienne.

C'est utile pour modéliser des poids présentant davantage de valeurs extrêmes.

PMG ne doit cependant pas affirmer que les poids de DeepSeek ou GLM suivent précisément une Student-t sans mesure directe des poids.

La distribution sera donc :

> **modèle statistique hypothétique configurable**

et non :

> vérité du modèle original.

---

# 16. Mélanges de distributions

Un modèle plus réaliste peut utiliser :

\[
P(X)
=
\sum_{k=1}^{K}\pi_kP_k(X)
\]

avec :

\[
\sum_k\pi_k=1
\]

Exemple :

```text
95.0 % Gaussian
4.5 % Student-t
0.4 % Laplace
0.1 % Outlier component
```

Les coefficients doivent être déterminés par :

- profils empiriques disponibles ;
- littérature ;
- famille de tenseurs ;
- architecture ;
- niveau de fidélité demandé.

Ils ne doivent jamais être inventés comme étant des statistiques mesurées du modèle réel.

---

# 17. Injection des super-poids

PMG doit implémenter une composante dédiée aux valeurs extrêmes.

Soit :

\[
W_{base}
\]

et un masque :

\[
M_{ij}\in\{0,1\}
\]

Alors :

\[
W_{outlier,ij}
=
M_{ij}\cdot\alpha_{ij}
\]

et :

\[
W'_{ij}
=
W_{ij}
+
W_{outlier,ij}
\]

Une meilleure formulation est multiplicative :

\[
W'_{ij}
=
W_{ij}
\left(
1 + M_{ij}\lambda_{ij}
\right)
\]

PMG appellera cette catégorie :

> **Super-Weights / Critical Outliers**

mais devra distinguer :

```text
statistical outlier
structural outlier
channel outlier
block outlier
synthetic super-weight
```

---

# 18. Pourquoi les outliers sont importants pour PMG

Un compresseur peut avoir un comportement très différent face à :

```text
[0.01, 0.02, 0.01, 0.02]
```

et :

```text
[0.01, 0.02, 8.7, 0.02]
```

Même moyenne approximative ne signifie pas même difficulté de quantification.

Pour un quantificateur uniforme :

\[
q(x)=
round
\left(
\frac{x}{s}
\right)
\]

une valeur extrême peut déterminer :

\[
s=
\frac{x_{max}-x_{min}}
{2^b-1}
\]

et donc modifier la résolution de toute la plage.

C'est pourquoi PMG doit tester explicitement les valeurs extrêmes.

---

# 19. Corrélations

PMG doit pouvoir générer :

\[
W_{corr}
\]

à partir d'une covariance :

\[
\Sigma
\]

avec :

\[
X\sim\mathcal{N}(0,\Sigma)
\]

Si :

\[
\Sigma=LL^T
\]

alors :

\[
X=LZ
\]

avec :

\[
Z\sim\mathcal{N}(0,I)
\]

Cette construction permet d'introduire des corrélations contrôlées.

---

# 20. Structure bas-rang

Une matrice peut être modélisée :

\[
W = UV^T + E
\]

avec :

\[
U\in\mathbb{R}^{m\times r}
\]

\[
V\in\mathbb{R}^{n\times r}
\]

et :

\[
r\ll\min(m,n)
\]

La partie :

\[
UV^T
\]

représente la structure dominante.

Le résidu :

\[
E
\]

représente les détails.

PMG peut ainsi produire des matrices plus réalistes qu'un bruit IID :

```text
W
│
├── Low-rank structure
│
└── Residual noise
     ├── Gaussian
     ├── heavy tail
     └── outliers
```

---

# 21. Modèle synthétique complet

Le générateur principal peut utiliser :

\[
W =
UV^T
+
LZ
+
\sigma E
+
O
\]

où :

- \(UV^T\) = structure bas-rang ;
- \(LZ\) = corrélation ;
- \(E\) = bruit résiduel ;
- \(O\) = outliers.

Une variante plus contrôlable :

\[
W =
\alpha W_{LR}
+
\beta W_{CORR}
+
\gamma W_{BASE}
+
\delta W_{OUT}
\]

avec :

\[
\alpha+\beta+\gamma+\delta=1
\]

---

# 22. Seed déterministe

PMG doit être reproductible.

Pour une même :

```text
model
tensor
seed
generation_profile
dtype
```

le résultat doit être identique.

On définit :

\[
seed_{tensor}
=
H(
seed_{global}
\Vert
model\_id
\Vert
tensor\_name
\Vert
layer\_id
\Vert
generation\_version
)
\]

Ainsi :

```text
tensor A
```

ne doit pas dépendre de l'ordre dans lequel les autres tenseurs ont été générés.

---

# 23. Génération par chunks

PMG ne doit jamais faire :

```text
allocate entire 5 GB tensor
```

pour générer un tenseur massif.

Il doit fonctionner :

```text
tensor
  ↓
chunk 0
  ↓
chunk 1
  ↓
chunk 2
  ↓
...
```

Complexité :

\[
O(N)
\]

où \(N\) est le nombre de valeurs générées.

Mémoire :

\[
O(C)
\]

où \(C\) est la taille du chunk.

Ainsi :

\[
C\ll N
\]

---

# 24. Génération parallèle

Lorsque cela est sûr :

```text
Rayon
 ├── tensor A
 ├── tensor B
 ├── tensor C
 └── tensor D
```

Mais la reproductibilité doit être indépendante de l'ordre d'exécution.

Donc chaque tâche doit recevoir son propre seed dérivé.

---

# 25. Gestion des dtypes

PMG doit représenter explicitement :

```rust
enum DType {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    F6E2M3,
    F6E3M2,
    F4,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    BOOL,
}
```

Le crate Rust `safetensors` actuel expose notamment BOOL, F4, F6, FP8, F16, BF16, F32, F64, entiers et autres variantes ; l'enum est non exhaustif. PMG doit donc prévoir un mécanisme d'extension plutôt qu'un `match` supposant que la liste est définitive.

---

# 26. Attention aux formats quantifiés

PMG ne doit pas confondre :

```text
Safetensors dtype
```

et :

```text
quantization scheme
```

Par exemple :

```text
NF4
GPTQ
AWQ
Q4
```

ne sont pas simplement des dtypes Safetensors universels.

Une représentation quantifiée peut nécessiter :

```text
packed values
+
scale
+
zero point
+
group metadata
```

PMG doit donc modéliser séparément :

```text
StorageDType
QuantizationScheme
ScaleType
BlockShape
ZeroPoint
Packing
```

---

# 27. Quantification

Pour une quantification uniforme :

\[
q =
clip
\left(
round
\left(
\frac{x}{s}
\right),
q_{min},
q_{max}
\right)
\]

Déquantification :

\[
\hat{x}=sq
\]

Erreur :

\[
e=x-\hat{x}
\]

Métriques :

\[
MAE=\frac{1}{N}\sum_i|e_i|
\]

\[
MSE=\frac{1}{N}\sum_i e_i^2
\]

\[
RMSE=\sqrt{MSE}
\]

PMG doit générer des pseudo-tenseurs permettant de tester ces phénomènes.

---

# 28. Test des outliers sur quantification

PMG doit proposer plusieurs profils :

```text
normal
heavy-tail
outlier-heavy
channel-outlier
extreme-outlier
```

Exemple :

```text
Profil normal:
sigma = 1

Profil outlier:
sigma = 1
P(outlier) = 0.001
scale_outlier = 20
```

Cela permet de tester la robustesse d'un compresseur.

---

# 29. Taille cible

PMG doit permettre :

```bash
pmg generate \
  --model deepseek-v4-flash \
  --size 1G
```

ou :

```bash
pmg generate \
  --model glm-5.2 \
  --size 1G
```

Mais cette fonctionnalité possède une contrainte fondamentale.

---

# 30. Deux concepts différents de taille

## Mode A — Architecture fidèle

On conserve :

\[
shape_{PMG}=shape_{original}
\]

et :

\[
params_{PMG}=params_{original}
\]

La taille disque dépend alors du dtype :

\[
S\approx N\times b
\]

où \(b\) est le nombre d'octets par paramètre.

Pour 284B paramètres, même 4 bits/paramètre donnent environ :

\[
284\times10^9\times0.5
\approx142\ GB
\]

avant les autres métadonnées.

Il est donc impossible d'avoir une représentation dense complète de DeepSeek-V4-Flash dans 1 Go.

---

# 31. Mode B — Pseudo-model compact

PMG peut générer un modèle de 1 Go en réduisant :

- nombre de couches ;
- hidden size ;
- experts ;
- vocabulaire ;
- dimensions internes ;
- ou en utilisant une représentation sparse/low-rank.

Mais alors :

\[
shape_{PMG}\neq shape_{original}
\]

et il ne s'agit plus d'un clone structurellement compatible avec le modèle original.

---

# 32. Mode C — Mannequin structurel compressé

C'est le mode recommandé.

Le fichier contient des représentations synthétiques :

```text
LowRank(U,V)
+
Statistics
+
Outlier descriptors
+
Distribution descriptors
+
Quantization metadata
```

au lieu de stocker toutes les valeurs.

Cette représentation peut être extrêmement petite.

Mais elle nécessite un moteur PMG-aware.

Elle ne doit pas être présentée comme un fichier Safetensors dense standard interchangeable avec n'importe quel moteur.

---

# 33. Règle de compatibilité

PMG doit produire deux classes :

### `standard-compatible`

Compatible avec les moteurs attendus :

```text
Transformers
vLLM
SGLang
etc.
```

mais la taille dépend du nombre réel de valeurs stockées.

### `PMG-surrogate`

Optimisé pour :

```text
compression testing
quantization testing
layout testing
benchmarking
```

et nécessite un interpréteur PMG.

---

# 34. Le problème du 1 Go

PMG doit donc refuser silencieusement les contradictions.

Exemple :

```bash
pmg generate \
    --model deepseek-v4-flash \
    --size 1G \
    --compatibility strict
```

Si une représentation dense est demandée :

```text
ERREUR PMG-204

La cible de 1 GiB est incompatible avec la représentation
dense complète de l'architecture demandée.

Paramètres estimés : 284.3B
Budget : 1 GiB

Solutions :
  --mode surrogate
  --mode reduced
  --dtype ...
```

C'est essentiel pour respecter la contrainte de vérité du projet.

---

# 35. Génération des fichiers

PMG doit produire un dossier complet :

```text
PMG-model/
│
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── model.safetensors.index.json
│
├── model-00001-of-000XX.safetensors
├── model-00002-of-000XX.safetensors
│
└── pmg/
    ├── profile.json
    ├── statistics.json
    └── surrogate.json
```

---

# 36. Compatibilité avec le modèle original

PMG doit conserver autant que possible :

```text
architecture
model_type
tensor names
tensor shapes
dtype declarations
tokenizer
special tokens
chat template
generation configuration
weight map
```

La configuration est essentielle parce que Transformers utilise la configuration pour déterminer la classe et les paramètres nécessaires à l'instanciation du modèle.

---

# 37. Tokenizer

PMG ne doit pas synthétiser arbitrairement le tokenizer.

Si :

```text
tokenizer.json
```

est disponible :

```text
copie exacte
```

Si le tokenizer est distant :

```text
metadata-only
```

doit être privilégié selon la politique d'utilisation.

Le tokenizer est fonctionnellement différent des poids : il transforme le texte en IDs de tokens et possède sa propre logique de normalisation, segmentation et décodage.

---

# 38. `model.safetensors.index.json`

Pour un modèle shardé :

```json
{
  "metadata": {
    "total_size": "...",
    "total_parameters": "..."
  },
  "weight_map": {
    "tensor_name": "model-00001-of-000XX.safetensors"
  }
}
```

Le champ `weight_map` associe les noms de tenseurs aux shards. L'interface Hugging Face documente également `metadata` et `weight_map` comme propriétés de l'index Safetensors.

PMG doit recalculer et valider :

\[
N_{parameters}
=
\sum_i
\prod_j shape_{ij}
\]

et :

\[
S_{estimated}
=
\sum_i
size(tensor_i)
\]

---

# 39. Validation d'un index

PMG doit vérifier :

### Invariant 1

Chaque tenseur possède un shard.

### Invariant 2

Chaque shard déclaré existe dans l'index.

### Invariant 3

Aucun tenseur n'est dupliqué.

### Invariant 4

Les shapes sont valides.

### Invariant 5

Les offsets sont monotones lorsque requis.

### Invariant 6

Les tailles sont cohérentes avec le dtype.

---

# 40. Inspection (`espec`)

La commande :

```bash
pmg espec ./model
```

doit produire :

```text
╔══════════════════════════════════════════════╗
║ PMG — Inspection du modèle                  ║
╚══════════════════════════════════════════════╝

Architecture : DeepseekV4ForCausalLM
Couches       : 43
Hidden size   : 4096
Experts       : 256
Experts/token : 6
Vocabulaire   : 129280

Paramètres :
  Total       : ~284.3B
  Activés     : ~13B

Stockage :
  Shards      : ...
  Dtype       : ...
  Taille      : ...

Attention :
  Type        : Hybrid
  Index heads : 64
  Top-K       : 512
```

---

# 41. Validation (`validate`)

La commande :

```bash
pmg validate ./model
```

doit produire un rapport :

```text
[OK] config.json
[OK] tokenizer.json
[OK] generation_config.json
[OK] index.json
[OK] tensor names
[OK] tensor shapes
[OK] shard mapping
[OK] dtype consistency
[OK] parameter count
[WARN] synthetic statistics
```

---

# 42. Comparaison (`compare`)

PMG doit distinguer :

```text
STRUCTURAL COMPARE
```

de :

```text
WEIGHT VALUE COMPARE
```

Le second est interdit en V1.

Donc :

```bash
pmg compare original/ pseudo/
```

compare :

```text
config
tensor names
shapes
dtypes
shards
parameter count
metadata
tokenizer metadata
generation config
```

mais pas :

```text
W_original - W_pseudo
```

puisque \(W_{original}\) n'est pas lu.

---

# 43. Rapport de comparaison

Exemple :

```text
Architecture       : IDENTIQUE
Nombre de couches  : IDENTIQUE
Hidden size        : IDENTIQUE
Tensor names       : 100 %
Shapes             : 100 %
Dtypes             : 100 %
Parameter count    : 100 %
Tokenizer          : IDENTIQUE
Poids numériques   : NON COMPARÉS
```

---

# 44. Commande `generate`

Syntaxe recommandée :

```bash
pmg generate \
    --model deepseek-v4-flash \
    --source ./model \
    --output ./pseudo \
    --size 1G \
    --dtype bf16 \
    --profile realistic \
    --seed 42
```

---

# 45. Commande `help`

```bash
pmg help
```

doit expliquer :

```text
Débutant
│
├── inspecter
├── générer
├── valider
└── comparer
```

Exemple :

```bash
pmg help generate
```

---

# 46. Flags globaux

PMG doit utiliser :

```text
-h, --help
-d, --dry-run
--debug
-b, --verbose
```

Attention : le cahier précédent utilisait deux fois `-h`. Cela doit être corrigé.

La convention recommandée est :

```text
-h = help
-d = dry-run
-b = verbose
--debug = debug
```

Le debug ne doit pas recevoir `-h`.

---

# 47. `--dry-run`

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1G \
    --dry-run
```

PMG calcule :

```text
Architecture
Tensor count
Estimated parameters
Estimated storage
Chunk size
Number of shards
Generation time estimate
```

mais ne crée aucun poids.

---

# 48. `--verbose`

Affiche :

```text
[INFO] Chargement config.json
[INFO] 78 couches détectées
[INFO] 256 experts
[INFO] Construction du profil statistique
[INFO] Génération du shard 12/282
```

---

# 49. `--debug`

Ajoute :

```text
seed
tensor seed
distribution parameters
chunk boundaries
byte offsets
internal invariants
RNG state identifiers
```

Les logs doivent rester en français.

---

# 50. Validation mathématique du générateur

Chaque génération doit produire un rapport statistique :

```text
mean
std
min
max
median
MAD
skewness
kurtosis
quantiles
outlier_rate
```

Pour un échantillon \(x_1,\ldots,x_n\) :

\[
\bar{x}
=
\frac{1}{n}
\sum_i x_i
\]

Variance :

\[
s^2
=
\frac{1}{n-1}
\sum_i(x_i-\bar{x})^2
\]

---

# 51. Détection d'outliers

PMG doit permettre plusieurs méthodes.

### Z-score

\[
z_i=
\frac{x_i-\mu}{\sigma}
\]

Par exemple :

\[
|z_i|>5
\]

peut définir un outlier synthétique.

### Méthode IQR

\[
IQR=Q_3-Q_1
\]

Outlier :

\[
x<Q_1-1.5IQR
\]

ou :

\[
x>Q_3+1.5IQR
\]

---

# 52. Métriques de corrélation

Pour deux variables :

\[
\rho_{XY}
=
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
\]

PMG peut générer une matrice :

\[
R\in[-1,1]^{n\times n}
\]

qui doit être symétrique :

\[
R=R^T
\]

et idéalement semi-définie positive :

\[
x^TRx\geq0
\]

pour tout \(x\).

---

# 53. Métriques de compression

Pour chaque pseudo-tenseur :

```text
original theoretical size
pseudo size
compression ratio
bits/parameter
sparsity
zero ratio
outlier ratio
entropy estimate
```

Ratio :

\[
CR=
\frac{S_{original}}
{S_{compressed}}
\]

---

# 54. Entropie

Pour une distribution discrète :

\[
H(X)
=
-\sum_i p_i\log_2p_i
\]

PMG doit permettre de générer des tenseurs dont la distribution présente une entropie non triviale.

Un tenseur composé presque uniquement de zéros est un mauvais mannequin pour certains tests de compression.

---

# 55. Sparsité

Définition :

\[
S=
\frac{\#\{x_i=0\}}
{N}
\]

PMG doit pouvoir contrôler :

```text
dense
semi-sparse
sparse
structured-sparse
```

---

# 56. Structure des MoE

PMG ne doit pas générer les experts comme 256 matrices indépendantes totalement IID.

Il doit pouvoir modéliser :

```text
shared component
+
expert-specific component
+
expert correlation
```

Par exemple :

\[
W_e=W_{shared}+\Delta W_e
\]

avec :

\[
\Delta W_e
=
U_eV_e^T+E_e
\]

Cela permet de tester des compresseurs sensibles aux similarités inter-experts.

---

# 57. Routing MoE synthétique

Pour \(E\) experts et top-\(k\) :

\[
g(x)\in\mathbb{R}^{E}
\]

PMG peut générer des logits synthétiques :

\[
g_i(x)
\]

puis :

\[
TopK(g,k)
\]

Le modèle doit conserver :

\[
k=6
\]

pour DeepSeek-V4-Flash et :

\[
k=8
\]

pour GLM-5.2, conformément aux configurations publiées.

---

# 58. Tests de déséquilibre MoE

PMG doit pouvoir simuler :

```text
balanced routing
moderate imbalance
severe imbalance
expert collapse
```

Une mesure simple :

\[
L_{imbalance}
=
\sum_e
(p_e-\frac1E)^2
\]

où \(p_e\) est la proportion d'utilisation de l'expert \(e\).

---

# 59. Attention sparse/hybrid

Pour DeepSeek-V4-Flash, les configurations publiées exposent notamment des ratios de compression d'attention et une architecture hybride.

Pour GLM-5.2, la configuration expose une architecture `glm_moe_dsa` avec indexation sparse et réutilisation de l'indexer.

PMG doit donc traiter l'attention comme une propriété d'architecture, et non comme une simple matrice `QKV`.

---

# 60. Modèle de données interne

```text
ModelSpec
│
├── ArchitectureSpec
│
├── ConfigSpec
│
├── TokenizerSpec
│
├── GenerationSpec
│
├── LayerSpecs[]
│
│   ├── AttentionSpec
│   ├── MoESpec
│   ├── NormSpec
│   └── MtpSpec
│
├── TensorSpecs[]
│
└── ShardSpecs[]
```

---

# 61. Pipeline complet

```text
SOURCE
  │
  ▼
Discovery
  │
  ▼
Configuration parser
  │
  ▼
Safetensors index parser
  │
  ▼
HTTP Range metadata parser
  │
  ▼
Canonical ModelSpec
  │
  ├───────────────┐
  ▼               ▼
Statistics      Architecture
  │               │
  └───────┬───────┘
          ▼
     Generator
          │
          ├── distribution
          ├── low-rank
          ├── correlation
          ├── outliers
          ├── sparsity
          └── quantization
          │
          ▼
   Streaming writer
          │
          ▼
   Safetensors shards
          │
          ▼
   Validation
          │
          ▼
     Final model
```

---

# 62. Écriture Safetensors

Le writer doit fonctionner en streaming.

Étapes :

```text
1. Construire TensorSpec
2. Calculer taille
3. Réserver offsets
4. Générer header
5. Écrire header
6. Générer chunk
7. Encoder dtype
8. Écrire chunk
9. Continuer
10. Finaliser
```

Le format Safetensors impose que les offsets décrivent les données dans le buffer et que le buffer soit entièrement indexé, sans trous.

---

# 63. Invariant de taille

Pour chaque tenseur :

\[
END-BEGIN
=
generated\_bytes
\]

Et pour les dtypes à taille fixe :

\[
generated\_bytes
=
\prod_i shape_i\times bytes(dtype)
\]

Tout écart doit provoquer :

```text
ERROR
```

et non une correction silencieuse.

---

# 64. Vérification des offsets

Pour les tenseurs :

\[
T_1,T_2,\ldots,T_n
\]

PMG doit vérifier :

\[
begin_1=0
\]

et :

\[
end_i=begin_{i+1}
\]

lorsque les tenseurs sont écrits contiguës sans trous.

Finalement :

\[
end_n=buffer\_size
\]

---

# 65. Gestion des shards

Si la taille cible impose plusieurs fichiers :

```text
model-00001-of-000XX.safetensors
...
model-000XX-of-000XX.safetensors
```

PMG doit mettre à jour :

```json
weight_map
```

et :

```json
metadata.total_size
```

sans divergence.

---

# 66. Gestion mémoire

Objectif :

\[
Memory_{PMG}\ll Size_{model}
\]

PMG ne doit jamais avoir besoin de charger :

```text
1.5 TB
```

pour construire les métadonnées d'un modèle de cette taille.

La lecture du header Safetensors est précisément conçue pour permettre l'inspection des métadonnées sans charger les données des tenseurs.

---

# 67. Tolérance réseau

HTTP Range doit gérer :

```text
206 Partial Content
```

mais PMG doit aussi détecter :

```text
200 OK
```

lorsqu'un serveur ignore Range.

Dans ce cas :

```text
PMG ne doit PAS télécharger le fichier entier.
```

Il doit retourner :

```text
ERR_RANGE_UNSUPPORTED
```

sauf si l'utilisateur a explicitement autorisé une autre stratégie.

---

# 68. Cache HTTP

PMG peut mettre en cache :

```text
URL
ETag
Last-Modified
header-size
header-hash
```

mais pas les poids.

Le cache peut être :

```text
~/.cache/pmg/metadata/
```

---

# 69. Hash des métadonnées

PMG doit calculer :

\[
H_{metadata}
=
SHA256(canonical\_metadata)
\]

Ce hash identifie le profil observé.

Il ne doit pas être appelé :

```text
hash of model weights
```

car les poids ne sont pas lus.

---

# 70. Profil de génération

Exemple :

```json
{
  "profile": "realistic",
  "seed": 42,
  "distribution": "hybrid",
  "low_rank": true,
  "correlation": true,
  "outliers": true,
  "sparsity": true,
  "dtype": "bf16"
}
```

PMG doit versionner ce profil.

---

# 71. Versionnement des générateurs

Le seed seul ne suffit pas.

Le résultat dépend de :

\[
R=
F(
model,
seed,
profile,
generator\_version,
dtype
)
\]

Donc :

```text
PMG 1.0 + seed 42
```

et :

```text
PMG 1.1 + seed 42
```

peuvent produire des résultats différents.

C'est acceptable et doit être documenté.

---

# 72. Tests unitaires essentiels

### Test 1

```text
shape → byte size
```

### Test 2

```text
dtype → byte size
```

### Test 3

```text
offsets contiguous
```

### Test 4

```text
parameter count
```

### Test 5

```text
seed determinism
```

### Test 6

```text
distribution moments
```

### Test 7

```text
outlier injection
```

### Test 8

```text
low-rank reconstruction
```

---

# 73. Test de reproductibilité

```text
generate(seed=42)
generate(seed=42)
```

doit donner :

\[
H(output_1)=H(output_2)
\]

à condition que :

- même version ;
- même architecture ;
- même configuration ;
- même plateforme compatible ;
- même profil.

---

# 74. Test d'intégrité Safetensors

Pour chaque fichier généré :

```text
parse header
↓
validate JSON
↓
validate dtype
↓
validate shape
↓
validate offsets
↓
validate byte size
```

Puis :

```text
read metadata only
```

pour vérifier que le fichier peut être interprété.

---

# 75. Test de compatibilité

PMG doit fournir des tests avec les loaders ciblés lorsque possible :

```text
Transformers
vLLM
SGLang
```

mais sans confondre :

```text
configuration compatible
```

et :

```text
fonctionnement numérique équivalent
```

---

# 76. Benchmark de compression

Le benchmark PMG doit mesurer :

```text
compression ratio
memory
throughput
latency
error
outlier preservation
```

Pour chaque méthode :

```text
FP32
BF16
FP16
FP8
INT8
INT4
```

lorsque le format et le moteur les supportent.

---

# 77. Benchmark de quantification

Pour chaque tenseur :

\[
MSE
\]

\[
MAE
\]

\[
MaxError
\]

\[
PSNR
\]

et :

\[
OutlierRetention
=
\frac{
\#outliers\ retained
}{
\#outliers\ original/synthetic
}
\]

---

# 78. Objectif réel du pseudo-model

Le pseudo-model ne doit pas essayer de reproduire :

```text
la connaissance exacte
```

mais plutôt les propriétés nécessaires au test :

```text
forme
distribution
dynamique
outliers
corrélation
rang
sparsité
quantification
routing
taille
layout
I/O
```

C'est cette distinction qui rend le projet scientifiquement défendable.

---

# 79. Architecture de profils

PMG doit proposer :

```text
--profile structural
--profile statistical
--profile realistic
--profile compression
--profile quantization
--profile stress
```

### Structural

Priorité :

\[
F_s
\]

### Statistical

Priorité :

\[
P(W)
\]

### Realistic

Combinaison :

\[
F_s + P(W)+correlation+outliers+lowrank
\]

### Compression

Accent sur :

```text
entropy
redundancy
low-rank
outliers
```

### Quantization

Accent sur :

```text
range
tails
outliers
block statistics
```

### Stress

Accent sur :

```text
worst-case
extreme outliers
high kurtosis
imbalanced routing
```

---

# 80. Principe de vérité des statistiques

PMG doit classer toutes les statistiques en trois catégories :

### `OBSERVED`

Observées directement dans une donnée autorisée.

### `DERIVED`

Calculées mathématiquement à partir de métadonnées observées.

### `SYNTHETIC`

Produites par le générateur.

Exemple :

```text
parameter_count = DERIVED
tensor_shape = OBSERVED
mean_weight = UNKNOWN
outlier_rate = SYNTHETIC
```

C'est une règle essentielle.

---

# 81. Interdiction de fabriquer de fausses métadonnées

PMG ne doit jamais écrire :

```json
{
  "weight_mean": 0.0031
}
```

en laissant croire que cette valeur provient du modèle réel si elle est synthétique.

Il doit écrire :

```json
{
  "value": 0.0031,
  "origin": "synthetic",
  "generator": "PMG-1.0"
}
```

---

# 82. Séparation des données

Le modèle final doit distinguer :

```text
Original metadata
Derived metadata
Synthetic metadata
```

Exemple :

```text
pmg/
├── observed.json
├── derived.json
└── synthetic.json
```

---

# 83. Journal scientifique

Chaque génération doit pouvoir produire :

```text
generation_report.json
```

contenant :

```text
model
architecture
source metadata hash
generator version
seed
profile
dtype
size target
actual size
statistics
synthetic assumptions
warnings
```

---

# 84. Niveau de confiance

PMG peut calculer un score de confiance :

\[
C=
w_sF_s+
w_dF_d+
w_cF_c+
w_oF_o+
w_rF_r
\]

où :

- \(F_s\) = fidélité structurelle ;
- \(F_d\) = fidélité distributionnelle ;
- \(F_c\) = fidélité corrélationnelle ;
- \(F_o\) = fidélité des outliers ;
- \(F_r\) = fidélité du rang.

Mais lorsqu'aucune observation des poids n'est disponible :

\[
F_d,F_c,F_o,F_r
\]

doivent être explicitement marqués comme **estimés**, pas mesurés.

---

# 85. Contraintes V1

La V1 ne doit pas :

- lire les valeurs des `.safetensors` ;
- télécharger les payloads via Range ;
- prétendre reconstruire les poids ;
- prétendre reproduire exactement les sorties ;
- prétendre reproduire exactement la connaissance du modèle ;
- inventer des statistiques comme étant observées.

---

# 86. V1 doit pouvoir

La V1 doit :

- analyser les configurations ;
- analyser les headers Safetensors ;
- analyser les shards ;
- utiliser HTTP Range ;
- reconstruire l'architecture ;
- compter les paramètres ;
- reconstruire les shapes ;
- générer des pseudo-tenseurs ;
- injecter des outliers synthétiques ;
- produire des corrélations ;
- produire des structures bas-rang ;
- produire des distributions à queues lourdes ;
- gérer plusieurs dtypes ;
- écrire des Safetensors ;
- produire les fichiers de configuration ;
- valider les fichiers ;
- comparer structurellement ;
- fonctionner en streaming ;
- fonctionner sans PyTorch.

---

# 87. Dépendances Rust

Dépendances minimales recommandées :

```text
serde
serde_json
thiserror
anyhow
clap
rand
rayon
safetensors
```

Selon les besoins :

```text
reqwest
sha2
indicatif
```

La politique reste :

> Une dépendance doit avoir une justification technique.

---

# 88. Règle `unsafe`

`unsafe` est interdit par défaut.

Une exception doit :

1. être documentée ;
2. être encapsulée ;
3. avoir des invariants explicites ;
4. posséder des tests.

---

# 89. Organisation finale du workspace

```text
Pseudo-Models-Generator/
│
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── README.md
│
├── crates/
│   ├── pmg-cli/
│   ├── pmg-core/
│   ├── pmg-config/
│   ├── pmg-io/
│   ├── pmg-math/
│   ├── pmg-models/
│   ├── pmg-generator/
│   ├── pmg-validator/
│   └── pmg-comparator/
│
├── tests/
├── benches/
│
├── docs/
│   ├── cahier_des_charges.md
│   ├── cahier_technique.md
│   ├── architecture.md
│   └── formats.md
│
└── .github/
    └── workflows/
```

---

# 90. Exemple utilisateur débutant

L'utilisateur possède :

```text
mon-modele/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
├── model.safetensors.index.json
└── ...
```

Il exécute :

```bash
pmg espec ./mon-modele
```

PMG analyse les fichiers.

Puis :

```bash
pmg generate \
  --model glm-5.2 \
  --source ./mon-modele \
  --output ./glm52-pseudo \
  --size 1G
```

PMG répond :

```text
Analyse terminée.

Architecture détectée : GLM-5.2
Paramètres déclarés   : ~753B
Budget demandé        : 1 GiB

Mode sélectionné :
  Pseudo-model compact

Important :
  Ce modèle conserve les propriétés structurelles principales,
  mais ne contient pas les valeurs originales des poids.

Génération...
```

---

# 91. Exemple expert

```bash
pmg generate \
  --model deepseek-v4-flash \
  --source ./metadata \
  --output ./pseudo \
  --size 4G \
  --dtype bf16 \
  --profile compression \
  --distribution student-t \
  --low-rank \
  --correlation \
  --outliers \
  --seed 0xDEADBEEF
```

Le générateur construit alors :

\[
W=
W_{LR}
+
W_{CORR}
+
W_{DIST}
+
W_{OUT}
\]

puis encode le résultat dans le format demandé.

---

# 92. Critères d'acceptation technique V1

PMG V1 sera considéré techniquement valide lorsque :

### A. Architecture

\[
Architecture_{PMG}
=
Architecture_{reference}
\]

pour les propriétés prises en charge.

### B. Shapes

\[
Shape_{PMG}=Shape_{reference}
\]

en mode structurel.

### C. Paramètres

\[
N_{PMG}=N_{reference}
\]

en mode structurel.

### D. Safetensors

Tous les offsets et tailles sont cohérents.

### E. Reproductibilité

Même seed + même version :

\[
Output_1=Output_2
\]

### F. Streaming

La mémoire utilisée reste bornée par la taille des chunks.

### G. No-weight-read

Aucune valeur du payload original n'est lue.

### H. Transparence

Toutes les données synthétiques sont identifiées comme telles.

---

# 93. Critère scientifique principal

Le critère principal de PMG n'est pas :

> « Le pseudo-model produit exactement les mêmes réponses que le modèle réel. »

Cette affirmation serait impossible à garantir sans les poids.

Le critère correct est :

> « Le pseudo-model reproduit suffisamment fidèlement les propriétés structurelles et statistiques pertinentes pour que les outils testés puissent être évalués sur des phénomènes représentatifs du modèle réel. »

Cette définition est testable.

---

# 94. Formulation finale du problème

Le problème PMG peut être formulé comme une optimisation :

\[
\theta_{PMG}
=
\arg\min_{\theta}
D
\left(
\Phi(\theta),
\Phi_{target}
\right)
\]

où :

\[
\Phi
\]

est un vecteur de propriétés observables ou dérivées :

\[
\Phi=
[
architecture,
shapes,
dtypes,
size,
distribution,
rank,
correlation,
sparsity,
outliers,
routing,
quantization
]
\]

et \(D\) une distance pondérée.

Ainsi PMG ne cherche pas à résoudre :

\[
\theta_{PMG}=\theta_{real}
\]

mais :

\[
\Phi(\theta_{PMG})
\approx
\Phi(\theta_{real})
\]

pour les propriétés pertinentes du test.

---

# 95. Conclusion technique

PMG V1 doit être conçu comme un **générateur de mannequins statistiques et structurels de LLM**, et non comme un mécanisme de reconstruction des poids.

La combinaison recommandée est :

\[
\boxed{
PseudoModel
=
Architecture
+
Metadata
+
Distribution
+
LowRank
+
Correlation
+
Outliers
+
Sparsity
+
Quantization
}
\]

avec trois garanties fondamentales :

\[
\boxed{\text{Structure exacte lorsque les métadonnées sont disponibles}}
\]

\[
\boxed{\text{Données synthétiques explicitement identifiées}}
\]

\[
\boxed{\text{Aucune lecture du payload des poids originaux}}
\]

DeepSeek-V4-Flash et GLM-5.2 doivent disposer de générateurs spécialisés, car leurs architectures MoE, leurs mécanismes d'attention et leurs paramètres structurels ne sont pas interchangeables. Les informations publiques actuelles confirment notamment les configurations MoE et les différences importantes entre les deux architectures.

**Ce document constitue la base technique de PMG V1.**