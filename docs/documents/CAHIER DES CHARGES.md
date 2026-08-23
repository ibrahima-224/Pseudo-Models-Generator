# CAHIER DES CHARGES
# Pseudo-Models Generator — PMG

**Version :** 1.0  
**Statut :** Approuvé / Spécification de référence  
**Licence du logiciel :** GPL-3.0  
**Langue de l'interface :** Français  
**Langue du code :** Anglais pour les identifiants Rust  
**Langue des commentaires, logs et documentation :** Français  
**Technologie principale :** Rust  
**Plateformes cibles :** Linux prioritairement, puis autres plateformes compatibles Rust  
**Modèles prioritaires v1 :**
1. GLM-5.2
2. DeepSeek-V4-Flash

---

# 1. Objet du projet

## 1.1 Définition

**PMG — Pseudo-Models Generator** est un logiciel permettant de construire un **pseudo-modèle de LLM** à partir des informations disponibles autour d'un modèle réel, sans avoir besoin de télécharger ni de charger intégralement les fichiers de poids `.safetensors`.

Le pseudo-modèle généré doit reproduire autant que possible :

- la structure du modèle ;
- l'organisation des couches ;
- les tenseurs ;
- les dimensions ;
- les dtypes ;
- les tailles mémoire ;
- les distributions statistiques ;
- les corrélations ;
- les structures de bas-rang ;
- les anomalies/outliers ;
- les propriétés pertinentes pour la compression ;
- les propriétés pertinentes pour la quantification ;
- les propriétés pertinentes pour l'optimisation ;
- les propriétés pertinentes pour certains tests d'inférence ;
- les caractéristiques nécessaires au chargement par les logiciels compatibles.

L'objectif n'est donc pas simplement de créer un fichier contenant des nombres aléatoires.

PMG doit produire un **mannequin numérique statistiquement structuré**.

---

# 2. Problème que PMG cherche à résoudre

Les modèles modernes peuvent atteindre :

- plusieurs dizaines de Go ;
- plusieurs centaines de Go ;
- voire plusieurs To dans certaines configurations.

Pour tester un logiciel de :

- compression ;
- quantification ;
- optimisation ;
- conversion ;
- analyse ;
- partitionnement ;
- parallélisation ;
- gestion mémoire ;
- chargement de modèle ;
- génération de fichiers Safetensors ;

il n'est pas toujours nécessaire de disposer des poids réels.

Cependant, un simple tenseur rempli de nombres aléatoires n'est pas suffisamment représentatif.

Exemple :

```text
Modèle réel
    ↓
Poids structurés
    ↓
corrélations
    ↓
outliers
    ↓
faible rang
    ↓
distributions non gaussiennes
    ↓
blocs avec statistiques différentes
    ↓
quantification non uniforme
```

Un générateur naïf :

```text
randn()
   ↓
tensor
```

ne reproduit pas correctement ces propriétés.

PMG doit donc générer un modèle synthétique **structurellement et statistiquement contrôlé**.

---

# 3. Principe scientifique fondamental

## 3.1 Ce que PMG peut connaître

Sans lire les poids, PMG peut exploiter :

- `config.json` ;
- `model.safetensors.index.json` ;
- `tokenizer.json` ;
- `tokenizer_config.json` ;
- `special_tokens_map.json` ;
- fichiers de génération ;
- templates ;
- métadonnées du dépôt ;
- noms des tenseurs ;
- formes des tenseurs ;
- dtypes déclarés lorsqu'ils sont disponibles ;
- organisation des shards ;
- tailles annoncées ;
- métadonnées accessibles par les fichiers associés ;
- éventuellement des informations récupérées via HTTP Range sur les fichiers Safetensors, sous réserve de ne télécharger que les zones nécessaires.

---

# 4. Limitation fondamentale : absence des poids

## 4.1 Principe d'impossibilité

Soit un modèle réel :

\[
M=(A,W,T)
\]

avec :

- \(A\) = architecture ;
- \(W\) = poids ;
- \(T\) = tokenizer et paramètres associés.

PMG observe :

\[
O=(A,T,H)
\]

où \(H\) représente les métadonnées observables des poids.

Il existe généralement plusieurs ensembles de poids :

\[
W_1 \neq W_2
\]

tels que :

\[
O(A,W_1,T)=O(A,W_2,T)
\]

Si deux modèles ont exactement les mêmes métadonnées mais des poids différents, aucune fonction déterministe :

\[
f(O)
\]

ne peut reconstruire simultanément \(W_1\) et \(W_2\).

Donc :

\[
f(O)\neq W
\]

en général.

### Conclusion

PMG **ne doit jamais prétendre reconstruire les poids originaux**.

Il doit générer :

\[
\hat W \sim P(W|O)
\]

c'est-à-dire un ensemble de poids synthétiques compatible avec les informations observables et avec un modèle statistique de génération.

---

# 5. Définition officielle du pseudo-modèle PMG

Un pseudo-modèle PMG est :

\[
\hat M=(A,\hat W,T,C)
\]

où :

- \(A\) = architecture réelle connue ;
- \(\hat W\) = poids synthétiques ;
- \(T\) = tokenizer ;
- \(C\) = métadonnées et informations de génération.

Le pseudo-modèle doit être :

### 5.1 Structurellement compatible

Les tenseurs doivent respecter :

- noms ;
- dimensions ;
- relations ;
- organisation ;
- architecture ;
- paramètres nécessaires au chargement.

### 5.2 Statistiquement réaliste

Les distributions générées doivent être contrôlées.

### 5.3 Numériquement réaliste

Les valeurs doivent respecter :

- plage ;
- précision ;
- densité ;
- dispersion ;
- asymétrie ;
- queues de distribution ;
- corrélations.

### 5.4 Systémiquement réaliste

Le modèle doit être utilisable par les outils qui attendent un modèle réel.

---

# 6. Objectifs de PMG v1.0

## 6.1 Objectif principal

Permettre de générer un pseudo-modèle compatible avec la structure de :

- GLM-5.2 ;
- DeepSeek-V4-Flash.

---

## 6.2 Objectifs secondaires

PMG doit permettre :

1. d'inspecter un modèle ;
2. d'analyser ses métadonnées ;
3. de générer un pseudo-modèle ;
4. de choisir sa taille cible ;
5. de choisir les dtypes ;
6. de produire plusieurs shards ;
7. de générer les fichiers de configuration ;
8. de valider le pseudo-modèle ;
9. de comparer métadonnées et structures ;
10. de simuler une opération avant génération réelle.

---

# 7. Non-objectifs de PMG v1

PMG v1 ne doit pas :

- télécharger intégralement un `.safetensors` réel ;
- charger intégralement les poids réels en RAM ;
- prétendre reconstruire les poids originaux ;
- effectuer un fine-tuning ;
- entraîner un LLM ;
- reproduire exactement les sorties token par token du modèle réel ;
- effectuer une comparaison numérique complète avec les poids réels ;
- remplacer un modèle réel pour une utilisation productive.

---

# 8. Modèles supportés

PMG v1 possède deux profils de référence :

```text
PMG
├── GLM-5.2
└── DeepSeek-V4-Flash
```

Chaque modèle doit posséder un profil interne.

Exemple conceptuel :

```text
ModelProfile
├── architecture
├── tensor_schema
├── dtype_policy
├── layer_policy
├── distribution_policy
├── outlier_policy
├── correlation_policy
├── low_rank_policy
├── quantization_policy
└── serialization_policy
```

Ces profils ne doivent pas être construits à partir d'hypothèses non vérifiées.

Lorsqu'une caractéristique précise du modèle réel n'est pas connue avec suffisamment de certitude, PMG doit :

1. l'indiquer ;
2. utiliser une valeur configurable ;
3. enregistrer la provenance de cette information ;
4. ne jamais présenter une estimation comme un fait.

---

# 9. Architecture générale

Architecture logique :

```text
                    ┌─────────────────────┐
                    │      Interface CLI  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Command Dispatcher│
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
     Inspector             Generator             Validator
          │                    │                     │
          ▼                    ▼                     ▼
    Model Parser        Synthetic Engine       Validator Engine
          │                    │
          ▼                    ▼
    Metadata Model       Tensor Generator
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
           Distributions    Outliers       Correlations
               │               │               │
               └───────────────┼───────────────┘
                               ▼
                       Tensor Serializer
                               │
                               ▼
                     Safetensors Writer
                               │
                               ▼
                    Model Output Directory
```

---

# 10. Organisation logicielle Rust

Structure recommandée :

```text
Pseudo-Models-Generator/
│
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── README.md
├── Cahier_des_charges.md
├── Cahier_developpement.md
│
├── crates/
│   │
│   ├── pmg-core/
│   │   └── src/
│   │
│   ├── pmg-cli/
│   │   └── src/
│   │
│   ├── pmg-config/
│   │   └── src/
│   │
│   ├── pmg-model/
│   │   └── src/
│   │
│   ├── pmg-math/
│   │   └── src/
│   │
│   ├── pmg-statistics/
│   │   └── src/
│   │
│   ├── pmg-generator/
│   │   └── src/
│   │
│   ├── pmg-safetensors/
│   │   └── src/
│   │
│   ├── pmg-validator/
│   │   └── src/
│   │
│   └── pmg-inspector/
│       └── src/
│
├── tests/
├── benches/
└── .github/
    └── workflows/
```

---

# 11. Responsabilités des crates

## 11.1 `pmg-core`

Contient les primitives communes :

- types ;
- erreurs ;
- identifiants ;
- configuration globale ;
- structures communes.

---

## 11.2 `pmg-cli`

Responsable de :

- parsing CLI ;
- commandes ;
- options ;
- affichage ;
- logs ;
- mode verbose ;
- mode debug ;
- mode dry-run.

---

## 11.3 `pmg-config`

Responsable de :

- lecture JSON ;
- validation ;
- normalisation ;
- gestion des configurations.

---

## 11.4 `pmg-model`

Représentation abstraite d'un modèle :

```text
Model
├── ModelArchitecture
├── ModelConfig
├── TensorRegistry
├── LayerRegistry
├── TokenizerMetadata
└── WeightMetadata
```

---

## 11.5 `pmg-math`

Contient :

- RNG ;
- distributions ;
- algorithmes statistiques ;
- matrices ;
- opérations vectorielles ;
- corrélations ;
- transformations.

---

## 11.6 `pmg-statistics`

Responsable de :

- moyenne ;
- variance ;
- écart-type ;
- skewness ;
- kurtosis ;
- quantiles ;
- histogrammes ;
- queues ;
- détection d'outliers ;
- modèles statistiques.

---

# 12. Moteur de génération synthétique

Le générateur doit fonctionner par étapes.

```text
Configuration
     ↓
Analyse architecture
     ↓
Construction Tensor Registry
     ↓
Planification mémoire
     ↓
Génération statistiques
     ↓
Génération structurelle
     ↓
Injection corrélations
     ↓
Injection bas-rang
     ↓
Injection outliers
     ↓
Conversion dtype
     ↓
Quantification éventuelle
     ↓
Validation
     ↓
Écriture Safetensors
     ↓
Écriture configurations
```

---

# 13. Génération des tenseurs

Chaque tenseur doit posséder une description interne :

```text
TensorDescriptor
├── name
├── shape
├── dtype
├── num_elements
├── byte_size
├── layer_id
├── tensor_role
├── statistical_profile
├── outlier_profile
├── correlation_profile
└── low_rank_profile
```

---

# 14. Calcul de la taille d'un tenseur

Pour un tenseur de forme :

\[
S=(d_1,d_2,\ldots,d_n)
\]

le nombre d'éléments est :

\[
N=\prod_{i=1}^{n}d_i
\]

Pour un dtype de \(b\) bits :

\[
B=N\frac{b}{8}
\]

octets lorsque le stockage est directement aligné sur des octets.

Exemple BF16 :

\[
b=16
\]

donc :

\[
B=2N
\]

---

# 15. Gestion de la taille cible

PMG doit permettre :

```bash
pmg generate \
  --model glm-5.2 \
  --size 1GB
```

ou :

```bash
pmg generate \
  --model deepseek-v4-flash \
  --size 1GiB
```

La distinction entre :

- GB ;
- GiB ;

doit être explicite.

\[
1\,GB=10^9\ bytes
\]

\[
1\,GiB=2^{30}\ bytes
\]

---

# 16. Problème important : réduction de taille

Un modèle de 100 Go ne peut pas devenir un pseudo-modèle de 1 Go tout en conservant simultanément :

- toutes ses dimensions ;
- tous ses tenseurs ;
- le même dtype ;
- le même nombre d'éléments.

Si :

\[
B_{original}>B_{target}
\]

alors PMG doit appliquer une stratégie explicitement choisie.

Exemples :

### Stratégie A — Quantification

Réduction de bits par poids.

### Stratégie B — Réduction structurelle

Créer un modèle miniature avec dimensions réduites.

### Stratégie C — Profil hybride

Conserver certaines couches/tensors importants et réduire les autres.

### Stratégie D — Modèle proxy

Conserver la topologie mais réduire les dimensions internes.

PMG doit **indiquer quelle stratégie a été utilisée**.

Il ne doit jamais produire un fichier de 1 Go et prétendre qu'il contient exactement les mêmes paramètres que le modèle réel de 100 Go.

---

# 17. Modes de génération

PMG v1 doit prévoir au minimum :

```text
FULL-STRUCTURAL
STRUCTURAL-PROXY
SIZE-CONSTRAINED
DTYPE-CONSTRAINED
```

Exemple :

```bash
pmg generate \
  --model glm-5.2 \
  --size 1GiB \
  --mode size-constrained
```

---

# 18. Dtypes

PMG doit disposer d'une abstraction :

```rust
enum DType {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}
```

Les dtypes réellement supportés par le writer Safetensors doivent être vérifiés contre la spécification de la version utilisée.

PMG ne doit pas inventer un dtype Safetensors simplement parce qu'un format de quantification existe dans un autre écosystème.

---

# 19. Quantification INT4 / NF4

PMG doit distinguer :

```text
dtype de stockage
```

et :

```text
schéma de quantification
```

Exemple :

```text
QuantizationProfile
├── method = GPTQ
├── bits = 4
├── group_size = ...
├── scale_dtype = F16
├── zero_point_dtype = ...
└── packing = ...
```

Un format tel que NF4 ne doit pas être traité comme s'il s'agissait automatiquement d'un dtype natif générique du fichier Safetensors.

---

# 20. Injection des outliers

Les outliers constituent une composante importante du projet PMG.

Cependant, PMG ne doit pas utiliser arbitrairement :

```text
1% de valeurs × 100
```

sans justification.

Le générateur doit utiliser un modèle statistique.

Pour une variable aléatoire \(X\), un outlier peut être défini par exemple par :

\[
|X-\mu|>k\sigma
\]

ou par un seuil quantile :

\[
|X|>Q_{0.999}(X)
\]

Le seuil doit dépendre du profil du tenseur.

---

# 21. Super-poids

PMG introduit le concept de :

**Super Weights / Super Poids**

Un super-poids est un poids synthétique exceptionnellement important relativement au reste de la distribution.

Exemple conceptuel :

```text
Poids normaux :
-0.2
 0.1
-0.4
 0.3
 0.05

Super-poids :
+8.4
-6.7
```

Mais leur fréquence doit être contrôlée.

PMG doit donc stocker :

```text
OutlierProfile
├── probability
├── amplitude_distribution
├── sign_distribution
├── locality
├── row_correlation
└── column_correlation
```

---

# 22. Corrélations

Un tenseur réel n'est pas nécessairement un ensemble de variables indépendantes.

PMG doit pouvoir générer :

\[
Cov(X)\neq I
\]

Une méthode possible consiste à produire :

\[
X=LZ
\]

où :

\[
Z\sim N(0,I)
\]

et :

\[
LL^T=\Sigma
\]

Ainsi :

\[
Cov(X)=\Sigma
\]

PMG pourra utiliser une matrice de covariance simplifiée ou factorisée afin d'éviter une consommation mémoire excessive.

---

# 23. Structures de bas-rang

PMG doit pouvoir générer une composante :

\[
W_{low-rank}=UV^T
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

Le tenseur final peut être :

\[
W=W_{base}+\alpha UV^T+W_{outlier}
\]

Cela permet de simuler :

- corrélations ;
- redondance ;
- directions dominantes ;
- structures compressibles.

---

# 24. Modèle statistique général

PMG doit pouvoir combiner plusieurs composantes :

\[
W=
W_{base}
+
\lambda_1W_{corr}
+
\lambda_2W_{lowrank}
+
\lambda_3W_{outlier}
\]

où :

- \(W_{base}\) = distribution fondamentale ;
- \(W_{corr}\) = structure corrélée ;
- \(W_{lowrank}\) = composante de bas-rang ;
- \(W_{outlier}\) = anomalies.

Les coefficients :

\[
\lambda_1,\lambda_2,\lambda_3
\]

doivent être contrôlables.

---

# 25. Distributions statistiques

PMG doit disposer d'un moteur de distributions.

Profils potentiels :

### Gaussienne

\[
X\sim N(\mu,\sigma^2)
\]

### Student-t

\[
X\sim t_\nu
\]

utile pour des queues plus lourdes.

### Laplace

\[
f(x)=\frac{1}{2b}e^{-|x-\mu|/b}
\]

### Log-normal

Pour certaines variables positives.

### Pareto

Pour modéliser des queues lourdes :

\[
P(X>x)=\left(\frac{x_m}{x}\right)^\alpha
\]

### Weibull

\[
f(x)=
\frac{k}{\lambda}
\left(\frac{x}{\lambda}\right)^{k-1}
e^{-(x/\lambda)^k}
\]

Mais **PMG ne doit jamais choisir une distribution uniquement parce qu'elle est mathématiquement sophistiquée**.

La distribution doit être sélectionnée selon :

- rôle du tenseur ;
- architecture ;
- profil observé ;
- littérature disponible ;
- paramètres de génération ;
- stratégie de validation.

---

# 26. Seed et reproductibilité

Toutes les générations doivent être reproductibles.

Exemple :

```bash
pmg generate \
  --model glm-5.2 \
  --size 1GiB \
  --seed 42
```

Pour :

\[
seed=42
\]

la même configuration doit produire le même pseudo-modèle, à condition que la version du moteur de génération soit identique.

PMG doit enregistrer :

```text
generation_seed
generator_version
profile_version
distribution_profile
```

---

# 27. Génération déterministe

La génération doit éviter les dépendances accidentelles à :

- l'ordre des threads ;
- l'ordre des HashMap ;
- le système ;
- l'endianness ;
- les architectures CPU lorsque cela affecte les résultats.

Le parallélisme doit donc être conçu pour conserver la reproductibilité lorsque le mode déterministe est activé.

---

# 28. Fichiers de sortie

PMG ne doit pas produire uniquement :

```text
model.safetensors
```

Il doit produire un répertoire complet.

Exemple :

```text
glm-5.2-pmg/
│
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── chat_template.jinja
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
│
└── pmg_metadata.json
```

Le nom exact des fichiers doit suivre le modèle source et les conventions attendues par les frameworks ciblés.

---

# 29. `pmg_metadata.json`

PMG doit produire un fichier spécifique permettant de distinguer le pseudo-modèle du modèle réel.

Exemple :

```json
{
  "generator": "PMG",
  "version": "1.0.0",
  "model_family": "glm-5.2",
  "pseudo_model": true,
  "generation_seed": 42,
  "generation_mode": "size-constrained",
  "target_size_bytes": 1073741824,
  "statistical_profile": "default",
  "weights_reconstructed": false
}
```

Cette information est importante pour empêcher qu'un pseudo-modèle soit présenté accidentellement comme les poids originaux.

---

# 30. Safetensors

PMG doit disposer d'un writer Safetensors dédié.

Le writer doit gérer :

1. l'en-tête ;
2. les noms de tenseurs ;
3. les shapes ;
4. les dtypes ;
5. les offsets ;
6. les tailles ;
7. les métadonnées ;
8. l'écriture streaming ;
9. le partitionnement en shards.

---

# 31. Streaming

PMG ne doit pas obligatoirement générer un énorme buffer mémoire.

Pour un tenseur :

\[
N=10^9
\]

éléments, le générateur doit pouvoir produire :

```text
chunk 1
chunk 2
chunk 3
...
chunk n
```

puis les écrire progressivement.

Objectif :

\[
Memory_{PMG}\ll Size_{Model}
\]

---

# 32. Génération par chunks

Exemple :

```text
Tensor
  │
  ├── Chunk 0
  ├── Chunk 1
  ├── Chunk 2
  ├── ...
  └── Chunk N
```

Chaque chunk doit être généré de manière déterministe.

La seed peut être dérivée de :

\[
S_{chunk}=Hash(S_{global},tensor\_id,chunk\_id)
\]

Cela permet de paralléliser sans perdre la reproductibilité.

---

# 33. HTTP Range

PMG peut exploiter HTTP Range lorsque cela est utile.

Principe :

```http
Range: bytes=start-end
```

Cela permet d'obtenir une partie d'un fichier distant.

Mais PMG doit respecter une règle fondamentale :

> Une requête Range ne doit jamais être utilisée comme moyen détourné de télécharger intégralement les poids.

L'objectif est uniquement de récupérer des informations minimales nécessaires.

PMG doit donc :

1. connaître la structure du fichier ;
2. récupérer l'en-tête nécessaire ;
3. parser les métadonnées ;
4. éviter les lectures inutiles.

---

# 34. Politique stricte de lecture des poids

Par défaut :

```text
.safetensors
       ↓
Aucune lecture des données de poids
```

PMG peut éventuellement lire :

```text
header / metadata
```

si l'utilisateur active explicitement :

```bash
pmg inspect --remote ...
```

Le système doit distinguer :

```text
METADATA ACCESS
```

de :

```text
WEIGHT DATA ACCESS
```

---

# 35. Commandes CLI

PMG v1 doit fournir :

```text
help
generate
espec
validate
compare
version
```

---

# 36. Commande `help`

Exemple :

```bash
pmg help
```

doit afficher un guide pour débutant.

Exemple :

```text
PMG — Pseudo-Models Generator

Utilisation :

    pmg generate ...
    pmg espec ...
    pmg validate ...
    pmg compare ...
    pmg version

Exemple débutant :

    pmg generate --model glm-5.2 --size 1GiB
```

---

# 37. Commande `generate`

Syntaxe conceptuelle :

```bash
pmg generate \
    --model <model> \
    --size <size> \
    --dtype <dtype> \
    --output <directory>
```

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GiB \
    --dtype bf16 \
    --output ./glm-pmg
```

---

# 38. Commande `espec`

`espec` signifie **inspection/spécification du modèle**.

Exemple :

```bash
pmg espec ./model
```

Il doit afficher :

```text
Architecture
Couches
Tenseurs
Shapes
Dtypes
Nombre de paramètres
Taille estimée
Shards
Tokenizer
Configuration
Statistiques disponibles
Informations manquantes
```

Il doit clairement séparer :

```text
OBSERVÉ
ESTIMÉ
GÉNÉRÉ
INCONNU
```

---

# 39. Commande `validate`

Exemple :

```bash
pmg validate ./glm-pmg
```

Validation :

```text
✓ config.json
✓ tokenizer.json
✓ index Safetensors
✓ noms des tenseurs
✓ shapes
✓ offsets
✓ dtypes
✓ tailles
✓ shards
✓ cohérence globale
```

---

# 40. Commande `compare`

Exemple :

```bash
pmg compare ./original ./pseudo
```

La comparaison v1 ne doit pas être une comparaison profonde des poids.

Elle compare :

- architecture ;
- configuration ;
- tokenizer ;
- noms ;
- shapes ;
- dtypes ;
- shards ;
- tailles ;
- métadonnées ;
- informations disponibles.

Elle doit afficher par exemple :

```text
Architecture       : compatible
Nombre de tenseurs : identique
Shapes             : 98.7 % compatibles
Dtypes             : différents
Taille             : différente
Poids réels        : non comparés
```

---

# 41. Commande `version`

```bash
pmg version
```

doit afficher :

```text
PMG
Pseudo-Models Generator
Version 1.0.0
Rust
GPL-3.0
```

ainsi que les informations de build pertinentes.

---

# 42. Options globales

Options recommandées :

```text
-h, --help
-d, --dry-run
-v, --verbose
    --debug
```

### Correction importante

La spécification précédente indiquait deux fois `-h` :

```text
-h = help
-h = debug
```

Cela est impossible dans une CLI classique.

La version officielle doit donc être :

```text
-h, --help
-d, --dry-run
-v, --verbose
    --debug
```

---

# 43. `--dry-run`

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GiB \
    --dry-run
```

PMG ne doit rien générer.

Il affiche :

```text
Modèle : GLM-5.2
Taille cible : 1 GiB
Mode : size-constrained

Tenseurs estimés : ...
Shards estimés : ...
Dtype : BF16
Mémoire temporaire : ...
Temps estimé : ...
```

---

# 44. `--verbose`

```bash
pmg generate ... --verbose
```

affiche davantage d'informations utilisateur, sans activer les logs internes détaillés.

---

# 45. `--debug`

```bash
pmg generate ... --debug
```

active les informations de diagnostic :

```text
seeds
chunks
offsets
distribution parameters
timings
allocation
threading
validation
```

---

# 46. Interface CLI

La CLI doit être riche mais compréhensible par un débutant.

Exemple :

```text
╭────────────────────────────────────╮
│ PMG — Pseudo-Models Generator 1.0  │
╰────────────────────────────────────╯

Modèle : GLM-5.2
Mode   : Size Constrained
Taille : 1 GiB

[1/6] Analyse de la configuration
[2/6] Construction du registre tensors
[3/6] Génération statistique
[4/6] Génération des tenseurs
[5/6] Écriture Safetensors
[6/6] Validation

✓ Génération terminée
```

---

# 47. Validation mathématique

PMG doit fournir des tests statistiques.

Pour un tenseur \(X\), calculer :

\[
\mu=\frac1N\sum_{i=1}^{N}x_i
\]

\[
\sigma^2=
\frac1N\sum_{i=1}^{N}(x_i-\mu)^2
\]

Ainsi que :

- minimum ;
- maximum ;
- quantiles ;
- skewness ;
- kurtosis ;
- taux d'outliers.

---

# 48. Validation des outliers

Si :

\[
O=\frac{N_{outlier}}{N}
\]

PMG doit afficher :

```text
Outlier ratio : O
```

et comparer cette valeur au profil cible.

---

# 49. Validation de corrélation

Pour deux variables :

\[
\rho_{XY}=
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
\]

PMG doit pouvoir vérifier que la corrélation produite se rapproche de la corrélation cible.

---

# 50. Validation bas-rang

Pour une matrice \(W\), PMG peut utiliser une approximation SVD :

\[
W=U\Sigma V^T
\]

et calculer l'énergie capturée par les \(k\) premières valeurs singulières :

\[
E_k=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_i\sigma_i^2}
\]

Cela permet d'évaluer la présence d'une structure basse dimensionnelle.

---

# 51. Validation de compression

Le pseudo-modèle doit être testé avec les pipelines ciblés.

Exemple :

```text
PMG
 ↓
Pseudo-model
 ↓
Compresseur
 ↓
Résultat
 ↓
Validation
```

L'objectif n'est pas de démontrer que :

```text
compression(pseudo) = compression(real)
```

mais de mesurer :

\[
D(C(\hat W),C(W))
\]

lorsque des données du modèle réel sont disponibles pour validation expérimentale.

---

# 52. Validation des quantifications

Pour une quantification :

\[
Q(W)
\]

PMG doit pouvoir mesurer des propriétés comme :

- distribution des valeurs ;
- saturation ;
- erreur de reconstruction ;
- taux d'outliers ;
- utilisation des bins ;
- distribution des scales ;
- distribution des zero-points ;
- erreur RMS.

---

# 53. Critère d'erreur

Une métrique possible est :

\[
RMSE=
\sqrt{
\frac1N
\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
}
\]

Pour un pseudo-modèle, cette métrique ne peut être calculée contre les poids réels que si ceux-ci sont effectivement accessibles.

Sinon PMG doit utiliser :

```text
distributional validation
```

et non :

```text
weight-by-weight validation
```

---

# 54. Provenance des informations

Chaque propriété importante doit avoir une provenance.

Exemple :

```text
Tensor shape
    source = model.safetensors.index.json
    confidence = exact

Dtype
    source = metadata
    confidence = exact

Outlier ratio
    source = synthetic profile
    confidence = estimated

Correlation
    source = statistical model
    confidence = synthetic
```

---

# 55. Niveaux de confiance

PMG doit utiliser :

```text
EXACT
DERIVED
ESTIMATED
SYNTHETIC
UNKNOWN
```

Exemple :

```text
Nombre de tenseurs : EXACT
Shape : EXACT
Nombre d'outliers : SYNTHETIC
Distribution réelle : UNKNOWN
```

---

# 56. Principe anti-hallucination

PMG ne doit jamais transformer :

```text
inconnu
```

en :

```text
certain
```

Exemple interdit :

```text
GLM-5.2 possède exactement 0.023 % d'outliers.
```

si aucune donnée vérifiable ne permet de le démontrer.

Exemple correct :

```text
Taux d'outliers :
UNKNOWN pour le modèle réel.
Profil synthétique utilisé :
0.02 %.
```

---

# 57. Configuration utilisateur

PMG doit permettre un profil personnalisé.

Exemple :

```toml
[statistics]
distribution = "student_t"
degrees_of_freedom = 5.0

[outliers]
enabled = true
ratio = 0.0002

[low_rank]
enabled = true
rank_ratio = 0.02

[correlation]
enabled = true
strength = 0.4
```

---

# 58. Presets

PMG doit fournir des presets :

```text
default
compression
quantization
inference
optimization
stress-test
```

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --profile compression \
    --size 1GiB
```

---

# 59. Profil `compression`

Il doit privilégier :

- outliers ;
- queues lourdes ;
- corrélations ;
- redondance ;
- bas-rang ;
- structures par blocs ;
- variations d'entropie.

---

# 60. Profil `quantization`

Il doit privilégier :

- plages dynamiques ;
- outliers ;
- distributions ;
- saturation ;
- scales ;
- zero-points ;
- groupes de quantification.

---

# 61. Profil `stress-test`

Il doit produire volontairement des cas difficiles :

```text
forts outliers
queues lourdes
forte dynamique
blocs hétérogènes
corrélations élevées
```

Cela permet de tester la robustesse d'un moteur.

---

# 62. Architecture des données

PMG doit conserver une représentation abstraite :

```text
Model
 ├── Config
 ├── Tokenizer
 ├── Layers
 │    ├── Layer 0
 │    │    ├── Tensor A
 │    │    ├── Tensor B
 │    │    └── Tensor C
 │    ├── Layer 1
 │    └── ...
 └── Metadata
```

---

# 63. Gestion mémoire

PMG doit privilégier :

- streaming ;
- chunks ;
- buffers réutilisables ;
- allocations contrôlées ;
- parallélisme limité ;
- absence de copie inutile.

Objectif :

\[
RAM \approx O(chunk\_size)
\]

plutôt que :

\[
RAM=O(model\_size)
\]

---

# 64. Parallélisation

Le calcul peut être parallélisé :

```text
Layer 0 ──┐
Layer 1 ──┤
Layer 2 ──┼──> Worker pool
Layer 3 ──┤
Layer N ──┘
```

La parallélisation ne doit cependant pas compromettre :

- la reproductibilité ;
- les offsets ;
- l'ordre logique ;
- l'intégrité du fichier.

---

# 65. Sécurité

PMG doit :

- éviter `unsafe` sauf nécessité démontrée ;
- vérifier les tailles ;
- éviter les overflows ;
- vérifier les offsets ;
- vérifier les chemins ;
- refuser les fichiers malformés ;
- limiter les allocations contrôlées par des données externes ;
- valider les JSON avant traitement.

---

# 66. Gestion des erreurs

Les erreurs doivent être explicites.

Exemple :

```text
Erreur PMG-E204 :

Le modèle demande 8 GiB alors que la limite configurée
est de 1 GiB.

Suggestion :
utilisez --mode size-constrained.
```

---

# 67. Tests

PMG doit disposer de :

### Tests unitaires

Tester :

- distributions ;
- RNG ;
- calculs de tailles ;
- offsets ;
- dtypes ;
- packing ;
- outliers ;
- corrélations.

### Tests d'intégration

Tester :

```text
configuration
→ génération
→ écriture
→ lecture
→ validation
```

### Tests E2E

Tester les commandes CLI.

### Benchmarks

Tester :

- génération ;
- écriture ;
- streaming ;
- compression ;
- quantification.

---

# 68. Tests de propriété

Les fonctions mathématiques importantes doivent utiliser des tests basés sur propriétés lorsque cela est pertinent.

Exemple :

Pour une matrice :

\[
W=UV^T
\]

vérifier que :

\[
rank(W)\leq r
\]

dans les conditions prévues.

---

# 69. Test de reproductibilité

Deux exécutions :

```bash
pmg generate --seed 42 ...
pmg generate --seed 42 ...
```

doivent produire des sorties identiques lorsque le mode déterministe est activé.

---

# 70. Test de différence des seeds

Avec :

```text
seed = 42
```

et :

```text
seed = 43
```

PMG doit produire des modèles différents tout en conservant les mêmes contraintes statistiques.

---

# 71. Critères d'acceptation v1

PMG v1 est considéré comme fonctionnel lorsque :

### Configuration

- [ ] `config.json` est correctement analysé.
- [ ] `model.safetensors.index.json` est correctement analysé.
- [ ] tokenizer et fichiers associés sont gérés.
- [ ] les métadonnées sont distinguées des estimations.

### Génération

- [ ] GLM-5.2 est supporté.
- [ ] DeepSeek-V4-Flash est supporté.
- [ ] génération streaming.
- [ ] génération multi-shards.
- [ ] taille cible configurable.
- [ ] dtype configurable.

### Statistiques

- [ ] distributions.
- [ ] outliers.
- [ ] super-poids.
- [ ] corrélations.
- [ ] bas-rang.
- [ ] validation statistique.

### Safetensors

- [ ] writer valide.
- [ ] index valide.
- [ ] offsets valides.
- [ ] shapes cohérentes.
- [ ] dtypes cohérents.

### CLI

- [ ] `help`
- [ ] `generate`
- [ ] `espec`
- [ ] `validate`
- [ ] `compare`
- [ ] `version`
- [ ] `--dry-run`
- [ ] `--verbose`
- [ ] `--debug`

---

# 72. Critères d'acceptation scientifique

PMG ne doit pas être déclaré « fidèle au modèle réel » uniquement parce que :

```text
le fichier est valide
```

La validation doit être multidimensionnelle.

## Niveau 1 — Structure

\[
S_{struct}
\]

Vérification :

- architecture ;
- couches ;
- tenseurs ;
- shapes ;
- dtypes.

## Niveau 2 — Statistique

\[
S_{stat}
\]

Vérification :

- moyenne ;
- variance ;
- quantiles ;
- kurtosis ;
- queues ;
- outliers.

## Niveau 3 — Structure matricielle

\[
S_{matrix}
\]

Vérification :

- corrélations ;
- rang ;
- valeurs singulières ;
- structure par blocs.

## Niveau 4 — Pipeline

\[
S_{pipeline}
\]

Vérification :

- compression ;
- quantification ;
- conversion ;
- chargement ;
- optimisation.

---

# 73. Score de fidélité

PMG pourra ultérieurement définir :

\[
F=
w_sS_{struct}
+w_tS_{stat}
+w_mS_{matrix}
+w_pS_{pipeline}
\]

avec :

\[
\sum_iw_i=1
\]

Mais ce score ne devra être introduit qu'après définition rigoureuse des métriques.

Un score arbitraire du type :

```text
PMG fidelity = 97 %
```

est interdit sans protocole expérimental permettant de justifier cette valeur.

---

# 74. Architecture de validation expérimentale

Lorsque des poids réels sont disponibles pour une expérience de laboratoire, ils peuvent être comparés au pseudo-modèle.

Important :

> Cette comparaison appartient au protocole de validation externe et ne signifie pas que PMG doit lire les poids réels pendant sa génération normale.

Pipeline :

```text
              ┌──────────────┐
              │ Modèle réel  │
              └──────┬───────┘
                     │
                 référence
                     │
                     ▼
              ┌──────────────┐
              │ Benchmark    │
              └──────────────┘

              ┌──────────────┐
              │ Modèle PMG   │
              └──────┬───────┘
                     │
                 benchmark
                     │
                     ▼
              ┌──────────────┐
              │ Comparaison  │
              └──────────────┘
```

---

# 75. Compatibilité avec les moteurs externes

PMG doit viser la compatibilité avec les écosystèmes qui consomment :

- configuration de modèle ;
- tokenizer ;
- Safetensors ;
- index de shards ;
- metadata.

Mais PMG ne doit jamais garantir la compatibilité avec un moteur donné sans test réel.

La règle est :

```text
Compatible par spécification
```

puis :

```text
Compatible validé expérimentalement
```

---

# 76. Documentation utilisateur

PMG doit fournir :

```text
README.md
MANUEL_UTILISATEUR.md
GUIDE_DEBUTANT.md
FAQ.md
Cahier_des_charges.md
Cahier_developpement.md
```

---

# 77. Exemple complet pour débutant

Un utilisateur possède :

```text
glm-5.2/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model.safetensors.index.json
```

mais pas les poids.

Il exécute :

```bash
pmg espec ./glm-5.2
```

PMG analyse la structure.

Ensuite :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1GiB \
    --output ./glm-pmg
```

PMG produit :

```text
glm-pmg/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── pmg_metadata.json
```

Puis :

```bash
pmg validate ./glm-pmg
```

Enfin :

```bash
pmg compare ./glm-5.2 ./glm-pmg
```

---

# 78. Exemple de workflow professionnel

```text
                 SOURCE
                   │
                   ▼
          Configuration modèle
                   │
                   ▼
              PMG espec
                   │
                   ▼
          Model Specification
                   │
                   ▼
          PMG dry-run
                   │
                   ▼
       Plan de génération validé
                   │
                   ▼
             PMG generate
                   │
                   ▼
        Pseudo-model Safetensors
                   │
                   ├─────────────┐
                   ▼             ▼
              validate       benchmark
                   │             │
                   └──────┬──────┘
                          ▼
                     Analyse finale
```

---

# 79. Contraintes de développement

Le projet doit respecter le Cahier de Développement PMG.

Principes principaux :

- Rust ;
- Rustfmt ;
- Clippy ;
- documentation des API publiques ;
- commentaires internes en français ;
- fichiers Rust ≤ 500 lignes ;
- `unsafe` exceptionnel ;
- tests obligatoires ;
- Conventional Commits ;
- Git Flow simplifié ;
- CI/CD ;
- benchmarks ;
- GPL-3.0.

---

# 80. Dépendances

Les dépendances doivent rester minimales.

Crates envisagées :

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

Le choix final doit être validé au moment de l'implémentation en fonction :

- des versions compatibles ;
- de la maintenance ;
- de la licence ;
- de la sécurité ;
- de l'overhead.

---

# 81. Licence

Le projet PMG est distribué sous :

# GNU General Public License v3.0 — GPL-3.0

Tous les composants du projet doivent respecter les obligations de cette licence.

Le fichier :

```text
LICENSE
```

doit contenir le texte officiel de la GPL-3.0.

---

# 82. Règle de vérité scientifique

Cette règle est fondamentale.

PMG doit toujours distinguer :

```text
DONNÉE OBSERVÉE
```

de :

```text
DONNÉE DÉDUITE
```

de :

```text
DONNÉE ESTIMÉE
```

de :

```text
DONNÉE SYNTHÉTIQUE
```

de :

```text
DONNÉE INCONNUE
```

Aucune donnée synthétique ne doit être présentée comme provenant des poids réels.

---

# 83. Positionnement officiel de PMG

PMG n'est pas :

```text
un reconstructeur de poids
```

PMG est :

```text
un générateur de modèles synthétiques structurellement compatibles
et statistiquement contrôlés.
```

La formulation officielle recommandée est :

> **PMG génère des pseudo-modèles synthétiques conçus pour reproduire les propriétés structurelles, numériques et statistiques pertinentes d'un modèle LLM réel, sans nécessiter le téléchargement ou la lecture complète de ses poids.**

---

# 84. Vision scientifique

Le cœur du projet peut être résumé par :

\[
\boxed{
\hat W =
G(A,T,H,\theta,S)
}
\]

où :

- \(A\) = architecture ;
- \(T\) = tokenizer ;
- \(H\) = métadonnées disponibles ;
- \(\theta\) = paramètres statistiques ;
- \(S\) = seed ;
- \(G\) = générateur PMG ;
- \(\hat W\) = poids synthétiques.

Le générateur doit chercher à satisfaire :

\[
P(\hat W)\approx P(W|A,T,H)
\]

sur les propriétés utiles aux outils cibles.

---

# 85. Architecture conceptuelle finale

```text
                         ┌───────────────────┐
                         │       PMG CLI     │
                         └─────────┬─────────┘
                                   │
              ┌────────────────────┼───────────────────┐
              │                    │                   │
              ▼                    ▼                   ▼
          Inspector             Generator           Validator
              │                    │                   │
              ▼                    ▼                   ▼
        Config Parser        Model Profile       Structural Tests
              │                    │                   │
              ▼                    ▼                   ▼
       Metadata Registry      Tensor Planner       Statistical Tests
                                   │                   │
                    ┌──────────────┼──────────────┐    │
                    │              │              │    │
                    ▼              ▼              ▼    │
               Distribution    Correlation     Low-Rank │
                    │              │              │    │
                    └──────────────┼──────────────┘    │
                                   ▼                   │
                               Outliers                │
                                   │                   │
                                   ▼                   │
                           Synthetic Tensor            │
                                   │                   │
                                   ▼                   │
                           DType / Packing             │
                                   │                   │
                                   ▼                   │
                         Safetensors Streaming         │
                                   │                   │
                                   ▼                   │
                            Model Directory            │
                                   │                   │
                                   └──────────┬────────┘
                                              ▼
                                           Validate
```

---

# 86. Définition finale de la v1.0

La version 1.0 de PMG doit être considérée comme un **moteur de génération de mannequins LLM**, spécialisé initialement dans :

```text
GLM-5.2
DeepSeek-V4-Flash
```

avec les capacités fondamentales suivantes :

```text
                 PMG v1.0
                    │
       ┌────────────┼─────────────┐
       │            │             │
       ▼            ▼             ▼
   STRUCTURE     STATISTIQUE    FORMAT
       │            │             │
       │            │             ├── Safetensors
       │            ├── distributions
       │            ├── outliers
       │            ├── super-poids
       │            ├── corrélations
       │            └── bas-rang
       │
       ├── layers
       ├── tensors
       ├── shapes
       ├── dtypes
       └── architecture
```

Le résultat final doit pouvoir être utilisé comme **banc d'essai logiciel** pour les moteurs de :

- compression ;
- quantification ;
- optimisation ;
- conversion ;
- analyse ;
- gestion de mémoire ;
- traitement de tenseurs ;
- pipelines compatibles avec les formats produits.

---

# 87. Formule directrice du projet

La philosophie technique de PMG peut être résumée par :

\[
\boxed{
\text{PMG} =
\text{Structure}
+
\text{Statistique}
+
\text{Numérique}
+
\text{Anomalies}
+
\text{Corrélations}
+
\text{Bas-rang}
+
\text{Compatibilité}
}
\]

et non simplement :

\[
\boxed{
\text{PMG}\neq\text{Random Tensor Generator}
}
\]

Le succès de PMG ne sera donc pas mesuré par la capacité à produire rapidement plusieurs gigaoctets de nombres, mais par la capacité à produire un **proxy expérimental contrôlé dont les propriétés pertinentes ressemblent à celles du modèle cible**, tout en indiquant explicitement ce qui est connu, déduit, estimé ou synthétique.

---

# 88. Statut

Le présent document constitue la spécification fonctionnelle et technique de référence de **PMG v1.0**.

Toute fonctionnalité nouvelle doit :

1. respecter cette architecture ;
2. respecter les contraintes scientifiques ;
3. respecter les contraintes Rust ;
4. respecter la licence GPL-3.0 ;
5. être testée ;
6. être documentée ;
7. distinguer les données réelles des données synthétiques ;
8. ne pas introduire de prétention de fidélité impossible à démontrer.

**Statut : CAHIER DES CHARGES PMG v1.0 — RÉFÉRENCE**