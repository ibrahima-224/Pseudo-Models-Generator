# CAHIER FONCTIONNEL
# Pseudo-Models Generator — PMG

**Version :** 1.0  
**Statut :** Approuvé  
**Licence :** GPL-3.0  
**Langue de l'interface :** Français  
**Technologie d'implémentation :** Rust  
**Type de logiciel :** CLI riche  
**Projet :** Pseudo-Models Generator — PMG

---

# 1. Objet du document

Le présent cahier fonctionnel définit précisément les fonctionnalités attendues du logiciel **Pseudo-Models Generator (PMG)**.

PMG est un logiciel destiné à générer des **pseudo-modèles de réseaux neuronaux/LLM** à partir des informations structurelles et métadonnées disponibles autour d'un modèle réel, notamment :

- `config.json`
- `model.safetensors.index.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`, lorsqu'il existe
- fichiers de templates éventuels
- métadonnées Safetensors accessibles
- headers Safetensors récupérables sans télécharger les données complètes
- autres fichiers de configuration explicitement supportés.

La version 1.0 cible prioritairement :

1. **DeepSeek-V4-Flash**
2. **GLM-5.2**

Le logiciel doit produire un répertoire ressemblant fonctionnellement à celui d'un modèle réel afin de permettre son utilisation dans des logiciels de :

- compression ;
- quantification ;
- optimisation ;
- conversion ;
- inspection ;
- validation ;
- benchmarking structurel ;
- expérimentation sur les pipelines de chargement ;
- tests d'outils de traitement de modèles.

---

# 2. Définition fonctionnelle de PMG

## 2.1 Principe général

PMG reçoit une description d'un modèle réel et produit un modèle synthétique.

Le pipeline général est :

```text
Entrée modèle
     │
     ├── config.json
     ├── model.safetensors.index.json
     ├── tokenizer.json
     ├── tokenizer_config.json
     ├── autres configurations
     └── métadonnées Safetensors
              │
              ▼
       Analyse du modèle
              │
              ▼
       Modèle abstrait PMG
              │
              ├── Architecture
              ├── Tenseurs
              ├── Shapes
              ├── Dtypes
              ├── Sharding
              ├── Paramètres
              ├── MoE
              ├── distributions
              ├── corrélations
              ├── outliers
              └── structures bas-rang
              │
              ▼
      Générateur synthétique
              │
              ▼
      Encodeur Safetensors
              │
              ▼
       Modèle pseudo-généré
```

---

# 3. Principe de vérité et limites fonctionnelles

## 3.1 Ce que PMG peut connaître

À partir des configurations et métadonnées, PMG peut connaître, selon les informations disponibles :

- architecture ;
- nombre de couches ;
- dimensions ;
- vocabulaire ;
- dimensions d'embedding ;
- nombre de têtes ;
- paramètres de RoPE ;
- paramètres MoE ;
- nombre d'experts ;
- dimensions des tenseurs ;
- noms des tenseurs ;
- shapes ;
- dtypes déclarés ;
- organisation des shards ;
- offsets ;
- tailles des tenseurs ;
- taille totale déclarée ;
- métadonnées Safetensors ;
- paramètres du tokenizer ;
- template conversationnel ;
- paramètres de génération.

---

## 3.2 Ce que PMG ne peut pas déduire exactement

Sans lecture des données réelles des tenseurs, PMG ne connaît pas exactement :

- chaque poids ;
- chaque valeur de biais ;
- chaque matrice réelle ;
- chaque distribution exacte ;
- chaque outlier réel ;
- chaque corrélation réelle ;
- chaque valeur singulière réelle ;
- chaque rang numérique réel ;
- chaque covariance réelle ;
- chaque quantification réelle effective si elle n'est pas entièrement décrite par les métadonnées.

Par conséquent :

```text
Configuration réelle
        ≠
Poids réels
```

et :

```text
Pseudo-poids générés
        ≠
Poids réels
```

PMG doit explicitement signaler cette distinction à l'utilisateur.

---

# 4. Objectifs fonctionnels

PMG doit permettre à l'utilisateur de :

1. sélectionner un modèle supporté ;
2. inspecter un modèle ;
3. analyser ses métadonnées ;
4. récupérer les métadonnées nécessaires ;
5. générer un pseudo-modèle ;
6. imposer une taille cible ;
7. choisir un dtype ;
8. générer plusieurs shards ;
9. reproduire la structure tensorielle ;
10. générer des distributions synthétiques réalistes ;
11. injecter des outliers contrôlés ;
12. reproduire des corrélations ;
13. reproduire des structures bas-rang ;
14. reproduire les caractéristiques structurelles d'un MoE ;
15. produire les fichiers de configuration nécessaires ;
16. valider le pseudo-modèle ;
17. comparer les métadonnées d'un pseudo-modèle avec celles du modèle source ;
18. inspecter les propriétés statistiques du pseudo-modèle ;
19. exécuter toutes les opérations en mode simulation avec `--dry-run`.

---

# 5. Modèle utilisateur

PMG doit être utilisable par trois catégories principales.

## 5.1 Débutant

Le débutant doit pouvoir exécuter :

```bash
pmg generate --model deepseek-v4-flash --size 1G
```

sans avoir à comprendre :

- les shards ;
- les offsets ;
- les distributions ;
- les matrices ;
- le format binaire ;
- les tenseurs.

PMG doit sélectionner automatiquement les paramètres sûrs.

---

## 5.2 Utilisateur avancé

L'utilisateur peut contrôler :

```bash
pmg generate \
    --model deepseek-v4-flash \
    --size 1G \
    --dtype bf16 \
    --seed 42 \
    --outliers realistic \
    --correlation medium \
    --low-rank realistic
```

---

## 5.3 Ingénieur / chercheur

L'utilisateur expert peut demander des statistiques détaillées :

```bash
pmg espec ./model
```

et obtenir notamment :

```text
Architecture
Paramètres
Tenseurs
Shapes
Dtypes
Distribution estimée
Kurtosis
Skewness
Outliers
Corrélations
Rang effectif
Sharding
Budget mémoire
```

---

# 6. Interface CLI

PMG possède une interface CLI riche en français.

Les commandes principales sont :

```text
help
generate
espec
validate
compare
version
```

---

# 7. Commande `help`

## 7.1 Fonction

Afficher l'aide générale.

Exemple :

```bash
pmg help
```

Résultat conceptuel :

```text
PMG — Pseudo-Models Generator

Génère des pseudo-modèles réalistes à partir des
métadonnées d'un modèle réel.

Commandes :

  generate   Générer un pseudo-modèle
  espec      Inspecter un modèle
  validate   Valider un modèle
  compare    Comparer deux modèles
  version    Afficher la version

Exemples :

  pmg generate --model glm-5.2 --size 1G
  pmg espec ./model
  pmg validate ./pseudo-model
  pmg compare ./original ./pseudo
```

---

# 8. Commande `generate`

## 8.1 Fonction

`generate` constitue la fonctionnalité principale de PMG.

Elle génère un pseudo-modèle complet.

Syntaxe conceptuelle :

```bash
pmg generate [OPTIONS]
```

---

## 8.2 Sélection du modèle

Exemple :

```bash
pmg generate --model deepseek-v4-flash
```

ou :

```bash
pmg generate --model glm-5.2
```

PMG doit refuser les modèles non supportés en version 1.0.

Exemple :

```text
Erreur :
Le modèle « llama-3 » n'est pas supporté par PMG 1.0.

Modèles supportés :
  deepseek-v4-flash
  glm-5.2
```

---

# 9. Source du modèle

PMG doit pouvoir travailler à partir d'un répertoire local contenant les configurations.

Exemple :

```bash
pmg generate \
    --input ./deepseek-v4-flash \
    --output ./pseudo-deepseek
```

Le répertoire peut contenir :

```text
deepseek-v4-flash/
├── config.json
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
└── ...
```

---

# 10. Règle fondamentale concernant les fichiers `.safetensors`

PMG **ne doit jamais charger intégralement les fichiers `.safetensors` pour générer un pseudo-modèle**.

Le fonctionnement normal est :

```text
configuration
     +
index
     +
métadonnées
     +
header Safetensors éventuellement récupéré
     ↓
PMG
     ↓
pseudo-modèle
```

Les données réelles des poids ne sont pas nécessaires à la génération standard.

---

# 11. Métadonnées récupérables par HTTP Range

Lorsque le modèle est distant et que le serveur prend en charge les requêtes HTTP Range, PMG peut récupérer uniquement les zones nécessaires à l'inspection des fichiers.

Principe :

```http
Range: bytes=0-N
```

PMG peut ainsi récupérer notamment le début d'un fichier Safetensors afin d'inspecter son header.

Cela ne signifie pas que PMG récupère les poids.

Le comportement doit être affiché :

```text
Analyse distante :
  Header : récupéré
  Poids : non téléchargés
  Données téléchargées : 96 KiB
```

Si le serveur ne supporte pas correctement Range :

```text
Avertissement :
Le serveur distant ne permet pas une récupération Range fiable.

PMG n'effectuera pas automatiquement le téléchargement
complet du fichier de poids.
```

---

# 12. Génération avec taille cible

L'utilisateur peut imposer une taille maximale ou cible.

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1G
```

PMG doit alors construire un pseudo-modèle dont la taille totale est compatible avec le budget demandé.

---

# 13. Budget de taille

Soit :

- \(B\) = budget demandé en octets ;
- \(H\) = taille des fichiers de configuration ;
- \(M\) = taille des métadonnées ;
- \(W\) = budget consacré aux données tensorielle.

Alors :

\[
W = B - H - M
\]

PMG doit tenir compte de la différence entre :

```text
taille demandée
```

et :

```text
taille réellement produite
```

en raison :

- des headers ;
- des alignements ;
- des shards ;
- des métadonnées ;
- des contraintes de représentation ;
- des tailles entières des tenseurs.

---

# 14. Réduction d'un modèle

Si le modèle original contient :

\[
P
\]

paramètres et que l'utilisateur demande un budget inférieur, PMG ne doit pas prétendre avoir conservé tous les paramètres originaux.

Il doit créer une **représentation réduite mais structurellement cohérente**.

Exemple :

```text
Modèle source :
  500 GB

Budget :
  1 GB

PMG :
  modèle pseudo-réduit
  architecture compatible avec le profil sélectionné
  dimensions/tensors adaptés au budget
```

Le rapport doit préciser :

```text
Attention :
Ce pseudo-modèle est une représentation réduite.
Il ne contient pas les 500 GB de poids originaux.
```

---

# 15. Dtypes

PMG doit proposer une sélection contrôlée des types de représentation.

Exemples d'interface :

```bash
--dtype fp32
--dtype fp16
--dtype bf16
--dtype int8
--dtype int4
```

La disponibilité exacte dépend du backend de stockage et de la représentation supportée par l'outil cible.

---

# 16. Important : dtype de stockage et format de quantification

PMG doit distinguer :

```text
dtype physique du tensor
```

de :

```text
format de quantification
```

Par exemple, un format INT4 peut nécessiter :

- valeurs packées ;
- scales ;
- zero-points ;
- groupes ;
- métadonnées supplémentaires.

Ainsi :

\[
\text{taille réelle}
\neq
N \times 4/8
\]

dans tous les cas.

PMG doit donc calculer la taille réelle à partir du schéma d'encodage.

---

# 17. Génération déterministe

PMG doit accepter une seed.

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1G \
    --seed 42
```

Pour un même :

```text
modèle
+
configuration
+
taille
+
dtype
+
seed
+
version PMG
```

la génération doit être reproductible autant que possible.

Conceptuellement :

\[
W = G(S, M, C, D, V)
\]

où :

- \(S\) = seed ;
- \(M\) = modèle ;
- \(C\) = configuration ;
- \(D\) = dtype ;
- \(V\) = version du générateur ;
- \(G\) = générateur déterministe ;
- \(W\) = pseudo-poids.

---

# 18. Génération des distributions

PMG ne doit pas remplir les tenseurs uniquement avec une distribution normale naïve.

Le moteur statistique doit pouvoir utiliser plusieurs familles.

Exemples :

- normale ;
- log-normale ;
- Student-t ;
- Weibull ;
- Pareto ;
- mélange de distributions.

Une distribution peut être définie par :

\[
X \sim D(\theta)
\]

où \(D\) est une famille statistique et \(\theta\) ses paramètres.

---

# 19. Mélange de distributions

Pour reproduire des queues lourdes :

\[
X =
\begin{cases}
X_1 & \text{avec probabilité } 1-p\\
X_2 & \text{avec probabilité } p
\end{cases}
\]

Exemple :

```text
99,5 % → distribution centrale
0,5 %   → distribution de queue
```

Cette stratégie permet d'introduire des valeurs extrêmes sans rendre tout le tensor irréaliste.

---

# 20. Injection des outliers

PMG doit posséder un système d'injection contrôlée des outliers.

Un outlier est une valeur ou un groupe de valeurs qui s'écarte fortement de la distribution centrale.

Une approche simple utilise :

\[
x' = s x
\]

avec :

\[
s > 1
\]

pour les positions sélectionnées.

Mais PMG doit éviter une simple multiplication uniforme.

Le système doit pouvoir contrôler :

- fréquence ;
- amplitude ;
- position ;
- regroupement ;
- corrélation ;
- signe ;
- localisation dans les matrices.

---

# 21. Outliers structurés

PMG doit supporter plusieurs formes :

```text
outlier ponctuel
outlier par ligne
outlier par colonne
outlier par bloc
outlier corrélé
outlier de queue lourde
```

Exemple :

```text
Tensor

[ 0.1   0.2   0.1 ]
[ 0.2  15.3   0.1 ]  ← outlier
[ 0.1   0.1   0.2 ]
```

PMG doit enregistrer la stratégie utilisée dans son rapport.

---

# 22. Corrélations

Les pseudo-poids doivent pouvoir présenter des corrélations synthétiques.

Une matrice de covariance peut être utilisée :

\[
\Sigma =
\mathbb{E}[(X-\mu)(X-\mu)^T]
\]

PMG peut générer :

\[
X = LZ + \mu
\]

où :

\[
LL^T = \Sigma
\]

et :

\[
Z \sim \mathcal{N}(0,I)
\]

Cette construction permet d'introduire une structure de covariance contrôlée.

---

# 23. Structures bas-rang

PMG doit pouvoir générer des composantes de faible rang.

Pour une matrice :

\[
W \in \mathbb{R}^{m \times n}
\]

PMG peut construire :

\[
W = UV^T + E
\]

avec :

\[
U \in \mathbb{R}^{m \times r}
\]

\[
V \in \mathbb{R}^{n \times r}
\]

et :

\[
r \ll \min(m,n)
\]

où \(E\) représente le résidu.

Cette structure permet de contrôler la présence d'une composante bas-rang.

---

# 24. Structure hybride

Le générateur PMG doit pouvoir combiner :

\[
W = W_{\text{base}}
+ W_{\text{corr}}
+ W_{\text{low-rank}}
+ W_{\text{outlier}}
\]

Cela permet de produire des tenseurs plus complexes qu'une simple matrice aléatoire.

---

# 25. Paramètres statistiques

PMG doit pouvoir associer à chaque tenseur des statistiques telles que :

- moyenne ;
- variance ;
- écart-type ;
- minimum ;
- maximum ;
- médiane ;
- quantiles ;
- skewness ;
- kurtosis ;
- fréquence d'outliers ;
- énergie ;
- norme \(L_1\) ;
- norme \(L_2\) ;
- rang effectif ;
- corrélation.

Exemple :

```text
Tensor : model.layers.10.attn.q_proj.weight

Shape       : [8192, 4096]
Dtype       : BF16
Mean        : -0.00013
Std         : 0.0184
Min         : -0.92
Max         : 1.07
Kurtosis    : 8.21
Outliers    : 0.031 %
Low-rank    : détectée
```

---

# 26. Profils de génération

PMG doit fournir des profils.

## Profil `safe`

Pour débutant :

```bash
pmg generate --model glm-5.2 --size 1G
```

PMG choisit automatiquement les paramètres.

---

## Profil `realistic`

```bash
pmg generate \
    --model glm-5.2 \
    --size 1G \
    --profile realistic
```

Le moteur active :

- distributions composites ;
- outliers ;
- corrélations ;
- structures bas-rang ;
- hétérogénéité par couche.

---

## Profil `compression`

```bash
pmg generate \
    --model deepseek-v4-flash \
    --size 4G \
    --profile compression
```

Le modèle est optimisé fonctionnellement pour tester :

- quantification ;
- packing ;
- compression ;
- décompression ;
- sparsité ;
- sensibilité aux outliers.

---

# 27. Profils de difficulté

PMG peut fournir :

```text
basic
realistic
compression
stress
research
```

Le profil `stress` peut volontairement accentuer :

- outliers ;
- queues lourdes ;
- distributions hétérogènes ;
- tensors atypiques.

Il doit être explicitement marqué comme synthétique.

---

# 28. Génération des fichiers

La sortie doit être un dossier complet.

Exemple :

```text
pseudo-glm/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── template_jinja.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── pmg/
    ├── manifest.json
    ├── statistics.json
    └── provenance.json
```

Les fichiers `pmg/` sont des métadonnées PMG et ne doivent pas être nécessaires au chargement par un moteur standard si celui-ci n'en a pas besoin.

---

# 29. Manifest PMG

PMG doit produire un manifeste permettant de savoir comment le pseudo-modèle a été construit.

Exemple conceptuel :

```json
{
  "pmg_version": "1.0.0",
  "model": "glm-5.2",
  "seed": 42,
  "target_size": 1073741824,
  "dtype": "bf16",
  "profile": "realistic",
  "weights_are_synthetic": true
}
```

Le champ :

```text
weights_are_synthetic
```

est obligatoire.

---

# 30. Commande `espec`

## 30.1 Fonction

`espec` signifie inspection/expertise du modèle.

Exemple :

```bash
pmg espec ./model
```

Elle ne génère rien.

Elle analyse le modèle.

---

# 31. Résultat de `espec`

Exemple :

```text
╔══════════════════════════════════════════╗
║          PMG — EXPERTISE MODÈLE          ║
╚══════════════════════════════════════════╝

Modèle :
  GLM-5.2

Architecture :
  ...

Paramètres :
  ...

Layers :
  ...

Tenseurs :
  ...

Dtypes :
  BF16

Sharding :
  16 fichiers

Tokenizer :
  Vocabulaire : ...

Statistiques disponibles :
  Header : OUI
  Poids : NON

Mode :
  Analyse métadonnées uniquement
```

---

# 32. Niveaux d'inspection

PMG doit permettre :

```bash
pmg espec ./model
```

et :

```bash
pmg espec ./model --verbose
```

Le niveau détaillé peut afficher :

- tous les tensors ;
- shapes ;
- dtype ;
- shard ;
- offsets ;
- tailles ;
- statistiques dérivées des métadonnées ;
- anomalies structurelles.

---

# 33. Commande `validate`

## 33.1 Fonction

Vérifier qu'un modèle respecte les contraintes attendues.

Exemple :

```bash
pmg validate ./pseudo-glm
```

---

# 34. Validation structurelle

PMG doit vérifier :

### Configuration

```text
config.json valide
```

### Index

```text
model.safetensors.index.json valide
```

### Tenseurs

Pour chaque tenseur :

\[
N = \prod_i shape_i
\]

PMG vérifie que la taille déclarée correspond au dtype.

---

# 35. Validation Safetensors

PMG doit vérifier :

- magic/header ;
- longueur du header ;
- JSON ;
- dtype ;
- shape ;
- offsets ;
- absence de chevauchement ;
- cohérence des tailles ;
- offsets dans les limites du fichier ;
- cohérence entre index et shards.

Exemple :

```text
Validation Safetensors

✓ Header
✓ JSON
✓ Dtypes
✓ Shapes
✓ Offsets
✓ Shards
✓ Index
✓ Configuration

Résultat : VALIDE
```

---

# 36. Validation du budget

Si l'utilisateur demande :

```text
1 GiB
```

PMG doit calculer :

\[
S_{\text{total}}
=
S_{\text{config}}
+
S_{\text{metadata}}
+
S_{\text{headers}}
+
S_{\text{tensors}}
\]

et comparer :

\[
S_{\text{total}} \leq B
\]

ou appliquer une tolérance explicitement définie.

---

# 37. Validation statistique

Pour un pseudo-modèle PMG, le validateur peut vérifier :

- présence des outliers attendus ;
- distribution ;
- variance ;
- kurtosis ;
- corrélation ;
- rang effectif ;
- diversité entre couches.

Exemple :

```text
Validation statistique

✓ Distribution centrale
✓ Queue lourde
✓ Outliers
✓ Corrélation
✓ Structure bas-rang

Score de conformité synthétique :
87.4 %

Attention :
Ce score mesure la conformité au profil PMG,
pas la ressemblance exacte aux poids originaux.
```

---

# 38. Commande `compare`

## 38.1 Fonction

Comparer deux modèles sans effectuer une comparaison profonde des poids.

Exemple :

```bash
pmg compare ./original ./pseudo
```

---

# 39. Comparaison autorisée

PMG peut comparer :

- architecture ;
- config ;
- tokenizer ;
- nombre de couches ;
- shapes ;
- nombres de tensors ;
- dtypes ;
- shards ;
- tailles ;
- noms des tensors ;
- paramètres déclarés ;
- métadonnées ;
- headers Safetensors.

---

# 40. Comparaison interdite par défaut

PMG ne doit pas télécharger les gigaoctets de poids uniquement pour faire un `compare`.

Il doit donc afficher :

```text
Comparaison des poids :
  NON EFFECTUÉE

Raison :
  PMG n'effectue pas de téléchargement complet des
  fichiers Safetensors en mode comparaison standard.
```

---

# 41. Comparaison des headers distants

Si les headers sont disponibles par Range :

```text
Source A :
  Header récupéré : 84 KiB

Source B :
  Header récupéré : 91 KiB

Poids :
  0 octet téléchargé
```

---

# 42. Résultat de comparaison

Exemple :

```text
COMPARAISON

                         Original      Pseudo
Architecture             GLM-5.2       GLM-5.2
Layers                    80            80
Tensors                   1 240         1 240
Dtype                     BF16          BF16
Shards                    32            4
Taille                    500 GB        1 GB

Architecture              ✓
Noms tensors              ✓
Shapes                    ✓
Dtype                     ✓
Taille                    ≠
Poids                     NON COMPARÉS
```

---

# 43. Commande `version`

Exemple :

```bash
pmg version
```

Résultat :

```text
PMG — Pseudo-Models Generator

Version       : 1.0.0
Rust          : ...
Format PMG    : 1
Licence       : GPL-3.0

Modèles :
  ✓ DeepSeek-V4-Flash
  ✓ GLM-5.2
```

---

# 44. Options globales

PMG doit fournir les options communes suivantes.

## `--help`

Afficher l'aide.

Exemple :

```bash
pmg generate --help
```

---

## `--dry-run`

Simuler une opération sans produire de fichier final.

Exemple :

```bash
pmg generate \
    --model glm-5.2 \
    --size 1G \
    --dry-run
```

Résultat :

```text
DRY-RUN

Modèle : GLM-5.2
Budget : 1 GiB
Dtype  : BF16

Taille estimée :
  Configuration : 18 KiB
  Headers       : 512 KiB
  Tenseurs      : 1023 MiB

Aucun fichier ne sera écrit.
```

---

# 45. `--debug`

Le flag debug affiche les informations internes nécessaires au diagnostic.

Exemple :

```bash
pmg generate ... --debug
```

Il peut afficher :

```text
DEBUG
seed=42
generator=realistic_v1
layer=23
tensor=...
distribution=student_t
...
```

Les logs doivent rester distingués de la sortie utilisateur normale.

---

# 46. `--verbose`

`--verbose` augmente le niveau d'information utilisateur sans afficher nécessairement les informations internes de debug.

Exemple :

```bash
pmg espec ./model --verbose
```

---

# 47. Différence `verbose` / `debug`

```text
normal
   ↓
résultat essentiel

verbose
   ↓
résultat détaillé

debug
   ↓
informations internes du logiciel
```

---

# 48. Architecture fonctionnelle

PMG est divisé fonctionnellement en composants.

```text
                 ┌──────────────────┐
                 │      PMG CLI      │
                 └────────┬─────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Generate      Espec       Validate
             │            │            │
             └────────────┼────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Model Analyzer  │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Model IR        │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Statistical     │
                 │ Generator       │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Tensor Generator│
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Safetensors     │
                 │ Writer          │
                 └─────────────────┘
```

---

# 49. Représentation intermédiaire du modèle

PMG doit utiliser une représentation intermédiaire, appelée conceptuellement :

```text
Model IR
```

Elle doit représenter :

```text
Model
 ├── Architecture
 ├── Configuration
 ├── Tokenizer
 ├── Layers
 │    ├── Tensor
 │    ├── Tensor
 │    └── ...
 ├── Dtypes
 ├── Shards
 └── Statistics
```

Cette représentation évite de coupler directement le générateur à `config.json`.

---

# 50. Support des architectures

La version 1.0 doit avoir des adaptateurs spécifiques :

```text
DeepSeekV4FlashAdapter
GLM52Adapter
```

Chaque adaptateur transforme les informations du modèle vers le Model IR.

---

# 51. Pourquoi des adaptateurs ?

Le générateur statistique ne doit pas connaître directement les particularités de chaque architecture.

Exemple :

```text
DeepSeek-V4-Flash
       ↓
DeepSeek Adapter
       ↓
      IR
       ↓
Generic Generator
```

et :

```text
GLM-5.2
       ↓
GLM Adapter
       ↓
      IR
       ↓
Generic Generator
```

Cela facilite l'ajout futur d'autres modèles.

---

# 52. Gestion du MoE

Si une architecture utilise un système Mixture-of-Experts, PMG doit représenter explicitement :

```text
Experts
Router
Expert weights
Routing parameters
```

Le générateur doit pouvoir produire des tensors correspondant à ces structures.

Il ne suffit pas de générer une matrice aléatoire appelée `expert.weight`.

---

# 53. Hétérogénéité des couches

PMG ne doit pas supposer que toutes les couches sont statistiquement identiques.

Pour la couche \(l\), les paramètres peuvent être :

\[
\theta_l = f(l)
\]

avec des variations selon :

- profondeur ;
- type de couche ;
- attention ;
- MLP ;
- expert ;
- embedding ;
- normalization.

Ainsi :

```text
Layer 0
  distribution A

Layer 20
  distribution B

Layer 60
  distribution C
```

---

# 54. Génération des embeddings

Les embeddings doivent être générés séparément des matrices internes.

PMG doit conserver :

- vocabulaire ;
- dimension d'embedding ;
- dtype ;
- shape ;
- éventuel partage avec la sortie.

---

# 55. Génération des normalisations

Les paramètres de normalisation doivent être traités comme une catégorie spécifique.

Par exemple :

```text
norm.weight
norm.bias
```

ne doivent pas être générés comme une matrice de poids dense ordinaire.

Le générateur doit utiliser un profil spécifique adapté à leur rôle.

---

# 56. Génération des matrices de projection

Les matrices :

```text
q_proj
k_proj
v_proj
o_proj
```

ou leurs équivalents architecturaux doivent être générées avec leur structure propre.

PMG doit connaître leur :

- shape ;
- rôle ;
- couche ;
- dtype ;
- relation avec les autres matrices.

---

# 57. Génération MLP

Les tensors MLP doivent être classés selon leur fonction :

```text
up projection
down projection
gate projection
```

Cela permet d'appliquer des profils statistiques différents.

---

# 58. Outliers spécifiques au rôle

PMG doit permettre :

```text
Embedding outlier profile
Attention outlier profile
MLP outlier profile
Router outlier profile
Expert outlier profile
```

Cela est préférable à :

```text
un seul profil d'outlier pour tout le modèle
```

---

# 59. Compression et quantification

Le pseudo-modèle doit pouvoir servir de cible à des logiciels de compression.

Exemple :

```text
PMG
 │
 ▼
Pseudo-model BF16
 │
 ▼
Compresseur
 │
 ▼
Pseudo-model INT4
```

Le but fonctionnel est de tester le pipeline sans avoir besoin de télécharger le modèle réel.

---

# 60. Mais PMG ne doit pas simuler de faux résultats

PMG ne doit jamais annoncer :

```text
Cette quantification donnera exactement 2.4 %
de perte sur le vrai modèle.
```

à partir de poids synthétiques.

Il doit annoncer :

```text
Résultat obtenu sur le pseudo-modèle synthétique.
Ce résultat ne constitue pas une mesure de la qualité
du modèle réel.
```

---

# 61. Compatibilité avec les outils externes

PMG doit chercher à produire une structure compatible avec les conventions attendues par les outils utilisant :

- configuration JSON ;
- tokenizer ;
- index Safetensors ;
- fichiers Safetensors ;
- shards ;
- templates de conversation.

La compatibilité réelle doit être vérifiée par `validate` et par des tests d'intégration avec les outils ciblés.

---

# 62. Sortie utilisateur

PMG doit distinguer :

### Sortie normale

```text
Génération terminée.

Modèle :
  GLM-5.2

Taille :
  1.00 GiB

Dtype :
  BF16

Shards :
  4

Seed :
  42

Résultat :
  ./pseudo-glm
```

### Warning

```text
AVERTISSEMENT :
Les poids sont synthétiques.
Ils ne correspondent pas aux poids originaux.
```

### Erreur

```text
ERREUR :
Le fichier model.safetensors.index.json est invalide.
```

---

# 63. Codes de sortie

PMG doit utiliser des codes d'exécution cohérents.

Conceptuellement :

```text
0  = succès
1  = erreur utilisateur
2  = modèle invalide
3  = entrée inaccessible
4  = erreur de génération
5  = erreur de format
6  = erreur interne
```

Les valeurs exactes doivent être centralisées dans le CLI.

---

# 64. Gestion des erreurs utilisateur

Exemple :

```bash
pmg generate --size abc
```

PMG doit répondre :

```text
Erreur :
« abc » n'est pas une taille valide.

Exemples :
  512M
  1G
  4G
  1GiB
  1024MiB
```

---

# 65. Sécurité fonctionnelle

PMG ne doit pas :

- exécuter du code provenant de `config.json` ;
- exécuter du contenu provenant du modèle ;
- faire confiance aveuglément aux métadonnées distantes ;
- écrire arbitrairement hors du répertoire demandé ;
- télécharger automatiquement des poids complets sans information explicite.

Les JSON doivent être traités comme des données non fiables.

---

# 66. Contrôle de la taille mémoire

La génération de très grands tensors ne doit pas nécessiter de charger tout le pseudo-modèle en RAM.

Le système doit privilégier :

```text
génération par blocs
        ↓
encodage
        ↓
écriture
        ↓
bloc suivant
```

Ainsi :

\[
M_{\text{RAM}} \ll S_{\text{modèle}}
\]

est l'objectif fonctionnel.

---

# 67. Génération streaming

Exemple :

```text
Tensor
  ↓
Chunk 1 → encode → write
  ↓
Chunk 2 → encode → write
  ↓
Chunk 3 → encode → write
```

Le générateur ne doit pas nécessiter :

```text
500 GB de RAM
```

pour générer un modèle de :

```text
500 GB
```

---

# 68. Progression

Pour les opérations longues :

```text
Génération

[████████████████░░░░] 78 %

Layer      : 61 / 80
Tensor     : 932 / 1200
Données    : 796 MiB / 1 GiB
Débit      : 428 MiB/s
ETA        : 00:31
```

---

# 69. Annulation

PMG doit gérer proprement l'interruption utilisateur.

Par exemple :

```text
CTRL+C
```

doit provoquer :

```text
Interruption demandée.

Finalisation sécurisée...
Nettoyage...
```

et éviter de laisser croire qu'un modèle incomplet est valide.

---

# 70. Fichier temporaire

La génération doit idéalement utiliser des fichiers temporaires ou un mécanisme équivalent afin d'éviter :

```text
pseudo-model/
    model.safetensors
```

présent mais incomplet et présenté comme valide après interruption.

---

# 71. Validation automatique après génération

Par défaut, après :

```bash
pmg generate
```

PMG doit pouvoir effectuer :

```text
generation
   ↓
validation
   ↓
rapport
```

Exemple :

```text
Génération terminée.
Validation automatique : OK.
```

---

# 72. Mode `--dry-run`

Le dry-run doit simuler :

- lecture des configurations ;
- calcul du budget ;
- choix du dtype ;
- nombre de tensors ;
- nombre de shards ;
- tailles estimées ;
- paramètres statistiques ;
- estimation du temps.

Il ne doit pas générer les poids.

---

# 73. Déterminisme et version

Le résultat dépend de :

\[
R = f(M,C,D,S,P,V)
\]

avec :

- \(M\) = modèle ;
- \(C\) = configuration ;
- \(D\) = dtype ;
- \(S\) = seed ;
- \(P\) = profil ;
- \(V\) = version PMG.

Le manifeste doit enregistrer ces paramètres.

---

# 74. Compatibilité ascendante

Une version future de PMG doit éviter de rendre silencieusement invalides les anciens pseudo-modèles.

Le manifeste doit contenir :

```text
pmg_format_version
pmg_generator_version
```

---

# 75. Critères fonctionnels de réussite

PMG 1.0 est considéré fonctionnel lorsqu'il peut :

### Analyse

- [x] lire les configurations ;
- [x] analyser l'index Safetensors ;
- [x] analyser les headers accessibles ;
- [x] identifier les tensors ;
- [x] reconstruire le Model IR.

### Génération

- [x] générer DeepSeek-V4-Flash ;
- [x] générer GLM-5.2 ;
- [x] imposer un budget de taille ;
- [x] choisir un dtype ;
- [x] générer plusieurs shards ;
- [x] écrire les configurations ;
- [x] générer les tensors.

### Réalisme statistique

- [x] distributions ;
- [x] queues lourdes ;
- [x] outliers ;
- [x] corrélations ;
- [x] bas-rang ;
- [x] hétérogénéité par couche.

### Validation

- [x] validation JSON ;
- [x] validation index ;
- [x] validation Safetensors ;
- [x] validation shapes ;
- [x] validation dtypes ;
- [x] validation offsets ;
- [x] validation taille.

### Inspection

- [x] statistiques ;
- [x] architecture ;
- [x] tensors ;
- [x] shards ;
- [x] métadonnées.

---

# 76. Critères de non-conformité

Le logiciel ne doit pas considérer comme succès :

```text
Un fichier de 1 GB rempli de nombres aléatoires
```

simplement parce que :

```text
taille = 1 GB
```

Le pseudo-modèle doit également respecter :

```text
structure
+
shapes
+
dtypes
+
organisation
+
distribution
+
statistiques
+
contraintes architecturales
```

---

# 77. Exemple complet débutant

L'utilisateur possède :

```text
glm-model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model.safetensors.index.json
```

Il veut un pseudo-modèle de 1 GiB.

Commande :

```bash
pmg generate \
    --input ./glm-model \
    --output ./glm-pseudo \
    --size 1GiB
```

PMG :

```text
1. Analyse du modèle
2. Détection GLM-5.2
3. Construction du Model IR
4. Calcul du budget
5. Sélection du dtype
6. Génération des tensors
7. Injection des structures statistiques
8. Écriture Safetensors
9. Génération des configurations
10. Validation
```

Sortie :

```text
glm-pseudo/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── ...
└── pmg/
    ├── manifest.json
    └── statistics.json
```

---

# 78. Exemple expert

```bash
pmg generate \
    --input ./deepseek \
    --output ./deepseek-pseudo \
    --size 4GiB \
    --dtype bf16 \
    --seed 1337 \
    --profile realistic \
    --verbose
```

Le générateur peut conceptuellement construire :

\[
W_l =
W_{\text{base},l}
+
\alpha_l W_{\text{corr},l}
+
\beta_l U_lV_l^T
+
W_{\text{outlier},l}
\]

où les coefficients :

\[
\alpha_l,\beta_l
\]

varient selon la couche.

---

# 79. Exemple de validation

```bash
pmg validate ./deepseek-pseudo
```

Résultat :

```text
PMG VALIDATE

Configuration
  ✓ config.json
  ✓ tokenizer.json
  ✓ tokenizer_config.json

Safetensors
  ✓ headers
  ✓ shapes
  ✓ dtypes
  ✓ offsets
  ✓ shards

Structure
  ✓ layers
  ✓ attention
  ✓ MLP
  ✓ experts

Statistiques
  ✓ distribution
  ✓ outliers
  ✓ corrélation
  ✓ structure bas-rang

Résultat :
  VALIDE

Nature :
  SYNTHÉTIQUE
```

---

# 80. Architecture fonctionnelle finale

La chaîne fonctionnelle officielle de PMG 1.0 est :

```text
                 ┌──────────────────────┐
                 │      UTILISATEUR      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       PMG CLI        │
                 └──────────┬───────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
       GENERATE            ESPEC           VALIDATE
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │  Model Acquisition  │
                 │  + HTTP Range       │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │    Model Analyzer    │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │       Model IR       │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Statistical Engine   │
                 ├──────────────────────┤
                 │ Distributions        │
                 │ Correlations         │
                 │ Low-rank             │
                 │ Outliers             │
                 │ Layer profiles       │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │  Tensor Generator    │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Streaming Safetensors│
                 │ Writer               │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Pseudo-Model         │
                 └──────────────────────┘
```

---

# 81. Règle fonctionnelle fondamentale de PMG

La règle centrale du projet est :

> **PMG doit reproduire la structure et les propriétés mesurables que ses entrées permettent de connaître, et générer de manière contrôlée les propriétés qui ne peuvent pas être observées sans les poids. Il ne doit jamais présenter une propriété synthétique comme une propriété réellement mesurée sur le modèle original.**

Cette règle doit être appliquée à :

- l'interface CLI ;
- les rapports ;
- les fichiers `manifest.json` ;
- les benchmarks ;
- les tests ;
- la documentation ;
- les résultats statistiques.

---

# 82. Définition officielle du pseudo-modèle PMG

Un pseudo-modèle PMG est défini comme :

\[
P =
(A,S,D,T,W,\Theta)
\]

où :

- \(A\) = architecture ;
- \(S\) = structure tensorielle ;
- \(D\) = dtypes ;
- \(T\) = tokenizer et configuration ;
- \(W\) = pseudo-poids ;
- \(\Theta\) = propriétés statistiques synthétiques.

Le modèle réel est :

\[
R =
(A,S,D,T,W_R)
\]

PMG peut reproduire :

\[
A,\ S,\ D,\ T
\]

lorsqu'ils sont connus depuis les métadonnées, mais il génère :

\[
W_P \neq W_R
\]

en général.

L'objectif devient donc :

\[
W_P \approx_{\mathcal{F}} W_R
\]

où \(\mathcal{F}\) représente un ensemble de propriétés mesurables :

\[
\mathcal{F} =
\{
\text{distribution},
\text{moments},
\text{queues},
\text{outliers},
\text{corrélations},
\text{rang},
\text{structure},
...
\}
\]

Cette formulation est la définition fonctionnelle la plus rigoureuse du concept de pseudo-modèle PMG.

---

# 83. Résultat attendu de PMG 1.0

À la fin du projet, un utilisateur doit pouvoir prendre un modèle supporté, par exemple :

```text
DeepSeek-V4-Flash
```

ou :

```text
GLM-5.2
```

et demander :

```bash
pmg generate --model glm-5.2 --size 1GiB
```

pour obtenir un répertoire autonome contenant :

```text
configuration
+
tokenizer
+
index
+
Safetensors
+
shards
+
métadonnées PMG
```

avec :

```text
structure cohérente
+
dtypes cohérents
+
taille contrôlée
+
poids synthétiques
+
distributions contrôlées
+
outliers
+
corrélations
+
structures bas-rang
+
hétérogénéité architecturale
+
validation automatique
```

tout en conservant une distinction explicite entre :

```text
PROPRIÉTÉ OBSERVÉE
```

et :

```text
PROPRIÉTÉ SYNTHÉTIQUE
```

---

# 84. Conclusion

Le rôle de PMG n'est donc pas de fabriquer un simple « faux fichier Safetensors ».

Son rôle fonctionnel est de construire un **mannequin numérique structurel et statistique** d'un modèle LLM.

Le mannequin doit être suffisamment fidèle sur les dimensions pertinentes pour permettre aux logiciels aval de tester leurs mécanismes :

```text
             MODÈLE RÉEL
                  │
       métadonnées observables
                  │
                  ▼
              ┌───────┐
              │  PMG  │
              └───┬───┘
                  │
       reconstruction structurelle
                  +
       génération statistique contrôlée
                  │
                  ▼
          PSEUDO-MODÈLE
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
 Compression  Quantification Optimisation
       │          │          │
       └──────────┼──────────┘
                  ▼
             Tests logiciels
```

La qualité de PMG doit donc être évaluée non pas uniquement par :

\[
\text{taille du fichier}
\]

mais par un ensemble multidimensionnel de critères :

\[
Q_{\text{PMG}}
=
f(Q_{\text{structure}},
Q_{\text{format}},
Q_{\text{statistique}},
Q_{\text{outliers}},
Q_{\text{corrélation}},
Q_{\text{bas-rang}},
Q_{\text{architecture}},
Q_{\text{compatibilité}})
\]

**PMG 1.0 est fonctionnellement centré sur DeepSeek-V4-Flash et GLM-5.2, avec une architecture suffisamment générique pour permettre l'ajout ultérieur d'autres modèles sans réécrire le moteur statistique ou le moteur Safetensors.**