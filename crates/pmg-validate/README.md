# Spécification d'architecture — PMG v1.0 (Sprints 0 à 17)

**Statut :** Contrat d'architecture implémentation-ready pour le mode Code.
**Version :** 1.0.0-rc.0 (spécification)
**Modèles cibles v1.0 :** GLM-5.2, DeepSeek-V4-Flash
**Licence :** GPL-3.0 — **Langage :** Rust — **Édition :** 2021 — **MSRV :** 1.80

---

## 1. Objet

Ce dossier contient la **spécification d'architecture unique** de PMG (Pseudo-Models Generator) v1.0, couvrant l'intégralité des sprints 0 à 17 des cahiers de développement. Elle est le contrat unique que le mode Code doit suivre pour implémenter, tester et livrer la v1.0.

Sources de vérité : les 9 documents de `docs/` (synthèse validée) et les artefacts de modèles de `Models/` (`config.json`, `model.safetensors.index.json`, tokenizer). Toute valeur présentée comme observée provient de ces artefacts ou des informations publiées citées.

## 2. Documents du dossier

| # | Document | Contenu |
|---|----------|---------|
| 1 | [`01-decisions-architecture.md`](01-decisions-architecture.md) | Les 10 décisions de stabilisation des contradictions + principes directeurs |
| 2 | [`02-workspace-et-crates.md`](02-workspace-et-crates.md) | Workspace Cargo, 12 crates, responsabilités, dépendances, versioning |
| 3 | [`03-modeles-de-donnees.md`](03-modeles-de-donnees.md) | Types du Core, Blueprint, TensorAtlas, Model IR, GenerationPlan, profils |
| 4 | [`04-moteurs-math-injection-generation.md`](04-moteurs-math-injection-generation.md) | RNG/seed, distributions, statistiques, covariance, low-rank, injection, pipeline, streaming |
| 5 | [`05-safetensors-io.md`](05-safetensors-io.md) | Writer/Reader Safetensors internes, invariants, sharding, index, émission des fichiers |
| 6 | [`06-outils-inspection-validation-comparaison.md`](06-outils-inspection-validation-comparaison.md) | `espec`, `validate` (4 niveaux), `compare` (metadata-only) |
| 7 | [`07-cli.md`](07-cli.md) | CLI complète (clap), commandes, options, codes de sortie, manifeste de sortie |
| 8 | [`08-plan-implementation.md`](08-plan-implementation.md) | Plan d'implémentation par lots ordonnés (sprints 0–17) avec dépendances |
| 9 | [`09-tests-benchmarks-ci.md`](09-tests-benchmarks-ci.md) | Stratégie de tests, benchmarks, CI, fichiers de projet |

## 3. Résumé exécutif des décisions structurantes

1. **Nomenclature des crates** : 12 crates à responsabilité unique (détail en D1).
2. **Flag verbose** : `-v, --verbose` ; `--debug` sans raccourci court.
3. **Codes de sortie** : `0` succès, `1` erreur générale, `2` argument invalide, `3` modèle invalide, `4` erreur I/O, `5` validation échouée, `6` comparaison incompatible.
4. **Manifeste PMG** : `pmg_metadata.json` à la racine (champ canonique `synthetic`) + dossier `pmg/` (statistics.json, provenance.json).
5. **Safetensors** : writer/parser **interne** (contrôle total des invariants, streaming, Zero-Payload) ; crate officielle `safetensors` en dev-dependency **optionnelle** pour tests d'interopérabilité uniquement.
6. **DType** : 19 variantes + `Bool`, enum `#[non_exhaustive]`, mécanisme d'extension ; écriture binaire v1.0 limitée aux dtypes à taille fixe ≥ 1 octet (F4/F6/F8E8M0 déclarés, taille calculable, écriture → erreur explicite).
7. **Couches GLM-5.2** : **78** (source de vérité = `config.json` publié, vérifié dans le dépôt).
8. **`--size`** : budget maximal du package généré (tous fichiers confondus) ; tolérance documentée 2 % ou +16 MiB (le plus petit) ; dépassement > tolérance → erreur explicite.
9. **`espec`** : « expertise de spécification » — inspection metadata-first sans chargement des poids, catégories OBSERVÉ/ESTIMÉ/GÉNÉRÉ/INCONNU.
10. **Édition Rust 2021**, MSRV `rust-version = "1.80"` (décision d'architecture).

## 4. Contraintes transversales (rappel)

- Rust, GPL-3.0, CLI en français, identifiants en anglais, commentaires/logs/docs en français.
- **Zero-Payload** : jamais de téléchargement ni de lecture du contenu des `.safetensors` source. `METADATA_ONLY` autorisé, `WEIGHTS_DATA` interdit en v1.0.
- **Streaming mémoire bornée** : `RAM = O(chunk_size)`, jamais `O(model_size)`.
- **Reproductibilité** : `seed_tensor = H(seed_global, model_id, tensor_name, layer_id, generation_version)` ; le numéro de version du générateur participe à l'identité du résultat.
- **`unsafe` interdit** sauf exception documentée/isolée (aucune en v1.0 : décision zéro-`unsafe`).
- **Limite 500 lignes** par fichier Rust hors commentaires/lignes blanches.
- **Aucune dépendance ML lourde** (pas de PyTorch/TF/JAX/Transformers/vLLM/CUDA/SciPy/NumPy).
- **Erreurs typées** `thiserror` dans les crates bibliothèque ; `anyhow` réservé au CLI.
- Distinction systématique **OBSERVÉ / ESTIMÉ / GÉNÉRÉ / INCONNU** et niveaux **EXACT / DERIVED / ESTIMATED / SYNTHETIC / UNKNOWN**.
- Phase 18+ (calibration empirique, Fidelity Score) : **hors périmètre v1.0** — mentionnée comme follow-up uniquement.

## 5. Vocabulaire stabilisé

| Terme | Définition |
|---|---|
| Pseudo-modèle | `M̂ = (A, Ŵ, T, C)` : architecture réelle connue, poids synthétiques, tokenizer, métadonnées de génération |
| Mannequin | Pseudo-modèle utilisé comme banc d'essai logiciel |
| `espec` | Expertise de spécification : inspection metadata-first |
| OBSERVÉ / EXACT | Lu directement dans un artefact autorisé |
| ESTIMÉ / DERIVED | Calculé mathématiquement depuis des métadonnées observées |
| GÉNÉRÉ / SYNTHETIC | Produit par le générateur PMG |
| INCONNU / UNKNOWN | Non établissable à partir des entrées autorisées |

## 6. État d'avancement (Sprints 0–11)

### Sprints complétés

| Sprint | Composant | Statut |
|--------|-----------|--------|
| 0 | Infrastructure | ✅ Complété |
| 1 | Core (pmg-core) | ✅ Complété |
| 2 | Blueprint (pmg-blueprint) | ✅ Complété |
| 3 | Mathématiques (pmg-math) | ✅ Complété |
| 4 | Injection (pmg-injector) | ✅ Complété |
| 5 | Génération (pmg-generator) | ✅ Complété |
| 6 | Génération tensorielle | ✅ Complété |
| 7 | Distributions statistiques | ✅ Complété |
| 8 | Structures de corrélation | ✅ Complété |
| 9 | Super-poids et outliers | ✅ Complété |
| 10 | Génération complète | ✅ Complété |
| 11 | Validation et comparaison | ✅ Complété |

### Composants ajoutés (Sprints 6-11)

#### pmg-core
- `generation_plan.rs` : Plan de génération tensorielle
- `distribution_config.rs` : Configuration des distributions statistiques
- `outlier_metadata.rs` : Métadonnées de validation des outliers
- `generator_config.rs` : Configuration globale du générateur
- `generation_pipeline.rs` : Pipeline de génération ordonné
- `tensor_generation.rs` : Génération d'un tenseur
- `streaming_generation.rs` : Génération par streaming
- `manifest.rs` : Manifeste du pseudo-modèle

#### pmg-math
- `generator.rs` : Générateur de base (normal, uniform)
- `outliers/model.rs` : Modèle de transformation des outliers
- `outliers/amplitude.rs` : Calcul de l'amplitude des anomalies
- `outliers/layer_policy.rs` : Politique par couche
- `outlier_analysis.rs` : Détection d'outliers
- `correlation_analysis.rs` : Analyse de corrélation
- `low_rank_analysis.rs` : Analyse bas-rang

#### pmg-validate
- `severity.rs` : Niveaux de sévérité
- `validator.rs` : Validateur principal
- `report.rs` : Rapports de validation

#### pmg-compare
- `comparator.rs` : Comparateur principal
- `config_comparator.rs` : Comparaison de configurations
- `tensor_comparator.rs` : Comparaison de structures
- `statistical_comparator.rs` : Comparaison statistique

#### pmg-cli
- Commandes : `generate`, `validate`, `compare`, `espec`, `version`
- Tests E2E complets

### Tests ajoutés

- **pmg-core** : 76 tests (dont 10 pour `generation_plan`)
- **pmg-math** : 174 tests (dont 13 nouveaux)
- **pmg-validate** : 16 tests (dont 15 nouveaux)
- **pmg-compare** : 18 tests (dont 18 nouveaux)
- **pmg-cli** : 9 tests (dont 7 E2E)
- **Total** : 300+ tests passent

### Documentation mise à jour

- `docs/PROJECT_SUMMARY.md` : Résumé complet du projet
- `docs/user_guide.md` : Guide d'utilisation du CLI
- `docs/architecture/04-moteurs-math-injection-generation.md` : Sprints 6-9
- `docs/architecture/06-outils-inspection-validation-comparaison.md` : Sprint 11
- `CHANGELOG.md` : Modifications des Sprints 6-11

## 7. Prochaines étapes

### Sprints 12–17 (prévu)
- Intégration complète des composants mathématiques
- Écriture SafeTensors réelle
- Tests d'intégration avec modèles réels
- Optimisation des performances
- Calibration empirique (phase 18+)

### Améliorations futures
- Support HTTP pour inspection distante
- Parallelisme avec rayon
- Benchmarking de performance
- Documentation API complète
