# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Unreleased]

### Changed

- **Refactoring** :
  - Découpage de `crates/pmg-validate/src/validator.rs` (618 lignes) en modules séparés pour respecter la limite de500 lignes.
  - Nouveau module `types.rs` contenant les types de base (`ValidationIssue`, `ValidationCategory`, `TensorValidationResult`, `ValidationResult`, `ValidationSummary`, `ValidationConfig`, `TensorData`).
  - Mise à jour de `validator.rs` pour réexporter les types via `pub use crate::types::*`.
  - Mise à jour de `lib.rs` pour déclarer le module `types`.
  - Préservation de l'API publique du crate.

### Added

#### Sprint 17 — Durcissement et Release Candidate
- **Documentation d'audit** :
  - `docs/audit_erreurs.md` — Audit complet des `unwrap()`, `expect()`, `panic!()`, `todo!()`, `unimplemented!()`
  - `docs/audit_panics.md` — Analyse des crashs brutaux et résistance aux fichiers corrompus
  - `docs/audit_unsafe.md` — Audit du code unsafe (aucune occurrence trouvée)
  - `docs/audit_dependances.md` — Audit des dépendances avec analyse des licences et sécurité
  - `docs/audit_documentation.md` — Évaluation de la complétude et qualité de la documentation

- **Tests de stress** :
  - `tests/stress/large_model_tests.rs` — Tests avec 100+ couches, 500+ tenseurs, 128+ experts MoE, grandes dimensions

- **Tests de corruption** :
  - `tests/corruption/corrupted_files_tests.rs` — Tests de résistance aux fichiers JSON invalides, Safetensors corrompus, fichiers tronqués

- **Benchmarks** :
  - `benches/final_benchmarks.rs` — Benchmarks criterion pour l'inspection, le parsing, la génération, la validation

- **CI/CD** :
  - `.github/workflows/release.yml` — Workflow de build multi-plateformes (Linux, Windows, macOS x86_64, macOS ARM64)

- **Documentation** :
  - `docs/release_checklist.md` — Checklist complète pour la release candidate

### Changed
- **Version** : Passage à v1.0.0-rc.1 (Release Candidate)
- **CHANGELOG** : Mise à jour avec les changements du Sprint 17

### Fixed
- **Bug critique** : Correction du bug d'écrasement de fichiers dans la commande `generate`
  - Ajout du flag `--force` (`-f`) pour permettre explicitement l'écrasement
  - Par défaut, demande confirmation avant d'écraser des fichiers existants
  - Vérification de l'existence du répertoire de destination avant la copie
  - Affichage d'un message d'avertissement clair quand des fichiers seraient écrasés
  - Protection contre les écrasements accidentels de fichiers de configuration
  - Tests unitaires ajoutés pour valider le comportement

- **Bug mineur** : Correction des messages d'erreur dupliqués dans toutes les commandes CLI
  - Les messages d'erreur contenaient une duplication du préfixe (ex: `Erreur PMG-4: Erreur PMG-4:`)
  - Problème identifié dans la conversion `From<anyhow::Error>` pour `CliError`
  - Solution : nettoyage des messages dans la conversion pour éviter la duplication
  - Tests de validation mis à jour et passant avec succès


## [1.0.0] - 2026-08-22

### Added
- Benchmarks de performance pour pmg-compression
- Script de tests ciblés avec monitoring mémoire
- Pipeline CI/CD GitHub Actions
- Pool de buffers pour pmg-compression et pmg-io
- Validations renforcées pour pmg-models
- 22 nouveaux tests unitaires pour pmg-validate
- 16 nouveaux tests pour pmg-cli (compare.rs, espec.rs)

### Changed
- Refactoring de pmg-gpu/allocator.rs en modules séparés
- Optimisation des calculs dans pmg-validate
- Amélioration du pooling dans pmg-io

### Fixed
- Bug de release dans OptimizedBufferPool
- Calculs redondants dans model_validator.rs

### Security
- Documentation des unsafe blocks dans pmg-gpu
- Protection contre les double frees avec AtomicBool


## [0.2.0] - 2026-08-09

### Added

#### Sprint 6 — Génération tensorielle
- **pmg-core** : `generation_plan.rs` — Plan de génération tensorielle
  - Structure `GenerationPlan` sérialisable, inspectable, déterministe
  - Méthodes : `new()`, `with_chunk_elements()`, `validate()`, `to_tensor_metadata()`
  - 10 tests unitaires
- **pmg-math** : `generator.rs` — Générateur de base
  - Fonctions : `generate_normal()`, `generate_uniform()`
  - Algorithmes : Box-Muller pour normale, formule directe pour uniforme
  - 9 tests unitaires
- **Tests d'intégration** : 16 tests (`generator_tests.rs`)

#### Sprint 7 — Distributions statistiques
- **pmg-core** : `distribution_config.rs` — Configuration des distributions
  - Structure `DistributionConfig` sérialisable (JSON/TOML)
  - Énumération `DistributionKind` (7 variantes)
  - 4 tests de sérialisation
- **pmg-math** : `distribution_tests.rs` — Validation statistique
  - 21 tests vérifiant propriétés statistiques essentielles
  - Tolérances conformes à l'architecture

#### Sprint 8 — Structures de corrélation
- **pmg-math** : Modules de structures (`structure/`)
  - `base_structure.rs`, `local_correlation.rs`, `block_structure.rs`, `correlation.rs`, `factors.rs`
  - 16+ tests dans `structure_tests.rs`

#### Sprint 9 — Super-poids et outliers
- **pmg-math** : Module `outliers/`
  - `model.rs` : Modèle de transformation (additif/multiplicatif)
  - `amplitude.rs` : Calcul de l'amplitude des anomalies
  - `layer_policy.rs` : Politique par couche
  - 16 tests dans `outlier_tests.rs`
- **pmg-core** : `outlier_metadata.rs` — Métadonnées de validation
  - Structure `OutlierMetadata` sérialisable
  - Méthodes : `new()`, `validate()`, `to_json()`, `from_json()`

#### Sprint 10 — Génération complète
- **pmg-core** : Pipeline de génération
  - `generator_config.rs` : Configuration globale
  - `generation_pipeline.rs` : Pipeline ordonné (4 étapes)
  - `tensor_generation.rs` : Génération d'un tenseur
  - `streaming_generation.rs` : Génération par streaming
  - `manifest.rs` : Manifeste du pseudo-modèle
- **pmg-io** : Écriture de configuration
  - `config_writer.rs` : Écriture de `config.json`
  - `metadata_writer.rs` : Écriture des métadonnées
- **Tests** : `full_generation.rs`, `determinism_tests.rs`

#### Sprint 11 — Validation et comparaison
- **pmg-math** : Modules d'analyse
  - `outlier_analysis.rs` : Détection d'outliers
  - `correlation_analysis.rs` : Analyse de corrélation
  - `low_rank_analysis.rs` : Analyse bas-rang
  - 13 nouveaux tests
- **pmg-validate** : Système de validation
  - `severity.rs` : Niveaux de sévérité (INFO, WARNING, ERROR, CRITICAL)
  - `validator.rs` : Validateur principal (5 catégories)
  - `report.rs` : Rapports (texte, JSON, console)
  - 15 nouveaux tests
- **pmg-compare** : Système de comparaison
  - `comparator.rs` : Comparateur principal (5 niveaux de similarité)
  - `config_comparator.rs` : Comparaison de configurations
  - `tensor_comparator.rs` : Comparaison de structures
  - `statistical_comparator.rs` : Comparaison statistique
  - 18 nouveaux tests
- **pmg-cli** : Commandes implémentées
  - `generate`, `validate`, `compare`, `espec`, `version`, `help`
  - 7 tests E2E (`tests/e2e.rs`)

### Changed
- Refactor de `covariance.rs` en sous-module (`covariance/mod.rs` + `covariance/tests.rs`)
- Refactor de `low_rank.rs` en sous-module (`low_rank/mod.rs` + `low_rank/tests.rs`)

### Fixed
- Aucun fix pour l'instant

### Documentation
- Création de `docs/PROJECT_SUMMARY.md` : Résumé complet du projet
- Création de `docs/user_guide.md` : Guide d'utilisation du CLI
- Mise à jour de `docs/architecture/README.md` : Ajout des Sprints 6-11
- Mise à jour de `docs/architecture/04-moteurs-math-injection-generation.md` : Sprints 6-9
- Mise à jour de `docs/architecture/06-outils-inspection-validation-comparaison.md` : Sprint 11

## [0.1.0] - 2026-08-09

### Added

#### Sprint 0 — Infrastructure (L0)
- Workspace Rust multi-crates avec 12 crates
- Configurations : `.rustfmt.toml`, `.clippy.toml`, `.editorconfig`, `rust-toolchain.toml`
- CI/CD : `.github/workflows/ci.yml`
- Scripts : `scripts/check_file_size.sh`
- Documentation : `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`, `README.md`

#### Sprint 1 — Core (L1) — pmg-core
- Types fondamentaux : `dtype.rs`, `shape.rs`, `tensor_metadata.rs`, `model_config.rs`, `tensor_role.rs`
- Gestion des erreurs : `error.rs`
- Validation : `validation.rs`
- Origin/Confidence : `origin.rs`
- MoE : `moe.rs`
- Storage vs Quantization : `storage_vs_quant.rs`

#### Sprint 2 — Blueprint (L2) — pmg-blueprint
- Blueprint : `blueprint.rs`
- Couche : `layer.rs`
- Spécification tenseur : `tensor_spec.rs`
- Architecture : `architecture.rs`
- MoE : `moe.rs`
- Nommage : `naming.rs`
- Planificateur : `planner.rs`
- Validation : `validation.rs`
- Erreurs : `error.rs`

#### Sprint 3 — Mathématiques (L3) — pmg-math
- RNG déterministe : `rng.rs`
- Statistiques : `statistics.rs`
- Covariance PSD (Cholesky) : `covariance/mod.rs`, `covariance/tests.rs`
- Structures bas-rang : `low_rank/mod.rs`, `low_rank/tests.rs`
- Distributions : `distributions/{mod,normal,student_t,laplace,log_normal,weibull,pareto,mixture}.rs`
- Fonctions spéciales : `special.rs`
- Trait Distribution : `distribution.rs`
- Erreurs : `error.rs`

#### Sprint 4 — Injection (L4) — pmg-injector
- Politique d'injection : `injection_policy.rs`
- Masque d'outliers : `outlier_mask.rs`
- Super-poids : `super_weight.rs`
- Corrélations : `correlated.rs`
- Bas-rang : `low_rank.rs`
- Structure sparse : `sparse_structure.rs`
- Motif de couche : `layer_pattern.rs`
- Injecteur tenseur : `tensor_injector.rs`
- Validateur : `injection_validator.rs`
- Erreurs : `error.rs`

#### Sprint 5 — Génération déterministe (L5) — pmg-generator
- Générateur : `generator.rs`
- Déterminisme : `deterministic.rs`
- Chunk : `chunk.rs`
- Plan de seed : `seed_plan.rs`
- Rapport de génération : `generation_report.rs`
- Erreurs : `error.rs`

### Changed
- Refactor de `covariance.rs` en sous-module (`covariance/mod.rs` + `covariance/tests.rs`)
- Refactor de `low_rank.rs` en sous-module (`low_rank/mod.rs` + `low_rank/tests.rs`)
