# PMG — Pseudo-Models Generator

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![Rust](https://img.shields.io/badge/rust-1.80+-brightgreen)
![Architecture](https://img.shields.io/badge/architecture-12--crates-blueviolet)

**Générateur déterministe de pseudo-modèles de réseaux de neurones** — conçu pour la génération zéro-payload, le déterminisme mathématique et la sécurité industrielle.

---

## Table des matières

1. [Description](#description)
2. [Fonctionnalités principales](#fonctionnalités-principales)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
6. [Astuces et conseils](#astuces-et-conseils)
7. [Limitations et avertissements](#limitations-et-avertissements)
8. [Documentation technique](#documentation-technique)
9. [Contribution](#contribution)
10. [FAQ](#faq)
11. [Licence](#licence)
12. [Contact et support](#contact-et-support)
13. [Remerciements](#remerciements)

---

## Description

PMG (Pseudo-Models Generator) est un outil de génération de pseudo-modèles pour réseaux de neurones. Il permet de créer des modèles synthétiques reproduisant les propriétés statistiques et structurelles de modèles réels, sans exposer les poids originaux.

### Principes fondamentaux

- **Zero-Payload** : Génération de tenseurs aléatoires avec propriétés statistiques contrôlées, sans données réelles
- **Déterminisme** : Reproductibilité garantie grâce à des graines cryptographiques (SHA-256)
- **Sécurité** : Validation statistique rigoureuse, auditabilité complète, conformité aux normes industrielles

### Cas d'utilisation

- **Benchmarking** : Évaluation de performances sans exposer les poids du modèle
- **Recherche** : Études comparatives de propriétés de modèles
- **Test** : Validation de pipelines de traitement de modèles
- **Sécurité** : Gestion de modèles sans risque de fuite de données

---

## Fonctionnalités principales

### Génération de modèles
- Génération de tenseurs avec distributions contrôlées (normale, log-normale, Pareto, Weibull)
- Support de l'architecture Mixture-of-Experts (MoE)
- Export au format Safetensors (standard industriel)
- Mode streaming pour les grands modèles

### Validation et analyse
- Validation statistique complète (distributions, corrélations, rang faible)
- Détection d'anomalies et d'outliers
- Analyse de similarité entre modèles
- Métriques de qualité (Kolmogorov-Smirnov, Anderson-Darling)

### Performance
- Support GPU (CUDA) avec kernels optimisés
- Parallélisme multi-GPU
- Pool de mémoire optimisé
- Compression de données intégrée

### Sécurité
- Validation des entrées (Zod, Pydantic)
- Protection contre les injections
- Audit complet des opérations
- Conformité aux normes industrielles

---

## Architecture

PMG est structuré en **12 crates spécialisées**, chacune responsable d'un domaine fonctionnel précis.

Pour plus de détails sur l'architecture et les dépendances entre crates, voir le fichier [Diagrammes d'architecture](docs/architecture-diagrams.md).

### Vue d'ensemble

- **Couche Interface** : CLI, noyau, planification
- **Couche Métier** : génération, validation, comparaison, analyse
- **Couche Infrastructure** : mathématiques, entrées/sorties, GPU, compression
- **Couche Données** : modèles, métadonnées, injection

---

## Installation

### Prérequis

- **Rust** 1.80+ (recommandé : stable)
- **Cargo** (gestionnaire de packages Rust)
- **Git** (pour le clonage)
- **CMake** (optionnel, pour GPU)
- **CUDA Toolkit** (optionnel, pour accélération GPU)

### Installation depuis les sources

```bash
# Cloner le dépôt
git clone https://github.com/Ibrahima-224/Pseudo-Models-Generator.git
cd Pseudo-Models-Generator

# Compiler le projet (mode release pour les performances)
cargo build --release

# Installer le binaire
cargo install --path crates/pmg-cli

# Vérifier l'installation
pmg --version
```

### Installation via crates.io (à venir)

```bash
cargo install pmg-cli
```

### Compilation avec support GPU

```bash
# Activer le support CUDA
cargo build --release --features gpu

# Ou compiler avec tous les features
cargo build --release --all-features
```

---

## Utilisation

### Commandes principales

#### Génération de modèles

```bash
# Générer un modèle GLM-5.2 (1GB)
pmg generate --model glm52 --size 1G --output glm52_model.safetensors

# Générer un modèle DeepSeek-V4-Flash (10GB)
pmg generate --model deepseek-v4-flash --size 10G --output deepseek_model.safetensors

# Générer avec des paramètres personnalisés
pmg generate \
  --layers 32 \
  --hidden-size 4096 \
  --num-heads 32 \
  --distribution normal \
  --seed 42 \
  --output custom_model.safetensors

# Mode dry-run (simulation sans écriture)
pmg generate --model glm52 --size 1G --dry-run
```

#### Validation de modèles

```bash
# Valider un modèle
pmg validate --model-path model.safetensors

# Validation détaillée avec rapport complet
pmg validate --model-path model.safetensors --verbose --report detailed

# Validation rapide (métriques de base)
pmg validate --model-path model.safetensors --quick
```

#### Comparaison de modèles

```bash
# Comparer deux modèles
pmg compare \
  --original model1.safetensors \
  --compared model2.safetensors

# Comparaison avec score de similarité
pmg compare \
  --original model1.safetensors \
  --compared model2.safetensors \
  --output comparison_report.json

# Comparaison spécifique (uniquement les tenseurs d'embedding)
pmg compare \
  --original model1.safetensors \
  --compared model2.safetensors \
  --filter "embedding"
```

#### Spécifications de modèles

```bash
# Afficher les spécifications d'un modèle
pmg espec --model-path model.safetensors

# Spécifications détaillées
pmg espec --model-path model.safetensors --verbose

# Export des spécifications
pmg espec --model-path model.safetensors --output specs.json
```

#### Options globales

```bash
# Mode debug (logs détaillés)
pmg --debug generate --model glm52

# Mode verbose (sortie détaillée)
pmg --verbose validate --model-path model.safetensors

# Version avec détails
pmg version --verbose
```

### Exemples d'utilisation

#### Pipeline complet de génération

```bash
# 1. Générer un modèle
pmg generate --model glm52 --size 1G --output glm52.safetensors

# 2. Valider le modèle généré
pmg validate --model-path glm52.safetensors

# 3. Afficher les spécifications
pmg espec --model-path glm52.safetensors

# 4. Comparer avec un autre modèle
pmg compare \
  --original glm52.safetensors \
  --compared deepseek_model.safetensors
```

#### Utilisation en script

```bash
#!/bin/bash
# Script de génération et validation

MODEL="glm52"
SIZE="2G"
OUTPUT="model.safetensors"

# Génération
echo "Génération du modèle $MODEL..."
pmg generate --model $MODEL --size $SIZE --output $OUTPUT

# Validation
echo "Validation du modèle..."
if pmg validate --model-path $OUTPUT; then
  echo "✅ Modèle valide"
else
  echo "❌ Modèle invalide"
  exit 1
fi

# Spécifications
echo "Spécifications du modèle:"
pmg espec --model-path $OUTPUT --verbose
```

---

## Astuces et conseils

### Optimisations de performance
- **Mode de génération** : `--mode size-constrained` (défaut), `--mode full-structural`, `--mode streaming` (>10GB)
- **Paramètres de chunk** : `--chunk-elements 2097152` (accélère) ou `524288` (réduit mémoire)
- **Compression** : `--compress`, niveau `--compression-level 1-9` (défaut : 3), algorithme `--compression-algorithm zstd|lz4`
- **Parallelisme** : `--parallel-tensors 4`, `--parallel-layers 2` (selon ressources)

### Bonnes pratiques d'utilisation
- **Profils** : créer dans `profiles/`, versionner, valider avec `pmg validate --profile custom.json`
- **Scripts** : utiliser `--dry-run`, valider après génération, stocker les graines (`--seed`)
- **Sorties** : structure standardisée, noms descriptifs avec timestamps, conserver manifestes et rapports
- **Documentation** : commenter paramètres, documenter choix de distribution, conserver rapports de validation

### Conseils pour le débogage
- **Logs** : `--debug` (développement), `--verbose` (production), `--report json` (analyse automatisée)
- **Validation** : `--quick` (rapide), `--report detailed` (complet), `--statistical-profile strict` (renforcé)
- **Erreurs** : codes de sortie (0-6), messages contextuels, rapports JSON
- **Profiling** : `cargo flamegraph`, `--benchmark`, `--memory-profile`

### Utilisation avancée des profils
- **Structure** : modèle JSON avec `model`, `architecture`, `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `distributions`
- **Personnalisation** : profils minimaux, paramètres par couche/tenseur, configurations MoE
- **Validation** : `pmg validate --profile custom.json`, tests d'intégration, comparaison avec profils de référence
- **Optimisation** : réduction paramètres, distributions optimisées, équilibrage précision/performance

---

## Limitations et avertissements

### Limitations de taille de modèle
- **Limite mémoire** : 100GB (recommandé : 50GB)
- **Limite disque** : 2x la taille cible
- **Limite tenseur** : 2^31 éléments (2.1 milliards)
- **Optimisations** : mode streaming >50GB, ajuster `--chunk-elements`, compression

### Problèmes connus
- **Formats non supportés** : sous-octets (F4, F6, F8E8M0), quantification (NF4, GPTQ, AWQ)
- **Fonctionnalités manquantes** : support distribué, HTTP pour grands fichiers
- **Bugs** : écrasement de fichiers (`--force`), messages dupliqués (corrigé v1.0.1)
- **Compatibilité** : Safetensors v0.3, CUDA ≥11.7, Rust 1.80+

### Compatibilité matérielle
- **Plateformes** : Linux (x86_64, ARM64), Windows (x86_64), macOS (x86_64, ARM64)
- **Configuration recommandée** : CPU 8+ cœurs, RAM 16-32GB, SSD, GPU NVIDIA (optionnel)
- **GPU** : Compute Capability ≥7.0, Mémoire ≥8GB, Pilotes ≥525.60, CUDA ≥11.7

---

## Documentation technique

### Architecture Decision Records (ADRs)

Les décisions architecturales sont documentées dans le dossier `docs/architecture/` :

- `ADR-001` : Choix du format Safetensors
- `ADR-002` : Architecture en crates
- `ADR-003` : Stratégie de validation statistique
- `ADR-004` : Support GPU
- `ADR-005` : Sécurité et auditabilité

### Audits de sécurité

- `SECURITY.md` : Politique de sécurité
- `docs/security/` : Rapports d'audit
- Tests de sécurité dans chaque crate

### Documentation des crates

Chaque crate contient sa propre documentation. Pour générer : `cargo doc --open`

### Benchmarks

Exécuter les benchmarks avec `cargo bench`. Spécifiques : `cargo bench -p pmg-bench`. GPU : `cargo bench --features gpu`.

### Profils de modèles

Les profils définissent les caractéristiques des modèles à générer. Voir `profiles/` pour les exemples.

### Configuration GPU avancée

Vérifier la compatibilité : `pmg generate --model glm52 --gpu-check`. Configurer la mémoire : `--gpu-memory-fraction 0.8`. Multi-GPU : `--multi-gpu --gpu-count 2`.

### Tests

Exécuter tous les tests : `cargo test`. Spécifiques : `cargo test -p pmg-validate`. Couverture : `cargo tarpaulin`.

---

## Contribution

### Guide de contribution

1. **Fourchonner** le dépôt
2. **Créer** une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. **Développer** en suivant les conventions du projet
4. **Tester** vos modifications
5. **Soumettre** une pull request

### Conventions de code

- **Langage** : Rust (édition 2021)
- **Formatage** : `rustfmt` avec configuration standard
- **Linting** : `clippy` avec les lints stricts
- **Documentation** : Toute fonction publique doit être documentée
- **Tests** : Couverture minimale de 80%

### Conventional Commits

Nous utilisons les [Conventional Commits](https://www.conventionalcommits.org/) :

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types :
- `feat` : Nouvelle fonctionnalité
- `fix` : Correction de bug
- `docs` : Documentation
- `style` : Formatage (sans changement de code)
- `refactor` : Refactorisation
- `test` : Ajout de tests
- `chore` : Maintenance

Exemples :
```
feat(pmg-math): ajout de la distribution de Student-T
fix(pmg-io): correction du buffer overflow
docs: mise à jour du README
```

### Structure des pull requests

1. **Titre** : Description concise des changements
2. **Description** : Détail des modifications
3. **Tests** : Nouveaux tests ajoutés
4. **Documentation** : Mise à jour de la doc si nécessaire
5. **Breaking changes** : Si applicable

### Revue de code

- Toutes les pull requests nécessitent au moins 2 approbations
- Les tests doivent passer
- La couverture de code ne doit pas diminuer
- La documentation doit être à jour

---

## FAQ

### Questions techniques courantes

#### Comment activer le support GPU ?
1. Installer CUDA Toolkit ≥ 11.7
2. Compiler avec `--features gpu`
3. Vérifier avec `nvidia-smi` que les GPU sont disponibles
4. Utiliser `pmg generate --gpu-check` pour tester la compatibilité

#### Quelle est la taille maximale des modèles supportés ?
- **Limite pratique** : 100GB (contraintes mémoire système)
- **Recommandation** : utiliser le mode streaming pour les modèles > 10GB
- **Optimisation** : ajuster `--chunk-elements` selon la mémoire disponible

#### Comment changer la distribution statistique ?
```bash
# Distribution normale (défaut)
pmg generate --distribution normal --mean 0.0 --std 1.0
# Distribution log-normale
pmg generate --distribution log_normal --mean 0.0 --std 0.5
# Distribution de Pareto
pmg generate --distribution pareto --shape 3.0 --scale 1.0
# Distribution de Weibull
pmg generate --distribution weibull --shape 2.0 --scale 1.0
```

#### Comment créer un profil personnalisé ?
1. Copier un profil existant : `cp profiles/glm52.json profiles/custom.json`
2. Modifier les paramètres selon vos besoins
3. Valider : `pmg validate --profile profiles/custom.json`
4. Utiliser : `pmg generate --profile profiles/custom.json`

### Dépannage des erreurs fréquentes

#### Erreur PMG-4 (Erreur I/O)
- **Cause** : Problèmes de permissions ou d'espace disque
- **Solution** :
  1. Vérifier les permissions : `ls -la /chemin/vers/sortie`
  2. Vérifier l'espace : `df -h /chemin/vers/sortie`
  3. Utiliser `--dry-run` pour tester sans écriture
  4. Changer de répertoire de sortie

#### Erreur PMG-5 (Validation échouée)
- **Cause** : Le modèle généré ne respecte pas les propriétés statistiques
- **Solution** :
  1. Utiliser `--verbose` pour voir les détails
  2. Ajuster les paramètres de distribution
  3. Utiliser `--statistical-profile strict` pour plus de tolérance
  4. Vérifier la cohérence du profil

#### Erreur PMG-6 (Comparaison incompatible)
- **Cause** : Les modèles n'ont pas la même architecture
- **Solution** :
  1. Vérifier les spécifications : `pmg espec --model-path model.safetensors`
  2. Utiliser `--filter` pour comparer des tenseurs spécifiques
  3. Générer des modèles avec le même profil
  4. Consulter les logs détaillés

### Configuration GPU

#### Problèmes courants GPU
- **Erreur "CUDA out of memory"** :
  1. Réduire `--chunk-elements` (ex: 524288)
  2. Utiliser `--gpu-memory-fraction 0.8`
  3. Fermer les autres applications GPU
  4. Utiliser un GPU avec plus de mémoire

- **Performance dégradée** :
  1. Vérifier les pilotes NVIDIA (`nvidia-smi`)
  2. Compiler en mode release (`--release`)
  3. Augmenter `--parallel-tensors`
  4. Vérifier la température GPU

- **Incompatibilité CUDA** :
  1. Vérifier la version : `nvcc --version`
  2. Installer la version correspondante
  3. Compiler avec `--features gpu`
  4. Tester avec `pmg generate --gpu-check`

### Problèmes connus et solutions

#### Fichiers corrompus en sortie
- **Cause** : Interruption pendant l'écriture
- **Solution** :
  1. Utiliser `--force` avec précaution
  2. Vérifier l'intégrité : `pmg validate`
  3. Activer les logs : `--debug`
  4. Utiliser un système de fichiers avec journalisation

#### Performance lente sur grands modèles
- **Optimisations** :
  1. Augmenter `--chunk-elements` (défaut : 1048576)
  2. Utiliser le mode streaming (`--streaming`)
  3. Compiler en mode release (`cargo build --release`)
  4. Activer la compression (`--compress`)

#### Déterminisme non garanti
- **Causes** :
  1. Graines différentes (`--seed`)
  2. Versions de PMG différentes
  3. Environnements différents (OS, compilateur)
- **Solutions** :
  1. Utiliser la même graine
  2. Utiliser la même version de PMG
  3. Documenter l'environnement de génération
  4. Utiliser des conteneurs Docker

---

## Licence

Ce projet est distribué sous la **GNU General Public License v3.0**.

```
PMG — Pseudo-Models Generator
Copyright (C) 2024 PMG Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

Pour plus d'informations, voir le fichier [LICENSE](LICENSE).

---

## Contact et support

- **Dépôt** : [GitHub](https://github.com/Ibrahima-224/Pseudo-Models-Generator)
- **Issues** : [GitHub Issues](https://github.com/Ibrahima-224/Pseudo-Models-Generator/issues)
- **Discussions** : [GitHub Discussions](https://github.com/Ibrahima-224/Pseudo-Models-Generator/discussions)

### Ressources supplémentaires
- **Documentation technique** : `docs/architecture/`
- **Changelog** : `CHANGELOG.md`
- **Guide de contribution** : `CONTRIBUTING.md`
- **Guide de déploiement** : `DEPLOYMENT_GUIDE.md`

---

## Remerciements

Merci à tous les contributeurs qui ont participé au développement de PMG.

---

*Dernière mise à jour : Août 2026*
