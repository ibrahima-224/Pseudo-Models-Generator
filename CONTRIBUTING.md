# Guide de contribution

Merci de contribuer au projet PMG ! Voici les directives à suivre.

## Outil de build

Le projet utilise Cargo comme outil de build. Les commandes principales :

```bash
# Compiler le workspace
cargo build --workspace

# Exécuter les tests
cargo test --workspace

# Vérifier le formatage
cargo fmt --all -- --check

# Vérifier les lints
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Règles de taille des fichiers

Tous les fichiers `.rs` doivent respecter les limites suivantes :

- **Code source** : maximum 500 lignes non-commentaires, non-blank
- **Tests** : maximum 500 lignes non-commentaires, non-blank
- **Kernels/shaders/GPU source** : maximum 300 lignes (non applicable à ce projet)

Le script `scripts/check_file_size.sh` vérifie automatiquement ces limites.

## Conventions de code

- **Langue de la documentation** : français
- **Noms techniques** : anglais (variables, fonctions, modules)
- **Formatage** : `cargo fmt` avec la configuration du projet (`.rustfmt.toml`)
- **Lints** : `cargo clippy` sans avertissements (`-D warnings`)

## Tests

Chaque modification significative doit être accompagnée de tests appropriés.

```bash
# Exécuter tous les tests
cargo test --workspace

# Exécuter les tests d'un crate spécifique
cargo test --package pmg-math
```

Les tests doivent être placés dans le répertoire `tests/` ou dans un sous-module `tests` au sein du fichier source.

## Commits

Le projet utilise le format [Conventional Commits](https://www.conventionalcommits.org/fr/v1.0.0/) :

- `feat(scope)` : ajout de fonctionnalité
- `fix(scope)` : correction de bug
- `test(scope)` : ajout de tests
- `refactor(scope)` : refactoring sans changement de comportement
- `docs` : documentation
- `chore` : tâches de maintenance

Exemples :
```
feat(math): ajout de la distribution de Student-t
fix(injector): correction du masque d'outliers
test(core): ajout de tests pour TensorAtlas
refactor(covariance): extraction des tests dans un sous-module
docs: mise à jour du README
```

## Revue de code

Toutes les modifications doivent passer par un processus de revue de code avant d'être fusionnées.

### Processus

1. **Soumettre une PR** : Créer une requête de fusion avec une description claire.
2. **Utiliser le template** : Remplir le template de revue de code disponible dans `docs/review-template.md`.
3. **Vérifications obligatoires** :
   - Tous les tests passent (`cargo test --workspace`)
   - Clippy sans warning (`cargo clippy --workspace -- -D warnings`)
   - Formatage correct (`cargo fmt --all -- --check`)
   - Documentation à jour
   - Tests unitaires pour les nouvelles fonctionnalités
4. **Revue par les pairs** : Au moins un relecteur doit approuver la PR.
5. **Merge** : Après approbation et vérification CI, la PR peut être fusionnée.

### Points de vérification

Le template de revue de code (`docs/review-template.md`) inclut :
- Vérifications de qualité du code
- Contrôles de sécurité
- Évaluation des performances
- Vérification de la documentation
- Tests de robustesse
- Conformité aux standards du projet

### Cas limites

Consulter `docs/cas-limites.md` pour les cas limites connus et les comportements attendus.

## Licence

Ce projet est sous licence GPL-3.0. En contribuant, vous acceptez que vos contributions soient soumises à cette licence.
