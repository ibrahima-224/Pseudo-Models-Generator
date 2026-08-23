# Profils Statistiques Externes

Ce dossier contient les profils statistiques externes pour la génération de pseudo-modèles dans PMG.

## Structure

```
statistical_profiles/
├── glm52.json                 # Profil pour GLM-5.2
├── deepseek_v4_flash.json     # Profil pour DeepSeek-V4-Flash
└── README.md                  # Ce fichier
```

## Schéma JSON

Chaque profil statistique suit le schéma suivant :

```json
{
  "name": "string",                    // Nom unique du profil
  "version": "string",                 // Version du profil (sémantique)
  "description": "string",             // Description du profil
  "distributions": {
    "weight_distribution": "string",   // Type de distribution pour les poids
    "outlier_distribution": "string",  // Type de distribution pour les outliers
    "correlation_strength": "float",   // Force des corrélations (0.0 à 1.0)
    "low_rank_strength": "float"       // Force de la structure à faible rang (0.0 à 1.0)
  },
  "outlier_config": {
    "probability": "float",            // Probabilité d'apparition (0.0 à 1.0)
    "severity_factor": "float",        // Facteur de sévérité (> 0.0)
    "layer_variation": "bool"          // Variation entre les couches
  },
  "correlation_config": {
    "enabled": "bool",                 // Activation des corrélations
    "max_correlation": "float",        // Corrélation maximale (0.0 à 1.0)
    "layer_decay": "float"             // Taux de décroissance (0.0 à 1.0)
  },
  "low_rank_config": {
    "enabled": "bool",                 // Activation de la structure à faible rang
    "rank_ratio": "float",             // Ratio du rang (0.0 à 1.0)
    "strength": "float"                // Force de la structure (0.0 à 1.0)
  },
  "super_weight_config": {
    "enabled": "bool",                 // Activation des super-poids
    "probability": "float",            // Probabilité d'apparition (0.0 à 1.0)
    "magnitude_factor": "float"        // Facteur de magnitude (> 0.0)
  }
}
```

## Types de Distributions Supportés

- `normal` : Distribution normale (gaussienne)
- `student_t` : Distribution de Student-t (queues lourdes)
- `laplace` : Distribution de Laplace
- `log_normal` : Distribution log-normale

## Utilisation

### Chargement depuis un fichier

```rust
use pmg_core::statistical_profile::StatisticalProfile;
use std::path::Path;

// Chargement depuis un fichier JSON
let path = Path::new("statistical_profiles/glm52.json");
let profile = StatisticalProfile::load_from_file(path)
    .expect("profil valide");
```

### Utilisation avec le pipeline de génération

```rust
use pmg_core::generator_config::GeneratorConfig;
use pmg_core::statistical_profile::StatisticalProfile;

// Chargement du profil
let profile = StatisticalProfile::glm52_default();

// Création de la configuration à partir du profil
let config = GeneratorConfig::from_statistical_profile(
    42,           // seed
    "glm-5.2",   // model_id
    &profile
).expect("configuration valide");
```

### Profils par défaut

Des profils par défaut sont disponibles directement dans le code :

```rust
use pmg_core::statistical_profile::StatisticalProfile;

// Profil GLM-5.2
let glm52_profile = StatisticalProfile::glm52_default();

// Profil DeepSeek-V4-Flash
let deepseek_profile = StatisticalProfile::deepseek_v4_flash_default();
```

## Validation

Les profils sont validés au chargement avec les règles suivantes :

1. **Nom et version** : ne doivent pas être vides
2. **Distributions** : les forces doivent être entre 0.0 et 1.0
3. **Outliers** : la probabilité doit être entre 0.0 et 1.0, le facteur de sévérité > 0.0
4. **Corrélations** : si activé, les valeurs doivent être entre 0.0 et 1.0
5. **Faible rang** : si activé, les ratios et forces doivent être entre 0.0 et 1.0
6. **Super-poids** : si activé, la probabilité doit être entre 0.0 et 1.0, le facteur > 0.0

## Création de Nouveaux Profils

Pour créer un nouveau profil :

1. Copier un profil existant comme base
2. Modifier les paramètres selon les besoins du modèle
3. Valider le profil avec `profile.validate()`
4. Sauvegarder en JSON dans ce dossier

## Intégration avec les Modèles

Les profils statistiques sont conçus pour être utilisés avec les profils de modèles définis dans `crates/pmg-models/`. La combinaison des deux permet de configurer complètement la génération pour un modèle spécifique.

## Notes Techniques

- Les profils sont chargés depuis des fichiers JSON externes pour permettre la personnalisation sans recompilation
- La validation est effectuée au chargement pour détecter les erreurs tôt
- Les profils par défaut sont compilés dans la crate pour une utilisation sans fichier externe
- Le mécanisme de fallback utilise les valeurs par défaut si le profil n'est pas disponible
