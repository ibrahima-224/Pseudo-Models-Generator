// Copyright (C) 2024 PMG Contributors
// This file is part of PMG (Pseudo-Model Generator).
//
// PMG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PMG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PMG.  If not, see <https://www.gnu.org/licenses/>.

//! Écriture du fichier de configuration du pseudo-modèle.
//!
//! Ce module fournit la fonction [`write_config`] qui produit le fichier
//! `config.json` avec les paramètres cohérents de génération.
//!
//! Conformité : Sprint 10, étape 10.5 « Fichier de configuration ».
//!
//! # Exemple
//!
//! ```rust
//! use pmg_io::config_writer::write_config;
//! use pmg_core::generator_config::GeneratorConfig;
//!
//! let config = GeneratorConfig::default();
//! let json = write_config(&config).unwrap();
//! assert!(json.contains("\"seed\""));
//! ```

use std::path::Path;

use pmg_core::core_config::AmplitudeStrategy;
use pmg_core::distribution_config::DistributionKind;
use pmg_core::error::{CoreError, CoreResult};
use pmg_core::outlier_metadata::OutlierStrategyKind;
use pmg_core::CoreConfig;
use serde::Serialize;

/// Structure de sortie pour le fichier de configuration.
///
/// Cette structure est sérialisable en JSON et contient tous les champs
/// nécessaires au format de configuration du pseudo-modèle.
#[derive(Debug, Serialize)]
struct ConfigOutput {
    /// Type de modèle (toujours "pseudo_model").
    model_type: &'static str,
    /// Architecture (toujours "transformer").
    architecture: &'static str,
    /// Version du générateur.
    generator_version: String,
    /// Seed de génération.
    seed: u64,
    /// Identifiant du modèle.
    model_id: String,
    /// Configuration de distribution.
    distribution: DistributionOutput,
    /// Configuration de structure.
    structure: StructureOutput,
    /// Configuration des outliers.
    expert_outlier: OutlierOutput,
}

/// Sortie de la distribution (format simplifié pour le JSON).
#[derive(Debug, Serialize)]
struct DistributionOutput {
    /// Type de distribution.
    kind: DistributionKind,
    /// Première valeur (mean, location, etc.).
    mean: f64,
    /// Deuxième valeur (std, scale, etc.).
    std: f64,
}

/// Sortie de la configuration de structure.
#[derive(Debug, Serialize)]
struct StructureOutput {
    /// Force structurelle (0.0 à 1.0).
    strength: f64,
    /// Type de structure (optionnel).
    structure_type: String,
}

/// Sortie de la configuration des outliers.
#[derive(Debug, Serialize)]
struct OutlierOutput {
    /// Stratégie d'injection.
    strategy: OutlierStrategyKind,
    /// Fraction cible d'outliers.
    target_fraction: f64,
    /// Stratégie d'amplitude.
    amplitude_strategy: AmplitudeStrategy,
    /// Valeur d'amplitude.
    amplitude_value: f64,
}

/// Écrit le fichier de configuration du modèle.
///
/// # Paramètres
/// - `config` : configuration de génération ;
/// - `path` : chemin du fichier à écrire.
///
/// # Erreurs
/// Retourne une erreur si l'écriture échoue.
pub fn write_config_file(config: &CoreConfig, path: &Path) -> CoreResult<()> {
    let json = write_config(config)?;
    std::fs::write(path, json)
        .map_err(|e| CoreError::Internal(format!("échec écriture config.json : {e}")))
}

/// Génère le contenu JSON de la configuration en utilisant la sérialisation serde.
///
/// La structure est validée avant sérialisation pour garantir la cohérence.
pub fn write_config(config: &CoreConfig) -> CoreResult<String> {
    // Valide la configuration avant sérialisation
    config.validate()?;

    // Construction de la structure de sortie avec les champs requis
    let output = ConfigOutput {
        model_type: "pseudo_model",
        architecture: "transformer",
        generator_version: config.generation_version.clone(),
        seed: config.seed,
        model_id: config.model_id.clone(),
        distribution: DistributionOutput {
            kind: config.distribution.kind,
            mean: config.distribution.p1,
            std: config.distribution.p2.unwrap_or(0.0),
        },
        structure: StructureOutput {
            strength: config.structure.strength().value(),
            structure_type: config
                .structure
                .structure_type()
                .unwrap_or("base")
                .to_string(),
        },
        expert_outlier: OutlierOutput {
            strategy: config.outlier.strategy,
            target_fraction: config.outlier.target_fraction,
            amplitude_strategy: config.outlier.amplitude_strategy,
            amplitude_value: config.outlier.amplitude_value,
        },
    };

    // Sérialisation en JSON pretty avec serde
    serde_json::to_string_pretty(&output)
        .map_err(|e| CoreError::Internal(format!("échec sérialisation JSON : {e}")))
}

#[cfg(test)]
mod tests {
    use super::{write_config, write_config_file};
    use pmg_core::generator_config::GeneratorConfig;

    #[test]
    fn write_config_basic() {
        let config = GeneratorConfig::new(42, "glm-5.2").unwrap();
        let json = write_config(&config).unwrap();
        assert!(json.contains("\"seed\": 42"));
        assert!(json.contains("\"model_id\": \"glm-5.2\""));
        assert!(json.contains("\"model_type\": \"pseudo_model\""));
    }

    #[test]
    fn write_config_json_structure() {
        let config = GeneratorConfig::default();
        let json = write_config(&config).unwrap();
        // Vérifie que c'est du JSON valide en cherchant des patterns
        assert!(json.contains("\"distribution\""));
        assert!(json.contains("\"structure\""));
        assert!(json.contains("\"expert_outlier\""));
        assert!(json.contains("\"seed\""));
    }

    #[test]
    fn write_config_file_success() {
        let config = GeneratorConfig::default();
        let temp_dir = std::env::temp_dir().join("pmg_test_config");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("config.json");

        let result = write_config_file(&config, &path);
        assert!(result.is_ok());

        // Vérifie que le fichier a été créé
        assert!(path.exists());

        // Nettoyage
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(temp_dir).unwrap();
    }

    #[test]
    fn serde_roundtrip() {
        let config = GeneratorConfig::new(123, "test-model").unwrap();
        let json = write_config(&config).unwrap();
        // Vérifie que le JSON contient les bonnes valeurs
        assert!(json.contains("\"seed\": 123"));
        assert!(json.contains("\"model_id\": \"test-model\""));
    }
}
