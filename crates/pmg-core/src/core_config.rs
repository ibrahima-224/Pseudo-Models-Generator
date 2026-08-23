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

//! Types fondamentaux de configuration du générateur.
//!
//! Ce module contient les types purs de configuration qui ne dépendent pas
//! de l'I/O, du RNG ou des algorithmes métier. Ces types sont partagés
//! entre pmg-core, pmg-io et pmg-generator.
//!
//! Conformité : ADR-002, étape 1 - Split GeneratorConfig.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_core::core_config::CoreConfig;
//!
//! let config = CoreConfig::new(42, "glm-5.2").unwrap();
//! assert_eq!(config.seed, 42);
//! assert_eq!(config.model_id, "glm-5.2");
//! ```

use serde::{Deserialize, Serialize};

use crate::distribution_config::DistributionConfig;
use crate::error::{CoreError, CoreResult};
use crate::outlier_metadata::OutlierStrategyKind;
use crate::statistical_profile::StatisticalProfile;
use crate::structure_config::StructureConfig;
use crate::structure_config::StructureStrength;

/// Stratégie d'amplitude pour les outliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmplitudeStrategy {
    /// Amplitude proportionnelle à l'écart-type du tenseur.
    StdDev,
    /// Amplitude fixe (valeur absolue).
    Fixed,
    /// Amplitude basée sur le percentile.
    Percentile,
}

impl AmplitudeStrategy {
    /// Nom lisible en français.
    pub fn display_name(self) -> &'static str {
        match self {
            AmplitudeStrategy::StdDev => "écart-type",
            AmplitudeStrategy::Fixed => "fixe",
            AmplitudeStrategy::Percentile => "percentile",
        }
    }
}

/// Modes de génération disponibles pour la CLI.
///
/// Ces modes contrôlent le comportement du générateur lors de la création
/// des tenseurs et de la sélection des distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationMode {
    /// Mode sûr : distributions conservatifs, pas d'outliers.
    ///
    /// Utilise des distributions normales standard sans injection
    /// d'outliers ni de structure complexe.
    Safe,
    /// Mode réaliste : distributions réalistes avec outliers.
    ///
    /// Utilise les distributions du modèle source avec injection
    /// d'outliers proportionnels pour un rendu plus réaliste.
    Realistic,
    /// Mode compression : optimiser pour la taille.
    ///
    /// Réduit la taille des tenseurs en utilisant des distributions
    /// plus compacts et en éliminant les éléments redondants.
    Compression,
    /// Mode stress : maximale complexité pour tests.
    ///
    /// Génère des tenseurs avec une complexité maximale pour tester
    /// les limites du système (outliers extrêmes, structures complexes).
    Stress,
}

impl GenerationMode {
    /// Retourne le nom lisible en français du mode.
    pub fn display_name(self) -> &'static str {
        match self {
            GenerationMode::Safe => "sûr",
            GenerationMode::Realistic => "réaliste",
            GenerationMode::Compression => "compression",
            GenerationMode::Stress => "stress",
        }
    }
}

/// Configuration des outliers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierConfig {
    /// Stratégie d'injection (additive ou multiplicative).
    pub strategy: OutlierStrategyKind,
    /// Fraction cible d'outliers (0.0 à 1.0).
    pub target_fraction: f64,
    /// Stratégie d'amplitude.
    pub amplitude_strategy: AmplitudeStrategy,
    /// Paramètre d'amplitude (valeur fixe ou percentile selon la stratégie).
    pub amplitude_value: f64,
}

impl OutlierConfig {
    /// Crée une configuration d'outliers avec des valeurs par défaut.
    pub fn default_additive() -> Self {
        Self {
            strategy: OutlierStrategyKind::Additive,
            target_fraction: 0.01,
            amplitude_strategy: AmplitudeStrategy::StdDev,
            amplitude_value: 5.0,
        }
    }

    /// Valide la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if !(0.0..=1.0).contains(&self.target_fraction) {
            return Err(CoreError::Validation(format!(
                "target_fraction hors [0.0, 1.0] : {}",
                self.target_fraction
            )));
        }
        if self.amplitude_value <= 0.0 {
            return Err(CoreError::Validation(format!(
                "amplitude_value doit être > 0 : {}",
                self.amplitude_value
            )));
        }
        Ok(())
    }
}

/// Configuration fondamentale du générateur (types purs).
///
/// Cette structure contient uniquement les paramètres de base nécessaires
/// à la génération, sans les spécificités CLI ou I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    /// Seed globale de génération (non nulle).
    pub seed: u64,
    /// Identifiant du modèle.
    pub model_id: String,
    /// Version du générateur.
    pub generation_version: String,
    /// Configuration de distribution de base.
    pub distribution: DistributionConfig,
    /// Configuration de structure.
    pub structure: StructureConfig,
    /// Configuration des outliers.
    pub outlier: OutlierConfig,
    /// Taille cible du pseudo-modèle en octets (0 = non spécifié).
    pub target_size_bytes: u64,
    /// Type de données de sortie (ex: "f32", "f16", "bf16", "i8").
    pub dtype: String,
}

impl CoreConfig {
    /// Crée une configuration avec des valeurs par défaut raisonnables.
    ///
    /// # Paramètres
    /// - `seed` : seed globale (non nulle) ;
    /// - `model_id` : identifiant du modèle.
    ///
    /// # Erreurs
    /// Retourne une erreur si `seed == 0` ou si `model_id` est vide.
    pub fn new(seed: u64, model_id: impl Into<String>) -> CoreResult<Self> {
        let model_id = model_id.into();
        if seed == 0 {
            return Err(CoreError::InvalidSeed(
                "seed globale nulle interdite".into(),
            ));
        }
        if model_id.trim().is_empty() {
            return Err(CoreError::Validation(
                "model_id ne peut pas être vide".into(),
            ));
        }

        // Distribution normale par défaut : N(0, 0.02)
        let distribution = DistributionConfig::normal(0.0, 0.02);

        // Structure avec force 0.3 (légère structure)
        let structure = StructureConfig::new(StructureStrength::new(0.3)?);

        // Outliers : 1% additive, amplitude 5σ
        let outlier = OutlierConfig::default_additive();

        Ok(Self {
            seed,
            model_id,
            generation_version: crate::PMG_VERSION.to_string(),
            distribution,
            structure,
            outlier,
            target_size_bytes: 0,
            dtype: "f32".to_string(),
        })
    }

    /// Crée une configuration sans structure ni outliers (pur).
    pub fn pure(seed: u64, model_id: impl Into<String>) -> CoreResult<Self> {
        let mut config = Self::new(seed, model_id)?;
        config.structure = StructureConfig::new(StructureStrength::new(0.0)?);
        config.outlier.target_fraction = 0.0;
        Ok(config)
    }

    /// Valide la cohérence interne de la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if self.seed == 0 {
            return Err(CoreError::InvalidSeed(
                "seed globale nulle interdite".into(),
            ));
        }
        if self.model_id.trim().is_empty() {
            return Err(CoreError::Validation(
                "model_id ne peut pas être vide".into(),
            ));
        }
        // Valide les sous-configurations
        self.outlier.validate()?;
        Ok(())
    }

    /// Sérialise la configuration en JSON pretty.
    pub fn to_json(&self) -> CoreResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| CoreError::Internal(format!("échec sérialisation JSON : {e}")))
    }

    /// Désérialise la configuration depuis du JSON.
    pub fn from_json(json: &str) -> CoreResult<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|e| CoreError::Internal(format!("échec désérialisation JSON : {e}")))?;
        config.validate()?;
        Ok(config)
    }
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self::new(42, "unknown").expect("configuration par défaut valide")
    }
}

impl CoreConfig {
    /// Crée une configuration à partir d'un profil statistique.
    ///
    /// Cette méthode permet d'intégrer les profils statistiques externes
    /// dans le pipeline de génération existant.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed globale de génération (non nulle).
    /// * `model_id` - Identifiant du modèle.
    /// * `profile` - Profil statistique à utiliser.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si `seed == 0`, si `model_id` est vide,
    /// ou si le profil est invalide.
    pub fn from_statistical_profile(
        seed: u64,
        model_id: impl Into<String>,
        profile: &StatisticalProfile,
    ) -> CoreResult<Self> {
        let model_id = model_id.into();
        if seed == 0 {
            return Err(CoreError::InvalidSeed(
                "seed globale nulle interdite".into(),
            ));
        }
        if model_id.trim().is_empty() {
            return Err(CoreError::Validation(
                "model_id ne peut pas être vide".into(),
            ));
        }

        // Valide le profil
        profile.validate()?;

        // Convertit la distribution du profil en DistributionConfig de pmg-core
        let distribution = match profile.distributions.weight_distribution {
            crate::statistical_profile::WeightDistribution::Normal => {
                DistributionConfig::normal(0.0, 0.02)
            },
            crate::statistical_profile::WeightDistribution::StudentT => {
                DistributionConfig::student_t(5.0)
            },
            crate::statistical_profile::WeightDistribution::Laplace => {
                DistributionConfig::laplace(0.0, 0.02)
            },
            crate::statistical_profile::WeightDistribution::LogNormal => {
                DistributionConfig::log_normal(0.0, 0.02)
            },
        };

        // Structure avec la force du profil
        let structure = StructureConfig::new(StructureStrength::new(
            profile.distributions.low_rank_strength,
        )?);

        // Outliers avec les paramètres du profil
        let outlier = OutlierConfig {
            strategy: OutlierStrategyKind::Additive,
            target_fraction: profile.outlier_config.probability,
            amplitude_strategy: AmplitudeStrategy::StdDev,
            amplitude_value: profile.outlier_config.severity_factor,
        };

        Ok(Self {
            seed,
            model_id,
            generation_version: crate::PMG_VERSION.to_string(),
            distribution,
            structure,
            outlier,
            target_size_bytes: 0,
            dtype: "f32".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_creation() {
        let config = CoreConfig::new(42, "glm-5.2").unwrap();
        assert_eq!(config.seed, 42);
        assert_eq!(config.model_id, "glm-5.2");
        assert_eq!(config.generation_version, crate::PMG_VERSION);
    }

    #[test]
    fn config_creation_zero_seed() {
        let result = CoreConfig::new(0, "glm-5.2");
        assert!(result.is_err());
    }

    #[test]
    fn config_creation_empty_model_id() {
        let result = CoreConfig::new(42, "");
        assert!(result.is_err());
        let result = CoreConfig::new(42, "   ");
        assert!(result.is_err());
    }

    #[test]
    fn config_pure() {
        let config = CoreConfig::pure(42, "test").unwrap();
        assert_eq!(config.structure.strength().value(), 0.0);
        assert_eq!(config.outlier.target_fraction, 0.0);
    }

    #[test]
    fn config_validation() {
        let mut config = CoreConfig::new(42, "test").unwrap();
        assert!(config.validate().is_ok());

        // Invalide : outlier fraction hors bornes
        config.outlier.target_fraction = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let config = CoreConfig::new(42, "glm-5.2").unwrap();
        let json = config.to_json().unwrap();
        let restored = CoreConfig::from_json(&json).unwrap();
        assert_eq!(config.seed, restored.seed);
        assert_eq!(config.model_id, restored.model_id);
    }

    #[test]
    fn outlier_config_default() {
        let outlier = OutlierConfig::default_additive();
        assert_eq!(outlier.strategy, OutlierStrategyKind::Additive);
        assert_eq!(outlier.target_fraction, 0.01);
        assert_eq!(outlier.amplitude_strategy, AmplitudeStrategy::StdDev);
        assert_eq!(outlier.amplitude_value, 5.0);
    }

    #[test]
    fn outlier_config_validation() {
        let mut outlier = OutlierConfig::default_additive();
        assert!(outlier.validate().is_ok());

        outlier.target_fraction = -0.1;
        assert!(outlier.validate().is_err());

        outlier.target_fraction = 0.5;
        outlier.amplitude_value = -1.0;
        assert!(outlier.validate().is_err());
    }

    #[test]
    fn amplitude_strategy_display() {
        assert_eq!(AmplitudeStrategy::StdDev.display_name(), "écart-type");
        assert_eq!(AmplitudeStrategy::Fixed.display_name(), "fixe");
        assert_eq!(AmplitudeStrategy::Percentile.display_name(), "percentile");
    }

    #[test]
    fn config_from_statistical_profile() {
        use crate::statistical_profile::StatisticalProfile;

        let profile = StatisticalProfile::glm52_default();
        let config = CoreConfig::from_statistical_profile(42, "glm-5.2", &profile).unwrap();

        assert_eq!(config.seed, 42);
        assert_eq!(config.model_id, "glm-5.2");
        assert_eq!(config.generation_version, crate::PMG_VERSION);

        // Vérifie que les paramètres du profil sont correctement convertis
        assert_eq!(
            config.structure.strength().value(),
            profile.distributions.low_rank_strength
        );
        assert_eq!(
            config.outlier.target_fraction,
            profile.outlier_config.probability
        );
        assert_eq!(
            config.outlier.amplitude_value,
            profile.outlier_config.severity_factor
        );
    }

    #[test]
    fn config_from_statistical_profile_zero_seed() {
        use crate::statistical_profile::StatisticalProfile;

        let profile = StatisticalProfile::glm52_default();
        let result = CoreConfig::from_statistical_profile(0, "glm-5.2", &profile);
        assert!(result.is_err());
    }

    #[test]
    fn config_from_statistical_profile_empty_model_id() {
        use crate::statistical_profile::StatisticalProfile;

        let profile = StatisticalProfile::glm52_default();
        let result = CoreConfig::from_statistical_profile(42, "", &profile);
        assert!(result.is_err());
    }

    #[test]
    fn config_from_statistical_profile_invalid_profile() {
        use crate::statistical_profile::StatisticalProfile;

        let mut profile = StatisticalProfile::glm52_default();
        profile.name = "".to_string(); // Profil invalide

        let result = CoreConfig::from_statistical_profile(42, "glm-5.2", &profile);
        assert!(result.is_err());
    }

    #[test]
    fn generation_mode_display() {
        assert_eq!(GenerationMode::Safe.display_name(), "sûr");
        assert_eq!(GenerationMode::Realistic.display_name(), "réaliste");
        assert_eq!(GenerationMode::Compression.display_name(), "compression");
        assert_eq!(GenerationMode::Stress.display_name(), "stress");
    }
}
