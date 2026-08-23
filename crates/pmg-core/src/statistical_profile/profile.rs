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

//! Sous-module contenant la structure principale du profil statistique.
//!
//! Ce module définit [`StatisticalProfile`], la structure qui représente
//! un profil statistique complet pour un modèle, ainsi que ses méthodes
//! de chargement, validation et création de profils par défaut.

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

use super::configs::{
    CorrelationConfig, LowRankConfig, OutlierProfileConfig, ProfileDistributionConfig,
    SuperWeightConfig, WeightDistribution,
};

/// Profil statistique complet pour un modèle.
///
/// Ce profil contient tous les paramètres nécessaires à la génération
/// des poids d'un modèle spécifique.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalProfile {
    /// Nom du profil.
    pub name: String,
    /// Version du profil.
    pub version: String,
    /// Description du profil.
    pub description: String,
    /// Configuration des distributions.
    pub distributions: ProfileDistributionConfig,
    /// Configuration des outliers.
    pub outlier_config: OutlierProfileConfig,
    /// Configuration des corrélations.
    pub correlation_config: CorrelationConfig,
    /// Configuration de la structure à faible rang.
    pub low_rank_config: LowRankConfig,
    /// Configuration des super-poids.
    pub super_weight_config: SuperWeightConfig,
}

impl StatisticalProfile {
    /// Valide le profil statistique.
    pub fn validate(&self) -> CoreResult<()> {
        if self.name.trim().is_empty() {
            return Err(CoreError::Validation(
                "le nom du profil ne peut pas être vide".into(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(CoreError::Validation(
                "la version du profil ne peut pas être vide".into(),
            ));
        }

        self.distributions.validate()?;
        self.outlier_config.validate()?;
        self.correlation_config.validate()?;
        self.low_rank_config.validate()?;
        self.super_weight_config.validate()?;

        Ok(())
    }

    /// Crée un profil par défaut pour GLM-5.2.
    pub fn glm52_default() -> Self {
        Self {
            name: "glm52_statistical_profile".to_string(),
            version: "1.0.0".to_string(),
            description: "Profil statistique par défaut pour GLM-5.2".to_string(),
            distributions: ProfileDistributionConfig {
                weight_distribution: WeightDistribution::Normal,
                outlier_distribution: WeightDistribution::StudentT,
                correlation_strength: 0.3,
                low_rank_strength: 0.1,
            },
            outlier_config: OutlierProfileConfig {
                probability: 0.01,
                severity_factor: 3.0,
                layer_variation: true,
            },
            correlation_config: CorrelationConfig {
                enabled: true,
                max_correlation: 0.5,
                layer_decay: 0.9,
            },
            low_rank_config: LowRankConfig {
                enabled: true,
                rank_ratio: 0.1,
                strength: 0.05,
            },
            super_weight_config: SuperWeightConfig {
                enabled: true,
                probability: 0.001,
                magnitude_factor: 10.0,
            },
        }
    }

    /// Crée un profil par défaut pour DeepSeek-V4-Flash.
    pub fn deepseek_v4_flash_default() -> Self {
        Self {
            name: "deepseek_v4_flash_statistical_profile".to_string(),
            version: "1.0.0".to_string(),
            description: "Profil statistique par défaut pour DeepSeek-V4-Flash".to_string(),
            distributions: ProfileDistributionConfig {
                weight_distribution: WeightDistribution::Normal,
                outlier_distribution: WeightDistribution::StudentT,
                correlation_strength: 0.25,
                low_rank_strength: 0.15,
            },
            outlier_config: OutlierProfileConfig {
                probability: 0.015,
                severity_factor: 4.0,
                layer_variation: true,
            },
            correlation_config: CorrelationConfig {
                enabled: true,
                max_correlation: 0.4,
                layer_decay: 0.85,
            },
            low_rank_config: LowRankConfig {
                enabled: true,
                rank_ratio: 0.12,
                strength: 0.06,
            },
            super_weight_config: SuperWeightConfig {
                enabled: true,
                probability: 0.0015,
                magnitude_factor: 12.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_profile_validation() {
        let profile = StatisticalProfile::glm52_default();
        assert!(profile.validate().is_ok());

        let mut invalid_profile = profile.clone();
        invalid_profile.name = "".to_string();
        assert!(invalid_profile.validate().is_err());
    }

    #[test]
    fn test_default_profiles() {
        let glm52 = StatisticalProfile::glm52_default();
        assert_eq!(glm52.name, "glm52_statistical_profile");
        assert_eq!(
            glm52.distributions.weight_distribution,
            WeightDistribution::Normal
        );
        assert_eq!(
            glm52.distributions.outlier_distribution,
            WeightDistribution::StudentT
        );
        assert_eq!(glm52.distributions.correlation_strength, 0.3);
        assert_eq!(glm52.distributions.low_rank_strength, 0.1);

        let deepseek = StatisticalProfile::deepseek_v4_flash_default();
        assert_eq!(deepseek.name, "deepseek_v4_flash_statistical_profile");
        assert_eq!(
            deepseek.distributions.weight_distribution,
            WeightDistribution::Normal
        );
        assert_eq!(
            deepseek.distributions.outlier_distribution,
            WeightDistribution::StudentT
        );
        assert_eq!(deepseek.distributions.correlation_strength, 0.25);
        assert_eq!(deepseek.distributions.low_rank_strength, 0.15);
    }

    #[test]
    fn test_serialization_round_trip() {
        let profile = StatisticalProfile::glm52_default();
        let json = serde_json::to_string_pretty(&profile).unwrap();
        let deserialized: StatisticalProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile.name, deserialized.name);
        assert_eq!(profile.version, deserialized.version);
        assert_eq!(profile.distributions, deserialized.distributions);
    }
}
