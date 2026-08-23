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

//! Sous-module contenant les structures de configuration des profils statistiques.
//!
//! Ce module regroupe toutes les sous-configurations utilisées dans un profil statistique :
//! - [`WeightDistribution`] : type de distribution pour les poids
//! - [`ProfileDistributionConfig`] : configuration des distributions statistiques
//! - [`OutlierProfileConfig`] : configuration des outliers
//! - [`CorrelationConfig`] : configuration des corrélations entre paramètres
//! - [`LowRankConfig`] : configuration de la structure à faible rang
//! - [`SuperWeightConfig`] : configuration des super-poids (magnitude élevée)

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// Type de distribution pour les poids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightDistribution {
    /// Distribution normale (gaussienne).
    Normal,
    /// Distribution de Student-t (queues lourdes).
    StudentT,
    /// Distribution de Laplace.
    Laplace,
    /// Distribution log-normale.
    LogNormal,
}

impl WeightDistribution {
    /// Nom lisible en français.
    pub fn display_name(self) -> &'static str {
        match self {
            WeightDistribution::Normal => "normale",
            WeightDistribution::StudentT => "student-t",
            WeightDistribution::Laplace => "laplace",
            WeightDistribution::LogNormal => "log-normale",
        }
    }
}

/// Configuration des distributions statistiques.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileDistributionConfig {
    /// Type de distribution pour les poids.
    pub weight_distribution: WeightDistribution,
    /// Type de distribution pour les outliers.
    pub outlier_distribution: WeightDistribution,
    /// Force des corrélations entre paramètres (0.0 à 1.0).
    pub correlation_strength: f64,
    /// Force de la structure à faible rang (0.0 à 1.0).
    pub low_rank_strength: f64,
}

impl ProfileDistributionConfig {
    /// Valide la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if !(0.0..=1.0).contains(&self.correlation_strength) {
            return Err(CoreError::Validation(format!(
                "correlation_strength hors [0.0, 1.0] : {}",
                self.correlation_strength
            )));
        }
        if !(0.0..=1.0).contains(&self.low_rank_strength) {
            return Err(CoreError::Validation(format!(
                "low_rank_strength hors [0.0, 1.0] : {}",
                self.low_rank_strength
            )));
        }
        Ok(())
    }
}

/// Configuration des outliers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierProfileConfig {
    /// Probabilité d'apparition d'un outlier (0.0 à 1.0).
    pub probability: f64,
    /// Facteur de sévérité (multiplicateur de l'amplitude).
    pub severity_factor: f64,
    /// Variation de la probabilité entre les couches.
    pub layer_variation: bool,
}

impl OutlierProfileConfig {
    /// Valide la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(CoreError::Validation(format!(
                "probability hors [0.0, 1.0] : {}",
                self.probability
            )));
        }
        if self.severity_factor <= 0.0 {
            return Err(CoreError::Validation(format!(
                "severity_factor doit être > 0 : {}",
                self.severity_factor
            )));
        }
        Ok(())
    }
}

/// Configuration des corrélations entre paramètres.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Activation des corrélations.
    pub enabled: bool,
    /// Corrélation maximale entre paramètres (0.0 à 1.0).
    pub max_correlation: f64,
    /// Taux de décroissance de la corrélation entre les couches (0.0 à 1.0).
    pub layer_decay: f64,
}

impl CorrelationConfig {
    /// Valide la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if self.enabled {
            if !(0.0..=1.0).contains(&self.max_correlation) {
                return Err(CoreError::Validation(format!(
                    "max_correlation hors [0.0, 1.0] : {}",
                    self.max_correlation
                )));
            }
            if !(0.0..=1.0).contains(&self.layer_decay) {
                return Err(CoreError::Validation(format!(
                    "layer_decay hors [0.0, 1.0] : {}",
                    self.layer_decay
                )));
            }
        }
        Ok(())
    }
}

/// Configuration de la structure à faible rang.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankConfig {
    /// Activation de la structure à faible rang.
    pub enabled: bool,
    /// Ratio du rang par rapport à la dimension complète (0.0 à 1.0).
    pub rank_ratio: f64,
    /// Force de la structure à faible rang (0.0 à 1.0).
    pub strength: f64,
}

impl LowRankConfig {
    /// Valide la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if self.enabled {
            if !(0.0..=1.0).contains(&self.rank_ratio) {
                return Err(CoreError::Validation(format!(
                    "rank_ratio hors [0.0, 1.0] : {}",
                    self.rank_ratio
                )));
            }
            if !(0.0..=1.0).contains(&self.strength) {
                return Err(CoreError::Validation(format!(
                    "strength hors [0.0, 1.0] : {}",
                    self.strength
                )));
            }
        }
        Ok(())
    }
}

/// Configuration des super-poids (magnitude élevée).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperWeightConfig {
    /// Activation des super-poids.
    pub enabled: bool,
    /// Probabilité d'apparition d'un super-poids (0.0 à 1.0).
    pub probability: f64,
    /// Facteur de magnitude (multiplicateur de la valeur absolue).
    pub magnitude_factor: f64,
}

impl SuperWeightConfig {
    /// Valide la configuration.
    pub fn validate(&self) -> CoreResult<()> {
        if self.enabled {
            if !(0.0..=1.0).contains(&self.probability) {
                return Err(CoreError::Validation(format!(
                    "probability hors [0.0, 1.0] : {}",
                    self.probability
                )));
            }
            if self.magnitude_factor <= 0.0 {
                return Err(CoreError::Validation(format!(
                    "magnitude_factor doit être > 0 : {}",
                    self.magnitude_factor
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_distribution_display() {
        assert_eq!(WeightDistribution::Normal.display_name(), "normale");
        assert_eq!(WeightDistribution::StudentT.display_name(), "student-t");
        assert_eq!(WeightDistribution::Laplace.display_name(), "laplace");
        assert_eq!(WeightDistribution::LogNormal.display_name(), "log-normale");
    }

    #[test]
    fn test_distribution_config_validation() {
        let valid_config = ProfileDistributionConfig {
            weight_distribution: WeightDistribution::Normal,
            outlier_distribution: WeightDistribution::StudentT,
            correlation_strength: 0.5,
            low_rank_strength: 0.2,
        };
        assert!(valid_config.validate().is_ok());

        let invalid_config = ProfileDistributionConfig {
            weight_distribution: WeightDistribution::Normal,
            outlier_distribution: WeightDistribution::StudentT,
            correlation_strength: 1.5,
            low_rank_strength: 0.2,
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_outlier_config_validation() {
        let valid_config = OutlierProfileConfig {
            probability: 0.01,
            severity_factor: 3.0,
            layer_variation: true,
        };
        assert!(valid_config.validate().is_ok());

        let invalid_config = OutlierProfileConfig {
            probability: 1.5,
            severity_factor: 3.0,
            layer_variation: true,
        };
        assert!(invalid_config.validate().is_err());
    }
}
