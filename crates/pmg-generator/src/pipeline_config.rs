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

//! Configuration du pipeline de génération.
//!
//! Ce module définit les paramètres de configuration pour chaque étape
//! du pipeline de génération. Les configurations par défaut sont conçues
//! pour être réalistes et conformes aux spécifications du projet.

/// Configuration de l'étape de distribution.
#[derive(Debug, Clone)]
pub struct DistributionConfig {
    /// Moyenne de la distribution normale (par défaut : 0.0).
    pub mean: f64,
    /// Écart-type de la distribution normale (par défaut : 1.0).
    pub std: f64,
}

impl Default for DistributionConfig {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
        }
    }
}

/// Configuration de l'étape de corrélation.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Matrice de covariance cible (dim × dim, stockée ligne par ligne).
    /// Si None, une corrélation identité (pas de corrélation) est utilisée.
    pub sigma: Option<Vec<f64>>,
    /// Dimension de la matrice de covariance.
    pub dim: usize,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            sigma: None,
            dim: 1,
        }
    }
}

/// Configuration de l'étape de bas-rang.
#[derive(Debug, Clone)]
pub struct LowRankConfig {
    /// Seuil d'énergie pour le rang effectif (par défaut : 0.9).
    pub energy_threshold: f64,
    /// Réduction de rang cible (par défaut : 0.5).
    pub rank_reduction: f64,
}

impl Default for LowRankConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.9,
            rank_reduction: 0.5,
        }
    }
}

/// Configuration de l'étape d'outliers.
#[derive(Debug, Clone)]
pub struct OutliersConfig {
    /// Seuil k pour la détection d'outliers (|x| > kσ, par défaut : 3.0).
    pub threshold_k: f64,
    /// Mode de remplacement (par défaut : clip au seuil).
    pub replacement_mode: OutlierReplacementMode,
}

impl Default for OutliersConfig {
    fn default() -> Self {
        Self {
            threshold_k: 3.0,
            replacement_mode: OutlierReplacementMode::ClipToThreshold,
        }
    }
}

/// Mode de remplacement des outliers.
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierReplacementMode {
    /// Clipper les valeurs au seuil (mean ± kσ).
    ClipToThreshold,
    /// Remplacer par la moyenne.
    ReplaceWithMean,
    /// Supprimer les outliers (réduire la taille du tableau).
    Remove,
}

/// Configuration de l'étape de super-poids.
#[derive(Debug, Clone)]
pub struct SuperWeightsConfig {
    /// Seuil k pour les super-poids (|x| > kσ, par défaut : 5.0).
    pub threshold_k: f64,
    /// Facteur de multiplication pour les super-poids (par défaut : 2.0).
    pub multiplier: f64,
    /// Proportion maximale de super-poids (par défaut : 0.01).
    pub max_proportion: f64,
}

impl Default for SuperWeightsConfig {
    fn default() -> Self {
        Self {
            threshold_k: 5.0,
            multiplier: 2.0,
            max_proportion: 0.01,
        }
    }
}

/// Configuration globale du pipeline de génération.
#[derive(Debug, Clone, Default)]
pub struct PipelineGlobalConfig {
    /// Configuration de l'étape de distribution.
    pub distribution: DistributionConfig,
    /// Configuration de l'étape de corrélation.
    pub correlation: CorrelationConfig,
    /// Configuration de l'étape de bas-rang.
    pub low_rank: LowRankConfig,
    /// Configuration de l'étape d'outliers.
    pub outliers: OutliersConfig,
    /// Configuration de l'étape de super-poids.
    pub super_weights: SuperWeightsConfig,
}
