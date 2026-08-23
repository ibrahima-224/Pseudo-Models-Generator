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

//! Politique d'injection d'outliers par couche.
//!
//! Ce module permet de définir des règles différentes selon l'indice de couche
//! d'un tenseur, conformément à la spécification Sprint 9, étape 5.3 :
//! « Super-poids par couche — politique différente selon la couche ».
//!
//! # Concepts
//!
//! - **Plage de couches** : intervalle `[start, end]` d'indices de couches (0-based, inclus) ;
//! - **Configuration par couche** : paramètres d'injection (probabilité, stratégie d'amplitude) ;
//! - **Politique** : ensemble ordonné de règles appliquées en priorité (première correspondance).
//!
//! # Exemple
//!
//! ```ignore
//! let mut policy = LayerPolicy::new();
//! policy.add_rule(0, 10, LayerOutlierConfig::low());
//! policy.add_rule(11, 20, LayerOutlierConfig::medium());
//! policy.add_rule(21, 30, LayerOutlierConfig::high());
//!
//! // Pour l'indice de couche 15 → configuration moyenne.
//! let config = policy.config_for_layer(15).unwrap();
//! ```

use super::amplitude::AmplitudeStrategy;
use crate::error::{MathError, MathResult};

/// Configuration d'injection pour une plage de couches.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerOutlierConfig {
    /// Probabilité cible d'outliers dans `[0, 1]`.
    pub probability: f64,
    /// Stratégie de calcul de l'amplitude.
    pub amplitude_strategy: AmplitudeStrategy,
}

impl LayerOutlierConfig {
    /// Configuration « faible » : faible probabilité, amplitude faible.
    pub fn low() -> Self {
        Self {
            probability: 0.01, // 1%
            amplitude_strategy: AmplitudeStrategy::Fixed(1.0),
        }
    }

    /// Configuration « moyenne » : probabilité modérée, amplitude modérée.
    pub fn medium() -> Self {
        Self {
            probability: 0.05, // 5%
            amplitude_strategy: AmplitudeStrategy::RelativeToStd { k: 2.0 },
        }
    }

    /// Configuration « élevée » : probabilité élevée, amplitude élevée.
    pub fn high() -> Self {
        Self {
            probability: 0.10, // 10%
            amplitude_strategy: AmplitudeStrategy::HeavyTail { df: 3.0 },
        }
    }

    /// Valide la configuration.
    pub fn validate(&self) -> MathResult<()> {
        if !self.probability.is_finite() || self.probability < 0.0 || self.probability > 1.0 {
            return Err(MathError::InvalidParameter(format!(
                "probabilité doit être dans [0, 1], reçue {}",
                self.probability
            )));
        }
        self.amplitude_strategy.validate()
    }
}

/// Règle liant une plage de couches à une configuration.
#[derive(Debug, Clone, PartialEq)]
struct LayerRule {
    start: usize,
    end: usize,
    config: LayerOutlierConfig,
}

/// Politique d'injection d'outliers par couche.
///
/// Les règles sont évaluées dans l'ordre d'ajout ; la première correspondance
/// est retournée. Si aucune règle ne correspond, la politique par défaut
/// (pas d'outlier) est appliquée.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerPolicy {
    rules: Vec<LayerRule>,
}

impl LayerPolicy {
    /// Crée une politique vide (aucune règle).
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Ajoute une règle pour la plage `[start, end]` (inclus).
    ///
    /// # Erreurs
    /// - [`MathError::InvalidParameter`] si `start > end` ;
    /// - [`MathError::InvalidParameter`] si la configuration est invalide.
    pub fn add_rule(
        &mut self,
        start: usize,
        end: usize,
        config: LayerOutlierConfig,
    ) -> MathResult<()> {
        if start > end {
            return Err(MathError::InvalidParameter(format!(
                "start ({start}) doit être ≤ end ({end})"
            )));
        }
        config.validate()?;
        self.rules.push(LayerRule { start, end, config });
        Ok(())
    }

    /// Retourne la configuration pour un indice de couche donné.
    ///
    /// Si aucune règle ne correspond, retourne `None` (pas d'outlier).
    pub fn config_for_layer(&self, layer_idx: usize) -> Option<&LayerOutlierConfig> {
        self.rules
            .iter()
            .find(|rule| layer_idx >= rule.start && layer_idx <= rule.end)
            .map(|rule| &rule.config)
    }

    /// Vérifie si la politique contient des règles.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Nombre de règles dans la politique.
    pub fn len(&self) -> usize {
        self.rules.len()
    }
}

impl Default for LayerPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Fonction utilitaire : retourne la configuration d'outlier pour une couche
/// selon une politique donnée.
///
/// # Arguments
/// - `policy` : politique à appliquer ;
/// - `layer_idx` : indice de la couche (0-based).
///
/// # Retour
/// `Some(config)` si une règle correspond, `None` sinon.
pub fn layer_outlier_config(policy: &LayerPolicy, layer_idx: usize) -> Option<&LayerOutlierConfig> {
    policy.config_for_layer(layer_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_policy() {
        let policy = LayerPolicy::new();
        assert!(policy.is_empty());
        assert!(policy.config_for_layer(0).is_none());
    }

    #[test]
    fn test_add_rule() {
        let mut policy = LayerPolicy::new();
        let config = LayerOutlierConfig::low();
        assert!(policy.add_rule(0, 10, config).is_ok());
        assert_eq!(policy.len(), 1);
    }

    #[test]
    fn test_invalid_rule() {
        let mut policy = LayerPolicy::new();
        let config = LayerOutlierConfig::low();
        assert!(policy.add_rule(10, 5, config).is_err());
    }

    #[test]
    fn test_config_for_layer() {
        let mut policy = LayerPolicy::new();
        let low = LayerOutlierConfig::low();
        let medium = LayerOutlierConfig::medium();
        let high = LayerOutlierConfig::high();
        policy.add_rule(0, 10, low).unwrap();
        policy.add_rule(11, 20, medium).unwrap();
        policy.add_rule(21, 30, high).unwrap();

        assert_eq!(policy.config_for_layer(5).unwrap().probability, 0.01);
        assert_eq!(policy.config_for_layer(15).unwrap().probability, 0.05);
        assert_eq!(policy.config_for_layer(25).unwrap().probability, 0.10);
        assert!(policy.config_for_layer(31).is_none());
    }

    #[test]
    fn test_layer_outlier_config_function() {
        let mut policy = LayerPolicy::new();
        let config = LayerOutlierConfig::high();
        policy.add_rule(0, 100, config).unwrap();
        let result = layer_outlier_config(&policy, 50);
        assert!(result.is_some());
        assert_eq!(result.unwrap().probability, 0.10);
    }
}
