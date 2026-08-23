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

//! Métadonnées d'injection d'outliers pour la validation et la traçabilité.
//!
//! Ce module définit les informations qui accompagnent chaque injection de
//! super-poids, permettant de valider les propriétés statistiques et de
//! garantir la reproductibilité.
//!
//! # Conformité
//!
//! Spécification Sprint 9, étape 5.4 : « Métadonnées d'anomalies —
//! informations pour la validation ».

use serde::{Deserialize, Serialize};

/// Stratégie d'injection utilisée (pour la traçabilité).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutlierStrategyKind {
    /// Stratégie additive (W' = W + O).
    Additive,
    /// Stratégie multiplicative (W' = W ⊙ M).
    Multiplicative,
}

/// Métadonnées d'injection d'outliers pour un tenseur donné.
///
/// Ces informations sont stockées avec le tenseur injecté et permettent :
/// - la validation du nombre réel d'outliers ;
/// - le contrôle de l'amplitude maximale ;
/// - la traçabilité de la stratégie et de la seed utilisée.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlierMetadata {
    /// Nombre total d'éléments dans le tenseur.
    pub total_elements: usize,
    /// Nombre d'éléments marqués comme outliers.
    pub outlier_count: usize,
    /// Fraction réelle d'outliers = outlier_count / total_elements.
    pub fraction: f64,
    /// Valeur absolue maximale parmi les outliers injectés.
    pub max_abs: f64,
    /// Stratégie utilisée pour l'injection.
    pub strategy: OutlierStrategyKind,
    /// Seed globale utilisée pour la génération déterministe.
    pub seed: u64,
    /// Version du générateur (pour compatibilité ascendante).
    pub generation_version: String,
}

impl OutlierMetadata {
    /// Crée les métadonnées à partir des informations brutes.
    ///
    /// # Arguments
    /// - `total_elements` : taille totale du tenseur ;
    /// - `outlier_count` : nombre d'outliers effectivement injectés ;
    /// - `max_abs` : valeur absolue maximale observée parmi les outliers ;
    /// - `strategy` : stratégie d'injection utilisée ;
    /// - `seed` : seed globale ;
    /// - `generation_version` : version du générateur.
    ///
    /// # Panique
    /// Ne panique pas, toutes les opérations sont sur `f64` et gèrent l'inf.
    pub fn new(
        total_elements: usize,
        outlier_count: usize,
        max_abs: f64,
        strategy: OutlierStrategyKind,
        seed: u64,
        generation_version: String,
    ) -> Self {
        let fraction = if total_elements == 0 {
            0.0
        } else {
            outlier_count as f64 / total_elements as f64
        };
        Self {
            total_elements,
            outlier_count,
            fraction,
            max_abs,
            strategy,
            seed,
            generation_version,
        }
    }

    /// Valide la cohérence des métadonnées.
    ///
    /// # Erreurs
    /// Retourne une erreur si les métadonnées sont incohérentes.
    pub fn validate(&self) -> Result<(), String> {
        if self.outlier_count > self.total_elements {
            return Err(format!(
                "outlier_count ({}) > total_elements ({})",
                self.outlier_count, self.total_elements
            ));
        }
        if self.total_elements == 0 && self.outlier_count > 0 {
            return Err("total_elements = 0 mais outlier_count > 0".into());
        }
        let expected_fraction = if self.total_elements == 0 {
            0.0
        } else {
            self.outlier_count as f64 / self.total_elements as f64
        };
        if (self.fraction - expected_fraction).abs() > 1e-10 {
            return Err(format!(
                "fraction ({}) ne correspond pas au ratio attendu ({})",
                self.fraction, expected_fraction
            ));
        }
        if !self.max_abs.is_finite() || self.max_abs < 0.0 {
            return Err(format!(
                "max_abs doit être fini et ≥ 0, reçu {}",
                self.max_abs
            ));
        }
        Ok(())
    }

    /// Sérialise les métadonnées en JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Désérialise les métadonnées depuis du JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let meta = OutlierMetadata::new(
            1000,
            50,
            10.5,
            OutlierStrategyKind::Multiplicative,
            42,
            "1.0.0".into(),
        );
        assert_eq!(meta.total_elements, 1000);
        assert_eq!(meta.outlier_count, 50);
        assert!((meta.fraction - 0.05).abs() < 1e-10);
        assert_eq!(meta.max_abs, 10.5);
        assert_eq!(meta.strategy, OutlierStrategyKind::Multiplicative);
        assert_eq!(meta.seed, 42);
        assert_eq!(meta.generation_version, "1.0.0");
    }

    #[test]
    fn test_metadata_validation() {
        let valid = OutlierMetadata::new(
            100,
            10,
            5.0,
            OutlierStrategyKind::Additive,
            123,
            "2.0.0".into(),
        );
        assert!(valid.validate().is_ok());

        let invalid = OutlierMetadata::new(
            100,
            150, // plus que total
            5.0,
            OutlierStrategyKind::Additive,
            123,
            "2.0.0".into(),
        );
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_json_roundtrip() {
        let meta = OutlierMetadata::new(
            200,
            20,
            #[allow(clippy::approx_constant)]
            3.14,
            OutlierStrategyKind::Additive,
            999,
            "3.0.0".into(),
        );
        let json = meta.to_json().unwrap();
        let restored = OutlierMetadata::from_json(&json).unwrap();
        assert_eq!(meta, restored);
    }
}
