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

//! Score de similarité — calcul du score structurel global.
//!
//! Ce module fournit les types et fonctions pour calculer le score
//! de similarité structurel entre deux modèles, en combinant les
//! résultats des différentes comparaisons.
//!
//! # Responsabilités
//!
//! - Structure `ComparisonScore` pour le score global ;
//! - Calcul du score : S = (N_match / N_total) × 100 ;
//! - Affichage séparé des anomalies bloquantes.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Le score est toujours affiché avec une décimale.

/// Score de similarité structurel.
///
/// # Exemple
///
/// ```
/// use pmg_compare::score::ComparisonScore;
/// use pmg_compare::comparison::ComparisonStatus;
///
/// let score = ComparisonScore::new(95.5, 100, 95, 0);
/// assert_eq!(score.percentage, 95.5);
/// assert_eq!(score.total_elements, 100);
/// assert_eq!(score.matching_elements, 95);
/// assert_eq!(score.blocking_anomalies, 0);
/// ```
#[derive(Debug, Clone)]
pub struct ComparisonScore {
    /// Pourcentage de similarité (0.0 à 100.0).
    pub percentage: f64,
    /// Nombre total d'éléments comparés.
    pub total_elements: usize,
    /// Nombre d'éléments identiques.
    pub matching_elements: usize,
    /// Nombre d'anomalies bloquantes.
    pub blocking_anomalies: usize,
}

impl ComparisonScore {
    /// Crée un nouveau score de similarité.
    pub fn new(
        percentage: f64,
        total_elements: usize,
        matching_elements: usize,
        blocking_anomalies: usize,
    ) -> Self {
        Self {
            percentage,
            total_elements,
            matching_elements,
            blocking_anomalies,
        }
    }

    /// Calcule le score à partir des compteurs.
    pub fn from_counts(
        total_elements: usize,
        matching_elements: usize,
        blocking_anomalies: usize,
    ) -> Self {
        let percentage = if total_elements == 0 {
            100.0
        } else {
            (matching_elements as f64 / total_elements as f64) * 100.0
        };

        Self::new(
            percentage,
            total_elements,
            matching_elements,
            blocking_anomalies,
        )
    }

    /// Vérifie si le score est parfait (100%).
    pub fn is_perfect(&self) -> bool {
        self.percentage >= 100.0 && self.blocking_anomalies == 0
    }

    /// Vérifie si le score est acceptable (>= 80%).
    pub fn is_acceptable(&self) -> bool {
        self.percentage >= 80.0 && self.blocking_anomalies == 0
    }

    /// Vérifie s'il y a des anomalies bloquantes.
    pub fn has_blocking_anomalies(&self) -> bool {
        self.blocking_anomalies > 0
    }
}

impl std::fmt::Display for ComparisonScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}%", self.percentage)?;

        if self.has_blocking_anomalies() {
            write!(f, " (⚠️  {} anomalies bloquantes)", self.blocking_anomalies)?;
        }

        Ok(())
    }
}

/// Calcule le score global à partir des résultats partiels.
///
/// # Entrées
/// - `config_score` : score de la configuration (0.0 à 1.0) ;
/// - `architecture_score` : score de l'architecture (0.0 à 1.0) ;
/// - `tensor_score` : score des tenseurs (0.0 à 1.0) ;
/// - `shape_score` : score des shapes (0.0 à 1.0) ;
/// - `dtype_score` : score des dtypes (0.0 à 1.0) ;
/// - `shard_score` : score du sharding (0.0 à 1.0) ;
/// - `blocking_anomalies` : nombre d'anomalies bloquantes.
///
/// # Sorties
/// Un [`ComparisonScore`] avec le score global.
pub fn calculate_global_score(
    config_score: f64,
    architecture_score: f64,
    tensor_score: f64,
    shape_score: f64,
    dtype_score: f64,
    shard_score: f64,
    blocking_anomalies: usize,
) -> ComparisonScore {
    // Pondération des scores (somme = 1.0)
    let weights = [
        0.20, // config
        0.25, // architecture
        0.20, // tensors
        0.15, // shapes
        0.10, // dtypes
        0.10, // sharding
    ];

    let scores = [
        config_score,
        architecture_score,
        tensor_score,
        shape_score,
        dtype_score,
        shard_score,
    ];

    // Calcul du score pondéré
    let weighted_sum: f64 = scores
        .iter()
        .zip(weights.iter())
        .map(|(score, weight)| score * weight)
        .sum();

    // Conversion en pourcentage
    let _percentage = weighted_sum * 100.0;

    // Nombre total d'éléments (estimé)
    let total_elements = 10; // Nombre de catégories de comparaison
    let matching_elements = (weighted_sum * total_elements as f64) as usize;

    ComparisonScore::from_counts(total_elements, matching_elements, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_score_display() {
        let score = ComparisonScore::new(95.5, 100, 95, 0);
        assert_eq!(score.to_string(), "95.5%");

        let score_with_anomalies = ComparisonScore::new(85.0, 100, 85, 2);
        assert!(score_with_anomalies
            .to_string()
            .contains("anomalies bloquantes"));
    }

    #[test]
    fn comparison_score_from_counts() {
        let score = ComparisonScore::from_counts(100, 90, 1);
        assert_eq!(score.percentage, 90.0);
        assert_eq!(score.total_elements, 100);
        assert_eq!(score.matching_elements, 90);
        assert_eq!(score.blocking_anomalies, 1);
    }

    #[test]
    fn calculate_global_score_perfect() {
        let score = calculate_global_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0);
        assert!(score.is_perfect());
        assert!(!score.has_blocking_anomalies());
    }

    #[test]
    fn calculate_global_score_with_anomalies() {
        let score = calculate_global_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1);
        assert!(!score.is_perfect());
        assert!(score.has_blocking_anomalies());
    }
}
