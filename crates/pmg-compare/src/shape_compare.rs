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

//! Comparaison de shapes — comparaison isolée des dimensions.
//!
//! Ce module fournit des fonctions pour comparer les shapes
//! (dimensions) des tenseurs entre deux modèles.
//!
//! # Responsabilités
//!
//! - Comparaison des shapes tenseur par tenseur ;
//! - Détection des différences de dimensions ;
//! - Calcul d'un score de similarité pour les shapes.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les comparaisons sont déterministes.

use crate::comparison::ComparisonStatus;
use crate::diff::Diff;

/// Représente la shape d'un tenseur pour la comparaison.
#[derive(Debug, Clone)]
pub struct ShapeInfo {
    /// Nom du tenseur.
    pub tensor_name: String,
    /// Dimensions du tenseur.
    pub dimensions: Vec<usize>,
}

impl ShapeInfo {
    /// Crée une nouvelle information de shape.
    pub fn new(tensor_name: String, dimensions: Vec<usize>) -> Self {
        Self {
            tensor_name,
            dimensions,
        }
    }
}

/// Résultat de la comparaison de deux ensembles de shapes.
#[derive(Debug, Clone)]
pub struct ShapeComparisonResult {
    /// Score de similarité (0.0 à 1.0).
    pub similarity_score: f64,
    /// Statut de la comparaison.
    pub status: ComparisonStatus,
    /// Différences détectées.
    pub differences: Vec<Diff>,
    /// Nombre total de shapes comparées.
    pub total_shapes: usize,
    /// Nombre de shapes identiques.
    pub matching_shapes: usize,
}

impl Default for ShapeComparisonResult {
    fn default() -> Self {
        Self {
            similarity_score: 1.0,
            status: ComparisonStatus::Match,
            differences: Vec::new(),
            total_shapes: 0,
            matching_shapes: 0,
        }
    }
}

impl ShapeComparisonResult {
    /// Crée un nouveau résultat de comparaison.
    pub fn new(
        similarity_score: f64,
        status: ComparisonStatus,
        differences: Vec<Diff>,
        total_shapes: usize,
        matching_shapes: usize,
    ) -> Self {
        Self {
            similarity_score,
            status,
            differences,
            total_shapes,
            matching_shapes,
        }
    }
}

/// Compare deux ensembles de shapes.
///
/// # Entrées
/// - `original` : shapes originales ;
/// - `compared` : shapes à comparer.
///
/// # Sorties
/// Un [`ShapeComparisonResult`] contenant les résultats de la comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::shape_compare::{compare_shapes, ShapeInfo};
///
/// let shapes1 = vec![
///     ShapeInfo::new("layer1.weight".to_string(), vec![4096, 4096]),
///     ShapeInfo::new("layer1.bias".to_string(), vec![4096]),
/// ];
///
/// let shapes2 = vec![
///     ShapeInfo::new("layer1.weight".to_string(), vec![4096, 4096]),
///     ShapeInfo::new("layer1.bias".to_string(), vec![4096]),
/// ];
///
/// let result = compare_shapes(&shapes1, &shapes2);
/// assert_eq!(result.similarity_score, 1.0);
/// assert!(result.differences.is_empty());
/// ```
pub fn compare_shapes(original: &[ShapeInfo], compared: &[ShapeInfo]) -> ShapeComparisonResult {
    let mut differences = Vec::new();
    let mut matching_count = 0;
    let mut total_count = 0;

    // Indexer les shapes par nom de tenseur
    let original_map: std::collections::HashMap<&str, &ShapeInfo> = original
        .iter()
        .map(|s| (s.tensor_name.as_str(), s))
        .collect();
    let compared_map: std::collections::HashMap<&str, &ShapeInfo> = compared
        .iter()
        .map(|s| (s.tensor_name.as_str(), s))
        .collect();

    // Comparer les shapes des tenseurs communs
    for (name, original_shape) in &original_map {
        if let Some(compared_shape) = compared_map.get(name) {
            total_count += 1;
            if original_shape.dimensions == compared_shape.dimensions {
                matching_count += 1;
            } else {
                differences.push(Diff::modified(
                    name.to_string(),
                    format!("{:?}", original_shape.dimensions),
                    format!("{:?}", compared_shape.dimensions),
                    format!("Shape différente pour le tenseur {}", name),
                ));
            }
        }
    }

    // Calcul du score de similarité
    let similarity_score = if total_count == 0 {
        1.0
    } else {
        matching_count as f64 / total_count as f64
    };

    // Déterminer le statut
    let status = if differences.is_empty() {
        ComparisonStatus::Match
    } else if similarity_score >= 0.8 {
        ComparisonStatus::Partial
    } else {
        ComparisonStatus::Different
    };

    ShapeComparisonResult::new(
        similarity_score,
        status,
        differences,
        total_count,
        matching_count,
    )
}

/// Compare les shapes avec détection d'anomalies bloquantes.
///
/// # Entrées
/// - `original` : shapes originales ;
/// - `compared` : shapes à comparer.
///
/// # Sorties
/// Un tuple (résultat, anomalies bloquantes).
pub fn compare_shapes_with_anomalies(
    original: &[ShapeInfo],
    compared: &[ShapeInfo],
) -> (ShapeComparisonResult, Vec<String>) {
    let result = compare_shapes(original, compared);
    let mut blocking_anomalies = Vec::new();

    // Vérifier les anomalies bloquantes
    for diff in &result.differences {
        // Toute différence de shape est potentiellement bloquante
        blocking_anomalies.push(format!(
            "Shape incompatible: {} - {}",
            diff.path, diff.description
        ));
    }

    (result, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_shapes_identical() {
        let shapes = vec![
            ShapeInfo::new("layer1.weight".to_string(), vec![100, 200]),
            ShapeInfo::new("layer1.bias".to_string(), vec![200]),
        ];
        let result = compare_shapes(&shapes, &shapes);
        assert_eq!(result.similarity_score, 1.0);
        assert!(result.differences.is_empty());
        assert_eq!(result.status, ComparisonStatus::Match);
    }

    #[test]
    fn compare_shapes_different() {
        let original = vec![
            ShapeInfo::new("layer1.weight".to_string(), vec![100, 200]),
            ShapeInfo::new("layer1.bias".to_string(), vec![200]),
        ];
        let compared = vec![
            ShapeInfo::new("layer1.weight".to_string(), vec![100, 300]),
            ShapeInfo::new("layer1.bias".to_string(), vec![300]),
        ];
        let result = compare_shapes(&original, &compared);
        assert_eq!(result.similarity_score, 0.0);
        assert_eq!(result.differences.len(), 2);
        assert_eq!(result.status, ComparisonStatus::Different);
    }

    #[test]
    fn compare_shapes_partial() {
        let original = vec![
            ShapeInfo::new("layer1.weight".to_string(), vec![100, 200]),
            ShapeInfo::new("layer1.bias".to_string(), vec![200]),
        ];
        let compared = vec![
            ShapeInfo::new("layer1.weight".to_string(), vec![100, 200]),
            ShapeInfo::new("layer1.bias".to_string(), vec![300]),
        ];
        let result = compare_shapes(&original, &compared);
        assert_eq!(result.similarity_score, 0.5);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.status, ComparisonStatus::Different);
    }
}
