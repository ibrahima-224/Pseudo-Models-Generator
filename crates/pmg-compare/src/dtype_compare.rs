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

//! Comparaison de types de données — comparaison isolée des dtypes.
//!
//! Ce module fournit des fonctions pour comparer les types de données
//! (dtypes) des tenseurs entre deux modèles.
//!
//! # Responsabilités
//!
//! - Comparaison des dtypes tenseur par tenseur ;
//! - Détection des différences de types ;
//! - Calcul d'un score de similarité pour les dtypes.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les comparaisons sont déterministes.

use crate::comparison::ComparisonStatus;
use crate::diff::Diff;

/// Représente le type de données d'un tenseur pour la comparaison.
#[derive(Debug, Clone)]
pub struct DtypeInfo {
    /// Nom du tenseur.
    pub tensor_name: String,
    /// Type de données (dtype).
    pub dtype: String,
}

impl DtypeInfo {
    /// Crée une nouvelle information de dtype.
    pub fn new(tensor_name: String, dtype: String) -> Self {
        Self { tensor_name, dtype }
    }
}

/// Résultat de la comparaison de deux ensembles de dtypes.
#[derive(Debug, Clone)]
pub struct DtypeComparisonResult {
    /// Score de similarité (0.0 à 1.0).
    pub similarity_score: f64,
    /// Statut de la comparaison.
    pub status: ComparisonStatus,
    /// Différences détectées.
    pub differences: Vec<Diff>,
    /// Nombre total de dtypes comparés.
    pub total_dtypes: usize,
    /// Nombre de dtypes identiques.
    pub matching_dtypes: usize,
}

impl Default for DtypeComparisonResult {
    fn default() -> Self {
        Self {
            similarity_score: 1.0,
            status: ComparisonStatus::Match,
            differences: Vec::new(),
            total_dtypes: 0,
            matching_dtypes: 0,
        }
    }
}

impl DtypeComparisonResult {
    /// Crée un nouveau résultat de comparaison.
    pub fn new(
        similarity_score: f64,
        status: ComparisonStatus,
        differences: Vec<Diff>,
        total_dtypes: usize,
        matching_dtypes: usize,
    ) -> Self {
        Self {
            similarity_score,
            status,
            differences,
            total_dtypes,
            matching_dtypes,
        }
    }
}

/// Compare deux ensembles de dtypes.
///
/// # Entrées
/// - `original` : dtypes originaux ;
/// - `compared` : dtypes à comparer.
///
/// # Sorties
/// Un [`DtypeComparisonResult`] contenant les résultats de la comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::dtype_compare::{compare_dtypes, DtypeInfo};
///
/// let dtypes1 = vec![
///     DtypeInfo::new("layer1.weight".to_string(), "f32".to_string()),
///     DtypeInfo::new("layer1.bias".to_string(), "f32".to_string()),
/// ];
///
/// let dtypes2 = vec![
///     DtypeInfo::new("layer1.weight".to_string(), "f32".to_string()),
///     DtypeInfo::new("layer1.bias".to_string(), "f32".to_string()),
/// ];
///
/// let result = compare_dtypes(&dtypes1, &dtypes2);
/// assert_eq!(result.similarity_score, 1.0);
/// assert!(result.differences.is_empty());
/// ```
pub fn compare_dtypes(original: &[DtypeInfo], compared: &[DtypeInfo]) -> DtypeComparisonResult {
    let mut differences = Vec::new();
    let mut matching_count = 0;
    let mut total_count = 0;

    // Indexer les dtypes par nom de tenseur
    let original_map: std::collections::HashMap<&str, &DtypeInfo> = original
        .iter()
        .map(|d| (d.tensor_name.as_str(), d))
        .collect();
    let compared_map: std::collections::HashMap<&str, &DtypeInfo> = compared
        .iter()
        .map(|d| (d.tensor_name.as_str(), d))
        .collect();

    // Comparer les dtypes des tenseurs communs
    for (name, original_dtype) in &original_map {
        if let Some(compared_dtype) = compared_map.get(name) {
            total_count += 1;
            if original_dtype.dtype == compared_dtype.dtype {
                matching_count += 1;
            } else {
                differences.push(Diff::modified(
                    name.to_string(),
                    original_dtype.dtype.clone(),
                    compared_dtype.dtype.clone(),
                    format!("dtype différent pour le tenseur {}", name),
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

    DtypeComparisonResult::new(
        similarity_score,
        status,
        differences,
        total_count,
        matching_count,
    )
}

/// Compare les dtypes avec détection d'anomalies bloquantes.
///
/// # Entrées
/// - `original` : dtypes originaux ;
/// - `compared` : dtypes à comparer.
///
/// # Sorties
/// Un tuple (résultat, anomalies bloquantes).
pub fn compare_dtypes_with_anomalies(
    original: &[DtypeInfo],
    compared: &[DtypeInfo],
) -> (DtypeComparisonResult, Vec<String>) {
    let result = compare_dtypes(original, compared);
    let mut blocking_anomalies = Vec::new();

    // Vérifier les anomalies bloquantes
    // Un changement de dtype peut être bloquant s'il est critique
    for diff in &result.differences {
        // Les changements de dtype sont généralement bloquants
        blocking_anomalies.push(format!(
            "dtype incompatible: {} - {}",
            diff.path, diff.description
        ));
    }

    (result, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_dtypes_identical() {
        let dtypes = vec![
            DtypeInfo::new("layer1.weight".to_string(), "float32".to_string()),
            DtypeInfo::new("layer1.bias".to_string(), "float32".to_string()),
        ];
        let result = compare_dtypes(&dtypes, &dtypes);
        assert_eq!(result.similarity_score, 1.0);
        assert!(result.differences.is_empty());
        assert_eq!(result.status, ComparisonStatus::Match);
    }

    #[test]
    fn compare_dtypes_different() {
        let original = vec![
            DtypeInfo::new("layer1.weight".to_string(), "float32".to_string()),
            DtypeInfo::new("layer1.bias".to_string(), "float32".to_string()),
        ];
        let compared = vec![
            DtypeInfo::new("layer1.weight".to_string(), "float16".to_string()),
            DtypeInfo::new("layer1.bias".to_string(), "float16".to_string()),
        ];
        let result = compare_dtypes(&original, &compared);
        assert_eq!(result.similarity_score, 0.0);
        assert_eq!(result.differences.len(), 2);
        assert_eq!(result.status, ComparisonStatus::Different);
    }

    #[test]
    fn compare_dtypes_partial() {
        let original = vec![
            DtypeInfo::new("layer1.weight".to_string(), "float32".to_string()),
            DtypeInfo::new("layer1.bias".to_string(), "float32".to_string()),
        ];
        let compared = vec![
            DtypeInfo::new("layer1.weight".to_string(), "float32".to_string()),
            DtypeInfo::new("layer1.bias".to_string(), "float16".to_string()),
        ];
        let result = compare_dtypes(&original, &compared);
        assert_eq!(result.similarity_score, 0.5);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.status, ComparisonStatus::Different);
    }
}
