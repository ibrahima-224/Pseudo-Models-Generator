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

//! Comparaison de tenseurs — comparaison des noms et présences.
//!
//! Ce module fournit des fonctions pour comparer les tenseurs
//! entre deux modèles, en se concentrant sur les noms et la présence
//! des tenseurs (sans lecture des poids).
//!
//! # Responsabilités
//!
//! - Comparaison des noms de tenseurs ;
//! - Détection des tenseurs manquants ou supplémentaires ;
//! - Calcul d'un score de similarité pour les tenseurs.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les comparaisons sont déterministes.

use crate::comparison::ComparisonStatus;
use crate::diff::{Diff, DiffType};

/// Représente un tenseur pour la comparaison.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Nom ou chemin du tenseur.
    pub name: String,
    /// Shape du tenseur (optionnel, pour référence).
    pub shape: Option<Vec<usize>>,
    /// Type de données (optionnel, pour référence).
    pub dtype: Option<String>,
}

impl TensorInfo {
    /// Crée un nouveau tenseur avec uniquement le nom.
    pub fn new(name: String) -> Self {
        Self {
            name,
            shape: None,
            dtype: None,
        }
    }

    /// Crée un tenseur avec toutes les informations.
    pub fn with_details(name: String, shape: Vec<usize>, dtype: String) -> Self {
        Self {
            name,
            shape: Some(shape),
            dtype: Some(dtype),
        }
    }
}

/// Résultat de la comparaison de deux ensembles de tenseurs.
#[derive(Debug, Clone)]
pub struct TensorComparisonResult {
    /// Score de similarité (0.0 à 1.0).
    pub similarity_score: f64,
    /// Statut de la comparaison.
    pub status: ComparisonStatus,
    /// Différences détectées.
    pub differences: Vec<Diff>,
    /// Nombre total de tenseurs uniques (union).
    pub total_tensors: usize,
    /// Nombre de tenseurs communs.
    pub common_tensors: usize,
    /// Nombre de tenseurs uniquement dans l'original.
    pub original_only: usize,
    /// Nombre de tenseurs uniquement dans la comparaison.
    pub compared_only: usize,
}

impl Default for TensorComparisonResult {
    fn default() -> Self {
        Self {
            similarity_score: 1.0,
            status: ComparisonStatus::Match,
            differences: Vec::new(),
            total_tensors: 0,
            common_tensors: 0,
            original_only: 0,
            compared_only: 0,
        }
    }
}

impl TensorComparisonResult {
    /// Crée un nouveau résultat de comparaison.
    pub fn new(
        similarity_score: f64,
        status: ComparisonStatus,
        differences: Vec<Diff>,
        total_tensors: usize,
        common_tensors: usize,
        original_only: usize,
        compared_only: usize,
    ) -> Self {
        Self {
            similarity_score,
            status,
            differences,
            total_tensors,
            common_tensors,
            original_only,
            compared_only,
        }
    }

    /// Vérifie si tous les tenseurs sont communs.
    pub fn all_common(&self) -> bool {
        self.original_only == 0 && self.compared_only == 0
    }
}

/// Compare deux ensembles de tenseurs.
///
/// # Entrées
/// - `original` : tenseurs originaux ;
/// - `compared` : tenseurs à comparer.
///
/// # Sorties
/// Un [`TensorComparisonResult`] contenant les résultats de la comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::tensor_compare::{compare_tensors, TensorInfo};
///
/// let tensors1 = vec![
///     TensorInfo::new("layer1.weight".to_string()),
///     TensorInfo::new("layer1.bias".to_string()),
/// ];
///
/// let tensors2 = vec![
///     TensorInfo::new("layer1.weight".to_string()),
///     TensorInfo::new("layer1.bias".to_string()),
/// ];
///
/// let result = compare_tensors(&tensors1, &tensors2);
/// assert_eq!(result.similarity_score, 1.0);
/// assert!(result.differences.is_empty());
/// ```
pub fn compare_tensors(original: &[TensorInfo], compared: &[TensorInfo]) -> TensorComparisonResult {
    let mut differences = Vec::new();
    let mut common_count = 0;
    let mut original_only_count = 0;
    let mut compared_only_count = 0;

    // Indexer les tenseurs par nom
    let original_map: std::collections::HashMap<&str, &TensorInfo> =
        original.iter().map(|t| (t.name.as_str(), t)).collect();
    let compared_map: std::collections::HashMap<&str, &TensorInfo> =
        compared.iter().map(|t| (t.name.as_str(), t)).collect();

    // Trouver les tenseurs communs et originaux uniquement
    for name in original_map.keys() {
        if compared_map.contains_key(name) {
            common_count += 1;
        } else {
            original_only_count += 1;
            differences.push(Diff::removed(
                name.to_string(),
                "présent".to_string(),
                format!("Tenseur {} uniquement dans l'original", name),
            ));
        }
    }

    // Trouver les tenseurs uniquement dans la comparaison
    for name in compared_map.keys() {
        if !original_map.contains_key(name) {
            compared_only_count += 1;
            differences.push(Diff::added(
                name.to_string(),
                "présent".to_string(),
                format!("Tenseur {} uniquement dans la comparaison", name),
            ));
        }
    }

    // Calcul du score de similarité (Jaccard-like)
    let total_unique = original_only_count + common_count + compared_only_count;
    let similarity_score = if total_unique == 0 {
        // Aucun tenseur à comparer : score de 0.0 et statut inconnu
        0.0
    } else {
        common_count as f64 / total_unique as f64
    };

    // Déterminer le statut
    let status = if total_unique == 0 {
        // Aucun tenseur : comparaison impossible
        ComparisonStatus::Unknown
    } else if differences.is_empty() {
        ComparisonStatus::Match
    } else if similarity_score >= 0.8 {
        ComparisonStatus::Partial
    } else {
        ComparisonStatus::Different
    };

    TensorComparisonResult::new(
        similarity_score,
        status,
        differences,
        total_unique,
        common_count,
        original_only_count,
        compared_only_count,
    )
}

/// Compare les tenseurs avec détection d'anomalies bloquantes.
///
/// # Entrées
/// - `original` : tenseurs originaux ;
/// - `compared` : tenseurs à comparer.
///
/// # Sorties
/// Un tuple (résultat, anomalies bloquantes).
pub fn compare_tensors_with_anomalies(
    original: &[TensorInfo],
    compared: &[TensorInfo],
) -> (TensorComparisonResult, Vec<String>) {
    let result = compare_tensors(original, compared);
    let mut blocking_anomalies = Vec::new();

    // Vérifier les anomalies bloquantes
    // Un tenseur manquant peut être bloquant s'il est critique
    for diff in &result.differences {
        if diff.diff_type == DiffType::Removed {
            // Certains tenseurs critiques sont bloquants
            if diff.path.contains("weight") || diff.path.contains("bias") {
                blocking_anomalies.push(format!("Tenseur critique manquant: {}", diff.path));
            }
        }
    }

    (result, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_tensors_identical() {
        let tensors = vec![
            TensorInfo::new("layer1.weight".to_string()),
            TensorInfo::new("layer1.bias".to_string()),
        ];
        let result = compare_tensors(&tensors, &tensors);
        assert_eq!(result.similarity_score, 1.0);
        assert!(result.differences.is_empty());
        assert_eq!(result.status, ComparisonStatus::Match);
    }

    #[test]
    fn compare_tensors_different() {
        let original = vec![
            TensorInfo::new("layer1.weight".to_string()),
            TensorInfo::new("layer1.bias".to_string()),
        ];
        let compared = vec![TensorInfo::new("layer1.weight".to_string())];
        let result = compare_tensors(&original, &compared);
        assert_eq!(result.similarity_score, 0.5);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.status, ComparisonStatus::Different);
    }

    #[test]
    fn test_compare_tensors_with_anomalies() {
        let original = vec![
            TensorInfo::new("layer1.weight".to_string()),
            TensorInfo::new("layer1.bias".to_string()),
        ];
        let compared = vec![TensorInfo::new("layer1.weight".to_string())];
        let (result, anomalies) = compare_tensors_with_anomalies(&original, &compared);
        assert_eq!(result.similarity_score, 0.5);
        assert!(!anomalies.is_empty());
        assert!(anomalies[0].contains("layer1.bias"));
    }
}
