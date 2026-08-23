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

//! Validation de la forme (shape) des tenseurs.
//!
//! Ce module vérifie que la forme observée d'un tenseur correspond à la forme
//! attendue, conformément aux spécifications du blueprint.
//!
//! # Responsabilités
//!
//! - Comparaison des shapes observées vs attendues ;
//! - Vérification de la compatibilité des dimensions ;
//! - Détection des incohérences structurelles.
//!
//! # Formule
//!
//! La validation est simple : `shape_observé == shape_attendu`.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::types::{ValidationCategory, ValidationIssue};

/// Résultat de la validation de forme pour un tenseur.
#[derive(Debug, Clone)]
pub struct ShapeValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// Forme observée.
    pub observed_shape: Vec<u64>,
    /// Forme attendue.
    pub expected_shape: Vec<u64>,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Valide la forme d'un tenseur.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_shape, Severity};
///
/// // Validation d'un tenseur avec la bonne forme
/// let result = validate_shape(
///     "layer1.weight",
///     &[768, 3072],  // forme observée
///     &[768, 3072],  // forme attendue
/// );
/// assert!(result.issues.is_empty());
///
/// // Validation d'un tenseur avec une forme incorrecte
/// let result2 = validate_shape(
///     "layer1.weight",
///     &[768, 1024],  // forme observée
///     &[768, 3072],  // forme attendue
/// );
/// assert!(!result2.issues.is_empty());
/// assert_eq!(result2.issues[0].severity, Severity::Error);
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `observed_shape` : forme observée dans le modèle ;
/// - `expected_shape` : forme attendue selon le blueprint.
///
/// # Sorties
/// Un [`ShapeValidationResult`] contenant les issues détectées.
pub fn validate_shape(
    tensor_path: &str,
    observed_shape: &[u64],
    expected_shape: &[u64],
) -> ShapeValidationResult {
    let mut issues = Vec::new();

    // Vérification de la correspondance des shapes
    if observed_shape != expected_shape {
        let severity = if observed_shape.len() != expected_shape.len() {
            // Différence de rang → erreur critique
            Severity::Critical
        } else {
            // Même rang mais dimensions différentes → erreur
            Severity::Error
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Structural,
            severity,
            message: format!(
                "Shape observée {:?} ne correspond pas à la shape attendue {:?}",
                observed_shape, expected_shape
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    // Vérification que les dimensions sont non nulles
    for (i, &dim) in observed_shape.iter().enumerate() {
        if dim == 0 {
            issues.push(ValidationIssue {
                category: ValidationCategory::Structural,
                severity: Severity::Critical,
                message: format!("Dimension {} de la shape observée est nulle (interdit)", i),
                tensor_path: Some(tensor_path.to_string()),
            });
        }
    }

    ShapeValidationResult {
        path: tensor_path.to_string(),
        observed_shape: observed_shape.to_vec(),
        expected_shape: expected_shape.to_vec(),
        issues,
    }
}

/// Valide la compatibilité des shapes pour une opération binaire.
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `shape_a` : première forme ;
/// - `shape_b` : seconde forme ;
/// - `operation` : nom de l'opération.
///
/// # Sorties
/// Un vecteur d'[`ValidationIssue`] contenant les incompatibilités.
pub fn validate_binary_compatibility(
    tensor_path: &str,
    shape_a: &[u64],
    shape_b: &[u64],
    operation: &str,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Vérification du rang
    if shape_a.len() != shape_b.len() {
        issues.push(ValidationIssue {
            category: ValidationCategory::Structural,
            severity: Severity::Error,
            message: format!(
                "Incompatibilité de rang pour {}: {:?} vs {:?}",
                operation, shape_a, shape_b
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
        return issues;
    }

    // Vérification de la compatibilité broadcast
    for (i, (da, db)) in shape_a.iter().zip(shape_b.iter()).enumerate() {
        if da != db && *da != 1 && *db != 1 {
            issues.push(ValidationIssue {
                category: ValidationCategory::Structural,
                severity: Severity::Error,
                message: format!(
                    "Incompatibilité de dimension {} pour {}: {} vs {} (pas de broadcast possible)",
                    i, operation, da, db
                ),
                tensor_path: Some(tensor_path.to_string()),
            });
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_matching() {
        let result = validate_shape("test", &[2, 3, 4], &[2, 3, 4]);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn shape_mismatch_different_rank() {
        let result = validate_shape("test", &[2, 3], &[2, 3, 4]);
        assert!(!result.issues.is_empty());
        assert!(result
            .issues
            .iter()
            .any(|i| i.severity == Severity::Critical));
    }

    #[test]
    fn shape_mismatch_same_rank() {
        let result = validate_shape("test", &[2, 3, 4], &[2, 3, 5]);
        assert!(!result.issues.is_empty());
        assert!(result.issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn shape_with_zero_dimension() {
        let result = validate_shape("test", &[2, 0, 4], &[2, 0, 4]);
        assert!(!result.issues.is_empty());
        assert!(result
            .issues
            .iter()
            .any(|i| i.severity == Severity::Critical));
    }

    #[test]
    fn binary_compatibility_matching() {
        let issues = validate_binary_compatibility("test", &[2, 3, 4], &[2, 3, 4], "add");
        assert!(issues.is_empty());
    }

    #[test]
    fn binary_compatibility_broadcast() {
        let issues = validate_binary_compatibility("test", &[2, 3, 4], &[1, 3, 4], "add");
        assert!(issues.is_empty());
    }

    #[test]
    fn binary_compatibility_incompatible() {
        let issues = validate_binary_compatibility("test", &[2, 3, 4], &[2, 4, 4], "add");
        assert!(!issues.is_empty());
    }
}
