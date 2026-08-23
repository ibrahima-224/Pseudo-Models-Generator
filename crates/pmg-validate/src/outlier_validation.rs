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

//! Validation des outliers dans les tenseurs.
//!
//! Ce module détecte les valeurs aberrantes (outliers) dans les données
//! en utilisant la règle simple basée sur l'écart-type.
//!
//! # Responsabilités
//!
//! - Détection des outliers par la règle |x - μ| > kσ ;
//! - Analyse de la fréquence, magnitude et concentration des outliers ;
//! - Vérification de la normalité des données.
//!
//! # Formule
//!
//! Règle simple : `|x - μ| > kσ`
//! où `μ` est la moyenne, `σ` l'écart-type et `k` le seuil (typiquement 3).
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::types::{ValidationCategory, ValidationIssue};

/// Constante pour éviter la division par zéro.
const EPSILON: f64 = 1e-10;

/// Résultat de la validation des outliers pour un tenseur.
#[derive(Debug, Clone)]
pub struct OutlierValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// Nombre total d'éléments.
    pub total_elements: usize,
    /// Nombre d'outliers détectés.
    pub outlier_count: usize,
    /// Fréquence des outliers (pourcentage).
    pub outlier_frequency: f64,
    /// Magnitude maximale des outliers (en nombre d'écarts-types).
    pub max_magnitude: f64,
    /// Seuil utilisé (k dans |x - μ| > kσ).
    pub threshold: f64,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Calcule la moyenne d'un slice de f64.
///
/// # Entrées
/// - `data` : slice de données.
///
/// # Sorties
/// La moyenne, ou `None` si le slice est vide.
pub fn calculate_mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        None
    } else {
        let sum: f64 = data.iter().sum();
        Some(sum / data.len() as f64)
    }
}

/// Calcule l'écart-type d'un slice de f64.
///
/// # Entrées
/// - `data` : slice de données.
///
/// # Sorties
/// L'écart-type, ou `None` si le slice est vide ou contient une seule valeur.
pub fn calculate_std(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        None
    } else {
        let mean = calculate_mean(data)?;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        Some(variance.sqrt())
    }
}

/// Détecte les outliers dans un ensemble de données.
///
/// # Entrées
/// - `data` : données à analyser ;
/// - `threshold` : seuil k pour la règle |x - μ| > kσ.
///
/// # Sorties
/// Un vecteur d'indices contenant les positions des outliers.
pub fn detect_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
    let mean = match calculate_mean(data) {
        Some(m) => m,
        None => return Vec::new(),
    };

    let std = match calculate_std(data) {
        Some(s) => s,
        None => return Vec::new(),
    };

    if std < EPSILON {
        return Vec::new(); // Pas d'outliers si écart-type nul
    }

    let k_std = threshold * std;
    data.iter()
        .enumerate()
        .filter(|(_, &x)| (x - mean).abs() > k_std)
        .map(|(i, _)| i)
        .collect()
}

/// Calcule la magnitude d'un outlier (en nombre d'écarts-types).
///
/// # Entrées
/// - `value` : valeur de l'outlier ;
/// - `mean` : moyenne ;
/// - `std` : écart-type.
///
/// # Sorties
/// La magnitude (valeur positive).
pub fn outlier_magnitude(value: f64, mean: f64, std: f64) -> f64 {
    if std < EPSILON {
        return 0.0;
    }
    (value - mean).abs() / std
}

/// Valide les outliers d'un tenseur.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_outliers, Severity};
///
/// // Données avec quelques outliers
/// let mut data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
/// data.push(10.0); // Outlier évident
///
/// let result = validate_outliers(
///     "layer1.weight",
///     &data,
///     3.0,    // seuil de 3 écarts-types
///     0.1,    // fréquence maximale de 10%
/// );
///
/// // Vérification des résultats
/// assert!(result.outlier_count > 0);
/// assert!(result.outlier_frequency <= 0.1 || !result.issues.is_empty());
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `data` : données du tenseur ;
/// - `threshold` : seuil k pour la règle |x - μ| > kσ ;
/// - `max_frequency` : fréquence maximale autorisée (entre 0 et 1).
///
/// # Sorties
/// Un [`OutlierValidationResult`] contenant les issues détectées.
pub fn validate_outliers(
    tensor_path: &str,
    data: &[f64],
    threshold: f64,
    max_frequency: f64,
) -> OutlierValidationResult {
    let mut issues = Vec::new();

    // Détection des outliers
    let outlier_indices = detect_outliers(data, threshold);
    let outlier_count = outlier_indices.len();
    let total_elements = data.len();

    // Calcul de la fréquence
    let outlier_frequency = if total_elements > 0 {
        outlier_count as f64 / total_elements as f64
    } else {
        0.0
    };

    // Calcul de la magnitude maximale
    let max_magnitude = if let (Some(mean), Some(std)) = (calculate_mean(data), calculate_std(data))
    {
        outlier_indices
            .iter()
            .map(|&i| outlier_magnitude(data[i], mean, std))
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    // Vérification de la fréquence
    if outlier_frequency > max_frequency {
        let severity = if outlier_frequency > max_frequency * 2.0 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Outlier,
            severity,
            message: format!(
                "Fréquence d'outliers ({:.6}) dépasse le maximum autorisé ({:.6})",
                outlier_frequency, max_frequency
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    // Vérification de la magnitude
    if max_magnitude > threshold * 2.0 {
        issues.push(ValidationIssue {
            category: ValidationCategory::Outlier,
            severity: Severity::Warning,
            message: format!(
                "Magnitude maximale des outliers ({:.6}σ) dépasse le double du seuil ({:.6}σ)",
                max_magnitude, threshold
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    // Information sur les outliers détectés
    if outlier_count > 0 && issues.is_empty() {
        issues.push(ValidationIssue {
            category: ValidationCategory::Outlier,
            severity: Severity::Info,
            message: format!(
                "{} outliers détectés (fréquence {:.6}, magnitude max {:.6}σ)",
                outlier_count, outlier_frequency, max_magnitude
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    OutlierValidationResult {
        path: tensor_path.to_string(),
        total_elements,
        outlier_count,
        outlier_frequency,
        max_magnitude,
        threshold,
        issues,
    }
}

/// Valide un ensemble de tenseurs pour les outliers.
///
/// # Entrées
/// - `tensor_data` : vecteur de (chemin, données) ;
/// - `threshold` : seuil k ;
/// - `max_frequency` : fréquence maximale autorisée.
///
/// # Sorties
/// Un vecteur de [`OutlierValidationResult`].
pub fn validate_tensor_outliers(
    tensor_data: &[(&str, &[f64])],
    threshold: f64,
    max_frequency: f64,
) -> Vec<OutlierValidationResult> {
    tensor_data
        .iter()
        .map(|&(path, data)| validate_outliers(path, data, threshold, max_frequency))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_mean_empty() {
        assert!(calculate_mean(&[]).is_none());
    }

    #[test]
    fn calculate_mean_values() {
        let mean = calculate_mean(&[1.0, 2.0, 3.0]).unwrap();
        assert!((mean - 2.0).abs() < EPSILON);
    }

    #[test]
    fn calculate_std_insufficient_data() {
        assert!(calculate_std(&[1.0]).is_none());
    }

    #[test]
    fn calculate_std_values() {
        let std = calculate_std(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((std - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn detect_outliers_normal() {
        // Use threshold such that only 100 is outlier.
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 100.0];
        let outliers = detect_outliers(&data, 1.0);
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 5);
    }

    #[test]
    fn detect_outliers_none() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let outliers = detect_outliers(&data, 3.0);
        assert!(outliers.is_empty());
    }

    #[test]
    fn outlier_magnitude_value() {
        let mag = outlier_magnitude(5.0, 2.0, 1.0);
        assert!((mag - 3.0).abs() < EPSILON);
    }

    #[test]
    fn validate_outliers_ok() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_outliers("test", &data, 3.0, 0.1);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn validate_outliers_too_many() {
        // Use a small threshold to have many outliers.
        let data = [0.0, 100.0, 200.0, 300.0, 400.0];
        let result = validate_outliers("test", &data, 0.5, 0.1);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn validate_outliers_empty_data() {
        // Données vides
        let data = [];
        let result = validate_outliers("test", &data, 3.0, 0.1);
        assert!(result.issues.is_empty());
        assert_eq!(result.outlier_count, 0);
    }

    #[test]
    fn validate_outliers_single_element() {
        // Un seul élément
        let data = [1.0];
        let result = validate_outliers("test", &data, 3.0, 0.1);
        assert!(result.issues.is_empty());
        assert_eq!(result.outlier_count, 0);
    }

    #[test]
    fn validate_outliers_all_same() {
        // Tous les éléments identiques
        let data = [5.0, 5.0, 5.0, 5.0, 5.0];
        let result = validate_outliers("test", &data, 3.0, 0.1);
        assert!(result.issues.is_empty());
        assert_eq!(result.outlier_count, 0);
    }
}
