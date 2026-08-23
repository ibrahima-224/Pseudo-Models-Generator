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

//! Validation des corrélations entre tenseurs.
//!
//! Ce module vérifie les corrélations observées entre tenseurs par rapport
//! aux corrélations attendues, en utilisant le coefficient de corrélation de Pearson.
//!
//! # Responsabilités
//!
//! - Calcul du coefficient de corrélation de Pearson ;
//! - Comparaison des corrélations observées vs attendues ;
//! - Détection des déviations significatives.
//!
//! # Formule
//!
//! Coefficient de corrélation de Pearson :
//! `ρ_{X,Y} = Cov(X,Y) / (σ_X σ_Y)`
//!
//! où :
//! - `Cov(X,Y)` est la covariance entre X et Y ;
//! - `σ_X` et `σ_Y` sont les écarts-types de X et Y.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::types::{ValidationCategory, ValidationIssue};

/// Constante pour éviter la division par zéro.
const EPSILON: f64 = 1e-10;

/// Résultat de la validation de corrélation.
#[derive(Debug, Clone)]
pub struct CorrelationValidationResult {
    /// Chemin du premier tenseur.
    pub path_a: String,
    /// Chemin du deuxième tenseur.
    pub path_b: String,
    /// Corrélation observée.
    pub observed_correlation: f64,
    /// Corrélation attendue.
    pub expected_correlation: f64,
    /// Écart absolu.
    pub absolute_diff: f64,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Calcule la covariance entre deux ensembles de données.
///
/// # Entrées
/// - `data_a` : premier ensemble de données ;
/// - `data_b` : deuxième ensemble de données.
///
/// # Sorties
/// La covariance, ou `None` si les données sont vides ou de tailles différentes.
pub fn covariance(data_a: &[f64], data_b: &[f64]) -> Option<f64> {
    if data_a.len() != data_b.len() || data_a.is_empty() {
        return None;
    }

    let n = data_a.len() as f64;
    let mean_a = data_a.iter().sum::<f64>() / n;
    let mean_b = data_b.iter().sum::<f64>() / n;

    let cov = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(&a, &b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>()
        / n;

    Some(cov)
}

/// Calcule l'écart-type d'un ensemble de données.
///
/// # Entrées
/// - `data` : ensemble de données.
///
/// # Sorties
/// L'écart-type, ou `None` si les données sont vides.
pub fn standard_deviation(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    Some(variance.sqrt())
}

/// Calcule le coefficient de corrélation de Pearson entre deux ensembles de données.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::pearson_correlation;
///
/// // Données parfaitement corrélées
/// let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let data_b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
///
/// let corr = pearson_correlation(&data_a, &data_b);
/// assert!((corr.unwrap() - 1.0).abs() < 1e-6);
///
/// // Données inversement corrélées
/// let data_c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
/// let corr2 = pearson_correlation(&data_a, &data_c);
/// assert!((corr2.unwrap() - (-1.0)).abs() < 1e-6);
/// ```
///
/// # Formule
/// `ρ_{X,Y} = Cov(X,Y) / (σ_X σ_Y)`
///
/// # Entrées
/// - `data_a` : premier ensemble de données ;
/// - `data_b` : deuxième ensemble de données.
///
/// # Sorties
/// Le coefficient de corrélation (entre -1 et 1), ou `None` si le calcul est impossible.
pub fn pearson_correlation(data_a: &[f64], data_b: &[f64]) -> Option<f64> {
    let cov = covariance(data_a, data_b)?;
    let std_a = standard_deviation(data_a)?;
    let std_b = standard_deviation(data_b)?;

    let denominator = std_a * std_b;
    if denominator < EPSILON {
        return None; // Éviter la division par zéro
    }

    Some(cov / denominator)
}

/// Valide la corrélation entre deux tenseurs.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_correlation, Severity};
///
/// // Deux tenseurs fortement corrélés
/// let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let data_b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
///
/// let result = validate_correlation(
///     "layer1.weight",
///     "layer1.bias",
///     &data_a,
///     &data_b,
///     1.0,    // corrélation attendue
///     0.1,    // tolérance
/// );
///
/// // Vérification des résultats
/// assert!(result.observed_correlation > 0.9);
/// assert!(result.issues.is_empty() || result.issues.iter().all(|i| i.severity == Severity::Info));
/// ```
///
/// # Entrées
/// - `path_a` : chemin du premier tenseur ;
/// - `path_b` : chemin du deuxième tenseur ;
/// - `data_a` : données du premier tenseur ;
/// - `data_b` : données du deuxième tenseur ;
/// - `expected_correlation` : corrélation attendue ;
/// - `tolerance` : seuil de tolérance.
///
/// # Sorties
/// Un [`CorrelationValidationResult`] contenant les issues détectées.
pub fn validate_correlation(
    path_a: &str,
    path_b: &str,
    data_a: &[f64],
    data_b: &[f64],
    expected_correlation: f64,
    tolerance: f64,
) -> CorrelationValidationResult {
    let mut issues = Vec::new();

    // Calcul de la corrélation observée
    let observed_correlation = match pearson_correlation(data_a, data_b) {
        Some(corr) => corr,
        None => {
            issues.push(ValidationIssue {
                category: ValidationCategory::Statistical,
                severity: Severity::Error,
                message: format!(
                    "Impossible de calculer la corrélation entre {} et {}",
                    path_a, path_b
                ),
                tensor_path: Some(path_a.to_string()),
            });

            return CorrelationValidationResult {
                path_a: path_a.to_string(),
                path_b: path_b.to_string(),
                observed_correlation: 0.0,
                expected_correlation,
                absolute_diff: 0.0,
                issues,
            };
        },
    };

    // Calcul de l'écart
    let absolute_diff = (observed_correlation - expected_correlation).abs();

    // Vérification de la tolérance
    if absolute_diff > tolerance {
        let severity = if absolute_diff > tolerance * 2.0 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Statistical,
            severity,
            message: format!(
                "Corrélation observée ({:.6}) dévie de l'attendue ({:.6}) : écart {:.6} > tolérance {:.6}",
                observed_correlation, expected_correlation, absolute_diff, tolerance
            ),
            tensor_path: Some(path_a.to_string()),
        });
    }

    CorrelationValidationResult {
        path_a: path_a.to_string(),
        path_b: path_b.to_string(),
        observed_correlation,
        expected_correlation,
        absolute_diff,
        issues,
    }
}

/// Valide une matrice de corrélation complète.
///
/// # Entrées
/// - `tensor_names` : noms des tenseurs ;
/// - `data` : données des tenseurs (un slice par tenseur) ;
/// - `expected_matrix` : matrice de corrélation attendue (vecteur plat) ;
/// - `tolerance` : seuil de tolérance.
///
/// # Sorties
/// Un vecteur de [`CorrelationValidationResult`].
pub fn validate_correlation_matrix(
    tensor_names: &[&str],
    data: &[&[f64]],
    expected_matrix: &[f64],
    tolerance: f64,
) -> Vec<CorrelationValidationResult> {
    let n = tensor_names.len();
    let mut results = Vec::new();

    // Vérification de la taille de la matrice attendue
    if expected_matrix.len() != n * n {
        results.push(CorrelationValidationResult {
            path_a: "matrix".to_string(),
            path_b: "expected".to_string(),
            observed_correlation: 0.0,
            expected_correlation: 0.0,
            absolute_diff: 0.0,
            issues: vec![ValidationIssue {
                category: ValidationCategory::Statistical,
                severity: Severity::Error,
                message: format!(
                    "Taille de la matrice attendue incohérente : {} éléments pour {} tenseurs",
                    expected_matrix.len(),
                    n
                ),
                tensor_path: None,
            }],
        });
        return results;
    }

    // Validation de chaque paire
    for i in 0..n {
        for j in i..n {
            let expected_corr = expected_matrix[i * n + j];
            let result = validate_correlation(
                tensor_names[i],
                tensor_names[j],
                data[i],
                data[j],
                expected_corr,
                tolerance,
            );
            results.push(result);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_empty() {
        assert!(covariance(&[], &[]).is_none());
    }

    #[test]
    fn covariance_different_lengths() {
        assert!(covariance(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn covariance_values() {
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = [2.0, 4.0, 6.0, 8.0, 10.0];
        let cov = covariance(&data_a, &data_b).unwrap();
        assert!((cov - 4.0).abs() < 1e-10);
    }

    #[test]
    fn standard_deviation_empty() {
        assert!(standard_deviation(&[]).is_none());
    }

    #[test]
    fn standard_deviation_values() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let std = standard_deviation(&data).unwrap();
        assert!((std - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn pearson_correlation_perfect() {
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = [2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&data_a, &data_b).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_correlation_inverse() {
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = [10.0, 8.0, 6.0, 4.0, 2.0];
        let corr = pearson_correlation(&data_a, &data_b).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn validate_correlation_ok() {
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = [2.0, 4.0, 6.0, 8.0, 10.0];
        let result = validate_correlation("a", "b", &data_a, &data_b, 1.0, 0.01);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn validate_correlation_bad() {
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = [2.0, 4.0, 6.0, 8.0, 10.0];
        let result = validate_correlation("a", "b", &data_a, &data_b, 0.5, 0.01);
        assert!(!result.issues.is_empty());
    }
}
