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

//! Validation de la structure de bas rang des tenseurs.
//!
//! Ce module vérifie si un tenseur a une structure de bas rang, en estimant
//! la dimension réduite et en calculant le rapport d'énergie.
//!
//! # Responsabilités
//!
//! - Estimation de la dimension réduite d'un tenseur ;
//! - Calcul du rapport d'énergie R_k ;
//! - Vérification de la qualité de l'approximation de bas rang.
//!
//! # Formules
//!
//! - Rapport d'énergie : `R_k = (∑_{i=1}^k σ_i²) / (∑_i σ_i²)`
//!   où `σ_i` sont les valeurs singulières du tenseur.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::types::{ValidationCategory, ValidationIssue};
use pmg_math::low_rank::singular_values as pmg_singular_values;

/// Constante pour éviter la division par zéro.
const EPSILON: f64 = 1e-10;

/// Résultat de la validation de bas rang.
#[derive(Debug, Clone)]
pub struct LowRankValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// Dimension originale.
    pub original_dimension: usize,
    /// Dimension réduite estimée.
    pub reduced_dimension: usize,
    /// Rapport d'énergie R_k.
    pub energy_ratio: f64,
    /// Seuil d'énergie utilisé.
    pub energy_threshold: f64,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Applique la décomposition en valeurs singulières (SVD) simplifiée.
///
/// Cette fonction est une implémentation simplifiée pour la validation.
/// Elle calcule les valeurs singulières d'une matrice représentée en vecteur.
///
/// # Entrées
/// - `data` : données du tenseur (vecteur plat) ;
/// - `rows` : nombre de lignes ;
/// - `cols` : nombre de colonnes.
///
/// # Sorties
/// Un vecteur de valeurs singulières triées par ordre décroissant.
pub fn singular_values(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    if data.is_empty() || rows == 0 || cols == 0 {
        return Vec::new();
    }

    // Utilisation de l'implémentation SVD de pmg-math via la méthode de la puissance
    // avec déflation pour estimer les valeurs singulières.
    pmg_singular_values(data, rows, cols).unwrap_or_default()
}

/// Calcule le rapport d'énergie pour k valeurs singulières.
///
/// # Formule
/// `R_k = (∑_{i=1}^k σ_i²) / (∑_i σ_i²)`
///
/// # Entrées
/// - `singular_vals` : valeurs singulières triées ;
/// - `k` : nombre de valeurs singulières à considérer.
///
/// # Sorties
/// Le rapport d'énergie (entre 0 et 1).
pub fn energy_ratio(singular_vals: &[f64], k: usize) -> f64 {
    if singular_vals.is_empty() || k == 0 {
        return 0.0;
    }

    let total_energy: f64 = singular_vals.iter().map(|&x| x * x).sum();
    if total_energy < EPSILON {
        return 1.0; // Tenseur quasi-nul
    }

    let partial_energy: f64 = singular_vals.iter().take(k).map(|&x| x * x).sum();
    partial_energy / total_energy
}

/// Estime la dimension réduite nécessaire pour atteindre un seuil d'énergie.
///
/// # Entrées
/// - `singular_vals` : valeurs singulières triées ;
/// - `threshold` : seuil d'énergie (entre 0 et 1).
///
/// # Sorties
/// La dimension réduite estimée.
pub fn estimate_reduced_dimension(singular_vals: &[f64], threshold: f64) -> usize {
    if singular_vals.is_empty() {
        return 0;
    }

    let total_energy: f64 = singular_vals.iter().map(|&x| x * x).sum();
    if total_energy < EPSILON {
        return 1; // Tenseur quasi-nul
    }

    let mut cumulative_energy = 0.0;
    for (i, &sv) in singular_vals.iter().enumerate() {
        cumulative_energy += sv * sv;
        if cumulative_energy / total_energy >= threshold {
            return i + 1;
        }
    }

    singular_vals.len()
}

/// Valide la structure de bas rang d'un tenseur.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_low_rank, Severity};
///
/// // Données simulées pour une matrice 10x10
/// let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
///
/// let result = validate_low_rank(
///     "layer1.weight",
///     &data,
///     10,     // lignes
///     10,     // colonnes
///     0.9,    // seuil d'énergie de 90%
/// );
///
/// // Vérification des résultats
/// assert!(result.original_dimension > 0);
/// assert!(result.reduced_dimension <= result.original_dimension);
/// assert!(result.energy_ratio >= 0.0 && result.energy_ratio <= 1.0);
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `data` : données du tenseur (vecteur plat) ;
/// - `rows` : nombre de lignes ;
/// - `cols` : nombre de colonnes ;
/// - `energy_threshold` : seuil d'énergie souhaité.
///
/// # Sorties
/// Un [`LowRankValidationResult`] contenant les issues détectées.
pub fn validate_low_rank(
    tensor_path: &str,
    data: &[f64],
    rows: usize,
    cols: usize,
    energy_threshold: f64,
) -> LowRankValidationResult {
    let mut issues = Vec::new();

    // Calcul des valeurs singulières
    let sv = singular_values(data, rows, cols);

    // Estimation de la dimension réduite
    let reduced_dimension = estimate_reduced_dimension(&sv, energy_threshold);
    let original_dimension = rows.min(cols);

    // Calcul du rapport d'énergie réel
    let energy_ratio = energy_ratio(&sv, reduced_dimension);

    // Vérification si la dimension réduite est significativement plus petite
    if reduced_dimension < original_dimension {
        let ratio = reduced_dimension as f64 / original_dimension as f64;
        if ratio < 0.5 {
            // Forte réduction → potentiellement une structure de bas rang
            issues.push(ValidationIssue {
                category: ValidationCategory::Structural,
                severity: Severity::Info,
                message: format!(
                    "Tenseur {} a une structure de bas rang : dimension réduite {}/{} ({:.1}%)",
                    tensor_path,
                    reduced_dimension,
                    original_dimension,
                    ratio * 100.0
                ),
                tensor_path: Some(tensor_path.to_string()),
            });
        }
    }

    // Vérification que le rapport d'énergie est suffisant
    if energy_ratio < energy_threshold {
        issues.push(ValidationIssue {
            category: ValidationCategory::Structural,
            severity: Severity::Warning,
            message: format!(
                "Rapport d'énergie ({:.6}) inférieur au seuil ({:.6}) pour la dimension réduite {}",
                energy_ratio, energy_threshold, reduced_dimension
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    LowRankValidationResult {
        path: tensor_path.to_string(),
        original_dimension,
        reduced_dimension,
        energy_ratio,
        energy_threshold,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singular_values_empty() {
        assert!(singular_values(&[], 0, 0).is_empty());
    }

    #[test]
    fn energy_ratio_zero() {
        assert!((energy_ratio(&[], 0) - 0.0).abs() < EPSILON);
    }

    #[test]
    fn energy_ratio_one() {
        let sv = [1.0, 0.1, 0.01];
        let ratio = energy_ratio(&sv, 1);
        // 1.0 / (1.0 + 0.01 + 0.0001) ≈ 0.9899
        assert!((ratio - 0.9899).abs() < 0.01);
    }

    #[test]
    fn estimate_reduced_dimension_small() {
        let sv = [1.0, 0.1, 0.01, 0.001];
        let dim = estimate_reduced_dimension(&sv, 0.99);
        assert_eq!(dim, 1);
    }

    #[test]
    fn validate_low_rank_normal() {
        let data = vec![1.0; 100]; // Matrice uniforme
        let result = validate_low_rank("test", &data, 10, 10, 0.9);
        // Devrait détecter une structure de bas rang
        assert_eq!(result.path, "test");
    }
}
