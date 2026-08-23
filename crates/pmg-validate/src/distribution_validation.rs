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

//! Validation des distributions des tenseurs.
//!
//! Ce module vérifie si les données observées suivent une distribution attendue
//! (normale, Student-t, Weibull, Pareto, log-normale) en utilisant le test
//! de Kolmogorov-Smirnov.
//!
//! # Responsabilités
//!
//! - Vérification de l'adéquation aux distributions attendues ;
//! - Test de Kolmogorov-Smirnov (KS) ;
//! - Estimation des paramètres de distribution ;
//! - Détection des déviations significatives.
//!
//! # Formules
//!
//! - Test KS : `D = sup_x |F_n(x) - F(x)|`
//!   où `F_n(x)` est la fonction de répartition empirique
//!   et `F(x)` est la fonction de répartition théorique.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::types::{ValidationCategory, ValidationIssue};

// Import des fonctions d'estimation des paramètres
use crate::distributions::{
    estimate_lognormal_params, estimate_pareto_params, estimate_student_t_params,
    estimate_weibull_params,
};

/// Représente une distribution statistique supportée.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Distribution {
    /// Distribution normale (gaussienne).
    Normal,
    /// Distribution de Student-t.
    StudentT,
    /// Distribution de Weibull.
    Weibull,
    /// Distribution de Pareto.
    Pareto,
    /// Distribution log-normale.
    LogNormal,
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Distribution::Normal => write!(f, "Normale"),
            Distribution::StudentT => write!(f, "Student-t"),
            Distribution::Weibull => write!(f, "Weibull"),
            Distribution::Pareto => write!(f, "Pareto"),
            Distribution::LogNormal => write!(f, "Log-normale"),
        }
    }
}

/// Résultat de la validation de distribution pour un tenseur.
#[derive(Debug, Clone)]
pub struct DistributionValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// Distribution testée.
    pub distribution: Distribution,
    /// Statistique KS observée.
    pub ks_statistic: f64,
    /// Seuil de signification (p-value).
    pub p_value: f64,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Calcule la fonction de répartition empirique (ECDF) en un point donné.
///
/// # Entrées
/// - `data` : données triées ;
/// - `x` : point d'évaluation.
///
/// # Sorties
/// La valeur de l'ECDF en x.
pub fn empirical_cdf(data: &[f64], x: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let count = data.iter().filter(|&&val| val <= x).count();
    count as f64 / data.len() as f64
}

/// Calcule la fonction de répartition normale (CDF) en un point donné.
///
/// # Entrées
/// - `x` : point d'évaluation ;
/// - `mu` : moyenne ;
/// - `sigma` : écart-type.
///
/// # Sorties
/// La valeur de la CDF normale en x.
pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return if x < mu { 0.0 } else { 1.0 };
    }
    let z = (x - mu) / sigma;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Fonction d'erreur (erf) approximée.
///
/// # Entrées
/// - `x` : argument.
///
/// # Sorties
/// Valeur approchée de erf(x).
fn erf(x: f64) -> f64 {
    // Approximation de Abramowitz et Stegun
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Calcule la statistique KS sur des données pré-triées.
///
/// # Description
/// Cette fonction optimisée calcule la distance maximale entre la fonction
/// de répartition empirique et une fonction de répartition théorique
/// sur des données déjà triées.
///
/// # Arguments
///
/// * `sorted_data` - Données triées en ordre croissant.
/// * `cdf` - Fonction de répartition théorique.
///
/// # Retourne
/// La statistique KS (distance maximale).
pub fn kolmogorov_smirnov_sorted(sorted_data: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let n = sorted_data.len() as f64;
    let mut max_diff: f64 = 0.0;

    for (i, &x) in sorted_data.iter().enumerate() {
        let empirical = (i as f64 + 1.0) / n;
        let theoretical = cdf(x);
        let diff = (empirical - theoretical).abs();
        max_diff = max_diff.max(diff);
    }

    max_diff
}

/// Effectue le test de Kolmogorov-Smirnov pour une distribution normale.
///
/// # Entrées
/// - `data` : données à tester ;
/// - `mu` : moyenne estimée ;
/// - `sigma` : écart-type estimé.
///
/// # Sorties
/// La statistique KS (distance maximale entre ECDF et CDF théorique).
pub fn kolmogorov_smirnov_normal(data: &[f64], mu: f64, sigma: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Pré-tri des données
    let mut sorted_data = data.to_vec();
    // Gestion safe des NaN : on les traite comme égaux pour éviter un panic
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    kolmogorov_smirnov_sorted(&sorted_data, |x| normal_cdf(x, mu, sigma))
}

/// Estime les paramètres de la distribution normale.
///
/// # Entrées
/// - `data` : données.
///
/// # Sorties
/// Tuple (mu, sigma).
pub fn estimate_normal_params(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0);
    }

    let n = data.len() as f64;
    let mu = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();

    (mu, sigma)
}

/// Test de Kolmogorov-Smirnov générique pour n'importe quelle distribution.
fn kolmogorov_smirnov_generic<F>(data: &[f64], cdf_fn: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if data.is_empty() {
        return 0.0;
    }

    // Pré-tri des données
    let mut sorted_data = data.to_vec();
    // Gestion safe des NaN : on les traite comme égaux pour éviter un panic
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    kolmogorov_smirnov_sorted(&sorted_data, cdf_fn)
}

/// Test de Kolmogorov-Smirnov pour la distribution de Student-t.
pub fn kolmogorov_smirnov_student_t(data: &[f64], df: f64, location: f64) -> f64 {
    kolmogorov_smirnov_generic(data, |x| pmg_math::special::student_t_cdf(x - location, df))
}

/// Test de Kolmogorov-Smirnov pour la distribution de Weibull.
pub fn kolmogorov_smirnov_weibull(data: &[f64], shape: f64, scale: f64) -> f64 {
    kolmogorov_smirnov_generic(data, |x| {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-(x / scale).powf(shape)).exp()
        }
    })
}

/// Test de Kolmogorov-Smirnov pour la distribution de Pareto.
pub fn kolmogorov_smirnov_pareto(data: &[f64], shape: f64, scale: f64) -> f64 {
    kolmogorov_smirnov_generic(data, |x| {
        if x < scale {
            0.0
        } else {
            1.0 - (scale / x).powf(shape)
        }
    })
}

/// Test de Kolmogorov-Smirnov pour la distribution log-normale.
pub fn kolmogorov_smirnov_lognormal(data: &[f64], mu: f64, sigma: f64) -> f64 {
    kolmogorov_smirnov_generic(data, |x| {
        if x <= 0.0 {
            0.0
        } else {
            // CDF log-normale = Φ((ln x - μ) / σ)
            let z = (x.ln() - mu) / sigma;
            0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
        }
    })
}

/// Valide la distribution d'un tenseur.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_distribution, Distribution, Severity};
///
/// // Données suivant approximativement une loi normale
/// let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
/// let result = validate_distribution(
///     "layer1.weight",
///     &data,
///     Distribution::Normal,
///     0.05,  // seuil de signification
/// );
///
/// // Interprétation des résultats
/// assert_eq!(result.distribution, Distribution::Normal);
/// if result.issues.is_empty() {
///     println!("Distribution normale acceptable");
/// } else {
///     for issue in &result.issues {
///         match issue.severity {
///             Severity::Error => eprintln!("Erreur: {}", issue.message),
///             Severity::Warning => println!("Avertissement: {}", issue.message),
///             _ => {}
///         }
///     }
/// }
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `data` : données du tenseur ;
/// - `expected_distribution` : distribution attendue ;
/// - `significance_level` : niveau de signification (seuil p-value).
///
/// # Sorties
/// Un [`DistributionValidationResult`] contenant les issues détectées.
pub fn validate_distribution(
    tensor_path: &str,
    data: &[f64],
    expected_distribution: Distribution,
    significance_level: f64,
) -> DistributionValidationResult {
    let mut issues = Vec::new();

    if data.is_empty() {
        issues.push(ValidationIssue {
            category: ValidationCategory::Distribution,
            severity: Severity::Error,
            message: "Données vides pour la validation de distribution".to_string(),
            tensor_path: Some(tensor_path.to_string()),
        });

        return DistributionValidationResult {
            path: tensor_path.to_string(),
            distribution: expected_distribution,
            ks_statistic: 0.0,
            p_value: 1.0,
            issues,
        };
    }

    // Pré-tri des données une seule fois pour optimiser les performances
    let mut sorted_data = data.to_vec();
    // Gestion safe des NaN : on les traite comme égaux pour éviter un panic
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Estimation des paramètres et calcul de la statistique KS avec données pré-triées
    let (ks_statistic, p_value) = match expected_distribution {
        Distribution::Normal => {
            let (mu, sigma) = estimate_normal_params(data); // estimation sur données non triées
            let ks = kolmogorov_smirnov_sorted(&sorted_data, |x| normal_cdf(x, mu, sigma));
            // Approximation de la p-value pour le test KS
            let p = ks_p_value(ks, data.len());
            (ks, p)
        },
        Distribution::StudentT => {
            let (df, location) = estimate_student_t_params(data);
            let ks = kolmogorov_smirnov_sorted(&sorted_data, |x| {
                pmg_math::special::student_t_cdf(x - location, df)
            });
            let p = ks_p_value(ks, data.len());
            (ks, p)
        },
        Distribution::Weibull => {
            let (shape, scale) = estimate_weibull_params(data);
            let ks = kolmogorov_smirnov_sorted(&sorted_data, |x| {
                if x < 0.0 {
                    0.0
                } else {
                    1.0 - (-(x / scale).powf(shape)).exp()
                }
            });
            let p = ks_p_value(ks, data.len());
            (ks, p)
        },
        Distribution::Pareto => {
            let (shape, scale) = estimate_pareto_params(data);
            let ks = kolmogorov_smirnov_sorted(&sorted_data, |x| {
                if x < scale {
                    0.0
                } else {
                    1.0 - (scale / x).powf(shape)
                }
            });
            let p = ks_p_value(ks, data.len());
            (ks, p)
        },
        Distribution::LogNormal => {
            let (mu, sigma) = estimate_lognormal_params(data);
            let ks = kolmogorov_smirnov_sorted(&sorted_data, |x| {
                if x <= 0.0 {
                    0.0
                } else {
                    // CDF log-normale = Φ((ln x - μ) / σ)
                    let z = (x.ln() - mu) / sigma;
                    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
                }
            });
            let p = ks_p_value(ks, data.len());
            (ks, p)
        },
    };

    // Vérification de la significativité
    if p_value < significance_level {
        let severity = if p_value < significance_level * 0.1 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Distribution,
            severity,
            message: format!(
                "Distribution {} rejetée (KS={:.6}, p={:.6} < {})",
                expected_distribution, ks_statistic, p_value, significance_level
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    DistributionValidationResult {
        path: tensor_path.to_string(),
        distribution: expected_distribution,
        ks_statistic,
        p_value,
        issues,
    }
}

/// Approximation de la p-value pour le test KS.
///
/// # Entrées
/// - `ks` : statistique KS ;
/// - `n` : taille de l'échantillon.
///
/// # Sorties
/// Approximation de la p-value.
fn ks_p_value(ks: f64, n: usize) -> f64 {
    if n == 0 || ks == 0.0 {
        return 1.0;
    }

    let lambda = (n as f64).sqrt() * ks;
    // Approximation asymptotique
    if lambda < 0.01 {
        1.0
    } else if lambda < 1.18 {
        1.0 - (-2.0 * lambda * lambda).exp()
    } else {
        // Formule de Kolmogorov
        let mut sum = 0.0;
        for k in 1..=100 {
            let term = (-2.0 * (k as f64) * (k as f64) * lambda * lambda).exp();
            if term.abs() < 1e-10 {
                break;
            }
            sum += term;
        }
        2.0 * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_cdf_empty() {
        assert!((empirical_cdf(&[], 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn empirical_cdf_values() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((empirical_cdf(&data, 3.0) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn normal_cdf_zero() {
        // L'approximation de erf peut avoir une erreur de ~1e-7
        assert!((normal_cdf(0.0, 0.0, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normal_cdf_negative() {
        assert!((normal_cdf(-1.0, 0.0, 1.0) - 0.15865525393145707).abs() < 1e-6);
    }

    #[test]
    fn erf_zero() {
        // L'approximation de erf peut avoir une erreur de ~1e-7
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn erf_one() {
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-6);
    }

    #[test]
    fn estimate_normal_params_values() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (mu, sigma) = estimate_normal_params(&data);
        assert!((mu - 3.0).abs() < 1e-10);
        assert!((sigma - std::f64::consts::SQRT_2).abs() < 1e-6);
    }
}
