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

//! Validation de l'injection : mesurer l'effet réel et le comparer au profil.
//!
//! Le principe fondamental (spécification étape 4.9) : une politique
//! `outlier_frequency = 0.01` ne doit pas être acceptée parce qu'elle est
//! configurée — PMG doit **mesurer** le ratio réel `p̂ = N_outliers/N` et
//! vérifier `|p̂ − p| < ε`.
//!
//! [`InjectionReport`] regroupe les métriques descriptives du tenseur final ;
//! [`validate_against_policy`] compare ces métriques à la politique demandée
//! avec des tolérances, et produit [`InjectionValidation`] (succès/échec par
//! critère).
//!
//! # Exemple
//!
//! ```
//! use pmg_injector::injection_policy::InjectionPolicy;
//! use pmg_injector::injection_validator::{validate_against_policy, ValidationTolerances};
//!
//! // Simuler un tenseur injecté
//! let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
//!
//! // Politique demandée
//! let policy = InjectionPolicy::none();
//!
//! // Tolérances par défaut
//! let tolerances = ValidationTolerances::default();
//!
//! // Validation
//! let result = validate_against_policy(
//!     &values,
//!     &policy,
//!     None,       // pas de cible std
//!     3.0,        // outlier_threshold
//!     &tolerances,
//! ).unwrap();
//!
//! // La validation passe car aucune politique n'est activée
//! assert!(result.passed);
//! ```

use pmg_math::statistics::{quantiles, summary, SummaryStats};

use crate::error::{InjectorError, InjectorResult};
use crate::injection_policy::InjectionPolicy;

/// Métriques descriptives d'un tenseur après injection.
///
/// Champs alignés sur la spécification (§4.9) : `outlier_ratio`, `mean`,
/// `std`, `max_abs`, quantiles, corrélation, rang estimé.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectionReport {
    /// Nombre d'éléments.
    pub count: usize,
    /// Ratio d'outliers mesuré `p̂ = N_outliers/N`.
    pub outlier_ratio: f64,
    /// Moyenne arithmétique.
    pub mean: f64,
    /// Écart-type de population.
    pub std_population: f64,
    /// Valeur absolue maximale.
    pub max_abs: f64,
    /// Quantiles demandés (copie, dans l'ordre de la requête).
    pub quantiles: Vec<f64>,
    /// Statistiques complètes (moyenne, variance, skewness, kurtosis…).
    pub summary: SummaryStats,
    /// Corrélation moyenne hors diagonale entre colonnes (matrices 2D,
    /// `None` si non mesurable).
    pub mean_column_correlation: Option<f64>,
    /// Rang effectif estimé (matrices 2D, énergie 99 %).
    pub estimated_rank: Option<usize>,
}

impl InjectionReport {
    /// Mesure les métriques d'un tenseur plat (sans connaissance 2D).
    ///
    /// # Entrées
    /// - `values` : tenseur final (non vide) ;
    /// - `outlier_threshold` : seuil de détection d'un outlier (une valeur
    ///   `|x| > seuil` est comptée) ;
    /// - `quantile_probs` : probabilités de quantiles demandées (défaut
    ///   `[0.01, 0.25, 0.5, 0.75, 0.99]` si vide).
    ///
    /// # Erreurs
    /// [`InjectorError::Math`] si les données sont vides (propagée depuis
    /// `pmg-math::statistics`).
    ///
    /// # Complexité
    /// O(n log n) — tri pour les quantiles.
    pub fn from_values(
        values: &[f64],
        outlier_threshold: f64,
        quantile_probs: &[f64],
    ) -> InjectorResult<Self> {
        let s = summary(values)?;
        let probs: Vec<f64> = if quantile_probs.is_empty() {
            vec![0.01, 0.25, 0.5, 0.75, 0.99]
        } else {
            quantile_probs.to_vec()
        };
        let qs = quantiles(values, &probs)?;
        let outlier_ratio = values
            .iter()
            .filter(|&&x| x.abs() > outlier_threshold)
            .count() as f64
            / values.len() as f64;
        let max_abs = values.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        Ok(Self {
            count: values.len(),
            outlier_ratio,
            mean: s.mean,
            std_population: s.std_population,
            max_abs,
            quantiles: qs,
            summary: s,
            mean_column_correlation: None,
            estimated_rank: None,
        })
    }

    /// Mesure les métriques d'une matrice 2D (ajoute corrélation et rang).
    ///
    /// # Erreurs
    /// [`InjectorError::Math`] si les données sont vides ;
    /// [`InjectorError::InvalidTensor`] si `values.len() != rows·cols`.
    ///
    /// # Complexité
    /// O(n log n + cols²·rows + m²·n) — quantiles + corrélation + rang.
    pub fn from_matrix(
        values: &[f64],
        rows: usize,
        cols: usize,
        outlier_threshold: f64,
        quantile_probs: &[f64],
    ) -> InjectorResult<Self> {
        if values.len() != rows * cols {
            return Err(InjectorError::InvalidTensor(format!(
                "matrice de longueur {} ≠ rows·cols = {rows}·{cols}",
                values.len()
            )));
        }
        let mut report = Self::from_values(values, outlier_threshold, quantile_probs)?;
        if rows >= 2 && cols >= 2 {
            let corr = crate::correlated::empirical_correlation(values, rows, cols)?;
            let mut sum = 0.0;
            let mut n = 0usize;
            for a in 0..cols {
                for b in (a + 1)..cols {
                    let c = corr[a * cols + b];
                    if c.is_finite() {
                        sum += c;
                        n += 1;
                    }
                }
            }
            report.mean_column_correlation = (n > 0).then_some(sum / n as f64);
            report.estimated_rank = Some(crate::low_rank::estimate_effective_rank(
                values, rows, cols, 0.99,
            )?);
        }
        Ok(report)
    }
}

/// Résultat de la validation : chaque critère est évalué séparément.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectionValidation {
    /// `true` si tous les critères demandés sont satisfaits.
    pub passed: bool,
    /// Écart mesuré par critère (nom → écart absolu).
    pub deviations: Vec<(String, f64)>,
    /// Liste des échecs (nom → message explicatif).
    pub failures: Vec<(String, String)>,
}

/// Tolérances de validation (écarts absolus maximaux acceptés).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationTolerances {
    /// Tolérance sur le ratio d'outliers `|p̂ − p|`.
    pub outlier_ratio: f64,
    /// Tolérance relative sur l'écart-type `|σ̂/σ − 1|`.
    pub std_relative: f64,
    /// Tolérance absolue sur la corrélation `|ρ̂ − ρ²|` (colonnes).
    pub correlation: f64,
}

impl ValidationTolerances {
    /// Tolérances par défaut (documentées, adaptées aux grands tenseurs).
    pub fn default_tolerances() -> Self {
        Self {
            outlier_ratio: 0.01,
            std_relative: 0.15,
            correlation: 0.1,
        }
    }
}

impl Default for ValidationTolerances {
    fn default() -> Self {
        Self::default_tolerances()
    }
}

/// Valide le tenseur injecté contre la politique demandée.
///
/// # Critères évalués (si la politique l'exige)
/// - `outlier_frequency > 0` : `|p̂ − p| < tol.outlier_ratio` ;
/// - `correlation_strength > 0` (matrice 2D) :
///   `|ρ̂_moyen − ρ²| < tol.correlation` (la corrélation entre colonnes
///   théorique est `ρ²`) ;
/// - `std_population` : écart-type global cohérent (vérifie
///   `|σ̂/σ_cible − 1| < tol.std_relative`, où `σ_cible = spec.stddev` — si
///   aucune cible n'est disponible, le critère est ignoré).
///
/// # Entrées
/// - `values` : tenseur final ;
/// - `policy` : politique demandée ;
/// - `target_stddev` : écart-type cible du spec (`Some` pour activer le
///   critère std) ;
/// - `outlier_threshold` : seuil `|x| > seuil` pour compter un outlier ;
/// - `tolerances` : tolérances.
///
/// # Sorties
/// [`InjectionValidation`] avec `passed = failures.is_empty()` — l'échec d'un
/// critère ne court-circuite pas l'évaluation des autres.
///
/// # Erreurs
/// [`InjectorError::Math`] si les valeurs sont vides.
///
/// # Complexité
/// O(n log n) — tri pour les quantiles.
///
/// # Exemple
///
/// ```
/// use pmg_injector::injection_policy::InjectionPolicy;
/// use pmg_injector::injection_validator::{validate_against_policy, ValidationTolerances};
///
/// // Tenseur injecté avec 1% d'outliers
/// let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 10.0];
///
/// // Politique demandée : outlier_frequency = 0.01
/// let mut policy = InjectionPolicy::none();
/// policy.outlier_frequency = 0.01;
///
/// // Tolérances
/// let tolerances = ValidationTolerances {
///     outlier_ratio: 0.02,  // 2% d'écart acceptable
///     std_relative: 0.1,
///     correlation: 0.1,
/// };
///
/// // Validation
/// let result = validate_against_policy(
///     &values,
///     &policy,
///     None,       // pas de cible std
///     3.0,        // seuil outlier
///     &tolerances,
/// ).unwrap();
///
/// // Le résultat indique si la validation passe
/// println!("Validation : {}", if result.passed { "réussie" } else { "échouée" });
/// ```
pub fn validate_against_policy(
    values: &[f64],
    policy: &InjectionPolicy,
    target_stddev: Option<f64>,
    outlier_threshold: f64,
    tolerances: &ValidationTolerances,
) -> InjectorResult<InjectionValidation> {
    let report = InjectionReport::from_values(values, outlier_threshold, &[])?;
    let mut deviations: Vec<(String, f64)> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    if policy.outlier_frequency > 0.0 {
        let dev = (report.outlier_ratio - policy.outlier_frequency).abs();
        deviations.push(("outlier_ratio".into(), dev));
        if dev >= tolerances.outlier_ratio {
            failures.push((
                "outlier_ratio".into(),
                format!(
                    "p̂ = {:.4} vs p = {:.4} (écart {:.4} ≥ {:.4})",
                    report.outlier_ratio, policy.outlier_frequency, dev, tolerances.outlier_ratio
                ),
            ));
        }
    }

    if let Some(target) = target_stddev {
        if target > 0.0 {
            let rel = (report.std_population / target - 1.0).abs();
            deviations.push(("std_relative".into(), rel));
            if rel >= tolerances.std_relative {
                failures.push((
                    "std_relative".into(),
                    format!(
                        "σ̂ = {:.4} vs cible {:.4} (écart relatif {:.4} ≥ {:.4})",
                        report.std_population, target, rel, tolerances.std_relative
                    ),
                ));
            }
        }
    }

    // Corrélation : seulement si la politique l'exige et que le tenseur est
    // une matrice mesurable.
    if policy.correlation_strength > 0.0 {
        match report.mean_column_correlation {
            Some(rho_hat) => {
                let expected = policy.correlation_strength * policy.correlation_strength;
                let dev = (rho_hat - expected).abs();
                deviations.push(("correlation".into(), dev));
                if dev >= tolerances.correlation {
                    failures.push((
                        "correlation".into(),
                        format!(
                            "ρ̂ = {rho_hat:.4} vs attendu {expected:.4} (écart {dev:.4} ≥ {:.4})",
                            tolerances.correlation
                        ),
                    ));
                }
            },
            None => {
                failures.push((
                    "correlation".into(),
                    "politique de corrélation mais tenseur non matriciel (1D)".into(),
                ));
            },
        }
    }

    let passed = failures.is_empty();
    Ok(InjectionValidation {
        passed,
        deviations,
        failures,
    })
}

#[cfg(test)]
#[path = "injection_validator_tests.rs"]
mod tests;
