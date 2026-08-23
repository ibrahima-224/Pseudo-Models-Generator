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

//! Sous-module contenant le validateur principal pour les pseudo-modèles.

use crate::correlation_validation::{validate_correlation, CorrelationValidationResult};
use crate::distribution_validation::{
    validate_distribution, Distribution, DistributionValidationResult,
};
use crate::dtype_validation::{validate_dtype, DTypeValidationResult, SimpleDType};
use crate::low_rank_validation::{validate_low_rank, LowRankValidationResult};
use crate::outlier_validation::{validate_outliers, OutlierValidationResult};
use crate::score::{calculate_global_score, GlobalScore, ScoreWeights};
use crate::severity::Severity;
use crate::shape_validation::{validate_shape, ShapeValidationResult};
use crate::statistical_helpers::compute_basic_stats;
use crate::statistical_validation::{validate_statistics, StatisticalValidationResult};

/// Import des types depuis le module dédié.
pub use crate::types::*;

/// Type alias pour les entrées de validation de tenseur.
/// Comprend : (chemin du tenseur, données, moyenne attendue, écart-type attendu).
type TensorValidationInput<'a> = (&'a str, &'a [f64], Option<f64>, Option<f64>);

/// Validateur principal pour les pseudo-modèles.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{ModelValidator, ValidationConfig, Severity};
///
/// // Création avec la configuration par défaut
/// let validator = ModelValidator::default();
///
/// // Validation d'un tenseur
/// let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let result = validator.validate_tensor(
///     "layer1.weight",
///     &data,
///     Some(0.3),
///     Some(0.1),
/// );
///
/// // Vérification des issues
/// for issue in &result.issues {
///     match issue.severity {
///         Severity::Critical => panic!("Erreur critique: {}", issue.message),
///         Severity::Error => eprintln!("Erreur: {}", issue.message),
///         Severity::Warning => println!("Avertissement: {}", issue.message),
///         Severity::Info => println!("Info: {}", issue.message),
///     }
/// }
/// ```
pub struct ModelValidator {
    config: ValidationConfig,
}

impl Default for ModelValidator {
    fn default() -> Self {
        Self::new(ValidationConfig::default())
    }
}

impl ModelValidator {
    /// Crée un nouveau validateur avec la configuration donnée.
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Valide un tenseur individuel.
    ///
    /// # Entrées
    /// - `tensor_path` : chemin du tenseur ;
    /// - `data` : données du tenseur (slice de f64) ;
    /// - `expected_mean` : moyenne attendue (optionnelle) ;
    /// - `expected_std` : écart-type attendu (optionnel).
    ///
    /// # Sorties
    /// Un [`TensorValidationResult`] contenant les issues détectées.
    pub fn validate_tensor(
        &self,
        tensor_path: &str,
        data: &[f64],
        expected_mean: Option<f64>,
        expected_std: Option<f64>,
    ) -> TensorValidationResult {
        let mut issues = Vec::new();

        // Validation structurelle
        if self.config.check_structural {
            self.check_structural(tensor_path, data, &mut issues);
        }

        // Validation statistique
        if self.config.check_statistical {
            self.check_statistical(tensor_path, data, expected_mean, expected_std, &mut issues);
        }

        // Validation des outliers
        if self.config.check_outliers {
            self.check_outliers(tensor_path, data, &mut issues);
        }

        TensorValidationResult {
            path: tensor_path.to_string(),
            issues,
        }
    }

    /// Vérifie la structure d'un tenseur.
    fn check_structural(&self, tensor_path: &str, data: &[f64], issues: &mut Vec<ValidationIssue>) {
        // Vérification que le tenseur n'est pas vide
        if data.is_empty() {
            issues.push(ValidationIssue {
                category: ValidationCategory::Structural,
                severity: Severity::Error,
                message: "Le tenseur est vide".to_string(),
                tensor_path: Some(tensor_path.to_string()),
            });
        }
    }

    /// Vérifie les propriétés statistiques d'un tenseur.
    fn check_statistical(
        &self,
        tensor_path: &str,
        data: &[f64],
        expected_mean: Option<f64>,
        expected_std: Option<f64>,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if data.is_empty() {
            return;
        }

        // Calcul des statistiques de base en une seule passe
        let (observed_mean, _variance, observed_std) = compute_basic_stats(data);

        // Comparaison avec la moyenne attendue
        if let Some(expected_mean) = expected_mean {
            let diff = (observed_mean - expected_mean).abs();
            if diff > self.config.statistical_tolerance {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Statistical,
                    severity: Severity::Warning,
                    message: format!(
                        "Moyenne observée ({:.4}) diffère de la moyenne attendue ({:.4})",
                        observed_mean, expected_mean
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            }
        }

        // Comparaison avec l'écart-type attendu
        if let Some(expected_std) = expected_std {
            let diff = (observed_std - expected_std).abs();
            if diff > self.config.statistical_tolerance {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Statistical,
                    severity: Severity::Warning,
                    message: format!(
                        "Écart-type observé ({:.4}) diffère de l'écart-type attendu ({:.4})",
                        observed_std, expected_std
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            }
        }
    }

    /// Vérifie les outliers d'un tenseur.
    fn check_outliers(&self, tensor_path: &str, data: &[f64], issues: &mut Vec<ValidationIssue>) {
        if data.is_empty() {
            return;
        }

        // Calcul des statistiques de base en une seule passe
        let (mean, _variance, std) = compute_basic_stats(data);

        // Détection des outliers par la règle des 3 écarts-types
        let outlier_count = data
            .iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * std)
            .count();

        if outlier_count > 0 {
            let frequency = outlier_count as f64 / data.len() as f64;
            if frequency > self.config.outlier_threshold {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Outlier,
                    severity: Severity::Warning,
                    message: format!(
                        "Détection de {} outliers ({:.2}%) dans le tenseur",
                        outlier_count,
                        frequency * 100.0
                    ),
                    tensor_path: Some(tensor_path.to_string()),
                });
            }
        }
    }

    /// Valide la forme d'un tenseur par rapport à une forme attendue.
    pub fn validate_tensor_shape(
        &self,
        tensor_path: &str,
        observed_shape: &[u64],
        expected_shape: &[u64],
    ) -> ShapeValidationResult {
        validate_shape(tensor_path, observed_shape, expected_shape)
    }

    /// Valide le dtype d'un tenseur par rapport à un dtype attendu.
    pub fn validate_tensor_dtype(
        &self,
        tensor_path: &str,
        observed_dtype: SimpleDType,
        expected_dtype: SimpleDType,
    ) -> DTypeValidationResult {
        validate_dtype(tensor_path, observed_dtype, expected_dtype)
    }

    /// Valide les propriétés statistiques d'un tenseur.
    pub fn validate_tensor_statistics(
        &self,
        tensor_path: &str,
        observed_mean: f64,
        target_mean: f64,
        observed_std: f64,
        target_std: f64,
    ) -> StatisticalValidationResult {
        validate_statistics(
            tensor_path,
            observed_mean,
            target_mean,
            observed_std,
            target_std,
            self.config.statistical_tolerance,
        )
    }

    /// Valide la distribution d'un tenseur.
    pub fn validate_tensor_distribution(
        &self,
        tensor_path: &str,
        data: &[f64],
        expected_distribution: Distribution,
        significance_level: f64,
    ) -> DistributionValidationResult {
        validate_distribution(tensor_path, data, expected_distribution, significance_level)
    }

    /// Valide la corrélation entre deux tenseurs.
    pub fn validate_tensor_correlation(
        &self,
        path_a: &str,
        path_b: &str,
        data_a: &[f64],
        data_b: &[f64],
        expected_correlation: f64,
    ) -> CorrelationValidationResult {
        validate_correlation(
            path_a,
            path_b,
            data_a,
            data_b,
            expected_correlation,
            self.config.statistical_tolerance,
        )
    }

    /// Valide la structure de bas rang d'un tenseur.
    pub fn validate_tensor_low_rank(
        &self,
        tensor_path: &str,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> LowRankValidationResult {
        validate_low_rank(tensor_path, data, rows, cols, self.config.energy_threshold)
    }

    /// Valide les outliers d'un tenseur.
    pub fn validate_tensor_outliers(
        &self,
        tensor_path: &str,
        data: &[f64],
        max_frequency: f64,
    ) -> OutlierValidationResult {
        validate_outliers(
            tensor_path,
            data,
            self.config.outlier_threshold,
            max_frequency,
        )
    }

    /// Calcule le score global de validation.
    pub fn calculate_score(
        &self,
        result: &ValidationResult,
        weights: &ScoreWeights,
    ) -> GlobalScore {
        calculate_global_score(result, weights)
    }

    /// Valide un modèle complet composé de plusieurs tenseurs.
    pub fn validate_model(
        &self,
        model_path: &str,
        tensors: &[TensorValidationInput<'_>],
    ) -> ValidationResult {
        let mut tensor_results = Vec::new();
        let mut summary = ValidationSummary::default();

        for (tensor_path, data, expected_mean, expected_std) in tensors {
            let result = self.validate_tensor(tensor_path, data, *expected_mean, *expected_std);
            // Mise à jour du résumé
            for issue in &result.issues {
                match issue.severity {
                    Severity::Info => summary.info_count += 1,
                    Severity::Warning => summary.warning_count += 1,
                    Severity::Error => summary.error_count += 1,
                    Severity::Critical => summary.critical_count += 1,
                }
            }
            tensor_results.push(result);
        }

        ValidationResult {
            model_name: model_path.to_string(),
            tensor_count: tensors.len(),
            tensor_results,
            summary,
        }
    }
}
