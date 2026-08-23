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

//! Validation de la cohérence de la génération.
//!
//! Ce module vérifie que la génération respecte les spécifications du blueprint
//! et les propriétés statistiques attendues.
//!
//! # Tests de validation
//!
//! - Nombre de tenseurs attendu vs réel
//! - Nombre de paramètres total
//! - Shapes et dtypes des tenseurs
//! - Seed utilisée
//! - Statistiques (moyenne, variance, quantiles)
//! - Injections (outlier_ratio, correlation)

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_math::statistics;

use crate::error::{GeneratorError, GeneratorResult};
use crate::generation_report::GenerationReport;

/// Résultat de la validation de génération.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation réussie.
    pub success: bool,
    /// Messages d'erreur éventuels.
    pub errors: Vec<String>,
    /// Avertissements non bloquants.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Crée un résultat de validation succès.
    pub fn success() -> Self {
        Self {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Crée un résultat de validation avec erreurs.
    pub fn failure(errors: Vec<String>) -> Self {
        Self {
            success: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Ajoute un avertissement.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Validateur de génération.
pub struct GenerationValidator {
    /// Rapport de génération à valider.
    report: GenerationReport,
    /// Spécifications des tenseurs attendues.
    specs: Vec<TensorSpec>,
}

impl GenerationValidator {
    /// Crée un nouveau validateur.
    pub fn new(report: GenerationReport, specs: Vec<TensorSpec>) -> Self {
        Self { report, specs }
    }

    /// Valide la cohérence de la génération.
    pub fn validate(&self) -> GeneratorResult<ValidationResult> {
        let mut result = ValidationResult::success();

        // Vérifier le nombre de tenseurs
        self.validate_tensor_count(&mut result)?;

        // Vérifier le nombre de paramètres
        self.validate_parameter_count(&mut result)?;

        // Vérifier les shapes et dtypes
        self.validate_specs(&mut result)?;

        // Vérifier la seed
        self.validate_seed(&mut result)?;

        // Vérifier les distributions
        self.validate_distribution_stats(&mut result)?;

        // Vérifier les injections
        self.validate_injection_stats(&mut result)?;

        Ok(result)
    }

    /// Valide le nombre de tenseurs.
    fn validate_tensor_count(&self, result: &mut ValidationResult) -> GeneratorResult<()> {
        let expected = self.specs.len() as u64;
        let actual = self.report.num_tensors;

        if expected != actual {
            result.success = false;
            result.errors.push(format!(
                "nombre de tenseurs inattendu : {} attendu(s), {} trouvé(s)",
                expected, actual
            ));
        }

        Ok(())
    }

    /// Valide le nombre total de paramètres.
    fn validate_parameter_count(&self, result: &mut ValidationResult) -> GeneratorResult<()> {
        let expected: u64 = self.specs.iter().try_fold(0u64, |acc, spec| {
            let n = spec.num_elements()?;
            acc.checked_add(n).ok_or_else(|| {
                GeneratorError::Validation("dépassement u64 du nombre de paramètres".into())
            })
        })?;

        let actual = self.report.parameter_count;

        if expected != actual {
            result.success = false;
            result.errors.push(format!(
                "nombre de paramètres inattendu : {} attendu, {} trouvé",
                expected, actual
            ));
        }

        Ok(())
    }

    /// Valide les spécifications (shapes, dtypes).
    fn validate_specs(&self, result: &mut ValidationResult) -> GeneratorResult<()> {
        // Pour l'instant, on vérifie juste que les specs sont cohérentes
        // Dans une implémentation plus complète, on vérifierait chaque tenseur généré
        for spec in &self.specs {
            if spec.name.is_empty() {
                result
                    .warnings
                    .push("tenseur avec nom vide trouvé".to_string());
            }
        }

        Ok(())
    }

    /// Valide la seed utilisée.
    fn validate_seed(&self, result: &mut ValidationResult) -> GeneratorResult<()> {
        if self.report.seed == 0 {
            result.success = false;
            result.errors.push("seed globale nulle interdite".into());
        }

        Ok(())
    }

    /// Valide les statistiques de distribution.
    fn validate_distribution_stats(&self, result: &mut ValidationResult) -> GeneratorResult<()> {
        let stats = &self.report.distribution_stats;

        // Si aucun tenseur n'a été analysé, les statistiques ne sont pas initialisées
        // (rapport par défaut), on ne génère pas d'avertissement
        if stats.total_analyzed == 0 {
            return Ok(());
        }

        // Vérifier que les pourcentages totalisent 100%
        let total = stats.normal_pct + stats.student_t_pct + stats.pareto_pct + stats.other_pct;
        if (total - 100.0).abs() > 0.1 {
            result.warnings.push(format!(
                "les pourcentages de distribution totalisent {:.1}% au lieu de 100%",
                total
            ));
        }

        // Vérifier que le nombre total analysé correspond au nombre de tenseurs
        if stats.total_analyzed != self.report.num_tensors {
            result.warnings.push(format!(
                "nombre de tenseurs analysés pour les distributions ({}) != nombre total ({})",
                stats.total_analyzed, self.report.num_tensors
            ));
        }

        Ok(())
    }

    /// Valide les statistiques d'injection.
    fn validate_injection_stats(&self, result: &mut ValidationResult) -> GeneratorResult<()> {
        let stats = &self.report.injection_stats;

        // Si aucun tenseur n'a été analysé, les statistiques ne sont pas initialisées
        // (rapport par défaut), on ne génère pas d'avertissement
        if stats.total_analyzed == 0 {
            return Ok(());
        }

        // Vérifier que le nombre total analysé correspond
        if stats.total_analyzed != self.report.num_tensors {
            result.warnings.push(format!(
                "nombre de tenseurs analysés pour les injections ({}) != nombre total ({})",
                stats.total_analyzed, self.report.num_tensors
            ));
        }

        // Vérifier que le pourcentage d'outliers est dans [0, 100]
        if stats.outlier_pct < 0.0 || stats.outlier_pct > 100.0 {
            result.warnings.push(format!(
                "pourcentage d'outliers hors bornes : {:.2}%",
                stats.outlier_pct
            ));
        }

        Ok(())
    }
}

/// Valide les statistiques d'un tableau de valeurs.
pub fn validate_tensor_stats(
    values: &[f64],
    expected_mean: Option<f64>,
    expected_std: Option<f64>,
    tolerance: f64,
) -> GeneratorResult<ValidationResult> {
    let mut result = ValidationResult::success();

    if values.is_empty() {
        result.add_warning("tableau vide".into());
        return Ok(result);
    }

    // Calculer les statistiques réelles
    let actual_mean = statistics::mean(values)?;
    let actual_std = statistics::std_sample(values)?;

    // Vérifier la moyenne
    if let Some(expected) = expected_mean {
        if (actual_mean - expected).abs() > tolerance {
            result.add_warning(format!(
                "moyenne inattendue : {:.6} attendue (±{:.6}), obtenue {:.6}",
                expected, tolerance, actual_mean
            ));
        }
    }

    // Vérifier l'écart-type
    if let Some(expected) = expected_std {
        if (actual_std - expected).abs() > tolerance {
            result.add_warning(format!(
                "écart-type inattendu : {:.6} attendu (±{:.6}), obtenu {:.6}",
                expected, tolerance, actual_std
            ));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::{DType, Shape, TensorRole};

    fn test_specs() -> Vec<TensorSpec> {
        vec![
            TensorSpec::new(
                "tensor1",
                Shape::new(vec![10, 10]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
            TensorSpec::new(
                "tensor2",
                Shape::new(vec![5, 5]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
        ]
    }

    fn test_report() -> GenerationReport {
        let mut report = GenerationReport::new("test-model", 42);
        report.num_tensors = 2;
        report.parameter_count = 125; // 10*10 + 5*5
        report
    }

    #[test]
    fn validator_creation() {
        let report = test_report();
        let specs = test_specs();
        let validator = GenerationValidator::new(report, specs);
        assert_eq!(validator.report.num_tensors, 2);
    }

    #[test]
    fn validation_success() {
        let report = test_report();
        let specs = test_specs();
        let validator = GenerationValidator::new(report, specs);
        let result = validator.validate().unwrap();
        assert!(result.success);
    }

    #[test]
    fn validation_wrong_tensor_count() {
        let mut report = test_report();
        report.num_tensors = 3; // Attendu: 2
        let specs = test_specs();
        let validator = GenerationValidator::new(report, specs);
        let result = validator.validate().unwrap();
        assert!(!result.success);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("nombre de tenseurs")));
    }

    #[test]
    fn validation_wrong_parameter_count() {
        let mut report = test_report();
        report.parameter_count = 100; // Attendu: 125
        let specs = test_specs();
        let validator = GenerationValidator::new(report, specs);
        let result = validator.validate().unwrap();
        assert!(!result.success);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("nombre de paramètres")));
    }

    #[test]
    fn validation_seed_zero() {
        let mut report = test_report();
        report.seed = 0;
        let specs = test_specs();
        let validator = GenerationValidator::new(report, specs);
        let result = validator.validate().unwrap();
        assert!(!result.success);
        assert!(result.errors.iter().any(|e| e.contains("seed")));
    }

    #[test]
    fn validate_tensor_stats_success() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_tensor_stats(&values, Some(3.0), Some(1.5811388), 0.001).unwrap();
        assert!(result.success);
    }

    #[test]
    fn validate_tensor_stats_warning() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = validate_tensor_stats(&values, Some(10.0), None, 0.001).unwrap();
        assert!(result.success); // Warning seulement
        assert!(!result.warnings.is_empty());
    }
}
