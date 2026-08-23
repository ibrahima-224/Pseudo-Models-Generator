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

//! Crate `pmg-validate` — validation des pseudo-modèles générés.
//!
//! Vérifie qu'un pseudo-modèle respecte les spécifications, en quatre niveaux :
//! **format**, **structure**, **mathématique** et **modèle** (détail dans
//! `docs/architecture/06-outils-inspection-validation-comparaison.md`).
//!
//! ## Responsabilité
//!
//! - `format` : JSON, headers, offsets, encodage binaire ;
//! - `structure` : tenseurs, shapes, dtypes ;
//! - `math` : `N = ∏shape`, `bytes = N × sizeof(dtype)` ;
//! - `model` : architecture déclarée vs profil ;
//! - rapport de validation exploitable par la CLI.
//!
//! ## Dépendances
//!
//! `pmg-io`, `pmg-core`, `pmg-math`, `pmg-models`.
//!
//! ## Exemple d'utilisation
//!
//! ```rust
//! use pmg_validate::{ModelValidator, ValidationConfig, Severity};
//!
//! // Création d'un validateur avec la configuration par défaut
//! let validator = ModelValidator::default();
//!
//! // Validation d'un tenseur simple
//! let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
//! let result = validator.validate_tensor(
//!     "layer1.weight",
//!     &data,
//!     Some(0.3),  // moyenne attendue
//!     Some(0.1),  // écart-type attendu
//! );
//!
//! // Vérification des résultats
//! assert!(result.issues.is_empty() || result.issues.iter().all(|i| i.severity == Severity::Info));
//! ```
//!
//! ## Validation de distribution
//!
//! ```rust
//! use pmg_validate::{validate_distribution, Distribution};
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let result = validate_distribution(
//!     "weight_tensor",
//!     &data,
//!     Distribution::Normal,
//!     0.05,  // seuil de signification
//! );
//!
//! // Vérifier si la distribution est acceptable
//! if result.p_value > 0.05 {
//!     println!("Distribution normale acceptable");
//! }
//! ```

pub mod correlation_validation;
pub mod distribution_validation;
pub mod distributions;
pub mod dtype_validation;
pub mod low_rank_validation;
pub mod outlier_validation;
pub mod report;
pub mod score;
pub mod severity;
pub mod shape_validation;
pub mod statistical_helpers;
pub mod statistical_validation;
pub mod types;
pub mod validator;

#[cfg(test)]
pub mod correlation_validation_tests;

#[cfg(test)]
pub mod statistical_helpers_tests;

pub use correlation_validation::{
    pearson_correlation, validate_correlation, CorrelationValidationResult,
};
pub use distribution_validation::{
    validate_distribution, Distribution, DistributionValidationResult,
};
pub use distributions::{
    estimate_lognormal_params, estimate_pareto_params, estimate_student_t_params,
    estimate_weibull_params,
};
pub use dtype_validation::{validate_dtype, DTypeValidationResult, SimpleDType};
pub use low_rank_validation::{validate_low_rank, LowRankValidationResult};
pub use outlier_validation::{validate_outliers, OutlierValidationResult};
pub use report::{
    generate_console_summary, generate_full_report, generate_json_report, generate_text_report,
};
pub use score::{calculate_global_score, format_global_score, GlobalScore, ScoreWeights};
pub use severity::Severity;
pub use shape_validation::{validate_shape, ShapeValidationResult};
pub use statistical_validation::{
    relative_error, validate_external_profile, validate_statistics, ProfileValidationResult,
    StatisticalProfile, StatisticalValidationResult,
};
pub use types::TensorData;
pub use types::{
    TensorValidationResult, ValidationCategory, ValidationConfig, ValidationIssue,
    ValidationResult, ValidationSummary,
};
pub use validator::ModelValidator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_compiles() {
        // Squelette de sprint 0 : aucune API métier à tester pour l'instant.
        let _ = 0u64;
    }

    #[test]
    fn severity_is_exported() {
        let _ = Severity::Info;
        let _ = Severity::Warning;
        let _ = Severity::Error;
        let _ = Severity::Critical;
    }

    #[test]
    fn validator_is_exported() {
        let _ = ModelValidator::default();
    }
}
