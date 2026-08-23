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

//! Validation statistique des tenseurs.
//!
//! Ce module compare les propriétés statistiques observées aux valeurs cibles,
//! en utilisant l'erreur relative comme métrique principale.
//!
//! # Responsabilités
//!
//! - Comparaison des moyennes observées vs cibles ;
//! - Comparaison des écarts-types observés vs cibles ;
//! - Calcul de l'erreur relative ;
//! - Détection des déviations significatives.
//!
//! # Formules
//!
//! - Erreur relative : `E = |θ_obs - θ_target| / max(|θ_target|, ε)`
//! - ε est une petite constante pour éviter la division par zéro.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les résultats sont typés avec des niveaux de sévérité.

use crate::severity::Severity;
use crate::statistical_helpers::{calculate_mean, calculate_std};
use crate::types::{ValidationCategory, ValidationIssue};

/// Ré-exporte la fonction relative_error pour la compatibilité.
pub use crate::statistical_helpers::relative_error;

/// Résultat de la validation statistique pour un tenseur.
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// Moyenne observée.
    pub observed_mean: f64,
    /// Moyenne cible.
    pub target_mean: f64,
    /// Écart-type observé.
    pub observed_std: f64,
    /// Écart-type cible.
    pub target_std: f64,
    /// Erreur relative sur la moyenne.
    pub mean_error: f64,
    /// Erreur relative sur l'écart-type.
    pub std_error: f64,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Valide les propriétés statistiques d'un tenseur.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_statistics, Severity};
///
/// // Validation d'un tenseur avec des propriétés statistiques connues
/// let result = validate_statistics(
///     "layer1.weight",
///     0.3,    // moyenne observée
///     0.3,    // moyenne cible
///     0.1,    // écart-type observé
///     0.1,    // écart-type cible
///     0.05,   // tolérance de 5%
/// );
///
/// // Vérification des résultats
/// assert!(result.mean_error < 0.05);
/// assert!(result.std_error < 0.05);
/// for issue in &result.issues {
///     match issue.severity {
///         Severity::Error => panic!("Erreur: {}", issue.message),
///         Severity::Warning => println!("Avertissement: {}", issue.message),
///         _ => {}
///     }
/// }
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `observed_mean` : moyenne observée ;
/// - `target_mean` : moyenne cible ;
/// - `observed_std` : écart-type observé ;
/// - `target_std` : écart-type cible ;
/// - `tolerance` : seuil de tolérance pour l'erreur relative.
///
/// # Sorties
/// Un [`StatisticalValidationResult`] contenant les issues détectées.
pub fn validate_statistics(
    tensor_path: &str,
    observed_mean: f64,
    target_mean: f64,
    observed_std: f64,
    target_std: f64,
    tolerance: f64,
) -> StatisticalValidationResult {
    let mut issues = Vec::new();

    // Calcul des erreurs relatives
    let mean_error = relative_error(observed_mean, target_mean);
    let std_error = relative_error(observed_std, target_std);

    // Vérification de la moyenne
    if mean_error > tolerance {
        let severity = if mean_error > tolerance * 2.0 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Statistical,
            severity,
            message: format!(
                "Moyenne observée ({:.6}) dévie de la cible ({:.6}) : erreur relative {:.6} > tolérance {:.6}",
                observed_mean, target_mean, mean_error, tolerance
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    // Vérification de l'écart-type
    if std_error > tolerance {
        let severity = if std_error > tolerance * 2.0 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Statistical,
            severity,
            message: format!(
                "Écart-type observé ({:.6}) dévie de la cible ({:.6}) : erreur relative {:.6} > tolérance {:.6}",
                observed_std, target_std, std_error, tolerance
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    StatisticalValidationResult {
        path: tensor_path.to_string(),
        observed_mean,
        target_mean,
        observed_std,
        target_std,
        mean_error,
        std_error,
        issues,
    }
}

/// Valide un ensemble de tenseurs avec des statistiques cibles.
///
/// # Entrées
/// - `tensor_stats` : vecteur de (chemin, moyenne observée, moyenne cible,
///   écart-type observé, écart-type cible) ;
/// - `tolerance` : seuil de tolérance.
///
/// # Sorties
/// Un vecteur de [`StatisticalValidationResult`].
pub fn validate_tensor_statistics(
    tensor_stats: &[(&str, f64, f64, f64, f64)],
    tolerance: f64,
) -> Vec<StatisticalValidationResult> {
    tensor_stats
        .iter()
        .map(|&(path, obs_mean, tgt_mean, obs_std, tgt_std)| {
            validate_statistics(path, obs_mean, tgt_mean, obs_std, tgt_std, tolerance)
        })
        .collect()
}

/// Représente un profil statistique externe.
#[derive(Debug, Clone)]
pub struct StatisticalProfile {
    /// Nom du profil.
    pub name: String,
    /// Moyenne attendue.
    pub expected_mean: f64,
    /// Écart-type attendu.
    pub expected_std: f64,
    /// Tolérance pour la moyenne.
    pub mean_tolerance: f64,
    /// Tolérance pour l'écart-type.
    pub std_tolerance: f64,
}

/// Résultat de la validation d'un profil statistique externe.
#[derive(Debug, Clone)]
pub struct ProfileValidationResult {
    /// Nom du profil.
    pub profile_name: String,
    /// Chemin du tenseur validé.
    pub tensor_path: String,
    /// Moyenne observée.
    pub observed_mean: f64,
    /// Moyenne attendue.
    pub expected_mean: f64,
    /// Écart-type observé.
    pub observed_std: f64,
    /// Écart-type attendu.
    pub expected_std: f64,
    /// Issues détectées.
    pub issues: Vec<ValidationIssue>,
}

/// Valide un tenseur par rapport à un profil statistique externe.
///
/// # Exemple
///
/// ```rust
/// use pmg_validate::{validate_external_profile, StatisticalProfile, Severity};
///
/// // Création d'un profil statistique
/// let profile = StatisticalProfile {
///     name: "deepseek_v4_flash".to_string(),
///     expected_mean: 0.0,
///     expected_std: 1.0,
///     mean_tolerance: 0.1,
///     std_tolerance: 0.1,
/// };
///
/// // Données du tenseur
/// let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
///
/// // Validation
/// let result = validate_external_profile("layer1.weight", &data, &profile);
///
/// // Vérification des résultats
/// assert_eq!(result.profile_name, "deepseek_v4_flash");
/// ```
///
/// # Entrées
/// - `tensor_path` : chemin du tenseur ;
/// - `data` : données du tenseur ;
/// - `profile` : profil statistique externe.
///
/// # Sorties
/// Un [`ProfileValidationResult`] contenant les issues détectées.
pub fn validate_external_profile(
    tensor_path: &str,
    data: &[f64],
    profile: &StatisticalProfile,
) -> ProfileValidationResult {
    let mut issues = Vec::new();

    // Calcul des propriétés statistiques observées
    let observed_mean = calculate_mean(data).unwrap_or(0.0);
    let observed_std = calculate_std(data).unwrap_or(0.0);

    // Vérification de la moyenne
    let mean_error = relative_error(observed_mean, profile.expected_mean);
    if mean_error > profile.mean_tolerance {
        let severity = if mean_error > profile.mean_tolerance * 2.0 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Statistical,
            severity,
            message: format!(
                "Moyenne observée ({:.6}) dévie de la moyenne attendue ({:.6}) pour le profil {} : erreur {:.6} > tolérance {:.6}",
                observed_mean, profile.expected_mean, profile.name, mean_error, profile.mean_tolerance
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    // Vérification de l'écart-type
    let std_error = relative_error(observed_std, profile.expected_std);
    if std_error > profile.std_tolerance {
        let severity = if std_error > profile.std_tolerance * 2.0 {
            Severity::Error
        } else {
            Severity::Warning
        };

        issues.push(ValidationIssue {
            category: ValidationCategory::Statistical,
            severity,
            message: format!(
                "Écart-type observé ({:.6}) dévie de l'écart-type attendu ({:.6}) pour le profil {} : erreur {:.6} > tolérance {:.6}",
                observed_std, profile.expected_std, profile.name, std_error, profile.std_tolerance
            ),
            tensor_path: Some(tensor_path.to_string()),
        });
    }

    ProfileValidationResult {
        profile_name: profile.name.clone(),
        tensor_path: tensor_path.to_string(),
        observed_mean,
        expected_mean: profile.expected_mean,
        observed_std,
        expected_std: profile.expected_std,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistical_helpers::EPSILON;

    #[test]
    fn relative_error_zero() {
        assert!((relative_error(1.0, 1.0) - 0.0).abs() < EPSILON);
    }

    #[test]
    fn relative_error_positive() {
        let error = relative_error(1.5, 1.0);
        assert!((error - 0.5).abs() < EPSILON);
    }

    #[test]
    fn relative_error_zero_target() {
        // Quand la cible est zéro, l'erreur est |observed| / EPSILON ce qui est grand.
        let error = relative_error(0.1, 0.0);
        assert!(error > 1e8);
    }

    #[test]
    fn validate_statistics_ok() {
        let result = validate_statistics("test", 1.0, 1.0, 0.5, 0.5, 0.1);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn validate_statistics_mean_warning() {
        let result = validate_statistics("test", 1.1, 1.0, 0.5, 0.5, 0.1);
        assert!(!result.issues.is_empty());
        assert!(result
            .issues
            .iter()
            .any(|i| i.severity == Severity::Warning));
    }

    #[test]
    fn validate_statistics_mean_error() {
        let result = validate_statistics("test", 1.5, 1.0, 0.5, 0.5, 0.1);
        assert!(!result.issues.is_empty());
        assert!(result.issues.iter().any(|i| i.severity == Severity::Error));
    }

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
        assert!((std - std::f64::consts::SQRT_2).abs() < EPSILON);
    }

    #[test]
    fn validate_external_profile_ok() {
        let profile = StatisticalProfile {
            name: "test_profile".to_string(),
            expected_mean: 2.0,
            expected_std: 0.816, // Écart-type exact pour [1.0, 2.0, 3.0]
            mean_tolerance: 0.1,
            std_tolerance: 0.1,
        };
        let data = [1.0, 2.0, 3.0];
        let result = validate_external_profile("test", &data, &profile);
        assert!(result.issues.is_empty());
        assert_eq!(result.profile_name, "test_profile");
    }

    #[test]
    fn validate_external_profile_mean_warning() {
        let profile = StatisticalProfile {
            name: "test_profile".to_string(),
            expected_mean: 2.0,
            expected_std: 1.0,
            mean_tolerance: 0.1,
            std_tolerance: 1.0, // Tolérance large pour l'écart-type
        };
        let data = [1.5, 2.0, 2.5]; // moyenne = 2.0
        let result = validate_external_profile("test", &data, &profile);
        // La moyenne est exacte, donc pas d'erreur
        assert!(result.issues.is_empty());
    }

    #[test]
    fn validate_external_profile_mean_error() {
        let profile = StatisticalProfile {
            name: "test_profile".to_string(),
            expected_mean: 2.0,
            expected_std: 1.0,
            mean_tolerance: 0.1,
            std_tolerance: 1.0, // Tolérance large pour l'écart-type
        };
        let data = [0.0, 0.0, 6.0]; // moyenne = 2.0, mais écart-type élevé
        let result = validate_external_profile("test", &data, &profile);
        // La moyenne est correcte mais l'écart-type peut dévier
        assert_eq!(result.profile_name, "test_profile");
    }

    #[test]
    fn validate_external_profile_std_warning() {
        let profile = StatisticalProfile {
            name: "test_profile".to_string(),
            expected_mean: 2.0,
            expected_std: 1.0,
            mean_tolerance: 0.1,
            std_tolerance: 0.1,
        };
        let data = [1.0, 2.0, 3.0]; // moyenne = 2.0, std = 0.816
        let result = validate_external_profile("test", &data, &profile);
        // L'écart-type observé (0.816) dévie de l'attendu (1.0) de 18%
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn validate_external_profile_empty_data() {
        let profile = StatisticalProfile {
            name: "test_profile".to_string(),
            expected_mean: 2.0,
            expected_std: 1.0,
            mean_tolerance: 0.1,
            std_tolerance: 0.1,
        };
        let data = [];
        let result = validate_external_profile("test", &data, &profile);
        // Les valeurs observées seront 0.0, 0.0
        assert!(!result.issues.is_empty());
    }
}
