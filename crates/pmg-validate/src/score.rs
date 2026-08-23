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

//! Score global de validation.
//!
//! Ce module calcule un score global combinant les différentes métriques
//! de validation (structure, distribution, corrélation, bas rang, outliers).
//!
//! # Responsabilités
//!
//! - Calcul du score global à partir des scores partiels ;
//! - Gestion des poids configurables ;
//! - Garantie que les erreurs critiques ne sont jamais masquées.
//!
//! # Formule
//!
//! Score global : `S = w_s S_s + w_d S_d + w_c S_c + w_r S_r + w_o S_o`
//!
//! où :
//! - `S_s` : score structurel (shape, dtype) ;
//! - `S_d` : score de distribution ;
//! - `S_c` : score de corrélation ;
//! - `S_r` : score de bas rang ;
//! - `S_o` : score des outliers ;
//! - `w_*` : poids configurables (somme = 1).
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les scores sont entre 0 et 1 (1 = parfait).

use crate::types::{ValidationCategory, ValidationIssue, ValidationResult, ValidationSummary};

/// Poids par défaut pour le calcul du score global.
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    /// Poids pour le score structurel.
    pub structural: f64,
    /// Poids pour le score de distribution.
    pub distribution: f64,
    /// Poids pour le score de corrélation.
    pub correlation: f64,
    /// Poids pour le score de bas rang.
    pub low_rank: f64,
    /// Poids pour le score des outliers.
    pub outlier: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            structural: 0.3,
            distribution: 0.25,
            correlation: 0.2,
            low_rank: 0.15,
            outlier: 0.1,
        }
    }
}

impl ScoreWeights {
    /// Vérifie que les poids sont valides (somme = 1, tous positifs).
    pub fn validate(&self) -> bool {
        let sum =
            self.structural + self.distribution + self.correlation + self.low_rank + self.outlier;
        (sum - 1.0).abs() < 1e-6
            && self.structural >= 0.0
            && self.distribution >= 0.0
            && self.correlation >= 0.0
            && self.low_rank >= 0.0
            && self.outlier >= 0.0
    }

    /// Normalise les poids pour qu'ils somment à 1.
    pub fn normalize(&mut self) {
        let sum =
            self.structural + self.distribution + self.correlation + self.low_rank + self.outlier;
        if sum > 0.0 {
            self.structural /= sum;
            self.distribution /= sum;
            self.correlation /= sum;
            self.low_rank /= sum;
            self.outlier /= sum;
        }
    }
}

/// Compte les issues d'une catégorie spécifique dans un vecteur d'issues.
///
/// # Entrées
/// - `issues` : vecteur d'issues ;
/// - `category` : catégorie recherchée.
///
/// # Sorties
/// Nombre d'issues de cette catégorie.
fn count_issues_by_category(issues: &[ValidationIssue], category: &ValidationCategory) -> usize {
    issues.iter().filter(|i| i.category == *category).count()
}

/// Calcule le score de distribution à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score de distribution entre 0 et 1.
fn calculate_distribution_score_from_issues(issues: &[ValidationIssue]) -> f64 {
    let distribution_issues = count_issues_by_category(issues, &ValidationCategory::Distribution);
    let total_issues = issues.len();

    if total_issues == 0 {
        return 1.0; // Pas d'issues → score parfait
    }

    // Pénalité basée sur le nombre d'issues de distribution
    let penalty = distribution_issues as f64 / total_issues as f64;
    (1.0 - penalty).max(0.0)
}

/// Calcule le score de corrélation à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score de corrélation entre 0 et 1.
fn calculate_correlation_score_from_issues(issues: &[ValidationIssue]) -> f64 {
    // Les issues de corrélation sont dans la catégorie Statistical (validation statistique)
    let statistical_issues = count_issues_by_category(issues, &ValidationCategory::Statistical);
    let total_issues = issues.len();

    if total_issues == 0 {
        return 1.0; // Pas d'issues → score parfait
    }

    // Pénalité basée sur le nombre d'issues statistiques
    let penalty = statistical_issues as f64 / total_issues as f64;
    (1.0 - penalty).max(0.0)
}

/// Calcule le score de bas rang à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score de bas rang entre 0 et 1.
fn calculate_low_rank_score_from_issues(issues: &[ValidationIssue]) -> f64 {
    // Les issues de bas rang sont dans la catégorie Structural
    let structural_issues = count_issues_by_category(issues, &ValidationCategory::Structural);
    let total_issues = issues.len();

    if total_issues == 0 {
        return 1.0; // Pas d'issues → score parfait
    }

    // Pénalité basée sur le nombre d'issues structurelles
    let penalty = structural_issues as f64 / total_issues as f64;
    (1.0 - penalty).max(0.0)
}

/// Calcule le score des outliers à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score des outliers entre 0 et 1.
fn calculate_outlier_score_from_issues(issues: &[ValidationIssue]) -> f64 {
    let outlier_issues = count_issues_by_category(issues, &ValidationCategory::Outlier);
    let total_issues = issues.len();

    if total_issues == 0 {
        return 1.0; // Pas d'issues → score parfait
    }

    // Pénalité basée sur le nombre d'issues d'outliers
    let penalty = outlier_issues as f64 / total_issues as f64;
    (1.0 - penalty * 10.0).max(0.0) // Facteur 10 pour les outliers
}

/// Résultat du score global.
#[derive(Debug, Clone)]
pub struct GlobalScore {
    /// Score global (entre 0 et 1).
    pub total: f64,
    /// Score structurel.
    pub structural: f64,
    /// Score de distribution.
    pub distribution: f64,
    /// Score de corrélation.
    pub correlation: f64,
    /// Score de bas rang.
    pub low_rank: f64,
    /// Score des outliers.
    pub outlier: f64,
    /// Poids utilisés.
    pub weights: ScoreWeights,
    /// Indique si le score est masqué par des erreurs critiques.
    pub masked_by_critical: bool,
}

/// Calcule le score structurel à partir du résumé de validation.
///
/// # Entrées
/// - `summary` : résumé des issues de validation.
///
/// # Sorties
/// Score structurel entre 0 et 1.
pub fn structural_score(summary: &ValidationSummary) -> f64 {
    let total = summary.total_issues() as f64;
    if total == 0.0 {
        return 1.0;
    }

    // Pénalités differentes selon la sévérité
    let penalty = summary.info_count as f64 * 0.01
        + summary.warning_count as f64 * 0.05
        + summary.error_count as f64 * 0.2
        + summary.critical_count as f64 * 0.5;

    (1.0 - penalty.min(1.0)).max(0.0)
}

/// Calcule le score de distribution à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score de distribution entre 0 et 1.
pub fn distribution_score(issues: &[ValidationIssue]) -> f64 {
    calculate_distribution_score_from_issues(issues)
}

/// Calcule le score de corrélation à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score de corrélation entre 0 et 1.
pub fn correlation_score(issues: &[ValidationIssue]) -> f64 {
    calculate_correlation_score_from_issues(issues)
}

/// Calcule le score de bas rang à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score de bas rang entre 0 et 1.
pub fn low_rank_score(issues: &[ValidationIssue]) -> f64 {
    calculate_low_rank_score_from_issues(issues)
}

/// Calcule le score des outliers à partir des issues de validation.
///
/// # Entrées
/// - `issues` : toutes les issues de validation.
///
/// # Sorties
/// Score des outliers entre 0 et 1.
pub fn outlier_score(issues: &[ValidationIssue]) -> f64 {
    calculate_outlier_score_from_issues(issues)
}

/// Calcule le score global à partir des résultats de validation.
///
/// # Entrées
/// - `result` : résultat de la validation ;
/// - `weights` : poids configurables.
///
/// # Sorties
/// Un [`GlobalScore`] contenant le score et les détails.
pub fn calculate_global_score(result: &ValidationResult, weights: &ScoreWeights) -> GlobalScore {
    // Vérification des erreurs critiques
    let has_critical = result.summary.critical_count > 0;

    // Extraction de toutes les issues de tous les tenseurs
    let mut all_issues: Vec<ValidationIssue> = Vec::new();
    for tensor_result in &result.tensor_results {
        all_issues.extend(tensor_result.issues.iter().cloned());
    }

    // Calcul des scores partiels à partir des issues
    let s_s = structural_score(&result.summary);
    let s_d = distribution_score(&all_issues);
    let s_c = correlation_score(&all_issues);
    let s_r = low_rank_score(&all_issues);
    let s_o = outlier_score(&all_issues);

    // Score total pondéré
    let total = weights.structural * s_s
        + weights.distribution * s_d
        + weights.correlation * s_c
        + weights.low_rank * s_r
        + weights.outlier * s_o;

    // Si erreurs critiques, le score est masqué
    let masked_by_critical = has_critical;

    GlobalScore {
        total,
        structural: s_s,
        distribution: s_d,
        correlation: s_c,
        low_rank: s_r,
        outlier: s_o,
        weights: weights.clone(),
        masked_by_critical,
    }
}

/// Formate le score global pour l'affichage.
///
/// # Entrées
/// - `score` : score global.
///
/// # Sorties
/// Chaîne de caractères formatée.
pub fn format_global_score(score: &GlobalScore) -> String {
    if score.masked_by_critical {
        format!(
            "SCORE: MASQUÉ (erreurs critiques) | Structure: {:.3} | Distribution: {:.3} | Corrélation: {:.3} | Bas rang: {:.3} | Outliers: {:.3}",
            score.structural, score.distribution, score.correlation, score.low_rank, score.outlier
        )
    } else {
        format!(
            "SCORE: {:.3} | Structure: {:.3} | Distribution: {:.3} | Corrélation: {:.3} | Bas rang: {:.3} | Outliers: {:.3}",
            score.total, score.structural, score.distribution, score.correlation, score.low_rank, score.outlier
        )
    }
}

/// Génère le message d'avertissement obligatoire sur la conformité au profil.
///
/// Ce message doit être systématiquement émis pour rappeler que le score
/// mesure la conformité au PROFIL PMG, jamais la ressemblance au modèle réel.
///
/// # Paramètres
/// - `score` : score de conformité (entre 0 et 1).
///
/// # Retour
/// Chaîne de caractères contenant le message d'avertissement.
pub fn conformite_profil_message(score: f64) -> String {
    format!(
        "⚠️ AVERTISSEMENT: Ce score mesure la conformité au PROFIL PMG, \
         jamais la ressemblance au modèle réel.\n\
         Score de conformité: {:.1}%",
        score * 100.0
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_weights_default_valid() {
        let weights = ScoreWeights::default();
        assert!(weights.validate());
    }

    #[test]
    fn score_weights_normalize() {
        let mut weights = ScoreWeights {
            structural: 0.5,
            distribution: 0.5,
            correlation: 0.0,
            low_rank: 0.0,
            outlier: 0.0,
        };
        weights.normalize();
        assert!((weights.structural - 0.5).abs() < 1e-6);
        assert!((weights.distribution - 0.5).abs() < 1e-6);
    }

    #[test]
    fn structural_score_perfect() {
        let summary = ValidationSummary::default();
        assert!((structural_score(&summary) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn structural_score_with_issues() {
        let summary = ValidationSummary {
            info_count: 1,
            warning_count: 1,
            error_count: 0,
            critical_count: 0,
        };
        let score = structural_score(&summary);
        assert!(score < 1.0 && score > 0.9);
    }

    #[test]
    fn global_score_no_critical() {
        let result = ValidationResult {
            model_name: "test".to_string(),
            tensor_count: 1,
            tensor_results: vec![],
            summary: ValidationSummary::default(),
        };
        let weights = ScoreWeights::default();
        let score = calculate_global_score(&result, &weights);
        assert!(!score.masked_by_critical);
        assert!(score.total > 0.9);
    }

    #[test]
    fn global_score_with_critical() {
        let result = ValidationResult {
            model_name: "test".to_string(),
            tensor_count: 1,
            tensor_results: vec![],
            summary: ValidationSummary {
                info_count: 0,
                warning_count: 0,
                error_count: 0,
                critical_count: 1,
            },
        };
        let weights = ScoreWeights::default();
        let score = calculate_global_score(&result, &weights);
        assert!(score.masked_by_critical);
    }

    #[test]
    fn distribution_score_no_issues() {
        let issues = vec![];
        let score = distribution_score(&issues);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distribution_score_with_issues() {
        let issues = vec![ValidationIssue {
            category: ValidationCategory::Distribution,
            severity: crate::severity::Severity::Warning,
            message: "Test".to_string(),
            tensor_path: None,
        }];
        let score = distribution_score(&issues);
        assert!(score < 1.0);
    }

    #[test]
    fn correlation_score_no_issues() {
        let issues = vec![];
        let score = correlation_score(&issues);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn correlation_score_with_issues() {
        let issues = vec![ValidationIssue {
            category: ValidationCategory::Statistical,
            severity: crate::severity::Severity::Warning,
            message: "Test".to_string(),
            tensor_path: None,
        }];
        let score = correlation_score(&issues);
        assert!(score < 1.0);
    }

    #[test]
    fn low_rank_score_no_issues() {
        let issues = vec![];
        let score = low_rank_score(&issues);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn low_rank_score_with_issues() {
        let issues = vec![ValidationIssue {
            category: ValidationCategory::Structural,
            severity: crate::severity::Severity::Warning,
            message: "Test".to_string(),
            tensor_path: None,
        }];
        let score = low_rank_score(&issues);
        assert!(score < 1.0);
    }

    #[test]
    fn outlier_score_no_issues() {
        let issues = vec![];
        let score = outlier_score(&issues);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn outlier_score_with_issues() {
        let issues = vec![ValidationIssue {
            category: ValidationCategory::Outlier,
            severity: crate::severity::Severity::Warning,
            message: "Test".to_string(),
            tensor_path: None,
        }];
        let score = outlier_score(&issues);
        assert!(score < 1.0);
    }
}
