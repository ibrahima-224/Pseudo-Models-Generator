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

//! Types de base pour la validation des pseudo-modèles.
//!
//! Ce module définit les structures de données utilisées pour représenter
//! les résultats de validation, les configurations et les catégories de problèmes.

use crate::severity::Severity;

/// Représente un problème détecté lors de la validation.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Catégorie du problème.
    pub category: ValidationCategory,
    /// Niveau de sévérité.
    pub severity: Severity,
    /// Message descriptif du problème.
    pub message: String,
    /// Chemin du tenseur concerné (le cas échéant).
    pub tensor_path: Option<String>,
}

/// Catégories de validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValidationCategory {
    /// Validation structurelle (shapes, dtypes, tailles).
    Structural,
    /// Validation statistique (moyennes, variances, quantiles).
    Statistical,
    /// Validation de distribution.
    Distribution,
    /// Validation des outliers.
    Outlier,
    /// Validation des métadonnées.
    Metadata,
}

impl std::fmt::Display for ValidationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationCategory::Structural => write!(f, "STRUCTURAL"),
            ValidationCategory::Statistical => write!(f, "STATISTICAL"),
            ValidationCategory::Distribution => write!(f, "DISTRIBUTION"),
            ValidationCategory::Outlier => write!(f, "OUTLIER"),
            ValidationCategory::Metadata => write!(f, "METADATA"),
        }
    }
}

/// Résultat de la validation d'un tenseur.
#[derive(Debug, Clone)]
pub struct TensorValidationResult {
    /// Chemin du tenseur.
    pub path: String,
    /// Issues détectées pour ce tenseur.
    pub issues: Vec<ValidationIssue>,
}

/// Résultat global de la validation d'un modèle.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Nom du modèle validé.
    pub model_name: String,
    /// Nombre total de tenseurs validés.
    pub tensor_count: usize,
    /// Résultats par tenseur.
    pub tensor_results: Vec<TensorValidationResult>,
    /// Nombre total d'issues par sévérité.
    pub summary: ValidationSummary,
}

/// Résumé des issues de validation.
#[derive(Debug, Clone, Default)]
pub struct ValidationSummary {
    pub info_count: usize,
    pub warning_count: usize,
    pub error_count: usize,
    pub critical_count: usize,
}

impl ValidationSummary {
    /// Retourne `true` si la validation est passée (pas d'errors ni de criticals).
    pub fn is_valid(&self) -> bool {
        self.error_count == 0 && self.critical_count == 0
    }

    /// Retourne le nombre total d'issues.
    pub fn total_issues(&self) -> usize {
        self.info_count + self.warning_count + self.error_count + self.critical_count
    }
}

/// Configuration pour la validation d'un modèle.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Seuil de détection d'outliers (nombre d'écarts-types).
    pub outlier_threshold: f64,
    /// Seuil d'énergie pour le rang effectif.
    pub energy_threshold: f64,
    /// Tolérance pour les comparaisons statistiques.
    pub statistical_tolerance: f64,
    /// Activer la validation structurelle.
    pub check_structural: bool,
    /// Activer la validation statistique.
    pub check_statistical: bool,
    /// Activer la validation de distribution.
    pub check_distribution: bool,
    /// Activer la validation des outliers.
    pub check_outliers: bool,
    /// Activer la validation des métadonnées.
    pub check_metadata: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            outlier_threshold: 3.0,
            energy_threshold: 0.9,
            statistical_tolerance: 0.1,
            check_structural: true,
            check_statistical: true,
            check_distribution: true,
            check_outliers: true,
            check_metadata: true,
        }
    }
}

/// Type alias pour simplifier la signature de `validate_model`.
pub type TensorData<'a> = (&'a str, &'a [f64], Option<f64>, Option<f64>);
