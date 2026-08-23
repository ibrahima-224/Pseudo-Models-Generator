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

//! Rapport de comparaison — structure unifiée pour les résultats.
//!
//! Ce module fournit les types principaux pour représenter les résultats
//! de comparaison entre deux modèles, incluant le rapport complet et
//! les statuts de comparaison.
//!
//! # Responsabilités
//!
//! - Structure `ComparisonReport` contenant les résultats de chaque type de comparaison ;
//! - Énumération `ComparisonStatus` pour les statuts de comparaison ;
//! - Méthodes utilitaires pour l'affichage et le calcul des scores.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les structures sont conçues pour être immuables après construction.

use crate::architecture_compare::ArchitectureComparisonResult;
use crate::config_compare::ConfigComparisonResult;
use crate::dtype_compare::DtypeComparisonResult;
use crate::score::ComparisonScore;
use crate::shape_compare::ShapeComparisonResult;
use crate::shard_compare::ShardComparisonResult;
use crate::tensor_compare::TensorComparisonResult;

/// Statut de comparaison pour un élément donné.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonStatus {
    /// Les éléments sont identiques.
    Match,
    /// Les éléments sont différents.
    Different,
    /// Les éléments sont partiellement similaires.
    Partial,
    /// Le statut est inconnu (pas de comparaison effectuée).
    Unknown,
}

impl std::fmt::Display for ComparisonStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonStatus::Match => write!(f, "✅ MATCH"),
            ComparisonStatus::Different => write!(f, "❌ DIFFERENT"),
            ComparisonStatus::Partial => write!(f, "⚠️  PARTIAL"),
            ComparisonStatus::Unknown => write!(f, "❓ UNKNOWN"),
        }
    }
}

/// Rapport complet de comparaison entre deux modèles.
///
/// # Exemple
///
/// ```
/// use pmg_compare::comparison::ComparisonReport;
/// use pmg_compare::config_compare::{ConfigComparisonResult, ModelConfig, ConfigValue};
/// use pmg_compare::architecture_compare::{ArchitectureComparisonResult, ArchitectureType};
/// use pmg_compare::tensor_compare::{TensorComparisonResult, TensorInfo};
/// use pmg_compare::shape_compare::ShapeComparisonResult;
/// use pmg_compare::dtype_compare::DtypeComparisonResult;
/// use pmg_compare::shard_compare::ShardComparisonResult;
/// use pmg_compare::score::ComparisonScore;
/// use pmg_compare::comparison::ComparisonStatus;
///
/// let config_result = ConfigComparisonResult::default();
/// let architecture_result = ArchitectureComparisonResult::new(
///     ArchitectureType::Identical,
///     1.0,
///     ComparisonStatus::Match,
///     vec![],
///     6,
///     6,
/// );
/// let tensor_result = TensorComparisonResult::default();
/// let shape_result = ShapeComparisonResult::default();
/// let dtype_result = DtypeComparisonResult::default();
/// let shard_result = ShardComparisonResult::default();
/// let global_score = ComparisonScore::new(100.0, 10, 10, 0);
///
/// let report = ComparisonReport::new(
///     "model_a".to_string(),
///     "model_b".to_string(),
///     config_result,
///     architecture_result,
///     tensor_result,
///     shape_result,
///     dtype_result,
///     shard_result,
///     global_score,
///     ComparisonStatus::Match,
///     vec![],
/// );
///
/// assert_eq!(report.original_model_name, "model_a");
/// assert_eq!(report.global_status, ComparisonStatus::Match);
/// ```
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Nom du modèle original.
    pub original_model_name: String,
    /// Nom du modèle comparé.
    pub compared_model_name: String,

    /// Résultat de la comparaison des configurations.
    pub config_result: ConfigComparisonResult,
    /// Résultat de la comparaison des architectures.
    pub architecture_result: ArchitectureComparisonResult,
    /// Résultat de la comparaison des tenseurs.
    pub tensor_result: TensorComparisonResult,
    /// Résultat de la comparaison des shapes.
    pub shape_result: ShapeComparisonResult,
    /// Résultat de la comparaison des dtypes.
    pub dtype_result: DtypeComparisonResult,
    /// Résultat de la comparaison du sharding.
    pub shard_result: ShardComparisonResult,

    /// Score global de similarité.
    pub global_score: ComparisonScore,
    /// Statut global de la comparaison.
    pub global_status: ComparisonStatus,

    /// Anomalies bloquantes (à afficher séparément).
    pub blocking_anomalies: Vec<String>,

    /// Indique si la comparaison est metadata-only (sans lecture des poids).
    pub metadata_only: bool,
}

impl ComparisonReport {
    /// Crée un nouveau rapport de comparaison.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        original_model_name: String,
        compared_model_name: String,
        config_result: ConfigComparisonResult,
        architecture_result: ArchitectureComparisonResult,
        tensor_result: TensorComparisonResult,
        shape_result: ShapeComparisonResult,
        dtype_result: DtypeComparisonResult,
        shard_result: ShardComparisonResult,
        global_score: ComparisonScore,
        global_status: ComparisonStatus,
        blocking_anomalies: Vec<String>,
    ) -> Self {
        Self {
            original_model_name,
            compared_model_name,
            config_result,
            architecture_result,
            tensor_result,
            shape_result,
            dtype_result,
            shard_result,
            global_score,
            global_status,
            blocking_anomalies,
            metadata_only: true, // Par défaut, la comparaison est metadata-only
        }
    }

    /// Vérifie s'il y a des anomalies bloquantes.
    pub fn has_blocking_anomalies(&self) -> bool {
        !self.blocking_anomalies.is_empty()
    }

    /// Retourne le nombre d'anomalies bloquantes.
    pub fn blocking_anomaly_count(&self) -> usize {
        self.blocking_anomalies.len()
    }

    /// Détermine le statut global en fonction des résultats.
    pub fn determine_global_status(&self) -> ComparisonStatus {
        // Si le score est parfait et pas d'anomalies bloquantes
        if self.global_score.percentage >= 100.0 && self.blocking_anomalies.is_empty() {
            ComparisonStatus::Match
        }
        // Si le score est inférieur à 50% ou il y a des anomalies bloquantes
        else if self.global_score.percentage < 50.0 || !self.blocking_anomalies.is_empty() {
            ComparisonStatus::Different
        }
        // Sinon, partiel
        else {
            ComparisonStatus::Partial
        }
    }
}

impl std::fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║           RAPPORT DE COMPARAISON DE MODÈLES                ║"
        )?;
        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════════╝"
        )?;
        writeln!(f)?;
        writeln!(f, "Modèle original: {}", self.original_model_name)?;
        writeln!(f, "Modèle comparé:  {}", self.compared_model_name)?;
        writeln!(f)?;

        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ Type de comparaison: Metadata-only (sans lecture des poids)│"
        )?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        writeln!(
            f,
            "Score global: {:.1}% ({})",
            self.global_score.percentage, self.global_status
        )?;
        writeln!(f)?;

        // Afficher les résultats par catégorie
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ RÉSULTATS PAR CATÉGORIE                                   │"
        )?;
        writeln!(
            f,
            "├─────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(f, "│ Configuration:    {:<40} │", self.config_result.status)?;
        writeln!(
            f,
            "│ Architecture:     {:<40} │",
            self.architecture_result.status
        )?;
        writeln!(f, "│ Tenseurs:         {:<40} │", self.tensor_result.status)?;
        writeln!(f, "│ Shapes:           {:<40} │", self.shape_result.status)?;
        writeln!(f, "│ Dtypes:           {:<40} │", self.dtype_result.status)?;
        writeln!(f, "│ Sharding:         {:<40} │", self.shard_result.status)?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Afficher les anomalies bloquantes s'il y en a
        if self.has_blocking_anomalies() {
            writeln!(
                f,
                "┌─────────────────────────────────────────────────────────────┐"
            )?;
            writeln!(
                f,
                "│ ⚠️  ANOMALIES BLOQUANTES ({:<2})                               │",
                self.blocking_anomaly_count()
            )?;
            writeln!(
                f,
                "├─────────────────────────────────────────────────────────────┤"
            )?;
            for (i, anomaly) in self.blocking_anomalies.iter().enumerate() {
                writeln!(f, "│ {}. {:<54} │", i + 1, anomaly)?;
            }
            writeln!(
                f,
                "└─────────────────────────────────────────────────────────────┘"
            )?;
            writeln!(f)?;
        }

        writeln!(
            f,
            "Note: Aucune lecture profonde des poids n'a été effectuée."
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::ComparisonScore;

    #[test]
    fn comparison_status_display() {
        assert_eq!(ComparisonStatus::Match.to_string(), "✅ MATCH");
        assert_eq!(ComparisonStatus::Different.to_string(), "❌ DIFFERENT");
        assert_eq!(ComparisonStatus::Partial.to_string(), "⚠️  PARTIAL");
        assert_eq!(ComparisonStatus::Unknown.to_string(), "❓ UNKNOWN");
    }

    #[test]
    fn comparison_report_creation() {
        let config_result = ConfigComparisonResult::default();
        let architecture_result = ArchitectureComparisonResult::default();
        let tensor_result = TensorComparisonResult::default();
        let shape_result = ShapeComparisonResult::default();
        let dtype_result = DtypeComparisonResult::default();
        let shard_result = ShardComparisonResult::default();

        let score = ComparisonScore::new(100.0, 10, 10, 0);

        let report = ComparisonReport::new(
            "model_a".to_string(),
            "model_b".to_string(),
            config_result,
            architecture_result,
            tensor_result,
            shape_result,
            dtype_result,
            shard_result,
            score,
            ComparisonStatus::Match,
            vec![],
        );

        assert_eq!(report.original_model_name, "model_a");
        assert_eq!(report.compared_model_name, "model_b");
        assert!(!report.has_blocking_anomalies());
        assert_eq!(report.blocking_anomaly_count(), 0);
        assert!(report.metadata_only);
    }

    #[test]
    fn global_status_determination() {
        let config_result = ConfigComparisonResult::default();
        let architecture_result = ArchitectureComparisonResult::default();
        let tensor_result = TensorComparisonResult::default();
        let shape_result = ShapeComparisonResult::default();
        let dtype_result = DtypeComparisonResult::default();
        let shard_result = ShardComparisonResult::default();

        // Cas 1: Score parfait, pas d'anomalies
        let score = ComparisonScore::new(100.0, 10, 10, 0);
        let report = ComparisonReport::new(
            "model_a".to_string(),
            "model_b".to_string(),
            config_result.clone(),
            architecture_result.clone(),
            tensor_result.clone(),
            shape_result.clone(),
            dtype_result.clone(),
            shard_result.clone(),
            score,
            ComparisonStatus::Match,
            vec![],
        );
        assert_eq!(report.determine_global_status(), ComparisonStatus::Match);

        // Cas 2: Score faible
        let score = ComparisonScore::new(30.0, 3, 10, 0);
        let report = ComparisonReport::new(
            "model_a".to_string(),
            "model_b".to_string(),
            config_result.clone(),
            architecture_result.clone(),
            tensor_result.clone(),
            shape_result.clone(),
            dtype_result.clone(),
            shard_result.clone(),
            score,
            ComparisonStatus::Different,
            vec![],
        );
        assert_eq!(
            report.determine_global_status(),
            ComparisonStatus::Different
        );

        // Cas 3: Anomalie bloquante
        let score = ComparisonScore::new(90.0, 9, 10, 0);
        let report = ComparisonReport::new(
            "model_a".to_string(),
            "model_b".to_string(),
            config_result,
            architecture_result,
            tensor_result,
            shape_result,
            dtype_result,
            shard_result,
            score,
            ComparisonStatus::Partial,
            vec!["Taille de embedding différente".to_string()],
        );
        assert_eq!(
            report.determine_global_status(),
            ComparisonStatus::Different
        );
    }
}
