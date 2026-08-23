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

//! Rapport de comparaison — formatage et affichage.
//!
//! Ce module fournit des fonctions pour formater et afficher les résultats
//! de comparaison de manière lisible, avec des tableaux, des icônes et
//! un formatage clair.
//!
//! # Responsabilités
//!
//! - Formatage du rapport de comparaison ;
//! - Affichage des résultats par catégorie ;
//! - Affichage des anomalies bloquantes ;
//! - Affichage du message "Aucune lecture profonde des poids effectuée".
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Le formatage est conçu pour être lisible par des débutants.

use crate::comparison::ComparisonReport;

/// Génère un rapport formaté à partir d'un rapport de comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::comparison::ComparisonReport;
/// use pmg_compare::report::format_report;
/// use pmg_compare::config_compare::{ConfigComparisonResult, ModelConfig, ConfigValue};
/// use pmg_compare::architecture_compare::{ArchitectureComparisonResult, ArchitectureType};
/// use pmg_compare::tensor_compare::TensorComparisonResult;
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
/// let formatted = format_report(&report);
/// assert!(formatted.contains("RAPPORT DE COMPARAISON"));
/// ```
pub fn format_report(report: &ComparisonReport) -> String {
    let mut output = String::new();

    // En-tête
    output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
    output.push_str("║           RAPPORT DE COMPARAISON DE MODÈLES                ║\n");
    output.push_str("╚══════════════════════════════════════════════════════════════╝\n");
    output.push('\n');

    // Informations sur les modèles
    output.push_str(&format!(
        "Modèle original: {}\n",
        report.original_model_name
    ));
    output.push_str(&format!(
        "Modèle comparé:  {}\n",
        report.compared_model_name
    ));
    output.push('\n');

    // Type de comparaison
    output.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    output.push_str("│ Type de comparaison: Metadata-only (sans lecture des poids)│\n");
    output.push_str("└─────────────────────────────────────────────────────────────┘\n");
    output.push('\n');

    // Score global
    output.push_str(&format!("Score global: {}\n", report.global_score));
    output.push_str(&format!("Statut global: {}\n", report.global_status));
    output.push('\n');

    // Tableau des résultats par catégorie
    output.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    output.push_str("│ RÉSULTATS PAR CATÉGORIE                                   │\n");
    output.push_str("├─────────────────────────────────────────────────────────────┤\n");
    output.push_str(&format!(
        "│ Configuration:    {:<40} │\n",
        report.config_result.status
    ));
    output.push_str(&format!(
        "│ Architecture:     {:<40} │\n",
        report.architecture_result.status
    ));
    output.push_str(&format!(
        "│ Tenseurs:         {:<40} │\n",
        report.tensor_result.status
    ));
    output.push_str(&format!(
        "│ Shapes:           {:<40} │\n",
        report.shape_result.status
    ));
    output.push_str(&format!(
        "│ Dtypes:           {:<40} │\n",
        report.dtype_result.status
    ));
    output.push_str(&format!(
        "│ Sharding:         {:<40} │\n",
        report.shard_result.status
    ));
    output.push_str("└─────────────────────────────────────────────────────────────┘\n");
    output.push('\n');

    // Anomalies bloquantes
    if report.has_blocking_anomalies() {
        output.push_str("┌─────────────────────────────────────────────────────────────┐\n");
        output.push_str(&format!(
            "│ ⚠️  ANOMALIES BLOQUANTES ({:<2})                               │\n",
            report.blocking_anomaly_count()
        ));
        output.push_str("├─────────────────────────────────────────────────────────────┤\n");
        for (i, anomaly) in report.blocking_anomalies.iter().enumerate() {
            output.push_str(&format!("│ {}. {:<54} │\n", i + 1, anomaly));
        }
        output.push_str("└─────────────────────────────────────────────────────────────┘\n");
        output.push('\n');
    }

    // Message final
    output.push_str("Note: Aucune lecture profonde des poids n'a été effectuée.\n");

    output
}

/// Génère un rapport condensé pour la CLI.
pub fn format_compact_report(report: &ComparisonReport) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "{} vs {}: {} ",
        report.original_model_name, report.compared_model_name, report.global_score
    ));

    if report.has_blocking_anomalies() {
        output.push_str(&format!(
            "({} anomalies bloquantes)",
            report.blocking_anomaly_count()
        ));
    }

    output.push_str("Poids : NON COMPARÉS ");
    output.push_str(" [metadata-only]");

    output
}

/// Génère un rapport détaillé avec toutes les différences.
pub fn format_detailed_report(report: &ComparisonReport) -> String {
    let mut output = format_report(report);

    output.push('\n');
    output.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    output.push_str("│ DIFFÉRENCES DÉTAILLÉES                                     │\n");
    output.push_str("├─────────────────────────────────────────────────────────────┤\n");

    // Différences de configuration
    if !report.config_result.differences.is_empty() {
        output.push_str("│ CONFIGURATION:                                             │\n");
        for diff in &report.config_result.differences {
            output.push_str(&format!("│   {} {:<52} │\n", diff.diff_type, diff.path));
        }
        output.push_str("│                                                            │\n");
    }

    // Différences d'architecture
    if !report.architecture_result.differences.is_empty() {
        output.push_str("│ ARCHITECTURE:                                              │\n");
        for diff in &report.architecture_result.differences {
            output.push_str(&format!("│   {} {:<52} │\n", diff.diff_type, diff.path));
        }
        output.push_str("│                                                            │\n");
    }

    // Différences de tenseurs
    if !report.tensor_result.differences.is_empty() {
        output.push_str("│ TENSEURS:                                                  │\n");
        for diff in &report.tensor_result.differences {
            output.push_str(&format!("│   {} {:<52} │\n", diff.diff_type, diff.path));
        }
        output.push_str("│                                                            │\n");
    }

    // Différences de shapes
    if !report.shape_result.differences.is_empty() {
        output.push_str("│ SHAPES:                                                    │\n");
        for diff in &report.shape_result.differences {
            output.push_str(&format!("│   {} {:<52} │\n", diff.diff_type, diff.path));
        }
        output.push_str("│                                                            │\n");
    }

    // Différences de dtypes
    if !report.dtype_result.differences.is_empty() {
        output.push_str("│ DTYPES:                                                    │\n");
        for diff in &report.dtype_result.differences {
            output.push_str(&format!("│   {} {:<52} │\n", diff.diff_type, diff.path));
        }
        output.push_str("│                                                            │\n");
    }

    // Différences de sharding
    if !report.shard_result.differences.is_empty() {
        output.push_str("│ SHARDING:                                                  │\n");
        for diff in &report.shard_result.differences {
            output.push_str(&format!("│   {} {:<52} │\n", diff.diff_type, diff.path));
        }
        output.push_str("│                                                            │\n");
    }

    output.push_str("└─────────────────────────────────────────────────────────────┘\n");

    output
}

/// Génère un rapport au format JSON.
pub fn format_json_report(report: &ComparisonReport) -> String {
    let mut json = String::new();
    json.push_str("{\n");

    // Informations générales
    json.push_str(&format!(
        "  \"original_model\": \"{}\",\n",
        report.original_model_name
    ));
    json.push_str(&format!(
        "  \"compared_model\": \"{}\",\n",
        report.compared_model_name
    ));
    json.push_str(&format!("  \"metadata_only\": {},\n", report.metadata_only));
    json.push_str(&format!(
        "  \"global_score\": {:.2},\n",
        report.global_score.percentage
    ));
    json.push_str(&format!(
        "  \"global_status\": \"{}\",\n",
        report.global_status
    ));

    // Résultats par catégorie
    json.push_str("  \"results\": {\n");
    json.push_str(&format!(
        "    \"config\": {{ \"status\": \"{}\", \"score\": {:.2} }},\n",
        report.config_result.status,
        report.config_result.similarity_score * 100.0
    ));
    json.push_str(&format!(
        "    \"architecture\": {{ \"status\": \"{}\", \"score\": {:.2} }},\n",
        report.architecture_result.status,
        report.architecture_result.compatibility_score * 100.0
    ));
    json.push_str(&format!(
        "    \"tensors\": {{ \"status\": \"{}\", \"score\": {:.2} }},\n",
        report.tensor_result.status,
        report.tensor_result.similarity_score * 100.0
    ));
    json.push_str(&format!(
        "    \"shapes\": {{ \"status\": \"{}\", \"score\": {:.2} }},\n",
        report.shape_result.status,
        report.shape_result.similarity_score * 100.0
    ));
    json.push_str(&format!(
        "    \"dtypes\": {{ \"status\": \"{}\", \"score\": {:.2} }},\n",
        report.dtype_result.status,
        report.dtype_result.similarity_score * 100.0
    ));
    json.push_str(&format!(
        "    \"sharding\": {{ \"status\": \"{}\", \"score\": {:.2} }}\n",
        report.shard_result.status,
        report.shard_result.similarity_score * 100.0
    ));
    json.push_str("  },\n");

    // Statistiques
    let total_diffs = report.config_result.differences.len()
        + report.architecture_result.differences.len()
        + report.tensor_result.differences.len()
        + report.shape_result.differences.len()
        + report.dtype_result.differences.len()
        + report.shard_result.differences.len();

    json.push_str("  \"statistics\": {\n");
    json.push_str(&format!("    \"total_differences\": {},\n", total_diffs));
    json.push_str(&format!(
        "    \"blocking_anomalies\": {}\n",
        report.blocking_anomaly_count()
    ));
    json.push_str("  },\n");

    // Anomalies bloquantes
    if !report.blocking_anomalies.is_empty() {
        json.push_str("  \"blocking_anomaly_details\": [\n");
        for (i, anomaly) in report.blocking_anomalies.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "    \"{}\"",
                anomaly.replace('\\', "\\\\").replace('"', "\\\"")
            ));
        }
        json.push_str("\n  ],\n");
    }

    // Différences (si demandé)
    json.push_str("  \"differences\": {\n");
    json.push_str(&format!(
        "    \"config\": {},\n",
        report.config_result.differences.len()
    ));
    json.push_str(&format!(
        "    \"architecture\": {},\n",
        report.architecture_result.differences.len()
    ));
    json.push_str(&format!(
        "    \"tensors\": {},\n",
        report.tensor_result.differences.len()
    ));
    json.push_str(&format!(
        "    \"shapes\": {},\n",
        report.shape_result.differences.len()
    ));
    json.push_str(&format!(
        "    \"dtypes\": {},\n",
        report.dtype_result.differences.len()
    ));
    json.push_str(&format!(
        "    \"sharding\": {}\n",
        report.shard_result.differences.len()
    ));
    json.push_str("  }\n");

    json.push('}');
    json
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_compare::ArchitectureComparisonResult;
    use crate::comparison::{ComparisonReport, ComparisonStatus};
    use crate::config_compare::ConfigComparisonResult;
    use crate::dtype_compare::DtypeComparisonResult;
    use crate::score::ComparisonScore;
    use crate::shape_compare::ShapeComparisonResult;
    use crate::shard_compare::ShardComparisonResult;
    use crate::tensor_compare::TensorComparisonResult;

    fn create_test_report() -> ComparisonReport {
        let config_result = ConfigComparisonResult::default();
        let architecture_result = ArchitectureComparisonResult::default();
        let tensor_result = TensorComparisonResult::default();
        let shape_result = ShapeComparisonResult::default();
        let dtype_result = DtypeComparisonResult::default();
        let shard_result = ShardComparisonResult::default();

        let score = ComparisonScore::new(100.0, 10, 10, 0);

        ComparisonReport::new(
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
        )
    }

    #[test]
    fn format_report_contains_key_elements() {
        let report = create_test_report();
        let formatted = format_report(&report);

        assert!(formatted.contains("RAPPORT DE COMPARAISON"));
        assert!(formatted.contains("model_a"));
        assert!(formatted.contains("model_b"));
        assert!(formatted.contains("Metadata-only"));
        assert!(formatted.contains("Aucune lecture profonde des poids"));
    }

    #[test]
    fn test_format_compact_report() {
        let report = create_test_report();
        let compact = format_compact_report(&report);

        assert!(compact.contains("model_a vs model_b"));
        assert!(compact.contains("metadata-only"));
    }
}
