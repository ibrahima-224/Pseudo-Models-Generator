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

//! Rapport de validation — génération et formatage des résultats.
//!
//! Ce module fournit des fonctions pour générer des rapports de validation
//! lisibles à partir des résultats de [`ValidationResult`].
//!
//! # Responsabilités
//!
//! - Formatage des résultats de validation en texte lisible ;
//! - Résumé des issues par catégorie et sévérité ;
//! - Export en différents formats (texte, JSON) ;
//! - Affichage des anomalies bloquantes séparément ;
//! - Inclusion du score global.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les rapports sont concis mais informatifs.

use crate::score::{format_global_score, GlobalScore};
use crate::types::ValidationResult;

/// Génère un rapport de validation en texte lisible.
///
/// # Entrées
/// - `result` : résultat de la validation.
///
/// # Sorties
/// Chaîne de caractères contenant le rapport formaté.
pub fn generate_text_report(result: &ValidationResult) -> String {
    let mut report = String::new();

    // En-tête
    report.push_str("=== RAPPORT DE VALIDATION ===\n\n");
    report.push_str(&format!("Modèle : {}\n", result.model_name));
    report.push_str(&format!("Nombre de tenseurs : {}\n\n", result.tensor_count));

    // Résumé
    report.push_str("--- Résumé ---\n");
    report.push_str(&format!(
        "Total issues : {}\n",
        result.summary.total_issues()
    ));
    report.push_str(&format!("  INFO : {}\n", result.summary.info_count));
    report.push_str(&format!("  WARNING : {}\n", result.summary.warning_count));
    report.push_str(&format!("  ERROR : {}\n", result.summary.error_count));
    report.push_str(&format!(
        "  CRITICAL : {}\n\n",
        result.summary.critical_count
    ));

    // Statut
    if result.summary.is_valid() {
        report.push_str("✅ VALIDATION RÉUSSIE\n\n");
    } else {
        report.push_str("❌ VALIDATION ÉCHOUÉE\n\n");
    }

    // Détails par tenseur
    report.push_str("--- Détails par tenseur ---\n\n");
    for tensor_result in &result.tensor_results {
        if tensor_result.issues.is_empty() {
            report.push_str(&format!("✅ {} : OK\n", tensor_result.path));
        } else {
            report.push_str(&format!(
                "❌ {} : {} issue(s)\n",
                tensor_result.path,
                tensor_result.issues.len()
            ));
            for issue in &tensor_result.issues {
                report.push_str(&format!(
                    "  [{}] {} : {}\n",
                    issue.severity, issue.category, issue.message
                ));
            }
        }
        report.push('\n');
    }

    report
}

/// Génère un rapport de validation en JSON.
///
/// # Entrées
/// - `result` : résultat de la validation.
///
/// # Sorties
/// Chaîne de caractères contenant le JSON formaté.
pub fn generate_json_report(result: &ValidationResult) -> String {
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"model_name\": \"{}\",\n", result.model_name));
    json.push_str(&format!("  \"tensor_count\": {},\n", result.tensor_count));

    // Résumé
    json.push_str("  \"summary\": {\n");
    json.push_str(&format!(
        "    \"total_issues\": {},\n",
        result.summary.total_issues()
    ));
    json.push_str(&format!("    \"info\": {},\n", result.summary.info_count));
    json.push_str(&format!(
        "    \"warning\": {},\n",
        result.summary.warning_count
    ));
    json.push_str(&format!("    \"error\": {},\n", result.summary.error_count));
    json.push_str(&format!(
        "    \"critical\": {},\n",
        result.summary.critical_count
    ));
    json.push_str(&format!(
        "    \"is_valid\": {}\n",
        result.summary.is_valid()
    ));
    json.push_str("  },\n");

    // Tenseurs
    json.push_str("  \"tensors\": [\n");
    for (i, tensor_result) in result.tensor_results.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"path\": \"{}\",\n", tensor_result.path));
        json.push_str(&format!(
            "      \"issue_count\": {},\n",
            tensor_result.issues.len()
        ));
        json.push_str("      \"issues\": [\n");
        for (j, issue) in tensor_result.issues.iter().enumerate() {
            json.push_str("        {\n");
            json.push_str(&format!(
                "          \"category\": \"{}\",\n",
                issue.category
            ));
            json.push_str(&format!(
                "          \"severity\": \"{}\",\n",
                issue.severity
            ));
            json.push_str(&format!("          \"message\": \"{}\"", issue.message));
            if let Some(ref path) = issue.tensor_path {
                json.push_str(&format!(",\n          \"tensor_path\": \"{}\"", path));
            }
            json.push_str("\n        }");
            if j < tensor_result.issues.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("      ]\n");
        json.push_str("    }");
        if i < result.tensor_results.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    json
}

/// Génère un résumé condensé pour la console.
///
/// # Entrées
/// - `result` : résultat de la validation.
///
/// # Sorties
/// Chaîne de caractères contenant le résumé condensé.
pub fn generate_console_summary(result: &ValidationResult) -> String {
    let status = if result.summary.is_valid() {
        "✅ OK"
    } else {
        "❌ ÉCHEC"
    };

    format!(
        "{} | {} tenseurs | {} issues ({}E, {}W, {}C)",
        status,
        result.tensor_count,
        result.summary.total_issues(),
        result.summary.error_count,
        result.summary.warning_count,
        result.summary.critical_count
    )
}

/// Génère un rapport complet avec score global et anomalies bloquantes.
///
/// # Entrées
/// - `result` : résultat de la validation ;
/// - `score` : score global (optionnel).
///
/// # Sorties
/// Chaîne de caractères contenant le rapport formaté.
pub fn generate_full_report(result: &ValidationResult, score: Option<&GlobalScore>) -> String {
    let mut report = String::new();

    // En-tête
    report.push_str("=== RAPPORT DE VALIDATION COMPLET ===\n\n");
    report.push_str(&format!("Modèle : {}\n", result.model_name));
    report.push_str(&format!("Nombre de tenseurs : {}\n\n", result.tensor_count));

    // Score global
    if let Some(score) = score {
        report.push_str("--- Score Global ---\n");
        report.push_str(&format!("{}\n\n", format_global_score(score)));
    }

    // Résumé
    report.push_str("--- Résumé ---\n");
    report.push_str(&format!(
        "Total issues : {}\n",
        result.summary.total_issues()
    ));
    report.push_str(&format!("  INFO : {}\n", result.summary.info_count));
    report.push_str(&format!("  WARNING : {}\n", result.summary.warning_count));
    report.push_str(&format!("  ERROR : {}\n", result.summary.error_count));
    report.push_str(&format!(
        "  CRITICAL : {}\n\n",
        result.summary.critical_count
    ));

    // Statut
    if result.summary.is_valid() {
        report.push_str("✅ VALIDATION RÉUSSIE\n\n");
    } else {
        report.push_str("❌ VALIDATION ÉCHOUÉE\n\n");
    }

    // Anomalies bloquantes (erreurs et critiques)
    let blocking_issues: Vec<_> = result
        .tensor_results
        .iter()
        .flat_map(|tr| &tr.issues)
        .filter(|i| {
            i.severity == crate::severity::Severity::Error
                || i.severity == crate::severity::Severity::Critical
        })
        .collect();

    if !blocking_issues.is_empty() {
        report.push_str("--- ANOMALIES BLOQUANTES ---\n\n");
        for issue in &blocking_issues {
            report.push_str(&format!(
                "❌ [{}] {} : {}\n",
                issue.severity, issue.category, issue.message
            ));
            if let Some(ref path) = issue.tensor_path {
                report.push_str(&format!("   Tenseur : {}\n", path));
            }
        }
        report.push('\n');
    }

    // Détails par tenseur
    report.push_str("--- Détails par tenseur ---\n\n");
    for tensor_result in &result.tensor_results {
        if tensor_result.issues.is_empty() {
            report.push_str(&format!("✅ {} : OK\n", tensor_result.path));
        } else {
            report.push_str(&format!(
                "❌ {} : {} issue(s)\n",
                tensor_result.path,
                tensor_result.issues.len()
            ));
            for issue in &tensor_result.issues {
                report.push_str(&format!(
                    "  [{}] {} : {}\n",
                    issue.severity, issue.category, issue.message
                ));
            }
        }
        report.push('\n');
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::severity::Severity;
    use crate::types::{
        TensorValidationResult, ValidationCategory, ValidationIssue, ValidationSummary,
    };

    fn create_test_result() -> ValidationResult {
        let tensor_results = vec![
            TensorValidationResult {
                path: "layer1.weight".to_string(),
                issues: vec![ValidationIssue {
                    category: ValidationCategory::Statistical,
                    severity: Severity::Warning,
                    message: "Moyenne différente de l'attendue".to_string(),
                    tensor_path: Some("layer1.weight".to_string()),
                }],
            },
            TensorValidationResult {
                path: "layer2.weight".to_string(),
                issues: vec![],
            },
        ];

        ValidationResult {
            model_name: "test_model".to_string(),
            tensor_count: 2,
            tensor_results,
            summary: ValidationSummary {
                info_count: 0,
                warning_count: 1,
                error_count: 0,
                critical_count: 0,
            },
        }
    }

    #[test]
    fn generate_text_report_contains_model_name() {
        let result = create_test_result();
        let report = generate_text_report(&result);
        assert!(report.contains("test_model"));
    }

    #[test]
    fn generate_text_report_contains_tensor_count() {
        let result = create_test_result();
        let report = generate_text_report(&result);
        assert!(report.contains("Nombre de tenseurs : 2"));
    }

    #[test]
    fn generate_json_report_valid_json() {
        let result = create_test_result();
        let json = generate_json_report(&result);
        // Vérification basique que c'est du JSON valide
        assert!(json.starts_with('{'));
        assert!(json.trim_end().ends_with('}'));
        assert!(json.contains("\"model_name\": \"test_model\""));
    }

    #[test]
    fn generate_console_summary_format() {
        let result = create_test_result();
        let summary = generate_console_summary(&result);
        assert!(summary.contains("2 tenseurs"));
        assert!(summary.contains("1 issues"));
    }
}
