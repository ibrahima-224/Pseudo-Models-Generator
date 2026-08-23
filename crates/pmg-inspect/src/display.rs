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

//! Formatage CLI pour les rapports d'inspection.
//!
//! Ce module fournit des fonctions pour formater les rapports d'inspection
//! selon le niveau de détail souhaité (brief, normal, verbose, debug) et
//! en différents formats (texte, JSON).
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::display::format_report;
//! use pmg_inspect::inspector::{InspectionReport, InspectionLevel};
//!
//! // Création d'un rapport fictif (non illustré ici)
//! // let report = InspectionReport::new(...);
//! // let output = format_report(&report, InspectionLevel::Normal, false);
//! // println!("{}", output);
//! ```

use crate::inspector::{InspectionLevel, InspectionReport};

/// Formate un rapport d'inspection en texte selon le niveau de détail.
///
/// # Paramètres
/// - `report` : rapport d'inspection.
/// - `level` : niveau de détail souhaité.
/// - `json` : si true, retourne le format JSON au lieu du texte.
pub fn format_report(report: &InspectionReport, level: InspectionLevel, json: bool) -> String {
    if json {
        format_report_json(report)
    } else {
        match level {
            InspectionLevel::Brief => format_brief(report),
            InspectionLevel::Normal => format_normal(report),
            InspectionLevel::Verbose => format_verbose(report),
            InspectionLevel::Debug => format_debug(report),
        }
    }
}

/// Formate le rapport en JSON.
fn format_report_json(report: &InspectionReport) -> String {
    // Conversion en structure JSON sérialisable
    let json_value = serde_json::json!({
        "model_path": report.model_path,
        "level": format!("{:?}", report.level),
        "config": report.config.as_ref().map(|c| {
            serde_json::json!({
                "model_type": c.model_type,
                "architectures": c.architectures,
                "hidden_size": c.hidden_size,
                "num_layers": c.num_layers,
                "num_attention_heads": c.num_attention_heads,
                "num_key_value_heads": c.num_key_value_heads,
                "intermediate_size": c.intermediate_size,
                "vocab_size": c.vocab_size,
                "dtype": format!("{:?}", c.dtype),
                "attention_type": format!("{:?}", c.attention_type),
                "max_position_embeddings": c.max_position_embeddings,
                "moe": c.moe.as_ref().map(|m| serde_json::json!({
                    "n_routed_experts": m.n_routed_experts,
                    "n_shared_experts": m.n_shared_experts,
                    "experts_per_tok": m.experts_per_tok,
                    "router_dtype": format!("{:?}", m.router_dtype),
                    "routed_scaling_factor": m.routed_scaling_factor,
                    "norm_topk_prob": m.norm_topk_prob,
                    "topk_method": m.topk_method,
                    "first_k_dense_replace": m.first_k_dense_replace,
                    "layer_types": m.layer_types,
                    "expert_dtype": m.expert_dtype.map(|d| format!("{:?}", d)),
                })),
            })
        }),
        "safetensors_headers": report.safetensors_headers.iter().map(|h| {
            serde_json::json!({
                "file_path": h.file_path,
                "tensor_count": h.tensor_count(),
                "total_bytes": h.total_bytes(),
                "file_size": h.file_size,
                "header_size": h.header_size,
            })
        }).collect::<Vec<_>>(),
        "shard_index": report.shard_index.as_ref().map(|idx| {
            serde_json::json!({
                "total_tensors": idx.total_tensors(),
                "shard_count": idx.shard_count(),
                "shards": idx.all_shard_paths().iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
            })
        }),
        "structural": {
            "total_tensors": report.structural.total_tensors,
            "num_layers": report.structural.num_layers,
            "num_shards": report.structural.num_shards,
            "num_experts": report.structural.num_experts,
            "total_parameters": report.structural.total_parameters,
            "total_elements": report.structural.total_elements,
            "dimensions": report.structural.dimensions,
            "dtypes": report.structural.dtypes.iter().map(|d| format!("{:?}", d)).collect::<Vec<_>>(),
        },
        "physical": {
            "total_memory_bytes": report.physical.total_memory_bytes,
            "total_file_size": report.physical.total_file_size,
            "theoretical_size_bytes": report.physical.theoretical_size_bytes,
            "primary_dtype": report.physical.primary_dtype.map(|d| format!("{:?}", d)),
            "average_density": report.physical.average_density,
            "total_parameters": report.physical.total_parameters,
            "average_bytes_per_parameter": report.physical.average_bytes_per_parameter,
        },
        "architecture": {
            "architecture_type": report.architecture.architecture_type,
            "attention_type": report.architecture.attention_type,
            "num_layers": report.architecture.num_layers,
            "hidden_size": report.architecture.hidden_size,
            "num_attention_heads": report.architecture.num_attention_heads,
            "num_key_value_heads": report.architecture.num_key_value_heads,
            "intermediate_size": report.architecture.intermediate_size,
            "vocab_size": report.architecture.vocab_size,
            "primary_dtype": report.architecture.primary_dtype,
            "total_parameters": report.architecture.total_parameters,
            "has_moe": report.architecture.has_moe,
            "num_experts": report.architecture.num_experts,
            "head_dim": report.architecture.head_dim,
        },
    });

    serde_json::to_string_pretty(&json_value).unwrap_or_else(|_| "{}".to_string())
}

/// Formate en mode brief (résumé).
fn format_brief(report: &InspectionReport) -> String {
    let mut output = String::new();
    output.push_str("=== Inspection bref du modèle ===\n");
    output.push_str(&format!("Chemin : {}\n", report.model_path.display()));
    output.push_str(&format!("{}\n", report.architecture));
    output.push_str(&format!(
        "Paramètres : {}\n",
        format_number(report.structural.total_parameters)
    ));
    output.push_str(&format!(
        "Mémoire   : {}\n",
        format_bytes(report.physical.total_memory_bytes)
    ));
    output
}

/// Formate en mode normal.
fn format_normal(report: &InspectionReport) -> String {
    let mut output = String::new();
    output.push_str("=== Rapport d'inspection du modèle ===\n");
    output.push_str(&format!("Chemin : {}\n", report.model_path.display()));
    output.push('\n');

    // Configuration
    if let Some(ref config) = report.config {
        output.push_str("--- Configuration ---\n");
        output.push_str(&format!("{}\n", config));
    }

    // Architecture
    output.push('\n');
    output.push_str("--- Architecture ---\n");
    output.push_str(&format!("{}\n", report.architecture));

    // Statistiques structurelles
    output.push('\n');
    output.push_str("--- Statistiques structurelles ---\n");
    output.push_str(&format!("{}\n", report.structural));

    // Statistiques physiques
    output.push('\n');
    output.push_str("--- Statistiques physiques ---\n");
    output.push_str(&format!("{}\n", report.physical));

    output
}

/// Formate en mode verbeux.
fn format_verbose(report: &InspectionReport) -> String {
    let mut output = String::new();
    // D'abord le format normal
    output.push_str(&format_normal(report));

    // Puis les détails supplémentaires
    output.push('\n');
    output.push_str("--- Détails Safetensors ---\n");
    for (i, header) in report.safetensors_headers.iter().enumerate() {
        output.push_str(&format!(
            "Shard {}: {}\n",
            i + 1,
            header.file_path.display()
        ));
        output.push_str(&format!("  Tenseurs : {}\n", header.tensor_count()));
        output.push_str(&format!(
            "  Taille  : {}\n",
            format_bytes(header.total_bytes())
        ));
    }

    if let Some(ref index) = report.shard_index {
        output.push('\n');
        output.push_str("--- Index des shards ---\n");
        output.push_str(&format!("{}\n", index));
    }

    output
}

/// Formate en mode debug.
fn format_debug(report: &InspectionReport) -> String {
    let mut output = String::new();
    // D'abord le format verbeux
    output.push_str(&format_verbose(report));

    // Puis les informations de débogage
    output.push('\n');
    output.push_str("--- Informations de débogage ---\n");
    output.push_str(&format!(
        "Nombre de shards : {}\n",
        report.safetensors_headers.len()
    ));
    output.push_str(&format!(
        "Nombre total de tenseurs : {}\n",
        report.structural.total_tensors
    ));
    output.push_str(&format!(
        "Densité moyenne : {:.4}\n",
        report.physical.average_density
    ));

    // Détails de chaque tenseur (premiers 10)
    if !report.safetensors_headers.is_empty() {
        output.push('\n');
        output.push_str("Premiers tenseurs (max 10) :\n");
        let mut count = 0;
        for header in &report.safetensors_headers {
            for tensor in header.tensors.iter().take(10 - count) {
                output.push_str(&format!(
                    "  {} : {:?} {:?} ({})\n",
                    tensor.name,
                    tensor.shape,
                    tensor.dtype,
                    format_bytes(tensor.size_bytes())
                ));
                count += 1;
                if count >= 10 {
                    break;
                }
            }
            if count >= 10 {
                break;
            }
        }
    }

    output
}

// ============================================================================
// Fonctions utilitaires de formatage
// ============================================================================

/// Formate un nombre avec des séparateurs de milliers.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Formate une taille en octets en unité lisible.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TiB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{} o", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inspector::InspectionReport;
    use std::path::PathBuf;

    #[test]
    fn test_format_report_text() {
        let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Normal);
        let output = format_report(&report, InspectionLevel::Normal, false);
        assert!(output.contains("=== Rapport d'inspection du modèle ==="));
        assert!(output.contains("Chemin : /fake/model"));
    }

    #[test]
    fn test_format_report_json() {
        let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Normal);
        let output = format_report(&report, InspectionLevel::Normal, true);
        // Vérifie que c'est du JSON valide
        assert!(serde_json::from_str::<serde_json::Value>(&output).is_ok());
        assert!(output.contains("model_path"));
        assert!(output.contains("/fake/model"));
    }

    #[test]
    fn test_format_brief() {
        let report = InspectionReport::new(PathBuf::from("/fake/model"), InspectionLevel::Brief);
        let output = format_brief(&report);
        assert!(output.contains("=== Inspection bref du modèle ==="));
        assert!(output.contains("Paramètres : 0"));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1_234");
        assert_eq!(format_number(1234567), "1_234_567");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 o");
        assert_eq!(format_bytes(1023), "1023 o");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1048576), "1.00 MiB");
        assert_eq!(format_bytes(1073741824), "1.00 GiB");
        assert_eq!(format_bytes(1099511627776), "1.00 TiB");
    }
}
