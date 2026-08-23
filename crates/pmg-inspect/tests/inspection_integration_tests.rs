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

//! Tests d'intégration pour la crate `pmg-inspect`.
//!
//! Ces tests valident le fonctionnement complet de l'inspection des modèles
//! sans charger les poids (principe Zero-Payload).

use pmg_inspect::inspector::{InspectionLevel, ModelInspector};
use std::path::PathBuf;

/// Crée un répertoire de test avec un fichier config.json minimal.
fn create_test_model_dir() -> (tempfile::TempDir, PathBuf) {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().to_path_buf();

    // Création du fichier config.json
    let config_json = r#"{
        "model_type": "test_model",
        "architectures": ["TestModelForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 2048,
        "moe": {
            "num_experts": 8,
            "top_k": 2,
            "n_shared_experts": 1,
            "routed_scaling_factor": 2.0,
            "norm_topk_prob": true,
            "topk_method": "noaux_tc",
            "layer_types": ["dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"]
        }
    }"#;
    std::fs::write(model_path.join("config.json"), config_json).unwrap();

    (temp_dir, model_path)
}

/// Teste l'inspection complète d'un modèle avec tous les modules.
#[test]
fn test_full_inspection() {
    let (_temp_dir, model_path) = create_test_model_dir();

    // Création d'un fichier Safetensors
    let header_json = r#"{
        "model.layers.0.weight": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 2097152]},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 1024], "data_offsets": [2097152, 67108864]}
    }"#;

    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    file_content.resize(8 + 67108864, 0);

    std::fs::write(model_path.join("model.safetensors"), &file_content).unwrap();

    // Inspection complète
    let inspector = ModelInspector::new(&model_path);
    let result = inspector.inspect();

    assert!(result.is_ok());

    let report = result.unwrap();

    // Vérification de la configuration
    assert!(report.config.is_some());
    let config = report.config.unwrap();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_layers, 12);

    // Vérification des headers Safetensors
    assert_eq!(report.safetensors_headers.len(), 1);
    assert_eq!(report.safetensors_headers[0].tensor_count(), 2);

    // Vérification de l'index
    assert!(report.shard_index.is_some());
    let index = report.shard_index.unwrap();
    assert_eq!(index.total_tensors(), 2);

    // Vérification des statistiques structurelles
    assert_eq!(report.structural.total_tensors, 2);
    assert_eq!(
        report.structural.total_parameters,
        1024 * 1024 + 32000 * 1024
    );

    // Vérification des statistiques physiques
    assert_eq!(
        report.physical.total_memory_bytes,
        (1024 * 1024 + 32000 * 1024) * 2 // BF16 = 2 octets
    );

    // Vérification du résumé architectural
    assert_eq!(report.architecture.hidden_size, 1024);
    assert_eq!(report.architecture.num_layers, 12);
    assert_eq!(report.architecture.vocab_size, 32000);
}

/// Teste la génération de rapports structurés en JSON.
#[test]
fn test_structured_report_json() {
    let (_temp_dir, model_path) = create_test_model_dir();

    // Création d'un fichier Safetensors
    let header_json = r#"{
        "model.layers.0.weight": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 2097152]},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32000, 1024], "data_offsets": [2097152, 67108864]}
    }"#;

    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    file_content.resize(8 + 67108864, 0);

    std::fs::write(model_path.join("model.safetensors"), &file_content).unwrap();

    // Inspection complète
    let inspector = ModelInspector::new(&model_path);
    let report = inspector.inspect().unwrap();

    // Conversion en rapport structuré
    let structured = pmg_inspect::StructuredReport::from_inspection_report(&report);

    // Vérification que le JSON est valide
    let json = structured.to_json();
    assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());

    // Vérification des champs essentiels
    let json_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(json_value["model_path"], model_path.to_str().unwrap());
    assert!(json_value["config"].is_object());
    assert!(json_value["structural"].is_object());
    assert!(json_value["physical"].is_object());
    assert!(json_value["architecture"].is_object());

    // Vérification du JSON compact
    let json_compact = structured.to_json_compact();
    assert!(serde_json::from_str::<serde_json::Value>(&json_compact).is_ok());
}

/// Teste les différents niveaux de détail en mode texte.
#[test]
fn test_text_report_levels() {
    let (_temp_dir, model_path) = create_test_model_dir();

    // Création d'un fichier Safetensors minimal
    let header_json = r#"{
        "model.layers.0.weight": {"dtype": "F32", "shape": [100, 100], "data_offsets": [0, 40000]}
    }"#;

    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    file_content.resize(8 + 40000, 0);

    std::fs::write(model_path.join("model.safetensors"), &file_content).unwrap();

    // Test de chaque niveau de détail
    for level in [
        InspectionLevel::Brief,
        InspectionLevel::Normal,
        InspectionLevel::Verbose,
        InspectionLevel::Debug,
    ] {
        let inspector = ModelInspector::new(&model_path).with_level(level);
        let report = inspector.inspect().unwrap();
        let structured = pmg_inspect::StructuredReport::from_inspection_report(&report);

        let text_output = structured.to_text(level);
        assert!(
            !text_output.is_empty(),
            "Le rapport texte ne doit pas être vide pour le niveau {:?}",
            level
        );

        // Vérifications spécifiques par niveau
        match level {
            InspectionLevel::Brief => {
                assert!(text_output.contains("=== Inspection bref du modèle ==="));
            },
            InspectionLevel::Normal => {
                assert!(text_output.contains("=== Rapport d'inspection du modèle ==="));
                assert!(text_output.contains("--- Configuration ---"));
                assert!(text_output.contains("--- Architecture ---"));
            },
            InspectionLevel::Verbose => {
                assert!(text_output.contains("=== Rapport d'inspection du modèle ==="));
                assert!(text_output.contains("--- Détails Safetensors ---"));
            },
            InspectionLevel::Debug => {
                assert!(text_output.contains("--- Informations de débogage ---"));
                assert!(text_output.contains("Nombre de shards"));
            },
        }
    }
}

/// Teste la conversion des types internes vers les types JSON.
#[test]
fn test_json_type_conversions() {
    use pmg_inspect::report::{ConfigInspectionJson, StructuralStatsJson};

    // Test de la conversion ConfigInspection -> ConfigInspectionJson
    let config_inspection = pmg_inspect::config_inspector::ConfigInspection {
        config_path: std::path::PathBuf::from("/fake/config.json"),
        model_type: "test_model".to_string(),
        architectures: vec!["TestModel".to_string()],
        hidden_size: 1024,
        num_layers: 12,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        intermediate_size: Some(4096),
        vocab_size: 32000,
        dtype: pmg_core::DType::F32,
        attention_type: pmg_core::model_config::AttentionKind::Dense,
        max_position_embeddings: 2048,
        moe: None,
        provenance: std::collections::BTreeMap::new(),
    };

    let config_json = ConfigInspectionJson::from(&config_inspection);
    assert_eq!(config_json.model_type, "test_model");
    assert_eq!(config_json.hidden_size, 1024);
    assert_eq!(config_json.num_layers, 12);
    assert_eq!(config_json.dtype, "F32");

    // Test de la conversion StructuralStats -> StructuralStatsJson
    let structural = pmg_inspect::structural_stats::StructuralStats {
        total_tensors: 100,
        num_layers: 12,
        num_shards: 2,
        num_experts: 0,
        total_parameters: 1000000,
        total_elements: 1000000,
        dimensions: std::collections::BTreeMap::new(),
        dtypes: vec![pmg_core::DType::F32],
        tensors_per_layer: std::collections::BTreeMap::new(),
        tensors_by_role: std::collections::BTreeMap::new(),
    };

    let structural_json = StructuralStatsJson::from(&structural);
    assert_eq!(structural_json.total_tensors, 100);
    assert_eq!(structural_json.num_layers, 12);
    assert_eq!(structural_json.total_parameters, 1000000);
    assert!(structural_json.dtypes.contains(&"F32".to_string()));
}
