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

//! Tests d'inspection principaux pour la crate `pmg-inspect`.
//!
//! Ces tests valident le fonctionnement de base de l'inspection des modèles.
//! Les tests d'intégration complets sont dans `inspection_integration_tests`.

use pmg_core::dtype::DType;
use pmg_inspect::config_inspector::inspect_config;
use pmg_inspect::index_inspector::build_shard_index;
use pmg_inspect::physical_stats::compute_physical_stats;
use pmg_inspect::safetensors_inspector::inspect_safetensors_headers;
use pmg_inspect::structural_stats::compute_structural_stats;
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

/// Teste l'inspection de la configuration.
#[test]
fn test_config_inspection() {
    let (_temp_dir, model_path) = create_test_model_dir();

    let result = inspect_config(&model_path);
    assert!(result.is_ok());

    let config = result.unwrap();
    assert_eq!(config.model_type, "test_model");
    assert_eq!(config.architectures, vec!["TestModelForCausalLM"]);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_layers, 12);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 16);
    assert_eq!(config.intermediate_size, Some(4096));
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.dtype, DType::Bf16);
    assert_eq!(config.max_position_embeddings, 2048);
    assert!(config.moe.is_some());

    let moe = config.moe.unwrap();
    assert_eq!(moe.n_routed_experts, 8);
    assert_eq!(moe.experts_per_tok, 2);
    assert_eq!(moe.n_shared_experts, 1);
}

/// Teste l'inspection des headers Safetensors.
#[test]
fn test_header_inspection() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path();

    // Création d'un fichier Safetensors fictif avec un header valide
    let header_json = r#"{
        "model.layers.0.weight": {
            "dtype": "F32",
            "shape": [100, 100],
            "data_offsets": [0, 40000]
        },
        "model.layers.1.weight": {
            "dtype": "F32",
            "shape": [100, 100],
            "data_offsets": [40000, 80000]
        }
    }"#;

    // Écriture du fichier Safetensors (header uniquement)
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    // Ajout de données factices pour atteindre la taille déclarée
    file_content.resize(8 + 80000, 0);

    let safetensors_path = model_path.join("model.safetensors");
    std::fs::write(&safetensors_path, &file_content).unwrap();

    let result = inspect_safetensors_headers(model_path);
    assert!(result.is_ok());

    let headers = result.unwrap();
    assert_eq!(headers.len(), 1);

    let header = &headers[0];
    assert_eq!(header.tensor_count(), 2);
    assert_eq!(header.file_size, file_content.len() as u64);

    // Vérification des tenseurs
    let tensor0 = &header.tensors[0];
    assert_eq!(tensor0.name, "model.layers.0.weight");
    assert_eq!(tensor0.dtype, DType::F32);
    assert_eq!(tensor0.shape.dims(), &[100, 100]);
    assert_eq!(tensor0.data_offsets, [0, 40000]);

    let tensor1 = &header.tensors[1];
    assert_eq!(tensor1.name, "model.layers.1.weight");
    assert_eq!(tensor1.dtype, DType::F32);
    assert_eq!(tensor1.shape.dims(), &[100, 100]);
    assert_eq!(tensor1.data_offsets, [40000, 80000]);
}

/// Teste la construction de l'index des shards.
#[test]
fn test_shard_index() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path();

    // Création de deux fichiers Safetensors fictifs
    let test_cases = vec![
        (
            "shard1.safetensors",
            vec!["tensor1", "tensor2"],
            r#"{"tensor1": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 400]}, "tensor2": {"dtype": "F32", "shape": [10, 10], "data_offsets": [400, 800]}}"#,
        ),
        (
            "shard2.safetensors",
            vec!["tensor3"],
            r#"{"tensor3": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 400]}}"#,
        ),
    ];

    for (file_name, _tensor_names, header_json) in &test_cases {
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut file_content = Vec::new();
        file_content.extend_from_slice(&header_len.to_le_bytes());
        file_content.extend_from_slice(header_bytes);
        file_content.resize(8 + 800, 0);

        std::fs::write(model_path.join(file_name), &file_content).unwrap();
    }

    let headers = inspect_safetensors_headers(model_path).unwrap();
    let index = build_shard_index(model_path, &headers).unwrap();

    assert_eq!(index.total_tensors(), 3);
    assert_eq!(index.shard_count(), 2);

    // Vérification du mapping tensor → shard
    let shard1 = index.shard_for_tensor("tensor1").unwrap();
    assert!(shard1.to_string_lossy().contains("shard1.safetensors"));

    let shard2 = index.shard_for_tensor("tensor3").unwrap();
    assert!(shard2.to_string_lossy().contains("shard2.safetensors"));

    // Vérification du mapping shard → tenseurs
    let tensors_shard1 = index.tensors_in_shard(shard1).unwrap();
    assert_eq!(tensors_shard1.len(), 2);
    assert!(tensors_shard1.contains(&"tensor1".to_string()));
    assert!(tensors_shard1.contains(&"tensor2".to_string()));
}

/// Teste le comptage des tenseurs.
#[test]
fn test_tensor_count() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path();

    // Création d'un fichier Safetensors avec 5 tenseurs
    let mut header_json = "{".to_string();
    for i in 0..5 {
        if i > 0 {
            header_json.push(',');
        }
        header_json.push_str(&format!(
            r#""tensor{}": {{"dtype": "F32", "shape": [10, 10], "data_offsets": [{}, {}]}}"#,
            i,
            i * 400,
            (i + 1) * 400
        ));
    }
    header_json.push('}');

    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    file_content.resize(8 + 2000, 0);

    std::fs::write(model_path.join("model.safetensors"), &file_content).unwrap();

    let headers = inspect_safetensors_headers(model_path).unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].tensor_count(), 5);
}

/// Teste l'estimation des paramètres.
#[test]
fn test_parameter_estimation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path();

    // Création d'un fichier Safetensors avec des tenseurs de tailles variées
    let header_json = r#"{
        "layer1": {"dtype": "F32", "shape": [100, 100], "data_offsets": [0, 40000]},
        "layer2": {"dtype": "F16", "shape": [200, 200], "data_offsets": [40000, 120000]},
        "layer3": {"dtype": "BF16", "shape": [50, 50], "data_offsets": [120000, 125000]}
    }"#;

    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    file_content.resize(8 + 125000, 0);

    std::fs::write(model_path.join("model.safetensors"), &file_content).unwrap();

    let headers = inspect_safetensors_headers(model_path).unwrap();
    let structural = compute_structural_stats(&None, &headers, &None);
    let physical = compute_physical_stats(&headers, &structural);

    // Vérification du nombre de paramètres
    assert_eq!(structural.total_tensors, 3);
    assert_eq!(structural.total_parameters, 100 * 100 + 200 * 200 + 50 * 50);

    // Vérification de la mémoire estimée
    assert_eq!(
        physical.total_memory_bytes,
        100 * 100 * 4 + // F32: 4 octets
        200 * 200 * 2 + // F16: 2 octets
        50 * 50 * 2 // BF16: 2 octets
    );
}

/// Teste le principe Zero-Payload : aucune donnée de poids n'est chargée.
#[test]
fn test_no_weight_loading() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path();

    // Création d'un fichier Safetensors avec un gros tenseur (simulé)
    // On utilise des offsets réalistes pour éviter les erreurs de lecture
    let header_json = r#"{
        "big_tensor": {"dtype": "F32", "shape": [100, 100], "data_offsets": [0, 40000]}
    }"#;

    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_content = Vec::new();
    file_content.extend_from_slice(&header_len.to_le_bytes());
    file_content.extend_from_slice(header_bytes);
    // Simulation d'un fichier avec des données
    file_content.resize(8 + 40000, 0);

    std::fs::write(model_path.join("model.safetensors"), &file_content).unwrap();

    // Création d'un config.json minimal pour l'inspection
    let config_json = r#"{
        "model_type": "test_model",
        "hidden_size": 100,
        "num_hidden_layers": 1
    }"#;
    std::fs::write(model_path.join("config.json"), config_json).unwrap();

    // Mesure de la mémoire avant et après l'inspection
    let memory_before = get_memory_usage();

    // Inspection du modèle
    let inspector = pmg_inspect::inspector::ModelInspector::new(model_path)
        .with_level(pmg_inspect::inspector::InspectionLevel::Debug);
    let result = inspector.inspect();
    assert!(result.is_ok(), "L'inspection a échoué : {:?}", result.err());

    let report = result.unwrap();

    // Vérification que l'inspection a fonctionné
    assert_eq!(report.structural.total_tensors, 1);
    assert_eq!(report.structural.total_parameters, 100 * 100);
    assert_eq!(report.physical.total_memory_bytes, 100 * 100 * 4);

    let memory_after = get_memory_usage();

    // Vérification que la mémoire utilisée est raisonnable
    // (pas plus de 10 Mo supplémentaires pour un header de quelques octets)
    let memory_diff = memory_after.saturating_sub(memory_before);
    assert!(
        memory_diff < 10 * 1024 * 1024,
        "L'inspection a utilisé trop de mémoire : {} octets",
        memory_diff
    );
}

/// Obtient l'utilisation mémoire actuelle (approximative).
fn get_memory_usage() -> u64 {
    // Sur Linux, on peut lire /proc/self/status
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(value) = line.strip_prefix("VmRSS:") {
                    if let Some(kb) = value.trim().strip_suffix(" kB") {
                        if let Ok(kb) = kb.parse::<u64>() {
                            return kb * 1024; // Conversion en octets
                        }
                    }
                }
            }
        }
    }
    // Fallback : retourne 0 si on ne peut pas mesurer
    0
}
