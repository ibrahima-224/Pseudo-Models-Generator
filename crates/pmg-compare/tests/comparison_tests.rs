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

//! Tests d'intégration pour la comparaison de modèles.
//!
//! Ce fichier contient des tests pour vérifier le bon fonctionnement
//! de la comparaison metadata-only de modèles.

use pmg_compare::architecture_compare::{compare_architectures, ArchitectureType};
use pmg_compare::comparison::{ComparisonReport, ComparisonStatus};
use pmg_compare::config_compare::{compare_configs, ConfigValue, ModelConfig};
use pmg_compare::dtype_compare::{compare_dtypes, DtypeInfo};
use pmg_compare::report::{format_compact_report, format_report};
use pmg_compare::score::{calculate_global_score, ComparisonScore};
use pmg_compare::shape_compare::{compare_shapes, ShapeInfo};
use pmg_compare::shard_compare::{compare_sharding, ShardConfig, ShardInfo};
use pmg_compare::tensor_compare::{compare_tensors, TensorInfo};

/// Test de comparaison de configurations identiques.
#[test]
fn test_config_comparison_identical() {
    let config = ModelConfig {
        name: "test_model".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ("num_layers".to_string(), ConfigValue::Integer(32)),
            ("num_heads".to_string(), ConfigValue::Integer(32)),
            ("num_experts".to_string(), ConfigValue::Integer(8)),
            ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
        ],
    };

    let result = compare_configs(&config, &config);
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
    assert_eq!(result.status, ComparisonStatus::Match);
}

/// Test de comparaison de configurations différentes.
#[test]
fn test_config_comparison_different() {
    let original = ModelConfig {
        name: "model_a".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
        ],
    };

    let compared = ModelConfig {
        name: "model_b".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(8192)),
        ],
    };

    let result = compare_configs(&original, &compared);
    // 6 paramètres architecturaux, 1 différent → 5/6 = 0.833...
    assert!((result.similarity_score - 0.8333333333333334).abs() < 1e-10);
    assert_eq!(result.differences.len(), 1);
    assert_eq!(result.status, ComparisonStatus::Partial);
}

/// Test de comparaison d'architectures identiques.
#[test]
fn test_architecture_comparison_identical() {
    let config = ModelConfig {
        name: "test_model".to_string(),
        parameters: vec![
            ("num_layers".to_string(), ConfigValue::Integer(32)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ("num_heads".to_string(), ConfigValue::Integer(32)),
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("num_experts".to_string(), ConfigValue::Integer(8)),
            ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
        ],
    };

    let result = compare_architectures(&config, &config);
    assert_eq!(result.architecture_type, ArchitectureType::Identical);
    assert_eq!(result.compatibility_score, 1.0);
    assert!(result.differences.is_empty());
}

/// Test de comparaison de tenseurs identiques.
#[test]
fn test_tensor_comparison_identical() {
    let tensors = vec![
        TensorInfo::new("layer1.weight".to_string()),
        TensorInfo::new("layer1.bias".to_string()),
    ];

    let result = compare_tensors(&tensors, &tensors);
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
    assert_eq!(result.status, ComparisonStatus::Match);
}

/// Test de comparaison de tenseurs différents.
#[test]
fn test_tensor_comparison_different() {
    let original = vec![
        TensorInfo::new("layer1.weight".to_string()),
        TensorInfo::new("layer1.bias".to_string()),
    ];

    let compared = vec![TensorInfo::new("layer1.weight".to_string())];

    let result = compare_tensors(&original, &compared);
    assert_eq!(result.similarity_score, 0.5);
    assert_eq!(result.differences.len(), 1);
    assert_eq!(result.status, ComparisonStatus::Different);
}

/// Test de comparaison de shapes identiques.
#[test]
fn test_shape_comparison_identical() {
    let shapes = vec![
        ShapeInfo::new("layer1.weight".to_string(), vec![100, 200]),
        ShapeInfo::new("layer1.bias".to_string(), vec![200]),
    ];

    let result = compare_shapes(&shapes, &shapes);
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
    assert_eq!(result.status, ComparisonStatus::Match);
}

/// Test de comparaison de dtypes identiques.
#[test]
fn test_dtype_comparison_identical() {
    let dtypes = vec![
        DtypeInfo::new("layer1.weight".to_string(), "float32".to_string()),
        DtypeInfo::new("layer1.bias".to_string(), "float32".to_string()),
    ];

    let result = compare_dtypes(&dtypes, &dtypes);
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
    assert_eq!(result.status, ComparisonStatus::Match);
}

/// Test de comparaison de sharding identique.
#[test]
fn test_shard_comparison_identical() {
    let config = ShardConfig::new(
        2,
        vec![
            ShardInfo::new("layer1.weight".to_string(), 0, 1000),
            ShardInfo::new("layer1.bias".to_string(), 1, 500),
        ],
    );

    let result = compare_sharding(&config, &config);
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
    assert_eq!(result.status, ComparisonStatus::Match);
}

/// Test de calcul de score global.
#[test]
fn test_global_score_calculation() {
    let score = calculate_global_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0);
    assert!(score.is_perfect());
    assert!(!score.has_blocking_anomalies());
    assert_eq!(score.percentage, 100.0);
}

/// Test de calcul de score global avec anomalies.
#[test]
fn test_global_score_with_anomalies() {
    let score = calculate_global_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2);
    assert!(!score.is_perfect());
    assert!(score.has_blocking_anomalies());
    assert_eq!(score.blocking_anomalies, 2);
}

/// Test de formatage de rapport.
#[test]
fn test_report_formatting() {
    let config_result = pmg_compare::config_compare::ConfigComparisonResult::default();
    let architecture_result =
        pmg_compare::architecture_compare::ArchitectureComparisonResult::default();
    let tensor_result = pmg_compare::tensor_compare::TensorComparisonResult::default();
    let shape_result = pmg_compare::shape_compare::ShapeComparisonResult::default();
    let dtype_result = pmg_compare::dtype_compare::DtypeComparisonResult::default();
    let shard_result = pmg_compare::shard_compare::ShardComparisonResult::default();

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

    let formatted = format_report(&report);
    assert!(formatted.contains("RAPPORT DE COMPARAISON"));
    assert!(formatted.contains("model_a"));
    assert!(formatted.contains("model_b"));
    assert!(formatted.contains("Metadata-only"));
    assert!(formatted.contains("Aucune lecture profonde des poids"));

    let compact = format_compact_report(&report);
    assert!(compact.contains("model_a vs model_b"));
    assert!(compact.contains("metadata-only"));
}

/// Test de comparaison complète avec les nouveaux modules.
#[test]
fn test_comparator_metadata_only() {
    let config_original = ModelConfig {
        name: "model_a".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ("num_layers".to_string(), ConfigValue::Integer(32)),
            ("num_heads".to_string(), ConfigValue::Integer(32)),
            ("num_experts".to_string(), ConfigValue::Integer(8)),
            ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
        ],
    };

    let config_compared = ModelConfig {
        name: "model_b".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ("num_layers".to_string(), ConfigValue::Integer(32)),
            ("num_heads".to_string(), ConfigValue::Integer(32)),
            ("num_experts".to_string(), ConfigValue::Integer(8)),
            ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
        ],
    };

    let tensor_infos = vec![
        TensorInfo::new("layer1.weight".to_string()),
        TensorInfo::new("layer1.bias".to_string()),
    ];

    let shape_infos = vec![
        ShapeInfo::new("layer1.weight".to_string(), vec![100, 200]),
        ShapeInfo::new("layer1.bias".to_string(), vec![200]),
    ];

    let shard_config = ShardConfig::new(1, vec![]);

    // Comparer les métadonnées individuellement
    let config_result = compare_configs(&config_original, &config_compared);
    let architecture_result = compare_architectures(&config_original, &config_compared);
    let tensor_result = compare_tensors(&tensor_infos, &tensor_infos);
    let shape_result = compare_shapes(&shape_infos, &shape_infos);
    let dtype_result = compare_dtypes(&[], &[]); // pas de dtypes dans ce test
    let shard_result = compare_sharding(&shard_config, &shard_config);

    // Calculer le score global
    let global_score = calculate_global_score(
        config_result.similarity_score,
        architecture_result.compatibility_score,
        tensor_result.similarity_score,
        shape_result.similarity_score,
        dtype_result.similarity_score,
        shard_result.similarity_score,
        0,
    );

    // Créer le rapport
    let report = ComparisonReport::new(
        "model_a".to_string(),
        "model_b".to_string(),
        config_result,
        architecture_result,
        tensor_result,
        shape_result,
        dtype_result,
        shard_result,
        global_score,
        ComparisonStatus::Match,
        vec![],
    );

    assert_eq!(report.global_status, ComparisonStatus::Match);
    assert!(report.global_score.is_perfect());
    assert!(!report.has_blocking_anomalies());
    assert!(report.metadata_only);

    // Vérifier que le rapport peut être formaté
    let formatted = format_report(&report);
    assert!(!formatted.is_empty());
}

/// Test de comparaison avec anomalies bloquantes.
#[test]
fn test_comparison_with_blocking_anomalies() {
    let config_original = ModelConfig {
        name: "model_a".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
        ],
    };

    let config_compared = ModelConfig {
        name: "model_b".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            // hidden_size manquant - anomalie bloquante
        ],
    };

    let (result, anomalies) = pmg_compare::config_compare::compare_configs_with_anomalies(
        &config_original,
        &config_compared,
    );

    // 6 paramètres architecturaux, 1 manquant → 5/6 = 0.833...
    assert!((result.similarity_score - 0.8333333333333334).abs() < 1e-10);
    assert!(!anomalies.is_empty());
    assert!(anomalies[0].contains("hidden_size"));
}

/// Test de comparaison de modèles vides (aucun paramètre, aucun tenseur).
#[test]
fn test_empty_models_comparison() {
    let config_original = ModelConfig {
        name: "empty_model_a".to_string(),
        parameters: vec![],
    };

    let config_compared = ModelConfig {
        name: "empty_model_b".to_string(),
        parameters: vec![],
    };

    let result = compare_configs(&config_original, &config_compared);
    // Aucun paramètre à comparer → score parfait
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
    assert_eq!(result.status, ComparisonStatus::Match);
}

/// Test de comparaison de modèles avec des structures très différentes.
#[test]
fn test_very_different_models_comparison() {
    let config_original = ModelConfig {
        name: "model_a".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ("num_layers".to_string(), ConfigValue::Integer(32)),
            ("num_heads".to_string(), ConfigValue::Integer(32)),
            ("num_experts".to_string(), ConfigValue::Integer(8)),
            ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
        ],
    };

    let config_compared = ModelConfig {
        name: "model_b".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(100000)),
            ("hidden_size".to_string(), ConfigValue::Integer(8192)),
            ("num_layers".to_string(), ConfigValue::Integer(64)),
            ("num_heads".to_string(), ConfigValue::Integer(64)),
            ("num_experts".to_string(), ConfigValue::Integer(16)),
            ("intermediate_size".to_string(), ConfigValue::Integer(32768)),
        ],
    };

    let result = compare_configs(&config_original, &config_compared);
    // Tous les paramètres sont différents → score 0.0
    assert_eq!(result.similarity_score, 0.0);
    assert_eq!(result.differences.len(), 6);
    assert_eq!(result.status, ComparisonStatus::Different);
}

/// Test de comparaison de modèles de tailles très différentes (beaucoup plus de tenseurs dans un modèle).
#[test]
fn test_different_size_models_comparison() {
    // Créer deux ensembles de tenseurs avec des tailles très différentes
    let mut tensors_original = Vec::new();
    let mut tensors_compared = Vec::new();

    // Modèle original : 100 tenseurs
    for i in 0..100 {
        tensors_original.push(TensorInfo::new(format!("layer_{}.weight", i)));
    }

    // Modèle comparé : seulement 5 tenseurs (très différents)
    for i in 0..5 {
        tensors_compared.push(TensorInfo::new(format!("layer_{}.weight", i)));
    }

    let result = compare_tensors(&tensors_original, &tensors_compared);

    // Seulement 5 tenseurs en commun sur 100 → score faible
    assert!(result.similarity_score < 0.1);
    assert_eq!(result.status, ComparisonStatus::Different);

    // Vérifier les compteurs
    assert_eq!(result.total_tensors, 100); // Union des tenseurs
    assert_eq!(result.common_tensors, 5); // Tenseurs communs
    assert_eq!(result.original_only, 95); // Tenseurs uniquement dans l'original
    assert_eq!(result.compared_only, 0); // Aucun tenseur uniquement dans la comparaison
}

/// Test de comparaison de configurations avec des types de valeurs très différents.
#[test]
fn test_different_value_types_comparison() {
    let config_original = ModelConfig {
        name: "model_a".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Float(4096.5)),
            ("use_bias".to_string(), ConfigValue::Boolean(true)),
            (
                "activation".to_string(),
                ConfigValue::String("relu".to_string()),
            ),
        ],
    };

    let config_compared = ModelConfig {
        name: "model_b".to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Float(4096.5)),
            ("use_bias".to_string(), ConfigValue::Boolean(true)),
            (
                "activation".to_string(),
                ConfigValue::String("relu".to_string()),
            ),
        ],
    };

    let result = compare_configs(&config_original, &config_compared);
    assert_eq!(result.similarity_score, 1.0);
    assert!(result.differences.is_empty());
}

/// Test de comparaison avec des shapes très différentes.
#[test]
fn test_very_different_shapes_comparison() {
    let shapes_original = vec![
        ShapeInfo::new("layer1.weight".to_string(), vec![4096, 4096]),
        ShapeInfo::new("layer1.bias".to_string(), vec![4096]),
        ShapeInfo::new("layer2.weight".to_string(), vec![4096, 4096]),
    ];

    let shapes_compared = vec![
        ShapeInfo::new("layer1.weight".to_string(), vec![2048, 2048]),
        ShapeInfo::new("layer1.bias".to_string(), vec![2048]),
        ShapeInfo::new("layer2.weight".to_string(), vec![2048, 2048]),
    ];

    let result = compare_shapes(&shapes_original, &shapes_compared);

    // Toutes les shapes sont différentes
    assert!(result.similarity_score < 0.5);
    assert_eq!(result.status, ComparisonStatus::Different);
    assert_eq!(result.differences.len(), 3);
}
