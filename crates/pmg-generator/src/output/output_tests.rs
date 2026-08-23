//! Tests unitaires pour le module de sortie du pipeline.
//!
//! Ce module contient les tests pour les fonctions utilitaires de sortie.

use super::*;
use crate::model_generator::ModelTensorResult;
use std::path::PathBuf;

/// Test de création d'une configuration de sortie.
#[test]
fn pipeline_output_config_creation() {
    let config = PipelineOutputConfig {
        output_dir: PathBuf::from("/tmp/test"),
        source_dir: PathBuf::from("/tmp/source"),
        source_model: SourceModel::Glm52,
        seed: 42,
        generator_version: "1.0.0".to_string(),
        generation_mode: "size-constrained".to_string(),
        target_size_bytes: 1024 * 1024,
        dtype: "f32".to_string(),
    };

    assert_eq!(config.seed, 42);
    assert_eq!(config.generator_version, "1.0.0");
}

/// Test de préparation des métadonnées avec une liste vide.
#[test]
fn prepare_tensors_metadata_empty() {
    let results = vec![];
    let metadata = prepare_tensors_metadata(&results);
    assert!(metadata.is_empty());
}

/// Test de préparation des métadonnées avec des résultats.
#[test]
fn prepare_tensors_metadata_with_results() {
    let results = vec![ModelTensorResult {
        name: "test".to_string(),
        values: vec![1.0, 2.0, 3.0],
        pipeline_results: vec![],
        category: "test".to_string(),
        layer_index: None,
    }];

    let metadata = prepare_tensors_metadata(&results);
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata[0].name, "test");
    assert_eq!(metadata[0].byte_size_declared, Some(12)); // 3 * 4 octets
}

/// Test de calcul de taille totale avec une liste vide.
#[test]
fn calculate_total_size_empty() {
    let results = vec![];
    assert_eq!(calculate_total_size(&results), 0);
}

/// Test de calcul de taille totale avec des résultats.
#[test]
fn calculate_total_size_with_results() {
    let results = vec![
        ModelTensorResult {
            name: "t1".to_string(),
            values: vec![0.0; 100],
            pipeline_results: vec![],
            category: "test".to_string(),
            layer_index: None,
        },
        ModelTensorResult {
            name: "t2".to_string(),
            values: vec![0.0; 200],
            pipeline_results: vec![],
            category: "test".to_string(),
            layer_index: None,
        },
    ];

    assert_eq!(calculate_total_size(&results), 1200); // (100 + 200) * 4
}
