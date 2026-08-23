//! Tests d'intégration pour le pipeline streaming optimisé mémoire (Phase 2).
//!
//! Ces tests valident que le pipeline streaming avec écriture directe sur disque
//! fonctionne correctement et réduit la consommation mémoire.

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_core::{DType, Shape, TensorRole};
use pmg_generator::streaming_config::StreamingConfig;
use pmg_generator::tensor_chunk_generator::TensorChunkGenerator;
use pmg_io::safetensors::ShardWriter;
use tempfile::tempdir;

/// Test d'intégration : création et écriture d'un tenseur via TensorChunkGenerator.
#[test]
fn test_tensor_chunk_generator_integration() {
    let dir = tempdir().unwrap();
    let shard_path = dir.path().join("test.safetensors");

    // Configuration du streaming avec des chunks de 1 Mo pour les tests
    let config = StreamingConfig::new(1024 * 1024, 10 * 1024 * 1024); // 1 Mo chunks, 10 Mo max
    let seed = 42;

    // Création du writer avec une réserve d'en-tête de 1 Ko
    let mut writer = ShardWriter::new(shard_path.clone(), 1024).unwrap();

    // Création du générateur
    let mut generator = TensorChunkGenerator::new(config, seed);

    // Spécification du tenseur (100x64 = 6400 éléments f32 = 25600 octets)
    let tensor_spec = TensorSpec::new(
        "model.embed_tokens.weight",
        Shape::new(vec![100, 64]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    // Génération et écriture
    let result = generator
        .generate_and_write_tensor(&tensor_spec, &mut writer, 0)
        .unwrap();

    // Vérifications
    assert_eq!(result.total_elements, 6400);
    assert!(result.chunks_written >= 1);
    assert!(result.total_bytes_written > 0);

    // Finalisation
    writer.finalize().unwrap();

    // Vérification que le fichier a été créé
    assert!(shard_path.exists());
}

/// Test d'intégration : écriture de plusieurs tenseurs.
#[test]
fn test_multiple_tensors_integration() {
    let dir = tempdir().unwrap();
    let shard_path = dir.path().join("multi_tensor.safetensors");

    let config = StreamingConfig::new(1024 * 1024, 10 * 1024 * 1024);
    let seed = 123;

    let mut writer = ShardWriter::new(shard_path.clone(), 2048).unwrap();
    let mut generator = TensorChunkGenerator::new(config, seed);

    // Premier tenseur : embedding
    let embedding_spec = TensorSpec::new(
        "model.embed_tokens.weight",
        Shape::new(vec![100, 64]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    let result1 = generator
        .generate_and_write_tensor(&embedding_spec, &mut writer, 0)
        .unwrap();
    assert_eq!(result1.total_elements, 6400);

    // Deuxième tenseur : projection attention
    let attention_spec = TensorSpec::new(
        "model.layers.0.self_attn.q_proj.weight",
        Shape::new(vec![64, 64]).unwrap(),
        DType::F32,
        TensorRole::AttentionQuery,
    )
    .unwrap();

    let result2 = generator
        .generate_and_write_tensor(&attention_spec, &mut writer, 1)
        .unwrap();
    assert_eq!(result2.total_elements, 4096);

    // Troisième tenseur : norme
    let norm_spec = TensorSpec::new(
        "model.norm.weight",
        Shape::new(vec![64]).unwrap(),
        DType::F32,
        TensorRole::Norm,
    )
    .unwrap();

    let result3 = generator
        .generate_and_write_tensor(&norm_spec, &mut writer, 2)
        .unwrap();
    assert_eq!(result3.total_elements, 64);

    // Finalisation
    writer.finalize().unwrap();

    // Vérification du fichier
    assert!(shard_path.exists());
}

/// Test d'intégration : monitoring mémoire pendant la génération.
#[test]
fn test_memory_monitoring_integration() {
    let config = StreamingConfig::new(1024 * 1024, 50 * 1024 * 1024); // 50 Mo max
    let seed = 456;

    let generator = TensorChunkGenerator::new(config, seed);

    // Vérification que le moniteur est initialisé
    let monitor = generator.memory_monitor();
    assert_eq!(monitor.usage_percentage(), 0.0);
    assert!(!monitor.is_near_limit());
}

/// Test d'intégration : déterminisme de la génération.
#[test]
fn test_determinism_integration() {
    let dir = tempdir().unwrap();
    let shard_path1 = dir.path().join("test1.safetensors");
    let shard_path2 = dir.path().join("test2.safetensors");

    let config = StreamingConfig::new(1024 * 1024, 10 * 1024 * 1024);
    let seed = 789;

    let tensor_spec = TensorSpec::new(
        "test_tensor",
        Shape::new(vec![32, 32]).unwrap(),
        DType::F32,
        TensorRole::Other,
    )
    .unwrap();

    // Première génération
    let mut writer1 = ShardWriter::new(shard_path1.clone(), 1024).unwrap();
    let mut generator1 = TensorChunkGenerator::new(config.clone(), seed);
    let result1 = generator1
        .generate_and_write_tensor(&tensor_spec, &mut writer1, 0)
        .unwrap();
    writer1.finalize().unwrap();

    // Deuxième génération avec la même seed
    let mut writer2 = ShardWriter::new(shard_path2.clone(), 1024).unwrap();
    let mut generator2 = TensorChunkGenerator::new(config, seed);
    let result2 = generator2
        .generate_and_write_tensor(&tensor_spec, &mut writer2, 0)
        .unwrap();
    writer2.finalize().unwrap();

    // Les résultats doivent être identiques
    assert_eq!(result1.total_elements, result2.total_elements);
    assert_eq!(result1.chunks_written, result2.chunks_written);
    assert_eq!(result1.total_bytes_written, result2.total_bytes_written);

    // Les fichiers doivent être identiques
    let data1 = std::fs::read(&shard_path1).unwrap();
    let data2 = std::fs::read(&shard_path2).unwrap();
    assert_eq!(data1, data2);
}

/// Test d'intégration : pipeline streaming complet avec écriture disque.
#[test]
fn test_streaming_pipeline_disk_write() {
    let dir = tempdir().unwrap();
    let shard_path = dir.path().join("pipeline_test.safetensors");

    // Configuration
    let config = StreamingConfig::new(1024 * 1024, 20 * 1024 * 1024);
    let seed = 101;

    // Création du writer et du générateur
    let mut writer = ShardWriter::new(shard_path.clone(), 2048).unwrap();
    let mut generator = TensorChunkGenerator::new(config, seed);

    // Spécifications des tenseurs
    let tensors = [
        TensorSpec::new(
            "layer1.weight",
            Shape::new(vec![64, 64]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap(),
        TensorSpec::new(
            "layer1.bias",
            Shape::new(vec![64]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap(),
        TensorSpec::new(
            "layer2.weight",
            Shape::new(vec![32, 64]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap(),
    ];

    // Génération et écriture de tous les tenseurs
    let mut total_elements = 0;
    for (idx, tensor) in tensors.iter().enumerate() {
        let result = generator
            .generate_and_write_tensor(tensor, &mut writer, idx)
            .unwrap();
        total_elements += result.total_elements;
    }

    // Finalisation
    writer.finalize().unwrap();

    // Vérifications
    assert_eq!(total_elements, 64 * 64 + 64 + 32 * 64); // 4096 + 64 + 2048 = 6208
    assert!(shard_path.exists());

    // Vérification de la taille du fichier
    let file_size = std::fs::metadata(&shard_path).unwrap().len();
    assert!(file_size > 0);
}
