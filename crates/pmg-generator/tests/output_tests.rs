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

//! Tests d'intégration de la sortie du pipeline de génération.
//!
//! Ces tests vérifient la création de la structure de sortie complète,
//! l'écriture Safetensors, et la validation après génération.

use std::fs;
use std::path::PathBuf;

use pmg_blueprint::architecture::ArchitectureKind;
use pmg_blueprint::layer::{LayerKind, LayerSpec};
use pmg_blueprint::naming::NamingRules;
use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_blueprint::ModelBlueprint;
use pmg_core::model_config::glm52_test_config;
use pmg_core::{DType, Shape, TensorRole};
use pmg_io::output_structure::SourceModel;

use pmg_generator::{execute_pipeline_output, GenerationPipeline, PipelineOutputConfig};

/// Blueprint de test simple pour les tests d'intégration.
fn test_blueprint() -> ModelBlueprint {
    let config = glm52_test_config();
    let mut bp = ModelBlueprint::new(
        "glm-5.2",
        ArchitectureKind::MoETransformer,
        config,
        NamingRules::glm52(),
    );

    // Embedding
    bp.embeddings.push(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
            TensorRole::Embedding,
        )
        .unwrap(),
    );

    // Couche 0
    let mut layer0 = LayerSpec::new(0, LayerKind::Dense);
    layer0.attention.push(
        TensorSpec::new(
            "model.layers.0.self_attn.q_proj.weight",
            Shape::new(vec![64, 64]).unwrap(),
            DType::F32,
            TensorRole::AttentionQuery,
        )
        .unwrap(),
    );
    layer0.mlp.push(
        TensorSpec::new(
            "model.layers.0.mlp.gate_proj.weight",
            Shape::new(vec![128, 64]).unwrap(),
            DType::F32,
            TensorRole::MlpGate,
        )
        .unwrap(),
    );
    bp.layers.push(layer0);

    // Norme finale
    bp.final_norm.push(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![64]).unwrap(),
            DType::F32,
            TensorRole::Norm,
        )
        .unwrap(),
    );

    // LM Head
    bp.lm_head.push(
        TensorSpec::new(
            "lm_head.weight",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
            TensorRole::LmHead,
        )
        .unwrap(),
    );

    bp
}

/// Test : exécution complète du pipeline de sortie.
#[test]
fn test_execute_pipeline_output() {
    let dir = tempfile::tempdir().unwrap();
    let output_dir = dir.path().join("output");
    let source_dir = PathBuf::from("Models/GLM-5.2");

    // Vérifier que le dossier source existe
    if !source_dir.exists() {
        // Si le dossier source n'existe pas, on utilise un dossier temporaire
        // et on crée des fichiers de config minimaux
        let temp_source = dir.path().join("source");
        fs::create_dir_all(&temp_source).unwrap();

        // Créer config.json minimal
        fs::write(
            temp_source.join("config.json"),
            r#"{"model_type": "glm52"}"#,
        )
        .unwrap();

        // Créer generation_config.json minimal
        fs::write(
            temp_source.join("generation_config.json"),
            r#"{"do_sample": false}"#,
        )
        .unwrap();

        // Créer tokenizer.json minimal
        fs::write(temp_source.join("tokenizer.json"), r#"{}"#).unwrap();

        // Créer tokenizer_config.json minimal
        fs::write(temp_source.join("tokenizer_config.json"), r#"{}"#).unwrap();

        let config = PipelineOutputConfig {
            output_dir: output_dir.clone(),
            source_dir: temp_source,
            source_model: SourceModel::Glm52,
            seed: 42,
            generator_version: "1.0.0".to_string(),
            generation_mode: "size-constrained".to_string(),
            target_size_bytes: 1024 * 1024,
            dtype: "f32".to_string(),
        };

        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let result = execute_pipeline_output(&config, blueprint, pipeline).unwrap();

        // Vérifier les résultats
        assert_eq!(result.tensor_count, 5); // 1 embedding + 2 couches + 1 norme + 1 lm_head
        assert!(result.parameter_count > 0);
        assert!(result.actual_size_bytes > 0);
        assert!(result.validation.success);

        // Vérifier que le dossier de sortie existe
        assert!(output_dir.exists());

        // Vérifier la structure de sortie
        assert!(output_dir.join("model-00001-of-00001.safetensors").exists());
        assert!(output_dir.join("model.safetensors.index.json").exists());
        assert!(output_dir.join("pmg_metadata.json").exists());
        assert!(output_dir.join("pmg").exists());
        assert!(output_dir.join("pmg").join("statistics.json").exists());
    } else {
        // Si le dossier source existe, on l'utilise directement
        let config = PipelineOutputConfig {
            output_dir: output_dir.clone(),
            source_dir,
            source_model: SourceModel::Glm52,
            seed: 42,
            generator_version: "1.0.0".to_string(),
            generation_mode: "size-constrained".to_string(),
            target_size_bytes: 1024 * 1024,
            dtype: "f32".to_string(),
        };

        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let result = execute_pipeline_output(&config, blueprint, pipeline).unwrap();

        // Vérifier les résultats
        assert_eq!(result.tensor_count, 5);
        assert!(result.parameter_count > 0);
        assert!(result.actual_size_bytes > 0);
        assert!(result.validation.success);

        // Vérifier que le dossier de sortie existe
        assert!(output_dir.exists());

        // Vérifier la structure de sortie
        assert!(output_dir.join("model-00001-of-00001.safetensors").exists());
        assert!(output_dir.join("model.safetensors.index.json").exists());
        assert!(output_dir.join("pmg_metadata.json").exists());
        assert!(output_dir.join("pmg").exists());
        assert!(output_dir.join("pmg").join("statistics.json").exists());
    }
}

/// Test : déterminisme de la sortie.
#[test]
fn test_output_determinism() {
    let dir = tempfile::tempdir().unwrap();
    let output_dir1 = dir.path().join("output1");
    let output_dir2 = dir.path().join("output2");

    // Créer un dossier source temporaire
    let source_dir = dir.path().join("source");
    fs::create_dir_all(&source_dir).unwrap();

    // Créer des fichiers de config minimaux
    fs::write(source_dir.join("config.json"), r#"{"model_type": "glm52"}"#).unwrap();
    fs::write(
        source_dir.join("generation_config.json"),
        r#"{"do_sample": false}"#,
    )
    .unwrap();
    fs::write(source_dir.join("tokenizer.json"), r#"{}"#).unwrap();
    fs::write(source_dir.join("tokenizer_config.json"), r#"{}"#).unwrap();

    // Première exécution
    let config1 = PipelineOutputConfig {
        output_dir: output_dir1.clone(),
        source_dir: source_dir.clone(),
        source_model: SourceModel::Glm52,
        seed: 42,
        generator_version: "1.0.0".to_string(),
        generation_mode: "size-constrained".to_string(),
        target_size_bytes: 1024 * 1024,
        dtype: "f32".to_string(),
    };

    let blueprint1 = test_blueprint();
    let pipeline1 = GenerationPipeline::full();
    let result1 = execute_pipeline_output(&config1, blueprint1, pipeline1).unwrap();

    // Deuxième exécution (même seed)
    let config2 = PipelineOutputConfig {
        output_dir: output_dir2.clone(),
        source_dir,
        source_model: SourceModel::Glm52,
        seed: 42,
        generator_version: "1.0.0".to_string(),
        generation_mode: "size-constrained".to_string(),
        target_size_bytes: 1024 * 1024,
        dtype: "f32".to_string(),
    };

    let blueprint2 = test_blueprint();
    let pipeline2 = GenerationPipeline::full();
    let result2 = execute_pipeline_output(&config2, blueprint2, pipeline2).unwrap();

    // Vérifier que les résultats sont identiques
    assert_eq!(result1.tensor_count, result2.tensor_count);
    assert_eq!(result1.parameter_count, result2.parameter_count);
    assert_eq!(result1.actual_size_bytes, result2.actual_size_bytes);

    // Vérifier que les fichiers Safetensors sont identiques
    let safetensors1 = fs::read(output_dir1.join("model-00001-of-00001.safetensors")).unwrap();
    let safetensors2 = fs::read(output_dir2.join("model-00001-of-00001.safetensors")).unwrap();
    assert_eq!(safetensors1, safetensors2);

    // Vérifier que les index sont identiques
    let index1 = fs::read(output_dir1.join("model.safetensors.index.json")).unwrap();
    let index2 = fs::read(output_dir2.join("model.safetensors.index.json")).unwrap();
    assert_eq!(index1, index2);
}

/// Test : atomicité de la sortie.
#[test]
fn test_output_atomicity() {
    let dir = tempfile::tempdir().unwrap();
    let output_dir = dir.path().join("output");

    // Créer un dossier source temporaire
    let source_dir = dir.path().join("source");
    fs::create_dir_all(&source_dir).unwrap();

    // Créer des fichiers de config minimaux
    fs::write(source_dir.join("config.json"), r#"{"model_type": "glm52"}"#).unwrap();
    fs::write(
        source_dir.join("generation_config.json"),
        r#"{"do_sample": false}"#,
    )
    .unwrap();
    fs::write(source_dir.join("tokenizer.json"), r#"{}"#).unwrap();
    fs::write(source_dir.join("tokenizer_config.json"), r#"{}"#).unwrap();

    let config = PipelineOutputConfig {
        output_dir: output_dir.clone(),
        source_dir,
        source_model: SourceModel::Glm52,
        seed: 42,
        generator_version: "1.0.0".to_string(),
        generation_mode: "size-constrained".to_string(),
        target_size_bytes: 1024 * 1024,
        dtype: "f32".to_string(),
    };

    let blueprint = test_blueprint();
    let pipeline = GenerationPipeline::full();

    // Exécuter la sortie
    let _result = execute_pipeline_output(&config, blueprint, pipeline).unwrap();

    // Vérifier que le dossier final existe et qu'aucun dossier temporaire n'est présent
    assert!(output_dir.exists());

    // Vérifier qu'il n'y a pas de dossiers .tmp- dans le répertoire parent
    let parent = output_dir.parent().unwrap();
    let entries: Vec<_> = fs::read_dir(parent)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains(".tmp-")
        })
        .collect();

    assert!(
        entries.is_empty(),
        "Des dossiers temporaires ont été laissés : {:?}",
        entries
    );
}

/// Test : écriture Safetensors.
#[test]
fn test_safetensors_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.safetensors");

    let values = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensors = vec![(
        "tensor1".to_string(),
        vec![2, 3],
        "f32".to_string(),
        values.as_slice(),
    )];

    pmg_generator::write_safetensors_atomic(&path, &tensors).unwrap();

    assert!(path.exists());
    let metadata = fs::metadata(&path).unwrap();
    assert!(metadata.len() > 0);

    // Vérifier que le fichier tmp n'existe pas
    let tmp_path = path.with_extension("tmp");
    assert!(!tmp_path.exists());
}

/// Test : le rapport de génération est correct.
#[test]
fn test_generation_report() {
    let dir = tempfile::tempdir().unwrap();
    let output_dir = dir.path().join("output");

    // Créer un dossier source temporaire
    let source_dir = dir.path().join("source");
    fs::create_dir_all(&source_dir).unwrap();

    // Créer des fichiers de config minimaux
    fs::write(source_dir.join("config.json"), r#"{"model_type": "glm52"}"#).unwrap();
    fs::write(
        source_dir.join("generation_config.json"),
        r#"{"do_sample": false}"#,
    )
    .unwrap();
    fs::write(source_dir.join("tokenizer.json"), r#"{}"#).unwrap();
    fs::write(source_dir.join("tokenizer_config.json"), r#"{}"#).unwrap();

    let config = PipelineOutputConfig {
        output_dir,
        source_dir,
        source_model: SourceModel::Glm52,
        seed: 42,
        generator_version: "1.0.0".to_string(),
        generation_mode: "size-constrained".to_string(),
        target_size_bytes: 1024 * 1024,
        dtype: "f32".to_string(),
    };

    let blueprint = test_blueprint();
    let pipeline = GenerationPipeline::full();

    let result = execute_pipeline_output(&config, blueprint, pipeline).unwrap();

    // Vérifier le rapport
    assert_eq!(result.report.seed, 42);
    assert_eq!(result.report.num_tensors, 5);
    assert!(result.report.parameter_count > 0);
}
