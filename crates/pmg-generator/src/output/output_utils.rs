//! Fonctions utilitaires pour la sortie du pipeline de génération.
//!
//! Ce module contient les helpers de validation, formatage et vérification
//! utilisés par le module de sortie principal.

use std::path::{Path, PathBuf};

use pmg_core::tensor_metadata::TensorMetadata;
use pmg_core::{DType, Shape};
use pmg_io::output_structure::{copy_config_files, write_pmg_metadata, OutputConfig};

use crate::error::{GeneratorError, GeneratorResult};
use crate::generation_report::GenerationReport;
use crate::generation_validator::{GenerationValidator, ValidationResult};
use crate::model_generator::ModelTensorResult;
use crate::writer::write_safetensors_atomic;

use super::PipelineOutputConfig;

/// Prépare les métadonnées des tenseurs pour pmg-io.
pub fn prepare_tensors_metadata(results: &[ModelTensorResult]) -> Vec<TensorMetadata> {
    results
        .iter()
        .filter_map(|result| {
            let shape_vec: Vec<u64> = if !result.values.is_empty() {
                // Estimer la forme à partir du nombre d'éléments
                // Pour l'instant, on utilise une forme 1D
                vec![result.values.len() as u64]
            } else {
                vec![0]
            };

            let shape = Shape::new(shape_vec).ok()?;
            let dtype = DType::F32;

            TensorMetadata::new(&result.name, shape, dtype).ok()
        })
        .collect()
}

/// Crée la configuration de sortie pour pmg-io.
pub fn create_output_config(
    config: &PipelineOutputConfig,
    stats: &crate::generation_stats::GenerationStats,
) -> GeneratorResult<OutputConfig> {
    let timestamp = chrono::Utc::now().to_rfc3339();

    Ok(OutputConfig {
        output_dir: config.output_dir.clone(),
        source_dir: config.source_dir.clone(),
        source_model: config.source_model.clone(),
        seed: config.seed,
        generator_version: config.generator_version.clone(),
        timestamp_utc: timestamp,
        parameter_count: stats.parameter_count,
        tensor_count: stats.tensor_count as u32,
        shards: 1,
        target_size_bytes: config.target_size_bytes,
        estimated_size_bytes: config.target_size_bytes,
        actual_size_bytes: calculate_total_size_from_stats(stats),
        dtype: config.dtype.clone(),
        generation_mode: config.generation_mode.clone(),
    })
}

/// Crée un répertoire temporaire pour l'écriture atomique.
pub fn create_temp_dir(output_dir: &Path) -> GeneratorResult<PathBuf> {
    let pid = std::process::id();
    let temp_name = format!("{}.tmp-{}", output_dir.display(), pid);
    let temp_path = PathBuf::from(temp_name);

    // Si le dossier existe déjà, le supprime
    if temp_path.exists() {
        std::fs::remove_dir_all(&temp_path).map_err(|e| {
            GeneratorError::Internal(format!(
                "échec suppression ancien dossier temporaire : {}",
                e
            ))
        })?;
    }

    std::fs::create_dir_all(&temp_path).map_err(|e| {
        GeneratorError::Internal(format!(
            "échec création dossier temporaire {} : {}",
            temp_path.display(),
            e
        ))
    })?;

    Ok(temp_path)
}

/// Renomme atomiquement un répertoire temporaire en répertoire final.
pub fn atomic_rename(temp_dir: &Path, final_dir: &Path) -> GeneratorResult<()> {
    // Si le dossier final existe déjà, le supprime
    if final_dir.exists() {
        std::fs::remove_dir_all(final_dir).map_err(|e| {
            GeneratorError::Internal(format!("échec suppression ancien dossier final : {}", e))
        })?;
    }

    std::fs::rename(temp_dir, final_dir).map_err(|e| {
        // Nettoyage en cas d'échec
        let _ = std::fs::remove_dir_all(temp_dir);
        GeneratorError::Internal(format!(
            "échec renommage atomique {} → {} : {}",
            temp_dir.display(),
            final_dir.display(),
            e
        ))
    })?;

    Ok(())
}

/// Crée la structure de dossier dans un répertoire spécifié.
pub fn create_structure_in_dir(
    dir: &Path,
    config: &OutputConfig,
    tensors_metadata: &[TensorMetadata],
) -> GeneratorResult<()> {
    // Crée le dossier pmg/
    let pmg_dir = dir.join("pmg");
    std::fs::create_dir_all(&pmg_dir)
        .map_err(|e| GeneratorError::Internal(format!("échec création dossier pmg/ : {}", e)))?;

    // Copie les fichiers de configuration depuis la source
    copy_config_files(dir, &config.source_dir, &config.source_model)
        .map_err(|e| GeneratorError::Internal(format!("échec copie fichiers config : {}", e)))?;

    // Écrit le manifeste pmg_metadata.json
    write_pmg_metadata(dir, config)
        .map_err(|e| GeneratorError::Internal(format!("échec écriture manifeste : {}", e)))?;

    // Écrit les statistiques dans pmg/statistics.json
    write_statistics(dir, tensors_metadata)?;

    Ok(())
}

/// Écrit les statistiques dans pmg/statistics.json.
fn write_statistics(dir: &Path, tensors_metadata: &[TensorMetadata]) -> GeneratorResult<()> {
    let total_size: u64 = tensors_metadata
        .iter()
        .filter_map(|t| t.byte_size_declared)
        .sum();

    let stats_json = serde_json::json!({
        "tensor_count": tensors_metadata.len(),
        "total_size_bytes": total_size,
    });

    let path = dir.join("pmg").join("statistics.json");
    let json_str = serde_json::to_string_pretty(&stats_json)?;

    std::fs::write(&path, json_str)
        .map_err(|e| GeneratorError::Internal(format!("échec écriture statistiques : {}", e)))?;

    Ok(())
}

/// Écrit les tenseurs Safetensors dans des fichiers.
pub fn write_safetensors_files(dir: &Path, results: &[ModelTensorResult]) -> GeneratorResult<()> {
    if results.is_empty() {
        return Ok(());
    }

    // Regrouper les tenseurs par shard (pour l'instant, un seul shard)
    let tensors: Vec<_> = results
        .iter()
        .map(|result| {
            let shape = if !result.values.is_empty() {
                vec![result.values.len()]
            } else {
                vec![0]
            };
            (
                result.name.clone(),
                shape,
                "f32".to_string(),
                result.values.as_slice(),
            )
        })
        .collect();

    // Écrire le fichier Safetensors
    let safetensors_path = dir.join("model-00001-of-00001.safetensors");
    write_safetensors_atomic(&safetensors_path, &tensors)?;

    // Écrire l'index
    write_safetensors_index(dir, results)?;

    Ok(())
}

/// Écrit l'index Safetensors (model.safetensors.index.json).
fn write_safetensors_index(dir: &Path, results: &[ModelTensorResult]) -> GeneratorResult<()> {
    let mut weight_map = std::collections::BTreeMap::new();

    for result in results {
        weight_map.insert(
            result.name.clone(),
            "model-00001-of-00001.safetensors".to_string(),
        );
    }

    let index = serde_json::json!({
        "metadata": {},
        "weight_map": weight_map,
    });

    let path = dir.join("model.safetensors.index.json");
    let json_str = serde_json::to_string_pretty(&index)?;

    std::fs::write(&path, json_str)
        .map_err(|e| GeneratorError::Internal(format!("échec écriture index : {}", e)))?;

    Ok(())
}

/// Génère le rapport de génération.
pub fn generate_report(
    config: &PipelineOutputConfig,
    results: &[ModelTensorResult],
) -> GeneratorResult<GenerationReport> {
    let mut report = GenerationReport::new(config.seed.to_string(), config.seed);
    report.num_tensors = results.len() as u64;
    report.parameter_count = results.iter().map(|r| r.values.len() as u64).sum();

    // Compter les couches uniques
    let mut layers = std::collections::BTreeSet::new();
    for result in results {
        if let Some(layer_index) = result.layer_index {
            layers.insert(layer_index);
        }
    }
    report.num_layers = layers.len() as u64;

    Ok(report)
}

/// Valide la sortie de génération.
pub fn validate_output(
    _config: &PipelineOutputConfig,
    results: &[ModelTensorResult],
    report: &GenerationReport,
) -> GeneratorResult<ValidationResult> {
    let specs: Vec<pmg_blueprint::TensorSpec> = results
        .iter()
        .filter_map(|result| {
            let shape_vec: Vec<u64> = if !result.values.is_empty() {
                vec![result.values.len() as u64]
            } else {
                vec![0]
            };

            let shape = Shape::new(shape_vec).ok()?;
            let dtype = DType::F32;

            pmg_blueprint::TensorSpec::new(&result.name, shape, dtype, pmg_core::TensorRole::Other)
                .ok()
        })
        .collect();

    let validator = GenerationValidator::new(report.clone(), specs);
    validator.validate()
}

/// Calcule la taille totale des tenseurs en octets.
pub fn calculate_total_size(results: &[ModelTensorResult]) -> u64 {
    results
        .iter()
        .map(|r| (r.values.len() * 4) as u64) // f32 = 4 octets
        .sum()
}

/// Calcule la taille totale à partir des statistiques.
pub fn calculate_total_size_from_stats(stats: &crate::generation_stats::GenerationStats) -> u64 {
    stats.parameter_count * 4 // f32 = 4 octets
}

/// Écrit l'index Safetensors pour le mode streaming (à partir du blueprint).
pub fn write_safetensors_index_from_blueprint(
    output_dir: &Path,
    blueprint: &pmg_blueprint::ModelBlueprint,
) -> GeneratorResult<()> {
    let index_path = output_dir.join("model.safetensors.index.json");

    // Construire le weight_map
    let mut weight_map = std::collections::BTreeMap::new();
    let shard_name = "model-00001-of-00001.safetensors";

    // Embeddings
    for tensor in &blueprint.embeddings {
        weight_map.insert(tensor.name.clone(), shard_name.to_string());
    }

    // Couches
    for (layer_idx, layer) in blueprint.layers.iter().enumerate() {
        for tensor in layer.all_tensors() {
            let name = format!("layers.{}.{}", layer_idx, tensor.name);
            weight_map.insert(name, shard_name.to_string());
        }
    }

    // Norme finale
    for tensor in &blueprint.final_norm {
        weight_map.insert(tensor.name.clone(), shard_name.to_string());
    }

    // Tête de langage
    for tensor in &blueprint.lm_head {
        weight_map.insert(tensor.name.clone(), shard_name.to_string());
    }

    // Autres tenseurs
    for tensor in &blueprint.extra_tensors {
        weight_map.insert(tensor.name.clone(), shard_name.to_string());
    }

    // Construire l'index complet
    let total_size = calculate_total_size_from_blueprint(blueprint);
    let index = serde_json::json!({
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    });

    // Écrire le fichier
    let index_json = serde_json::to_string_pretty(&index)?;
    std::fs::write(&index_path, index_json)
        .map_err(|e| GeneratorError::Internal(format!("échec écriture index : {}", e)))?;

    Ok(())
}

/// Calcule la taille totale des tenseurs à partir du blueprint.
fn calculate_total_size_from_blueprint(blueprint: &pmg_blueprint::ModelBlueprint) -> u64 {
    let mut total = 0u64;
    for tensor in blueprint.all_tensors() {
        if let Ok(Some(bytes)) = tensor.byte_size() {
            total += bytes;
        }
    }
    total
}

/// Valide la configuration de sortie pour le streaming.
pub fn validate_streaming_config(config: &PipelineOutputConfig) -> GeneratorResult<()> {
    if config.output_dir.as_os_str().is_empty() {
        return Err(GeneratorError::Internal(
            "répertoire de sortie vide".to_string(),
        ));
    }
    Ok(())
}

/// Valide un fichier Safetensors généré en streaming.
pub fn validate_streaming_output(safetensors_path: &Path) -> GeneratorResult<ValidationResult> {
    // Pour l'instant, validation basique : vérifier que le fichier existe
    if !safetensors_path.exists() {
        return Ok(ValidationResult::failure(vec![
            "fichier Safetensors non trouvé".to_string(),
        ]));
    }

    // Vérifier la taille minimale
    let metadata = std::fs::metadata(safetensors_path)
        .map_err(|e| GeneratorError::Internal(format!("erreur lecture métadonnées : {}", e)))?;
    if metadata.len() < 8 {
        return Ok(ValidationResult::failure(vec![
            "fichier Safetensors trop petit".to_string(),
        ]));
    }

    Ok(ValidationResult::success())
}
