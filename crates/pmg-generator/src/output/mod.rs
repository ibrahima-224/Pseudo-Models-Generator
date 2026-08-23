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

//! Sortie du pipeline de génération.
//!
//! Ce module orchestre la création de la structure de sortie complète :
//! - Dossier de sortie avec copie des fichiers de configuration
//! - Écriture des tenseurs Safetensors
//! - Génération du manifeste `pmg_metadata.json`
//! - Statistiques et provenance dans `pmg/`
//!
//! ## Contraintes
//!
//! - **Atomicité** : écriture dans dossier temporaire puis renommage
//! - **Mémoire bornée** : écriture par chunks
//! - **Déterminisme** : même seed = même sortie binaire
//!
//! ## Organisation
//!
//! Le module est divisé en sous-modules pour respecter la limite de 500 lignes :
//! - `output_utils` : Fonctions utilitaires de validation et formatage
//! - `output_tests` : Tests unitaires

/// Type alias pour le callback de progression.
/// Prend (tensor_index, total_tensors, tensor_name) en paramètres.
type ProgressCallback<'a> = Option<&'a dyn Fn(usize, usize, &str)>;

use std::path::PathBuf;

use pmg_io::output_structure::{copy_config_files, write_pmg_metadata, SourceModel};

use crate::error::{GeneratorError, GeneratorResult};
use crate::generation_report::GenerationReport;
use crate::generation_validator::ValidationResult;
use crate::model_generator::streaming;
use crate::model_generator::ModelGeneratorComplete;
use crate::pipeline::GenerationPipeline;

// Déclaration des sous-modules
#[cfg(test)]
pub mod output_tests;
pub mod output_utils;

// Réexports pour rétrocompatibilité API
pub use output_utils::{
    atomic_rename, calculate_total_size, calculate_total_size_from_stats, create_output_config,
    create_structure_in_dir, create_temp_dir, generate_report, prepare_tensors_metadata,
    validate_output, validate_streaming_config, validate_streaming_output, write_safetensors_files,
    write_safetensors_index_from_blueprint,
};

/// Configuration de la sortie du pipeline.
#[derive(Debug, Clone)]
pub struct PipelineOutputConfig {
    /// Chemin du répertoire de sortie.
    pub output_dir: PathBuf,
    /// Chemin du répertoire source contenant les fichiers de configuration.
    pub source_dir: PathBuf,
    /// Modèle source.
    pub source_model: SourceModel,
    /// Seed utilisé pour la génération.
    pub seed: u64,
    /// Version du générateur.
    pub generator_version: String,
    /// Mode de génération.
    pub generation_mode: String,
    /// Taille cible en octets.
    pub target_size_bytes: u64,
    /// Type de données.
    pub dtype: String,
}

/// Résultat de la sortie du pipeline.
#[derive(Debug)]
pub struct PipelineOutputResult {
    /// Chemin du répertoire de sortie créé.
    pub output_dir: PathBuf,
    /// Nombre de tenseurs générés.
    pub tensor_count: usize,
    /// Nombre total de paramètres.
    pub parameter_count: u64,
    /// Taille réelle en octets.
    pub actual_size_bytes: u64,
    /// Résultat de la validation.
    pub validation: ValidationResult,
    /// Rapport de génération.
    pub report: GenerationReport,
}

/// Orchestre la sortie complète du pipeline de génération.
///
/// # Paramètres
/// - `config` : configuration de la sortie
/// - `blueprint` : blueprint du modèle
/// - `pipeline` : pipeline de génération
///
/// # Retourne
/// Le résultat de la sortie contenant les statistiques et la validation.
///
/// # Erreurs
/// Retourne une erreur si la génération ou l'écriture échoue.
pub fn execute_pipeline_output(
    config: &PipelineOutputConfig,
    blueprint: pmg_blueprint::ModelBlueprint,
    pipeline: GenerationPipeline,
) -> GeneratorResult<PipelineOutputResult> {
    // 1. Générer tous les tenseurs du modèle
    let chunk_size = 1024 * 1024; // 1 Mo par défaut
    let generator = ModelGeneratorComplete::new(
        blueprint,
        config.seed,
        &config.generator_version,
        pipeline,
        chunk_size,
    );

    let results = generator.generate_all()?;
    let stats = generator.compute_stats(&results);

    // 2. Préparer les métadonnées des tenseurs pour l'écriture
    let tensors_metadata = prepare_tensors_metadata(&results);

    // 3. Créer la structure de sortie avec pmg-io
    let output_config = create_output_config(config, &stats)?;

    // Créer le dossier temporaire pour l'atomicité
    let temp_dir = create_temp_dir(&config.output_dir)?;
    let temp_path = temp_dir.as_path();

    // Créer la structure dans le dossier temporaire
    create_structure_in_dir(temp_path, &output_config, &tensors_metadata)?;

    // 4. Écrire les tenseurs Safetensors
    write_safetensors_files(temp_path, &results)?;

    // 5. Renommer atomiquement le dossier temporaire en dossier final
    atomic_rename(&temp_dir, &config.output_dir)?;

    // 6. Valider la génération
    let report = generate_report(config, &results)?;
    let validation = validate_output(config, &results, &report)?;

    Ok(PipelineOutputResult {
        output_dir: config.output_dir.clone(),
        tensor_count: results.len(),
        parameter_count: stats.parameter_count,
        actual_size_bytes: calculate_total_size(&results),
        validation,
        report,
    })
}

/// Exécute le pipeline de sortie en mode streaming (sans accumulation mémoire).
///
/// Adapté pour les modèles de grande taille (> 10 GB). Écrit chaque tenseur
/// directement dans le fichier Safetensors sans charger tous les tenseurs en mémoire.
pub fn execute_pipeline_output_streaming(
    config: &PipelineOutputConfig,
    blueprint: pmg_blueprint::ModelBlueprint,
    pipeline: GenerationPipeline,
    progress_callback: ProgressCallback<'_>,
) -> GeneratorResult<PipelineOutputResult> {
    let _start_time = std::time::Instant::now();

    // 1. Valider la configuration
    validate_streaming_config(config)?;

    // 2. Créer le répertoire de sortie
    let output_dir = &config.output_dir;
    std::fs::create_dir_all(output_dir).map_err(|e| {
        GeneratorError::Internal(format!("échec création répertoire de sortie : {}", e))
    })?;

    // 3. Estimer la taille de l'en-tête Safetensors
    let tensor_count = streaming::count_total_tensors(&blueprint);
    let header_reserve = streaming::estimate_header_size(tensor_count);

    // 4. Créer le ShardWriter avec réserve d'en-tête
    let safetensors_path = output_dir.join("model-00001-of-00001.safetensors");
    let mut writer =
        pmg_io::safetensors::ShardWriter::new(safetensors_path.clone(), header_reserve).map_err(
            |e| GeneratorError::Internal(format!("erreur création ShardWriter : {}", e)),
        )?;

    // 5. Créer le générateur
    let chunk_size = 1024 * 1024; // 1 Mo par défaut
    let generator = ModelGeneratorComplete::new(
        blueprint.clone(),
        config.seed,
        &config.generator_version,
        pipeline,
        chunk_size,
    );

    // 6. Générer et écrire chaque tenseur (streaming)
    let stats = generator.generate_and_write(&mut writer, progress_callback)?;

    // 7. Finaliser le shard (écrit l'en-tête)
    let _shard_result = writer
        .finalize()
        .map_err(|e| GeneratorError::Internal(format!("erreur finalisation shard : {}", e)))?;

    // 8. Écrire l'index Safetensors
    write_safetensors_index_from_blueprint(output_dir, &blueprint)?;

    // 9. Copier les fichiers de configuration
    copy_config_files(output_dir, &config.source_dir, &config.source_model)?;

    // 10. Écrire les métadonnées PMG
    let output_config = create_output_config(config, &stats)?;
    write_pmg_metadata(output_dir, &output_config)?;

    // 11. Calculer les statistiques finales
    let total_size = std::fs::metadata(&safetensors_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // 12. Valider la génération
    let mut report = GenerationReport::new(&config.generator_version, config.seed);
    report.num_tensors = stats.tensor_count as u64;
    report.parameter_count = stats.parameter_count;
    let validation = validate_streaming_output(&safetensors_path)?;

    Ok(PipelineOutputResult {
        output_dir: config.output_dir.clone(),
        tensor_count: stats.tensor_count,
        parameter_count: stats.parameter_count,
        actual_size_bytes: total_size,
        validation,
        report,
    })
}
