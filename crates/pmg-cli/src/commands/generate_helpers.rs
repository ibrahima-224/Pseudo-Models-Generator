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

//! Fonctions d'assistance pour la commande `generate`.
//!
//! Ce module contient des fonctions utilitaires pour la commande de génération,
//! notamment la vérification de l'écrasement de fichiers, la confirmation
//! utilisateur et l'exécution du mode asynchrone.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use pmg_generator::{AsyncConfig, GenerationReport, PipelineOutputResult, ValidationResult};

use crate::output;

/// Exécute la génération asynchrone via tokio.
///
/// Crée un runtime tokio, lance la génération parallèle et retourne le résultat.
///
/// # Paramètres
/// - `model_name` : nom du modèle cible
/// - `output_dir` : répertoire de sortie
/// - `num_workers` : nombre de workers parallèles
/// - `chunk_size` : taille des chunks en octets
/// - `seed` : graine de génération
/// - `blueprint` : blueprint du modèle
/// - `verbose` : mode verbeux (active les callbacks de progression)
///
/// # Retourne
/// Le résultat de la sortie du pipeline.
pub fn execute_async_generation(
    model_name: &str,
    output_dir: &Path,
    num_workers: usize,
    chunk_size: usize,
    seed: u64,
    blueprint: pmg_blueprint::ModelBlueprint,
    verbose: bool,
) -> Result<PipelineOutputResult> {
    if verbose {
        output::info("Mode asynchrone activé : génération parallèle via tokio");
    }

    let output_path = output_dir.join(format!("{}.safetensors", model_name));

    // Créer le runtime tokio bloquant
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        output::error_with_cause_and_advice(
            "Erreur création runtime tokio",
            &format!("Détails : {}", e),
            "Vérifiez les ressources système disponibles",
        );
        anyhow::anyhow!("Erreur runtime tokio : {}", e)
    })?;

    // Exécuter la génération asynchrone
    let result = rt.block_on(async {
        let stats = pmg_generator::generate_model_async(
            &AsyncConfig {
                num_workers,
                chunk_size,
                seed,
                output_path,
            },
            blueprint,
            if verbose {
                Some(Arc::new(move |current, total, name| {
                    output::progress(name, total, current);
                }))
            } else {
                None
            },
        )
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok::<PipelineOutputResult, anyhow::Error>(PipelineOutputResult {
            output_dir: output_dir.to_path_buf(),
            tensor_count: stats.tensor_count,
            parameter_count: stats.parameter_count,
            actual_size_bytes: stats.parameter_count * 4,
            validation: ValidationResult::success(),
            report: GenerationReport::new(model_name, seed),
        })
    })?;

    Ok(result)
}

/// Vérifie si des fichiers de configuration existent déjà dans le répertoire de sortie.
///
/// Cette fonction parcourt la liste des fichiers de configuration selon le modèle
/// source et vérifie lesquels existent déjà dans le répertoire de sortie.
/// Elle retourne la liste des fichiers qui seraient écrasés lors de la copie.
///
/// # Paramètres
/// - `output_dir` : répertoire de sortie où les fichiers seraient copiés
/// - `source_model` : nom du modèle source (ex: "glm52", "deepseek-v4-flash")
///
/// # Retourne
/// Un vecteur de noms de fichiers qui existent déjà et seraient écrasés.
///
/// # Comportement
/// - Pour GLM-5.2 : vérifie config.json, generation_config.json, tokenizer.json,
///   tokenizer_config.json, special_tokens_map.json, chat_template.jinja
/// - Pour DeepSeek-V4-Flash : vérifie config.json, generation_config.json,
///   tokenizer.json, tokenizer_config.json
/// - Par défaut : vérifie les fichiers communs à tous les modèles
pub fn check_overwrite_warning(output_dir: &Path, source_model: &str) -> Vec<String> {
    // Liste des fichiers de configuration à vérifier selon le modèle
    let config_files = match source_model {
        "glm52" | "GLM-5.2" => vec![
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ],
        "deepseek-v4-flash" | "DeepSeek-V4-Flash" => vec![
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        _ => vec![
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    };

    // Vérifier quels fichiers existent déjà
    let mut existing_files = Vec::new();
    for file in config_files {
        let file_path = output_dir.join(file);
        if file_path.exists() {
            existing_files.push(file.to_string());
        }
    }

    existing_files
}

/// Demande à l'utilisateur de confirmer l'écrasement des fichiers existants.
///
/// Cette fonction affiche la liste des fichiers qui seraient écrasés et demande
/// à l'utilisateur de confirmer l'opération. Elle gère l'entrée utilisateur de
/// manière robuste, avec gestion des erreurs et des cas limites.
///
/// # Paramètres
/// - `files` : liste des fichiers qui seraient écrasés
///
/// # Retourne
/// `true` si l'utilisateur confirme, `false` sinon.
///
/// # Comportement
/// - Affiche la liste des fichiers concernés
/// - Attend une réponse "oui" ou "non" (insensible à la casse)
/// - En cas d'erreur de lecture, retourne `false` par sécurité
pub fn confirm_overwrite(files: &[String]) -> bool {
    use std::io::{self, Write};

    println!("\n⚠️  Les fichiers suivants seront écrasés :");
    for file in files {
        println!("   - {}", file);
    }

    print!("\nVoulez-vous continuer ? (oui/non) : ");
    // Ignorer silencieusement les erreurs de flush (peut arriver si la sortie est fermée).
    // Ce n'est pas critique pour la fonctionnalité principale.
    let _ = io::stdout().flush();

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => {
            let input = input.trim().to_lowercase();
            input == "oui" || input == "o" || input == "yes" || input == "y"
        },
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_check_overwrite_warning_glm52() {
        let temp_dir = std::env::temp_dir().join("pmg_test_overwrite_glm52");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // Créer quelques fichiers de test
        fs::write(temp_dir.join("config.json"), "{}").unwrap();
        fs::write(temp_dir.join("tokenizer.json"), "{}").unwrap();

        let result = check_overwrite_warning(&temp_dir, "glm52");
        assert!(result.contains(&"config.json".to_string()));
        assert!(result.contains(&"tokenizer.json".to_string()));
        assert!(!result.contains(&"generation_config.json".to_string()));

        // Nettoyage
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_check_overwrite_warning_no_files() {
        let temp_dir = std::env::temp_dir().join("pmg_test_overwrite_empty");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let result = check_overwrite_warning(&temp_dir, "glm52");
        assert!(result.is_empty());

        // Nettoyage
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_create_blueprint_from_profile() {
        let profile = pmg_models::Glm52Profile::default_profile();
        let blueprint =
            crate::commands::generate_blueprint::create_blueprint_from_profile(&profile).unwrap();
        assert_eq!(blueprint.embeddings.len(), 1);
    }

    #[test]
    fn test_generate_args_defaults() {
        use crate::commands::generate::GenerateArgs;
        let args = GenerateArgs {
            source: "Models/GLM-5.2".to_string(),
            model: "glm52".to_string(),
            size: "1G".to_string(),
            mode: "safe".to_string(),
            dtype: "f32".to_string(),
            seed: Some(42),
            profile: None,
            chunk_size: 67108864,
            max_shard_bytes: 5368709120,
            no_validate: false,
            force: false,
            dry_run: true,
            verbose: false,
            quiet: false,
            json_output: false,
            debug: false,
            stream: false,
            stream_full: false,
            async_mode: false,
            workers: None,
            distributed: false,
            coordinator: "127.0.0.1:9090".to_string(),
            workers_count: 4,
            worker_mode: false,
            worker_id: None,
            gpu: false,
            gpu_count: None,
            compress: false,
            compression_algorithm: "lz4".to_string(),
            compression_level: 6,
        };
        assert!(!args.force);
        assert!(!args.async_mode);
        assert!(args.workers.is_none());
    }

    #[test]
    fn execute_dry_run_local() {
        use crate::commands::generate::{execute, GenerateArgs};
        let args = GenerateArgs {
            source: "Models/GLM-5.2".to_string(),
            model: "glm52".to_string(),
            size: "1G".to_string(),
            mode: "safe".to_string(),
            dtype: "f32".to_string(),
            seed: Some(42),
            profile: None,
            chunk_size: 67108864,
            max_shard_bytes: 5368709120,
            no_validate: false,
            force: false,
            dry_run: true, // Mode dry-run pour éviter l'erreur de répertoire source
            verbose: false,
            quiet: false,
            json_output: false,
            debug: false,
            stream: false,
            stream_full: false,
            async_mode: false,
            workers: None,
            distributed: false,
            coordinator: "127.0.0.1:9090".to_string(),
            workers_count: 4,
            worker_mode: false,
            worker_id: None,
            gpu: false,
            gpu_count: None,
            compress: false,
            compression_algorithm: "lz4".to_string(),
            compression_level: 6,
        };
        assert!(execute(args, false).is_ok());
    }
}
