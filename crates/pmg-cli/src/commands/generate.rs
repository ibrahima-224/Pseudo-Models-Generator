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

//! Commande `generate` — génération de pseudo-modèles.
//!
//! Cette commande permet de générer un pseudo-modèle en utilisant
//! les configurations et paramètres spécifiés. Elle intègre le pipeline
//! complet de génération via `pmg-generator`.
//!
//! # Intégration
//!
//! La commande utilise `execute_pipeline_output` de `pmg-generator` pour créer
//! un modèle déterministe selon les paramètres fournis. Elle gère
//! également la création de la structure de sortie via `pmg-io`.
//!
//! # Sécurité
//!
//! **Protection contre l'écrasement** : Cette commande inclut une protection
//! contre l'écrasement accidentel de fichiers existants. Par défaut, si des
//! fichiers de configuration existent déjà dans le répertoire de sortie,
//! l'utilisateur est invité à confirmer avant l'écrasement. Le flag `--force`
//! permet de passer cette vérification.
//!
//! # Options
//!
//! - `--output` : chemin de sortie pour le modèle généré
//! - `--layers` : nombre de couches (défaut: 12)
//! - `--hidden-size` : taille cachée (défaut: 4096)
//! - `--seed` : seed pour la génération déterministe (défaut: 42)
//! - `--source` : répertoire source du modèle (défaut: Models/GLM-5.2)
//! - `--force` : forcer l'écrasement des fichiers existants sans confirmation
//! - `--verbose` : afficher les détails de la génération
//! - `--dry-run` : mode sec (pas de génération réelle)

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::Result;
use pmg_core::DType;
use pmg_generator::{
    create_progress_tracker, execute_full_pipeline_streaming, execute_pipeline_output_streaming,
    GenerationPipeline, PipelineOutputConfig,
};
use pmg_io::output_structure::SourceModel;
use pmg_models::{DeepseekV4FlashProfile, Glm52Profile, ModelProfile};

use crate::commands::generate_blueprint::create_blueprint_from_profile;
use crate::commands::generate_distributed::execute_distributed;
use crate::commands::generate_helpers::{
    check_overwrite_warning, confirm_overwrite, execute_async_generation,
};
use crate::options::{format_size, parse_mode, parse_size};
use crate::output;

/// Arguments de la commande `generate`.
#[derive(Debug, clap::Args)]
pub struct GenerateArgs {
    /// Chemin vers le dossier du modèle source (contenant config.json, tokenizer.json, etc.).
    #[clap(long, default_value = "Models/GLM-5.2")]
    pub source: String,

    /// Modèle cible (glm52, deepseek-v4-flash).
    #[clap(long, short = 'm', default_value = "glm52")]
    pub model: String,

    /// Taille cible du pseudo-modèle (ex: "1G", "500M", "2T").
    #[clap(long, short = 's', default_value = "1G")]
    pub size: String,

    /// Mode de génération (safe, realistic, compression, stress).
    #[clap(long, short = 'M', default_value = "safe")]
    pub mode: String,

    /// Type de données de sortie (f32, f16, bf16, i8, etc.).
    #[clap(long, short = 'd', default_value = "f32")]
    pub dtype: String,

    /// Graine aléatoire pour la reproductibilité.
    #[clap(long, short = 'S')]
    pub seed: Option<u64>,

    /// Chemin vers un fichier de profil personnalisé.
    #[clap(long)]
    pub profile: Option<String>,

    /// Taille des chunks en octets (défaut: 64Mo).
    #[clap(long, default_value = "67108864")]
    pub chunk_size: u64,

    /// Taille maximale par shard en octets (défaut: 5Go).
    #[clap(long, default_value = "5368709120")]
    pub max_shard_bytes: u64,

    /// Désactiver la validation post-génération.
    #[clap(long)]
    pub no_validate: bool,

    /// Forcer l'écrasement des fichiers existants sans demander confirmation.
    #[clap(long, short = 'f')]
    pub force: bool,

    /// Mode sec (simuler sans écrire).
    #[clap(long, short = 'n')]
    pub dry_run: bool,

    /// Affichage verbeux.
    #[clap(long, short = 'v')]
    pub verbose: bool,

    /// Mode silencieux (pas de sortie).
    #[clap(long, help = "Mode silencieux (pas de sortie)")]
    pub quiet: bool,

    /// Sortie au format JSON.
    #[clap(long, help = "Sortie au format JSON")]
    pub json_output: bool,

    /// Mode debug (très verbeux).
    #[clap(long)]
    pub debug: bool,

    /// Activer le mode streaming tension par tension (recommandé pour les modèles > 10 GB).
    #[clap(long)]
    pub stream: bool,

    /// Active le mode streaming complet (recommandé pour > 10 GB).
    #[clap(
        long,
        help = "Active le mode streaming complet (recommandé pour > 10 GB)"
    )]
    pub stream_full: bool,

    /// Active le mode asynchrone avec tokio (recommandé pour multi-core).
    #[clap(
        long,
        help = "Active le mode asynchrone pour paralléliser la génération (recommandé pour multi-core)"
    )]
    pub async_mode: bool,

    /// Nombre de workers parallèles pour le mode asynchrone (défaut : nb cœurs).
    #[clap(long, help = "Nombre de workers parallèles pour le mode asynchrone")]
    pub workers: Option<usize>,

    /// Active le mode distribué pour la génération sur plusieurs machines.
    #[clap(
        long,
        help = "Active le mode distribué pour la génération sur plusieurs machines"
    )]
    pub distributed: bool,

    /// Adresse du coordinateur distribué (défaut: 127.0.0.1:9090).
    #[clap(
        long,
        default_value = "127.0.0.1:9090",
        help = "Adresse du coordinateur distribué"
    )]
    pub coordinator: String,

    /// Nombre de workers distribués (défaut: 4).
    #[clap(long, default_value = "4", help = "Nombre de workers distribués")]
    pub workers_count: usize,

    /// Mode worker (au lieu de coordinateur) pour le mode distribué.
    #[clap(
        long,
        help = "Mode worker (au lieu de coordinateur) pour le mode distribué"
    )]
    pub worker_mode: bool,

    /// Identifiant du worker pour le mode distribué.
    #[clap(long, help = "Identifiant du worker pour le mode distribué")]
    pub worker_id: Option<String>,

    /// Active l'accélération GPU (si disponible).
    #[clap(long, help = "Active l'accélération GPU (si disponible)")]
    pub gpu: bool,

    /// Nombre de GPU à utiliser (défaut: tous les GPU disponibles).
    #[clap(long, help = "Nombre de GPU à utiliser")]
    pub gpu_count: Option<usize>,

    /// Active la compression des tenseurs.
    #[clap(long, help = "Active la compression des tenseurs")]
    pub compress: bool,

    /// Algorithme de compression (lz4, zstd, none).
    #[clap(long, default_value = "lz4", help = "Algorithme de compression")]
    pub compression_algorithm: String,

    /// Niveau de compression (0-22 pour Zstd, 0-16 pour LZ4).
    #[clap(long, default_value = "6", help = "Niveau de compression")]
    pub compression_level: u32,
}

/// Exécute la commande `generate`.
pub fn execute(args: GenerateArgs, global_dry_run: bool) -> Result<()> {
    // 1. Parser la taille cible
    let target_size_bytes = parse_size(&args.size)?;

    // 2. Sélectionner le modèle
    let model_profile: Box<dyn ModelProfile> = match args.model.as_str() {
        "glm52" | "GLM-5.2" => Box::new(Glm52Profile::default_profile()),
        "deepseek-v4-flash" | "DeepSeek-V4-Flash" => {
            Box::new(DeepseekV4FlashProfile::default_profile())
        },
        _ => {
            output::error_with_cause_and_advice(
                "Modèle non supporté",
                &format!("Modèle demandé: {}", args.model),
                "Utilisez glm52 ou deepseek-v4-flash",
            );
            return Err(anyhow::anyhow!("Modèle non supporté: {}", args.model));
        },
    };

    // 3. Parser le dtype (utilisé pour validation)
    let _dtype = DType::from_str(&args.dtype.to_uppercase())?;

    // 4. Générer ou utiliser la seed
    let seed = args.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        // Génération de la seed basée sur le temps système.
        // En cas d'erreur (système antérieur à UNIX_EPOCH), utilise 0 comme fallback.
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs()
    });

    // 5. Parser le mode
    let mode = parse_mode(&args.mode)?;

    // 6. Vérification du mode sec (argument local ou global)
    let dry_run = args.dry_run || global_dry_run;

    // 7. Affichage des paramètres
    if args.verbose || dry_run {
        output::section("PARAMÈTRES DE GÉNÉRATION");
        output::key_value("Modèle", &args.model);
        output::key_value("Taille cible", &format_size(target_size_bytes));
        output::key_value("Dtype", &args.dtype);
        output::key_value("Mode", mode.display_name());
        output::key_value_numeric("Seed", seed);
        output::key_value("Source", &args.source);
        if let Some(ref profile_path) = args.profile {
            output::key_value("Profil", profile_path);
        }
        output::key_value_numeric("Chunk size", args.chunk_size);
        output::key_value_numeric("Max shard bytes", args.max_shard_bytes);
        output::key_value(
            "Validation",
            if args.no_validate {
                "désactivée"
            } else {
                "activée"
            },
        );
        output::blank_line();
    }

    if dry_run {
        output::section("MODE SEC — Simulation de la génération");
        output::key_value("Modèle", &args.model);
        output::key_value("Taille cible", &format_size(target_size_bytes));
        output::key_value("Dtype", &args.dtype);
        output::key_value("Seed", &seed.to_string());
        output::key_value("Mode", mode.display_name());
        return Ok(());
    }

    // 8. Créer la configuration de génération
    let _config = pmg_generator::GeneratorConfig::from_cli_args(
        seed,
        model_profile.model_family(),
        target_size_bytes,
        &args.dtype,
        mode,
        args.chunk_size,
        args.max_shard_bytes,
        !args.no_validate,
        dry_run,
        args.verbose,
        args.debug,
    )?;

    // 8.1 Support GPU (si activé)
    if args.gpu {
        #[cfg(feature = "gpu-acceleration")]
        {
            use pmg_gpu::GpuConfig;
            use pmg_gpu::GpuSupportManager;

            let gpu_config = GpuConfig {
                enabled: true,
                gpu_count: args.gpu_count,
                block_size: 256,
                shared_memory: 0,
            };

            let gpu_manager = GpuSupportManager::new(gpu_config);

            if gpu_manager.is_gpu_active() {
                output::info(&format!(
                    "GPU actif: {} devices disponibles",
                    gpu_manager.active_device_count()
                ));

                // Note: L'intégration complète des kernels GPU dans le pipeline
                // sera ajoutée dans une prochaine itération.
                // Pour l'instant, on utilise le fallback CPU.
                output::info(
                    "Mode GPU: utilisation du CPU pour le pipeline (GPU pas encore intégré)",
                );
            } else {
                output::warning("GPU demandé mais non disponible, fallback CPU");
            }
        }

        #[cfg(not(feature = "gpu-acceleration"))]
        {
            output::warning("Support GPU non compilé. Activez la feature 'gpu-acceleration'.");
        }
    }

    // 9. Créer le blueprint à partir du profil du modèle
    let blueprint = create_blueprint_from_profile(&*model_profile)?;

    // 10. Configuration de sortie
    let output_dir = std::env::current_dir()?;
    let source_dir = PathBuf::from(&args.source);

    // Vérification que le répertoire source existe
    if !source_dir.exists() {
        output::error_io(
            "Répertoire source introuvable",
            &format!("Chemin : {}", source_dir.display()),
            "Assurez-vous que le répertoire source contient les fichiers de configuration",
        );
        return Err(anyhow::anyhow!(
            "Erreur PMG-4: Répertoire source introuvable"
        ));
    }

    // 11. Vérification de l'écrasement de fichiers existants (sécurité)
    // Cette étape empêche l'écrasement accidentel de fichiers existants
    // sauf si l'utilisateur passe explicitement le flag --force
    if output_dir.exists() && !dry_run {
        let files_to_overwrite = check_overwrite_warning(&output_dir, &args.model);

        if !files_to_overwrite.is_empty() && !args.force {
            // Des fichiers seraient écrasés et --force n'est pas passé
            if !confirm_overwrite(&files_to_overwrite) {
                output::warning("Opération annulée par l'utilisateur");
                return Ok(());
            }
        } else if !files_to_overwrite.is_empty() && args.force {
            // Mode force : afficher un avertissement mais continuer
            output::warning(&format!(
                "Mode force activé : écrasement de {} fichier(s) existant(s)",
                files_to_overwrite.len()
            ));
        }
    }

    let pipeline_config = PipelineOutputConfig {
        output_dir,
        source_dir,
        // Sélection dynamique du modèle source selon l'argument --model
        source_model: match args.model.as_str() {
            "glm52" | "GLM-5.2" => SourceModel::Glm52,
            "deepseek-v4-flash" | "DeepSeek-V4-Flash" => SourceModel::DeepSeekV4Flash,
            _ => SourceModel::Glm52, // Par défaut, modèle GLM-5.2
        },
        seed,
        generator_version: "1.0.0".to_string(),
        generation_mode: mode.display_name().to_string(),
        target_size_bytes,
        dtype: args.dtype.clone(),
    };

    // 11. Pipeline de génération complet
    let pipeline = GenerationPipeline::full();

    // 12. Calcul du nombre total de tenseurs pour la progression
    let total_tensors = blueprint.embeddings.len()
        + blueprint
            .layers
            .iter()
            .map(|l| {
                l.attention.len()
                    + l.mlp.len()
                    + l.norms.len()
                    + l.hyper_connections.len()
                    + if l.moe_block.is_some() { 1 } else { 0 }
            })
            .sum::<usize>();

    // Création du suiveur de progression
    let progress_tracker =
        create_progress_tracker(total_tensors, args.quiet, args.json_output, args.verbose);

    // 13. Exécution du pipeline (5 modes)
    let result = if args.distributed {
        // Mode distribué pour la génération sur plusieurs machines
        execute_distributed(
            &blueprint,
            &args.coordinator,
            args.workers_count,
            args.worker_mode,
            args.worker_id.clone(),
            args.verbose,
            seed,
        )?;
        // Le mode distribué retourne Ok(()) directement
        return Ok(());
    } else if args.async_mode {
        // Mode asynchrone avec tokio (parallélisation multi-core)
        Ok(execute_async_generation(
            &args.model,
            &pipeline_config.output_dir,
            args.workers.unwrap_or_else(num_cpus::get),
            args.chunk_size as usize,
            seed,
            blueprint,
            args.verbose,
        )?)
    } else if args.stream_full {
        // Mode streaming complet (pipeline + écriture)
        if args.verbose {
            output::info("Mode streaming complet activé : pipeline + écriture en streaming");
        }
        execute_full_pipeline_streaming(
            &pipeline_config,
            blueprint,
            Some(Arc::new(progress_tracker.callback())),
        )
    } else if args.stream {
        // Mode streaming tension par tension (recommandé pour les grands modèles)
        if args.verbose {
            output::info("Mode streaming activé : écriture tension par tension");
        }
        execute_pipeline_output_streaming(
            &pipeline_config,
            blueprint,
            pipeline,
            Some(&progress_tracker.callback()),
        )
    } else {
        // Mode par défaut : streaming tension par tension (optimisé mémoire)
        // Utilise execute_pipeline_output_streaming pour éviter l'accumulation
        // de tous les tenseurs en mémoire, réduisant ainsi la consommation RAM.
        execute_pipeline_output_streaming(
            &pipeline_config,
            blueprint,
            pipeline,
            Some(&progress_tracker.callback()),
        )
    }
    .map_err(|e| {
        output::error_with_cause_and_advice(
            "Échec de la génération",
            &format!("Détails : {}", e),
            "Vérifiez les paramètres et l'espace disque disponible",
        );
        anyhow::anyhow!("Échec de la génération : {}", e)
    })?;

    // Finalisation de la progression
    progress_tracker.finish();

    // 13. Affichage des résultats
    if args.verbose {
        output::blank_line();
        output::success("Modèle généré avec succès !");
        output::subsection("Statistiques");
        output::key_value_numeric("Tenseurs générés", result.tensor_count as u64);
        output::key_value_numeric("Paramètres totaux", result.parameter_count);
        output::key_value_numeric("Taille réelle (octets)", result.actual_size_bytes);
        output::key_value("Validation", &format!("{:?}", result.validation));
    } else {
        output::success(&format!(
            "Modèle généré avec {} paramètres",
            result.parameter_count
        ));
    }

    Ok(())
}
