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

//! Commande `compare` — comparaison de pseudo-modèles.
//!
//! Cette commande permet de comparer deux pseudo-modèles générés en analysant
//! leurs similarités et différences. Elle utilise `pmg-compare` pour effectuer
//! la comparaison metadata-only.
//!
//! # Intégration
//!
//! La commande utilise `ModelComparator` de `pmg-compare` pour charger
//! les modèles et générer un rapport de comparaison structuré.
//!
//! # Options
//!
//! - `--original` : chemin vers le modèle original
//! - `--compared` : chemin vers le modèle à comparer
//! - `--tolerance` : tolérance pour les comparaisons statistiques (défaut: 0.1)
//! - `--verbose` : afficher les détails des différences
//! - `--format` : format de sortie (text, json)

use anyhow::Result;
use pmg_compare::comparison::{ComparisonReport, ComparisonStatus};
use pmg_compare::dtype_compare::{compare_dtypes, DtypeInfo};
use pmg_compare::shape_compare::{compare_shapes, ShapeInfo};
use pmg_compare::shard_compare::{compare_sharding, ShardConfig, ShardInfo};
use pmg_compare::tensor_compare::{compare_tensors, TensorInfo};
use pmg_io::safetensors::read_header_from;
use std::fs::File;
use std::io::BufReader;

use crate::cli_error::CliError;
use crate::output;

/// Arguments de la commande `compare`.
#[derive(Debug, clap::Args)]
pub struct CompareArgs {
    /// Chemin vers le modèle original.
    #[clap(short, long)]
    pub original: String,

    /// Chemin vers le modèle à comparer.
    #[clap(short, long)]
    pub compared: String,

    /// Tolérance pour les comparaisons statistiques.
    #[clap(short, long, default_value = "0.1")]
    pub tolerance: f64,

    /// Afficher les détails des différences.
    #[clap(short, long)]
    pub verbose: bool,

    /// Format de sortie (text, json).
    #[clap(long, default_value = "text")]
    pub format: String,
}

/// Exécute la commande `compare`.
pub fn execute(args: CompareArgs, global_verbose: bool) -> Result<()> {
    // Validation des chemins
    if args.original.is_empty() || args.compared.is_empty() {
        output::error_with_cause_and_advice(
            "Les chemins des modèles ne peuvent pas être vides",
            "Argument --original ou --compared manquant",
            "Spécifiez les chemins avec --original et --compared",
        );
        return Err(anyhow::anyhow!(
            "Erreur PMG-2: Chemins des modèles invalides"
        ));
    }

    // Vérification que les fichiers existent
    if !std::path::Path::new(&args.original).exists() {
        output::error_io(
            "Le fichier original n'existe pas",
            &format!("Chemin : {}", args.original),
            "Vérifiez le chemin spécifié",
        );
        return Err(anyhow::anyhow!(
            "Erreur PMG-4: Fichier original introuvable"
        ));
    }

    if !std::path::Path::new(&args.compared).exists() {
        output::error_io(
            "Le fichier comparé n'existe pas",
            &format!("Chemin : {}", args.compared),
            "Vérifiez le chemin spécifié",
        );
        return Err(anyhow::anyhow!("Erreur PMG-4: Fichier comparé introuvable"));
    }

    // Chargement des métadonnées du fichier original
    let original_file = File::open(&args.original).map_err(|e| {
        CliError::io_error(format!("Impossible d'ouvrir le fichier original: {}", e))
    })?;
    let mut original_reader = BufReader::new(original_file);
    let original_metadata = read_header_from(&mut original_reader)
        .map_err(|e| CliError::io_error(format!("Erreur de lecture du fichier original: {}", e)))?;

    // Chargement des métadonnées du fichier comparé
    let compared_file = File::open(&args.compared).map_err(|e| {
        CliError::io_error(format!("Impossible d'ouvrir le fichier comparé: {}", e))
    })?;
    let mut compared_reader = BufReader::new(compared_file);
    let compared_metadata = read_header_from(&mut compared_reader)
        .map_err(|e| CliError::io_error(format!("Erreur de lecture du fichier comparé: {}", e)))?;

    // Extraction des noms de tenseurs depuis les métadonnées réelles
    let original_tensor_names: Vec<String> = original_metadata
        .metadata_only()
        .into_iter()
        .map(|(name, _)| name.to_string())
        .collect();
    let compared_tensor_names: Vec<String> = compared_metadata
        .metadata_only()
        .into_iter()
        .map(|(name, _)| name.to_string())
        .collect();

    // Créer les TensorInfo pour la comparaison
    let original_tensors: Vec<TensorInfo> = original_tensor_names
        .into_iter()
        .map(TensorInfo::new)
        .collect();
    let compared_tensors: Vec<TensorInfo> = compared_tensor_names
        .into_iter()
        .map(TensorInfo::new)
        .collect();

    // Comparaison des métadonnées (noms de tenseurs)
    let tensor_result = compare_tensors(&original_tensors, &compared_tensors);

    // EXTRACTION DES INFORMATIONS RÉELLES pour les comparaisons
    // Créer les ShapeInfo à partir des métadonnées réelles
    // Conversion de Vec<u64> en Vec<usize> pour les dimensions
    let original_shapes: Vec<ShapeInfo> = original_metadata
        .metadata_only()
        .iter()
        .map(|(name, entry)| {
            let dims: Vec<usize> = entry.shape.iter().map(|&d| d as usize).collect();
            ShapeInfo::new(name.to_string(), dims)
        })
        .collect();
    let compared_shapes: Vec<ShapeInfo> = compared_metadata
        .metadata_only()
        .iter()
        .map(|(name, entry)| {
            let dims: Vec<usize> = entry.shape.iter().map(|&d| d as usize).collect();
            ShapeInfo::new(name.to_string(), dims)
        })
        .collect();

    // Créer les DtypeInfo à partir des métadonnées réelles
    // Conversion de DType en String pour la comparaison
    let original_dtypes: Vec<DtypeInfo> = original_metadata
        .metadata_only()
        .iter()
        .map(|(name, entry)| DtypeInfo::new(name.to_string(), format!("{:?}", entry.dtype)))
        .collect();
    let compared_dtypes: Vec<DtypeInfo> = compared_metadata
        .metadata_only()
        .iter()
        .map(|(name, entry)| DtypeInfo::new(name.to_string(), format!("{:?}", entry.dtype)))
        .collect();

    // Comparaison des formes (shapes)
    let shape_result = compare_shapes(&original_shapes, &compared_shapes);

    // Comparaison des types de données (dtypes)
    let dtype_result = compare_dtypes(&original_dtypes, &compared_dtypes);

    // Comparaison des shards (basée sur la taille du fichier et le nombre de tenseurs)
    // Créer les ShardInfo pour chaque tenseur
    let original_shards: Vec<ShardInfo> = original_metadata
        .metadata_only()
        .iter()
        .map(|(name, entry)| {
            let byte_size = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
            ShardInfo::new(name.to_string(), 0, byte_size)
        })
        .collect();
    let compared_shards: Vec<ShardInfo> = compared_metadata
        .metadata_only()
        .iter()
        .map(|(name, entry)| {
            let byte_size = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
            ShardInfo::new(name.to_string(), 0, byte_size)
        })
        .collect();

    let original_shard_config = ShardConfig::new(1, original_shards);
    let compared_shard_config = ShardConfig::new(1, compared_shards);
    let shard_result = compare_sharding(&original_shard_config, &compared_shard_config);

    // Calculer les scores individuels pour chaque aspect
    let config_score = if original_metadata.header_size == compared_metadata.header_size {
        1.0
    } else {
        0.5
    };

    let architecture_score = if original_metadata.tensor_count() == compared_metadata.tensor_count()
    {
        1.0
    } else {
        0.5
    };

    // Calculer un score global avec les scores réels
    let global_score = pmg_compare::score::calculate_global_score(
        config_score,
        architecture_score,
        tensor_result.similarity_score,
        shape_result.similarity_score,
        dtype_result.similarity_score,
        shard_result.similarity_score,
        0, // blocking_anomalies
    );

    // Créer les résultats de comparaison pour config et architecture
    // Basés sur les métadonnées réelles des fichiers
    let config_result = pmg_compare::config_compare::ConfigComparisonResult {
        similarity_score: config_score,
        status: if config_score >= 0.9 {
            ComparisonStatus::Match
        } else if config_score >= 0.5 {
            ComparisonStatus::Partial
        } else {
            ComparisonStatus::Different
        },
        differences: vec![],
        parameter_count: 2,
        matching_count: if config_score >= 0.9 { 2 } else { 1 },
    };

    let architecture_result = pmg_compare::architecture_compare::ArchitectureComparisonResult {
        architecture_type: if architecture_score >= 0.9 {
            pmg_compare::architecture_compare::ArchitectureType::Identical
        } else if architecture_score >= 0.5 {
            pmg_compare::architecture_compare::ArchitectureType::Compatible
        } else {
            pmg_compare::architecture_compare::ArchitectureType::Different
        },
        compatibility_score: architecture_score,
        status: if architecture_score >= 0.9 {
            ComparisonStatus::Match
        } else if architecture_score >= 0.5 {
            ComparisonStatus::Partial
        } else {
            ComparisonStatus::Different
        },
        differences: vec![],
        properties_compared: 2,
        properties_compatible: if architecture_score >= 0.9 { 2 } else { 1 },
    };

    // Déterminer le statut global basé sur le score
    let global_status = if global_score.percentage >= 90.0 {
        ComparisonStatus::Match
    } else if global_score.percentage >= 50.0 {
        ComparisonStatus::Partial
    } else {
        ComparisonStatus::Different
    };

    // Créer un rapport de comparaison avec les résultats réels
    let report = ComparisonReport::new(
        args.original.clone(),
        args.compared.clone(),
        config_result,
        architecture_result,
        tensor_result,
        shape_result,
        dtype_result,
        shard_result,
        global_score,
        global_status,
        vec![],
    );

    output::info("Poids : NON COMPARÉS");
    // Affichage des résultats
    match args.format.as_str() {
        "json" => {
            let json = format!(
                "{{\n  \"original\": \"{}\",\n  \"compared\": \"{}\",\n  \"global_score\": {:.6},\n  \"global_status\": \"{}\",\n  \"tensor_similarity_score\": {:.6},\n  \"tensor_count\": {},\n  \"tensor_common\": {},\n  \"tensor_original_only\": {},\n  \"tensor_compared_only\": {}\n}}",
                report.original_model_name,
                report.compared_model_name,
                report.global_score.percentage,
                report.global_status,
                report.tensor_result.similarity_score,
                report.tensor_result.total_tensors,
                report.tensor_result.common_tensors,
                report.tensor_result.original_only,
                report.tensor_result.compared_only
            );
            println!("{}", json);
        },
        _ => {
            if args.verbose || global_verbose {
                output::section("RAPPORT DE COMPARAISON");
                output::key_value("Modèle original", &report.original_model_name);
                output::key_value("Modèle comparé", &report.compared_model_name);
                output::key_value_decimal("Score global", report.global_score.percentage);
                output::key_value("Statut global", &format!("{}", report.global_status));
                output::blank_line();

                output::section("COMPARAISON DES TENSEURS");
                output::key_value_decimal(
                    "Score de similarité",
                    report.tensor_result.similarity_score,
                );
                output::key_value("Statut", &format!("{}", report.tensor_result.status));
                output::key_value_numeric(
                    "Total tenseurs",
                    report.tensor_result.total_tensors as u64,
                );
                output::key_value_numeric(
                    "Tenseurs communs",
                    report.tensor_result.common_tensors as u64,
                );
                output::key_value_numeric(
                    "Uniquement dans l'original",
                    report.tensor_result.original_only as u64,
                );
                output::key_value_numeric(
                    "Uniquement dans le comparé",
                    report.tensor_result.compared_only as u64,
                );
                output::blank_line();

                // Afficher les différences des tenseurs
                for diff in &report.tensor_result.differences {
                    output::info(&format!("[{}] {}", diff.diff_type, diff.description));
                    output::key_value("Chemin", &diff.path);
                    if let Some(orig) = &diff.original_value {
                        output::key_value("Original", orig);
                    }
                    if let Some(comp) = &diff.compared_value {
                        output::key_value("Comparé", comp);
                    }
                    output::blank_line();
                }
            } else {
                let status_str = match report.global_status {
                    ComparisonStatus::Match => "✅ IDENTIQUE",
                    ComparisonStatus::Partial => "⚠️ PARTIEL",
                    ComparisonStatus::Different => "❌ DIFFÉRENT",
                    ComparisonStatus::Unknown => "❓ INCONNU",
                };
                output::success(&format!(
                    "{} | Score : {:.6} | {} tenseurs | {} différences",
                    status_str,
                    report.global_score.percentage,
                    report.tensor_result.total_tensors,
                    report.tensor_result.differences.len()
                ));
            }
        },
    }

    // Code de sortie
    if report.global_score.percentage > 90.0 {
        output::success("Comparaison réussie");
        Ok(())
    } else {
        output::error_comparison(
            "Les modèles sont trop différents",
            &format!("Score global : {:.6}", report.global_score.percentage),
            "Vérifiez que les modèles sont comparables",
        );
        Err(anyhow::anyhow!("Erreur PMG-6: Modèles trop différents"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_empty_original() {
        let args = CompareArgs {
            original: String::new(),
            compared: "model2.safetensors".to_string(),
            tolerance: 0.1,
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }

    #[test]
    fn validate_empty_compared() {
        let args = CompareArgs {
            original: "model1.safetensors".to_string(),
            compared: String::new(),
            tolerance: 0.1,
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }

    #[test]
    fn validate_nonexistent_original() {
        let args = CompareArgs {
            original: "/chemin/inexistant/model1.safetensors".to_string(),
            compared: "model2.safetensors".to_string(),
            tolerance: 0.1,
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }
}
