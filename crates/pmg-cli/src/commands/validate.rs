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

//! Commande `validate` — validation de pseudo-modèles.
//!
//! Cette commande permet de valider un pseudo-modèle généré en vérifiant
//! ses propriétés statistiques, structurelles et de distribution.
//! Elle utilise `pmg-validate` pour effectuer la validation complète.
//!
//! # Intégration
//!
//! La commande utilise `ModelValidator` de `pmg-validate` pour charger
//! le modèle et effectuer les vérifications nécessaires.
//!
//! # Options
//!
//! - `--model-path` : chemin vers le modèle à valider
//! - `--tolerance` : tolérance pour les comparaisons statistiques (défaut: 0.1)
//! - `--outlier-threshold` : seuil de détection d'outliers (défaut: 3.0)
//! - `--verbose` : afficher le rapport détaillé
//! - `--format` : format de sortie (text, json)

use anyhow::Result;
use pmg_validate::{generate_text_report, ValidationCategory, ValidationConfig, ValidationIssue};
use std::fs::File;
use std::io::BufReader;

use crate::cli_error::CliError;
use crate::output;
use pmg_io::safetensors::{DType, SafetensorsReader};

/// Arguments de la commande `validate`.
#[derive(Debug, clap::Args)]
pub struct ValidateArgs {
    /// Chemin vers le modèle à valider.
    #[clap(short, long)]
    pub model_path: String,

    /// Tolérance pour les comparaisons statistiques.
    #[clap(short, long, default_value = "0.1")]
    pub tolerance: f64,

    /// Seuil de détection d'outliers (nombre d'écarts-types).
    #[clap(long, default_value = "3.0")]
    pub outlier_threshold: f64,

    /// Afficher le rapport détaillé.
    #[clap(short, long)]
    pub verbose: bool,

    /// Format de sortie (text, json).
    #[clap(long, default_value = "text")]
    pub format: String,
}

/// Exécute la commande `validate` en mode Zero-Payload.
///
/// Cette implémentation respecte le principe Zero-Payload : elle ne lit que
/// les métadonnées (header JSON) du fichier Safetensors, sans jamais
/// charger les données binaires (payload) en mémoire.
///
/// # Paramètres
/// - `args` : arguments de la commande
/// - `global_verbose` : mode verbeux global
///
/// # Retour
/// `Result<()>` avec error si la validation échoue ou si le fichier est invalide.
pub fn execute(args: ValidateArgs, global_verbose: bool) -> Result<()> {
    // Validation basique du chemin
    if args.model_path.is_empty() {
        output::error_with_cause_and_advice(
            "Le chemin du modèle ne peut pas être vide",
            "Argument --model-path manquant ou vide",
            "Spécifiez un chemin avec --model-path",
        );
        return Err(anyhow::anyhow!("Erreur PMG-2: Chemin du modèle invalide"));
    }

    // Vérification que le fichier existe
    if !std::path::Path::new(&args.model_path).exists() {
        output::error_io(
            "Le fichier modèle n'existe pas",
            &format!("Chemin : {}", args.model_path),
            "Vérifiez le chemin spécifié",
        );
        return Err(anyhow::anyhow!("Erreur PMG-4: Fichier modèle introuvable"));
    }

    // Création de la configuration de validation
    let config = ValidationConfig {
        outlier_threshold: args.outlier_threshold,
        statistical_tolerance: args.tolerance,
        ..ValidationConfig::default()
    };

    // Chargement du fichier safetensors en mode header-only (Zero-Payload)
    let file = File::open(&args.model_path)
        .map_err(|e| CliError::io_error(format!("Impossible d'ouvrir le fichier modèle: {}", e)))?;
    let mut reader = BufReader::new(file);

    // Lecture des métadonnées du header uniquement (Zero-Payload)
    let safetensors_reader: SafetensorsReader = pmg_io::safetensors::read_header_from(&mut reader)
        .map_err(|e| CliError::io_error(format!("Erreur de lecture du header: {}", e)))?;

    // Extraction des métadonnées des tenseurs
    let metadata = safetensors_reader.metadata_only();

    // Vérification qu'il y a au moins un tenseur
    if metadata.is_empty() {
        return Err(CliError::invalid_model("Aucun tenseur trouvé dans le fichier modèle").into());
    }

    // Création des résultats de validation pour chaque tenseur
    let mut tensor_results = Vec::new();
    let mut summary = pmg_validate::ValidationSummary::default();

    for (name, entry) in &metadata {
        let mut issues = Vec::new();

        // Validation structurelle basée sur les métadonnées du header
        // Vérification que les offsets sont cohérents avec la forme
        let element_count: u64 = entry.shape.iter().product();
        let expected_bytes = element_count
            .checked_mul(entry.dtype.size_bytes() as u64)
            .unwrap_or(0);

        let actual_bytes = entry.data_offsets[1].saturating_sub(entry.data_offsets[0]);

        if actual_bytes != expected_bytes {
            issues.push(ValidationIssue {
                category: ValidationCategory::Structural,
                severity: pmg_validate::severity::Severity::Error,
                message: format!(
                    "Incohérence de taille pour le tenseur '{}': {} octets attendus, {} octets trouvés",
                    name, expected_bytes, actual_bytes
                ),
                tensor_path: Some(name.to_string()),
            });
        }

        // Vérification que les offsets sont dans les limites du buffer
        if entry.data_offsets[1] > safetensors_reader.buffer_size {
            issues.push(ValidationIssue {
                category: ValidationCategory::Structural,
                severity: pmg_validate::severity::Severity::Error,
                message: format!(
                    "Offsets hors limites pour le tenseur '{}': fin {} > buffer_size {}",
                    name, entry.data_offsets[1], safetensors_reader.buffer_size
                ),
                tensor_path: Some(name.to_string()),
            });
        }

        // Vérification que le dtype est supporté
        match entry.dtype {
            DType::F32
            | DType::F16
            | DType::BF16
            | DType::I32
            | DType::I64
            | DType::U32
            | DType::U64
            | DType::I8
            | DType::I16
            | DType::U8
            | DType::U16
            | DType::F8E4M3
            | DType::F8E5M2 => {
                // Dtype supporté
            },
            _ => {
                issues.push(ValidationIssue {
                    category: ValidationCategory::Structural,
                    severity: pmg_validate::severity::Severity::Warning,
                    message: format!(
                        "Dtype non supporté pour la validation des données: {:?}",
                        entry.dtype
                    ),
                    tensor_path: Some(name.to_string()),
                });
            },
        }

        // Information sur les validations non disponibles en mode Zero-Payload
        if config.check_statistical {
            issues.push(ValidationIssue {
                category: ValidationCategory::Statistical,
                severity: pmg_validate::severity::Severity::Info,
                message: "Validation statistique non disponible en mode Zero-Payload (nécessite la lecture des données)".to_string(),
                tensor_path: Some(name.to_string()),
            });
        }

        if config.check_distribution {
            issues.push(ValidationIssue {
                category: ValidationCategory::Distribution,
                severity: pmg_validate::severity::Severity::Info,
                message: "Validation de distribution non disponible en mode Zero-Payload (nécessite la lecture des données)".to_string(),
                tensor_path: Some(name.to_string()),
            });
        }

        if config.check_outliers {
            issues.push(ValidationIssue {
                category: ValidationCategory::Outlier,
                severity: pmg_validate::severity::Severity::Info,
                message: "Détection d'outliers non disponible en mode Zero-Payload (nécessite la lecture des données)".to_string(),
                tensor_path: Some(name.to_string()),
            });
        }

        // Comptage des issues
        for issue in &issues {
            match issue.severity {
                pmg_validate::severity::Severity::Info => summary.info_count += 1,
                pmg_validate::severity::Severity::Warning => summary.warning_count += 1,
                pmg_validate::severity::Severity::Error => summary.error_count += 1,
                pmg_validate::severity::Severity::Critical => summary.critical_count += 1,
            }
        }

        tensor_results.push(pmg_validate::TensorValidationResult {
            path: name.to_string(),
            issues,
        });
    }

    // Création du résultat de validation global
    let result = pmg_validate::ValidationResult {
        model_name: "pseudo_model".to_string(),
        tensor_count: metadata.len(),
        tensor_results,
        summary,
    };

    // Affichage des résultats
    match args.format.as_str() {
        "json" => {
            let json = pmg_validate::generate_json_report(&result);
            println!("{}", json);
        },
        _ => {
            if args.verbose || global_verbose {
                let report = generate_text_report(&result);
                println!("{}", report);
            } else {
                let summary = pmg_validate::generate_console_summary(&result);
                println!("{}", summary);
            }
        },
    }

    // Code de sortie
    if result.summary.is_valid() {
        output::success("Validation réussie (mode Zero-Payload - métadonnées uniquement)");
        Ok(())
    } else {
        output::error_validation(
            "Validation échouée",
            &format!("{} erreurs détectées", result.summary.error_count),
            "Corrigez les erreurs mentionnées dans le rapport",
        );
        Err(anyhow::anyhow!("Erreur PMG-5: Validation échouée"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Crée un fichier Safetensors minimal valide pour les tests.
    fn create_test_safetensors() -> Vec<u8> {
        let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let json_len = header_json.len();
        let padding = (8 - (json_len % 8)) % 8;
        let padded_json = format!("{}{}", header_json, " ".repeat(padding));
        let header_size = padded_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_size.to_le_bytes());
        file.extend_from_slice(padded_json.as_bytes());
        // Ajoute 24 octets de données (payload)
        file.extend_from_slice(&[0u8; 24]);

        file
    }

    #[test]
    fn validate_empty_model_path() {
        let args = ValidateArgs {
            model_path: String::new(),
            tolerance: 0.1,
            outlier_threshold: 3.0,
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }

    #[test]
    fn validate_nonexistent_model() {
        let args = ValidateArgs {
            model_path: "/chemin/inexistant/model.safetensors".to_string(),
            tolerance: 0.1,
            outlier_threshold: 3.0,
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }

    #[test]
    fn validate_zero_payload_mode() {
        // Crée un fichier temporaire pour le test
        let test_data = create_test_safetensors();
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_validate_zero_payload.safetensors");

        // Écriture du fichier de test
        std::fs::write(&test_file, &test_data).expect("Impossible d'écrire le fichier de test");

        let args = ValidateArgs {
            // Conversion du chemin en chaîne UTF-8 avec gestion des caractères invalides.
            // to_string_lossy() remplace les caractères invalides par U+FFFD.
            model_path: test_file.to_string_lossy().to_string(),
            tolerance: 0.1,
            outlier_threshold: 3.0,
            verbose: true,
            format: "text".to_string(),
        };

        // La validation devrait réussir en mode Zero-Payload
        let result = execute(args, false);

        // Nettoyage
        let _ = std::fs::remove_file(&test_file);

        // Vérification que la validation a réussi
        assert!(result.is_ok(), "La validation Zero-Payload devrait réussir");
    }
}
