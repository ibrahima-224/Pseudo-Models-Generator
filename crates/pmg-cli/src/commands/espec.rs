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

//! Commande `espec` — affichage des spécifications d'un modèle.
//!
//! Cette commande permet d'afficher les spécifications détaillées
//! d'un pseudo-modèle généré. Elle utilise `pmg-inspect` pour analyser
//! les métadonnées du modèle sans charger les poids.
//!
//! # Intégration
//!
//! La commande utilise `ModelInspector` de `pmg-inspect` pour lire
//! les métadonnées du modèle et générer un rapport structuré.
//!
//! # Options
//!
//! - `--model-path` : chemin vers le modèle à inspecter
//! - `--verbose` : afficher les détails complets
//! - `--format` : format de sortie (text, json)

use anyhow::{Context, Result};
use pmg_inspect::safetensors_inspector::inspect_single_safetensors_file;
use pmg_inspect::{InspectionLevel, ModelInspector};
use std::path::Path;

use crate::output;

/// Arguments de la commande `espec`.
#[derive(Debug, clap::Args)]
pub struct EspecArgs {
    /// Chemin vers le modèle.
    #[clap(short, long)]
    pub model_path: String,

    /// Afficher les détails complets.
    #[clap(short, long)]
    pub verbose: bool,

    /// Format de sortie (text, json).
    #[clap(long, default_value = "text")]
    pub format: String,
}

/// Exécute la commande `espec`.
pub fn execute(args: EspecArgs, global_verbose: bool) -> Result<()> {
    // Validation du chemin
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

    // Niveau d'inspection
    let level = if args.verbose || global_verbose {
        InspectionLevel::Verbose
    } else {
        InspectionLevel::Normal
    };

    // Création du chemin
    let model_path = Path::new(&args.model_path);

    // Vérification si c'est un fichier ou un répertoire
    if model_path.is_file() {
        // Vérification de l'extension .safetensors
        if model_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            output::error_with_cause_and_advice(
                "Le fichier n'est pas un fichier Safetensors",
                &format!("Chemin : {}", args.model_path),
                "Spécifiez un fichier .safetensors ou un répertoire de modèle",
            );
            return Err(anyhow::anyhow!(
                "Erreur PMG-2: Extension de fichier invalide"
            ));
        }

        // Inspection directe du fichier Safetensors
        let header = inspect_single_safetensors_file(model_path)
            .context("Échec de l'inspection du fichier Safetensors")?;

        // Affichage des résultats pour un fichier unique
        match args.format.as_str() {
            "json" => {
                let json_report = serde_json::json!({
                    "model_path": header.file_path.display().to_string(),
                    "tensor_count": header.tensor_count(),
                    "total_bytes": header.total_bytes(),
                    "file_size": header.file_size,
                    "header_size": header.header_size,
                    "density": header.density(),
                    "valid": header.validate(),
                });
                let json = serde_json::to_string_pretty(&json_report)
                    .context("Échec de la sérialisation en JSON")?;
                println!("{}", json);
            },
            _ => {
                if args.verbose || global_verbose {
                    println!("{}", header);
                } else {
                    output::section("Spécifications du fichier Safetensors");
                    output::key_value("Fichier", &header.file_path.display().to_string());
                    output::key_value_numeric("Tenseurs", header.tensor_count() as u64);
                    output::key_value_numeric("Taille totale (octets)", header.total_bytes());
                    output::key_value_numeric("Taille fichier (octets)", header.file_size);
                    output::key_value_numeric("Taille header (octets)", header.header_size);
                    output::key_value_numeric("Densité (%)", (header.density() * 100.0) as u64);
                }
            },
        }

        return Ok(());
    }

    // Cas répertoire : comportement existant avec ModelInspector
    let inspector = ModelInspector::new(model_path).with_level(level);

    // Inspection du modèle
    let report = inspector
        .inspect()
        .context("Échec de l'inspection du modèle")?;

    // Affichage des résultats
    match args.format.as_str() {
        "json" => {
            // Création d'une structure simplifiée pour la sérialisation JSON
            let json_report = serde_json::json!({
                "model_path": report.model_path.display().to_string(),
                "level": format!("{:?}", report.level),
                "config": report.config.is_some(),
                "tensor_count": report.safetensors_headers.len(),
                "total_parameters": report.structural.total_parameters,
                "total_memory_bytes": report.physical.total_memory_bytes,
                "architecture": format!("{}", report.architecture),
            });
            let json = serde_json::to_string_pretty(&json_report)
                .context("Échec de la sérialisation en JSON")?;
            println!("{}", json);
        },
        _ => {
            if args.verbose || global_verbose {
                println!("{}", report);
            } else {
                // Affichage condensé avec le module output
                output::section("Spécifications du modèle");
                output::key_value("Modèle", &report.model_path.display().to_string());
                output::key_value_numeric("Tenseurs", report.safetensors_headers.len() as u64);
                output::key_value_numeric("Paramètres", report.structural.total_parameters);
                output::key_value_numeric("Mémoire (octets)", report.physical.total_memory_bytes);
            }
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_empty_model_path() {
        let args = EspecArgs {
            model_path: String::new(),
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }

    #[test]
    fn validate_nonexistent_model() {
        let args = EspecArgs {
            model_path: "/chemin/inexistant/model.safetensors".to_string(),
            verbose: false,
            format: "text".to_string(),
        };
        assert!(execute(args, false).is_err());
    }
}
