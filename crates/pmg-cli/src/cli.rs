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

//! Structure Clap principale de la CLI PMG.
//!
//! Ce module définit la structure de la ligne de commande avec toutes
//! les sous-commandes et options disponibles. Il utilise Clap pour
//! l'analyse des arguments et la génération automatique de l'aide.
//!
//! # Architecture
//!
//! La CLI est organisée en sous-commandes :
//! - `generate` : génération de pseudo-modèles
//! - `espec` : affichage des spécifications d'un modèle
//! - `validate` : validation de pseudo-modèles
//! - `compare` : comparaison de pseudo-modèles
//! - `version` : affichage de la version
//! - `help` : affichage de l'aide détaillée
//!
//! Les options globales (`--dry-run`, `--debug`, `--verbose`) sont
//! héritées par toutes les sous-commandes.

use clap::{Parser, Subcommand};

use crate::commands::{CompareArgs, EspecArgs, GenerateArgs, ValidateArgs, VersionArgs};
use crate::options::GlobalOptions;

/// Point d'entrée de la CLI PMG.
///
/// Parse les arguments et redirige vers la sous-commande appropriée.
///
/// # Exemple
///
/// ```bash
/// pmg generate --output model.safetensors --layers 12
/// pmg validate --model-path model.safetensors
/// pmg compare --original model1.safetensors --compared model2.safetensors
/// pmg espec --model-path model.safetensors
/// pmg version --verbose
/// pmg help generate
/// ```
#[derive(Debug, Parser)]
#[clap(
    name = "pmg",
    about = "Outil de génération et analyse de pseudo-modèles (PMG)",
    version
)]
pub struct Cli {
    /// Options globales héritées par toutes les sous-commandes.
    #[clap(flatten)]
    pub global: GlobalOptions,

    /// Sous-commande à exécuter.
    #[clap(subcommand)]
    pub command: Commands,
}

/// Sous-commandes disponibles dans la CLI PMG.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Génère un pseudo-modèle selon les paramètres spécifiés.
    ///
    /// Crée un fichier de modèle au format Safetensors avec les
    /// tenseurs générés aléatoirement selon la configuration donnée.
    Generate(Box<GenerateArgs>),

    /// Affiche les spécifications détaillées d'un modèle.
    ///
    /// Charge le modèle spécifié et affiche son architecture,
    /// ses tenseurs, et ses propriétés statistiques.
    Espec(EspecArgs),

    /// Valide un pseudo-modèle.
    ///
    /// Vérifie la cohérence statistique, structurelle et de distribution
    /// du modèle spécifié.
    Validate(ValidateArgs),

    /// Compare deux pseudo-modèles.
    ///
    /// Analyse les similarités et différences entre deux modèles
    /// et fournit un score de similarité.
    Compare(CompareArgs),

    /// Affiche la version de l'outil PMG.
    ///
    /// Sans option, affiche le numéro de version. Avec l'option
    /// `--verbose`, affiche les détails des composants.
    Version(VersionArgs),
}

impl Commands {
    /// Retourne le nom de la sous-commande.
    pub fn name(&self) -> &'static str {
        match self {
            Commands::Generate(_) => "generate",
            Commands::Espec(_) => "espec",
            Commands::Validate(_) => "validate",
            Commands::Compare(_) => "compare",
            Commands::Version(_) => "version",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_with_no_args() {
        // Sans arguments, Clap affiche l'aide et quitte.
        // Nous testons juste que la structure est correcte.
        let args = Cli::try_parse_from(["pmg"]);
        assert!(args.is_err()); // Pas de sous-commande → erreur
    }

    #[test]
    fn cli_parses_generate_command() {
        let args = Cli::try_parse_from([
            "pmg", "generate", "--model", "glm52", "--size", "1G", "--mode", "safe",
        ]);
        assert!(args.is_ok());
        let cli = args.unwrap();
        assert!(matches!(cli.command, Commands::Generate(_)));
    }

    #[test]
    fn cli_parses_global_options() {
        let args = Cli::try_parse_from([
            "pmg",
            "--dry-run",
            "--debug",
            "--verbose",
            "generate",
            "--model",
            "glm52",
            "--size",
            "1G",
        ]);
        assert!(args.is_ok());
        let cli = args.unwrap();
        assert!(cli.global.dry_run);
        assert!(cli.global.debug);
        assert!(cli.global.verbose);
    }

    #[test]
    fn cli_command_names() {
        let generate = Commands::Generate(Box::new(GenerateArgs {
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
            dry_run: false,
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
        }));
        assert_eq!(generate.name(), "generate");

        let version = Commands::Version(VersionArgs { verbose: false });
        assert_eq!(version.name(), "version");
    }
}
