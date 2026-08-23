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

//! Commande `help` — affichage de l'aide détaillée.
//!
//! Cette commande permet d'afficher l'aide générale ou l'aide
//! spécifique à une commande. Elle remplace l'ancien module `help`
//! pour être cohérente avec l'architecture des commandes.

/// Arguments de la commande `help`.
#[derive(Debug, clap::Args)]
pub struct HelpArgs {
    /// Commande pour laquelle afficher l'aide (optionnel).
    /// Si non spécifié, affiche l'aide générale.
    #[clap(required = false)]
    pub command: Option<String>,
}

use crate::output;

/// Exécute la commande `help`.
pub fn execute(args: HelpArgs) -> anyhow::Result<()> {
    match args.command {
        Some(cmd) => show_command_help(&cmd),
        None => show_general_help(),
    }
    Ok(())
}

/// Affiche l'aide générale de PMG.
fn show_general_help() {
    output::section("PMG — Pseudo Model Generator");

    output::subsection("Commandes disponibles");
    output::key_value("generate", "Génère un pseudo-modèle");
    output::key_value("validate", "Valide un pseudo-modèle généré");
    output::key_value("compare", "Compare deux pseudo-modèles");
    output::key_value("espec", "Affiche les spécifications d'un modèle");
    output::key_value("version", "Affiche la version de l'outil");
    output::key_value("help", "Affiche cette aide");

    output::blank_line();
    output::subsection("Options globales");
    output::key_value("-d, --dry-run", "Mode sec (pas de modification réelle)");
    output::key_value("--debug", "Active les messages de débogage");
    output::key_value("-v, --verbose", "Affiche des informations supplémentaires");
    output::key_value("-h, --help", "Affiche cette aide");

    output::blank_line();
    output::subsection("Pour plus d'informations sur une commande");
    output::key_value("pmg help <commande>", "Affiche l'aide spécifique");
    output::key_value("pmg <commande> --help", "Affiche l'aide de la commande");

    output::blank_line();
    output::subsection("Exemples");
    output::info("pmg generate --output ./model.safetensors --layers 12");
    output::info("pmg validate --model-path ./model.safetensors");
    output::info("pmg compare --original ./model1.safetensors --compared ./model2.safetensors");
    output::info("pmg espec --model-path ./model.safetensors");
    output::info("pmg version --verbose");
}

/// Affiche l'aide pour une commande spécifique.
fn show_command_help(command: &str) {
    match command {
        "generate" => {
            output::section("Commande generate");
            output::info("Génère un pseudo-modèle en utilisant les paramètres spécifiés.");

            output::blank_line();
            output::subsection("Usage");
            output::info("pmg generate [OPTIONS]");

            output::blank_line();
            output::subsection("Options");
            output::key_value(
                "--source <PATH>",
                "Chemin vers le dossier du modèle source (défaut: Models/GLM-5.2)",
            );
            output::key_value(
                "-m, --model <MODEL>",
                "Modèle cible (glm52, deepseek-v4-flash) (défaut: glm52)",
            );
            output::key_value(
                "-s, --size <SIZE>",
                "Taille cible du pseudo-modèle (ex: \"1G\", \"500M\") (défaut: 1G)",
            );
            output::key_value(
                "-M, --mode <MODE>",
                "Mode de génération (safe, realistic, compression, stress) (défaut: safe)",
            );
            output::key_value(
                "-d, --dtype <DTYPE>",
                "Type de données de sortie (f32, f16, bf16, i8) (défaut: f32)",
            );
            output::key_value(
                "-S, --seed <N>",
                "Graine aléatoire pour la reproductibilité",
            );
            output::key_value(
                "--profile <PATH>",
                "Chemin vers un fichier de profil personnalisé",
            );
            output::key_value(
                "--chunk-size <N>",
                "Taille des chunks en octets (défaut: 64Mo)",
            );
            output::key_value(
                "--max-shard-bytes <N>",
                "Taille maximale par shard en octets (défaut: 5Go)",
            );
            output::key_value("--no-validate", "Désactiver la validation post-génération");
            output::key_value(
                "-f, --force",
                "Forcer l'écrasement des fichiers existants sans confirmation",
            );
            output::key_value("-n, --dry-run", "Mode sec (simuler sans écrire)");
            output::key_value("-v, --verbose", "Affichage verbeux");
            output::key_value("--quiet", "Mode silencieux (pas de sortie)");
            output::key_value("--json-output", "Sortie au format JSON");
            output::key_value("--debug", "Mode debug (très verbeux)");
            output::key_value(
                "--stream",
                "Activer le mode streaming tension par tension (recommandé pour > 10 GB)",
            );
            output::key_value(
                "--stream-full",
                "Active le mode streaming complet (recommandé pour > 10 GB)",
            );
            output::key_value(
                "--async",
                "Active le mode asynchrone avec tokio (recommandé pour multi-core)",
            );
            output::key_value("-h, --help", "Affiche cette aide");
        },
        "validate" => {
            output::section("Commande validate");
            output::info("Valide un pseudo-modèle généré en vérifiant ses propriétés.");

            output::blank_line();
            output::subsection("Usage");
            output::info("pmg validate [OPTIONS]");

            output::blank_line();
            output::subsection("Options");
            output::key_value("-m, --model-path <PATH>", "Chemin vers le modèle à valider");
            output::key_value("-t, --tolerance <F>", "Tolérance statistique (défaut: 0.1)");
            output::key_value("--outlier-threshold <F>", "Seuil d'outliers (défaut: 3.0)");
            output::key_value("-v, --verbose", "Afficher le rapport détaillé");
            output::key_value("--format <FMT>", "Format de sortie (text, json)");
            output::key_value("-h, --help", "Affiche cette aide");
        },
        "compare" => {
            output::section("Commande compare");
            output::info("Compare deux pseudo-modèles en analysant leurs similarités.");

            output::blank_line();
            output::subsection("Usage");
            output::info("pmg compare [OPTIONS]");

            output::blank_line();
            output::subsection("Options");
            output::key_value("-o, --original <PATH>", "Chemin vers le modèle original");
            output::key_value("-c, --compared <PATH>", "Chemin vers le modèle à comparer");
            output::key_value("-t, --tolerance <F>", "Tolérance statistique (défaut: 0.1)");
            output::key_value("-v, --verbose", "Afficher les détails des différences");
            output::key_value("--format <FMT>", "Format de sortie (text, json)");
            output::key_value("-h, --help", "Affiche cette aide");
        },
        "espec" => {
            output::section("Commande espec");
            output::info("Affiche les spécifications détaillées d'un modèle.");

            output::blank_line();
            output::subsection("Usage");
            output::info("pmg espec [OPTIONS]");

            output::blank_line();
            output::subsection("Options");
            output::key_value("-m, --model-path <PATH>", "Chemin vers le modèle");
            output::key_value("-v, --verbose", "Afficher les détails complets");
            output::key_value("--format <FMT>", "Format de sortie (text, json)");
            output::key_value("-h, --help", "Affiche cette aide");
        },
        "version" => {
            output::section("Commande version");
            output::info("Affiche la version de l'outil PMG.");

            output::blank_line();
            output::subsection("Usage");
            output::info("pmg version [OPTIONS]");

            output::blank_line();
            output::subsection("Options");
            output::key_value("-v, --verbose", "Afficher les détails complets");
            output::key_value("-h, --help", "Affiche cette aide");
        },
        _ => {
            output::warning(&format!("Commande inconnue : {}", command));
            output::info("Tapez 'pmg help' pour voir les commandes disponibles.");
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_args_creation() {
        let args = HelpArgs { command: None };
        assert!(args.command.is_none());

        let args = HelpArgs {
            command: Some("generate".to_string()),
        };
        assert_eq!(args.command.unwrap(), "generate");
    }

    #[test]
    fn execute_with_no_command() {
        let args = HelpArgs { command: None };
        assert!(execute(args).is_ok());
    }

    #[test]
    fn execute_with_valid_command() {
        let args = HelpArgs {
            command: Some("generate".to_string()),
        };
        assert!(execute(args).is_ok());
    }

    #[test]
    fn execute_with_unknown_command() {
        let args = HelpArgs {
            command: Some("inconnue".to_string()),
        };
        assert!(execute(args).is_ok()); // Affiche un message mais ne plante pas
    }
}
