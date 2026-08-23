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

//! Point d'entrée binaire de la CLI PMG.
//!
//! Ce fichier est le point d'entrée principal de l'application en ligne
//! de commande. Il analyse les arguments, redirige vers la sous-commande
//! appropriée et gère les erreurs avec les codes de sortie correspondants.
//!
//! # Architecture
//!
//! L'application suit le pattern `cli::Cli → Commands → execute()`.
//! Chaque commande est responsable de sa logique métier et retourne
//! un `Result<()>` qui est converti en code de sortie.
//!
//! # Codes de sortie
//!
//! | Code | Signification |
//! |------|---------------|
//! | 0    | Succès |
//! | 1    | Erreur générale |
//! | 2    | Argument invalide |
//! | 3    | Modèle invalide |
//! | 4    | Erreur d'entrée/sortie |
//! | 5    | Validation échouée |
//! | 6    | Comparaison incompatible |

use clap::Parser;
use std::process;

use pmg_cli::cli::{Cli, Commands};
use pmg_cli::cli_error::CliError;
use pmg_cli::exit_codes;

fn main() {
    // Analyse des arguments
    let cli = Cli::parse();

    // Affichage des options actives en mode débogage
    cli.global.display_active_options();

    // Exécution de la commande
    let result = match cli.command {
        Commands::Generate(args) => pmg_cli::commands::generate::execute(*args, cli.global.dry_run),
        Commands::Espec(args) => pmg_cli::commands::espec::execute(args, cli.global.verbose),
        Commands::Validate(args) => pmg_cli::commands::validate::execute(args, cli.global.verbose),
        Commands::Compare(args) => pmg_cli::commands::compare::execute(args, cli.global.verbose),
        Commands::Version(args) => pmg_cli::commands::version::execute(args, cli.global.verbose),
    };

    // Gestion des erreurs et codes de sortie
    match result {
        Ok(()) => {
            process::exit(exit_codes::SUCCESS as i32);
        },
        Err(err) => {
            // Conversion de l'erreur en CliError pour obtenir le code de sortie
            let cli_err = CliError::from(err);
            let exit_code = cli_err.exit_code();

            // Toujours afficher l'erreur sur stderr pour garantir la visibilité
            eprintln!("{}", cli_err);

            process::exit(exit_code as i32);
        },
    }
}
