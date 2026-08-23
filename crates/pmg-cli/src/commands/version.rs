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

//! Commande `version` — affichage de la version.
//!
//! Cette commande permet d'afficher la version de l'outil PMG
//! et des informations détaillées sur les composants.
//!
//! # Options
//!
//! - `--verbose` : afficher les détails complets des composants

use crate::output;

/// Arguments de la commande `version`.
#[derive(Debug, clap::Args)]
pub struct VersionArgs {
    /// Afficher les détails complets.
    #[clap(short, long)]
    pub verbose: bool,
}

/// Exécute la commande `version`.
pub fn execute(args: VersionArgs, global_verbose: bool) -> anyhow::Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");
    let authors = env!("CARGO_PKG_AUTHORS");
    let license = env!("CARGO_PKG_LICENSE");
    let description = env!("CARGO_PKG_DESCRIPTION");
    let repository = env!("CARGO_PKG_REPOSITORY");

    if args.verbose || global_verbose {
        output::section(&format!("{} — {}", name, description));
        output::key_value("Version", version);
        output::key_value("Auteurs", authors);
        output::key_value("Licence", license);
        output::key_value("Dépôt", repository);

        output::blank_line();
        output::subsection("Composants");
        output::key_value("pmg-math", "Moteur mathématique et statistique");
        output::key_value("pmg-core", "Modèles de données et types de base");
        output::key_value("pmg-blueprint", "Architecture et spécifications");
        output::key_value("pmg-generator", "Pipeline de génération");
        output::key_value("pmg-io", "Entrée/sortie Safetensors");
        output::key_value("pmg-inspect", "Inspection de modèles");
        output::key_value("pmg-validate", "Validation de modèles");
        output::key_value("pmg-compare", "Comparaison de modèles");
        output::key_value("pmg-meta", "Métadonnées et index");
        output::key_value("pmg-cli", "Interface utilisateur en ligne de commande");

        output::blank_line();
        output::subsection("Informations techniques");
        output::key_value("Format de sortie", "Safetensors");
        output::key_value("Déterminisme", "Garanti par seed");
        output::key_value("Validation", "Méthode Zero-Payload");
    } else {
        output::info(&format!("{} v{}", name, version));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_args_default() {
        let args = VersionArgs { verbose: false };
        assert!(!args.verbose);
    }

    #[test]
    fn execute_returns_ok() {
        let args = VersionArgs { verbose: false };
        assert!(execute(args, false).is_ok());
    }

    #[test]
    fn execute_verbose_returns_ok() {
        let args = VersionArgs { verbose: true };
        assert!(execute(args, false).is_ok());
    }
}
