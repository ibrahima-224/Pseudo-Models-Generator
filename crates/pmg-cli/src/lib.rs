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

//! Crate `pmg-cli` — interface utilisateur en ligne de commande (binaire).
//!
//! Point d'entrée du projet : parse les arguments (clap), orchestre les
//! commandes, affiche les messages en français et traduit les erreurs internes
//! en codes de sortie (D3 : `0` succès, `1` erreur générale, `2` argument
//! invalide, `3` modèle invalide, `4` erreur I/O, `5` validation échouée,
//! `6` comparaison incompatible).
//!
//! ## Responsabilité
//!
//! - commandes : `help`, `generate`, `espec`, `validate`, `compare`, `version` ;
//! - options : `-v/--verbose`, `--debug`, `--dry-run`, `-h/-d` ;
//! - seule crate avec `anyhow` et `std::process::exit` ;
//! - **interdit** : algorithmes métier (orchestration uniquement).
//!
//! ## Dépendances
//!
//! `pmg-generator`, `pmg-inspect`, `pmg-validate`, `pmg-compare`, `pmg-meta`
//! (graphe normatif `docs/architecture/02-workspace-et-crates.md` §4 ; toutes
//! les autres crates métier sont atteignables transitivement).
//!
//! # Exemple
//!
//! ```
//! // Point d'entrée : voir src/main.rs
//! ```

pub mod cli;
pub mod cli_error;
pub mod commands;
pub mod error_display;
pub mod exit_codes;
pub mod options;
pub mod output;

// Ré-exports des types publics
pub use cli::{Cli, Commands};
pub use commands::{CompareArgs, EspecArgs, GenerateArgs, ValidateArgs, VersionArgs};
pub use error_display::PmgError;
pub use options::GlobalOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_structure_compiles() {
        // Vérifie que la structure CLI compile correctement
        let _ = std::mem::size_of::<Cli>();
    }

    #[test]
    fn exit_codes_are_unique() {
        let codes = [
            exit_codes::SUCCESS,
            exit_codes::GENERAL_ERROR,
            exit_codes::INVALID_ARGUMENT,
            exit_codes::INVALID_MODEL,
            exit_codes::IO_ERROR,
            exit_codes::VALIDATION_FAILED,
            exit_codes::INCOMPATIBLE_COMPARISON,
        ];
        let mut sorted = codes.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(codes.len(), sorted.len());
    }

    #[test]
    fn error_display_format() {
        let err = error_display::invalid_argument("Test", "Conseil");
        let display = format!("{}", err);
        assert!(display.contains("Erreur PMG-2:"));
    }
}
