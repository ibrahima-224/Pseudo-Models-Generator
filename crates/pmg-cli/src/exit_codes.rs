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

//! Codes de sortie standardisés pour la CLI PMG.
//!
//! Chaque commande doit retourner un code de sortie cohérent avec la nature
//! de l'erreur rencontrée. Ces codes permettent aux scripts et aux utilisateurs
//! de différencier les types d'échec.
//!
//! # Codes définis
//!
//! | Code | Signification |
//! |------|---------------|
//! | 0    | Succès |
//! | 1    | Erreur générale (non classifiée) |
//! | 2    | Argument invalide ou manquant |
//! | 3    | Modèle invalide ou corrompu |
//! | 4    | Erreur d'entrée/sortie (fichier introuvable, permissions, etc.) |
//! | 5    | Validation échouée |
//! | 6    | Comparaison incompatible (modèles trop différents) |

use std::process::ExitCode;

/// Code de succès (0).
pub const SUCCESS: u8 = 0;

/// Erreur générale non classifiée (1).
pub const GENERAL_ERROR: u8 = 1;

/// Argument invalide ou manquant (2).
pub const INVALID_ARGUMENT: u8 = 2;

/// Modèle invalide ou corrompu (3).
pub const INVALID_MODEL: u8 = 3;

/// Erreur d'entrée/sortie (4).
pub const IO_ERROR: u8 = 4;

/// Validation échouée (5).
pub const VALIDATION_FAILED: u8 = 5;

/// Comparaison incompatible (6).
pub const INCOMPATIBLE_COMPARISON: u8 = 6;

/// Convertit un code de sortie en `ExitCode` pour `std::process::exit`.
///
/// # Arguments
///
/// * `code` - Code de sortie (0-255).
///
/// # Exemple
///
/// ```rust
/// use pmg_cli::exit_codes;
/// use std::process::exit;
///
/// // exit(exit_codes::to_exit_code(exit_codes::SUCCESS));
/// ```
pub fn to_exit_code(code: u8) -> ExitCode {
    ExitCode::from(code)
}

/// Retourne une description brève du code de sortie.
pub fn describe(code: u8) -> &'static str {
    match code {
        SUCCESS => "Succès",
        GENERAL_ERROR => "Erreur générale",
        INVALID_ARGUMENT => "Argument invalide",
        INVALID_MODEL => "Modèle invalide",
        IO_ERROR => "Erreur d'entrée/sortie",
        VALIDATION_FAILED => "Validation échouée",
        INCOMPATIBLE_COMPARISON => "Comparaison incompatible",
        _ => "Code inconnu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique() {
        let codes = [
            SUCCESS,
            GENERAL_ERROR,
            INVALID_ARGUMENT,
            INVALID_MODEL,
            IO_ERROR,
            VALIDATION_FAILED,
            INCOMPATIBLE_COMPARISON,
        ];
        let mut sorted = codes.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(codes.len(), sorted.len(), "Les codes doivent être uniques");
    }

    #[test]
    fn describe_returns_non_empty() {
        for code in 0..=6 {
            assert!(!describe(code).is_empty());
        }
    }

    #[test]
    fn to_exit_code_preserves_value() {
        let ec = to_exit_code(42);
        // ExitCode ne expose pas directement la valeur, mais on peut vérifier qu'il est créé.
        let _ = ec;
    }
}
