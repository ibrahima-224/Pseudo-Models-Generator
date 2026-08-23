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

//! Affichage des erreurs en français pour la CLI PMG.
//!
//! Ce module fournit un mécanisme d'affichage des erreurs structurées
//! et compréhensibles, même pour les débutants. Chaque erreur est
//! présentée avec un code, une description, la cause technique et
//! un conseil correctif.
//!
//! # Format
//!
//! ```text
//! Erreur PMG-<CODE>: <description>
//! Cause : <cause technique>
//! Conseil : <action corrective>
//! ```

use std::fmt;

/// Structure représentant une erreur PMG affichable.
#[derive(Debug)]
pub struct PmgError {
    /// Code de l'erreur (correspond aux codes de sortie).
    pub code: u8,
    /// Description de l'erreur en français.
    pub description: String,
    /// Cause technique détaillée.
    pub cause: String,
    /// Conseil pour corriger l'erreur.
    pub advice: String,
}

impl PmgError {
    /// Crée une nouvelle erreur PMG.
    pub fn new(code: u8, description: &str, cause: &str, advice: &str) -> Self {
        Self {
            code,
            description: description.to_string(),
            cause: cause.to_string(),
            advice: advice.to_string(),
        }
    }

    /// Affiche l'erreur sur stderr.
    pub fn display(&self) {
        eprintln!("Erreur PMG-{}: {}", self.code, self.description);
        eprintln!("Cause : {}", self.cause);
        eprintln!("Conseil : {}", self.advice);
    }

    /// Retourne le code de sortie correspondant.
    pub fn exit_code(&self) -> u8 {
        self.code
    }
}

impl fmt::Display for PmgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Erreur PMG-{}: {}\nCause : {}\nConseil : {}",
            self.code, self.description, self.cause, self.advice
        )
    }
}

impl std::error::Error for PmgError {}

/// Crée une erreur pour argument invalide (code 2).
pub fn invalid_argument(cause: &str, advice: &str) -> PmgError {
    PmgError::new(2, "Argument invalide ou manquant", cause, advice)
}

/// Crée une erreur pour modèle invalide (code 3).
pub fn invalid_model(cause: &str, advice: &str) -> PmgError {
    PmgError::new(3, "Modèle invalide ou corrompu", cause, advice)
}

/// Crée une erreur pour erreur d'entrée/sortie (code 4).
pub fn io_error(cause: &str, advice: &str) -> PmgError {
    PmgError::new(4, "Erreur d'entrée/sortie", cause, advice)
}

/// Crée une erreur pour validation échouée (code 5).
pub fn validation_failed(cause: &str, advice: &str) -> PmgError {
    PmgError::new(5, "Validation échouée", cause, advice)
}

/// Crée une erreur pour comparaison incompatible (code 6).
pub fn incompatible_comparison(cause: &str, advice: &str) -> PmgError {
    PmgError::new(6, "Comparaison incompatible", cause, advice)
}

/// Crée une erreur générale (code 1).
pub fn general_error(cause: &str, advice: &str) -> PmgError {
    PmgError::new(1, "Erreur générale", cause, advice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_creation() {
        let err = PmgError::new(2, "Test", "Cause technique", "Conseil");
        assert_eq!(err.code, 2);
        assert_eq!(err.description, "Test");
        assert_eq!(err.cause, "Cause technique");
        assert_eq!(err.advice, "Conseil");
    }

    #[test]
    fn error_display_format() {
        let err = invalid_argument("Paramètre manquant", "Spécifiez le paramètre");
        let display = format!("{}", err);
        assert!(display.contains("Erreur PMG-2:"));
        assert!(display.contains("Argument invalide"));
        assert!(display.contains("Cause :"));
        assert!(display.contains("Conseil :"));
    }

    #[test]
    fn error_exit_code() {
        let err = io_error("Fichier introuvable", "Vérifiez le chemin");
        assert_eq!(err.exit_code(), 4);
    }

    #[test]
    fn all_error_factories() {
        let _ = invalid_argument("a", "b");
        let _ = invalid_model("a", "b");
        let _ = io_error("a", "b");
        let _ = validation_failed("a", "b");
        let _ = incompatible_comparison("a", "b");
        let _ = general_error("a", "b");
    }
}
