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

//! Définition des erreurs structurées de la CLI PMG.
//!
//! Ce module définit l'enum `CliError` qui encapsule toutes les erreurs
//! spécifiques à la CLI avec leurs codes de sortie associés. Cela remplace
//! le string matching fragile par du pattern matching typé.

use thiserror::Error;

/// Erreurs structurées de la CLI PMG.
///
/// Chaque variante correspond à un code de sortie spécifique et fournit
/// un message d'erreur contextuel.
#[derive(Debug, Error)]
pub enum CliError {
    /// Erreur PMG-2 : argument invalide ou manquant.
    #[error("Erreur PMG-2: {message}")]
    InvalidArgument {
        /// Message descriptif de l'erreur.
        message: String,
    },

    /// Erreur PMG-3 : modèle invalide ou corrompu.
    #[error("Erreur PMG-3: {message}")]
    InvalidModel {
        /// Message descriptif de l'erreur.
        message: String,
    },

    /// Erreur PMG-4 : erreur d'entrée/sortie.
    #[error("Erreur PMG-4: {message}")]
    IoError {
        /// Message descriptif de l'erreur.
        message: String,
    },

    /// Erreur PMG-5 : validation échouée.
    #[error("Erreur PMG-5: {message}")]
    ValidationFailed {
        /// Message descriptif de l'erreur.
        message: String,
    },

    /// Erreur PMG-6 : comparaison incompatible.
    #[error("Erreur PMG-6: {message}")]
    IncompatibleComparison {
        /// Message descriptif de l'erreur.
        message: String,
    },

    /// Erreur générale non classifiée.
    #[error("{0}")]
    Other(String),
}

impl CliError {
    /// Retourne le code de sortie associé à cette erreur.
    ///
    /// # Retour
    /// Un `u8` représentant le code de sortie (0-255).
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::InvalidArgument { .. } => crate::exit_codes::INVALID_ARGUMENT,
            Self::InvalidModel { .. } => crate::exit_codes::INVALID_MODEL,
            Self::IoError { .. } => crate::exit_codes::IO_ERROR,
            Self::ValidationFailed { .. } => crate::exit_codes::VALIDATION_FAILED,
            Self::IncompatibleComparison { .. } => crate::exit_codes::INCOMPATIBLE_COMPARISON,
            Self::Other(_) => crate::exit_codes::GENERAL_ERROR,
        }
    }

    /// Crée une erreur `InvalidArgument` avec le message donné.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
        }
    }

    /// Crée une erreur `InvalidModel` avec le message donné.
    pub fn invalid_model(message: impl Into<String>) -> Self {
        Self::InvalidModel {
            message: message.into(),
        }
    }

    /// Crée une erreur `IoError` avec le message donné.
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
        }
    }

    /// Crée une erreur `ValidationFailed` avec le message donné.
    pub fn validation_failed(message: impl Into<String>) -> Self {
        Self::ValidationFailed {
            message: message.into(),
        }
    }

    /// Crée une erreur `IncompatibleComparison` avec le message donné.
    pub fn incompatible_comparison(message: impl Into<String>) -> Self {
        Self::IncompatibleComparison {
            message: message.into(),
        }
    }

    /// Crée une erreur `Other` avec le message donné.
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}

/// Conversion depuis `anyhow::Error` vers `CliError`.
///
/// Cette implémentation permet de convertir les erreurs `anyhow` en `CliError`
/// en analysant le message pour déterminer le type d'erreur.
impl From<anyhow::Error> for CliError {
    fn from(err: anyhow::Error) -> Self {
        let err_str = err.to_string();

        // Fonction utilitaire pour nettoyer le message en retirant les préfixes connus
        fn clean_message(msg: &str) -> String {
            if let Some(pos) = msg.find("Erreur PMG-") {
                // Trouver le premier ":" après "Erreur PMG-"
                if let Some(colon_pos) = msg[pos..].find(':') {
                    let prefix_len = pos + colon_pos + 1; // "Erreur PMG-X:" inclus
                    let rest = &msg[prefix_len..];
                    // Retirer l'espace leading
                    let cleaned = rest.trim_start();
                    return cleaned.to_string();
                }
            }
            msg.to_string()
        }

        // Analyse du message d'erreur pour déterminer le type
        if err_str.contains("Erreur PMG-2:") || err_str.contains("PMG-2:") {
            Self::InvalidArgument {
                message: clean_message(&err_str),
            }
        } else if err_str.contains("Erreur PMG-3:") || err_str.contains("PMG-3:") {
            Self::InvalidModel {
                message: clean_message(&err_str),
            }
        } else if err_str.contains("Erreur PMG-4:") || err_str.contains("PMG-4:") {
            Self::IoError {
                message: clean_message(&err_str),
            }
        } else if err_str.contains("Erreur PMG-5:") || err_str.contains("PMG-5:") {
            Self::ValidationFailed {
                message: clean_message(&err_str),
            }
        } else if err_str.contains("Erreur PMG-6:") || err_str.contains("PMG-6:") {
            Self::IncompatibleComparison {
                message: clean_message(&err_str),
            }
        } else {
            Self::Other(err_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_codes() {
        // Vérifie que chaque variante retourne le bon code de sortie
        assert_eq!(
            CliError::invalid_argument("test").exit_code(),
            crate::exit_codes::INVALID_ARGUMENT
        );
        assert_eq!(
            CliError::invalid_model("test").exit_code(),
            crate::exit_codes::INVALID_MODEL
        );
        assert_eq!(
            CliError::io_error("test").exit_code(),
            crate::exit_codes::IO_ERROR
        );
        assert_eq!(
            CliError::validation_failed("test").exit_code(),
            crate::exit_codes::VALIDATION_FAILED
        );
        assert_eq!(
            CliError::incompatible_comparison("test").exit_code(),
            crate::exit_codes::INCOMPATIBLE_COMPARISON
        );
        assert_eq!(
            CliError::other("test").exit_code(),
            crate::exit_codes::GENERAL_ERROR
        );
    }

    #[test]
    fn test_from_anyhow() {
        // Test de conversion depuis anyhow::Error
        let anyhow_err = anyhow::anyhow!("Erreur PMG-2: argument invalide");
        let cli_err = CliError::from(anyhow_err);
        assert!(matches!(cli_err, CliError::InvalidArgument { .. }));
        assert_eq!(cli_err.exit_code(), crate::exit_codes::INVALID_ARGUMENT);

        let anyhow_err = anyhow::anyhow!("Erreur generique");
        let cli_err = CliError::from(anyhow_err);
        assert!(matches!(cli_err, CliError::Other(_)));
        assert_eq!(cli_err.exit_code(), crate::exit_codes::GENERAL_ERROR);
    }
}
