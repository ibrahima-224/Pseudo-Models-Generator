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

//! Types d'erreurs pour l'inspection des modèles.
//!
//! Ce module définit le type [`InspectError`] qui encapsule toutes les erreurs
//! pouvant survenir lors de l'inspection d'un modèle.

use std::path::PathBuf;

/// Erreurs spécifiques à l'inspection des modèles.
#[derive(Debug)]
pub enum InspectError {
    /// Fichier config.json non trouvé.
    ConfigNotFound(PathBuf),
    /// Erreur d'entrée/sortie.
    Io(std::io::Error, PathBuf),
    /// Erreur de parsing JSON.
    Json(serde_json::Error, PathBuf),
    /// Fichier Safetensors invalide.
    InvalidSafetensors(PathBuf, String),
    /// Index des shards invalide.
    InvalidIndex(String),
    /// Autre erreur interne.
    Internal(String),
}

impl std::fmt::Display for InspectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectError::ConfigNotFound(path) => {
                write!(f, "Fichier config.json non trouvé : {}", path.display())
            },
            InspectError::Io(e, path) => {
                write!(f, "Erreur d'entrée/sortie pour {} : {}", path.display(), e)
            },
            InspectError::Json(e, path) => {
                write!(f, "Erreur de parsing JSON pour {} : {}", path.display(), e)
            },
            InspectError::InvalidSafetensors(path, msg) => {
                write!(
                    f,
                    "Fichier Safetensors invalide {} : {}",
                    path.display(),
                    msg
                )
            },
            InspectError::InvalidIndex(msg) => {
                write!(f, "Index des shards invalide : {}", msg)
            },
            InspectError::Internal(msg) => {
                write!(f, "Erreur interne : {}", msg)
            },
        }
    }
}

impl std::error::Error for InspectError {}

impl From<std::io::Error> for InspectError {
    fn from(e: std::io::Error) -> Self {
        InspectError::Internal(format!("Erreur d'entrée/sortie : {}", e))
    }
}

impl From<serde_json::Error> for InspectError {
    fn from(e: serde_json::Error) -> Self {
        InspectError::Internal(format!("Erreur JSON : {}", e))
    }
}
