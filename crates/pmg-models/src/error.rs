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

//! Erreurs spécifiques à la crate `pmg-models`.

use std::path::PathBuf;
use thiserror::Error;

/// Erreurs pouvant survenir lors du chargement ou de l'utilisation des profils.
#[derive(Error, Debug)]
pub enum ModelProfileError {
    /// Le fichier de profil est introuvable.
    #[error("fichier de profil introuvable : {path}")]
    ProfileNotFound { path: PathBuf },

    /// Le fichier de profil n'est pas un fichier valide.
    #[error("le chemin n'est pas un fichier : {path}")]
    NotAFile { path: PathBuf },

    /// Erreur de lecture du fichier de profil.
    #[error("erreur de lecture du fichier {path} : {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Erreur de décodage JSON du fichier de profil.
    #[error("erreur de décodage JSON du fichier {path} : {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Nom de modèle inconnu lors du chargement.
    #[error("nom de modèle inconnu : {name}")]
    UnknownModel { name: String },

    /// Autre erreur de décodage.
    #[error("erreur de décodage JSON : {0}")]
    Decode(String),

    /// Valeur invalide pour un champ du profil.
    #[error("valeur invalide pour le champ '{field}' : {message}")]
    InvalidValue { field: String, message: String },

    /// Champ requis manquant dans le profil.
    #[error("champ requis manquant : {field}")]
    MissingField { field: String },

    /// Incohérence entre les valeurs du profil.
    #[error("incohérence d'architecture : {message}")]
    InconsistentArchitecture { message: String },
}

/// Type `Result` spécialisé pour les opérations de profils.
pub type Result<T> = std::result::Result<T, ModelProfileError>;
